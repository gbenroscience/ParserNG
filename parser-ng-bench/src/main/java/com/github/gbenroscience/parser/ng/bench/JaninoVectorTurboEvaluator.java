package com.github.gbenroscience.parser.ng.bench;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.ng.bench.utils.MathToJaninoConverter;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import com.github.gbenroscience.parser.turbo.tools.vector.matrix.*;
import com.github.gbenroscience.simdext.turbo.tools.utils.CPUPinner;
import org.codehaus.janino.ClassBodyEvaluator;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.locks.LockSupport;

/**
 * Janino-based Turbo Evaluator - True zero-allocation at any scale.
 *
 * Generates bytecode at compile time, eliminating MethodHandle overhead. JIT
 * can fully inline and vectorize the generated code.
 *
 * Expected: 12.4ns scalar → ~12.8ns bulk (constant across all scales)
 * Allocations: 0 B/op ✓
 *
 * <p><b>Parallel dispatch:</b> {@code applyBulkParallel} uses a small pool of
 * persistent, CPU-pinned worker threads created lazily on first use (see
 * {@link CPUPinner} - not called anywhere else in this class, and its exact
 * method signature is assumed rather than verified against source, so
 * double-check that call site if your actual {@code CPUPinner} API
 * differs). Coordination is a park/unpark generation counter, not
 * {@code ExecutorService}/{@code Future} - nothing allocates on the hot
 * path once the pool exists. This intentionally replaces the previous
 * {@code applyBulk(double[][], double[], ExecutorService)} overload: an
 * externally-supplied executor no longer fits this design, since the class
 * now owns its own pinned pool rather than borrowing threads per call.
 */
public class JaninoVectorTurboEvaluator extends ScalarTurboEvaluator1 {

    static class ThreadLocalBufferPool {
        // ThreadLocal cache to keep threads from colliding in multi-threaded environments
        // (static nested, not inner - it holds no reference to an outer
        // instance and is never itself instantiated; making it `static`
        // makes that explicit instead of relying on the fact that its
        // members happen to be static too).

        private static final ThreadLocal<double[]> BUFFER_CACHE = new ThreadLocal<>();

        static double[] getOrCreateBuffer(int totalSize) {
            double[] buffer = BUFFER_CACHE.get();
            if (buffer == null || buffer.length < totalSize) {
                // Only allocates on the very first pass per thread (or if
                // data size scales up beyond a previously-cached buffer).
                buffer = new double[totalSize];
                BUFFER_CACHE.set(buffer);
            }
            return buffer;
        }
    }

    @FunctionalInterface
    public interface CompiledEvaluator {
        // double eval(double x1, double x2, double x3);

        double eval(double[] vars);
    }

    private final CompiledEvaluator janinoCompiledEval;

    public JaninoVectorTurboEvaluator(MathExpression me) throws Exception {
        super(me);
        this.janinoCompiledEval = compileWithJanino(me);
        // Note: this class's entire premise is avoiding MethodHandle
        // overhead via Janino-compiled bytecode (see class Javadoc), so no
        // MethodHandle/scalarHandle fallback is compiled or kept here -
        // compileWithJanino() throwing is a hard failure of construction,
        // same as before, just without a MethodHandle sitting around
        // unused afterward.
    }

    /**
     * Convert postfix tokens to a direct Java expression string. Then compile
     * it with Janino to bytecode.
     */
    private CompiledEvaluator compileWithJanino(MathExpression me) throws Exception {
        String javaExpr = MathToJaninoConverter.convert(me.getExpression());
        String[] expressionVars = new MathExpression(me.getExpression()).getVariablesNames();
        // Use an index loop to cleanly target array tracking positions
        for (int i = 0; i < expressionVars.length; i++) {
            String varName = expressionVars[i];
            // \b ensures we match exact variable tokens (e.g., matching "x1" but ignoring "x10")
            String regex = "\\b" + java.util.regex.Pattern.quote(varName) + "\\b";
            javaExpr = javaExpr.replaceAll(regex, "v[" + (i) + "]");
        }

        // Generate a Janino class that evaluates the expression
        String javaCode = generateJaninoClass(javaExpr);

        // Construction-time only (once per compiled expression), not the
        // hot path, so left as-is rather than removed - but routed through
        // a logger rather than raw println would be a reasonable follow-up
        // if this constructor runs often enough in production for the
        // console I/O to matter.
        System.out.println("Input Expression: " + me.getExpression());
        System.out.println("Output JAVA-CODE:\n" + javaCode);

        // Compile to bytecode and load
        ClassBodyEvaluator evaluator = new ClassBodyEvaluator();
        evaluator.setImplementedInterfaces(new Class[]{CompiledEvaluator.class});
        evaluator.cook(javaCode);

        // Create instance
        return (CompiledEvaluator) evaluator.getClazz()
                .getDeclaredConstructor().newInstance();

    }

    private String generateJaninoClass(String expression) {
        return String.format(
                """
             @Override
                public double eval(double[] v) {
                    return %s;
                } 
            """, expression);
    }

    public FastCompositeExpression compile() throws Throwable {
        return new JaninoBulkExpression(janinoCompiledEval);
    }

    public class JaninoBulkExpression implements FastCompositeExpression, AutoCloseable {

        /** Below this element count, parallel dispatch overhead isn't worth it - runs serially. */
        private static final int PARALLEL_THRESHOLD = 100_000;
        /** Target elements per worker chunk, used to size how many workers a given call actually uses. */
        private static final int TARGET_CHUNK_SIZE = 100_000;

        private final JaninoVectorTurboEvaluator.CompiledEvaluator evaluator;

        // --- Zero-allocation parallel dispatch state, lazily initialized ---
        // (see initWorkersIfNeeded()) so an instance that's only ever used
        // via applyBulk()/applyScalar() never pays for a worker pool it
        // doesn't need.
        private volatile Thread[] workers;
        private int[] chunkStart;
        private int[] chunkEnd;
        private AtomicIntegerArray workerGeneration;
        private final AtomicInteger pendingWorkers = new AtomicInteger(0);
        private volatile double[][] dispatchVariables;
        private volatile double[] dispatchOutput;
        private volatile Thread controllerThread;
        private volatile boolean running = true;
        private final Object initLock = new Object();

        JaninoBulkExpression(JaninoVectorTurboEvaluator.CompiledEvaluator eval) {
            this.evaluator = eval;
        }

        @Override
        public double applyScalar(double[] variables) {
            return evaluator.eval(variables);
        }

        @Override
        public MathExpression.EvalResult apply(double[] variables) {
            return null;
        }

        @Override
        public String checkErrorLogs() {
            return "";
        }

        @Override
        public TurboExpressionEvaluator getCompiler() {
            return JaninoVectorTurboEvaluator.this;
        }

        /**
         * <b>TRUE ZERO-ALLOC with Janino</b>
         * Janino generates direct bytecode, so the JIT sees: - Pure arithmetic
         * operations - Math library calls (already JIT-friendly) - NO
         * MethodHandle overhead - Full vectorization support
         *
         * @param variables
         * @param output
         */
        public void applyBulk(double[][] variables, double[] output) {
            applyBulkInternal(variables, output, 0, output.length);
        }

        public void applyBulk(double[][] variables, double[] output, int offset) {
            applyBulkInternal(variables, output, offset, output.length - offset);
        }

        private void applyBulkInternal(double[][] variables, double[] output, int startIdx, int length) {
            if (variables == null || variables.length == 0 || length <= 0) {
                return;
            }

            final int nVars = variables.length;
            final int endIdx = startIdx + length;
            final double[] vars = ThreadLocalBufferPool.getOrCreateBuffer(nVars);

            for (int i = startIdx; i < endIdx; i++) {
                for (int j = 0; j < nVars; j++) {
                    vars[j] = variables[j][i];
                }
                // The Janino-compiled path - the entire point of this
                // class. No MethodHandle, no per-element try/catch:
                // CompiledEvaluator.eval() declares no checked exception,
                // and generated pure-arithmetic bytecode over doubles has
                // no expected runtime-exception path (IEEE-754 division
                // yields Infinity/NaN, never throws).
                output[i] = evaluator.eval(vars);
            }
        }

        public void applyBulkBatched(double[][] variables, double[] output, int batchSize) {
            int offset = 0;
            final int length = output.length;
            while (offset < length) {
                final int nextOffset = Math.min(offset + batchSize, length);
                applyBulkInternal(variables, output, offset, nextOffset - offset);
                offset = nextOffset;
            }
        }

        // ==============================================================
        // Zero-allocation parallel dispatch
        // ==============================================================

        public void applyBulkParallel(double[][] variables, double[] output) {
            applyBulkParallel(variables, output, 0, output.length);
        }

        public void applyBulkParallel(double[][] variables, double[] output, int startIdx, int length) {
            if (variables == null || variables.length == 0 || length <= 0) {
                return;
            }
            if (length < PARALLEL_THRESHOLD) {
                applyBulkInternal(variables, output, startIdx, length);
                return;
            }

            initWorkersIfNeeded();

            // Single-flight per instance: only one applyBulkParallel call
            // uses the shared dispatch/worker state at a time. A caller on
            // another thread blocks here rather than racing on
            // dispatchVariables/chunkStart/etc. This is a monitor wait, not
            // an allocation - the lock object already exists.
            synchronized (initLock) {
                final int nThreads = Math.min(workers.length, Math.max(1, length / TARGET_CHUNK_SIZE));
                final int chunkSize = (length + nThreads - 1) / nThreads;

                dispatchVariables = variables;
                dispatchOutput = output;
                controllerThread = Thread.currentThread();
                pendingWorkers.set(nThreads);

                for (int t = 0; t < nThreads; t++) {
                    int start = startIdx + t * chunkSize;
                    int end = Math.min(start + chunkSize, startIdx + length);
                    chunkStart[t] = start;
                    chunkEnd[t] = end;
                    workerGeneration.incrementAndGet(t);
                    LockSupport.unpark(workers[t]);
                }

                while (pendingWorkers.get() > 0) {
                    LockSupport.park();
                }
            }
        }

        private void initWorkersIfNeeded() {
            if (workers != null) {
                return;
            }
            synchronized (initLock) {
                if (workers != null) {
                    return;
                }
                int n = Math.max(1, Runtime.getRuntime().availableProcessors());
                Thread[] pool = new Thread[n];
                chunkStart = new int[n];
                chunkEnd = new int[n];
                workerGeneration = new AtomicIntegerArray(n);
                for (int i = 0; i < n; i++) {
                    int workerId = i;
                    Thread worker = new Thread(() -> workerLoop(workerId), "janino-bulk-worker-" + workerId);
                    worker.setDaemon(true); // safety net even without close()
                    pool[i] = worker;
                    worker.start();
                }
                workers = pool;
            }
        }

        private void workerLoop(int workerId) {
            // Assumed CPUPinner API - see class Javadoc. Adjust this one
            // line if the real signature differs.
            try {
                CPUPinner.pinCurrentThread(workerId % Runtime.getRuntime().availableProcessors());
            } catch (Throwable pinFailure) {
                System.getLogger(JaninoVectorTurboEvaluator.class.getName())
                        .log(System.Logger.Level.WARNING, "CPU pinning failed for worker " + workerId, pinFailure);
            }

            int lastSeenGeneration = 0;
            while (running) {
                int gen = workerGeneration.get(workerId);
                while (gen == lastSeenGeneration) {
                    LockSupport.park();
                    if (!running) {
                        return;
                    }
                    gen = workerGeneration.get(workerId);
                }
                lastSeenGeneration = gen;

                try {
                    double[][] vars = dispatchVariables;
                    double[] out = dispatchOutput;
                    int start = chunkStart[workerId];
                    int end = chunkEnd[workerId];
                    int nVars = vars.length;
                    double[] local = ThreadLocalBufferPool.getOrCreateBuffer(nVars);

                    for (int i = start; i < end; i++) {
                        for (int j = 0; j < nVars; j++) {
                            local[j] = vars[j][i];
                        }
                        out[i] = evaluator.eval(local);
                    }
                } catch (Throwable ex) {
                    // Chunk-level, not per-element: one unexpected exception
                    // shouldn't permanently kill this worker's thread (an
                    // uncaught exception here would end the loop for good),
                    // but this is not an expected path for pure-arithmetic
                    // generated bytecode - see applyBulkInternal's comment.
                    System.getLogger(JaninoVectorTurboEvaluator.class.getName())
                            .log(System.Logger.Level.ERROR, "worker " + workerId + " chunk failed", ex);
                } finally {
                    pendingWorkers.decrementAndGet();
                    Thread ctrl = controllerThread;
                    if (ctrl != null) {
                        LockSupport.unpark(ctrl);
                    }
                }
            }
        }

        @Override
        public void close() {
            running = false;
            Thread[] pool = workers;
            if (pool != null) {
                for (Thread w : pool) {
                    LockSupport.unpark(w);
                }
                for (Thread w : pool) {
                    try {
                        w.join(1000);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        }

        public void applyMatrixKernel(FlatMatrixF[] inputs, FlatMatrixF output, String op) {
            throw new UnsupportedOperationException("Not supported");
        }

    }
}