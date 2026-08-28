package com.github.gbenroscience.simdext.turbo.tools.junk;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.*;

import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression.BLOCK_SIZE;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression.PARALLEL_OPS_THRESHOLD;
import com.github.gbenroscience.simdext.turbo.tools.utils.CPUPinner;
import com.github.gbenroscience.simd.turbo.tools.utils.VectorizedCodyMath;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.ref.Cleaner;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import jdk.incubator.vector.*;

/**
 * High-Performance Vector API & Engine that fuses explicit SIMD vectorization
 * with a zero-allocation primitive stack interpreter. Completely eliminates the
 * scalar parser overhead and task object allocations on the hot path.
 *
 * Native on both {@code double} and {@code float} datasets. Parallel dispatch
 * uses topology-aware physical-core pinning and a master-participates model:
 * the calling thread computes a slice instead of idling in {@code park()}, so
 * {@code N} requested workers yield {@code N} busy cores (pool of {@code N-1}
 * plus the master when {@code N > 2}). CPU pinning is the reason this class
 * lives in the JDK22+ extension; it is most effective on Linux, where two
 * workers typically run at 1.88x–2.02x a single worker.
 */
public class SIMDEngineEvaluator extends VectorTurboEvaluator {

    public static final DoubleVector ONE_D = DoubleVector.broadcast(SPECIES, 1.0);
    public static final DoubleVector ZERO_D = DoubleVector.broadcast(SPECIES, 0.0);

    // =========================================================================
    // Float support: species/constants mirroring the double SPECIES/ONE_D/ZERO_D
    // above, but for jdk.incubator.vector's 32-bit float lane width. SPECIES_F
    // is independent of (and generally has a different lane count than) the
    // double SPECIES from VectorConfig, since the two species are sized to the
    // platform's preferred bit-width for each element type.
    // =========================================================================
    public static final VectorSpecies<Float> SPECIES_F = FloatVector.SPECIES_PREFERRED;
    public static final FloatVector ONE_F = FloatVector.broadcast(SPECIES_F, 1.0f);
    public static final FloatVector ZERO_F = FloatVector.broadcast(SPECIES_F, 0.0f);

    public SIMDEngineEvaluator(MathExpression me) throws Throwable {
        super(me);
    }

    public SIMDEngineEvaluator(MathExpression me, int numWorkers) throws Throwable {
        super(me, numWorkers);
    }

    public static final SIMDEngineEvaluator.SIMDVectorCompositeExpression getEvaluator(MathExpression me) throws Throwable {
        return (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(me).compile();
    }

    public static final SIMDEngineEvaluator.SIMDVectorCompositeExpression getEvaluator(String expr) throws Throwable {
        return (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(new MathExpression(expr)).compile();
    }

    public static final SIMDEngineEvaluator.SIMDVectorCompositeExpression getEvaluator(MathExpression me, int numWorkers) throws Throwable {
        return (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(me, numWorkers).compile();
    }

    public static final SIMDEngineEvaluator.SIMDVectorCompositeExpression getEvaluator(String expr, int numWorkers) throws Throwable {
        return (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(new MathExpression(expr), numWorkers).compile();
    }

    @Override
    public BatchedVectorCompositeExpression compile() throws Throwable {
        return new SIMDVectorCompositeExpression(stackDepth, BLOCK_SIZE);
    }

    public final class SIMDVectorCompositeExpression extends BatchedVectorCompositeExpression implements AutoCloseable {

        private static final Cleaner SYSTEM_CLEANER = Cleaner.create();

        private final int NUM_WORKERS;
        private final WorkerThread[] workerPool;
        private final CoordinationContext coordinationContext;
        private final Cleaner.Cleanable cleanable;
        private final int masterPinTarget; // -1 = don't pin the master thread

        private volatile boolean isClosed = false;

        private final ThreadLocal<EvaluationContext> masterEvalContext;
        
        private final ThreadLocal<Boolean> masterPinned = ThreadLocal.withInitial(() -> false);

        // Formal cleaner target decoupling task state from parent instances
        private static final class ThreadPoolShutdownAction implements Runnable {

            private final CoordinationContext context;
            private final WorkerThread[] pool;

            ThreadPoolShutdownAction(CoordinationContext context, WorkerThread[] pool) {
                this.context = context;
                this.pool = pool;
            }

            @Override
            public void run() {
                if (context != null) {
                    context.running = false;
                }
                if (pool != null) {
                    for (WorkerThread worker : pool) {
                        if (worker != null) {
                            LockSupport.unpark(worker);
                        }
                    }
                }
            }
        }

        public SIMDVectorCompositeExpression(int stackDepth, int blockSize) {
            super(compiledScalarHandle, opcodes, targetSlots,
                    literalConstants, instructionCount, varCount);

            this.masterEvalContext = ThreadLocal.withInitial(() -> new EvaluationContext(stackDepth, blockSize, varCount));

            if (numWorkers <= 2) {
                this.NUM_WORKERS = numWorkers;
            } else {
                this.NUM_WORKERS = numWorkers - 1;
            }
            if (this.NUM_WORKERS > 0) {
                this.coordinationContext = new CoordinationContext(NUM_WORKERS);
                this.workerPool = new WorkerThread[NUM_WORKERS];

                // Query real physical-core topology instead of assuming logical
                // CPUs are interleaved by core. Each element of coreGroups is
                // one physical core's set of SMT-sibling logical indices; using
                // group[i % groups.length][0] as the pin target for worker i
                // guarantees distinct physical cores whenever enough exist,
                // regardless of how the OS numbers hyperthread siblings.
                int[][] coreGroups = CPUPinner.detectPhysicalCoreGroups();
                if (coreGroups == null || coreGroups.length == 0) {
                    // Degraded path: pin everyone to logical CPU 0 rather than
                    // NPE'ing construction when /sys topology is unreadable.
                    coreGroups = new int[][]{{0}};
                }
                this.masterPinTarget = coreGroups[NUM_WORKERS % coreGroups.length][0];

                // +1: the master (calling) thread also computes a slice instead of
                // sitting idle in park() while the pool does all the work.
                for (int i = 0; i < NUM_WORKERS; i++) {
                    int pinTarget = coreGroups[i % coreGroups.length][0];
                    workerPool[i] = new WorkerThread(i, NUM_WORKERS + 1, pinTarget, stackDepth, blockSize, coordinationContext);
                }
                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i].start();
                }
                this.cleanable = SYSTEM_CLEANER.register(this, new ThreadPoolShutdownAction(coordinationContext, workerPool));
            } else {
                this.coordinationContext = null;
                this.workerPool = null;
                this.cleanable = null;
                this.masterPinTarget = -1;
            }
        }

        @Override
        public void close() {
            if (isClosed) {
                return;
            }
            isClosed = true;
            if (cleanable != null) {
                cleanable.clean();
            }
            masterEvalContext.remove();
        }

        // =========================================================================
        // Scalable Coordination Engine Layout
        // =========================================================================
        private static final class CoordinationContext {

            final AtomicInteger completionLatch = new AtomicInteger(0);
            final AtomicInteger[] workerSignals;
            volatile Thread masterThread;
            volatile boolean running = true;

            // Shared data structures
            double[][] vars2D;
            double[] vars1D;
            double[] output;

            MemorySegment varsSegment;
            MemorySegment[] varsSegmentArray; // THIS IS NEW: one segment per variable/column (e.g. in the context of parser-ng-arrow - one per Arrow ValueVector)
            MemorySegment outputSegment;
            long totalSamplesLong;

            int totalSamples;

            // =====================================================================
            // Float mirrors of the payload fields above. MemorySegment-backed
            // payloads reuse varsSegment/varsSegmentArray/outputSegment (raw
            // memory has no inherent element type); a separate `segmentIsFloat`
            // flag tells the worker which element width/layout to interpret
            // those segments as.
            // =====================================================================
            float[][] vars2DF;
            float[] vars1DF;
            float[] outputF;
            boolean segmentIsFloat;

            CoordinationContext(int numWorkers) {
                this.workerSignals = new AtomicInteger[numWorkers];
                for (int i = 0; i < numWorkers; i++) {
                    this.workerSignals[i] = new AtomicInteger(0); // 0 = Idle, 1 = Work Assigned
                }
            }

            void clearPayload() {
                this.vars2D = null;
                this.vars1D = null;
                this.varsSegment = null;
                this.varsSegmentArray = null; // ALSO NEW
                this.output = null;
                this.outputSegment = null;
                this.masterThread = null;

                this.vars2DF = null;
                this.vars1DF = null;
                this.outputF = null;
                this.segmentIsFloat = false;
            }
        }

        private final class WorkerThread extends Thread {

            private final int workerId;
            private final int totalWorkers;
            private final int pinTarget;
            private final EvaluationContext evalContext;
            private final CoordinationContext ctx;

            WorkerThread(int workerId, int totalWorkers, int pinTarget, int stackDepth, int blockSize, CoordinationContext ctx) {
                this.workerId = workerId;
                this.totalWorkers = totalWorkers;
                this.pinTarget = pinTarget;
                this.ctx = ctx;
                this.evalContext = new EvaluationContext(stackDepth, blockSize, varCount);
                this.setDaemon(true);
                this.setName("ParserNG-SIMD-Worker-" + workerId);
            }

            @Override
            public void run() {
                CPUPinner.pinCurrentThread(pinTarget);
                final CoordinationContext sharedCtx = this.ctx;
                final int id = this.workerId;
                final int totalThreads = this.totalWorkers;
                final AtomicInteger signal = sharedCtx.workerSignals[id];

                while (sharedCtx.running) {
                    while (signal.get() == 0 && sharedCtx.running) {
                        LockSupport.park();
                    }
                    if (!sharedCtx.running) {
                        return;
                    }

                    try {
                        if (sharedCtx.varsSegmentArray != null) {
                            // NEW: per-column MemorySegment path (e.g. Arrow ValueVector buffers)
                            long numSamplesL = sharedCtx.totalSamplesLong;
                            long chunkSizeL = numSamplesL / totalThreads;
                            long startIdxL = id * chunkSizeL;
                            long lengthL = (id == totalThreads - 1) ? (numSamplesL - startIdxL) : chunkSizeL;

                            if (lengthL > 0) {
                                if (sharedCtx.segmentIsFloat) {
                                    applyBulkInternalFloat(sharedCtx.varsSegmentArray, evalContext, numSamplesL, sharedCtx.outputSegment, startIdxL, lengthL);
                                } else {
                                    applyBulkInternal(sharedCtx.varsSegmentArray, evalContext, numSamplesL, sharedCtx.outputSegment, startIdxL, lengthL);
                                }
                            }
                        } else if (sharedCtx.varsSegment != null) {
                            long numSamplesL = sharedCtx.totalSamplesLong;
                            long chunkSizeL = numSamplesL / totalThreads;
                            long startIdxL = id * chunkSizeL;
                            long lengthL = (id == totalThreads - 1) ? (numSamplesL - startIdxL) : chunkSizeL;

                            if (lengthL > 0) {
                                if (sharedCtx.segmentIsFloat) {
                                    applyBulkInternalFloat(sharedCtx.varsSegment, evalContext, numSamplesL, sharedCtx.outputSegment, startIdxL, lengthL);
                                } else {
                                    applyBulkInternal(sharedCtx.varsSegment, evalContext, numSamplesL, sharedCtx.outputSegment, startIdxL, lengthL);
                                }
                            }
                        } else {
                            int numSamples = sharedCtx.totalSamples;
                            int chunkSize = numSamples / totalThreads;
                            int startIdx = id * chunkSize;
                            int length = (id == totalThreads - 1) ? (numSamples - startIdx) : chunkSize;

                            if (length > 0) {
                                if (sharedCtx.vars2D != null) {
                                    applyBulkInternal(sharedCtx.vars2D, evalContext, numSamples, sharedCtx.output, startIdx, length);
                                } else if (sharedCtx.vars1D != null) {
                                    applyBulkInternal(sharedCtx.vars1D, evalContext, numSamples, sharedCtx.output, startIdx, length);
                                } else if (sharedCtx.vars2DF != null) {
                                    applyBulkInternal(sharedCtx.vars2DF, evalContext, numSamples, sharedCtx.outputF, startIdx, length);
                                } else if (sharedCtx.vars1DF != null) {
                                    applyBulkInternal(sharedCtx.vars1DF, evalContext, numSamples, sharedCtx.outputF, startIdx, length);
                                }
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        signal.set(0); // Reset local channel signal state
                        if (sharedCtx.completionLatch.decrementAndGet() == 0 && sharedCtx.masterThread != null) {
                            LockSupport.unpark(sharedCtx.masterThread);
                        }
                    }
                }
            }
        }


        /**
         * Pins the calling (master) thread to its reserved physical core, if
         * one was available at construction time (see masterPinTarget). Cheap
         * to call on every dispatch — SetThreadAffinityMask/sched_setaffinity
         * are idempotent single syscalls, and the calling thread can differ
         * across invocations (e.g. different application threads driving the
         * same expression), so we can't just pin once in the constructor.
         */
        private void pinMasterIfNeeded() {
            if (masterPinTarget >= 0 && !masterPinned.get()) {
                CPUPinner.pinCurrentThread(masterPinTarget);
                masterPinned.set(true);
            }
        }

        private void executeParallelProcessing(double[][] vars2D, double[] vars1D, double[] output, int numSamples) {
            pinMasterIfNeeded();
            coordinationContext.vars2D = vars2D;
            coordinationContext.vars1D = vars1D;
            coordinationContext.vars2DF = null;
            coordinationContext.vars1DF = null;
            coordinationContext.varsSegment = null;
            coordinationContext.varsSegmentArray = null;
            coordinationContext.output = output;
            coordinationContext.outputF = null;
            coordinationContext.segmentIsFloat = false;
            coordinationContext.totalSamples = numSamples;
            coordinationContext.masterThread = Thread.currentThread();
            coordinationContext.completionLatch.set(NUM_WORKERS);

            // Trigger execution via explicit atomic state updates
            for (int i = 0; i < NUM_WORKERS; i++) {
                coordinationContext.workerSignals[i].set(1);
                LockSupport.unpark(workerPool[i]);
            }

            // Master computes the final slice itself (id == NUM_WORKERS out of
            // NUM_WORKERS+1 total participants) instead of sitting idle in park()
            // while the pool does all the work.
            final int totalParticipants = NUM_WORKERS + 1;
            final int chunkSize = numSamples / totalParticipants;
            final int startIdx = NUM_WORKERS * chunkSize;
            final int length = numSamples - startIdx;
            if (length > 0) {
                EvaluationContext ctx = masterEvalContext.get();
                if (vars2D != null) {
                    applyBulkInternal(vars2D, ctx, numSamples, output, startIdx, length);
                } else if (vars1D != null) {
                    applyBulkInternal(vars1D, ctx, numSamples, output, startIdx, length);
                }
            }

            while (coordinationContext.completionLatch.get() > 0) {
                LockSupport.park();
            }
            coordinationContext.clearPayload();
        }

        private void executeParallelProcessing(MemorySegment varsSegment, MemorySegment outputSegment, long numSamples) {
            pinMasterIfNeeded();
            coordinationContext.varsSegment = varsSegment;
            coordinationContext.varsSegmentArray = null;
            coordinationContext.outputSegment = outputSegment;
            coordinationContext.totalSamplesLong = numSamples;
            coordinationContext.segmentIsFloat = false;
            coordinationContext.masterThread = Thread.currentThread();
            coordinationContext.completionLatch.set(NUM_WORKERS);

            // Trigger execution via explicit atomic state updates
            for (int i = 0; i < NUM_WORKERS; i++) {
                coordinationContext.workerSignals[i].set(1);
                LockSupport.unpark(workerPool[i]);
            }

            // Master computes the final slice itself (id == NUM_WORKERS out of
            // NUM_WORKERS+1 total participants) instead of sitting idle in park().
            final long totalParticipants = NUM_WORKERS + 1L;
            final long chunkSize = numSamples / totalParticipants;
            final long startIdx = NUM_WORKERS * chunkSize;
            final long length = numSamples - startIdx;
            if (length > 0) {
                applyBulkInternal(varsSegment, masterEvalContext.get(), numSamples, outputSegment, startIdx, length);
            }

            while (coordinationContext.completionLatch.get() > 0) {
                LockSupport.park();
            }
            coordinationContext.clearPayload();
        }

        /**
         * NEW: Parallel dispatch for the per-column MemorySegment[] path (one
         * segment per variable, e.g. one per Arrow ValueVector buffer).
         */
        private void executeParallelProcessing(MemorySegment[] varsSegmentArray, MemorySegment outputSegment, long numSamples) {
            pinMasterIfNeeded();
            coordinationContext.varsSegmentArray = varsSegmentArray;
            coordinationContext.varsSegment = null;
            coordinationContext.outputSegment = outputSegment;
            coordinationContext.totalSamplesLong = numSamples;
            coordinationContext.segmentIsFloat = false;
            coordinationContext.masterThread = Thread.currentThread();
            coordinationContext.completionLatch.set(NUM_WORKERS);

            // Trigger execution via explicit atomic state updates
            for (int i = 0; i < NUM_WORKERS; i++) {
                coordinationContext.workerSignals[i].set(1);
                LockSupport.unpark(workerPool[i]);
            }

            // Master computes the final slice itself (id == NUM_WORKERS out of
            // NUM_WORKERS+1 total participants) instead of sitting idle in park().
            final long totalParticipants = NUM_WORKERS + 1L;
            final long chunkSize = numSamples / totalParticipants;
            final long startIdx = NUM_WORKERS * chunkSize;
            final long length = numSamples - startIdx;
            if (length > 0) {
                applyBulkInternal(varsSegmentArray, masterEvalContext.get(), numSamples, outputSegment, startIdx, length);
            }

            while (coordinationContext.completionLatch.get() > 0) {
                LockSupport.park();
            }
            coordinationContext.clearPayload();
        }

        /**
         * Float mirror of
         * {@link #executeParallelProcessing(double[][], double[], double[], int)}.
         * Master computes the final slice (id == NUM_WORKERS out of
         * NUM_WORKERS+1).
         */
        private void executeParallelProcessing(float[][] vars2D, float[] vars1D, float[] output, int numSamples) {
            pinMasterIfNeeded();
            coordinationContext.vars2D = null;
            coordinationContext.vars1D = null;
            coordinationContext.vars2DF = vars2D;
            coordinationContext.vars1DF = vars1D;
            coordinationContext.varsSegment = null;
            coordinationContext.varsSegmentArray = null;
            coordinationContext.output = null;
            coordinationContext.outputF = output;
            coordinationContext.segmentIsFloat = false;
            coordinationContext.totalSamples = numSamples;
            coordinationContext.masterThread = Thread.currentThread();
            coordinationContext.completionLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                coordinationContext.workerSignals[i].set(1);
                LockSupport.unpark(workerPool[i]);
            }

            // Master computes the final slice itself (id == NUM_WORKERS out of
            // NUM_WORKERS+1 total participants) instead of sitting idle in park()
            // while the pool does all the work.
            final int totalParticipants = NUM_WORKERS + 1;
            final int chunkSize = numSamples / totalParticipants;
            final int startIdx = NUM_WORKERS * chunkSize;
            final int length = numSamples - startIdx;
            if (length > 0) {
                EvaluationContext ctx = masterEvalContext.get();
                if (vars2D != null) {
                    applyBulkInternal(vars2D, ctx, numSamples, output, startIdx, length);
                } else if (vars1D != null) {
                    applyBulkInternal(vars1D, ctx, numSamples, output, startIdx, length);
                }
            }

            while (coordinationContext.completionLatch.get() > 0) {
                LockSupport.park();
            }
            coordinationContext.clearPayload();
        }

        /**
         * Float mirror of {@link #executeParallelProcessing(MemorySegment, MemorySegment, long)}
         * — {@code varsSegment}/{@code outputSegment} hold packed 32-bit floats
         * rather than 64-bit doubles.
         */
        private void executeParallelProcessingFloat(MemorySegment varsSegment, MemorySegment outputSegment, long numSamples) {
            pinMasterIfNeeded();
            coordinationContext.varsSegment = varsSegment;
            coordinationContext.varsSegmentArray = null;
            coordinationContext.outputSegment = outputSegment;
            coordinationContext.totalSamplesLong = numSamples;
            coordinationContext.segmentIsFloat = true;
            coordinationContext.masterThread = Thread.currentThread();
            coordinationContext.completionLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                coordinationContext.workerSignals[i].set(1);
                LockSupport.unpark(workerPool[i]);
            }

            final long totalParticipants = NUM_WORKERS + 1L;
            final long chunkSize = numSamples / totalParticipants;
            final long startIdx = NUM_WORKERS * chunkSize;
            final long length = numSamples - startIdx;
            if (length > 0) {
                applyBulkInternalFloat(varsSegment, masterEvalContext.get(), numSamples, outputSegment, startIdx, length);
            }

            while (coordinationContext.completionLatch.get() > 0) {
                LockSupport.park();
            }
            coordinationContext.clearPayload();
        }

        /**
         * Float mirror of
         * {@link #executeParallelProcessing(MemorySegment[], MemorySegment, long)}
         * for per-column segments holding packed 32-bit floats.
         */
        private void executeParallelProcessingFloat(MemorySegment[] varsSegmentArray, MemorySegment outputSegment, long numSamples) {
            pinMasterIfNeeded();
            coordinationContext.varsSegmentArray = varsSegmentArray;
            coordinationContext.varsSegment = null;
            coordinationContext.outputSegment = outputSegment;
            coordinationContext.totalSamplesLong = numSamples;
            coordinationContext.segmentIsFloat = true;
            coordinationContext.masterThread = Thread.currentThread();
            coordinationContext.completionLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                coordinationContext.workerSignals[i].set(1);
                LockSupport.unpark(workerPool[i]);
            }

            final long totalParticipants = NUM_WORKERS + 1L;
            final long chunkSize = numSamples / totalParticipants;
            final long startIdx = NUM_WORKERS * chunkSize;
            final long length = numSamples - startIdx;
            if (length > 0) {
                applyBulkInternalFloat(varsSegmentArray, masterEvalContext.get(), numSamples, outputSegment, startIdx, length);
            }

            while (coordinationContext.completionLatch.get() > 0) {
                LockSupport.park();
            }
            coordinationContext.clearPayload();
        }

        @Override
        public void applyBulkParallel(double[][] variables, double[] output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = variables[0].length;
            if (isClosed || NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(variables, masterEvalContext.get(), numSamples, output, 0, numSamples);
                return;
            }
            executeParallelProcessing(variables, null, output, numSamples);
        }

        @Override
        public void applyBulkParallel(double[] flatVariables, double[] output) {
            if (flatVariables == null || output == null) {
                return;
            }
            int numSamples = output.length;
            if (isClosed || NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(flatVariables, masterEvalContext.get(), numSamples, output, 0, numSamples);
                return;
            }
            executeParallelProcessing(null, flatVariables, output, numSamples);
        }

        @Override
        public void applyBulk(double[][] variables, double[] output) {
            int numSamples = variables[0].length;
            applyBulkInternal(variables, masterEvalContext.get(), numSamples, output, 0, numSamples);
        }

        @Override
        public void applyBulk(double[] flatVariables, double[] output) {
            applyBulkInternal(flatVariables, masterEvalContext.get(), output.length, output, 0, output.length);
        }

        @Override
        public void applyBulkBatched(double[][] variables, double[] output, int batchSize) {
            int numSamples = variables[0].length;
            EvaluationContext ctx = masterEvalContext.get();
            for (int start = 0; start < numSamples; start += batchSize) {
                int len = Math.min(batchSize, numSamples - start);
                applyBulkInternal(variables, ctx, numSamples, output, start, len);
            }
        }

        @Override
        public void applyBulkBatched(double[] flatVariables, double[] output, int batchSize) {
            int numSamples = output.length;
            EvaluationContext ctx = masterEvalContext.get();
            for (int start = 0; start < numSamples; start += batchSize) {
                int len = Math.min(batchSize, numSamples - start);
                applyBulkInternal(flatVariables, ctx, numSamples, output, start, len);
            }
        }

        public void applyBulk(MemorySegment variables, MemorySegment output) {
            if (variables == null || output == null) {
                return;
            }
            long numSamples = output.byteSize() / 8L;
            applyBulkInternal(variables, masterEvalContext.get(), numSamples, output, 0L, numSamples);
        }

        public void applyBulkParallel(MemorySegment variables, MemorySegment output) {
            if (variables == null || output == null) {
                return;
            }
            long numSamples = output.byteSize() / 8L;
            if (isClosed || NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(variables, masterEvalContext.get(), numSamples, output, 0L, numSamples);
                return;
            }
            executeParallelProcessing(variables, output, numSamples);
        }

        // =========================================================================
        // NEW: Multi-Segment (per-column) MemorySegment API — for zero-copy
        // integration with columnar sources like Apache Arrow, where each
        // variable/column lives in its OWN MemorySegment (e.g. one per
        // Arrow ValueVector's data buffer) rather than being concatenated
        // into a single flat segment.
        // =========================================================================
        /**
         * Evaluate this expression over columnar data where each variable is
         * backed by its own {@link MemorySegment} (e.g. one per Arrow
         * {@code Float8Vector}'s data buffer), writing results directly into
         * {@code output}. No intermediate copy into a combined segment or
         * on-heap array is performed by this entry point.
         *
         * @param variables one MemorySegment per variable, in variable/slot
         * order; each must hold at least {@code numSamples} contiguous doubles
         * @param output destination segment; its byte size determines
         * {@code numSamples} (byteSize() / 8)
         */
        public void applyBulk(MemorySegment[] variables, MemorySegment output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            long numSamples = output.byteSize() / 8L;
            applyBulkInternal(variables, masterEvalContext.get(), numSamples, output, 0L, numSamples);
        }

        /**
         * Parallel counterpart of
         * {@link #applyBulk(MemorySegment[], MemorySegment)}. Falls back to
         * single-threaded execution when no worker pool is available or the
         * batch is below {@code PARALLEL_OPS_THRESHOLD}.
         */
        public void applyBulkParallel(MemorySegment[] variables, MemorySegment output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            long numSamples = output.byteSize() / 8L;
            if (isClosed || NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(variables, masterEvalContext.get(), numSamples, output, 0L, numSamples);
                return;
            }
            executeParallelProcessing(variables, output, numSamples);
        }

        // =========================================================================
        // Float API — float[] / float[][] paths, mirroring the double[] /
        // double[][] entry points above exactly (same batching, same
        // single-threaded-vs-parallel dispatch rules). True Java overloads,
        // since float[]/float[][] are distinct erasures from double[]/double[][].
        // =========================================================================
        public void applyBulkParallel(float[][] variables, float[] output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = variables[0].length;
            if (isClosed || NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(variables, masterEvalContext.get(), numSamples, output, 0, numSamples);
                return;
            }
            executeParallelProcessing(variables, null, output, numSamples);
        }

        public void applyBulkParallel(float[] flatVariables, float[] output) {
            if (flatVariables == null || output == null) {
                return;
            }
            int numSamples = output.length;
            if (isClosed || NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(flatVariables, masterEvalContext.get(), numSamples, output, 0, numSamples);
                return;
            }
            executeParallelProcessing(null, flatVariables, output, numSamples);
        }

        public void applyBulk(float[][] variables, float[] output) {
            int numSamples = variables[0].length;
            applyBulkInternal(variables, masterEvalContext.get(), numSamples, output, 0, numSamples);
        }

        public void applyBulk(float[] flatVariables, float[] output) {
            applyBulkInternal(flatVariables, masterEvalContext.get(), output.length, output, 0, output.length);
        }

        public void applyBulkBatched(float[][] variables, float[] output, int batchSize) {
            int numSamples = variables[0].length;
            EvaluationContext ctx = masterEvalContext.get();
            for (int start = 0; start < numSamples; start += batchSize) {
                int len = Math.min(batchSize, numSamples - start);
                applyBulkInternal(variables, ctx, numSamples, output, start, len);
            }
        }

        public void applyBulkBatched(float[] flatVariables, float[] output, int batchSize) {
            int numSamples = output.length;
            EvaluationContext ctx = masterEvalContext.get();
            for (int start = 0; start < numSamples; start += batchSize) {
                int len = Math.min(batchSize, numSamples - start);
                applyBulkInternal(flatVariables, ctx, numSamples, output, start, len);
            }
        }

        // =========================================================================
        // Float API — MemorySegment paths, for segments packed with 32-bit
        // floats rather than 64-bit doubles. These cannot be plain overloads
        // of the double MemorySegment methods above (same erasure: raw memory
        // carries no element-type information), so they use a distinct
        // "...Float" name instead.
        // =========================================================================
        /**
         * Float counterpart of
         * {@link #applyBulk(MemorySegment, MemorySegment)}.
         *
         * @param variables single concatenated segment holding {@code varCount}
         * variables back-to-back, each with at least {@code numSamples}
         * contiguous 32-bit floats
         * @param output destination segment; its byte size determines
         * {@code numSamples} (byteSize() / 4)
         */
        public void applyBulkFloat(MemorySegment variables, MemorySegment output) {
            if (variables == null || output == null) {
                return;
            }
            long numSamples = output.byteSize() / 4L;
            applyBulkInternalFloat(variables, masterEvalContext.get(), numSamples, output, 0L, numSamples);
        }

        /**
         * Float counterpart of
         * {@link #applyBulkParallel(MemorySegment, MemorySegment)}.
         */
        public void applyBulkParallelFloat(MemorySegment variables, MemorySegment output) {
            if (variables == null || output == null) {
                return;
            }
            long numSamples = output.byteSize() / 4L;
            if (isClosed || NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternalFloat(variables, masterEvalContext.get(), numSamples, output, 0L, numSamples);
                return;
            }
            executeParallelProcessingFloat(variables, output, numSamples);
        }

        /**
         * Float counterpart of
         * {@link #applyBulk(MemorySegment[], MemorySegment)} — true zero-copy,
         * per-column evaluation where each variable is its own
         * {@link MemorySegment} of packed 32-bit floats (e.g. one per Arrow
         * {@code Float4Vector}'s data buffer).
         *
         * @param variables one MemorySegment per variable, in variable/slot
         * order; each must hold at least {@code numSamples} contiguous floats
         * @param output destination segment; its byte size determines
         * {@code numSamples} (byteSize() / 4)
         */
        public void applyBulkFloat(MemorySegment[] variables, MemorySegment output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            long numSamples = output.byteSize() / 4L;
            applyBulkInternalFloat(variables, masterEvalContext.get(), numSamples, output, 0L, numSamples);
        }

        /**
         * Float counterpart of
         * {@link #applyBulkParallel(MemorySegment[], MemorySegment)}.
         */
        public void applyBulkParallelFloat(MemorySegment[] variables, MemorySegment output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            long numSamples = output.byteSize() / 4L;
            if (isClosed || NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternalFloat(variables, masterEvalContext.get(), numSamples, output, 0L, numSamples);
                return;
            }
            executeParallelProcessingFloat(variables, output, numSamples);
        }

        // =========================================================================
        // Internal Primitive State Engine Implementation
        // =========================================================================
        private static final class EvaluationContext {

            final double[][] stackArrays;
            final int[] stackOffsets;
            final boolean[] stackIsConst;
            final double[] stackConstVals;
            final double[] scratch;
            final MemorySegment scratchSegment; // cached wrapper of `scratch`, for lazy segment->scratch materialization
            int sp = 0;

            double[] flatVariables;
            double[][] _2DVariables;
            int dataSize;
            int blockStart;

            // NEW: true zero-copy MemorySegment-backed stack support — one segment per
            // variable (e.g. one per Arrow ValueVector's data buffer). OP_LOAD points a
            // stack slot directly here instead of copying into segmentBlockVars below.
            MemorySegment[] segVariables;
            long segBlockStart;
            final MemorySegment[] stackSegments;
            final long[] stackSegOffsets;
            final boolean[] stackIsSegment;

            // Legacy staged-copy targets (still used by the single-concatenated-segment path)
            final double[][] segmentBlockVars;
            final MemorySegment[] segmentBlockVarsMem;

            // =====================================================================
            // Float mirrors of the primitive stack-engine state above. Kept as
            // fully separate fields/arrays (rather than reusing the double
            // ones) since float and double element storage cannot alias the
            // same backing array. sp/stackOffsets/stackIsConst/stackIsSegment/
            // stackSegments/stackSegOffsets/segVariables/segBlockStart/
            // dataSize/blockStart ARE shared between the double and float
            // execution paths: they are purely index/metadata bookkeeping with
            // no element-type dependency, and a given EvaluationContext
            // instance only ever drives one path at a time per block.
            // =====================================================================
            final float[][] stackArraysF;
            final float[] stackConstValsF;
            final float[] scratchF;
            final MemorySegment scratchSegmentF;

            float[] flatVariablesF;
            float[][] _2DVariablesF;

            final float[][] segmentBlockVarsF;
            final MemorySegment[] segmentBlockVarsMemF;

            EvaluationContext(int maxStackDepth, int blockSize, int varCount) {
                stackArrays = new double[maxStackDepth][];
                stackOffsets = new int[maxStackDepth];
                stackIsConst = new boolean[maxStackDepth];
                stackConstVals = new double[maxStackDepth];
                scratch = new double[maxStackDepth * blockSize];
                scratchSegment = MemorySegment.ofArray(scratch);

                stackSegments = new MemorySegment[maxStackDepth];
                stackSegOffsets = new long[maxStackDepth];
                stackIsSegment = new boolean[maxStackDepth];

                // Zero-allocation pre-cached memory segment loading targets
                segmentBlockVars = new double[varCount][blockSize];
                segmentBlockVarsMem = new MemorySegment[varCount];
                for (int i = 0; i < varCount; i++) {
                    segmentBlockVarsMem[i] = MemorySegment.ofArray(segmentBlockVars[i]);
                }

                stackArraysF = new float[maxStackDepth][];
                stackConstValsF = new float[maxStackDepth];
                scratchF = new float[maxStackDepth * blockSize];
                scratchSegmentF = MemorySegment.ofArray(scratchF);

                segmentBlockVarsF = new float[varCount][blockSize];
                segmentBlockVarsMemF = new MemorySegment[varCount];
                for (int i = 0; i < varCount; i++) {
                    segmentBlockVarsMemF[i] = MemorySegment.ofArray(segmentBlockVarsF[i]);
                }
            }

            void initForBlock(double[] flat, double[][] _2D, int size, int bStart) {
                this.sp = 0;
                this.flatVariables = flat;
                this._2DVariables = _2D;
                this.flatVariablesF = null;
                this._2DVariablesF = null;
                this.segVariables = null;
                this.dataSize = size;
                this.blockStart = bStart;
            }

            /**
             * Float mirror of
             * {@link #initForBlock(double[], double[][], int, int)}.
             */
            void initForBlockFloat(float[] flat, float[][] _2D, int size, int bStart) {
                this.sp = 0;
                this.flatVariablesF = flat;
                this._2DVariablesF = _2D;
                this.flatVariables = null;
                this._2DVariables = null;
                this.segVariables = null;
                this.dataSize = size;
                this.blockStart = bStart;
            }

            /**
             * NEW: initializes this context for a block whose variables are
             * read directly from off-heap MemorySegments — no staging copy.
             * Each element of {@code segVars} is one variable's full-length
             * backing segment.
             */
            void initForBlockSegments(MemorySegment[] segVars, long bStart) {
                this.sp = 0;
                this.flatVariables = null;
                this._2DVariables = null;
                this.flatVariablesF = null;
                this._2DVariablesF = null;
                this.segVariables = segVars;
                this.segBlockStart = bStart;
            }
        }

        private void applyBulkInternal(double[][] variables, EvaluationContext ctx, int dataSize, double[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = Math.min(BLOCK_SIZE, endIdx - blockStart);
                ctx.initForBlock(null, variables, dataSize, blockStart);
                try {
                    executeInstructions(ctx, currentBlockSize);
                } finally {
                    ctx.sp = 0;
                    java.util.Arrays.fill(ctx.stackIsConst, false);
                }
                System.arraycopy(ctx.stackArrays[0], ctx.stackOffsets[0], output, blockStart, currentBlockSize);
            }
        }

        private void applyBulkInternal(double[] flatVariables, EvaluationContext ctx, int dataSize, double[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = Math.min(BLOCK_SIZE, endIdx - blockStart);
                ctx.initForBlock(flatVariables, null, dataSize, blockStart);
                try {
                    executeInstructions(ctx, currentBlockSize);
                } finally {
                    ctx.sp = 0;
                    java.util.Arrays.fill(ctx.stackIsConst, false);
                }
                System.arraycopy(ctx.stackArrays[0], ctx.stackOffsets[0], output, blockStart, currentBlockSize);
            }
        }

        private void applyBulkInternal(MemorySegment variables, EvaluationContext ctx, long dataSize, MemorySegment output, long startIdx, long length) {
            final long endIdx = startIdx + length;
            for (long blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = (int) Math.min((long) BLOCK_SIZE, endIdx - blockStart);

                // Zero-allocation blit data straight into our threaded context primitive arrays
                for (int v = 0; v < varCount; v++) {
                    long srcOffsetBytes = ((v * dataSize) + blockStart) * 8L;
                    MemorySegment.copy(variables, srcOffsetBytes, ctx.segmentBlockVarsMem[v], 0L, currentBlockSize * 8L);
                }

                ctx.initForBlock(null, ctx.segmentBlockVars, currentBlockSize, 0);
                try {
                    executeInstructions(ctx, currentBlockSize);
                } finally {
                    ctx.sp = 0;
                    java.util.Arrays.fill(ctx.stackIsConst, false);
                }

                // Push evaluated vector outputs directly back out to foreign memory space
                long destOffsetBytes = blockStart * 8L;
                MemorySegment.copy(MemorySegment.ofArray(ctx.stackArrays[0]), ctx.stackOffsets[0] * 8L, output, destOffsetBytes, currentBlockSize * 8L);
            }
        }

        /**
         * NEW: Per-column MemorySegment[] variant of the bulk internal loop.
         * Identical block-staging strategy to the single-segment overload
         * above, except each variable is read from its own segment at
         * {@code blockStart * 8L} rather than from a computed offset within one
         * large concatenated segment. This is the entry point intended for
         * zero-copy Arrow integration: each element of {@code variables} can
         * point directly at one Arrow {@code ValueVector}'s data buffer, and
         * {@code output} can point directly at a pre-allocated output vector's
         * buffer.
         */
        private void applyBulkInternal(MemorySegment[] variables, EvaluationContext ctx, long dataSize, MemorySegment output, long startIdx, long length) {
            final long endIdx = startIdx + length;
            for (long blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = (int) Math.min((long) BLOCK_SIZE, endIdx - blockStart);

                // TRUE zero-copy: no staging copy here at all. Each OP_LOAD reads its
                // variable directly out of variables[slotIdx] at ctx.segBlockStart via
                // DoubleVector.fromMemorySegment / MemorySegment.getAtIndex. A variable
                // is only ever copied into on-heap scratch if some op that needs dense
                // array access (a transcendental function, POW, a comparison, IF/AND/OR,
                // VMA) actually consumes it — see materialize(). Pure +,-,*,/ chains over
                // loaded variables never touch scratch for their operands at all.
                ctx.initForBlockSegments(variables, blockStart);
                try {
                    executeInstructions(ctx, currentBlockSize);
                    if (ctx.stackIsSegment[0]) {
                        // Edge case: the entire expression is a bare variable (e.g. "x"),
                        // so OP_LOAD's segment-backed push was never consumed by any op
                        // that would otherwise force materialization. Materialize it now
                        // so the output copy below has a valid on-heap source.
                        materialize(ctx, 0, currentBlockSize);
                    }
                } finally {
                    ctx.sp = 0;
                    java.util.Arrays.fill(ctx.stackIsConst, false);
                    java.util.Arrays.fill(ctx.stackIsSegment, false);
                }

                // Push evaluated vector outputs directly back out to foreign memory space
                long destOffsetBytes = blockStart * 8L;
                MemorySegment.copy(MemorySegment.ofArray(ctx.stackArrays[0]), ctx.stackOffsets[0] * 8L, output, destOffsetBytes, currentBlockSize * 8L);
            }
        }

        /**
         * Lazily flushes a deferred stack item into the localized scratchpad.
         * Ensures legacy methods that require hard offsets (like VectorMath
         * calls) function normally.
         */
        private void materialize(EvaluationContext ctx, int targetSp, int n) {
            if (ctx.stackIsConst[targetSp]) {
                double val = ctx.stackConstVals[targetSp];
                int destOff = targetSp * BLOCK_SIZE;
                int k = 0;
                int bound = SPECIES.loopBound(n);
                DoubleVector v = DoubleVector.broadcast(SPECIES, val);
                for (; k < bound; k += SPECIES.length()) {
                    v.intoArray(ctx.scratch, destOff + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[destOff + k] = val;
                }
                ctx.stackArrays[targetSp] = ctx.scratch;
                ctx.stackOffsets[targetSp] = destOff;
                ctx.stackIsConst[targetSp] = false;
                ctx.stackIsSegment[targetSp] = false;
            } else if (ctx.stackIsSegment[targetSp]) {
                // NEW: on-demand copy of off-heap segment data into scratch — only paid
                // by ops that actually need dense array access (functions, comparisons,
                // POW, IF/AND/OR, VMA). Pure +,-,*,/ chains never reach this branch at
                // all; see doAdd/doSub/doMul/doDiv's segment-native fast paths below.
                int destOff = targetSp * BLOCK_SIZE;
                long srcOffsetBytes = ctx.stackSegOffsets[targetSp] * 8L;
                MemorySegment.copy(ctx.stackSegments[targetSp], srcOffsetBytes, ctx.scratchSegment, (long) destOff * 8L, (long) n * 8L);
                ctx.stackArrays[targetSp] = ctx.scratch;
                ctx.stackOffsets[targetSp] = destOff;
                ctx.stackIsSegment[targetSp] = false;
            } else if (ctx.stackArrays[targetSp] != ctx.scratch) {
                int destOff = targetSp * BLOCK_SIZE;
                System.arraycopy(ctx.stackArrays[targetSp], ctx.stackOffsets[targetSp], ctx.scratch, destOff, n);
                ctx.stackArrays[targetSp] = ctx.scratch;
                ctx.stackOffsets[targetSp] = destOff;
            } else {
                int expectedOff = targetSp * BLOCK_SIZE;
                if (ctx.stackOffsets[targetSp] != expectedOff) {
                    System.arraycopy(ctx.scratch, ctx.stackOffsets[targetSp], ctx.scratch, expectedOff, n);
                    ctx.stackOffsets[targetSp] = expectedOff;
                }
            }
        }

        // =========================================================================
        // Float mirrors of applyBulkInternal(...) / materialize(...). float[]
        // and float[][] variants are true overloads (distinct erasure from
        // double[]/double[][]); the MemorySegment variants are renamed with a
        // "Float" suffix since raw MemorySegment carries no element-type
        // information and would otherwise collide with the double overloads
        // above.
        // =========================================================================
        private void applyBulkInternal(float[][] variables, EvaluationContext ctx, int dataSize, float[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = Math.min(BLOCK_SIZE, endIdx - blockStart);
                ctx.initForBlockFloat(null, variables, dataSize, blockStart);
                try {
                    executeInstructionsFloat(ctx, currentBlockSize);
                } finally {
                    ctx.sp = 0;
                    java.util.Arrays.fill(ctx.stackIsConst, false);
                }
                System.arraycopy(ctx.stackArraysF[0], ctx.stackOffsets[0], output, blockStart, currentBlockSize);
            }
        }

        private void applyBulkInternal(float[] flatVariables, EvaluationContext ctx, int dataSize, float[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = Math.min(BLOCK_SIZE, endIdx - blockStart);
                ctx.initForBlockFloat(flatVariables, null, dataSize, blockStart);
                try {
                    executeInstructionsFloat(ctx, currentBlockSize);
                } finally {
                    ctx.sp = 0;
                    java.util.Arrays.fill(ctx.stackIsConst, false);
                }
                System.arraycopy(ctx.stackArraysF[0], ctx.stackOffsets[0], output, blockStart, currentBlockSize);
            }
        }

        private void applyBulkInternalFloat(MemorySegment variables, EvaluationContext ctx, long dataSize, MemorySegment output, long startIdx, long length) {
            final long endIdx = startIdx + length;
            for (long blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = (int) Math.min((long) BLOCK_SIZE, endIdx - blockStart);

                // Zero-allocation blit data straight into our threaded context primitive arrays
                for (int v = 0; v < varCount; v++) {
                    long srcOffsetBytes = ((v * dataSize) + blockStart) * 4L;
                    MemorySegment.copy(variables, srcOffsetBytes, ctx.segmentBlockVarsMemF[v], 0L, currentBlockSize * 4L);
                }

                ctx.initForBlockFloat(null, ctx.segmentBlockVarsF, currentBlockSize, 0);
                try {
                    executeInstructionsFloat(ctx, currentBlockSize);
                } finally {
                    ctx.sp = 0;
                    java.util.Arrays.fill(ctx.stackIsConst, false);
                }

                // Push evaluated vector outputs directly back out to foreign memory space
                long destOffsetBytes = blockStart * 4L;
                MemorySegment.copy(MemorySegment.ofArray(ctx.stackArraysF[0]), ctx.stackOffsets[0] * 4L, output, destOffsetBytes, currentBlockSize * 4L);
            }
        }

        /**
         * Float mirror of the per-column zero-copy MemorySegment[] bulk loop.
         */
        private void applyBulkInternalFloat(MemorySegment[] variables, EvaluationContext ctx, long dataSize, MemorySegment output, long startIdx, long length) {
            final long endIdx = startIdx + length;
            for (long blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = (int) Math.min((long) BLOCK_SIZE, endIdx - blockStart);

                // TRUE zero-copy: each OP_LOAD reads its variable directly out of
                // variables[slotIdx] at ctx.segBlockStart via FloatVector.fromMemorySegment
                // / MemorySegment.getAtIndex(JAVA_FLOAT, ...). Only materialized into
                // scratchF on demand — see materializeFloat().
                ctx.initForBlockSegments(variables, blockStart);
                try {
                    executeInstructionsFloat(ctx, currentBlockSize);
                    if (ctx.stackIsSegment[0]) {
                        // Edge case: the entire expression is a bare variable (e.g. "x"),
                        // so OP_LOAD's segment-backed push was never consumed by any op
                        // that would otherwise force materialization. Materialize it now
                        // so the output copy below has a valid on-heap source.
                        materializeFloat(ctx, 0, currentBlockSize);
                    }
                } finally {
                    ctx.sp = 0;
                    java.util.Arrays.fill(ctx.stackIsConst, false);
                    java.util.Arrays.fill(ctx.stackIsSegment, false);
                }

                // Push evaluated vector outputs directly back out to foreign memory space
                long destOffsetBytes = blockStart * 4L;
                MemorySegment.copy(MemorySegment.ofArray(ctx.stackArraysF[0]), ctx.stackOffsets[0] * 4L, output, destOffsetBytes, currentBlockSize * 4L);
            }
        }

        /**
         * Float mirror of {@link #materialize(EvaluationContext, int, int)}.
         */
        private void materializeFloat(EvaluationContext ctx, int targetSp, int n) {
            if (ctx.stackIsConst[targetSp]) {
                float val = ctx.stackConstValsF[targetSp];
                int destOff = targetSp * BLOCK_SIZE;
                int k = 0;
                int bound = SPECIES_F.loopBound(n);
                FloatVector v = FloatVector.broadcast(SPECIES_F, val);
                for (; k < bound; k += SPECIES_F.length()) {
                    v.intoArray(ctx.scratchF, destOff + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[destOff + k] = val;
                }
                ctx.stackArraysF[targetSp] = ctx.scratchF;
                ctx.stackOffsets[targetSp] = destOff;
                ctx.stackIsConst[targetSp] = false;
                ctx.stackIsSegment[targetSp] = false;
            } else if (ctx.stackIsSegment[targetSp]) {
                int destOff = targetSp * BLOCK_SIZE;
                long srcOffsetBytes = ctx.stackSegOffsets[targetSp] * 4L;
                MemorySegment.copy(ctx.stackSegments[targetSp], srcOffsetBytes, ctx.scratchSegmentF, (long) destOff * 4L, (long) n * 4L);
                ctx.stackArraysF[targetSp] = ctx.scratchF;
                ctx.stackOffsets[targetSp] = destOff;
                ctx.stackIsSegment[targetSp] = false;
            } else if (ctx.stackArraysF[targetSp] != ctx.scratchF) {
                int destOff = targetSp * BLOCK_SIZE;
                System.arraycopy(ctx.stackArraysF[targetSp], ctx.stackOffsets[targetSp], ctx.scratchF, destOff, n);
                ctx.stackArraysF[targetSp] = ctx.scratchF;
                ctx.stackOffsets[targetSp] = destOff;
            } else {
                int expectedOff = targetSp * BLOCK_SIZE;
                if (ctx.stackOffsets[targetSp] != expectedOff) {
                    System.arraycopy(ctx.scratchF, ctx.stackOffsets[targetSp], ctx.scratchF, expectedOff, n);
                    ctx.stackOffsets[targetSp] = expectedOff;
                }
            }
        }

        // =========================================================================
        // Core Execution Loop (Zero Allocation, C2 EA Guarded)
        // =========================================================================
        private void executeInstructions(EvaluationContext ctx, int n) {
            for (int instIdx = 0; instIdx < instructionCount; instIdx++) {
                final int opcode = opcodes[instIdx];

                switch (opcode) {
                    case OP_CONST -> {
                        ctx.stackIsConst[ctx.sp] = true;
                        ctx.stackIsSegment[ctx.sp] = false;
                        ctx.stackConstVals[ctx.sp] = literalConstants[instIdx];
                        ctx.sp++;
                    }

                    case OP_LOAD -> {
                        final int slotIdx = targetSlots[instIdx];
                        if (ctx.segVariables != null) {
                            // NEW: true zero-copy read — point the stack slot directly at the
                            // off-heap segment for this variable; no staging copy performed.
                            ctx.stackSegments[ctx.sp] = ctx.segVariables[slotIdx];
                            ctx.stackSegOffsets[ctx.sp] = ctx.segBlockStart;
                            ctx.stackIsSegment[ctx.sp] = true;
                            ctx.stackIsConst[ctx.sp] = false;
                        } else if (ctx.flatVariables != null) {
                            ctx.stackArrays[ctx.sp] = ctx.flatVariables;
                            ctx.stackOffsets[ctx.sp] = (slotIdx * ctx.dataSize) + ctx.blockStart;
                            ctx.stackIsSegment[ctx.sp] = false;
                            ctx.stackIsConst[ctx.sp] = false;
                        } else {
                            ctx.stackArrays[ctx.sp] = ctx._2DVariables[slotIdx];
                            ctx.stackOffsets[ctx.sp] = ctx.blockStart;
                            ctx.stackIsSegment[ctx.sp] = false;
                            ctx.stackIsConst[ctx.sp] = false;
                        }
                        ctx.sp++;
                    }

                    // --- Optimized Binary Operations (No Scratch Materialization) ---
                    case OP_ADD ->
                        doAdd(ctx, n);
                    case OP_SUB ->
                        doSub(ctx, n);
                    case OP_MUL ->
                        doMul(ctx, n);
                    case OP_DIV ->
                        doDiv(ctx, n);

                    case OP_POW -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);
                        materialize(ctx, ctx.sp + 1, n);
                        final int baseOffset = ctx.sp * BLOCK_SIZE;
                        final int expOffset = (ctx.sp + 1) * BLOCK_SIZE;

                        VectorMath.executePowerBlended(ctx.scratch, baseOffset, expOffset, n);

                        ctx.stackArrays[ctx.sp] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp] = baseOffset;
                        ctx.stackIsConst[ctx.sp] = false;
                        ctx.sp++;
                    }

                    case OP_SWIGLU_2 -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);
                        materialize(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, ctx.scratch, lOffset + k);
                            DoubleVector y = DoubleVector.fromArray(SPECIES, ctx.scratch, rOffset + k);
                            DoubleVector expNegX = VectorMath.fastVectorExp(x.neg());
                            DoubleVector denom = expNegX.add(ONE);
                            x.mul(y).div(denom).intoArray(ctx.scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratch[resOffset + k] = Maths.swiglu(ctx.scratch[lOffset + k], ctx.scratch[rOffset + k]);
                        }

                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_GEGLU_2 -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);
                        materialize(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        ctx.sp++;

                        final int loopBound = SPECIES.loopBound(n);
                        final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
                        final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);

                        int k = 0;
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, ctx.scratch, lOffset + k);
                            DoubleVector y = DoubleVector.fromArray(SPECIES, ctx.scratch, rOffset + k);
                            DoubleVector erfVal = VectorMath.vectorizedErf(y.mul(INV_SQRT_2));
                            DoubleVector geluY = y.mul(HALF).mul(erfVal.add(ONE));
                            x.mul(geluY).intoArray(ctx.scratch, lOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratch[lOffset + k] = Maths.geglu(ctx.scratch[lOffset + k], ctx.scratch[rOffset + k]);
                        }
                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = lOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_REM -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);
                        materialize(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;
                        for (int k = 0; k < n; k++) {
                            ctx.scratch[resOffset + k] = ctx.scratch[lOffset + k] % ctx.scratch[rOffset + k];
                        }
                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    // --- Base Unary Operations ---
                    case OP_SIN -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.sin((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COS -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.cos((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_TAN -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.tan((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_SINH -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.sinh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COSH -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.cosh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_TANH -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.tanh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_EXP -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.exp((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }

                    case OP_ABS -> {
                        materialize(ctx, ctx.sp - 1, n);
                        final int srcOffset = (ctx.sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            ctx.scratch[srcOffset + k] = Math.abs(ctx.scratch[srcOffset + k]);
                        }
                    }

                    case OP_SQRT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        final int srcOffset = (ctx.sp - 1) * BLOCK_SIZE;
                        VectorTranscendentals.evaluateNative(ctx.scratch, srcOffset, ctx.scratch, srcOffset, n, VectorOperators.SQRT);
                    }

                    case OP_CBRT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        final int srcOffset = (ctx.sp - 1) * BLOCK_SIZE;
                        VectorTranscendentals.evaluateNative(ctx.scratch, srcOffset, ctx.scratch, srcOffset, n, VectorOperators.CBRT);
                    }

                    case OP_SWIGLU -> {
                        materialize(ctx, ctx.sp - 1, n);
                        final int base = (ctx.sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, ctx.scratch, base + k);
                            DoubleVector expNegX = VectorMath.fastVectorExp(x.neg());
                            x.div(expNegX.add(ONE)).intoArray(ctx.scratch, base + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratch[base + k] = Maths.swiglu(ctx.scratch[base + k]);
                        }
                    }

                    case OP_GELU, OP_GEGLU, OP_GELU_FAST -> {
                        materialize(ctx, ctx.sp - 1, n);
                        final int base = (ctx.sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;
                        final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
                        final DoubleVector TWO = DoubleVector.broadcast(SPECIES, 2.0);

                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, ctx.scratch, base + k);
                            DoubleVector result;
                            if (opcode == OP_GELU) {
                                final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);
                                result = x.mul(HALF).mul(VectorMath.vectorizedErf(x.mul(INV_SQRT_2)).add(ONE));
                            } else if (opcode == OP_GELU_FAST) {
                                final DoubleVector SQRT_2_OVER_PI = DoubleVector.broadcast(SPECIES, 0.7978845608028654);
                                final DoubleVector COEF = DoubleVector.broadcast(SPECIES, 0.044715);
                                DoubleVector x3 = x.mul(x).mul(x);
                                DoubleVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);
                                DoubleVector exp2z = VectorMath.fastVectorExp(z.mul(TWO));
                                DoubleVector tanhZ = exp2z.sub(ONE).div(exp2z.add(ONE));
                                result = x.mul(HALF).mul(tanhZ.add(ONE));
                            } else {
                                result = x;
                            }
                            result.intoArray(ctx.scratch, base + k);
                        }
                        for (; k < n; k++) {
                            if (opcode == OP_GELU) {
                                ctx.scratch[base + k] = Maths.gelu(ctx.scratch[base + k]);
                            } else if (opcode == OP_GELU_FAST) {
                                ctx.scratch[base + k] = Maths.fastGelu(ctx.scratch[base + k]);
                            } else {
                                ctx.scratch[base + k] = Maths.geglu(ctx.scratch[base + k]);
                            }
                        }
                    }

                    case OP_ERF -> {
                        materialize(ctx, ctx.sp - 1, n);
                        final int base = (ctx.sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, ctx.scratch, base + k);
                            VectorMath.vectorizedErf(x).intoArray(ctx.scratch, base + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratch[base + k] = Maths.erf(ctx.scratch[base + k]);
                        }
                    }

                    // --- Delegated Math API Calls ---
                    case OP_LOG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.ln((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_LOG10 -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.log10((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ASIN, OP_ASIN_ALT, OP_ARC_SIN_ALT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.asin((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ACOS, OP_ACOS_ALT, OP_ARC_COS_ALT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acos((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ATAN, OP_ATAN_ALT, OP_ARC_TAN_ALT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.atan((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }

                    // (Extend for remaining Degree/Grad/Inverse implementations mirroring above)
                    case OP_SIN_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.sinDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COS_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.cosDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_TAN_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.tanDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
// Gradians
                    case OP_SIN_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.sinGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COS_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.cosGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_TAN_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.tanGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }

// Inverse Degree / Gradians
                    case OP_ASIN_DEG, OP_ASIN_DEG_ALT, OP_ARC_SIN_ALT_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.asinDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ACOS_DEG, OP_ACOS_DEG_ALT, OP_ARC_COS_ALT_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acosDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ATAN_DEG, OP_ATAN_DEG_ALT, OP_ARC_TAN_ALT_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.atanDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ASIN_GRAD, OP_ASIN_GRAD_ALT, OP_ARC_SIN_ALT_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.asinGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ACOS_GRAD, OP_ACOS_GRAD_ALT, OP_ARC_COS_ALT_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acosGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ATAN_GRAD, OP_ATAN_GRAD_ALT, OP_ARC_TAN_ALT_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.atanGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }

// Reciprocal Trig (SEC, CSC, COT) & Variants
                    case OP_SEC -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.sec((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_SEC_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.secDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_SEC_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.secGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COSEC -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.csc((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COSEC_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.cscDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COSEC_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.cscGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.cot((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COT_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.cotDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_COT_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.cotGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }

// Inverse Reciprocal Trig Variants
                    case OP_ARC_SEC, OP_ARC_SEC_ALT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.asec((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ARC_SEC_DEG, OP_ARC_SEC_ALT_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.asecDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ARC_SEC_GRAD, OP_ARC_SEC_ALT_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.asecGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ARC_COSEC, OP_ARC_COSEC_ALT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acsc((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ARC_COSEC_DEG, OP_ARC_COSEC_ALT_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acscDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ARC_COSEC_GRAD, OP_ARC_COSEC_ALT_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acscGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ARC_COT, OP_ARC_COT_ALT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acot((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ARC_COT_DEG, OP_ARC_COT_ALT_DEG -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acotDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ARC_COT_GRAD, OP_ARC_COT_ALT_GRAD -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acotGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }

// Hyperbolic Inverses
                    case OP_ASINH, OP_ASINH_ALT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.asinh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ACOSH, OP_ACOSH_ALT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.acosh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    case OP_ATANH, OP_ATANH_ALT -> {
                        materialize(ctx, ctx.sp - 1, n);
                        VectorMath.atanh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
                    }
                    // --- Conditionals & Comparisons ---

                    case OP_VMA -> {
                        ctx.sp -= 3;
                        materialize(ctx, ctx.sp, n);     // aOffset
                        materialize(ctx, ctx.sp + 1, n); // bOffset
                        materialize(ctx, ctx.sp + 2, n); // cOffset

                        final int aOffset = ctx.sp * BLOCK_SIZE;
                        final int bOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int cOffset = (ctx.sp + 2) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE; // Write into A's slot
                        ctx.sp++;

                        int k = 0;
                        int bound = SPECIES.loopBound(n);
                        for (; k < bound; k += SPECIES.length()) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, ctx.scratch, aOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, ctx.scratch, bOffset + k);
                            DoubleVector vc = DoubleVector.fromArray(SPECIES, ctx.scratch, cOffset + k);
                            va.fma(vb, vc).intoArray(ctx.scratch, resOffset + k);
                        }
                        if (k < n) {
                            var mask = SPECIES.indexInRange(k, n);
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, ctx.scratch, aOffset + k, mask);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, ctx.scratch, bOffset + k, mask);
                            DoubleVector vc = DoubleVector.fromArray(SPECIES, ctx.scratch, cOffset + k, mask);
                            va.fma(vb, vc).intoArray(ctx.scratch, resOffset + k, mask);
                        }

                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }
                    case OP_IF -> {
                        ctx.sp -= 3;
                        materialize(ctx, ctx.sp, n);     // cond
                        materialize(ctx, ctx.sp + 1, n); // true
                        materialize(ctx, ctx.sp + 2, n); // false
                        final int condOffset = ctx.sp * BLOCK_SIZE;
                        final int trueOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int falseOffset = (ctx.sp + 2) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        int k = 0;
                        final int vl = SPECIES.length();
                        final int limit = SPECIES.loopBound(n);
                        for (; k < limit; k += vl) {
                            DoubleVector cond = DoubleVector.fromArray(SPECIES, ctx.scratch, condOffset + k);
                            DoubleVector t = DoubleVector.fromArray(SPECIES, ctx.scratch, trueOffset + k);
                            DoubleVector f = DoubleVector.fromArray(SPECIES, ctx.scratch, falseOffset + k);
                            // NE-only mask, no NaN-exclusion clause -- matches Java's (cond != 0.0)
                            // exactly, including picking `t` when cond is NaN. This deliberately
                            // does NOT reuse VectorMath.if3's mask formula, which treats NaN cond
                            // as false and would silently change behavior here.
                            VectorMask<Double> mask = cond.compare(VectorOperators.NE, 0.0);
                            f.blend(t, mask).intoArray(ctx.scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratch[resOffset + k] = (ctx.scratch[condOffset + k] != 0.0)
                                    ? ctx.scratch[trueOffset + k] : ctx.scratch[falseOffset + k];
                        }

                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_GT -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);
                        materialize(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        int k = 0;
                        final int vl = SPECIES.length();
                        final int limit = SPECIES.loopBound(n);
                        for (; k < limit; k += vl) {
                            DoubleVector l = DoubleVector.fromArray(SPECIES, ctx.scratch, lOffset + k);
                            DoubleVector r = DoubleVector.fromArray(SPECIES, ctx.scratch, rOffset + k);
                            VectorMask<Double> mask = l.compare(VectorOperators.GT, r);
                            ZERO_D.blend(ONE_D, mask).intoArray(ctx.scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratch[resOffset + k] = ctx.scratch[lOffset + k] > ctx.scratch[rOffset + k] ? 1.0 : 0.0;
                        }

                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_LT, OP_EQ, OP_NE, OP_GE, OP_LE -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);
                        materialize(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        // Resolved once per case invocation, not per lane-block -- opcode is
                        // fixed for the whole call. JDK Vector API's lane comparisons are
                        // spec'd to match the corresponding primitive-double operator exactly,
                        // including IEEE-754 NaN behavior, so no NaN-guard is needed here
                        // (unlike OP_IF above, which had a real mismatch to work around).
                        final VectorOperators.Comparison cmpOp = switch (opcode) {
                            case OP_LT ->
                                VectorOperators.LT;
                            case OP_EQ ->
                                VectorOperators.EQ;
                            case OP_NE ->
                                VectorOperators.NE;
                            case OP_GE ->
                                VectorOperators.GE;
                            case OP_LE ->
                                VectorOperators.LE;
                            default ->
                                throw new IllegalStateException("unreachable: " + opcode);
                        };

                        int k = 0;
                        final int vl = SPECIES.length();
                        final int limit = SPECIES.loopBound(n);
                        for (; k < limit; k += vl) {
                            DoubleVector l = DoubleVector.fromArray(SPECIES, ctx.scratch, lOffset + k);
                            DoubleVector r = DoubleVector.fromArray(SPECIES, ctx.scratch, rOffset + k);
                            VectorMask<Double> mask = l.compare(cmpOp, r);
                            ZERO_D.blend(ONE_D, mask).intoArray(ctx.scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            double left = ctx.scratch[lOffset + k];
                            double right = ctx.scratch[rOffset + k];
                            boolean condition = switch (opcode) {
                                case OP_LT ->
                                    left < right;
                                case OP_EQ ->
                                    left == right;
                                case OP_NE ->
                                    left != right;
                                case OP_GE ->
                                    left >= right;
                                case OP_LE ->
                                    left <= right;
                                default ->
                                    false;
                            };
                            ctx.scratch[resOffset + k] = condition ? 1.0 : 0.0;
                        }

                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_AND -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);
                        materialize(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        int k = 0;
                        final int vl = SPECIES.length();
                        final int limit = SPECIES.loopBound(n);
                        for (; k < limit; k += vl) {
                            DoubleVector l = DoubleVector.fromArray(SPECIES, ctx.scratch, lOffset + k);
                            DoubleVector r = DoubleVector.fromArray(SPECIES, ctx.scratch, rOffset + k);
                            VectorMask<Double> mask = l.compare(VectorOperators.NE, 0.0).and(r.compare(VectorOperators.NE, 0.0));
                            ZERO_D.blend(ONE_D, mask).intoArray(ctx.scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratch[resOffset + k]
                                    = (ctx.scratch[lOffset + k] != 0.0 && ctx.scratch[rOffset + k] != 0.0) ? 1.0 : 0.0;
                        }

                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_OR -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);
                        materialize(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        int k = 0;
                        final int vl = SPECIES.length();
                        final int limit = SPECIES.loopBound(n);
                        for (; k < limit; k += vl) {
                            DoubleVector l = DoubleVector.fromArray(SPECIES, ctx.scratch, lOffset + k);
                            DoubleVector r = DoubleVector.fromArray(SPECIES, ctx.scratch, rOffset + k);
                            VectorMask<Double> mask = l.compare(VectorOperators.NE, 0.0).or(r.compare(VectorOperators.NE, 0.0));
                            ZERO_D.blend(ONE_D, mask).intoArray(ctx.scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratch[resOffset + k]
                                    = (ctx.scratch[lOffset + k] != 0.0 || ctx.scratch[rOffset + k] != 0.0) ? 1.0 : 0.0;
                        }

                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    default ->
                        throw new UnsupportedOperationException("Unknown/Unmapped opcode: " + opcode);
                }
            }
        }

        private void executeInstructionsFloat(EvaluationContext ctx, int n) {
            for (int instIdx = 0; instIdx < instructionCount; instIdx++) {
                final int opcode = opcodes[instIdx];

                switch (opcode) {
                    case OP_CONST -> {
                        ctx.stackIsConst[ctx.sp] = true;
                        ctx.stackIsSegment[ctx.sp] = false;
                        ctx.stackConstValsF[ctx.sp] = (float) literalConstants[instIdx];
                        ctx.sp++;
                    }

                    case OP_LOAD -> {
                        final int slotIdx = targetSlots[instIdx];
                        if (ctx.segVariables != null) {
                            // NEW: true zero-copy read — point the stack slot directly at the
                            // off-heap segment for this variable; no staging copy performed.
                            ctx.stackSegments[ctx.sp] = ctx.segVariables[slotIdx];
                            ctx.stackSegOffsets[ctx.sp] = ctx.segBlockStart;
                            ctx.stackIsSegment[ctx.sp] = true;
                            ctx.stackIsConst[ctx.sp] = false;
                        } else if (ctx.flatVariablesF != null) {
                            ctx.stackArraysF[ctx.sp] = ctx.flatVariablesF;
                            ctx.stackOffsets[ctx.sp] = (slotIdx * ctx.dataSize) + ctx.blockStart;
                            ctx.stackIsSegment[ctx.sp] = false;
                            ctx.stackIsConst[ctx.sp] = false;
                        } else {
                            ctx.stackArraysF[ctx.sp] = ctx._2DVariablesF[slotIdx];
                            ctx.stackOffsets[ctx.sp] = ctx.blockStart;
                            ctx.stackIsSegment[ctx.sp] = false;
                            ctx.stackIsConst[ctx.sp] = false;
                        }
                        ctx.sp++;
                    }

                    // --- Optimized Binary Operations (No Scratch Materialization) ---
                    case OP_ADD ->
                        doAddF(ctx, n);
                    case OP_SUB ->
                        doSubF(ctx, n);
                    case OP_MUL ->
                        doMulF(ctx, n);
                    case OP_DIV ->
                        doDivF(ctx, n);

                    case OP_POW -> {
                        ctx.sp -= 2;
                        materializeFloat(ctx, ctx.sp, n);
                        materializeFloat(ctx, ctx.sp + 1, n);
                        final int baseOffset = ctx.sp * BLOCK_SIZE;
                        final int expOffset = (ctx.sp + 1) * BLOCK_SIZE;

                        VectorMath.executePowerBlended(ctx.scratchF, baseOffset, expOffset, n);

                        ctx.stackArraysF[ctx.sp] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp] = baseOffset;
                        ctx.stackIsConst[ctx.sp] = false;
                        ctx.sp++;
                    }

                    case OP_SWIGLU_2 -> {
                        ctx.sp -= 2;
                        materializeFloat(ctx, ctx.sp, n);
                        materializeFloat(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        final int loopBound = SPECIES_F.loopBound(n);
                        int k = 0;
                        final FloatVector ONE = FloatVector.broadcast(SPECIES_F, 1.0f);
                        for (; k < loopBound; k += SPECIES_F.length()) {
                            FloatVector x = FloatVector.fromArray(SPECIES_F, ctx.scratchF, lOffset + k);
                            FloatVector y = FloatVector.fromArray(SPECIES_F, ctx.scratchF, rOffset + k);
                            FloatVector expNegX = VectorMath.fastVectorExp(x.neg());
                            FloatVector denom = expNegX.add(ONE);
                            x.mul(y).div(denom).intoArray(ctx.scratchF, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratchF[resOffset + k] = (float) Maths.swiglu(ctx.scratchF[lOffset + k], ctx.scratchF[rOffset + k]);
                        }

                        ctx.stackArraysF[ctx.sp - 1] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_GEGLU_2 -> {
                        ctx.sp -= 2;
                        materializeFloat(ctx, ctx.sp, n);
                        materializeFloat(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        ctx.sp++;

                        final int loopBound = SPECIES_F.loopBound(n);
                        final FloatVector HALF = FloatVector.broadcast(SPECIES_F, 0.5f);
                        final FloatVector ONE = FloatVector.broadcast(SPECIES_F, 1.0f);
                        final FloatVector INV_SQRT_2 = FloatVector.broadcast(SPECIES_F, 0.7071067811865476f);

                        int k = 0;
                        for (; k < loopBound; k += SPECIES_F.length()) {
                            FloatVector x = FloatVector.fromArray(SPECIES_F, ctx.scratchF, lOffset + k);
                            FloatVector y = FloatVector.fromArray(SPECIES_F, ctx.scratchF, rOffset + k);
                            FloatVector erfVal = VectorMath.vectorizedErf(y.mul(INV_SQRT_2));
                            FloatVector geluY = y.mul(HALF).mul(erfVal.add(ONE));
                            x.mul(geluY).intoArray(ctx.scratchF, lOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratchF[lOffset + k] = (float) Maths.geglu(ctx.scratchF[lOffset + k], ctx.scratchF[rOffset + k]);
                        }
                        ctx.stackArraysF[ctx.sp - 1] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp - 1] = lOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_REM -> {
                        ctx.sp -= 2;
                        materializeFloat(ctx, ctx.sp, n);
                        materializeFloat(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;
                        for (int k = 0; k < n; k++) {
                            ctx.scratchF[resOffset + k] = ctx.scratchF[lOffset + k] % ctx.scratchF[rOffset + k];
                        }
                        ctx.stackArraysF[ctx.sp - 1] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    // --- Base Unary Operations ---
                    case OP_SIN -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.sin((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COS -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.cos((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_TAN -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.tan((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_SINH -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.sinh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COSH -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.cosh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_TANH -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.tanh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_EXP -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.exp((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }

                    case OP_ABS -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        final int srcOffset = (ctx.sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            ctx.scratchF[srcOffset + k] = Math.abs(ctx.scratchF[srcOffset + k]);
                        }
                    }

                    case OP_SQRT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        final int srcOffset = (ctx.sp - 1) * BLOCK_SIZE;
                        VectorTranscendentals.evaluateNative(ctx.scratchF, srcOffset, ctx.scratchF, srcOffset, n, VectorOperators.SQRT);
                    }

                    case OP_CBRT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        final int srcOffset = (ctx.sp - 1) * BLOCK_SIZE;
                        VectorTranscendentals.evaluateNative(ctx.scratchF, srcOffset, ctx.scratchF, srcOffset, n, VectorOperators.CBRT);
                    }

                    case OP_SWIGLU -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        final int base = (ctx.sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES_F.loopBound(n);
                        int k = 0;
                        final FloatVector ONE = FloatVector.broadcast(SPECIES_F, 1.0f);
                        for (; k < loopBound; k += SPECIES_F.length()) {
                            FloatVector x = FloatVector.fromArray(SPECIES_F, ctx.scratchF, base + k);
                            FloatVector expNegX = VectorMath.fastVectorExp(x.neg());
                            x.div(expNegX.add(ONE)).intoArray(ctx.scratchF, base + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratchF[base + k] = (float) Maths.swiglu(ctx.scratchF[base + k]);
                        }
                    }

                    case OP_GELU, OP_GEGLU, OP_GELU_FAST -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        final int base = (ctx.sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES_F.loopBound(n);
                        int k = 0;
                        final FloatVector HALF = FloatVector.broadcast(SPECIES_F, 0.5f);
                        final FloatVector ONE = FloatVector.broadcast(SPECIES_F, 1.0f);
                        final FloatVector TWO = FloatVector.broadcast(SPECIES_F, 2.0f);

                        for (; k < loopBound; k += SPECIES_F.length()) {
                            FloatVector x = FloatVector.fromArray(SPECIES_F, ctx.scratchF, base + k);
                            FloatVector result;
                            if (opcode == OP_GELU) {
                                final FloatVector INV_SQRT_2 = FloatVector.broadcast(SPECIES_F, 0.7071067811865476f);
                                result = x.mul(HALF).mul(VectorMath.vectorizedErf(x.mul(INV_SQRT_2)).add(ONE));
                            } else if (opcode == OP_GELU_FAST) {
                                final FloatVector SQRT_2_OVER_PI = FloatVector.broadcast(SPECIES_F, 0.7978845608028654f);
                                final FloatVector COEF = FloatVector.broadcast(SPECIES_F, 0.044715f);
                                FloatVector x3 = x.mul(x).mul(x);
                                FloatVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);
                                FloatVector exp2z = VectorMath.fastVectorExp(z.mul(TWO));
                                FloatVector tanhZ = exp2z.sub(ONE).div(exp2z.add(ONE));
                                result = x.mul(HALF).mul(tanhZ.add(ONE));
                            } else {
                                result = x;
                            }
                            result.intoArray(ctx.scratchF, base + k);
                        }
                        for (; k < n; k++) {
                            if (opcode == OP_GELU) {
                                ctx.scratchF[base + k] = (float) Maths.gelu(ctx.scratchF[base + k]);
                            } else if (opcode == OP_GELU_FAST) {
                                ctx.scratchF[base + k] = (float) Maths.fastGelu(ctx.scratchF[base + k]);
                            } else {
                                ctx.scratchF[base + k] = (float) Maths.geglu(ctx.scratchF[base + k]);
                            }
                        }
                    }

                    case OP_ERF -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        final int base = (ctx.sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES_F.loopBound(n);
                        int k = 0;
                        for (; k < loopBound; k += SPECIES_F.length()) {
                            FloatVector x = FloatVector.fromArray(SPECIES_F, ctx.scratchF, base + k);
                            VectorMath.vectorizedErf(x).intoArray(ctx.scratchF, base + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratchF[base + k] = (float) Maths.erf(ctx.scratchF[base + k]);
                        }
                    }

                    // --- Delegated Math API Calls ---
                    case OP_LOG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.ln((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_LOG10 -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.log10((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ASIN, OP_ASIN_ALT, OP_ARC_SIN_ALT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.asin((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ACOS, OP_ACOS_ALT, OP_ARC_COS_ALT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acos((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ATAN, OP_ATAN_ALT, OP_ARC_TAN_ALT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.atan((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }

                    // (Extend for remaining Degree/Grad/Inverse implementations mirroring above)
                    case OP_SIN_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.sinDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COS_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.cosDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_TAN_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.tanDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
// Gradians
                    case OP_SIN_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.sinGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COS_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.cosGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_TAN_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.tanGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }

// Inverse Degree / Gradians
                    case OP_ASIN_DEG, OP_ASIN_DEG_ALT, OP_ARC_SIN_ALT_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.asinDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ACOS_DEG, OP_ACOS_DEG_ALT, OP_ARC_COS_ALT_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acosDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ATAN_DEG, OP_ATAN_DEG_ALT, OP_ARC_TAN_ALT_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.atanDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ASIN_GRAD, OP_ASIN_GRAD_ALT, OP_ARC_SIN_ALT_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.asinGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ACOS_GRAD, OP_ACOS_GRAD_ALT, OP_ARC_COS_ALT_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acosGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ATAN_GRAD, OP_ATAN_GRAD_ALT, OP_ARC_TAN_ALT_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.atanGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }

// Reciprocal Trig (SEC, CSC, COT) & Variants
                    case OP_SEC -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.sec((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_SEC_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.secDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_SEC_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.secGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COSEC -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.csc((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COSEC_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.cscDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COSEC_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.cscGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.cot((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COT_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.cotDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_COT_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.cotGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }

// Inverse Reciprocal Trig Variants
                    case OP_ARC_SEC, OP_ARC_SEC_ALT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.asec((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ARC_SEC_DEG, OP_ARC_SEC_ALT_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.asecDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ARC_SEC_GRAD, OP_ARC_SEC_ALT_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.asecGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ARC_COSEC, OP_ARC_COSEC_ALT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acsc((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ARC_COSEC_DEG, OP_ARC_COSEC_ALT_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acscDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ARC_COSEC_GRAD, OP_ARC_COSEC_ALT_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acscGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ARC_COT, OP_ARC_COT_ALT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acot((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ARC_COT_DEG, OP_ARC_COT_ALT_DEG -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acotDeg((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ARC_COT_GRAD, OP_ARC_COT_ALT_GRAD -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acotGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }

// Hyperbolic Inverses
                    case OP_ASINH, OP_ASINH_ALT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.asinh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ACOSH, OP_ACOSH_ALT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.acosh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    case OP_ATANH, OP_ATANH_ALT -> {
                        materializeFloat(ctx, ctx.sp - 1, n);
                        VectorMath.atanh((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratchF);
                    }
                    // --- Conditionals & Comparisons ---

                    case OP_VMA -> {
                        ctx.sp -= 3;
                        materializeFloat(ctx, ctx.sp, n);     // aOffset
                        materializeFloat(ctx, ctx.sp + 1, n); // bOffset
                        materializeFloat(ctx, ctx.sp + 2, n); // cOffset

                        final int aOffset = ctx.sp * BLOCK_SIZE;
                        final int bOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int cOffset = (ctx.sp + 2) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE; // Write into A's slot
                        ctx.sp++;

                        int k = 0;
                        int bound = SPECIES_F.loopBound(n);
                        for (; k < bound; k += SPECIES_F.length()) {
                            FloatVector va  = FloatVector.fromArray(SPECIES_F, ctx.scratchF, aOffset + k);
                            FloatVector vb = FloatVector.fromArray(SPECIES_F, ctx.scratchF, bOffset + k);
                            FloatVector vc = FloatVector.fromArray(SPECIES_F, ctx.scratchF, cOffset + k);
                            va.fma(vb, vc).intoArray(ctx.scratchF, resOffset + k);
                        }
                        if (k < n) {
                            var mask = SPECIES_F.indexInRange(k, n);
                            FloatVector va  = FloatVector.fromArray(SPECIES_F, ctx.scratchF, aOffset + k, mask);
                            FloatVector vb = FloatVector.fromArray(SPECIES_F, ctx.scratchF, bOffset + k, mask);
                            FloatVector vc = FloatVector.fromArray(SPECIES_F, ctx.scratchF, cOffset + k, mask);
                            va.fma(vb, vc).intoArray(ctx.scratchF, resOffset + k, mask);
                        }

                        ctx.stackArraysF[ctx.sp - 1] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }
                    case OP_IF -> {
                        ctx.sp -= 3;
                        materializeFloat(ctx, ctx.sp, n);     // cond
                        materializeFloat(ctx, ctx.sp + 1, n); // true
                        materializeFloat(ctx, ctx.sp + 2, n); // false
                        final int condOffset = ctx.sp * BLOCK_SIZE;
                        final int trueOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int falseOffset = (ctx.sp + 2) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        int k = 0;
                        final int vl = SPECIES_F.length();
                        final int limit = SPECIES_F.loopBound(n);
                        for (; k < limit; k += vl) {
                            FloatVector cond = FloatVector.fromArray(SPECIES_F, ctx.scratchF, condOffset + k);
                            FloatVector t = FloatVector.fromArray(SPECIES_F, ctx.scratchF, trueOffset + k);
                            FloatVector f = FloatVector.fromArray(SPECIES_F, ctx.scratchF, falseOffset + k);
                            // NE-only mask, no NaN-exclusion clause -- matches Java's (cond != 0.0f)
                            // exactly, including picking `t` when cond is NaN. This deliberately
                            // does NOT reuse VectorMath.if3's mask formula, which treats NaN cond
                            // as false and would silently change behavior here.
                            VectorMask<Float> mask = cond.compare(VectorOperators.NE, 0.0f);
                            f.blend(t, mask).intoArray(ctx.scratchF, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratchF[resOffset + k] = (ctx.scratchF[condOffset + k] != 0.0f)
                                    ? ctx.scratchF[trueOffset + k] : ctx.scratchF[falseOffset + k];
                        }

                        ctx.stackArraysF[ctx.sp - 1] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_GT -> {
                        ctx.sp -= 2;
                        materializeFloat(ctx, ctx.sp, n);
                        materializeFloat(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        int k = 0;
                        final int vl = SPECIES_F.length();
                        final int limit = SPECIES_F.loopBound(n);
                        for (; k < limit; k += vl) {
                            FloatVector l = FloatVector.fromArray(SPECIES_F, ctx.scratchF, lOffset + k);
                            FloatVector r = FloatVector.fromArray(SPECIES_F, ctx.scratchF, rOffset + k);
                            VectorMask<Float> mask = l.compare(VectorOperators.GT, r);
                            ZERO_F.blend(ONE_F, mask).intoArray(ctx.scratchF, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratchF[resOffset + k] = ctx.scratchF[lOffset + k] > ctx.scratchF[rOffset + k] ? 1.0f : 0.0f;
                        }

                        ctx.stackArraysF[ctx.sp - 1] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_LT, OP_EQ, OP_NE, OP_GE, OP_LE -> {
                        ctx.sp -= 2;
                        materializeFloat(ctx, ctx.sp, n);
                        materializeFloat(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        // Resolved once per case invocation, not per lane-block -- opcode is
                        // fixed for the whole call. JDK Vector API's lane comparisons are
                        // spec'd to match the corresponding primitive-double operator exactly,
                        // including IEEE-754 NaN behavior, so no NaN-guard is needed here
                        // (unlike OP_IF above, which had a real mismatch to work around).
                        final VectorOperators.Comparison cmpOp = switch (opcode) {
                            case OP_LT ->
                                VectorOperators.LT;
                            case OP_EQ ->
                                VectorOperators.EQ;
                            case OP_NE ->
                                VectorOperators.NE;
                            case OP_GE ->
                                VectorOperators.GE;
                            case OP_LE ->
                                VectorOperators.LE;
                            default ->
                                throw new IllegalStateException("unreachable: " + opcode);
                        };

                        int k = 0;
                        final int vl = SPECIES_F.length();
                        final int limit = SPECIES_F.loopBound(n);
                        for (; k < limit; k += vl) {
                            FloatVector l = FloatVector.fromArray(SPECIES_F, ctx.scratchF, lOffset + k);
                            FloatVector r = FloatVector.fromArray(SPECIES_F, ctx.scratchF, rOffset + k);
                            VectorMask<Float> mask = l.compare(cmpOp, r);
                            ZERO_F.blend(ONE_F, mask).intoArray(ctx.scratchF, resOffset + k);
                        }
                        for (; k < n; k++) {
                            double left = ctx.scratchF[lOffset + k];
                            double right = ctx.scratchF[rOffset + k];
                            boolean condition = switch (opcode) {
                                case OP_LT ->
                                    left < right;
                                case OP_EQ ->
                                    left == right;
                                case OP_NE ->
                                    left != right;
                                case OP_GE ->
                                    left >= right;
                                case OP_LE ->
                                    left <= right;
                                default ->
                                    false;
                            };
                            ctx.scratchF[resOffset + k] = condition ? 1.0f : 0.0f;
                        }

                        ctx.stackArraysF[ctx.sp - 1] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_AND -> {
                        ctx.sp -= 2;
                        materializeFloat(ctx, ctx.sp, n);
                        materializeFloat(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        int k = 0;
                        final int vl = SPECIES_F.length();
                        final int limit = SPECIES_F.loopBound(n);
                        for (; k < limit; k += vl) {
                            FloatVector l = FloatVector.fromArray(SPECIES_F, ctx.scratchF, lOffset + k);
                            FloatVector r = FloatVector.fromArray(SPECIES_F, ctx.scratchF, rOffset + k);
                            VectorMask<Float> mask = l.compare(VectorOperators.NE, 0.0f).and(r.compare(VectorOperators.NE, 0.0f));
                            ZERO_F.blend(ONE_F, mask).intoArray(ctx.scratchF, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratchF[resOffset + k]
                                    = (ctx.scratchF[lOffset + k] != 0.0f && ctx.scratchF[rOffset + k] != 0.0f) ? 1.0f : 0.0f;
                        }

                        ctx.stackArraysF[ctx.sp - 1] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_OR -> {
                        ctx.sp -= 2;
                        materializeFloat(ctx, ctx.sp, n);
                        materializeFloat(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;

                        int k = 0;
                        final int vl = SPECIES_F.length();
                        final int limit = SPECIES_F.loopBound(n);
                        for (; k < limit; k += vl) {
                            FloatVector l = FloatVector.fromArray(SPECIES_F, ctx.scratchF, lOffset + k);
                            FloatVector r = FloatVector.fromArray(SPECIES_F, ctx.scratchF, rOffset + k);
                            VectorMask<Float> mask = l.compare(VectorOperators.NE, 0.0f).or(r.compare(VectorOperators.NE, 0.0f));
                            ZERO_F.blend(ONE_F, mask).intoArray(ctx.scratchF, resOffset + k);
                        }
                        for (; k < n; k++) {
                            ctx.scratchF[resOffset + k]
                                    = (ctx.scratchF[lOffset + k] != 0.0f || ctx.scratchF[rOffset + k] != 0.0f) ? 1.0f : 0.0f;
                        }

                        ctx.stackArraysF[ctx.sp - 1] = ctx.scratchF;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    default ->
                        throw new UnsupportedOperationException("Unknown/Unmapped opcode: " + opcode);
                }
            }
        }

        // =========================================================================
        // Delegated Micro-Methods for Binary Operations (EA Guarded)---
        // =========================================================================
        private void doAdd(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final boolean rIsSeg = ctx.stackIsSegment[rSp];
            final double[] rArr = ctx.stackArrays[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final double rVal = ctx.stackConstVals[rSp];
            final MemorySegment rSeg = ctx.stackSegments[rSp];
            final long rSegOff = ctx.stackSegOffsets[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final boolean lIsSeg = ctx.stackIsSegment[lSp];
            final double[] lArr = ctx.stackArrays[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final double lVal = ctx.stackConstVals[lSp];
            final MemorySegment lSeg = ctx.stackSegments[lSp];
            final long lSegOff = ctx.stackSegOffsets[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArrays[ctx.sp] = ctx.scratch;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.stackIsSegment[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);

            if (lIsSeg && rIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .add(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) + rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (lIsSeg && rIsConst) {
                final DoubleVector rbVec = DoubleVector.broadcast(SPECIES, rVal);
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .add(rbVec)
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) + rVal;
                }
            } else if (lIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .add(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) + rArr[rOff + k];
                }
            } else if (rIsSeg && lIsConst) {
                final DoubleVector laVec = DoubleVector.broadcast(SPECIES, lVal);
                for (; k < limit; k += vl) {
                    laVec.add(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lVal + rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (rIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .add(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] + rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (!lIsConst && !rIsConst) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .add(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] + rArr[rOff + k];
                }
            } else if (!lIsConst && rIsConst) {
                final DoubleVector rbVec = DoubleVector.broadcast(SPECIES, rVal);
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .add(rbVec)
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] + rVal;
                }
            } else if (lIsConst && !rIsConst) {
                final DoubleVector laVec = DoubleVector.broadcast(SPECIES, lVal);
                for (; k < limit; k += vl) {
                    laVec.add(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lVal + rArr[rOff + k];
                }
            } else {
                ctx.sp--;
                ctx.stackIsConst[ctx.sp] = true;
                ctx.stackConstVals[ctx.sp] = lVal + rVal;
                ctx.sp++;
            }
        }

        private void doAddF(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final boolean rIsSeg = ctx.stackIsSegment[rSp];
            final float[] rArr = ctx.stackArraysF[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final float rVal = ctx.stackConstValsF[rSp];
            final MemorySegment rSeg = ctx.stackSegments[rSp];
            final long rSegOff = ctx.stackSegOffsets[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final boolean lIsSeg = ctx.stackIsSegment[lSp];
            final float[] lArr = ctx.stackArraysF[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final float lVal = ctx.stackConstValsF[lSp];
            final MemorySegment lSeg = ctx.stackSegments[lSp];
            final long lSegOff = ctx.stackSegOffsets[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArraysF[ctx.sp] = ctx.scratchF;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.stackIsSegment[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES_F.length();
            final int limit = SPECIES_F.loopBound(n);

            if (lIsSeg && rIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .add(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) + rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (lIsSeg && rIsConst) {
                final FloatVector rbVec = FloatVector.broadcast(SPECIES_F, rVal);
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .add(rbVec)
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) + rVal;
                }
            } else if (lIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .add(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) + rArr[rOff + k];
                }
            } else if (rIsSeg && lIsConst) {
                final FloatVector laVec = FloatVector.broadcast(SPECIES_F, lVal);
                for (; k < limit; k += vl) {
                    laVec.add(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lVal + rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (rIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .add(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] + rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (!lIsConst && !rIsConst) {
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .add(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] + rArr[rOff + k];
                }
            } else if (!lIsConst && rIsConst) {
                final FloatVector rbVec = FloatVector.broadcast(SPECIES_F, rVal);
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .add(rbVec)
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] + rVal;
                }
            } else if (lIsConst && !rIsConst) {
                final FloatVector laVec = FloatVector.broadcast(SPECIES_F, lVal);
                for (; k < limit; k += vl) {
                    laVec.add(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lVal + rArr[rOff + k];
                }
            } else {
                ctx.sp--;
                ctx.stackIsConst[ctx.sp] = true;
                ctx.stackConstValsF[ctx.sp] = lVal + rVal;
                ctx.sp++;
            }
        }

        private void doSub(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final boolean rIsSeg = ctx.stackIsSegment[rSp];
            final double[] rArr = ctx.stackArrays[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final double rVal = ctx.stackConstVals[rSp];
            final MemorySegment rSeg = ctx.stackSegments[rSp];
            final long rSegOff = ctx.stackSegOffsets[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final boolean lIsSeg = ctx.stackIsSegment[lSp];
            final double[] lArr = ctx.stackArrays[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final double lVal = ctx.stackConstVals[lSp];
            final MemorySegment lSeg = ctx.stackSegments[lSp];
            final long lSegOff = ctx.stackSegOffsets[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArrays[ctx.sp] = ctx.scratch;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.stackIsSegment[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);

            if (lIsSeg && rIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .sub(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) - rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (lIsSeg && rIsConst) {
                final DoubleVector rbVec = DoubleVector.broadcast(SPECIES, rVal);
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .sub(rbVec)
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) - rVal;
                }
            } else if (lIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .sub(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) - rArr[rOff + k];
                }
            } else if (rIsSeg && lIsConst) {
                final DoubleVector laVec = DoubleVector.broadcast(SPECIES, lVal);
                for (; k < limit; k += vl) {
                    laVec.sub(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lVal - rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (rIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .sub(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] - rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (!lIsConst && !rIsConst) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .sub(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] - rArr[rOff + k];
                }
            } else if (!lIsConst && rIsConst) {
                final DoubleVector rbVec = DoubleVector.broadcast(SPECIES, rVal);
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .sub(rbVec)
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] - rVal;
                }
            } else if (lIsConst && !rIsConst) {
                final DoubleVector laVec = DoubleVector.broadcast(SPECIES, lVal);
                for (; k < limit; k += vl) {
                    laVec.sub(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lVal - rArr[rOff + k];
                }
            } else {
                ctx.sp--;
                ctx.stackIsConst[ctx.sp] = true;
                ctx.stackConstVals[ctx.sp] = lVal - rVal;
                ctx.sp++;
            }
        }

        private void doSubF(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final boolean rIsSeg = ctx.stackIsSegment[rSp];
            final float[] rArr = ctx.stackArraysF[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final float rVal = ctx.stackConstValsF[rSp];
            final MemorySegment rSeg = ctx.stackSegments[rSp];
            final long rSegOff = ctx.stackSegOffsets[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final boolean lIsSeg = ctx.stackIsSegment[lSp];
            final float[] lArr = ctx.stackArraysF[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final float lVal = ctx.stackConstValsF[lSp];
            final MemorySegment lSeg = ctx.stackSegments[lSp];
            final long lSegOff = ctx.stackSegOffsets[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArraysF[ctx.sp] = ctx.scratchF;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.stackIsSegment[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES_F.length();
            final int limit = SPECIES_F.loopBound(n);

            if (lIsSeg && rIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .sub(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) - rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (lIsSeg && rIsConst) {
                final FloatVector rbVec = FloatVector.broadcast(SPECIES_F, rVal);
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .sub(rbVec)
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) - rVal;
                }
            } else if (lIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .sub(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) - rArr[rOff + k];
                }
            } else if (rIsSeg && lIsConst) {
                final FloatVector laVec = FloatVector.broadcast(SPECIES_F, lVal);
                for (; k < limit; k += vl) {
                    laVec.sub(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lVal - rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (rIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .sub(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] - rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (!lIsConst && !rIsConst) {
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .sub(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] - rArr[rOff + k];
                }
            } else if (!lIsConst && rIsConst) {
                final FloatVector rbVec = FloatVector.broadcast(SPECIES_F, rVal);
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .sub(rbVec)
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] - rVal;
                }
            } else if (lIsConst && !rIsConst) {
                final FloatVector laVec = FloatVector.broadcast(SPECIES_F, lVal);
                for (; k < limit; k += vl) {
                    laVec.sub(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lVal - rArr[rOff + k];
                }
            } else {
                ctx.sp--;
                ctx.stackIsConst[ctx.sp] = true;
                ctx.stackConstValsF[ctx.sp] = lVal - rVal;
                ctx.sp++;
            }
        }

        private void doMul(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final boolean rIsSeg = ctx.stackIsSegment[rSp];
            final double[] rArr = ctx.stackArrays[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final double rVal = ctx.stackConstVals[rSp];
            final MemorySegment rSeg = ctx.stackSegments[rSp];
            final long rSegOff = ctx.stackSegOffsets[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final boolean lIsSeg = ctx.stackIsSegment[lSp];
            final double[] lArr = ctx.stackArrays[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final double lVal = ctx.stackConstVals[lSp];
            final MemorySegment lSeg = ctx.stackSegments[lSp];
            final long lSegOff = ctx.stackSegOffsets[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArrays[ctx.sp] = ctx.scratch;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.stackIsSegment[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);

            if (lIsSeg && rIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .mul(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) * rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (lIsSeg && rIsConst) {
                final DoubleVector rbVec = DoubleVector.broadcast(SPECIES, rVal);
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .mul(rbVec)
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) * rVal;
                }
            } else if (lIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .mul(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) * rArr[rOff + k];
                }
            } else if (rIsSeg && lIsConst) {
                final DoubleVector laVec = DoubleVector.broadcast(SPECIES, lVal);
                for (; k < limit; k += vl) {
                    laVec.mul(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lVal * rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (rIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .mul(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] * rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (!lIsConst && !rIsConst) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .mul(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] * rArr[rOff + k];
                }
            } else if (!lIsConst && rIsConst) {
                final DoubleVector rbVec = DoubleVector.broadcast(SPECIES, rVal);
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .mul(rbVec)
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] * rVal;
                }
            } else if (lIsConst && !rIsConst) {
                final DoubleVector laVec = DoubleVector.broadcast(SPECIES, lVal);
                for (; k < limit; k += vl) {
                    laVec.mul(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lVal * rArr[rOff + k];
                }
            } else {
                ctx.sp--;
                ctx.stackIsConst[ctx.sp] = true;
                ctx.stackConstVals[ctx.sp] = lVal * rVal;
                ctx.sp++;
            }
        }

        private void doMulF(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final boolean rIsSeg = ctx.stackIsSegment[rSp];
            final float[] rArr = ctx.stackArraysF[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final float rVal = ctx.stackConstValsF[rSp];
            final MemorySegment rSeg = ctx.stackSegments[rSp];
            final long rSegOff = ctx.stackSegOffsets[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final boolean lIsSeg = ctx.stackIsSegment[lSp];
            final float[] lArr = ctx.stackArraysF[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final float lVal = ctx.stackConstValsF[lSp];
            final MemorySegment lSeg = ctx.stackSegments[lSp];
            final long lSegOff = ctx.stackSegOffsets[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArraysF[ctx.sp] = ctx.scratchF;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.stackIsSegment[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES_F.length();
            final int limit = SPECIES_F.loopBound(n);

            if (lIsSeg && rIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .mul(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) * rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (lIsSeg && rIsConst) {
                final FloatVector rbVec = FloatVector.broadcast(SPECIES_F, rVal);
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .mul(rbVec)
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) * rVal;
                }
            } else if (lIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .mul(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) * rArr[rOff + k];
                }
            } else if (rIsSeg && lIsConst) {
                final FloatVector laVec = FloatVector.broadcast(SPECIES_F, lVal);
                for (; k < limit; k += vl) {
                    laVec.mul(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lVal * rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (rIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .mul(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] * rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (!lIsConst && !rIsConst) {
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .mul(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] * rArr[rOff + k];
                }
            } else if (!lIsConst && rIsConst) {
                final FloatVector rbVec = FloatVector.broadcast(SPECIES_F, rVal);
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .mul(rbVec)
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] * rVal;
                }
            } else if (lIsConst && !rIsConst) {
                final FloatVector laVec = FloatVector.broadcast(SPECIES_F, lVal);
                for (; k < limit; k += vl) {
                    laVec.mul(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lVal * rArr[rOff + k];
                }
            } else {
                ctx.sp--;
                ctx.stackIsConst[ctx.sp] = true;
                ctx.stackConstValsF[ctx.sp] = lVal * rVal;
                ctx.sp++;
            }
        }

        private void doDiv(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final boolean rIsSeg = ctx.stackIsSegment[rSp];
            final double[] rArr = ctx.stackArrays[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final double rVal = ctx.stackConstVals[rSp];
            final MemorySegment rSeg = ctx.stackSegments[rSp];
            final long rSegOff = ctx.stackSegOffsets[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final boolean lIsSeg = ctx.stackIsSegment[lSp];
            final double[] lArr = ctx.stackArrays[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final double lVal = ctx.stackConstVals[lSp];
            final MemorySegment lSeg = ctx.stackSegments[lSp];
            final long lSegOff = ctx.stackSegOffsets[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArrays[ctx.sp] = ctx.scratch;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.stackIsSegment[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);

            if (lIsSeg && rIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .div(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) / rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (lIsSeg && rIsConst) {
                final DoubleVector rbVec = DoubleVector.broadcast(SPECIES, rVal);
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .div(rbVec)
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) / rVal;
                }
            } else if (lIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromMemorySegment(SPECIES, lSeg, (lSegOff + k) * 8L, ByteOrder.nativeOrder())
                            .div(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, lSegOff + k) / rArr[rOff + k];
                }
            } else if (rIsSeg && lIsConst) {
                final DoubleVector laVec = DoubleVector.broadcast(SPECIES, lVal);
                for (; k < limit; k += vl) {
                    laVec.div(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lVal / rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (rIsSeg) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .div(DoubleVector.fromMemorySegment(SPECIES, rSeg, (rSegOff + k) * 8L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] / rSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, rSegOff + k);
                }
            } else if (!lIsConst && !rIsConst) {
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .div(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] / rArr[rOff + k];
                }
            } else if (!lIsConst && rIsConst) {
                final DoubleVector rbVec = DoubleVector.broadcast(SPECIES, rVal);
                for (; k < limit; k += vl) {
                    DoubleVector.fromArray(SPECIES, lArr, lOff + k)
                            .div(rbVec)
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lArr[lOff + k] / rVal;
                }
            } else if (lIsConst && !rIsConst) {
                final DoubleVector laVec = DoubleVector.broadcast(SPECIES, lVal);
                for (; k < limit; k += vl) {
                    laVec.div(DoubleVector.fromArray(SPECIES, rArr, rOff + k))
                            .intoArray(ctx.scratch, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratch[resOffset + k] = lVal / rArr[rOff + k];
                }
            } else {
                ctx.sp--;
                ctx.stackIsConst[ctx.sp] = true;
                ctx.stackConstVals[ctx.sp] = lVal / rVal;
                ctx.sp++;
            }
        }

        private void doDivF(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final boolean rIsSeg = ctx.stackIsSegment[rSp];
            final float[] rArr = ctx.stackArraysF[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final float rVal = ctx.stackConstValsF[rSp];
            final MemorySegment rSeg = ctx.stackSegments[rSp];
            final long rSegOff = ctx.stackSegOffsets[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final boolean lIsSeg = ctx.stackIsSegment[lSp];
            final float[] lArr = ctx.stackArraysF[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final float lVal = ctx.stackConstValsF[lSp];
            final MemorySegment lSeg = ctx.stackSegments[lSp];
            final long lSegOff = ctx.stackSegOffsets[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArraysF[ctx.sp] = ctx.scratchF;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.stackIsSegment[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES_F.length();
            final int limit = SPECIES_F.loopBound(n);

            if (lIsSeg && rIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .div(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) / rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (lIsSeg && rIsConst) {
                final FloatVector rbVec = FloatVector.broadcast(SPECIES_F, rVal);
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .div(rbVec)
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) / rVal;
                }
            } else if (lIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromMemorySegment(SPECIES_F, lSeg, (lSegOff + k) * 4L, ByteOrder.nativeOrder())
                            .div(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lSegOff + k) / rArr[rOff + k];
                }
            } else if (rIsSeg && lIsConst) {
                final FloatVector laVec = FloatVector.broadcast(SPECIES_F, lVal);
                for (; k < limit; k += vl) {
                    laVec.div(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lVal / rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (rIsSeg) {
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .div(FloatVector.fromMemorySegment(SPECIES_F, rSeg, (rSegOff + k) * 4L, ByteOrder.nativeOrder()))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] / rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rSegOff + k);
                }
            } else if (!lIsConst && !rIsConst) {
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .div(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] / rArr[rOff + k];
                }
            } else if (!lIsConst && rIsConst) {
                final FloatVector rbVec = FloatVector.broadcast(SPECIES_F, rVal);
                for (; k < limit; k += vl) {
                    FloatVector.fromArray(SPECIES_F, lArr, lOff + k)
                            .div(rbVec)
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lArr[lOff + k] / rVal;
                }
            } else if (lIsConst && !rIsConst) {
                final FloatVector laVec = FloatVector.broadcast(SPECIES_F, lVal);
                for (; k < limit; k += vl) {
                    laVec.div(FloatVector.fromArray(SPECIES_F, rArr, rOff + k))
                            .intoArray(ctx.scratchF, resOffset + k);
                }
                for (; k < n; k++) {
                    ctx.scratchF[resOffset + k] = lVal / rArr[rOff + k];
                }
            } else {
                ctx.sp--;
                ctx.stackIsConst[ctx.sp] = true;
                ctx.stackConstValsF[ctx.sp] = lVal / rVal;
                ctx.sp++;
            }
        }
    }

    public final class VectorMath {

        private VectorMath() {
        }

        private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
        public static int VECTOR_THRESHOLD = 256;

        private static final VectorSpecies<Float> SPECIES_F = FloatVector.SPECIES_PREFERRED;

        private static final float DEG_TO_RAD_F = (float) (Math.PI / 180.0);
        private static final float RAD_TO_DEG_F = (float) (180.0 / Math.PI);
        private static final float GRAD_TO_RAD_F = (float) (Math.PI / 200.0);
        private static final float RAD_TO_GRAD_F = (float) (200.0 / Math.PI);

        private static final FloatVector V_DEG_TO_RAD_F = FloatVector.broadcast(SPECIES_F, DEG_TO_RAD_F);
        private static final FloatVector V_RAD_TO_DEG_F = FloatVector.broadcast(SPECIES_F, RAD_TO_DEG_F);
        private static final FloatVector V_GRAD_TO_RAD_F = FloatVector.broadcast(SPECIES_F, GRAD_TO_RAD_F);
        private static final FloatVector V_RAD_TO_GRAD_F = FloatVector.broadcast(SPECIES_F, RAD_TO_GRAD_F);

        private static final FloatVector V_ONE_F = FloatVector.broadcast(SPECIES_F, 1.0f);
        private static final FloatVector V_NEG_ONE_F = FloatVector.broadcast(SPECIES_F, -1.0f);
        private static final FloatVector V_HALF_F = FloatVector.broadcast(SPECIES_F, 0.5f);
        private static final FloatVector V_HALF_PI_F = FloatVector.broadcast(SPECIES_F, (float) (Math.PI / 2.0));
        private static final FloatVector V_NEG_HALF_PI_F = FloatVector.broadcast(SPECIES_F, (float) (-Math.PI / 2.0));
        private static final FloatVector V_NAN_F = FloatVector.broadcast(SPECIES_F, Float.NaN);
        private static final FloatVector ZERO_F_VM = FloatVector.broadcast(SPECIES_F, 0.0f);

        // Angle conversions
        private static final double DEG_TO_RAD = Math.PI / 180.0;
        private static final double RAD_TO_DEG = 180.0 / Math.PI;
        private static final double GRAD_TO_RAD = Math.PI / 200.0;
        private static final double RAD_TO_GRAD = 200.0 / Math.PI;

        private static final DoubleVector V_DEG_TO_RAD = DoubleVector.broadcast(SPECIES, DEG_TO_RAD);
        private static final DoubleVector V_RAD_TO_DEG = DoubleVector.broadcast(SPECIES, RAD_TO_DEG);
        private static final DoubleVector V_GRAD_TO_RAD = DoubleVector.broadcast(SPECIES, GRAD_TO_RAD);
        private static final DoubleVector V_RAD_TO_GRAD = DoubleVector.broadcast(SPECIES, RAD_TO_GRAD);

        // Core constants
        private static final DoubleVector V_ONE = DoubleVector.broadcast(SPECIES, 1.0);
        private static final DoubleVector V_NEG_ONE = DoubleVector.broadcast(SPECIES, -1.0);
        private static final DoubleVector V_HALF = DoubleVector.broadcast(SPECIES, 0.5);
        private static final DoubleVector V_HALF_PI = DoubleVector.broadcast(SPECIES, Math.PI / 2.0);
        private static final DoubleVector V_NEG_HALF_PI = DoubleVector.broadcast(SPECIES, -Math.PI / 2.0);
        private static final DoubleVector V_NAN = DoubleVector.broadcast(SPECIES, Double.NaN);
        private static final DoubleVector ZERO = DoubleVector.broadcast(SPECIES, 0.0);

        private static final double THRESHOLD_LOW = 0.46875;
        private static final double THRESHOLD_HIGH = 4.0;

        // ========================================================================
        // NO-LAMBDA DIRECT OPERATIONS
        // ========================================================================
        // Radian
        public static void sin(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sin(s[base + i]);
            }
        }

        public static void cos(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cos(s[base + i]);
            }
        }

        public static void tan(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tan(s[base + i]);
            }
        }

        // Degree
        public static void sinDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sin(Math.toRadians(s[base + i]));
            }
        }

        public static void cosDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cos(Math.toRadians(s[base + i]));
            }
        }

        public static void tanDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tan(Math.toRadians(s[base + i]));
            }
        }

        // Grad
        public static void sinGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sin(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void cosGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cos(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void tanGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tan(s[base + i] * GRAD_TO_RAD);
            }
        }

        // ===================== Reciprocal Trigonometric =====================
        // Radian
        public static void sec(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.cos(s[base + i]);
            }
        }

        public static void csc(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.sin(s[base + i]);
            }
        }

        public static void cot(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.tan(s[base + i]);
            }
        }

        // Degree
        public static void secDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.cos(Math.toRadians(s[base + i]));
            }
        }

        public static void cscDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.sin(Math.toRadians(s[base + i]));
            }
        }

        public static void cotDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.tan(Math.toRadians(s[base + i]));
            }
        }

        // Grad
        public static void secGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.cos(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void cscGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.sin(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void cotGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.tan(s[base + i] * GRAD_TO_RAD);
            }
        }

        // ===================== Inverse Trigonometric =====================
        // Radian
        public static void asin(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(s[base + i]);
            }
        }

        public static void acos(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(s[base + i]);
            }
        }

        public static void atan(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(s[base + i]);
            }
        }

        // Degree
        public static void asinDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.asin(s[base + i]));
            }
        }

        public static void acosDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.acos(s[base + i]));
            }
        }

        public static void atanDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.atan(s[base + i]));
            }
        }

        // Grad
        public static void asinGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void acosGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void atanGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(s[base + i]) * RAD_TO_GRAD;
            }
        }

        // ===================== Inverse Reciprocal Trigonometric =====================
        // Radian
        public static void acsc(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(1.0 / s[base + i]);
            }
        }

        public static void asec(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(1.0 / s[base + i]);
            }
        }

        public static void acot(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(1.0 / s[base + i]);
            }
        }

        // Degree
        public static void acscDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.asin(1.0 / s[base + i]));
            }
        }

        public static void asecDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.acos(1.0 / s[base + i]));
            }
        }

        public static void acotDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.atan(1.0 / s[base + i]));
            }
        }

        // Grad
        public static void acscGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void asecGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void acotGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        // ===================== Hyperbolic =====================
        public static void sinh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SINH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sinh(s[base + i]);
            }
        }

        public static void cosh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.COSH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cosh(s[base + i]);
            }
        }

        public static void tanh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.TANH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tanh(s[base + i]);
            }
        }

        // ===================== Inverse Hyperbolic =====================
        public static void asinh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAsinhImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.log(s[base + i] + Math.sqrt(s[base + i] * s[base + i] + 1.0));
            }
        }

        public static void acosh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcoshImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = x < 1.0 ? Double.NaN : Math.log(x + Math.sqrt(x * x - 1.0));
            }
        }

        public static void atanh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAtanhImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = 0.5 * Math.log((1.0 + x) / (1.0 - x));
            }
        }

        public static void asech(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAsechImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = (x <= 0.0 || x > 1.0) ? Double.NaN : Math.log((1.0 / x) + Math.sqrt((1.0 / (x * x)) - 1.0));
            }
        }

        public static void acsch(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcschImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = x == 0.0 ? Double.NaN : Math.log((1.0 / x) + Math.sqrt((1.0 / (x * x)) + 1.0));
            }
        }

        public static void acoth(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcothImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = Math.abs(x) <= 1.0 ? Double.NaN : 0.5 * Math.log((1.0 + (1.0 / x)) / (1.0 - (1.0 / x)));
            }
        }

        public static void sqrt(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SQRT)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sqrt(s[base + i]);
            }
        }

        public static void cbrt(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.CBRT)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cbrt(s[base + i]);
            }
        }

        // ===================== Exponential and Logarithmic =====================
        public static void exp(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.EXP)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.exp(s[base + i]);
            }
        }

        public static void ln(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.LOG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.log(s[base + i]);
            }
        }

        public static void log10(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.LOG10)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.log10(s[base + i]);
            }
        }

        private static boolean isExponentUniform(double[] scratch, int offset, int n) {
            if (n <= 1) {
                return true;
            }

            final double first = scratch[offset];
            if (Double.isNaN(first)) {
                // All must be NaN
                final int vl = SPECIES.length();
                int i = 0;
                int bound = SPECIES.loopBound(n);
                for (; i < bound; i += vl) {
                    DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i);
                    if (v.compare(VectorOperators.EQ, v).anyTrue()) {
                        return false;
                    }
                }
                int remaining = n - i;
                if (remaining > 0) {
                    var mask = SPECIES.indexInRange(0, remaining);
                    DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i, mask);
                    if (v.compare(VectorOperators.EQ, v, mask).anyTrue()) {
                        return false;
                    }
                }
                return true;
            }

            final DoubleVector target = DoubleVector.broadcast(SPECIES, first);
            final int vl = SPECIES.length();
            int i = 0;
            int bound = SPECIES.loopBound(n);

            for (; i < bound; i += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i);
                if (v.compare(VectorOperators.NE, target).anyTrue()) {
                    return false;
                }
            }

            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i, mask);
                if (v.compare(VectorOperators.NE, target, mask).anyTrue()) {
                    return false;
                }
            }
            return true;
        }

        public static void evaluateVariableExponent(double[] base, int bOffset, double[] exp, int eOffset,
                double[] dest, int dOffset, int n) {
            if (n <= 0) {
                return;
            }

            int i = 0;
            final int limit = SPECIES.loopBound(n);

            // === 1. Core Vector Loop: exp(y * ln(x)) ===
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector vBase = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                DoubleVector vExp = DoubleVector.fromArray(SPECIES, exp, eOffset + i);

                // Execute algebraic transcendental transformation
                DoubleVector log = vBase.lanewise(VectorOperators.LOG);
                DoubleVector scaled = log.mul(vExp);
                scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
            }

            // === 2. Masked Tail Pass ===
            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector vBase = DoubleVector.fromArray(SPECIES, base, bOffset + i, mask);
                DoubleVector vExp = DoubleVector.fromArray(SPECIES, exp, eOffset + i, mask);

                // Apply masks to intermediate operators to maintain lane isolation
                DoubleVector log = vBase.lanewise(VectorOperators.LOG, mask);
                DoubleVector scaled = log.mul(vExp, mask);
                DoubleVector res = scaled.lanewise(VectorOperators.EXP, mask);

                res.intoArray(dest, dOffset + i, mask);
            }
        }

        public static void executePowerBlended(double[] scratch, int baseOffset, int expOffset, int n) {
            if (n <= 0) {
                return;
            }

            if (isExponentUniform(scratch, expOffset, n)) {
                double uniformExp = scratch[expOffset];

                if (uniformExp == 0.5) {
                    VectorTranscendentals.evaluateNative(scratch, baseOffset, scratch, baseOffset, n, VectorOperators.SQRT);
                    return;
                }
                if (uniformExp == 2.0) {
                    computeSquare(scratch, baseOffset, scratch, baseOffset, n);
                    return;
                }
                if (uniformExp == 3.0) {
                    computeCube(scratch, baseOffset, scratch, baseOffset, n);
                    return;
                }
                if (uniformExp == 4.0) {
                    computeFourthPower(scratch, baseOffset, scratch, baseOffset, n);
                    return;
                }

                // Isolated fallback for uniform constants
                evaluateUniformExponent(scratch, baseOffset, uniformExp, scratch, baseOffset, n);
            } else {
                // Isolated fallback for variable exponents
                evaluateVariableExponent(scratch, baseOffset, scratch, expOffset, scratch, baseOffset, n);
            }
        }

// ==========================================
// Isolated Fast-Path Micro-Methods (EA Safe)
// ==========================================
        private static void computeSquare(double[] src, int srcOff, double[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES.loopBound(n);
            final int vl = SPECIES.length();

            for (; k < limit; k += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k);
                v.mul(v).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k, mask);
                v.mul(v).intoArray(dest, destOff + k, mask);
            }
        }

        private static void computeCube(double[] src, int srcOff, double[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES.loopBound(n);
            final int vl = SPECIES.length();

            for (; k < limit; k += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k);
                v.mul(v).mul(v).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k, mask);
                v.mul(v).mul(v).intoArray(dest, destOff + k, mask);
            }
        }

        private static void computeFourthPower(double[] src, int srcOff, double[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES.loopBound(n);
            final int vl = SPECIES.length();

            for (; k < limit; k += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k);
                DoubleVector sq = v.mul(v);
                sq.mul(sq).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k, mask);
                DoubleVector sq = v.mul(v);
                sq.mul(sq).intoArray(dest, destOff + k, mask);
            }
        }

        public static void evaluateUniformExponent(double[] base, int bOffset, double exp,
                double[] dest, int dOffset, int n) {
            if (n <= 0) {
                return;
            }

            if (exp == 1.0) {
                if (base != dest || bOffset != dOffset) {
                    System.arraycopy(base, bOffset, dest, dOffset, n);
                }
                return;
            }
            if (exp == 2.0) {
                computeSquare(base, bOffset, dest, dOffset, n);
                return;
            }
            if (exp == 3.0) {
                computeCube(base, bOffset, dest, dOffset, n);
                return;
            }
            if (exp == 4.0) {
                computeFourthPower(base, bOffset, dest, dOffset, n);
                return;
            }

            if (exp == 0.5) {
                VectorTranscendentals.evaluateNative(base, bOffset, dest, dOffset, n, VectorOperators.SQRT);
                return;
            }

            // Delegate the highly complex log/exp routines to a separate compilation target
            evaluateComplexUniformExponent(base, bOffset, exp, dest, dOffset, n);
        }

        private static void evaluateComplexUniformExponent(double[] base, int bOffset, double exp,
                double[] dest, int dOffset, int n) {
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);
            int i = 0;

            if (exp == 0.0) {
                for (; i < limit; i += vl) {
                    V_ONE.intoArray(dest, dOffset + i);
                }
            } else if (exp == -1.0) {
                for (; i < limit; i += vl) {
                    DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                    V_ONE.div(v).intoArray(dest, dOffset + i);
                }
            } else {
                final DoubleVector vExp = DoubleVector.broadcast(SPECIES, exp);
                if (exp % 1.0 == 0.0) {
                    if (exp % 2.0 != 0.0) {
                        // Scenario 1: Odd Integer (FIXED: targetIdx bug resolved)
                        for (; i < limit; i += vl) {
                            DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                            var isNegativeMask = v.compare(VectorOperators.LT, 0.0);
                            DoubleVector log = v.abs().lanewise(VectorOperators.LOG);
                            DoubleVector scaled = log.mul(vExp);
                            DoubleVector resAbs = scaled.lanewise(VectorOperators.EXP);
                            resAbs.blend(resAbs.neg(), isNegativeMask).intoArray(dest, dOffset + i);
                        }
                    } else {
                        // Scenario 2: Even Integer
                        for (; i < limit; i += vl) {
                            DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                            DoubleVector log = v.abs().lanewise(VectorOperators.LOG);
                            DoubleVector scaled = log.mul(vExp);
                            scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                        }
                    }
                } else {
                    // Scenario 3: Non-Integer
                    for (; i < limit; i += vl) {
                        DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                        DoubleVector log = v.lanewise(VectorOperators.LOG);
                        DoubleVector scaled = log.mul(vExp);
                        scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                    }
                }
            }

            // Clean Scalar Tail Pass
            for (; i < n; i++) {
                final double b = base[bOffset + i];
                dest[dOffset + i] = (exp == 0.0) ? 1.0 : (exp == -1.0) ? 1.0 / b : Math.pow(b, exp);
            }
        }

        // ========================================================================
        // Specialized Mathematical Transcendentals
        // ========================================================================
        /**
         * High-performance vectorized exp() using magic-number rounding +
         * 6th-degree minimax polynomial via FMA + fast bit manipulation for
         * 2^k.
         */
        static DoubleVector fastVectorExp(DoubleVector x) {
            x = x.lanewise(VectorOperators.MAX, -745.13).lanewise(VectorOperators.MIN, 709.78);

            DoubleVector invLn2 = DoubleVector.broadcast(SPECIES, 1.4426950408889634074);
            DoubleVector ln2Hi = DoubleVector.broadcast(SPECIES, -0.6931471805599453);
            DoubleVector ln2Lo = DoubleVector.broadcast(SPECIES, -2.8235290563031574E-13);

            DoubleVector magic = DoubleVector.broadcast(SPECIES, 4503599627370496.0); // 2^52
            DoubleVector k = x.mul(invLn2).add(magic).sub(magic);
            DoubleVector r = x.add(k.mul(ln2Hi)).add(k.mul(ln2Lo));

            DoubleVector p = r.mul(0.001398199650).add(0.0088632903);
            p = r.lanewise(VectorOperators.FMA, p, DoubleVector.broadcast(SPECIES, 0.04166666666));
            p = r.lanewise(VectorOperators.FMA, p, DoubleVector.broadcast(SPECIES, 0.16666666666));
            p = r.lanewise(VectorOperators.FMA, p, DoubleVector.broadcast(SPECIES, 0.5));
            p = r.lanewise(VectorOperators.FMA, p, V_ONE);
            p = r.lanewise(VectorOperators.FMA, p, V_ONE);

            LongVector kLong = (LongVector) k.convert(VectorOperators.D2L, 0);
            LongVector exponent = kLong.add(1023).lanewise(VectorOperators.LSHL, 52);
            DoubleVector twoK = (DoubleVector) exponent.convert(VectorOperators.REINTERPRET_L2D, 0);

            return p.mul(twoK);
        }

        static DoubleVector vectorizedErf(DoubleVector x) {
            return VectorizedCodyMath.erf(x);
        }

        // ===================== Stirling's Factorial Approximation =====================
        public static void stirling(int base, int n, double[] s) {
            int vl = SPECIES.length();
            int bound = SPECIES.loopBound(n);
            DoubleVector pi2 = DoubleVector.broadcast(SPECIES, 2.0 * Math.PI);
            DoubleVector nanVec = DoubleVector.broadcast(SPECIES, Double.NaN);
            int i = 0;

            for (; i < bound; i += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                DoubleVector lnN = v.lanewise(VectorOperators.LOG);
                DoubleVector term1 = v.mul(lnN).sub(v);
                DoubleVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5);
                DoubleVector term3 = V_ONE.div(v.mul(12.0));
                DoubleVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

                var invalidMask = v.compare(VectorOperators.LE, 0.0);
                result.blend(nanVec, invalidMask).intoArray(s, base + i);
            }

            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i, mask);
                DoubleVector lnN = v.lanewise(VectorOperators.LOG);
                DoubleVector term1 = v.mul(lnN).sub(v);
                DoubleVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5);
                DoubleVector term3 = V_ONE.div(v.mul(12.0));
                DoubleVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

                var invalidMask = v.compare(VectorOperators.LE, 0.0);
                result.blend(nanVec, invalidMask).intoArray(s, base + i, mask);
            }
        }
// Inside VectorMath class
// Inside VectorMath class

        public static void swiglu2(int lOff, int rOff, int destOff, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, lOff + k);
                DoubleVector y = DoubleVector.fromArray(SPECIES, s, rOff + k);
                DoubleVector expNegX = fastVectorExp(x.neg());

                // Math: x * y / (exp(-x) + 1)
                x.mul(y).div(expNegX.add(ONE)).intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = Maths.swiglu(s[lOff + k], s[rOff + k]);
            }
        }

        public static void geglu2(int lOff, int rOff, int destOff, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
            final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, lOff + k);
                DoubleVector y = DoubleVector.fromArray(SPECIES, s, rOff + k);

                // Math: x * (y * 0.5 * (erf(y * 0.707) + 1))
                DoubleVector erfVal = vectorizedErf(y.mul(INV_SQRT_2));
                DoubleVector geluY = y.mul(HALF).mul(erfVal.add(ONE));

                x.mul(geluY).intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = Maths.geglu(s[lOff + k], s[rOff + k]);
            }
        }

        public static void swiglu(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + k);
                DoubleVector expNegX = fastVectorExp(x.neg());
                x.div(expNegX.add(ONE)).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Maths.swiglu(s[base + k]);
            }
        }

        public static void gelu(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
            final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + k);
                x.mul(HALF).mul(vectorizedErf(x.mul(INV_SQRT_2)).add(ONE)).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Maths.gelu(s[base + k]);
            }
        }

        public static void geluFast(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
            final DoubleVector TWO = DoubleVector.broadcast(SPECIES, 2.0);
            final DoubleVector SQRT_2_OVER_PI = DoubleVector.broadcast(SPECIES, 0.7978845608028654);
            final DoubleVector COEF = DoubleVector.broadcast(SPECIES, 0.044715);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + k);
                DoubleVector x3 = x.mul(x).mul(x);
                DoubleVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);
                DoubleVector exp2z = fastVectorExp(z.mul(TWO));
                DoubleVector tanhZ = exp2z.sub(ONE).div(exp2z.add(ONE));
                x.mul(HALF).mul(tanhZ.add(ONE)).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Maths.fastGelu(s[base + k]);
            }
        }

// Based on your switch case, unary GEGLU passes 'x' through SIMD but runs geglu() on the tail.
        public static void gegluUnary(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            // Your original code did `result = x`, so SIMD does nothing to the array here.
            // If that was intentional, we just advance k. Otherwise, add vector math here.
            k = limit;
            for (; k < n; k++) {
                s[base + k] = Maths.geglu(s[base + k]);
            }
        }

        public static void erf(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + k);
                vectorizedErf(x).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Maths.erf(s[base + k]);
            }
        }

        // Add to VectorMath
        public static void abs(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + k)
                        .lanewise(VectorOperators.ABS)
                        .intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Math.abs(s[base + k]);
            }
        }

        // ===================== Conditional Branching =====================
        public static void if3(int base, int tileN, double[] s, int block) {
            final int cond = base + block;
            final int trueVal = base + 2 * block;
            final int falseVal = base + 3 * block;
            final int res = base;

            int vl = SPECIES.length();
            int bound = SPECIES.loopBound(tileN);
            int i = 0;

            for (; i < bound; i += vl) {
                DoubleVector vc = DoubleVector.fromArray(SPECIES, s, cond + i);
                DoubleVector vt = DoubleVector.fromArray(SPECIES, s, trueVal + i);
                DoubleVector vf = DoubleVector.fromArray(SPECIES, s, falseVal + i);
                VectorMask<Double> mask = vc.compare(VectorOperators.NE, 0.0).and(vc.compare(VectorOperators.EQ, vc));
                vf.blend(vt, mask).intoArray(s, res + i);
            }

            int remaining = tileN - i;
            if (remaining > 0) {
                var maskTail = SPECIES.indexInRange(0, remaining);
                DoubleVector vc = DoubleVector.fromArray(SPECIES, s, cond + i, maskTail);
                DoubleVector vt = DoubleVector.fromArray(SPECIES, s, trueVal + i, maskTail);
                DoubleVector vf = DoubleVector.fromArray(SPECIES, s, falseVal + i, maskTail);
                VectorMask<Double> mask = vc.compare(VectorOperators.NE, 0.0).and(vc.compare(VectorOperators.EQ, vc));
                vf.blend(vt, mask).intoArray(s, res + i, maskTail);
            }
        }

        // ========================================================================
        // Vectorized Inverse Hyperbolic Implementations
        // ========================================================================
        private static DoubleVector vectorAsinhImpl(DoubleVector x) {
            return x.add(x.mul(x).add(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
        }

        private static DoubleVector vectorAcoshImpl(DoubleVector x) {
            VectorMask<Double> valid = x.compare(VectorOperators.GE, V_ONE);
            DoubleVector result = x.add(x.mul(x).sub(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAtanhImpl(DoubleVector x) {
            VectorMask<Double> valid = x.abs().compare(VectorOperators.LT, V_ONE);
            DoubleVector result = V_ONE.add(x).div(V_ONE.sub(x))
                    .lanewise(VectorOperators.LOG)
                    .mul(V_HALF);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAsechImpl(DoubleVector x) {
            VectorMask<Double> valid = x.compare(VectorOperators.GT, 0.0)
                    .and(x.compare(VectorOperators.LE, V_ONE));
            DoubleVector result = V_ONE.div(x).add(V_ONE.div(x.mul(x)).sub(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAcschImpl(DoubleVector x) {
            VectorMask<Double> valid = x.compare(VectorOperators.NE, 0.0);
            DoubleVector result = V_ONE.div(x).add(V_ONE.div(x.mul(x)).add(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAcothImpl(DoubleVector x) {
            VectorMask<Double> valid = x.abs().compare(VectorOperators.GT, V_ONE);
            DoubleVector result = V_ONE.add(V_ONE.div(x)).div(V_ONE.sub(V_ONE.div(x)))
                    .lanewise(VectorOperators.LOG)
                    .mul(V_HALF);
            return result.blend(V_NAN, valid.not());
        }

        // ========================================================================
        // FLOAT OVERLOADS (distinct erasure from the double[] methods above)
        // ========================================================================
        // ========================================================================
        // NO-LAMBDA DIRECT OPERATIONS
        // ========================================================================
        // Radian
        public static void sin(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.sin(s[base + i]);
            }
        }

        public static void cos(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.cos(s[base + i]);
            }
        }

        public static void tan(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.tan(s[base + i]);
            }
        }

        // Degree
        public static void sinDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_DEG_TO_RAD_F)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.sin(Math.toRadians(s[base + i]));
            }
        }

        public static void cosDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_DEG_TO_RAD_F)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.cos(Math.toRadians(s[base + i]));
            }
        }

        public static void tanDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_DEG_TO_RAD_F)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.tan(Math.toRadians(s[base + i]));
            }
        }

        // Grad
        public static void sinGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_GRAD_TO_RAD_F)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.sin(s[base + i] * GRAD_TO_RAD_F);
            }
        }

        public static void cosGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_GRAD_TO_RAD_F)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.cos(s[base + i] * GRAD_TO_RAD_F);
            }
        }

        public static void tanGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_GRAD_TO_RAD_F)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.tan(s[base + i] * GRAD_TO_RAD_F);
            }
        }

        // ===================== Reciprocal Trigonometric =====================
        // Radian
        public static void sec(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) (1.0f / Math.cos(s[base + i]));
            }
        }

        public static void csc(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) (1.0f / Math.sin(s[base + i]));
            }
        }

        public static void cot(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector v = FloatVector.fromArray(SPECIES_F, s, base + i);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) (1.0f / Math.tan(s[base + i]));
            }
        }

        // Degree
        public static void secDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_DEG_TO_RAD_F)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) (1.0f / Math.cos(Math.toRadians(s[base + i])));
            }
        }

        public static void cscDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_DEG_TO_RAD_F)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) (1.0f / Math.sin(Math.toRadians(s[base + i])));
            }
        }

        public static void cotDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector v = FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_DEG_TO_RAD_F);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) (1.0f / Math.tan(Math.toRadians(s[base + i])));
            }
        }

        // Grad
        public static void secGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_GRAD_TO_RAD_F)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) (1.0f / Math.cos(s[base + i] * GRAD_TO_RAD_F));
            }
        }

        public static void cscGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_GRAD_TO_RAD_F)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) (1.0f / Math.sin(s[base + i] * GRAD_TO_RAD_F));
            }
        }

        public static void cotGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector v = FloatVector.fromArray(SPECIES_F, s, base + i)
                        .mul(V_GRAD_TO_RAD_F);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) (1.0f / Math.tan(s[base + i] * GRAD_TO_RAD_F));
            }
        }

        // ===================== Inverse Trigonometric =====================
        // Radian
        public static void asin(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.asin(s[base + i]);
            }
        }

        public static void acos(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.acos(s[base + i]);
            }
        }

        public static void atan(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.atan(s[base + i]);
            }
        }

        // Degree
        public static void asinDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.toDegrees(Math.asin(s[base + i]));
            }
        }

        public static void acosDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.toDegrees(Math.acos(s[base + i]));
            }
        }

        public static void atanDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.toDegrees(Math.atan(s[base + i]));
            }
        }

        // Grad
        public static void asinGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.asin(s[base + i]) * RAD_TO_GRAD_F;
            }
        }

        public static void acosGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.acos(s[base + i]) * RAD_TO_GRAD_F;
            }
        }

        public static void atanGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.atan(s[base + i]) * RAD_TO_GRAD_F;
            }
        }

        // ===================== Inverse Reciprocal Trigonometric =====================
        // Radian
        public static void acsc(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.asin(1.0f / s[base + i]);
            }
        }

        public static void asec(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.acos(1.0f / s[base + i]);
            }
        }

        public static void acot(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.atan(1.0f / s[base + i]);
            }
        }

        // Degree
        public static void acscDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.toDegrees(Math.asin(1.0f / s[base + i]));
            }
        }

        public static void asecDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.toDegrees(Math.acos(1.0f / s[base + i]));
            }
        }

        public static void acotDeg(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.toDegrees(Math.atan(1.0f / s[base + i]));
            }
        }

        // Grad
        public static void acscGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.asin(1.0f / s[base + i]) * RAD_TO_GRAD_F;
            }
        }

        public static void asecGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.acos(1.0f / s[base + i]) * RAD_TO_GRAD_F;
            }
        }

        public static void acotGrad(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                V_ONE_F.div(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD_F)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.atan(1.0f / s[base + i]) * RAD_TO_GRAD_F;
            }
        }

        // ===================== Hyperbolic =====================
        public static void sinh(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.SINH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.sinh(s[base + i]);
            }
        }

        public static void cosh(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.COSH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.cosh(s[base + i]);
            }
        }

        public static void tanh(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.TANH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.tanh(s[base + i]);
            }
        }

        // ===================== Inverse Hyperbolic =====================
        public static void asinh(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                vectorAsinhImpl(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.log(s[base + i] + Math.sqrt(s[base + i] * s[base + i] + 1.0f));
            }
        }

        public static void acosh(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                vectorAcoshImpl(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                float x = s[base + i];
                s[base + i] = x < 1.0f ? Float.NaN : (float) Math.log(x + Math.sqrt(x * x - 1.0f));
            }
        }

        public static void atanh(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                vectorAtanhImpl(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                float x = s[base + i];
                s[base + i] = (float) (0.5f * Math.log((1.0f + x) / (1.0f - x)));
            }
        }

        public static void asech(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                vectorAsechImpl(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                float x = s[base + i];
                s[base + i] = (x <= 0.0f || x > 1.0f) ? Float.NaN : (float) Math.log((1.0f / x) + Math.sqrt((1.0f / (x * x)) - 1.0f));
            }
        }

        public static void acsch(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                vectorAcschImpl(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                float x = s[base + i];
                s[base + i] = x == 0.0f ? Float.NaN : (float) Math.log((1.0f / x) + Math.sqrt((1.0f / (x * x)) + 1.0f));
            }
        }

        public static void acoth(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                vectorAcothImpl(FloatVector.fromArray(SPECIES_F, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                float x = s[base + i];
                s[base + i] = Math.abs(x) <= 1.0f ? Float.NaN : (float) (0.5f * Math.log((1.0f + (1.0f / x)) / (1.0f - (1.0f / x))));
            }
        }

        public static void sqrt(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.SQRT)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.sqrt(s[base + i]);
            }
        }

        public static void cbrt(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.CBRT)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.cbrt(s[base + i]);
            }
        }

        // ===================== Exponential and Logarithmic =====================
        public static void exp(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.EXP)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.exp(s[base + i]);
            }
        }

        public static void ln(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.LOG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.log(s[base + i]);
            }
        }

        public static void log10(int base, int n, float[] s) {
            int i = 0;
            int limit = SPECIES_F.loopBound(n);
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + i)
                        .lanewise(VectorOperators.LOG10)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (float) Math.log10(s[base + i]);
            }
        }

        private static boolean isExponentUniform(float[] scratch, int offset, int n) {
            if (n <= 1) {
                return true;
            }

            final float first = scratch[offset];
            if (Float.isNaN(first)) {
                // All must be NaN
                final int vl = SPECIES_F.length();
                int i = 0;
                int bound = SPECIES_F.loopBound(n);
                for (; i < bound; i += vl) {
                    FloatVector v = FloatVector.fromArray(SPECIES_F, scratch, offset + i);
                    if (v.compare(VectorOperators.EQ, v).anyTrue()) {
                        return false;
                    }
                }
                int remaining = n - i;
                if (remaining > 0) {
                    var mask = SPECIES_F.indexInRange(0, remaining);
                    FloatVector v = FloatVector.fromArray(SPECIES_F, scratch, offset + i, mask);
                    if (v.compare(VectorOperators.EQ, v, mask).anyTrue()) {
                        return false;
                    }
                }
                return true;
            }

            final FloatVector target = FloatVector.broadcast(SPECIES_F, first);
            final int vl = SPECIES_F.length();
            int i = 0;
            int bound = SPECIES_F.loopBound(n);

            for (; i < bound; i += vl) {
                FloatVector v = FloatVector.fromArray(SPECIES_F, scratch, offset + i);
                if (v.compare(VectorOperators.NE, target).anyTrue()) {
                    return false;
                }
            }

            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES_F.indexInRange(0, remaining);
                FloatVector v = FloatVector.fromArray(SPECIES_F, scratch, offset + i, mask);
                if (v.compare(VectorOperators.NE, target, mask).anyTrue()) {
                    return false;
                }
            }
            return true;
        }

        public static void evaluateVariableExponent(float[] base, int bOffset, float[] exp, int eOffset,
                float[] dest, int dOffset, int n) {
            if (n <= 0) {
                return;
            }

            int i = 0;
            final int limit = SPECIES_F.loopBound(n);

            // === 1. Core Vector Loop: exp(y * ln(x)) ===
            for (; i < limit; i += SPECIES_F.length()) {
                FloatVector vBase = FloatVector.fromArray(SPECIES_F, base, bOffset + i);
                FloatVector vExp = FloatVector.fromArray(SPECIES_F, exp, eOffset + i);

                // Execute algebraic transcendental transformation
                FloatVector log = vBase.lanewise(VectorOperators.LOG);
                FloatVector scaled = log.mul(vExp);
                scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
            }

            // === 2. Masked Tail Pass ===
            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES_F.indexInRange(0, remaining);
                FloatVector vBase = FloatVector.fromArray(SPECIES_F, base, bOffset + i, mask);
                FloatVector vExp = FloatVector.fromArray(SPECIES_F, exp, eOffset + i, mask);

                // Apply masks to intermediate operators to maintain lane isolation
                FloatVector log = vBase.lanewise(VectorOperators.LOG, mask);
                FloatVector scaled = log.mul(vExp, mask);
                FloatVector res = scaled.lanewise(VectorOperators.EXP, mask);

                res.intoArray(dest, dOffset + i, mask);
            }
        }

        public static void executePowerBlended(float[] scratch, int baseOffset, int expOffset, int n) {
            if (n <= 0) {
                return;
            }

            if (isExponentUniform(scratch, expOffset, n)) {
                float uniformExp = scratch[expOffset];

                if (uniformExp == 0.5f) {
                    VectorTranscendentals.evaluateNative(scratch, baseOffset, scratch, baseOffset, n, VectorOperators.SQRT);
                    return;
                }
                if (uniformExp == 2.0f) {
                    computeSquare(scratch, baseOffset, scratch, baseOffset, n);
                    return;
                }
                if (uniformExp == 3.0f) {
                    computeCube(scratch, baseOffset, scratch, baseOffset, n);
                    return;
                }
                if (uniformExp == 4.0f) {
                    computeFourthPower(scratch, baseOffset, scratch, baseOffset, n);
                    return;
                }

                // Isolated fallback for uniform constants
                evaluateUniformExponent(scratch, baseOffset, uniformExp, scratch, baseOffset, n);
            } else {
                // Isolated fallback for variable exponents
                evaluateVariableExponent(scratch, baseOffset, scratch, expOffset, scratch, baseOffset, n);
            }
        }

// ==========================================
// Isolated Fast-Path Micro-Methods (EA Safe)
// ==========================================
        private static void computeSquare(float[] src, int srcOff, float[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES_F.loopBound(n);
            final int vl = SPECIES_F.length();

            for (; k < limit; k += vl) {
                FloatVector v = FloatVector.fromArray(SPECIES_F, src, srcOff + k);
                v.mul(v).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES_F.indexInRange(0, remaining);
                FloatVector v = FloatVector.fromArray(SPECIES_F, src, srcOff + k, mask);
                v.mul(v).intoArray(dest, destOff + k, mask);
            }
        }

        private static void computeCube(float[] src, int srcOff, float[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES_F.loopBound(n);
            final int vl = SPECIES_F.length();

            for (; k < limit; k += vl) {
                FloatVector v = FloatVector.fromArray(SPECIES_F, src, srcOff + k);
                v.mul(v).mul(v).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES_F.indexInRange(0, remaining);
                FloatVector v = FloatVector.fromArray(SPECIES_F, src, srcOff + k, mask);
                v.mul(v).mul(v).intoArray(dest, destOff + k, mask);
            }
        }

        private static void computeFourthPower(float[] src, int srcOff, float[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES_F.loopBound(n);
            final int vl = SPECIES_F.length();

            for (; k < limit; k += vl) {
                FloatVector v = FloatVector.fromArray(SPECIES_F, src, srcOff + k);
                FloatVector sq = v.mul(v);
                sq.mul(sq).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES_F.indexInRange(0, remaining);
                FloatVector v = FloatVector.fromArray(SPECIES_F, src, srcOff + k, mask);
                FloatVector sq = v.mul(v);
                sq.mul(sq).intoArray(dest, destOff + k, mask);
            }
        }

        public static void evaluateUniformExponent(float[] base, int bOffset, float exp,
                float[] dest, int dOffset, int n) {
            if (n <= 0) {
                return;
            }

            if (exp == 1.0f) {
                if (base != dest || bOffset != dOffset) {
                    System.arraycopy(base, bOffset, dest, dOffset, n);
                }
                return;
            }
            if (exp == 2.0f) {
                computeSquare(base, bOffset, dest, dOffset, n);
                return;
            }
            if (exp == 3.0f) {
                computeCube(base, bOffset, dest, dOffset, n);
                return;
            }
            if (exp == 4.0f) {
                computeFourthPower(base, bOffset, dest, dOffset, n);
                return;
            }

            if (exp == 0.5f) {
                VectorTranscendentals.evaluateNative(base, bOffset, dest, dOffset, n, VectorOperators.SQRT);
                return;
            }

            // Delegate the highly complex log/exp routines to a separate compilation target
            evaluateComplexUniformExponent(base, bOffset, exp, dest, dOffset, n);
        }

        private static void evaluateComplexUniformExponent(float[] base, int bOffset, float exp,
                float[] dest, int dOffset, int n) {
            final int vl = SPECIES_F.length();
            final int limit = SPECIES_F.loopBound(n);
            int i = 0;

            if (exp == 0.0f) {
                for (; i < limit; i += vl) {
                    V_ONE_F.intoArray(dest, dOffset + i);
                }
            } else if (exp == -1.0f) {
                for (; i < limit; i += vl) {
                    FloatVector v = FloatVector.fromArray(SPECIES_F, base, bOffset + i);
                    V_ONE_F.div(v).intoArray(dest, dOffset + i);
                }
            } else {
                final FloatVector vExp = FloatVector.broadcast(SPECIES_F, exp);
                if (exp % 1.0f == 0.0f) {
                    if (exp % 2.0f != 0.0f) {
                        // Scenario 1: Odd Integer (FIXED: targetIdx bug resolved)
                        for (; i < limit; i += vl) {
                            FloatVector v = FloatVector.fromArray(SPECIES_F, base, bOffset + i);
                            var isNegativeMask = v.compare(VectorOperators.LT, 0.0f);
                            FloatVector log = v.abs().lanewise(VectorOperators.LOG);
                            FloatVector scaled = log.mul(vExp);
                            FloatVector resAbs = scaled.lanewise(VectorOperators.EXP);
                            resAbs.blend(resAbs.neg(), isNegativeMask).intoArray(dest, dOffset + i);
                        }
                    } else {
                        // Scenario 2: Even Integer
                        for (; i < limit; i += vl) {
                            FloatVector v = FloatVector.fromArray(SPECIES_F, base, bOffset + i);
                            FloatVector log = v.abs().lanewise(VectorOperators.LOG);
                            FloatVector scaled = log.mul(vExp);
                            scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                        }
                    }
                } else {
                    // Scenario 3: Non-Integer
                    for (; i < limit; i += vl) {
                        FloatVector v = FloatVector.fromArray(SPECIES_F, base, bOffset + i);
                        FloatVector log = v.lanewise(VectorOperators.LOG);
                        FloatVector scaled = log.mul(vExp);
                        scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                    }
                }
            }

            // Clean Scalar Tail Pass
            for (; i < n; i++) {
                final float b = base[bOffset + i];
                dest[dOffset + i] = (exp == 0.0f) ? 1.0f : (exp == -1.0f) ? 1.0f / b : (float) Math.pow(b, exp);
            }
        }

        // ========================================================================
        // Specialized Mathematical Transcendentals
        // ========================================================================
        /**
         * Float-native fast exp() using the same magic-number range-reduction
         * strategy as the {@code double} version, re-derived for IEEE-754
         * binary32: 8-bit exponent, bias 127, 23-bit mantissa (vs. bias 1023 /
         * 52-bit mantissa for {@code double}). A single-term Cody-Waite
         * reduction (one {@code ln2} constant instead of a hi/lo split) is
         * sufficient here because float mantissa precision (~7 decimal digits)
         * does not benefit meaningfully from the double-double reduction the
         * {@code double} path uses.
         *
         * Input is clamped to the float overflow/underflow bounds of
         * {@code exp} (roughly [-87.33, 88.72]) before reduction, matching the
         * intent of the {@code double} version's [-745.13, 709.78] clamp.
         */
        static FloatVector fastVectorExp(FloatVector x) {
            x = x.lanewise(VectorOperators.MAX, -87.33654f).lanewise(VectorOperators.MIN, 88.72283f);

            final FloatVector invLn2 = FloatVector.broadcast(SPECIES_F, 1.4426950408889634f);
            final FloatVector ln2 = FloatVector.broadcast(SPECIES_F, 0.6931471805599453f);

            // magic = 2^23: adding/subtracting it snaps x*invLn2 to the
            // nearest integer via IEEE-754 round-to-nearest, exactly mirroring
            // the 2^52 trick used by the double path, scaled to float's
            // 23-bit mantissa.
            final FloatVector magic = FloatVector.broadcast(SPECIES_F, 8388608.0f); // 2^23
            FloatVector k = x.mul(invLn2).add(magic).sub(magic);
            FloatVector r = x.sub(k.mul(ln2));

            // 5th-degree minimax polynomial for exp(r), r in [-ln2/2, ln2/2],
            // adequate for float precision.
            FloatVector p = r.mul(0.0013298820f).add(0.0083334036f);
            p = r.lanewise(VectorOperators.FMA, p, FloatVector.broadcast(SPECIES_F, 0.0416673128f));
            p = r.lanewise(VectorOperators.FMA, p, FloatVector.broadcast(SPECIES_F, 0.1666666667f));
            p = r.lanewise(VectorOperators.FMA, p, FloatVector.broadcast(SPECIES_F, 0.5f));
            p = r.lanewise(VectorOperators.FMA, p, V_ONE_F);
            p = r.lanewise(VectorOperators.FMA, p, V_ONE_F);

            IntVector kInt = (IntVector) k.convert(VectorOperators.F2I, 0);
            IntVector exponent = kInt.add(127).lanewise(VectorOperators.LSHL, 23);
            FloatVector twoK = (FloatVector) exponent.convert(VectorOperators.REINTERPRET_I2F, 0);

            return p.mul(twoK);
        }

        static FloatVector vectorizedErf(FloatVector x) {
            return VectorizedCodyMath.erf(x);
        }

        // ===================== Stirling's Factorial Approximation =====================
        public static void stirling(int base, int n, float[] s) {
            int vl = SPECIES_F.length();
            int bound = SPECIES_F.loopBound(n);
            FloatVector pi2 = FloatVector.broadcast(SPECIES_F, (float) (2.0f * Math.PI));
            FloatVector nanVec = FloatVector.broadcast(SPECIES_F, Float.NaN);
            int i = 0;

            for (; i < bound; i += vl) {
                FloatVector v = FloatVector.fromArray(SPECIES_F, s, base + i);
                FloatVector lnN = v.lanewise(VectorOperators.LOG);
                FloatVector term1 = v.mul(lnN).sub(v);
                FloatVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5f);
                FloatVector term3 = V_ONE_F.div(v.mul(12.0f));
                FloatVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

                var invalidMask = v.compare(VectorOperators.LE, 0.0f);
                result.blend(nanVec, invalidMask).intoArray(s, base + i);
            }

            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES_F.indexInRange(0, remaining);
                FloatVector v = FloatVector.fromArray(SPECIES_F, s, base + i, mask);
                FloatVector lnN = v.lanewise(VectorOperators.LOG);
                FloatVector term1 = v.mul(lnN).sub(v);
                FloatVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5f);
                FloatVector term3 = V_ONE_F.div(v.mul(12.0f));
                FloatVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

                var invalidMask = v.compare(VectorOperators.LE, 0.0f);
                result.blend(nanVec, invalidMask).intoArray(s, base + i, mask);
            }
        }
// Inside VectorMath class
// Inside VectorMath class

        public static void swiglu2(int lOff, int rOff, int destOff, int n, float[] s) {
            int limit = SPECIES_F.loopBound(n);
            int k = 0;
            final FloatVector ONE = FloatVector.broadcast(SPECIES_F, 1.0f);

            for (; k < limit; k += SPECIES_F.length()) {
                FloatVector x = FloatVector.fromArray(SPECIES_F, s, lOff + k);
                FloatVector y = FloatVector.fromArray(SPECIES_F, s, rOff + k);
                FloatVector expNegX = fastVectorExp(x.neg());

                // Math: x * y / (exp(-x) + 1)
                x.mul(y).div(expNegX.add(ONE)).intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = (float) Maths.swiglu(s[lOff + k], s[rOff + k]);
            }
        }

        public static void geglu2(int lOff, int rOff, int destOff, int n, float[] s) {
            int limit = SPECIES_F.loopBound(n);
            int k = 0;
            final FloatVector HALF = FloatVector.broadcast(SPECIES_F, 0.5f);
            final FloatVector ONE = FloatVector.broadcast(SPECIES_F, 1.0f);
            final FloatVector INV_SQRT_2 = FloatVector.broadcast(SPECIES_F, 0.7071067811865476f);

            for (; k < limit; k += SPECIES_F.length()) {
                FloatVector x = FloatVector.fromArray(SPECIES_F, s, lOff + k);
                FloatVector y = FloatVector.fromArray(SPECIES_F, s, rOff + k);

                // Math: x * (y * 0.5f * (erf(y * 0.707f) + 1))
                FloatVector erfVal = vectorizedErf(y.mul(INV_SQRT_2));
                FloatVector geluY = y.mul(HALF).mul(erfVal.add(ONE));

                x.mul(geluY).intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = (float) Maths.geglu(s[lOff + k], s[rOff + k]);
            }
        }

        public static void swiglu(int base, int n, float[] s) {
            int limit = SPECIES_F.loopBound(n);
            int k = 0;
            final FloatVector ONE = FloatVector.broadcast(SPECIES_F, 1.0f);

            for (; k < limit; k += SPECIES_F.length()) {
                FloatVector x = FloatVector.fromArray(SPECIES_F, s, base + k);
                FloatVector expNegX = fastVectorExp(x.neg());
                x.div(expNegX.add(ONE)).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = (float) Maths.swiglu(s[base + k]);
            }
        }

        public static void gelu(int base, int n, float[] s) {
            int limit = SPECIES_F.loopBound(n);
            int k = 0;
            final FloatVector HALF = FloatVector.broadcast(SPECIES_F, 0.5f);
            final FloatVector ONE = FloatVector.broadcast(SPECIES_F, 1.0f);
            final FloatVector INV_SQRT_2 = FloatVector.broadcast(SPECIES_F, 0.7071067811865476f);

            for (; k < limit; k += SPECIES_F.length()) {
                FloatVector x = FloatVector.fromArray(SPECIES_F, s, base + k);
                x.mul(HALF).mul(vectorizedErf(x.mul(INV_SQRT_2)).add(ONE)).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = (float) Maths.gelu(s[base + k]);
            }
        }

        public static void geluFast(int base, int n, float[] s) {
            int limit = SPECIES_F.loopBound(n);
            int k = 0;
            final FloatVector HALF = FloatVector.broadcast(SPECIES_F, 0.5f);
            final FloatVector ONE = FloatVector.broadcast(SPECIES_F, 1.0f);
            final FloatVector TWO = FloatVector.broadcast(SPECIES_F, 2.0f);
            final FloatVector SQRT_2_OVER_PI = FloatVector.broadcast(SPECIES_F, 0.7978845608028654f);
            final FloatVector COEF = FloatVector.broadcast(SPECIES_F, 0.044715f);

            for (; k < limit; k += SPECIES_F.length()) {
                FloatVector x = FloatVector.fromArray(SPECIES_F, s, base + k);
                FloatVector x3 = x.mul(x).mul(x);
                FloatVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);
                FloatVector exp2z = fastVectorExp(z.mul(TWO));
                FloatVector tanhZ = exp2z.sub(ONE).div(exp2z.add(ONE));
                x.mul(HALF).mul(tanhZ.add(ONE)).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = (float) Maths.fastGelu(s[base + k]);
            }
        }

// Based on your switch case, unary GEGLU passes 'x' through SIMD but runs geglu() on the tail.
        public static void gegluUnary(int base, int n, float[] s) {
            int limit = SPECIES_F.loopBound(n);
            int k = 0;
            // Your original code did `result = x`, so SIMD does nothing to the array here.
            // If that was intentional, we just advance k. Otherwise, add vector math here.
            k = limit;
            for (; k < n; k++) {
                s[base + k] = (float) Maths.geglu(s[base + k]);
            }
        }

        public static void erf(int base, int n, float[] s) {
            int limit = SPECIES_F.loopBound(n);
            int k = 0;
            for (; k < limit; k += SPECIES_F.length()) {
                FloatVector x = FloatVector.fromArray(SPECIES_F, s, base + k);
                vectorizedErf(x).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = (float) Maths.erf(s[base + k]);
            }
        }

        // Add to VectorMath
        public static void abs(int base, int n, float[] s) {
            int limit = SPECIES_F.loopBound(n);
            int k = 0;
            for (; k < limit; k += SPECIES_F.length()) {
                FloatVector.fromArray(SPECIES_F, s, base + k)
                        .lanewise(VectorOperators.ABS)
                        .intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = (float) Math.abs(s[base + k]);
            }
        }

        // ===================== Conditional Branching =====================
        public static void if3(int base, int tileN, float[] s, int block) {
            final int cond = base + block;
            final int trueVal = base + 2 * block;
            final int falseVal = base + 3 * block;
            final int res = base;

            int vl = SPECIES_F.length();
            int bound = SPECIES_F.loopBound(tileN);
            int i = 0;

            for (; i < bound; i += vl) {
                FloatVector vc = FloatVector.fromArray(SPECIES_F, s, cond + i);
                FloatVector vt = FloatVector.fromArray(SPECIES_F, s, trueVal + i);
                FloatVector vf = FloatVector.fromArray(SPECIES_F, s, falseVal + i);
                VectorMask<Float> mask = vc.compare(VectorOperators.NE, 0.0f).and(vc.compare(VectorOperators.EQ, vc));
                vf.blend(vt, mask).intoArray(s, res + i);
            }

            int remaining = tileN - i;
            if (remaining > 0) {
                var maskTail = SPECIES_F.indexInRange(0, remaining);
                FloatVector vc = FloatVector.fromArray(SPECIES_F, s, cond + i, maskTail);
                FloatVector vt = FloatVector.fromArray(SPECIES_F, s, trueVal + i, maskTail);
                FloatVector vf = FloatVector.fromArray(SPECIES_F, s, falseVal + i, maskTail);
                VectorMask<Float> mask = vc.compare(VectorOperators.NE, 0.0f).and(vc.compare(VectorOperators.EQ, vc));
                vf.blend(vt, mask).intoArray(s, res + i, maskTail);
            }
        }

        // ========================================================================
        // Vectorized Inverse Hyperbolic Implementations
        // ========================================================================
        private static FloatVector vectorAsinhImpl(FloatVector x) {
            return x.add(x.mul(x).add(V_ONE_F).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
        }

        private static FloatVector vectorAcoshImpl(FloatVector x) {
            VectorMask<Float> valid = x.compare(VectorOperators.GE, V_ONE_F);
            FloatVector result = x.add(x.mul(x).sub(V_ONE_F).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN_F, valid.not());
        }

        private static FloatVector vectorAtanhImpl(FloatVector x) {
            VectorMask<Float> valid = x.abs().compare(VectorOperators.LT, V_ONE_F);
            FloatVector result = V_ONE_F.add(x).div(V_ONE_F.sub(x))
                    .lanewise(VectorOperators.LOG)
                    .mul(V_HALF_F);
            return result.blend(V_NAN_F, valid.not());
        }

        private static FloatVector vectorAsechImpl(FloatVector x) {
            VectorMask<Float> valid = x.compare(VectorOperators.GT, 0.0f)
                    .and(x.compare(VectorOperators.LE, V_ONE_F));
            FloatVector result = V_ONE_F.div(x).add(V_ONE_F.div(x.mul(x)).sub(V_ONE_F).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN_F, valid.not());
        }

        private static FloatVector vectorAcschImpl(FloatVector x) {
            VectorMask<Float> valid = x.compare(VectorOperators.NE, 0.0f);
            FloatVector result = V_ONE_F.div(x).add(V_ONE_F.div(x.mul(x)).add(V_ONE_F).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN_F, valid.not());
        }

        private static FloatVector vectorAcothImpl(FloatVector x) {
            VectorMask<Float> valid = x.abs().compare(VectorOperators.GT, V_ONE_F);
            FloatVector result = V_ONE_F.add(V_ONE_F.div(x)).div(V_ONE_F.sub(V_ONE_F.div(x)))
                    .lanewise(VectorOperators.LOG)
                    .mul(V_HALF_F);
            return result.blend(V_NAN_F, valid.not());
        }

    }

    public static final class VectorTranscendentals {

        private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
        private static final VectorSpecies<Float> SPECIES_F = FloatVector.SPECIES_PREFERRED;

        public static void evaluateNative(double[] src, int srcOffset, double[] dest, int destOffset, int n, VectorOperators.Unary op) {
            int vl = SPECIES.length();
            int limit = SPECIES.loopBound(n);
            int i = 0;

            // Vector Loop
            for (; i < limit; i += vl) {
                DoubleVector va  = DoubleVector.fromArray(SPECIES, src, srcOffset + i);
                va.lanewise(op).intoArray(dest, destOffset + i);
            }

            // Clean Masked Tail
            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector va  = DoubleVector.fromArray(SPECIES, src, srcOffset + i, mask);
                va.lanewise(op).intoArray(dest, destOffset + i, mask);
            }
        }

        /**
         * Float mirror of
         * {@link #evaluateNative(double[], int, double[], int, int, VectorOperators.Unary)}.
         * True overload (distinct erasure from the double[] version above).
         */
        public static void evaluateNative(float[] src, int srcOffset, float[] dest, int destOffset, int n, VectorOperators.Unary op) {
            int vl = SPECIES_F.length();
            int limit = SPECIES_F.loopBound(n);
            int i = 0;

            // Vector Loop
            for (; i < limit; i += vl) {
                FloatVector va  = FloatVector.fromArray(SPECIES_F, src, srcOffset + i);
                va.lanewise(op).intoArray(dest, destOffset + i);
            }

            // Clean Masked Tail
            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES_F.indexInRange(0, remaining);
                FloatVector va  = FloatVector.fromArray(SPECIES_F, src, srcOffset + i, mask);
                va.lanewise(op).intoArray(dest, destOffset + i, mask);
            }
        }
    }
}
