package com.github.gbenroscience.simd.turbo.tools;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.*;

import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression.BLOCK_SIZE;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression.PARALLEL_OPS_THRESHOLD;
import com.github.gbenroscience.simd.turbo.tools.utils.CPUPinner;
import com.github.gbenroscience.simd.turbo.tools.utils.VectorizedCodyMath;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import jdk.incubator.vector.*;

/**
 * High-Performance Vector API & Engine that fuses explicit SIMD vectorization
 * with a zero-allocation primitive stack interpreter. Completely eliminates the
 * scalar parser overhead and task object allocations on the hot path.
 *
 * This version is the fastest of all the SIMD evaluators.
 * Combines near zero-allocation with parallel operations greatly enhanced with cpu-pinning.
 * Cpu pinning is the reason why this class is a native of this extension and is the main reason
 * why this extension is JDK22+
 * Note that CPU PINNING works best on Linux, so the worker efficiency of these classes
 * is best seen on Linux. Where 2 workers perform at almost 2x the rate of one worker.. usually between 1.88x to 2.02x
 * 
 *
 */
public class SIMDEngineEvaluator extends VectorTurboEvaluator {

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
        return (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(me).compile();
    }

    public static final SIMDEngineEvaluator.SIMDVectorCompositeExpression getEvaluator(String expr, int numWorkers) throws Throwable {
        return (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(new MathExpression(expr)).compile();
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

        private volatile boolean isClosed = false;

        private final ThreadLocal<EvaluationContext> masterEvalContext;

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

                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i] = new WorkerThread(i, NUM_WORKERS, stackDepth, blockSize, coordinationContext);
                }
                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i].start();
                }
                this.cleanable = SYSTEM_CLEANER.register(this, new ThreadPoolShutdownAction(coordinationContext, workerPool));
            } else {
                this.coordinationContext = null;
                this.workerPool = null;
                this.cleanable = null;
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
            MemorySegment outputSegment;
            long totalSamplesLong;
            
            int totalSamples;

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
                this.output = null;
                this.outputSegment = null;
                this.masterThread = null;
            }
        }

        private final class WorkerThread extends Thread {

            private final int workerId;
            private final int totalWorkers;
            private final EvaluationContext evalContext;
            private final CoordinationContext ctx;

            WorkerThread(int workerId, int totalWorkers, int stackDepth, int blockSize, CoordinationContext ctx) {
                this.workerId = workerId;
                this.totalWorkers = totalWorkers;
                this.ctx = ctx;
                this.evalContext = new EvaluationContext(stackDepth, blockSize, varCount);
                this.setDaemon(true);
                this.setName("ParserNG-SIMD-Worker-" + workerId);
            }

            @Override
            public void run() {
                CPUPinner.pinCurrentThread(workerId);
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
                        if (sharedCtx.varsSegment != null) {
                            long numSamplesL = sharedCtx.totalSamplesLong;
                            long chunkSizeL = numSamplesL / totalThreads;
                            long startIdxL = id * chunkSizeL;
                            long lengthL = (id == totalThreads - 1) ? (numSamplesL - startIdxL) : chunkSizeL;

                            if (lengthL > 0) {
                                applyBulkInternal(sharedCtx.varsSegment, evalContext, numSamplesL, sharedCtx.outputSegment, startIdxL, lengthL);
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

        private void executeParallelProcessing(double[][] vars2D, double[] vars1D, double[] output, int numSamples) {
            coordinationContext.vars2D = vars2D;
            coordinationContext.vars1D = vars1D;
            coordinationContext.output = output;
            coordinationContext.totalSamples = numSamples;
            coordinationContext.masterThread = Thread.currentThread();
            coordinationContext.completionLatch.set(NUM_WORKERS);

            // Trigger execution via explicit atomic state updates
            for (int i = 0; i < NUM_WORKERS; i++) {
                coordinationContext.workerSignals[i].set(1);
                LockSupport.unpark(workerPool[i]);
            }

            while (coordinationContext.completionLatch.get() > 0) {
                LockSupport.park();
            }
            coordinationContext.clearPayload();
        }

        private void executeParallelProcessing(MemorySegment varsSegment, MemorySegment outputSegment, long numSamples) {
            coordinationContext.varsSegment = varsSegment;
            coordinationContext.outputSegment = outputSegment;
            coordinationContext.totalSamplesLong = numSamples;
            coordinationContext.masterThread = Thread.currentThread();
            coordinationContext.completionLatch.set(NUM_WORKERS);

            // Trigger execution via explicit atomic state updates
            for (int i = 0; i < NUM_WORKERS; i++) {
                coordinationContext.workerSignals[i].set(1);
                LockSupport.unpark(workerPool[i]);
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
            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
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
            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
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
             if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                 applyBulkInternal(variables, masterEvalContext.get(), numSamples, output, 0L, numSamples);
                 return;
             }
             executeParallelProcessing(variables, output, numSamples);
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
            int sp = 0;

            double[] flatVariables;
            double[][] _2DVariables;
            int dataSize;
            int blockStart;
            
            final double[][] segmentBlockVars;
            final MemorySegment[] segmentBlockVarsMem;

            EvaluationContext(int maxStackDepth, int blockSize, int varCount) {
                stackArrays = new double[maxStackDepth][];
                stackOffsets = new int[maxStackDepth];
                stackIsConst = new boolean[maxStackDepth];
                stackConstVals = new double[maxStackDepth];
                scratch = new double[maxStackDepth * blockSize];
                
                // Zero-allocation pre-cached memory segment loading targets
                segmentBlockVars = new double[varCount][blockSize];
                segmentBlockVarsMem = new MemorySegment[varCount];
                for (int i = 0; i < varCount; i++) {
                    segmentBlockVarsMem[i] = MemorySegment.ofArray(segmentBlockVars[i]);
                }
            }

            void initForBlock(double[] flat, double[][] _2D, int size, int bStart) {
                this.sp = 0;
                this.flatVariables = flat;
                this._2DVariables = _2D;
                this.dataSize = size;
                this.blockStart = bStart;
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
        // Core Execution Loop (Zero Allocation, C2 EA Guarded)
        // =========================================================================
        private void executeInstructions(EvaluationContext ctx, int n) {
            for (int instIdx = 0; instIdx < instructionCount; instIdx++) {
                final int opcode = opcodes[instIdx];

                switch (opcode) {
                    case OP_CONST -> {
                        ctx.stackIsConst[ctx.sp] = true;
                        ctx.stackConstVals[ctx.sp] = literalConstants[instIdx];
                        ctx.sp++;
                    }

                    case OP_LOAD -> {
                        final int slotIdx = targetSlots[instIdx];
                        if (ctx.flatVariables != null) {
                            ctx.stackArrays[ctx.sp] = ctx.flatVariables;
                            ctx.stackOffsets[ctx.sp] = (slotIdx * ctx.dataSize) + ctx.blockStart;
                        } else {
                            ctx.stackArrays[ctx.sp] = ctx._2DVariables[slotIdx];
                            ctx.stackOffsets[ctx.sp] = ctx.blockStart;
                        }
                        ctx.stackIsConst[ctx.sp] = false;
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
                        SIMDCommandEvaluator.VectorMath.cotGrad((ctx.sp - 1) * BLOCK_SIZE, n, ctx.scratch);
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
                    case OP_GT -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);
                        materialize(ctx, ctx.sp + 1, n);
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        ctx.sp++;
                        for (int k = 0; k < n; k++) {
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

                        for (int k = 0; k < n; k++) {
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
                                    false; // Should never hit
                            };
                            ctx.scratch[resOffset + k] = condition ? 1.0 : 0.0;
                        }

                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }
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
                        for (int k = 0; k < n; k++) {
                            ctx.scratch[resOffset + k] = (ctx.scratch[condOffset + k] != 0.0)
                                    ? ctx.scratch[trueOffset + k] : ctx.scratch[falseOffset + k];
                        }
                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }
                    case OP_AND -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);     // left operand
                        materialize(ctx, ctx.sp + 1, n); // right operand
                        
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        
                        ctx.sp++;
                        
                        for (int k = 0; k < n; k++) {
                            ctx.scratch[resOffset + k] = 
                                (ctx.scratch[lOffset + k] != 0.0 && ctx.scratch[rOffset + k] != 0.0) ? 1.0 : 0.0;
                        }
                        
                        ctx.stackArrays[ctx.sp - 1] = ctx.scratch;
                        ctx.stackOffsets[ctx.sp - 1] = resOffset;
                        ctx.stackIsConst[ctx.sp - 1] = false;
                    }

                    case OP_OR -> {
                        ctx.sp -= 2;
                        materialize(ctx, ctx.sp, n);     // left operand
                        materialize(ctx, ctx.sp + 1, n); // right operand
                        
                        final int lOffset = ctx.sp * BLOCK_SIZE;
                        final int rOffset = (ctx.sp + 1) * BLOCK_SIZE;
                        final int resOffset = ctx.sp * BLOCK_SIZE;
                        
                        ctx.sp++;
                        
                        for (int k = 0; k < n; k++) {
                            ctx.scratch[resOffset + k] = 
                                (ctx.scratch[lOffset + k] != 0.0 || ctx.scratch[rOffset + k] != 0.0) ? 1.0 : 0.0;
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

        // =========================================================================
        // Delegated Micro-Methods for Binary Operations (EA Guarded)---
        // =========================================================================
        private void doAdd(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final double[] rArr = ctx.stackArrays[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final double rVal = ctx.stackConstVals[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final double[] lArr = ctx.stackArrays[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final double lVal = ctx.stackConstVals[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArrays[ctx.sp] = ctx.scratch;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);

            if (!lIsConst && !rIsConst) {
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

        private void doSub(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final double[] rArr = ctx.stackArrays[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final double rVal = ctx.stackConstVals[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final double[] lArr = ctx.stackArrays[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final double lVal = ctx.stackConstVals[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArrays[ctx.sp] = ctx.scratch;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);

            if (!lIsConst && !rIsConst) {
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

        private void doMul(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final double[] rArr = ctx.stackArrays[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final double rVal = ctx.stackConstVals[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final double[] lArr = ctx.stackArrays[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final double lVal = ctx.stackConstVals[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArrays[ctx.sp] = ctx.scratch;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);

            if (!lIsConst && !rIsConst) {
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

        private void doDiv(EvaluationContext ctx, int n) {
            final int rSp = --ctx.sp;
            final boolean rIsConst = ctx.stackIsConst[rSp];
            final double[] rArr = ctx.stackArrays[rSp];
            final int rOff = ctx.stackOffsets[rSp];
            final double rVal = ctx.stackConstVals[rSp];

            final int lSp = --ctx.sp;
            final boolean lIsConst = ctx.stackIsConst[lSp];
            final double[] lArr = ctx.stackArrays[lSp];
            final int lOff = ctx.stackOffsets[lSp];
            final double lVal = ctx.stackConstVals[lSp];

            final int resOffset = ctx.sp * BLOCK_SIZE;
            ctx.stackArrays[ctx.sp] = ctx.scratch;
            ctx.stackOffsets[ctx.sp] = resOffset;
            ctx.stackIsConst[ctx.sp] = false;
            ctx.sp++;

            int k = 0;
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);

            if (!lIsConst && !rIsConst) {
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
    }

    public final class VectorMath {

        private VectorMath() {
        }

        private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
        public static int VECTOR_THRESHOLD = 256;

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

            int remaining = k - n; // Wait, original had n - k, let's keep it safe:
            remaining = n - k;
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
            DoubleVector absX = x.abs();

            VectorMask<Double> maskLow = absX.compare(VectorOperators.LE, THRESHOLD_LOW);
            VectorMask<Double> maskHigh = absX.compare(VectorOperators.LE, THRESHOLD_HIGH);

            DoubleVector xSq = absX.mul(absX);
            DoubleVector p = xSq.mul(0.260194122534674).add(30.59022585250011).mul(xSq)
                    .add(573.9507736045833).mul(xSq).add(2801.752391065013).mul(xSq).add(3204.677458505002);
            DoubleVector q = xSq.add(159.0884090976454).mul(xSq).add(1422.080683811422).mul(xSq)
                    .add(4423.613442045816).mul(xSq).add(3204.677458506958);

            DoubleVector resLow = x.mul(p.div(q));

            // Note: Assumes VectorizedCodyMath is available in your classpath
            DoubleVector resMed = VectorizedCodyMath.evaluateMediumVector(x, absX, maskHigh);
            DoubleVector resHigh = VectorizedCodyMath.evaluateLargeVector(x, absX, maskHigh.not());
            DoubleVector erfcVal = resMed.blend(resHigh, maskHigh.not());

            DoubleVector resMidHigh = V_ONE.sub(erfcVal);
            VectorMask<Double> isNegative = x.compare(VectorOperators.LT, 0.0);
            resMidHigh = resMidHigh.blend(resMidHigh.neg(), isNegative);

            return resMidHigh.blend(resLow, maskLow);
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

    }

    public static final class VectorTranscendentals {

        private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

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
    }
}