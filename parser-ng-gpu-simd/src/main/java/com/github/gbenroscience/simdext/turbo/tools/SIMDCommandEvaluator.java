package com.github.gbenroscience.simdext.turbo.tools;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*; 
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression.*;
import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.*;

import com.github.gbenroscience.simdext.turbo.tools.utils.CPUPinner;
import com.github.gbenroscience.simd.turbo.tools.utils.VectorizedCodyMath;
import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import jdk.incubator.vector.*;

/**
 * High-Performance Vector API & Engine that fuses explicit SIMD vectorization
 * with a zero-allocation primitive stack interpreter. Completely eliminates the
 * scalar parser overhead and task object allocations on the hot path.
 *
 * This version is the second fastest of all the SIMD evaluators.
 * Combines near zero-allocation with parallel operations greatly enhanced with cpu-pinning.
 * Cpu pinning is the reason why this class is a native of this extension and is the main reason
 * why this extension is JDK22+
 * Note that CPU PINNING works best on Linux, so the worker efficiency of these classes
 * is best seen on Linux. Where 2 workers perform at almost 2x the rate of one worker.. usually between 1.88x to 2.02x
 * 
 *
 */
public class SIMDCommandEvaluator extends VectorTurboEvaluator {

    public SIMDCommandEvaluator(MathExpression me) throws Throwable {
        super(me);
    }

    public SIMDCommandEvaluator(MathExpression me, int numWorkers) throws Throwable {
        super(me, numWorkers);
    }
 
    public static final SIMDCommandEvaluator.SIMDVectorCompositeExpression getEvaluator(MathExpression me) throws Throwable {
        return (SIMDCommandEvaluator.SIMDVectorCompositeExpression) new SIMDCommandEvaluator(me).compile();
    }

    public static final SIMDCommandEvaluator.SIMDVectorCompositeExpression getEvaluator(String expr) throws Throwable {
        return (SIMDCommandEvaluator.SIMDVectorCompositeExpression) new SIMDCommandEvaluator(new MathExpression(expr)).compile();
    }

    public static final SIMDCommandEvaluator.SIMDVectorCompositeExpression getEvaluator(MathExpression me, int numWorkers) throws Throwable {
        return (SIMDCommandEvaluator.SIMDVectorCompositeExpression) new SIMDCommandEvaluator(me).compile();
    }

    public static final SIMDCommandEvaluator.SIMDVectorCompositeExpression getEvaluator(String expr, int numWorkers) throws Throwable {
        return (SIMDCommandEvaluator.SIMDVectorCompositeExpression) new SIMDCommandEvaluator(new MathExpression(expr)).compile();
    }

    // 1. Updated Command Interface
    @FunctionalInterface
    static interface VectorCommand {

        void execute(EvaluationContext ctx, int n);
    }

// 2. Ultra-lean Context (Zero dynamic stack allocation)
    private static final class EvaluationContext {

        final double[] scratch;
        double[] flatVariables;
        double[][] _2DVariables;
        int dataSize;
        int blockStart;

        EvaluationContext(int maxStackDepth, int blockSize) {
            // Only one flat scratch pad is needed!
            scratch = new double[maxStackDepth * blockSize];
        }

        void initForBlock(double[] flat, double[][] _2D, int size, int bStart) {
            this.flatVariables = flat;
            this._2DVariables = _2D;
            this.dataSize = size;
            this.blockStart = bStart;
        }
    }
// --- Memory Operations ---

    record ConstCommand(double value, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            DoubleVector v = DoubleVector.broadcast(SPECIES, value);
            for (; k < limit; k += SPECIES.length()) {
                v.intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = value;
            }
        }
    }

    record LoadCommand(int slotIdx, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            if (ctx.flatVariables != null) {
                int srcOff = (slotIdx * ctx.dataSize) + ctx.blockStart;
                System.arraycopy(ctx.flatVariables, srcOff, ctx.scratch, destOff, n);
            } else {
                System.arraycopy(ctx._2DVariables[slotIdx], ctx.blockStart, ctx.scratch, destOff, n);
            }
        }
    }

// --- Core Binary Operations ---
    record AddCommand(int lOff, int rOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, lOff + k)
                        .add(DoubleVector.fromArray(SPECIES, s, rOff + k))
                        .intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = s[lOff + k] + s[rOff + k];
            }
        }
    }

    record SubCommand(int lOff, int rOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, lOff + k)
                        .sub(DoubleVector.fromArray(SPECIES, s, rOff + k))
                        .intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = s[lOff + k] - s[rOff + k];
            }
        }
    }

    record MulCommand(int lOff, int rOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, lOff + k)
                        .mul(DoubleVector.fromArray(SPECIES, s, rOff + k))
                        .intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = s[lOff + k] * s[rOff + k];
            }
        }
    }

    record DivCommand(int lOff, int rOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, lOff + k)
                        .div(DoubleVector.fromArray(SPECIES, s, rOff + k))
                        .intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = s[lOff + k] / s[rOff + k];
            }
        }
    }

    record PowCommand(int lOff, int rOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            VectorMath.executePowerBlended(ctx.scratch, lOff, rOff, n);
            // Note: executePowerBlended writes to lOff. If dest != lOff, we must copy.
            // The compiler guarantees dest == lOff by reusing stack slots.
        }
    }

    record RemCommand(int lOff, int rOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                s[destOff + k] = s[lOff + k] % s[rOff + k];
            }
        }
    }

// --- Comparisons ---
    record CompareCommand(int lOff, int rOff, int destOff, int opcode) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int l = lOff;
            int r = rOff;
            int d = destOff;

            switch (opcode) {
                case OP_GT -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] > s[r + k]) ? 1.0 : 0.0;
                    }
                }
                case OP_LT -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] < s[r + k]) ? 1.0 : 0.0;
                    }
                }
                case OP_EQ -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] == s[r + k]) ? 1.0 : 0.0;
                    }
                }
                case OP_NE -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] != s[r + k]) ? 1.0 : 0.0;
                    }
                }
                case OP_GE -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] >= s[r + k]) ? 1.0 : 0.0;
                    }
                }
                case OP_LE -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] <= s[r + k]) ? 1.0 : 0.0;
                    }
                }
                // Standard C-style floating-point truthiness: non-zero is true
                case OP_AND -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] != 0.0 && s[r + k] != 0.0) ? 1.0 : 0.0;
                    }
                }
                case OP_OR -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] != 0.0 || s[r + k] != 0.0) ? 1.0 : 0.0;
                    }
                }
                default ->
                    throw new IllegalArgumentException("Unknown comparison opcode: " + opcode);
            }
        }
    }

// --- Ternary / Branching ---
    record VmaCommand(int aOff, int bOff, int cOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, bound = SPECIES.loopBound(n);
            for (; k < bound; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, aOff + k)
                        .fma(DoubleVector.fromArray(SPECIES, s, bOff + k),
                                DoubleVector.fromArray(SPECIES, s, cOff + k))
                        .intoArray(s, destOff + k);
            }
            if (k < n) {
                var mask = SPECIES.indexInRange(k, n);
                DoubleVector.fromArray(SPECIES, s, aOff + k, mask)
                        .fma(DoubleVector.fromArray(SPECIES, s, bOff + k, mask),
                                DoubleVector.fromArray(SPECIES, s, cOff + k, mask))
                        .intoArray(s, destOff + k, mask);
            }
        }
    }

    record IfCommand(int condOff, int trueOff, int falseOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                s[destOff + k] = (s[condOff + k] != 0.0) ? s[trueOff + k] : s[falseOff + k];
            }
        }
    }

// --- Unified Unary Operations (Delegates to VectorMath) ---
    @FunctionalInterface
    interface UnaryMathOp {

        void apply(int base, int n, double[] scratch);
    }

    record UnaryMathCommand(UnaryMathOp op, int baseOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            op.apply(baseOff, n, ctx.scratch);
        }
    }

    @FunctionalInterface
    interface BinaryMathOp {

        void apply(int lOff, int rOff, int destOff, int n, double[] scratch);
    }

    record BinaryMathCommand(BinaryMathOp op, int lOff, int rOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            op.apply(lOff, rOff, destOff, n, ctx.scratch);
        }
    }

    @Override
    public BatchedVectorCompositeExpression compile() throws Throwable {
        List<VectorCommand> plan = new ArrayList<>(instructionCount);

        int BLOCK_SIZE = VectorTurboEvaluator.BatchedVectorCompositeExpression.BLOCK_SIZE;
        // Virtual stack to track memory offsets during compilation
        int[] virtualStack = new int[stackDepth];
        int sp = 0;

        for (int i = 0; i < instructionCount; i++) {
            final int opcode = opcodes[i];

            switch (opcode) {
                case OP_CONST -> {
                    int dest = sp * BLOCK_SIZE;
                    plan.add(new ConstCommand(literalConstants[i], dest));
                    virtualStack[sp++] = dest;
                }
                case OP_LOAD -> {
                    int dest = sp * BLOCK_SIZE;
                    plan.add(new LoadCommand(targetSlots[i], dest));
                    virtualStack[sp++] = dest;
                }

                // --- Binary Operations ---
                case OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_REM, OP_POW -> {
                    int rOff = virtualStack[--sp];
                    int lOff = virtualStack[--sp];
                    int destOff = lOff; // Reuse left slot to save space

                    plan.add(switch (opcode) {
                        case OP_ADD ->
                            new AddCommand(lOff, rOff, destOff);
                        case OP_SUB ->
                            new SubCommand(lOff, rOff, destOff);
                        case OP_MUL ->
                            new MulCommand(lOff, rOff, destOff);
                        case OP_DIV ->
                            new DivCommand(lOff, rOff, destOff);
                        case OP_REM ->
                            new RemCommand(lOff, rOff, destOff);
                        case OP_POW ->
                            new PowCommand(lOff, rOff, destOff);

                        default ->
                            throw new IllegalStateException();
                    });
                    virtualStack[sp++] = destOff;
                }

                // --- Binary Mathematical Activations ---
                case OP_SWIGLU_2, OP_GEGLU_2 -> {
                    int rOff = virtualStack[--sp];
                    int lOff = virtualStack[--sp];
                    int destOff = lOff; // Re-use the left stack slot to save memory

                    BinaryMathOp mathOp = switch (opcode) {
                        case OP_SWIGLU_2 ->
                            VectorMath::swiglu2;
                        case OP_GEGLU_2 ->
                            VectorMath::geglu2;
                        default ->
                            throw new IllegalStateException();
                    };

                    plan.add(new BinaryMathCommand(mathOp, lOff, rOff, destOff));
                    virtualStack[sp++] = destOff;
                }

                // --- Comparisons ---
                case OP_GT, OP_LT, OP_EQ, OP_NE, OP_GE, OP_LE, OP_AND, OP_OR -> {
                    int rOff = virtualStack[--sp];
                    int lOff = virtualStack[--sp];
                    int destOff = lOff;
                    plan.add(new CompareCommand(lOff, rOff, destOff, opcode));
                    virtualStack[sp++] = destOff;
                }

                // --- Ternary ---
                case OP_VMA -> {
                    int cOff = virtualStack[--sp];
                    int bOff = virtualStack[--sp];
                    int aOff = virtualStack[--sp];
                    int destOff = aOff;
                    plan.add(new VmaCommand(aOff, bOff, cOff, destOff));
                    virtualStack[sp++] = destOff;
                }
                case OP_IF -> {
                    int falseOff = virtualStack[--sp];
                    int trueOff = virtualStack[--sp];
                    int condOff = virtualStack[--sp];
                    int destOff = condOff;
                    plan.add(new IfCommand(condOff, trueOff, falseOff, destOff));
                    virtualStack[sp++] = destOff;
                }

                // --- Unary Math Operations ---
                default -> {
                    // All remaining valid opcodes are Unary/In-place operations.
                    int baseOff = virtualStack[sp - 1]; // Peak at top of stack (in-place)

                    UnaryMathOp mathOp = switch (opcode) {

                        case OP_SQRT ->
                            VectorMath::sqrt;
                        case OP_CBRT -> 
                            VectorMath::cbrt;

                        case OP_GELU ->
                            VectorMath::gelu;

                        case OP_GELU_FAST ->
                            VectorMath::geluFast;
                        case OP_SWIGLU ->
                            VectorMath::swiglu;
                        case OP_GEGLU ->
                            VectorMath::gegluUnary;
                        case OP_ERF ->
                            VectorMath::erf;
                        case OP_ABS ->
                            VectorMath::abs;

                        // Standard Trig
                        case OP_SIN ->
                            VectorMath::sin;
                        case OP_COS ->
                            VectorMath::cos;
                        case OP_TAN ->
                            VectorMath::tan;

                        // Degree Variants
                        case OP_SIN_DEG ->
                            VectorMath::sinDeg;
                        case OP_COS_DEG ->
                            VectorMath::cosDeg;
                        case OP_TAN_DEG ->
                            VectorMath::tanDeg;

                        case OP_SIN_GRAD ->
                            VectorMath::sinGrad;
                        case OP_COS_GRAD ->
                            VectorMath::cosGrad;
                        case OP_TAN_GRAD ->
                            VectorMath::tanGrad;

                        // Standard Inverse
                        case OP_ASIN, OP_ASIN_ALT, OP_ARC_SIN_ALT ->
                            VectorMath::asin;
                        case OP_ACOS, OP_ACOS_ALT, OP_ARC_COS_ALT ->
                            VectorMath::acos;
                        case OP_ATAN, OP_ATAN_ALT, OP_ARC_TAN_ALT ->
                            VectorMath::atan;

                        case OP_ASIN_DEG, OP_ASIN_DEG_ALT, OP_ARC_SIN_ALT_DEG ->
                            VectorMath::asinDeg;
                        case OP_ACOS_DEG, OP_ACOS_DEG_ALT, OP_ARC_COS_ALT_DEG ->
                            VectorMath::acosDeg;
                        case OP_ATAN_DEG, OP_ATAN_DEG_ALT, OP_ARC_TAN_ALT_DEG ->
                            VectorMath::atanDeg;

                        case OP_ASIN_GRAD, OP_ASIN_GRAD_ALT, OP_ARC_SIN_ALT_GRAD ->
                            VectorMath::asinGrad;
                        case OP_ACOS_GRAD, OP_ACOS_GRAD_ALT, OP_ARC_COS_ALT_GRAD ->
                            VectorMath::acosGrad;
                        case OP_ATAN_GRAD, OP_ATAN_GRAD_ALT, OP_ARC_TAN_ALT_GRAD ->
                            VectorMath::atanGrad;

                        // Degree Variants
                        case OP_SEC_DEG ->
                            VectorMath::secDeg;
                        case OP_COSEC_DEG ->
                            VectorMath::cscDeg;
                        case OP_COT_DEG ->
                            VectorMath::cotDeg;

                        case OP_SEC_GRAD ->
                            VectorMath::secGrad;
                        case OP_COSEC_GRAD ->
                            VectorMath::cscGrad;
                        case OP_COT_GRAD ->
                            VectorMath::cotGrad;

                        // Standard Inverse
                        case OP_ARC_SEC, OP_ARC_SEC_ALT ->
                            VectorMath::asec;
                        case OP_ARC_COSEC, OP_ARC_COSEC_ALT ->
                            VectorMath::acsc;
                        case OP_ARC_COT, OP_ARC_COT_ALT ->
                            VectorMath::acot;

                        case OP_ARC_SEC_DEG, OP_ARC_SEC_ALT_DEG ->
                            VectorMath::asecDeg;
                        case OP_ARC_SEC_GRAD, OP_ARC_SEC_ALT_GRAD ->
                            VectorMath::asecGrad;

                        case OP_ARC_COSEC_DEG, OP_ARC_COSEC_ALT_DEG ->
                            VectorMath::acscDeg;
                        case OP_ARC_COSEC_GRAD, OP_ARC_COSEC_ALT_GRAD ->
                            VectorMath::acscGrad;

                        case OP_ARC_COT_DEG, OP_ARC_COT_ALT_DEG ->
                            VectorMath::acotDeg;
                        case OP_ARC_COT_GRAD, OP_ARC_COT_ALT_GRAD ->
                            VectorMath::acotGrad;

                        case OP_SINH ->
                            VectorMath::sinh;
                        case OP_COSH ->
                            VectorMath::cosh;
                        case OP_TANH ->
                            VectorMath::tanh;
                        case OP_ASINH, OP_ASINH_ALT ->
                            VectorMath::asinh;
                        case OP_ACOSH, OP_ACOSH_ALT ->
                            VectorMath::acosh;
                        case OP_ATANH, OP_ATANH_ALT ->
                            VectorMath::atanh;

                        // Exp/Log
                        case OP_EXP ->
                            VectorMath::exp;
                        case OP_LOG ->
                            VectorMath::ln;
                        case OP_LOG10 ->
                            VectorMath::log10;

                        default ->
                            throw new UnsupportedOperationException("Unmapped opcode: " + opcode);
                    };

                    plan.add(new UnaryMathCommand(mathOp, baseOff));
                }
            }
        }

        return new SIMDVectorCompositeExpression(plan.toArray(new VectorCommand[0]), stackDepth, BLOCK_SIZE);
    }

    public final class SIMDVectorCompositeExpression extends BatchedVectorCompositeExpression implements AutoCloseable {

        private static final Cleaner SYSTEM_CLEANER = Cleaner.create();

        private final int NUM_WORKERS;
        private final WorkerThread[] workerPool;
        private final AtomicInteger reuseLatch;
        private final VectorCommand[] executionPlan;
        private final ThreadLocal<EvaluationContext> masterEvalContext;
        private final Cleaner.Cleanable cleanable;

        private volatile boolean isClosed = false;

        private static final class ThreadPoolShutdownAction implements Runnable {

            private final WorkerThread[] pool;

            ThreadPoolShutdownAction(WorkerThread[] pool) {
                this.pool = pool;
            }

            @Override
            public void run() {
                if (pool != null) {
                    for (WorkerThread worker : pool) {
                        if (worker != null) {
                            worker.terminate();
                        }
                    }
                }
            }
        }

        public SIMDVectorCompositeExpression(VectorCommand[] executionPlan, int stackDepth, int blockSize) {
            super(compiledScalarHandle, opcodes, targetSlots, literalConstants, instructionCount, varCount);
            this.executionPlan = executionPlan;
            this.masterEvalContext = ThreadLocal.withInitial(() -> new EvaluationContext(stackDepth, blockSize));

            if (numWorkers <= 2) {
                this.NUM_WORKERS = numWorkers;
            } else {
                this.NUM_WORKERS = numWorkers - 1;
            }

            if (this.NUM_WORKERS > 0) {
                this.workerPool = new WorkerThread[NUM_WORKERS];
                this.reuseLatch = new AtomicInteger(0);

                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i] = new WorkerThread(i, reuseLatch, executionPlan, stackDepth, blockSize);
                }

                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i].start();
                }

                this.cleanable = SYSTEM_CLEANER.register(this, new ThreadPoolShutdownAction(workerPool));
            } else {
                this.workerPool = null;
                this.reuseLatch = null;
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

        private static final class WorkerThread extends Thread {

            private final int workerId;
            private final AtomicInteger reuseLatch;
            private final EvaluationContext evalContext;
            private final VectorCommand[] executionPlan;
            private final int blockSize;

            private volatile boolean isRunning = true;
            private volatile int taskState = 0;
            private volatile Thread masterThread;

            private double[][] vars2D;
            private double[] vars1D;
            private double[] output;
            private int dataSize;
            private int startIdx;
            private int length;

            public WorkerThread(int workerId, AtomicInteger reuseLatch, VectorCommand[] executionPlan, int stackDepth, int blockSize) {
                this.workerId = workerId;
                this.reuseLatch = reuseLatch;
                this.executionPlan = executionPlan;
                this.blockSize = blockSize;
                this.evalContext = new EvaluationContext(stackDepth, blockSize);
                this.setDaemon(true);
                this.setName("ParserNG-SIMD-Worker-" + workerId);
            }

            public void submitTask2D(double[][] vars, double[] output, int dataSize, int startIdx, int length, Thread master) {
                this.vars2D = vars;
                this.vars1D = null;
                this.output = output;
                this.dataSize = dataSize;
                this.startIdx = startIdx;
                this.length = length;
                this.masterThread = master;
                this.taskState = 1;
                LockSupport.unpark(this);
            }

            public void submitTask1D(double[] vars, double[] output, int dataSize, int startIdx, int length, Thread master) {
                this.vars1D = vars;
                this.vars2D = null;
                this.output = output;
                this.dataSize = dataSize;
                this.startIdx = startIdx;
                this.length = length;
                this.masterThread = master;
                this.taskState = 1;
                LockSupport.unpark(this);
            }

            public void terminate() {
                this.isRunning = false;
                this.interrupt();
            }

            @Override
            public void run() {
                CPUPinner.pinCurrentThread(this.workerId);
                while (isRunning) {
                    while (taskState == 0 && isRunning) {
                        LockSupport.park();
                        if (Thread.interrupted()) {
                            return;
                        }
                    }
                    if (!isRunning) {
                        return;
                    }

                    if (vars2D != null) {
                        applyBulkInternal(vars2D, evalContext, executionPlan, blockSize, dataSize, output, startIdx, length);
                    } else if (vars1D != null) {
                        applyBulkInternal(vars1D, evalContext, executionPlan, blockSize, dataSize, output, startIdx, length);
                    }

                    this.taskState = 0;
                    this.vars2D = null;
                    this.vars1D = null;
                    this.output = null;

                    if (reuseLatch.decrementAndGet() == 0 && masterThread != null) {
                        LockSupport.unpark(masterThread);
                    }
                }
            }
        }

        @Override
        public void applyBulk(double[][] variables, double[] output) {
            int numSamples = variables[0].length;
            applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        @Override
        public void applyBulkParallel(double[][] variables, double[] output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = variables[0].length;

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            Thread masterThread = Thread.currentThread();
            int chunkSize = numSamples / NUM_WORKERS;
            reuseLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                int startIdx = i * chunkSize;
                int length = (i == NUM_WORKERS - 1) ? (numSamples - startIdx) : chunkSize;
                workerPool[i].submitTask2D(variables, output, numSamples, startIdx, length, masterThread);
            }

            while (reuseLatch.get() > 0) {
                LockSupport.park();
            }
        }

        @Override
        public void applyBulkParallel(double[] flatVariables, double[] output) {
            if (flatVariables == null || output == null) {
                return;
            }
            int numSamples = output.length;

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            Thread masterThread = Thread.currentThread();
            int chunkSize = numSamples / NUM_WORKERS;
            reuseLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                int startIdx = i * chunkSize;
                int length = (i == NUM_WORKERS - 1) ? (numSamples - startIdx) : chunkSize;
                workerPool[i].submitTask1D(flatVariables, output, numSamples, startIdx, length, masterThread);
            }

            while (reuseLatch.get() > 0) {
                LockSupport.park();
            }
        }

        @Override
        public void applyBulkBatched(double[][] variables, double[] output, int batchSize) {
            EvaluationContext ctx = masterEvalContext.get();
            int numSamples = variables[0].length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(variables, ctx, executionPlan, BLOCK_SIZE, numSamples, output, start, length);
            }
        }

        @Override
        public void applyBulk(double[] flatVariables, double[] output) {
            applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, output.length, output, 0, output.length);
        }

        @Override
        public void applyBulkBatched(double[] flatVariables, double[] output, int batchSize) {
            EvaluationContext ctx = masterEvalContext.get();
            int numSamples = output.length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(flatVariables, ctx, executionPlan, BLOCK_SIZE, numSamples, output, start, length);
            }
        }

        // --- Core Internal Hot-Loops with Vectorized Copy Defenses ---
        private static void applyBulkInternal(double[] flatVariables, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, double[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            double[] s = ctx.scratch;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlock(flatVariables, null, dataSize, blockStart);

                for (int i = 0; i < executionPlan.length; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                // Vectorized output write back (assumes result is at scratch offset 0)
                int k = 0, limit = SPECIES.loopBound(currentBlockSize);
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, s, k)
                            .intoArray(output, blockStart + k);
                }
                for (; k < currentBlockSize; k++) {
                    output[blockStart + k] = s[k];
                }
            }
        }

        private static void applyBulkInternal(double[][] variables, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, double[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            double[] s = ctx.scratch;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlock(null, variables, dataSize, blockStart);

                for (int i = 0; i < executionPlan.length; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                // Vectorized output write back
                int k = 0, limit = SPECIES.loopBound(currentBlockSize);
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, s, k)
                            .intoArray(output, blockStart + k);
                }
                for (; k < currentBlockSize; k++) {
                    output[blockStart + k] = s[k];
                }
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
