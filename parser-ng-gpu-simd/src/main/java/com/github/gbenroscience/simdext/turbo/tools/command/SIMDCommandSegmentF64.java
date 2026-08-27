package com.github.gbenroscience.simdext.turbo.tools.command;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*; 
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression.*;
import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.*;

import com.github.gbenroscience.simdext.turbo.tools.utils.CPUPinner;

import com.github.gbenroscience.simdext.turbo.tools.utils.VectorMath;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.ref.Cleaner;
import java.nio.ByteOrder;
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
public class SIMDCommandSegmentF64 extends VectorTurboEvaluator {

    public SIMDCommandSegmentF64(MathExpression me) throws Throwable {
        super(me);
    }

    public SIMDCommandSegmentF64(MathExpression me, int numWorkers) throws Throwable {
        super(me, numWorkers);
    }
 
    public static final SIMDCommandSegmentF64.SIMDVectorCompositeExpression getEvaluator(MathExpression me) throws Throwable {
        return (SIMDCommandSegmentF64.SIMDVectorCompositeExpression) new SIMDCommandSegmentF64(me).compile();
    }

    public static final SIMDCommandSegmentF64.SIMDVectorCompositeExpression getEvaluator(String expr) throws Throwable {
        return (SIMDCommandSegmentF64.SIMDVectorCompositeExpression) new SIMDCommandSegmentF64(new MathExpression(expr)).compile();
    }

    public static final SIMDCommandSegmentF64.SIMDVectorCompositeExpression getEvaluator(MathExpression me, int numWorkers) throws Throwable {
        return (SIMDCommandSegmentF64.SIMDVectorCompositeExpression) new SIMDCommandSegmentF64(me, numWorkers).compile();
    }

    public static final SIMDCommandSegmentF64.SIMDVectorCompositeExpression getEvaluator(String expr, int numWorkers) throws Throwable {
        return (SIMDCommandSegmentF64.SIMDVectorCompositeExpression) new SIMDCommandSegmentF64(new MathExpression(expr), numWorkers).compile();
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
        // MemorySegment-backed variable sources (off-heap / zero-copy path).
        // Mutually exclusive with the double[] / double[][] fields above -
        // whichever init method was called last clears the other pair.
        MemorySegment flatVariablesSeg;
        MemorySegment[] _2DVariablesSeg;
        int dataSize;
        int blockStart;

        EvaluationContext(int maxStackDepth, int blockSize) {
            // Only one flat scratch pad is needed!
            scratch = new double[maxStackDepth * blockSize];
        }

        void initForBlock(double[] flat, double[][] _2D, int size, int bStart) {
            this.flatVariables = flat;
            this._2DVariables = _2D;
            this.flatVariablesSeg = null;
            this._2DVariablesSeg = null;
            this.dataSize = size;
            this.blockStart = bStart;
        }

        void initForBlockSeg(MemorySegment flatSeg, MemorySegment[] _2DSeg, int size, int bStart) {
            this.flatVariables = null;
            this._2DVariables = null;
            this.flatVariablesSeg = flatSeg;
            this._2DVariablesSeg = _2DSeg;
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
            } else if (ctx._2DVariables != null) {
                System.arraycopy(ctx._2DVariables[slotIdx], ctx.blockStart, ctx.scratch, destOff, n);
            } else if (ctx.flatVariablesSeg != null) {
                long srcOff = ((long) slotIdx * ctx.dataSize) + ctx.blockStart;
                MemorySegment.copy(ctx.flatVariablesSeg, ValueLayout.JAVA_DOUBLE, srcOff * ValueLayout.JAVA_DOUBLE.byteSize(),
                        ctx.scratch, destOff, n);
            } else {
                MemorySegment.copy(ctx._2DVariablesSeg[slotIdx], ValueLayout.JAVA_DOUBLE,
                        (long) ctx.blockStart * ValueLayout.JAVA_DOUBLE.byteSize(), ctx.scratch, destOff, n);
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
        /**
         * 
         * @param executionPlan
         * @param stackDepth
         * @param blockSize 
         */
        public SIMDVectorCompositeExpression(VectorCommand[] executionPlan, int stackDepth, int blockSize) {
            super(compiledScalarHandle, opcodes, targetSlots, literalConstants, instructionCount, varCount, false);
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
            // MemorySegment-backed task inputs/output (off-heap / zero-copy path)
            private MemorySegment[] vars2DSeg;
            private MemorySegment vars1DSeg;
            private MemorySegment outputSeg;
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
                this.vars2DSeg = null;
                this.vars1DSeg = null;
                this.output = output;
                this.outputSeg = null;
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
                this.vars2DSeg = null;
                this.vars1DSeg = null;
                this.output = output;
                this.outputSeg = null;
                this.dataSize = dataSize;
                this.startIdx = startIdx;
                this.length = length;
                this.masterThread = master;
                this.taskState = 1;
                LockSupport.unpark(this);
            }

            public void submitTaskSeg2D(MemorySegment[] vars, MemorySegment output, int dataSize, int startIdx, int length, Thread master) {
                this.vars2DSeg = vars;
                this.vars1DSeg = null;
                this.vars2D = null;
                this.vars1D = null;
                this.outputSeg = output;
                this.output = null;
                this.dataSize = dataSize;
                this.startIdx = startIdx;
                this.length = length;
                this.masterThread = master;
                this.taskState = 1;
                LockSupport.unpark(this);
            }

            public void submitTaskSeg1D(MemorySegment vars, MemorySegment output, int dataSize, int startIdx, int length, Thread master) {
                this.vars1DSeg = vars;
                this.vars2DSeg = null;
                this.vars1D = null;
                this.vars2D = null;
                this.outputSeg = output;
                this.output = null;
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
                    } else if (vars2DSeg != null) {
                        applyBulkInternalSeg(vars2DSeg, evalContext, executionPlan, blockSize, dataSize, outputSeg, startIdx, length);
                    } else if (vars1DSeg != null) {
                        applyBulkInternalSeg(vars1DSeg, evalContext, executionPlan, blockSize, dataSize, outputSeg, startIdx, length);
                    }

                    this.taskState = 0;
                    this.vars2D = null;
                    this.vars1D = null;
                    this.output = null;
                    this.vars2DSeg = null;
                    this.vars1DSeg = null;
                    this.outputSeg = null;

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

        // --- MemorySegment (off-heap / zero-copy) entry points ---
        // Same public contract as the double[] / double[][] variants above,
        // reusing the same masterEvalContext / workerPool / reuseLatch machinery.
        public void applyBulk(MemorySegment[] variables, MemorySegment output) {
            int numSamples = (int) (variables[0].byteSize() / ValueLayout.JAVA_DOUBLE.byteSize());
            applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(MemorySegment[] variables, MemorySegment output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = (int) (variables[0].byteSize() / ValueLayout.JAVA_DOUBLE.byteSize());

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            Thread masterThread = Thread.currentThread();
            int chunkSize = numSamples / NUM_WORKERS;
            reuseLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                int startIdx = i * chunkSize;
                int length = (i == NUM_WORKERS - 1) ? (numSamples - startIdx) : chunkSize;
                workerPool[i].submitTaskSeg2D(variables, output, numSamples, startIdx, length, masterThread);
            }

            while (reuseLatch.get() > 0) {
                LockSupport.park();
            }
        }

        public void applyBulk(MemorySegment variables, MemorySegment output) {
            int numSamples = (int) (output.byteSize() / ValueLayout.JAVA_DOUBLE.byteSize());
            applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(MemorySegment variables, MemorySegment output) {
            if (variables == null || output == null) {
                return;
            }
            int numSamples = (int) (output.byteSize() / ValueLayout.JAVA_DOUBLE.byteSize());

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            Thread masterThread = Thread.currentThread();
            int chunkSize = numSamples / NUM_WORKERS;
            reuseLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                int startIdx = i * chunkSize;
                int length = (i == NUM_WORKERS - 1) ? (numSamples - startIdx) : chunkSize;
                workerPool[i].submitTaskSeg1D(variables, output, numSamples, startIdx, length, masterThread);
            }

            while (reuseLatch.get() > 0) {
                LockSupport.park();
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

        // --- MemorySegment Core Internal Hot-Loops (mirrors the double[] / double[][] loops above) ---
        private static void applyBulkInternalSeg(MemorySegment flatVariablesSeg, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, MemorySegment output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            final long elemBytes = ValueLayout.JAVA_DOUBLE.byteSize();
            double[] s = ctx.scratch;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlockSeg(flatVariablesSeg, null, dataSize, blockStart);

                for (int i = 0; i < executionPlan.length; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                // Vectorized output write back (assumes result is at scratch offset 0)
                int k = 0, limit = SPECIES.loopBound(currentBlockSize);
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, s, k)
                            .intoMemorySegment(output, (blockStart + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < currentBlockSize; k++) {
                    output.setAtIndex(ValueLayout.JAVA_DOUBLE, blockStart + k, s[k]);
                }
            }
        }

        private static void applyBulkInternalSeg(MemorySegment[] variablesSeg, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, MemorySegment output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            final long elemBytes = ValueLayout.JAVA_DOUBLE.byteSize();
            double[] s = ctx.scratch;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlockSeg(null, variablesSeg, dataSize, blockStart);

                for (int i = 0; i < executionPlan.length; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                // Vectorized output write back
                int k = 0, limit = SPECIES.loopBound(currentBlockSize);
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, s, k)
                            .intoMemorySegment(output, (blockStart + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < currentBlockSize; k++) {
                    output.setAtIndex(ValueLayout.JAVA_DOUBLE, blockStart + k, s[k]);
                }
            }
        }
    }

    

}