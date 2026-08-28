package com.github.gbenroscience.simdext.turbo.tools.command.v2;
 
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*; 
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression.*; 

import com.github.gbenroscience.simdext.turbo.tools.utils.CPUPinner;
import com.github.gbenroscience.simdext.turbo.tools.utils.VectorMathF;
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
public class SIMDCommandSegmentF32 extends VectorTurboEvaluator {

    // NOTE: VectorConfig's statically-imported SPECIES is a VectorSpecies<Double>.
    // This class-local SPECIES shadows it for every float-based command/record below,
    // so the parallelism architecture and everything else pulled in from
    // VectorTurboEvaluator / VectorConfig is left completely untouched.
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public SIMDCommandSegmentF32(MathExpression me) throws Throwable {
        super(me);
    }

    public SIMDCommandSegmentF32(MathExpression me, int numWorkers) throws Throwable {
        super(me, numWorkers);
    }
 
    public static final SIMDCommandSegmentF32.SIMDVectorCompositeExpression getEvaluator(MathExpression me) throws Throwable {
        return (SIMDCommandSegmentF32.SIMDVectorCompositeExpression) new SIMDCommandSegmentF32(me).compile();
    }

    public static final SIMDCommandSegmentF32.SIMDVectorCompositeExpression getEvaluator(String expr) throws Throwable {
        return (SIMDCommandSegmentF32.SIMDVectorCompositeExpression) new SIMDCommandSegmentF32(new MathExpression(expr)).compile();
    }

    public static final SIMDCommandSegmentF32.SIMDVectorCompositeExpression getEvaluator(MathExpression me, int numWorkers) throws Throwable {
        return (SIMDCommandSegmentF32.SIMDVectorCompositeExpression) new SIMDCommandSegmentF32(me, numWorkers).compile();
    }

    public static final SIMDCommandSegmentF32.SIMDVectorCompositeExpression getEvaluator(String expr, int numWorkers) throws Throwable {
        return (SIMDCommandSegmentF32.SIMDVectorCompositeExpression) new SIMDCommandSegmentF32(new MathExpression(expr), numWorkers).compile();
    }

    // 1. Updated Command Interface
    @FunctionalInterface
    static interface VectorCommand {

        void execute(EvaluationContext ctx, int n);
    }

// 2. Ultra-lean Context (Zero dynamic stack allocation)
    private static final class EvaluationContext {

        final float[] scratch;
        float[] flatVariables;
        float[][] _2DVariables;
        // MemorySegment-backed variable sources (off-heap / zero-copy path).
        // Mutually exclusive with the float[] / float[][] fields above -
        // whichever init method was called last clears the other pair.
        MemorySegment flatVariablesSeg;
        MemorySegment[] _2DVariablesSeg;
        int dataSize;
        int blockStart;

        EvaluationContext(int maxStackDepth, int blockSize) {
            // Only one flat scratch pad is needed!
            scratch = new float[maxStackDepth * blockSize];
        }

        void initForBlock(float[] flat, float[][] _2D, int size, int bStart) {
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

    record ConstCommand(float value, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            FloatVector v = FloatVector.broadcast(SPECIES, value);
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
                MemorySegment.copy(ctx.flatVariablesSeg, ValueLayout.JAVA_FLOAT, srcOff * ValueLayout.JAVA_FLOAT.byteSize(),
                        ctx.scratch, destOff, n);
            } else {
                MemorySegment.copy(ctx._2DVariablesSeg[slotIdx], ValueLayout.JAVA_FLOAT,
                        (long) ctx.blockStart * ValueLayout.JAVA_FLOAT.byteSize(), ctx.scratch, destOff, n);
            }
        }
    }

// --- Core Binary Operations ---
    record AddCommand(int lOff, int rOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .add(FloatVector.fromArray(SPECIES, s, rOff + k))
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
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .sub(FloatVector.fromArray(SPECIES, s, rOff + k))
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
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .mul(FloatVector.fromArray(SPECIES, s, rOff + k))
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
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .div(FloatVector.fromArray(SPECIES, s, rOff + k))
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
            VectorMathF.executePowerBlended(ctx.scratch, lOff, rOff, n);
            // Note: executePowerBlended writes to lOff. If dest != lOff, we must copy.
            // The compiler guarantees dest == lOff by reusing stack slots.
        }
    }

    record RemCommand(int lOff, int rOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                s[destOff + k] = s[lOff + k] % s[rOff + k];
            }
        }
    }

// --- Comparisons ---
    record CompareCommand(int lOff, int rOff, int destOff, int opcode) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            int l = lOff;
            int r = rOff;
            int d = destOff;

            switch (opcode) {
                case OP_GT -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] > s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_LT -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] < s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_EQ -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] == s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_NE -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] != s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_GE -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] >= s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_LE -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] <= s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                // Standard C-style floating-point truthiness: non-zero is true
                case OP_AND -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] != 0.0f && s[r + k] != 0.0f) ? 1.0f : 0.0f;
                    }
                }
                case OP_OR -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] != 0.0f || s[r + k] != 0.0f) ? 1.0f : 0.0f;
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
            float[] s = ctx.scratch;
            int k = 0, bound = SPECIES.loopBound(n);
            for (; k < bound; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, aOff + k)
                        .fma(FloatVector.fromArray(SPECIES, s, bOff + k),
                                FloatVector.fromArray(SPECIES, s, cOff + k))
                        .intoArray(s, destOff + k);
            }
            if (k < n) {
                var mask = SPECIES.indexInRange(k, n);
                FloatVector.fromArray(SPECIES, s, aOff + k, mask)
                        .fma(FloatVector.fromArray(SPECIES, s, bOff + k, mask),
                                FloatVector.fromArray(SPECIES, s, cOff + k, mask))
                        .intoArray(s, destOff + k, mask);
            }
        }
    }

    record IfCommand(int condOff, int trueOff, int falseOff, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                s[destOff + k] = (s[condOff + k] != 0.0f) ? s[trueOff + k] : s[falseOff + k];
            }
        }
    }

// --- Unified Unary Operations (Delegates to VectorMath) ---
    @FunctionalInterface
    interface UnaryMathOp {

        void apply(int base, int n, float[] scratch);
    }

    record UnaryMathCommand(UnaryMathOp op, int baseOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            op.apply(baseOff, n, ctx.scratch);
        }
    }

    @FunctionalInterface
    interface BinaryMathOp {

        void apply(int lOff, int rOff, int destOff, int n, float[] scratch);
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
                    // literalConstants is an inherited double[] field from VectorTurboEvaluator (untouched);
                    // narrow to float here at the boundary of our float-only command stream.
                    plan.add(new ConstCommand((float) literalConstants[i], dest));
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
                            VectorMathF::swiglu2;
                        case OP_GEGLU_2 ->
                            VectorMathF::geglu2;
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
                            VectorMathF::sqrt;
                        case OP_CBRT -> 
                            VectorMathF::cbrt;

                        case OP_GELU ->
                            VectorMathF::gelu;

                        case OP_GELU_FAST ->
                            VectorMathF::geluFast;
                        case OP_SWIGLU ->
                            VectorMathF::swiglu;
                        case OP_GEGLU ->
                            VectorMathF::gegluUnary;
                        case OP_ERF ->
                            VectorMathF::erf;
                        case OP_ABS ->
                            VectorMathF::abs;

                        // Standard Trig
                        case OP_SIN ->
                            VectorMathF::sin;
                        case OP_COS ->
                            VectorMathF::cos;
                        case OP_TAN ->
                            VectorMathF::tan;

                        // Degree Variants
                        case OP_SIN_DEG ->
                            VectorMathF::sinDeg;
                        case OP_COS_DEG ->
                            VectorMathF::cosDeg;
                        case OP_TAN_DEG ->
                            VectorMathF::tanDeg;

                        case OP_SIN_GRAD ->
                            VectorMathF::sinGrad;
                        case OP_COS_GRAD ->
                            VectorMathF::cosGrad;
                        case OP_TAN_GRAD ->
                            VectorMathF::tanGrad;

                        // Standard Inverse
                        case OP_ASIN, OP_ASIN_ALT, OP_ARC_SIN_ALT ->
                            VectorMathF::asin;
                        case OP_ACOS, OP_ACOS_ALT, OP_ARC_COS_ALT ->
                            VectorMathF::acos;
                        case OP_ATAN, OP_ATAN_ALT, OP_ARC_TAN_ALT ->
                            VectorMathF::atan;

                        case OP_ASIN_DEG, OP_ASIN_DEG_ALT, OP_ARC_SIN_ALT_DEG ->
                            VectorMathF::asinDeg;
                        case OP_ACOS_DEG, OP_ACOS_DEG_ALT, OP_ARC_COS_ALT_DEG ->
                            VectorMathF::acosDeg;
                        case OP_ATAN_DEG, OP_ATAN_DEG_ALT, OP_ARC_TAN_ALT_DEG ->
                            VectorMathF::atanDeg;

                        case OP_ASIN_GRAD, OP_ASIN_GRAD_ALT, OP_ARC_SIN_ALT_GRAD ->
                            VectorMathF::asinGrad;
                        case OP_ACOS_GRAD, OP_ACOS_GRAD_ALT, OP_ARC_COS_ALT_GRAD ->
                            VectorMathF::acosGrad;
                        case OP_ATAN_GRAD, OP_ATAN_GRAD_ALT, OP_ARC_TAN_ALT_GRAD ->
                            VectorMathF::atanGrad;

                        // Degree Variants
                        case OP_SEC_DEG ->
                            VectorMathF::secDeg;
                        case OP_COSEC_DEG ->
                            VectorMathF::cscDeg;
                        case OP_COT_DEG ->
                            VectorMathF::cotDeg;

                        case OP_SEC_GRAD ->
                            VectorMathF::secGrad;
                        case OP_COSEC_GRAD ->
                            VectorMathF::cscGrad;
                        case OP_COT_GRAD ->
                            VectorMathF::cotGrad;

                        // Standard Inverse
                        case OP_ARC_SEC, OP_ARC_SEC_ALT ->
                            VectorMathF::asec;
                        case OP_ARC_COSEC, OP_ARC_COSEC_ALT ->
                            VectorMathF::acsc;
                        case OP_ARC_COT, OP_ARC_COT_ALT ->
                            VectorMathF::acot;

                        case OP_ARC_SEC_DEG, OP_ARC_SEC_ALT_DEG ->
                            VectorMathF::asecDeg;
                        case OP_ARC_SEC_GRAD, OP_ARC_SEC_ALT_GRAD ->
                            VectorMathF::asecGrad;

                        case OP_ARC_COSEC_DEG, OP_ARC_COSEC_ALT_DEG ->
                            VectorMathF::acscDeg;
                        case OP_ARC_COSEC_GRAD, OP_ARC_COSEC_ALT_GRAD ->
                            VectorMathF::acscGrad;

                        case OP_ARC_COT_DEG, OP_ARC_COT_ALT_DEG ->
                            VectorMathF::acotDeg;
                        case OP_ARC_COT_GRAD, OP_ARC_COT_ALT_GRAD ->
                            VectorMathF::acotGrad;

                        case OP_SINH ->
                            VectorMathF::sinh;
                        case OP_COSH ->
                            VectorMathF::cosh;
                        case OP_TANH ->
                            VectorMathF::tanh;
                        case OP_ASINH, OP_ASINH_ALT ->
                            VectorMathF::asinh;
                        case OP_ACOSH, OP_ACOSH_ALT ->
                            VectorMathF::acosh;
                        case OP_ATANH, OP_ATANH_ALT ->
                            VectorMathF::atanh;

                        // Exp/Log
                        case OP_EXP ->
                            VectorMathF::exp;
                        case OP_LOG ->
                            VectorMathF::ln;
                        case OP_LOG10 ->
                            VectorMathF::log10;

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
        private final int masterPinTarget; // -1 = don't pin the master thread

        private volatile boolean isClosed = false;
        private final ThreadLocal<Boolean> masterPinned = ThreadLocal.withInitial(() -> false);

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

                // Query real physical-core topology instead of assuming logical CPUs
                // are interleaved by core. Each element of coreGroups is one physical
                // core's set of SMT-sibling logical indices; using group[i % groups.length][0]
                // as the pin target for worker i guarantees distinct physical cores
                // whenever enough exist, regardless of how the OS numbers hyperthread
                // siblings. The master gets its own reserved group beyond the workers'.
                int[][] coreGroups = CPUPinner.detectPhysicalCoreGroups();
                if (coreGroups == null || coreGroups.length == 0) {
                    // Degraded path: pin everyone to logical CPU 0 rather than
                    // NPE'ing construction when /sys topology is unreadable.
                    coreGroups = new int[][]{{0}};
                }
                this.masterPinTarget = coreGroups[NUM_WORKERS % coreGroups.length][0];

                for (int i = 0; i < NUM_WORKERS; i++) {
                    int pinTarget = coreGroups[i % coreGroups.length][0];
                    workerPool[i] = new WorkerThread(i, pinTarget, reuseLatch, executionPlan, stackDepth, blockSize);
                }

                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i].start();
                }

                this.cleanable = SYSTEM_CLEANER.register(this, new ThreadPoolShutdownAction(workerPool));
            } else {
                this.workerPool = null;
                this.reuseLatch = null;
                this.masterPinTarget = -1;
                this.cleanable = null;
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
            private final int pinTarget;
            private final AtomicInteger reuseLatch;
            private final EvaluationContext evalContext;
            private final VectorCommand[] executionPlan;
            private final int blockSize;

            private volatile boolean isRunning = true;
            private volatile int taskState = 0;
            private volatile Thread masterThread;

            private float[][] vars2D;
            private float[] vars1D;
            private float[] output;
            // MemorySegment-backed task inputs/output (off-heap / zero-copy path)
            private MemorySegment[] vars2DSeg;
            private MemorySegment vars1DSeg;
            private MemorySegment outputSeg;
            private int dataSize;
            private int startIdx;
            private int length;

            public WorkerThread(int workerId, int pinTarget, AtomicInteger reuseLatch, VectorCommand[] executionPlan, int stackDepth, int blockSize) {
                this.workerId = workerId;
                this.pinTarget = pinTarget;
                this.reuseLatch = reuseLatch;
                this.executionPlan = executionPlan;
                this.blockSize = blockSize;
                this.evalContext = new EvaluationContext(stackDepth, blockSize);
                this.setDaemon(true);
                this.setName("ParserNG-SIMD-Worker-" + workerId);
            }

            public void submitTask2D(float[][] vars, float[] output, int dataSize, int startIdx, int length, Thread master) {
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

            public void submitTask1D(float[] vars, float[] output, int dataSize, int startIdx, int length, Thread master) {
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
                CPUPinner.pinCurrentThread(this.pinTarget);
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

  
        public void applyBulk(float[][] variables, float[] output) {
            int numSamples = variables[0].length;
            applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }


        public void applyBulkParallel(float[][] variables, float[] output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = variables[0].length;

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            pinMasterIfNeeded();
            Thread masterThread = Thread.currentThread();
            // Master participates as slice NUM_WORKERS of NUM_WORKERS+1 total
            // participants, instead of sitting idle in park() while the pool
            // does all the work.
            final int totalParticipants = NUM_WORKERS + 1;
            int chunkSize = numSamples / totalParticipants;
            reuseLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                int startIdx = i * chunkSize;
                workerPool[i].submitTask2D(variables, output, numSamples, startIdx, chunkSize, masterThread);
            }

            int masterStartIdx = NUM_WORKERS * chunkSize;
            int masterLength = numSamples - masterStartIdx;
            if (masterLength > 0) {
                applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, masterStartIdx, masterLength);
            }

            while (reuseLatch.get() > 0) {
                LockSupport.park();
            }
        }

         
        public void applyBulkParallel(float[] flatVariables, float[] output) {
            if (flatVariables == null || output == null) {
                return;
            }
            int numSamples = output.length;

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            pinMasterIfNeeded();
            Thread masterThread = Thread.currentThread();
            // Master participates as slice NUM_WORKERS of NUM_WORKERS+1 total
            // participants, instead of sitting idle in park() while the pool
            // does all the work.
            final int totalParticipants = NUM_WORKERS + 1;
            int chunkSize = numSamples / totalParticipants;
            reuseLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                int startIdx = i * chunkSize;
                workerPool[i].submitTask1D(flatVariables, output, numSamples, startIdx, chunkSize, masterThread);
            }

            int masterStartIdx = NUM_WORKERS * chunkSize;
            int masterLength = numSamples - masterStartIdx;
            if (masterLength > 0) {
                applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, masterStartIdx, masterLength);
            }

            while (reuseLatch.get() > 0) {
                LockSupport.park();
            }
        }

        
        public void applyBulkBatched(float[][] variables, float[] output, int batchSize) {
            EvaluationContext ctx = masterEvalContext.get();
            int numSamples = variables[0].length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(variables, ctx, executionPlan, BLOCK_SIZE, numSamples, output, start, length);
            }
        }

       
        public void applyBulk(float[] flatVariables, float[] output) {
            applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, output.length, output, 0, output.length);
        }

 
        public void applyBulkBatched(float[] flatVariables, float[] output, int batchSize) {
            EvaluationContext ctx = masterEvalContext.get();
            int numSamples = output.length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(flatVariables, ctx, executionPlan, BLOCK_SIZE, numSamples, output, start, length);
            }
        }

        // --- MemorySegment (off-heap / zero-copy) entry points ---
        // Same public contract as the float[] / float[][] variants above,
        // reusing the same masterEvalContext / workerPool / reuseLatch machinery.
        public void applyBulk(MemorySegment[] variables, MemorySegment output) {
            int numSamples = (int) (variables[0].byteSize() / ValueLayout.JAVA_FLOAT.byteSize());
            applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(MemorySegment[] variables, MemorySegment output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = (int) (variables[0].byteSize() / ValueLayout.JAVA_FLOAT.byteSize());

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            pinMasterIfNeeded();
            Thread masterThread = Thread.currentThread();
            // Master participates as slice NUM_WORKERS of NUM_WORKERS+1 total
            // participants, instead of sitting idle in park() while the pool
            // does all the work.
            final int totalParticipants = NUM_WORKERS + 1;
            int chunkSize = numSamples / totalParticipants;
            reuseLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                int startIdx = i * chunkSize;
                workerPool[i].submitTaskSeg2D(variables, output, numSamples, startIdx, chunkSize, masterThread);
            }

            int masterStartIdx = NUM_WORKERS * chunkSize;
            int masterLength = numSamples - masterStartIdx;
            if (masterLength > 0) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, masterStartIdx, masterLength);
            }

            while (reuseLatch.get() > 0) {
                LockSupport.park();
            }
        }

        public void applyBulk(MemorySegment variables, MemorySegment output) {
            int numSamples = (int) (output.byteSize() / ValueLayout.JAVA_FLOAT.byteSize());
            applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(MemorySegment variables, MemorySegment output) {
            if (variables == null || output == null) {
                return;
            }
            int numSamples = (int) (output.byteSize() / ValueLayout.JAVA_FLOAT.byteSize());

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            pinMasterIfNeeded();
            Thread masterThread = Thread.currentThread();
            // Master participates as slice NUM_WORKERS of NUM_WORKERS+1 total
            // participants, instead of sitting idle in park() while the pool
            // does all the work.
            final int totalParticipants = NUM_WORKERS + 1;
            int chunkSize = numSamples / totalParticipants;
            reuseLatch.set(NUM_WORKERS);

            for (int i = 0; i < NUM_WORKERS; i++) {
                int startIdx = i * chunkSize;
                workerPool[i].submitTaskSeg1D(variables, output, numSamples, startIdx, chunkSize, masterThread);
            }

            int masterStartIdx = NUM_WORKERS * chunkSize;
            int masterLength = numSamples - masterStartIdx;
            if (masterLength > 0) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, masterStartIdx, masterLength);
            }

            while (reuseLatch.get() > 0) {
                LockSupport.park();
            }
        }

        // --- Core Internal Hot-Loops with Vectorized Copy Defenses ---
        private static void applyBulkInternal(float[] flatVariables, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, float[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            float[] s = ctx.scratch;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlock(flatVariables, null, dataSize, blockStart);

                for (int i = 0; i < executionPlan.length; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                // Vectorized output write back (assumes result is at scratch offset 0)
                int k = 0, limit = SPECIES.loopBound(currentBlockSize);
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, s, k)
                            .intoArray(output, blockStart + k);
                }
                for (; k < currentBlockSize; k++) {
                    output[blockStart + k] = s[k];
                }
            }
        }

        private static void applyBulkInternal(float[][] variables, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, float[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            float[] s = ctx.scratch;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlock(null, variables, dataSize, blockStart);

                for (int i = 0; i < executionPlan.length; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                // Vectorized output write back
                int k = 0, limit = SPECIES.loopBound(currentBlockSize);
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, s, k)
                            .intoArray(output, blockStart + k);
                }
                for (; k < currentBlockSize; k++) {
                    output[blockStart + k] = s[k];
                }
            }
        }

        // --- MemorySegment Core Internal Hot-Loops (mirrors the float[] / float[][] loops above) ---
        private static void applyBulkInternalSeg(MemorySegment flatVariablesSeg, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, MemorySegment output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            final long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            float[] s = ctx.scratch;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlockSeg(flatVariablesSeg, null, dataSize, blockStart);

                for (int i = 0; i < executionPlan.length; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                // Vectorized output write back (assumes result is at scratch offset 0)
                int k = 0, limit = SPECIES.loopBound(currentBlockSize);
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, s, k)
                            .intoMemorySegment(output, (blockStart + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < currentBlockSize; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, blockStart + k, s[k]);
                }
            }
        }

        private static void applyBulkInternalSeg(MemorySegment[] variablesSeg, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, MemorySegment output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            final long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            float[] s = ctx.scratch;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlockSeg(null, variablesSeg, dataSize, blockStart);

                for (int i = 0; i < executionPlan.length; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                // Vectorized output write back
                int k = 0, limit = SPECIES.loopBound(currentBlockSize);
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, s, k)
                            .intoMemorySegment(output, (blockStart + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < currentBlockSize; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, blockStart + k, s[k]);
                }
            }
        }
    }
 

}