package com.github.gbenroscience.simdext.turbo.tools.command.v2;
  
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*; 
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression.*;
import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.*;


import com.github.gbenroscience.simdext.turbo.tools.utils.CPUPinner;
import com.github.gbenroscience.simdext.turbo.tools.utils.VectorMath; 
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
public class SIMDCommandF64 extends VectorTurboEvaluator {

    public SIMDCommandF64(MathExpression me) throws Throwable {
        super(me);
    }

    public SIMDCommandF64(MathExpression me, int numWorkers) throws Throwable {
        super(me, numWorkers);
    }
 
    public static final SIMDCommandF64.SIMDVectorCompositeExpression getEvaluator(MathExpression me) throws Throwable {
        return (SIMDCommandF64.SIMDVectorCompositeExpression) new SIMDCommandF64(me).compile();
    }

    public static final SIMDCommandF64.SIMDVectorCompositeExpression getEvaluator(String expr) throws Throwable {
        return (SIMDCommandF64.SIMDVectorCompositeExpression) new SIMDCommandF64(new MathExpression(expr)).compile();
    }

    public static final SIMDCommandF64.SIMDVectorCompositeExpression getEvaluator(MathExpression me, int numWorkers) throws Throwable {
        return (SIMDCommandF64.SIMDVectorCompositeExpression) new SIMDCommandF64(me, numWorkers).compile();
    }

    public static final SIMDCommandF64.SIMDVectorCompositeExpression getEvaluator(String expr, int numWorkers) throws Throwable {
        return (SIMDCommandF64.SIMDVectorCompositeExpression) new SIMDCommandF64(new MathExpression(expr), numWorkers).compile();
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

    // --- Fused Leaf Commands: Load(var) OP Load(var) ---
    // When both operands of a binary op are plain variable loads (the most
    // common shape for shallow expressions like a+b), compiling them as two
    // separate LoadCommands + one BinaryOp forces both operands through an
    // extra round-trip into ctx.scratch before the op even runs, and the op
    // itself writes a third copy back into scratch. These fused commands read
    // straight from the source arrays (flatVariables / _2DVariables) and skip
    // that intermediate materialization entirely. Selected at compile() time
    // via peephole-fusion over the emitted plan — see tryFuseLoadLoad().
    // Numerically identical to LoadCommand+LoadCommand+BinaryOp: same IEEE
    // op, same operand order, just a different source array.
    record LoadLoadAddCommand(int lSlot, int rSlot, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, lBase + k)
                            .add(DoubleVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flat[lBase + k] + flat[rBase + k];
                }
            } else {
                double[] l = ctx._2DVariables[lSlot];
                double[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, l, base + k)
                            .add(DoubleVector.fromArray(SPECIES, r, base + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = l[base + k] + r[base + k];
                }
            }
        }
    }

    record LoadLoadSubCommand(int lSlot, int rSlot, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, lBase + k)
                            .sub(DoubleVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flat[lBase + k] - flat[rBase + k];
                }
            } else {
                double[] l = ctx._2DVariables[lSlot];
                double[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, l, base + k)
                            .sub(DoubleVector.fromArray(SPECIES, r, base + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = l[base + k] - r[base + k];
                }
            }
        }
    }

    record LoadLoadMulCommand(int lSlot, int rSlot, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, lBase + k)
                            .mul(DoubleVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flat[lBase + k] * flat[rBase + k];
                }
            } else {
                double[] l = ctx._2DVariables[lSlot];
                double[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, l, base + k)
                            .mul(DoubleVector.fromArray(SPECIES, r, base + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = l[base + k] * r[base + k];
                }
            }
        }
    }

    record LoadLoadDivCommand(int lSlot, int rSlot, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, lBase + k)
                            .div(DoubleVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flat[lBase + k] / flat[rBase + k];
                }
            } else {
                double[] l = ctx._2DVariables[lSlot];
                double[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, l, base + k)
                            .div(DoubleVector.fromArray(SPECIES, r, base + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = l[base + k] / r[base + k];
                }
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

                    VectorCommand fused = tryFuseLoadLoad(plan, opcode, lOff, rOff, destOff);
                    if (fused != null) {
                        // Both LoadCommands are dead after this point: codegen is
                        // a strict LIFO stack machine, so lOff/rOff cannot be
                        // referenced by anything else once this op consumes them.
                        // Removing them skips materializing both operands into
                        // ctx.scratch - the fused command reads straight from
                        // flatVariables/_2DVariables instead.
                        plan.remove(plan.size() - 1);
                        plan.remove(plan.size() - 1);
                        plan.add(fused);
                    } else {
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
                    }
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

    /**
     * Peephole fusion: when both operands of a binary arithmetic op are
     * plain variable loads - i.e. the last two entries in the plan are
     * LoadCommands feeding directly into this op and nothing else - collapse
     * them into a single fused command that reads straight from the source
     * arrays instead of round-tripping both operands through ctx.scratch.
     * Returns null (no fusion) for anything that doesn't match that exact
     * shape, including OP_REM (not vectorized regardless) and OP_POW
     * (routes through VectorMath.executePowerBlended, not a plain lane op).
     */
    private static VectorCommand tryFuseLoadLoad(List<VectorCommand> plan, int opcode, int lOff, int rOff, int destOff) {
        int size = plan.size();
        if (size < 2) {
            return null;
        }
        if (!(plan.get(size - 1) instanceof LoadCommand rLoad) || rLoad.destOff() != rOff) {
            return null;
        }
        if (!(plan.get(size - 2) instanceof LoadCommand lLoad) || lLoad.destOff() != lOff) {
            return null;
        }

        return switch (opcode) {
            case OP_ADD ->
                new LoadLoadAddCommand(lLoad.slotIdx(), rLoad.slotIdx(), destOff);
            case OP_SUB ->
                new LoadLoadSubCommand(lLoad.slotIdx(), rLoad.slotIdx(), destOff);
            case OP_MUL ->
                new LoadLoadMulCommand(lLoad.slotIdx(), rLoad.slotIdx(), destOff);
            case OP_DIV ->
                new LoadLoadDivCommand(lLoad.slotIdx(), rLoad.slotIdx(), destOff);
            default ->
                null;
        };
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

            private double[][] vars2D;
            private double[] vars1D;
            private double[] output;
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

    

}