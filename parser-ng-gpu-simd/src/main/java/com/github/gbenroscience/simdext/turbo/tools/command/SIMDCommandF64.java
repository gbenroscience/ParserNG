package com.github.gbenroscience.simdext.turbo.tools.command;
 
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
import java.util.InputMismatchException;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import jdk.incubator.vector.*;

/**
 * High-Performance float64(Java's double type) Vector API & Engine that fuses
 * explicit SIMD vectorization with a zero-allocation primitive stack
 * interpreter. Completely eliminates the scalar parser overhead and task object
 * allocations on the hot path.
 *
 * These are the fastest of all the SIMD evaluators. Combines near
 * zero-allocation with parallel operations greatly enhanced with cpu-pinning.
 * Cpu pinning is the reason why this class is a native of this extension and is
 * the main reason why this extension is JDK22+ Note that CPU PINNING works best
 * on Linux, so the worker efficiency of these classes is best seen on Linux.
 * Where 2 workers perform at almost 2x the rate of one worker.. usually between
 * 1.4x to 2.0x
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

    /**
     * Optional extension of {@link VectorCommand} for command types that can
     * write their result directly into the plan's final output array --
     * {@code output[outputOffset..+n]} -- instead of into {@code ctx.scratch}.
     *
     * <p>Implemented only by command types whose math is plain {@code double[]}
     * lane arithmetic. {@link PowCommand} (routes through
     * {@code VectorMath.executePowerBlended(ctx.scratch, ...)}),
     * {@link UnaryMathCommand}/{@link LoadUnaryMathCommand}, and
     * {@link BinaryMathCommand} deliberately do NOT implement this: their
     * math is delegated to a functional interface whose signature is fixed
     * to {@code double[] scratch}, and retargeting that would mean either
     * changing those interfaces' signatures (a much larger change touching
     * every one of the ~50 unary/binary math ops) or duplicating
     * {@code VectorMath}'s internals outside {@code VectorMath} - neither is
     * done here.
     *
     * <p>When the LAST command in a compiled plan implements this interface,
     * {@code applyBulkInternal} calls {@link #executeToOutput} for it
     * instead of {@code execute(...)} followed by the separate
     * scratch-to-output writeback pass - eliminating one full extra
     * read+write pass over the block for exactly that case. Every command
     * before the last one in the plan is completely unaffected either way;
     * this only ever changes how the FINAL result reaches the output
     * array. When the last command does not implement this interface,
     * {@code applyBulkInternal} falls back to the original
     * {@code execute()}+writeback path, byte-for-byte unchanged.
     */
    interface DirectOutputCommand {

        void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset);
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

    record ConstCommand(double value, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            DoubleVector v = DoubleVector.broadcast(SPECIES, value);
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                v.intoArray(output, outputOffset + k);
            }
            for (; k < n; k++) {
                output[outputOffset + k] = value;
            }
        }
    }

    record LoadCommand(int slotIdx, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            if (ctx.flatVariables != null) {
                int srcOff = (slotIdx * ctx.dataSize) + ctx.blockStart;
                System.arraycopy(ctx.flatVariables, srcOff, ctx.scratch, destOff, n);
            } else {
                System.arraycopy(ctx._2DVariables[slotIdx], ctx.blockStart, ctx.scratch, destOff, n);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            if (ctx.flatVariables != null) {
                int srcOff = (slotIdx * ctx.dataSize) + ctx.blockStart;
                System.arraycopy(ctx.flatVariables, srcOff, output, outputOffset, n);
            } else {
                System.arraycopy(ctx._2DVariables[slotIdx], ctx.blockStart, output, outputOffset, n);
            }
        }
    }

// --- Core Binary Operations ---
    record AddCommand(int lOff, int rOff, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, lOff + k)
                        .add(DoubleVector.fromArray(SPECIES, s, rOff + k))
                        .intoArray(output, outputOffset + k);
            }
            for (; k < n; k++) {
                output[outputOffset + k] = s[lOff + k] + s[rOff + k];
            }
        }
    }

    record SubCommand(int lOff, int rOff, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, lOff + k)
                        .sub(DoubleVector.fromArray(SPECIES, s, rOff + k))
                        .intoArray(output, outputOffset + k);
            }
            for (; k < n; k++) {
                output[outputOffset + k] = s[lOff + k] - s[rOff + k];
            }
        }
    }

    record MulCommand(int lOff, int rOff, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, lOff + k)
                        .mul(DoubleVector.fromArray(SPECIES, s, rOff + k))
                        .intoArray(output, outputOffset + k);
            }
            for (; k < n; k++) {
                output[outputOffset + k] = s[lOff + k] * s[rOff + k];
            }
        }
    }

    record DivCommand(int lOff, int rOff, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, lOff + k)
                        .div(DoubleVector.fromArray(SPECIES, s, rOff + k))
                        .intoArray(output, outputOffset + k);
            }
            for (; k < n; k++) {
                output[outputOffset + k] = s[lOff + k] / s[rOff + k];
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
    record LoadLoadAddCommand(int lSlot, int rSlot, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, lBase + k)
                            .add(DoubleVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flat[lBase + k] + flat[rBase + k];
                }
            } else {
                double[] l = ctx._2DVariables[lSlot];
                double[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, l, base + k)
                            .add(DoubleVector.fromArray(SPECIES, r, base + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = l[base + k] + r[base + k];
                }
            }
        }
    }

    record LoadLoadSubCommand(int lSlot, int rSlot, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, lBase + k)
                            .sub(DoubleVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flat[lBase + k] - flat[rBase + k];
                }
            } else {
                double[] l = ctx._2DVariables[lSlot];
                double[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, l, base + k)
                            .sub(DoubleVector.fromArray(SPECIES, r, base + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = l[base + k] - r[base + k];
                }
            }
        }
    }

    record LoadLoadMulCommand(int lSlot, int rSlot, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, lBase + k)
                            .mul(DoubleVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flat[lBase + k] * flat[rBase + k];
                }
            } else {
                double[] l = ctx._2DVariables[lSlot];
                double[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, l, base + k)
                            .mul(DoubleVector.fromArray(SPECIES, r, base + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = l[base + k] * r[base + k];
                }
            }
        }
    }

    record LoadLoadDivCommand(int lSlot, int rSlot, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, lBase + k)
                            .div(DoubleVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flat[lBase + k] / flat[rBase + k];
                }
            } else {
                double[] l = ctx._2DVariables[lSlot];
                double[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, l, base + k)
                            .div(DoubleVector.fromArray(SPECIES, r, base + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = l[base + k] / r[base + k];
                }
            }
        }
    }

    // --- Fused Scale (const*var) and Scale-Accumulate (axpy chain) ---
    // Together these collapse a linear-combination-shaped expression --
    // a1*x1 + a2*x2 + ... + an*xn, or any mix of + / - between such terms --
    // from 4N-1 commands (with no fusion at all: a separate ConstCommand,
    // LoadCommand, MulCommand per term, plus an AddCommand/SubCommand
    // chaining each term into a running total, each command a full pass
    // through ctx.scratch) down to exactly N commands: the first term
    // becomes one ScaleCommand seeding the accumulator, and every
    // subsequent term becomes one ScaleAccumulateCommand -- a single SIMD
    // pass per term, each doing exactly one hardware fused-multiply-add per
    // lane. See tryFuseConstLoad() and tryFuseScaleAccumulate() for the
    // peephole rules that emit these; they compose without either knowing
    // about the other -- tryFuseConstLoad only ever looks at the immediate
    // ConstCommand/LoadCommand pair beneath an OP_MUL, and
    // tryFuseScaleAccumulate only ever looks at the ScaleCommand beneath an
    // OP_ADD/OP_SUB -- so an arbitrarily long chain of terms folds correctly
    // without any whole-expression pattern matching.
    record ScaleCommand(double coeff, int varSlot, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            DoubleVector coeffVec = DoubleVector.broadcast(SPECIES, coeff);
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int base = (varSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, base + k)
                            .mul(coeffVec)
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = coeff * flat[base + k];
                }
            } else {
                double[] v = ctx._2DVariables[varSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, v, base + k)
                            .mul(coeffVec)
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = coeff * v[base + k];
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            DoubleVector coeffVec = DoubleVector.broadcast(SPECIES, coeff);
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int base = (varSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, base + k)
                            .mul(coeffVec)
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = coeff * flat[base + k];
                }
            } else {
                double[] v = ctx._2DVariables[varSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, v, base + k)
                            .mul(coeffVec)
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = coeff * v[base + k];
                }
            }
        }
    }

    // acc[k] = acc[k] + coeff*var[k] (or acc - coeff*var, folded into a
    // negated coeff at fusion time -- see tryFuseScaleAccumulate), computed
    // as a single hardware fused-multiply-add per SIMD lane, in place on
    // scratch at accOff. Numerically this is strictly at least as accurate
    // as the unfused mul-then-add: one IEEE-754 rounding instead of two.
    record ScaleAccumulateCommand(double coeff, int varSlot, int accOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            DoubleVector coeffVec = DoubleVector.broadcast(SPECIES, coeff);
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int base = (varSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, base + k)
                            .fma(coeffVec, DoubleVector.fromArray(SPECIES, s, accOff + k))
                            .intoArray(s, accOff + k);
                }
                for (; k < n; k++) {
                    s[accOff + k] = Math.fma(flat[base + k], coeff, s[accOff + k]);
                }
            } else {
                double[] v = ctx._2DVariables[varSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, v, base + k)
                            .fma(coeffVec, DoubleVector.fromArray(SPECIES, s, accOff + k))
                            .intoArray(s, accOff + k);
                }
                for (; k < n; k++) {
                    s[accOff + k] = Math.fma(v[base + k], coeff, s[accOff + k]);
                }
            }
        }

        // Note: the accumulator itself is still read from ctx.scratch here --
        // it's a running value built up by prior commands, not a source --
        // only the FINAL fused-multiply-add result is written to `output`
        // instead of back into scratch. This is exactly the "accumulator +
        // coeff*var" shape ScaleAccumulateCommand's ordinary execute() computes,
        // just landing its one write in a different place.
        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            double[] s = ctx.scratch;
            DoubleVector coeffVec = DoubleVector.broadcast(SPECIES, coeff);
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int base = (varSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, base + k)
                            .fma(coeffVec, DoubleVector.fromArray(SPECIES, s, accOff + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = Math.fma(flat[base + k], coeff, s[accOff + k]);
                }
            } else {
                double[] v = ctx._2DVariables[varSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, v, base + k)
                            .fma(coeffVec, DoubleVector.fromArray(SPECIES, s, accOff + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = Math.fma(v[base + k], coeff, s[accOff + k]);
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

    record RemCommand(int lOff, int rOff, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                s[destOff + k] = s[lOff + k] % s[rOff + k];
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            double[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                output[outputOffset + k] = s[lOff + k] % s[rOff + k];
            }
        }
    }

// --- Comparisons ---
    record CompareCommand(int lOff, int rOff, int destOff, int opcode) implements VectorCommand, DirectOutputCommand {

        // Computes s[lOff+k..] OP s[rOff+k..] as a VectorMask, per the same
        // truthiness rules compareScalar() below applies to the tail. Shared
        // by execute() and executeToOutput() so the opcode -> comparison
        // mapping exists in exactly one place instead of being duplicated
        // across two switch statements (as it was before this vectorization).
        // OP_AND/OP_OR are built from two NE-vs-zero masks combined with
        // mask.and()/mask.or() -- the vectorized form of the same C-style
        // "nonzero is true" rule the scalar path already used.
        private VectorMask<Double> compareMask(double[] s, int k) {
            DoubleVector lv = DoubleVector.fromArray(SPECIES, s, lOff + k);
            DoubleVector rv = DoubleVector.fromArray(SPECIES, s, rOff + k);
            return switch (opcode) {
                case OP_GT ->
                    lv.compare(VectorOperators.GT, rv);
                case OP_LT ->
                    lv.compare(VectorOperators.LT, rv);
                case OP_EQ ->
                    lv.compare(VectorOperators.EQ, rv);
                case OP_NE ->
                    lv.compare(VectorOperators.NE, rv);
                case OP_GE ->
                    lv.compare(VectorOperators.GE, rv);
                case OP_LE ->
                    lv.compare(VectorOperators.LE, rv);
                case OP_AND ->
                    lv.compare(VectorOperators.NE, 0.0).and(rv.compare(VectorOperators.NE, 0.0));
                case OP_OR ->
                    lv.compare(VectorOperators.NE, 0.0).or(rv.compare(VectorOperators.NE, 0.0));
                default ->
                    throw new IllegalArgumentException("Unknown comparison opcode: " + opcode);
            };
        }

        // Scalar ground truth for the tail -- must stay in lockstep with
        // compareMask()'s per-lane semantics above.
        private static boolean compareScalar(int opcode, double l, double r) {
            return switch (opcode) {
                case OP_GT ->
                    l > r;
                case OP_LT ->
                    l < r;
                case OP_EQ ->
                    l == r;
                case OP_NE ->
                    l != r;
                case OP_GE ->
                    l >= r;
                case OP_LE ->
                    l <= r;
                case OP_AND ->
                    l != 0.0 && r != 0.0;
                case OP_OR ->
                    l != 0.0 || r != 0.0;
                default ->
                    throw new IllegalArgumentException("Unknown comparison opcode: " + opcode);
            };
        }

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                VectorMask<Double> mask = compareMask(s, k);
                DoubleVector.zero(SPECIES).blend(1.0, mask).intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = compareScalar(opcode, s[lOff + k], s[rOff + k]) ? 1.0 : 0.0;
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                VectorMask<Double> mask = compareMask(s, k);
                DoubleVector.zero(SPECIES).blend(1.0, mask).intoArray(output, outputOffset + k);
            }
            for (; k < n; k++) {
                output[outputOffset + k] = compareScalar(opcode, s[lOff + k], s[rOff + k]) ? 1.0 : 0.0;
            }
        }
    }

// --- Ternary / Branching ---
    record VmaCommand(int aOff, int bOff, int cOff, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            double[] s = ctx.scratch;
            int k = 0, bound = SPECIES.loopBound(n);
            for (; k < bound; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, aOff + k)
                        .fma(DoubleVector.fromArray(SPECIES, s, bOff + k),
                                DoubleVector.fromArray(SPECIES, s, cOff + k))
                        .intoArray(output, outputOffset + k);
            }
            if (k < n) {
                var mask = SPECIES.indexInRange(k, n);
                DoubleVector.fromArray(SPECIES, s, aOff + k, mask)
                        .fma(DoubleVector.fromArray(SPECIES, s, bOff + k, mask),
                                DoubleVector.fromArray(SPECIES, s, cOff + k, mask))
                        .intoArray(output, outputOffset + k, mask);
            }
        }
    }

    record IfCommand(int condOff, int trueOff, int falseOff, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                s[destOff + k] = (s[condOff + k] != 0.0) ? s[trueOff + k] : s[falseOff + k];
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            double[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                output[outputOffset + k] = (s[condOff + k] != 0.0) ? s[trueOff + k] : s[falseOff + k];
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

    // --- Fused Load+UnaryMathOp ---
    // A unary math op applied directly to a bare variable load -- e.g.
    // sin(x), not sin(x+1) -- otherwise compiles to two separate commands:
    // a LoadCommand materializing x into ctx.scratch, then a
    // UnaryMathCommand reading that scratch range and transforming it in
    // place. LoadUnaryMathCommand collapses that into ONE VectorCommand:
    // it inlines the exact same source dispatch LoadCommand.execute()
    // performs (flatVariables / _2DVariables), then immediately calls the
    // existing UnaryMathOp on the freshly-copied range. This still
    // delegates the actual math to the same op table used everywhere else
    // -- no new per-op vectorized primitives, no risk of a hand-written
    // fused implementation drifting from VectorMath's own numerics -- so
    // what's eliminated is purely dispatch: one fewer entry in
    // executionPlan, one fewer pass through the interpreter's outer
    // (inherently megamorphic, since it iterates over every VectorCommand
    // subtype) dispatch loop.
    //
    // This generic form does NOT eliminate the copy-into-scratch pass
    // itself -- op.apply still reads and writes that same scratch range,
    // so the operand is still touched twice in memory (once to copy it in,
    // once to transform it). For ops where the per-element compute cost
    // already dominates (sin, cos, tan, exp, ln, ...: many instructions for
    // range reduction + polynomial evaluation), that's a rounding error and
    // this generic fusion captures effectively all of the available win.
    // For ops cheap enough that the extra pass is a real fraction of total
    // cost -- one hardware instruction, like sqrt -- a fully dedicated
    // command that never materializes at all recovers strictly more; see
    // LoadSqrtCommand immediately below for that treatment, and
    // tryFuseLoadUnary() in compile() for how OP_SQRT is special-cased to
    // prefer it over this generic path.
    record LoadUnaryMathCommand(UnaryMathOp op, int varSlot, int destOff) implements VectorCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            if (ctx.flatVariables != null) {
                int srcOff = (varSlot * ctx.dataSize) + ctx.blockStart;
                System.arraycopy(ctx.flatVariables, srcOff, ctx.scratch, destOff, n);
            } else {
                System.arraycopy(ctx._2DVariables[varSlot], ctx.blockStart, ctx.scratch, destOff, n);
            }
            op.apply(destOff, n, ctx.scratch);
        }
    }

    // --- Fully-fused Load+Sqrt ---
    // Unlike LoadUnaryMathCommand, this never materializes the operand into
    // scratch at all: it reads straight from the source array, computes
    // sqrt via the hardware-mapped VectorOperators.SQRT in the SAME SIMD
    // pass, and writes the result straight to ctx.scratch. sqrt is cheap
    // enough (one SQRTPD instruction per lane) that the extra
    // read-then-transform pass LoadUnaryMathCommand still pays is a real
    // cost relative to the operation itself -- this is the dedicated escape
    // hatch for that case. No VectorMath dependency at all;
    // VectorOperators.SQRT is used exactly as VectorMath's own sqrt()
    // implementation uses it, so results are bit-identical to the unfused
    // path, just computed in one pass instead of two.
    record LoadSqrtCommand(int varSlot, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            double[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int base = (varSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, base + k)
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = Math.sqrt(flat[base + k]);
                }
            } else {
                double[] v = ctx._2DVariables[varSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, v, base + k)
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = Math.sqrt(v[base + k]);
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, double[] output, int outputOffset) {
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                double[] flat = ctx.flatVariables;
                int base = (varSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, flat, base + k)
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = Math.sqrt(flat[base + k]);
                }
            } else {
                double[] v = ctx._2DVariables[varSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    DoubleVector.fromArray(SPECIES, v, base + k)
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = Math.sqrt(v[base + k]);
                }
            }
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

                    // Peephole fusions, tried in order of specificity. Each
                    // helper is self-contained: it inspects the tail of `plan`,
                    // removes whatever entries it consumes on success, and
                    // returns null (leaving `plan` untouched) on no match -- see
                    // each method's own javadoc for exactly what shape it looks
                    // for and why removal is always safe (strict LIFO stack
                    // machine: once an operand's stack slot is popped here,
                    // nothing else in the program can reference it again).
                    VectorCommand fused = tryFuseLoadLoad(plan, opcode, lOff, rOff, destOff);
                    if (fused == null && opcode == OP_MUL) {
                        // const*var or var*const, e.g. the a_i*x_i term of a
                        // linear combination -- see ScaleCommand.
                        fused = tryFuseConstLoad(plan, lOff, rOff, destOff);
                    }
                    if (fused == null && (opcode == OP_ADD || opcode == OP_SUB)) {
                        // accumulator +/- (a_i*x_i term just computed above) --
                        // see ScaleAccumulateCommand. Composes with the
                        // tryFuseConstLoad case above to fold an entire
                        // a1*x1 + a2*x2 + ... + an*xn chain into N commands.
                        fused = tryFuseScaleAccumulate(plan, opcode, lOff, rOff, destOff);
                    }
                    if (fused == null && (opcode == OP_ADD || opcode == OP_SUB)) {
                        // accumulator +/- (bare variable), e.g. the x3 term of
                        // x1+x2+x3 -- an unweighted sum is just the coeff=1.0
                        // special case of the axpy chain above, so it reuses
                        // the exact same ScaleAccumulateCommand rather than a
                        // new command class. Tried after tryFuseScaleAccumulate
                        // since the two match mutually exclusive shapes (one
                        // needs a ScaleCommand on top of the plan, this one
                        // needs a bare LoadCommand) and tryFuseLoadLoad above
                        // already claims the case where BOTH operands are bare
                        // loads, so this only ever fires for the "accumulator
                        // so far, plus one more plain variable" shape.
                        fused = tryFuseLoadAccumulate(plan, opcode, lOff, rOff, destOff);
                    }

                    if (fused != null) {
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

                    // Peephole: this unary op is applied directly to a bare
                    // variable load (sin(x), not sin(x+1)) -- fuse the pending
                    // LoadCommand into the unary op itself. See
                    // tryFuseLoadUnary()'s javadoc for the generic-vs-dedicated
                    // (LoadSqrtCommand) distinction.
                    VectorCommand loadFused = tryFuseLoadUnary(plan, opcode, baseOff);
                    if (loadFused != null) {
                        plan.add(loadFused);
                    } else {
                        plan.add(new UnaryMathCommand(resolveUnaryMathOp(opcode), baseOff));
                    }
                }
            }
        }

        return new SIMDVectorCompositeExpression(plan.toArray(new VectorCommand[0]), stackDepth, BLOCK_SIZE);
    }

    /**
     * The opcode -&gt; {@link UnaryMathOp} mapping, factored out of the
     * {@code compile()} switch so both the ordinary (unfused) unary path and
     * {@link #tryFuseLoadUnary} can share one source of truth instead of two
     * copies of the same ~50-case table drifting apart over time.
     */
    private static UnaryMathOp resolveUnaryMathOp(int opcode) {
        return switch (opcode) {

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
    }

    /**
     * Peephole fusion: when both operands of a binary arithmetic op are plain
     * variable loads - i.e. the last two entries in the plan are LoadCommands
     * feeding directly into this op and nothing else - collapse them into a
     * single fused command that reads straight from the source arrays instead
     * of round-tripping both operands through ctx.scratch. Returns null (no
     * fusion) for anything that doesn't match that exact shape, including
     * OP_REM (not vectorized regardless) and OP_POW (routes through
     * VectorMath.executePowerBlended, not a plain lane op).
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
    
     /**
     * Peephole fusion for unweighted running sums/differences -
     * {@code x1 + x2 + x3 + ... + xn}, or any mix of {@code +}/{@code -}
     * between bare variables, with no coefficients anywhere. This is the
     * {@code coeff = 1.0} (or {@code -1.0}, for {@code OP_SUB}) special case
     * of {@link #tryFuseScaleAccumulate} - reusing the exact same
     * {@link ScaleAccumulateCommand} (one hardware fused-multiply-add per
     * SIMD lane, {@code acc = acc + 1.0*var}) rather than a separate
     * plain-accumulate command class - for when the right-hand operand of an
     * {@code OP_ADD}/{@code OP_SUB} is a bare {@link LoadCommand} rather
     * than a {@link ScaleCommand}.
     *
     * <p>Tried after {@link #tryFuseScaleAccumulate}, since the two match
     * mutually exclusive shapes (that one needs a {@code ScaleCommand} on
     * top of the plan; this one needs a bare {@code LoadCommand}), and after
     * {@link #tryFuseLoadLoad} already has first claim on the case where
     * BOTH operands are bare loads (e.g. the {@code x1+x2} seed of
     * {@code x1+x2+x3}) - so this only ever fires for "the accumulator so
     * far, plus one more plain variable", which is exactly the shape every
     * term after the first takes in an unweighted sum.
     *
     * <p>Composes with {@link #tryFuseLoadLoad} the same way
     * {@link #tryFuseConstLoad}/{@link #tryFuseScaleAccumulate} compose for
     * a weighted chain: an N-term unweighted sum collapses from
     * {@code 2N-3} commands (a standalone {@code LoadCommand} plus a plain
     * {@code AddCommand}/{@code SubCommand} for every term after the first
     * two) down to {@code N-1} (one {@code LoadLoadAddCommand} seeding the
     * accumulator from the first two terms, then one
     * {@code ScaleAccumulateCommand} per remaining term).
     */
    private static VectorCommand tryFuseLoadAccumulate(List<VectorCommand> plan, int opcode, int lOff, int rOff, int destOff) {
        int size = plan.size();
        if (size < 1) {
            return null;
        }
        if (!(plan.get(size - 1) instanceof LoadCommand load) || load.destOff() != rOff) {
            return null;
        }

        double coeff = (opcode == OP_SUB) ? -1.0 : 1.0;
        plan.remove(size - 1);
        return new ScaleAccumulateCommand(coeff, load.slotIdx(), destOff);
    }
    
      /**
     * Peephole fusion for linear-combination-shaped expressions
     * ({@code a1*x1 + a2*x2 + ... + an*xn}, and the equivalent with any mix
     * of {@code +}/{@code -} between terms): when the right-hand operand of
     * an {@code OP_ADD}/{@code OP_SUB} is a {@link ScaleCommand} that was
     * JUST emitted - the last entry in the plan, nothing has consumed it yet
     * - collapse the pair into a single {@link ScaleAccumulateCommand}: one
     * hardware fused-multiply-add per SIMD lane
     * ({@code acc = acc + coeff*var}, or {@code acc = acc - coeff*var} via a
     * negated coefficient for {@code OP_SUB}), reading the variable straight
     * from its source.
     *
     * <p>Unlike {@link #tryFuseLoadLoad}/{@link #tryFuseConstLoad}, only ONE
     * plan entry is ever removed here - the left-hand (accumulator) operand
     * is never a single fresh command to delete, it's whatever arbitrary
     * chain of prior commands already left its value at {@code lOff} in
     * scratch (itself possibly a previous {@code ScaleAccumulateCommand}),
     * and that chain is left completely untouched. {@code destOff} is
     * {@code lOff} by the caller's existing "reuse the left slot" convention
     * for {@code OP_ADD}/{@code OP_SUB}, so the accumulator is written back
     * into exactly the slot it already occupies.
     *
     * <p>Composing this with {@link #tryFuseConstLoad} is what collapses an
     * entire N-term linear combination into exactly N commands: the first
     * term becomes one {@code ScaleCommand} (seeding the accumulator), and
     * every subsequent term becomes one {@code ScaleAccumulateCommand} -
     * versus {@code 4N-1} commands (multiple full materialization passes per
     * term) with no fusion at all.
     */
    private static VectorCommand tryFuseScaleAccumulate(List<VectorCommand> plan, int opcode, int lOff, int rOff, int destOff) {
        int size = plan.size();
        if (size < 1) {
            return null;
        }
        if (!(plan.get(size - 1) instanceof ScaleCommand scale) || scale.destOff() != rOff) {
            return null;
        }

        double coeff = (opcode == OP_SUB) ? -scale.coeff() : scale.coeff();
        plan.remove(size - 1);
        return new ScaleAccumulateCommand(coeff, scale.varSlot(), destOff);
    }
    /**
     * Peephole fusion: when one operand of an {@code OP_MUL} is a plain
     * numeric constant and the other is a plain variable load - the last two
     * plan entries are exactly a {@link ConstCommand} and a
     * {@link LoadCommand} feeding this multiplication, in either order
     * (multiplication is commutative, so {@code a*x} and {@code x*a} both
     * match) - collapse them into a single {@link ScaleCommand} that reads
     * the variable straight from its source and multiplies by the constant
     * in one pass. This is the building block
     * {@link #tryFuseScaleAccumulate} depends on: {@code a1*x1} compiles to
     * one {@code ScaleCommand} instead of three separate commands
     * ({@code ConstCommand}, {@code LoadCommand}, {@code MulCommand}).
     *
     * <p>Only fires for {@code OP_MUL} - the caller is responsible for not
     * calling this for other opcodes, since e.g. {@code a/x} and
     * {@code x/a} are not interchangeable.
     */
    private static VectorCommand tryFuseConstLoad(List<VectorCommand> plan, int lOff, int rOff, int destOff) {
        int size = plan.size();
        if (size < 2) {
            return null;
        }

        VectorCommand last = plan.get(size - 1);
        VectorCommand secondLast = plan.get(size - 2);

        ConstCommand constCmd;
        LoadCommand loadCmd;
        if (last instanceof LoadCommand l && l.destOff() == rOff
                && secondLast instanceof ConstCommand c && c.destOff() == lOff) {
            constCmd = c;
            loadCmd = l;
        } else if (last instanceof ConstCommand c && c.destOff() == rOff
                && secondLast instanceof LoadCommand l && l.destOff() == lOff) {
            constCmd = c;
            loadCmd = l;
        } else {
            return null;
        }

        plan.remove(size - 1);
        plan.remove(size - 2);
        return new ScaleCommand(constCmd.value(), loadCmd.slotIdx(), destOff);
    }

    /**
     * Peephole fusion for any unary math opcode applied directly to a bare
     * variable load: if the last plan entry is exactly the
     * {@link LoadCommand} that produced {@code baseOff}, fuse the pair into
     * a single command that reads the variable straight from its source
     * instead of paying for a separate materialization pass first.
     *
     * <p>{@code OP_SQRT} gets the fully-dedicated {@link LoadSqrtCommand} -
     * no separate materialization pass at all, see that class's javadoc for
     * why sqrt specifically earns the hand-written treatment. Every other
     * unary opcode gets the generic {@link LoadUnaryMathCommand}, which
     * still delegates the actual math to the shared {@link #resolveUnaryMathOp}
     * table but skips the extra {@code VectorCommand} dispatch a standalone
     * {@code LoadCommand} would otherwise cost.
     *
     * <p>Returns {@code null} (no fusion) when the operand isn't a bare load
     * - e.g. {@code sqrt(x+1)}, where the operand is the result of a prior
     * {@code ADD}, not a {@code LoadCommand} - in which case the ordinary
     * {@link UnaryMathCommand} path handles it exactly as before.
     */
    private static VectorCommand tryFuseLoadUnary(List<VectorCommand> plan, int opcode, int baseOff) {
        int size = plan.size();
        if (size < 1) {
            return null;
        }
        if (!(plan.get(size - 1) instanceof LoadCommand load) || load.destOff() != baseOff) {
            return null;
        }

        plan.remove(size - 1);
        if (opcode == OP_SQRT) {
            return new LoadSqrtCommand(load.slotIdx(), baseOff);
        }
        return new LoadUnaryMathCommand(resolveUnaryMathOp(opcode), load.slotIdx(), baseOff);
    }

    public final class SIMDVectorCompositeExpression extends BatchedVectorCompositeExpression implements AutoCloseable {

        private static final Cleaner SYSTEM_CLEANER = Cleaner.create();

        // Bounded spin budget before parking, shared by the worker dispatch
        // wait and the master completion wait. Keeps best-case wake latency
        // in the tens-of-nanoseconds range instead of an OS park/unpark
        // round trip on every call, while still falling back to park() so a
        // waiter never burns a core forever if something stalls.
        private static final int SPIN_LIMIT = 2000;

        private final int NUM_WORKERS;
        private final WorkerThread[] workerPool;
        private final VectorCommand[] executionPlan;
        private final ThreadLocal<EvaluationContext> masterEvalContext;
        // Reusable per-caller-thread scratch buffer for computeChunkLengths(),
        // sized NUM_WORKERS + 1. Kept per-thread (not a single shared array)
        // because this class supports concurrent calls to applyBulkParallel
        // from multiple external threads on the same instance — a shared
        // buffer would let two callers stomp on each other's chunk math.
        private final ThreadLocal<int[]> chunkLengthsScratch;
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
                final int totalSlices = NUM_WORKERS + 1;
                this.chunkLengthsScratch = ThreadLocal.withInitial(() -> new int[totalSlices]);

                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i] = new WorkerThread(i, executionPlan, stackDepth, blockSize);
                }

                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i].start();
                }

                this.cleanable = SYSTEM_CLEANER.register(this, new ThreadPoolShutdownAction(workerPool));
            } else {
                this.workerPool = null;
                this.chunkLengthsScratch = null;
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
            if (chunkLengthsScratch != null) {
                chunkLengthsScratch.remove();
            }
        }

        /**
         * Splits {@code numSamples} into {@code totalSlices} chunks, each an
         * exact multiple of the SIMD lane width except for the very last
         * slice, which also absorbs whatever scalar remainder is left over.
         * The last slice is always handed to the calling (master) thread
         * (see the {@code applyBulkParallel} overloads below), so every
         * background worker gets a perfectly vector-aligned chunk with zero
         * scalar tail, and no slice carries more than one extra lane-group
         * versus its neighbours.
         *
         * Writes into the calling thread's {@link #chunkLengthsScratch}
         * buffer rather than allocating, so repeated calls from the same
         * thread cost zero garbage.
         */
        private int[] computeChunkLengths(int numSamples, int totalSlices) {
            final int vlen = SPECIES.length();
            final int units = numSamples / vlen;
            final int scalarRemainder = numSamples - (units * vlen);
            final int baseUnits = units / totalSlices;
            final int extraUnits = units % totalSlices;

            int[] lengths = chunkLengthsScratch.get();
            for (int i = 0; i < totalSlices; i++) {
                int u = baseUnits + (i < extraUnits ? 1 : 0);
                lengths[i] = u * vlen;
            }
            lengths[totalSlices - 1] += scalarRemainder;
            return lengths;
        }

        /**
         * Waits for the first {@code used} workers in {@link #workerPool} to
         * finish. Each worker only ever writes its own {@code done} flag, so
         * unlike a shared decrementing latch this generates no cache-line
         * contention between workers as they finish at roughly the same
         * time. Spins briefly before parking to avoid paying OS wake latency
         * in the common case where the wait is short.
         */
        private void awaitWorkers(int used) {
            for (int i = 0; i < used; i++) {
                WorkerThread w = workerPool[i];
                int spins = 0;
                while (!w.done) {
                    if (spins < SPIN_LIMIT) {
                        Thread.onSpinWait();
                        spins++;
                    } else {
                        LockSupport.park();
                    }
                }
            }
        }

        private static final class WorkerThread extends Thread {

            // Manual cache-line padding around the hot per-worker dispatch
            // state below. Java gives no field-layout guarantee, but keeping
            // this state clustered and padded discourages the JVM/hardware
            // from letting one worker's dispatch writes false-share a line
            // with a neighbouring WorkerThread's, without depending on
            // JDK-internal @Contended (which needs module opens we can't
            // assume the embedding application has granted).
            private long p0, p1, p2, p3, p4, p5, p6, p7;

            private final int workerId;
            private final EvaluationContext evalContext;
            private final VectorCommand[] executionPlan;
            private final int blockSize;

            private volatile boolean isRunning = true;
            private volatile int taskState = 0;
            // Written only by this worker; polled by the master in
            // awaitWorkers(). Deliberately not shared/aggregated so workers
            // never contend with each other while reporting completion.
            private volatile boolean done = true;
            private volatile Thread masterThread;

            private double[][] vars2D;
            private double[] vars1D;
            private double[] output;
            private int dataSize;
            private int startIdx;
            private int length;

            private long q0, q1, q2, q3, q4, q5, q6, q7;

            public WorkerThread(int workerId, VectorCommand[] executionPlan, int stackDepth, int blockSize) {
                this.workerId = workerId;
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
                this.done = false;
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
                this.done = false;
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
                    int spins = 0;
                    while (taskState == 0 && isRunning) {
                        if (spins < SPIN_LIMIT) {
                            Thread.onSpinWait();
                            spins++;
                        } else {
                            LockSupport.park();
                            if (Thread.interrupted()) {
                                return;
                            }
                        }
                    }
                    if (!isRunning) {
                        return;
                    }

                    // try/finally: guarantees `done` is always raised and the
                    // master always unparked, even if a bad expression or
                    // malformed input throws mid-block. Without this a single
                    // faulting task would leave the master parked forever.
                    try {
                        if (vars2D != null) {
                            applyBulkInternal(vars2D, evalContext, executionPlan, blockSize, dataSize, output, startIdx, length);
                        } else if (vars1D != null) {
                            applyBulkInternal(vars1D, evalContext, executionPlan, blockSize, dataSize, output, startIdx, length);
                        }
                    } finally {
                        this.taskState = 0;
                        this.vars2D = null;
                        this.vars1D = null;
                        this.output = null;
                        this.done = true;
                        Thread master = this.masterThread;
                        if (master != null) {
                            LockSupport.unpark(master);
                        }
                    }
                }
            }
        }

        public void validate(double[][] variables, double[] output) {
            // 1. Fail fast, avoid String.format unless throwing
            if (variables == null || output == null) {
                throw new IllegalArgumentException("Null input");
            }

            // 2. Cache values to local variables to avoid multiple array lookups
            final int varLen = variables.length;
            final int outLen = output.length;
            int stride = getVarCount();
            if (varLen != stride) {
                throw new IllegalArgumentException("Stride mismatch");
            }

            // 3. Optional: Only check inner length if you really need absolute safety
            // Only perform this if the performance impact of O(varCount) is acceptable.
            for (int i = 0; i < varLen; i++) {
                if (variables[i] == null || variables[i].length < outLen) {
                    throw new IllegalArgumentException("Jagged array or size mismatch");
                }
            }
        }

        public void validate(double[] flatVariables, double[] output) {
            int totalSamples = flatVariables != null && flatVariables.length > 0 && output != null && output.length > 0 ? flatVariables.length : -1;
            int stride = getVarCount();
            if (totalSamples != stride * output.length) {
                throw new IllegalStateException(String.format("array sizes not correct[totalSamples=%d vs computed(var-count*output-array-size)=%d]",
                        totalSamples, stride * output.length));
            }
        }

        public void validate(float[][] variables, double[] output) {
            throw new InputMismatchException("float[][] not supported only double[] and double[][]");
        }

        public void validate(float[] flatVariables, float[] output) {
            throw new InputMismatchException("float[][] not supported only double[] and double[][]");
        }

        @Override
        public void applyBulk(double[][] variables, double[] output) {
            if (varCount == 0) {
                fillOutput(SIMDCommandF64.this.constantAnswer, output);
                return;
            }
            int numSamples = variables[0].length;
            applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        @Override
        public void applyBulkParallel(double[][] variables, double[] output) {
            if (varCount == 0) {
                fillOutput(SIMDCommandF64.this.constantAnswer, output);
                return;
            }
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = variables[0].length;

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            // NUM_WORKERS background threads + the calling thread itself.
            // The master no longer sits idle while it waits: it takes the
            // last (vector-aligned-plus-remainder) slice and computes it
            // while the background workers are running.
            final int totalSlices = NUM_WORKERS + 1;
            final int[] lengths = computeChunkLengths(numSamples, totalSlices);
            final Thread masterThread = Thread.currentThread();

            int startIdx = 0;
            int used = 0;
            for (int i = 0; i < NUM_WORKERS; i++) {
                int length = lengths[i];
                if (length > 0) {
                    workerPool[i].submitTask2D(variables, output, numSamples, startIdx, length, masterThread);
                    used++;
                }
                startIdx += length;
            }

            int masterLength = lengths[totalSlices - 1];
            if (masterLength > 0) {
                applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, startIdx, masterLength);
            }

            awaitWorkers(used);
        }

        @Override
        public void applyBulkParallel(double[] flatVariables, double[] output) {
            if (varCount == 0) {
                fillOutput(SIMDCommandF64.this.constantAnswer, output);
                return;
            }
            if (flatVariables == null || output == null) {
                return;
            }
            int numSamples = output.length;

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            final int totalSlices = NUM_WORKERS + 1;
            final int[] lengths = computeChunkLengths(numSamples, totalSlices);
            final Thread masterThread = Thread.currentThread();

            int startIdx = 0;
            int used = 0;
            for (int i = 0; i < NUM_WORKERS; i++) {
                int length = lengths[i];
                if (length > 0) {
                    workerPool[i].submitTask1D(flatVariables, output, numSamples, startIdx, length, masterThread);
                    used++;
                }
                startIdx += length;
            }

            int masterLength = lengths[totalSlices - 1];
            if (masterLength > 0) {
                applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, startIdx, masterLength);
            }

            awaitWorkers(used);
        }

        @Override
        public void applyBulkBatched(double[][] variables, double[] output, int batchSize) {
            if (varCount == 0) {
                fillOutput(SIMDCommandF64.this.constantAnswer, output);
                return;
            }
            EvaluationContext ctx = masterEvalContext.get();
            int numSamples = variables[0].length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(variables, ctx, executionPlan, BLOCK_SIZE, numSamples, output, start, length);
            }
        }

        @Override
        public void applyBulk(double[] flatVariables, double[] output) {
            if (varCount == 0) {
                fillOutput(SIMDCommandF64.this.constantAnswer, output);
                return;
            }
            applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, output.length, output, 0, output.length);
        }

        @Override
        public void applyBulkBatched(double[] flatVariables, double[] output, int batchSize) {
            if (varCount == 0) {
                fillOutput(SIMDCommandF64.this.constantAnswer, output);
                return;
            }
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
            final int planLen = executionPlan.length;
            // If the last command in the plan can write its result straight to
            // `output` (see DirectOutputCommand), run every command before it
            // as usual but skip BOTH running the last one via the ordinary
            // execute() path AND the separate scratch-to-output writeback loop
            // below entirely -- one fewer full read+write pass over every
            // block. When the last command doesn't implement DirectOutputCommand
            // (PowCommand, or anything that delegates to VectorMath's
            // scratch-shaped UnaryMathOp/BinaryMathOp), `terminal` is null and
            // this falls back to the original path, byte-for-byte unchanged.
            // Resolved once outside the block loop since it's the same for
            // every block.
            final DirectOutputCommand terminal = (planLen > 0 && executionPlan[planLen - 1] instanceof DirectOutputCommand doc) ? doc : null;
            final int runLen = terminal != null ? planLen - 1 : planLen;

            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlock(flatVariables, null, dataSize, blockStart);

                for (int i = 0; i < runLen; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                if (terminal != null) {
                    terminal.executeToOutput(ctx, currentBlockSize, output, blockStart);
                } else {
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
        }

        private static void applyBulkInternal(double[][] variables, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, double[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            double[] s = ctx.scratch;
            final int planLen = executionPlan.length;
            // See the flatVariables overload above for the full explanation of
            // this DirectOutputCommand check -- identical logic here.
            final DirectOutputCommand terminal = (planLen > 0 && executionPlan[planLen - 1] instanceof DirectOutputCommand doc) ? doc : null;
            final int runLen = terminal != null ? planLen - 1 : planLen;

            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlock(null, variables, dataSize, blockStart);

                for (int i = 0; i < runLen; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                if (terminal != null) {
                    terminal.executeToOutput(ctx, currentBlockSize, output, blockStart);
                } else {
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

}