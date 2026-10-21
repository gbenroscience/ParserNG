package com.github.gbenroscience.simdext.turbo.tools.command.temp;

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
import java.util.InputMismatchException;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import jdk.incubator.vector.*;

/**
 * High-Performance MemorySegment(of floats) Vector API & Engine that fuses
 * explicit SIMD vectorization with a zero-allocation primitive stack
 * interpreter. Completely eliminates the scalar parser overhead and task object
 * allocations on the hot path.
 *
 * This version is the second fastest of all the SIMD evaluators. Combines near
 * zero-allocation with parallel operations greatly enhanced with cpu-pinning.
 * Cpu pinning is the reason why this class is a native of this extension and is
 * the main reason why this extension is JDK22+ Note that CPU PINNING works best
 * on Linux, so the worker efficiency of these classes is best seen on Linux.
 * Where 2 workers perform at almost 2x the rate of one worker.. usually between
 * 1.4x to 2.0x
 *
 *
 */
public class SIMDCommandSegmentF32 extends VectorTurboEvaluator {

    // NOTE: VectorConfig's statically-imported `SPECIES` constant is typed VectorSpecies<Double>
    // (it backs the F64 sibling of this class). It cannot be reused for a float32-only evaluator,
    // so this field shadows that import for every command/record in this file that references
    // the bare name `SPECIES`. This does not alter the parallelism architecture in any way -
    // it only supplies a species of the correct lane type.
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

    /**
     * A command that can write its final result for the current block straight
     * into the caller's output buffer, instead of into {@code ctx.scratch}.
     *
     * <p>
     * When the LAST command in a compiled plan implements this interface, the
     * {@code applyBulkInternal*} loops call {@link #executeToOutput} for it
     * instead of {@code execute(...)} followed by the separate scratch-to-output
     * writeback pass -- one fewer full read+write pass over every block. Every
     * command before the last one is unaffected; this only changes how the FINAL
     * result reaches the output. A last command that does not implement this
     * interface (PowCommand, UnaryMathCommand, BinaryMathCommand -- anything
     * that delegates to VectorMathF's scratch-shaped ops) takes the original
     * execute()+writeback path, byte-for-byte unchanged.
     *
     * <p>
     * There are two flavours because the plan can be driven with either a
     * {@link MemorySegment} output or a {@code float[]} output.
     */
    interface DirectOutputCommand {

        void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset);

        void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset);
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

    record ConstCommand(float value, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            FloatVector v = FloatVector.broadcast(SPECIES, value);
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                v.intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
            }
            for (; k < n; k++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, value);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            FloatVector v = FloatVector.broadcast(SPECIES, value);
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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            if (ctx.flatVariables != null) {
                int srcOff = (slotIdx * ctx.dataSize) + ctx.blockStart;
                MemorySegment.copy(ctx.flatVariables, srcOff, output, ValueLayout.JAVA_FLOAT, outputElemOffset * elemBytes, n);
            } else if (ctx._2DVariables != null) {
                MemorySegment.copy(ctx._2DVariables[slotIdx], ctx.blockStart, output, ValueLayout.JAVA_FLOAT, outputElemOffset * elemBytes, n);
            } else if (ctx.flatVariablesSeg != null) {
                long srcOff = ((long) slotIdx * ctx.dataSize) + ctx.blockStart;
                MemorySegment.copy(ctx.flatVariablesSeg, srcOff * elemBytes, output, outputElemOffset * elemBytes, (long) n * elemBytes);
            } else {
                MemorySegment.copy(ctx._2DVariablesSeg[slotIdx], (long) ctx.blockStart * elemBytes, output, outputElemOffset * elemBytes, (long) n * elemBytes);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            if (ctx.flatVariables != null) {
                int srcOff = (slotIdx * ctx.dataSize) + ctx.blockStart;
                System.arraycopy(ctx.flatVariables, srcOff, output, outputOffset, n);
            } else if (ctx._2DVariables != null) {
                System.arraycopy(ctx._2DVariables[slotIdx], ctx.blockStart, output, outputOffset, n);
            } else if (ctx.flatVariablesSeg != null) {
                long srcOff = ((long) slotIdx * ctx.dataSize) + ctx.blockStart;
                MemorySegment.copy(ctx.flatVariablesSeg, ValueLayout.JAVA_FLOAT, srcOff * elemBytes, output, outputOffset, n);
            } else {
                MemorySegment.copy(ctx._2DVariablesSeg[slotIdx], ValueLayout.JAVA_FLOAT, (long) ctx.blockStart * elemBytes, output, outputOffset, n);
            }
        }
    }

// --- Core Binary Operations ---
    record AddCommand(int lOff, int rOff, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            float[] s = ctx.scratch;
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .add(FloatVector.fromArray(SPECIES, s, rOff + k))
                        .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
            }
            for (; k < n; k++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, s[lOff + k] + s[rOff + k]);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .add(FloatVector.fromArray(SPECIES, s, rOff + k))
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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            float[] s = ctx.scratch;
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .sub(FloatVector.fromArray(SPECIES, s, rOff + k))
                        .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
            }
            for (; k < n; k++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, s[lOff + k] - s[rOff + k]);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .sub(FloatVector.fromArray(SPECIES, s, rOff + k))
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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            float[] s = ctx.scratch;
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .mul(FloatVector.fromArray(SPECIES, s, rOff + k))
                        .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
            }
            for (; k < n; k++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, s[lOff + k] * s[rOff + k]);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .mul(FloatVector.fromArray(SPECIES, s, rOff + k))
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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            float[] s = ctx.scratch;
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .div(FloatVector.fromArray(SPECIES, s, rOff + k))
                        .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
            }
            for (; k < n; k++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, s[lOff + k] / s[rOff + k]);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, lOff + k)
                        .div(FloatVector.fromArray(SPECIES, s, rOff + k))
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
    // straight from the source (flatVariables / _2DVariables / the
    // MemorySegment-backed equivalents) and skip that intermediate
    // materialization entirely. Selected at compile() time via
    // peephole-fusion over the emitted plan - see tryFuseLoadLoad().
    // Numerically identical to LoadCommand+LoadCommand+BinaryOp: same IEEE
    // op, same operand order, just a different source.
    record LoadLoadAddCommand(int lSlot, int rSlot, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .add(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flat[lBase + k] + flat[rBase + k];
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .add(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = l[base + k] + r[base + k];
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .add(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) + flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k);
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .add(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) + rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k);
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .add(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, flat[lBase + k] + flat[rBase + k]);
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .add(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, l[base + k] + r[base + k]);
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .add(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k,
                            flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) + flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k));
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .add(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k,
                            lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) + rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k));
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .add(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flat[lBase + k] + flat[rBase + k];
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .add(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = l[base + k] + r[base + k];
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .add(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) + flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k);
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .add(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) + rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k);
                }
            }
        }
    }

    record LoadLoadSubCommand(int lSlot, int rSlot, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .sub(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flat[lBase + k] - flat[rBase + k];
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .sub(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = l[base + k] - r[base + k];
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .sub(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) - flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k);
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .sub(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) - rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k);
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .sub(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, flat[lBase + k] - flat[rBase + k]);
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .sub(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, l[base + k] - r[base + k]);
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .sub(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k,
                            flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) - flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k));
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .sub(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k,
                            lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) - rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k));
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .sub(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flat[lBase + k] - flat[rBase + k];
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .sub(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = l[base + k] - r[base + k];
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .sub(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) - flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k);
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .sub(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) - rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k);
                }
            }
        }
    }

    record LoadLoadMulCommand(int lSlot, int rSlot, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .mul(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flat[lBase + k] * flat[rBase + k];
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .mul(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = l[base + k] * r[base + k];
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .mul(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) * flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k);
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .mul(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) * rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k);
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .mul(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, flat[lBase + k] * flat[rBase + k]);
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .mul(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, l[base + k] * r[base + k]);
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .mul(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k,
                            flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) * flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k));
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .mul(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k,
                            lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) * rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k));
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .mul(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flat[lBase + k] * flat[rBase + k];
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .mul(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = l[base + k] * r[base + k];
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .mul(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) * flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k);
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .mul(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) * rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k);
                }
            }
        }
    }

    record LoadLoadDivCommand(int lSlot, int rSlot, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .div(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flat[lBase + k] / flat[rBase + k];
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .div(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = l[base + k] / r[base + k];
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .div(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) / flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k);
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .div(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(s, destOff + k);
                }
                for (; k < n; k++) {
                    s[destOff + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) / rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k);
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .div(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, flat[lBase + k] / flat[rBase + k]);
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .div(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, l[base + k] / r[base + k]);
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .div(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k,
                            flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) / flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k));
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .div(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k,
                            lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) / rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k));
                }
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, limit = SPECIES.loopBound(n);
            if (ctx.flatVariables != null) {
                float[] flat = ctx.flatVariables;
                int lBase = (lSlot * ctx.dataSize) + ctx.blockStart;
                int rBase = (rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, flat, lBase + k)
                            .div(FloatVector.fromArray(SPECIES, flat, rBase + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flat[lBase + k] / flat[rBase + k];
                }
            } else if (ctx._2DVariables != null) {
                float[] l = ctx._2DVariables[lSlot];
                float[] r = ctx._2DVariables[rSlot];
                int base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, l, base + k)
                            .div(FloatVector.fromArray(SPECIES, r, base + k))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = l[base + k] / r[base + k];
                }
            } else if (ctx.flatVariablesSeg != null) {
                MemorySegment flatSeg = ctx.flatVariablesSeg;
                long lBase = ((long) lSlot * ctx.dataSize) + ctx.blockStart;
                long rBase = ((long) rSlot * ctx.dataSize) + ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, flatSeg, (lBase + k) * elemBytes, ByteOrder.nativeOrder())
                            .div(FloatVector.fromMemorySegment(SPECIES, flatSeg, (rBase + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, lBase + k) / flatSeg.getAtIndex(ValueLayout.JAVA_FLOAT, rBase + k);
                }
            } else {
                MemorySegment lSeg = ctx._2DVariablesSeg[lSlot];
                MemorySegment rSeg = ctx._2DVariablesSeg[rSlot];
                long base = ctx.blockStart;
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, lSeg, (base + k) * elemBytes, ByteOrder.nativeOrder())
                            .div(FloatVector.fromMemorySegment(SPECIES, rSeg, (base + k) * elemBytes, ByteOrder.nativeOrder()))
                            .intoArray(output, outputOffset + k);
                }
                for (; k < n; k++) {
                    output[outputOffset + k] = lSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) / rSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + k);
                }
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

    record RemCommand(int lOff, int rOff, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                s[destOff + k] = s[lOff + k] % s[rOff + k];
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            float[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, s[lOff + k] % s[rOff + k]);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            float[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                output[outputOffset + k] = s[lOff + k] % s[rOff + k];
            }
        }
    }

// --- Comparisons ---
    record CompareCommand(int lOff, int rOff, int destOff, int opcode) implements VectorCommand, DirectOutputCommand {

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
                        s[d + k] = (s[l + k] != 0.0 && s[r + k] != 0.0) ? 1.0f : 0.0f;
                    }
                }
                case OP_OR -> {
                    for (int k = 0; k < n; k++) {
                        s[d + k] = (s[l + k] != 0.0 || s[r + k] != 0.0) ? 1.0f : 0.0f;
                    }
                }
                default ->
                    throw new IllegalArgumentException("Unknown comparison opcode: " + opcode);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            float[] s = ctx.scratch;
            int l = lOff;
            int r = rOff;
            int d = destOff;

            switch (opcode) {
                case OP_GT -> {
                    for (int k = 0; k < n; k++) {
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, (s[l + k] > s[r + k]) ? 1.0f : 0.0f);
                    }
                }
                case OP_LT -> {
                    for (int k = 0; k < n; k++) {
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, (s[l + k] < s[r + k]) ? 1.0f : 0.0f);
                    }
                }
                case OP_EQ -> {
                    for (int k = 0; k < n; k++) {
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, (s[l + k] == s[r + k]) ? 1.0f : 0.0f);
                    }
                }
                case OP_NE -> {
                    for (int k = 0; k < n; k++) {
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, (s[l + k] != s[r + k]) ? 1.0f : 0.0f);
                    }
                }
                case OP_GE -> {
                    for (int k = 0; k < n; k++) {
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, (s[l + k] >= s[r + k]) ? 1.0f : 0.0f);
                    }
                }
                case OP_LE -> {
                    for (int k = 0; k < n; k++) {
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, (s[l + k] <= s[r + k]) ? 1.0f : 0.0f);
                    }
                }
                // Standard C-style floating-point truthiness: non-zero is true
                case OP_AND -> {
                    for (int k = 0; k < n; k++) {
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, (s[l + k] != 0.0 && s[r + k] != 0.0) ? 1.0f : 0.0f);
                    }
                }
                case OP_OR -> {
                    for (int k = 0; k < n; k++) {
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k, (s[l + k] != 0.0 || s[r + k] != 0.0) ? 1.0f : 0.0f);
                    }
                }
                default ->
                    throw new IllegalArgumentException("Unknown comparison opcode: " + opcode);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            float[] s = ctx.scratch;
            int l = lOff;
            int r = rOff;
            int d = destOff;

            switch (opcode) {
                case OP_GT -> {
                    for (int k = 0; k < n; k++) {
                        output[outputOffset + k] = (s[l + k] > s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_LT -> {
                    for (int k = 0; k < n; k++) {
                        output[outputOffset + k] = (s[l + k] < s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_EQ -> {
                    for (int k = 0; k < n; k++) {
                        output[outputOffset + k] = (s[l + k] == s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_NE -> {
                    for (int k = 0; k < n; k++) {
                        output[outputOffset + k] = (s[l + k] != s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_GE -> {
                    for (int k = 0; k < n; k++) {
                        output[outputOffset + k] = (s[l + k] >= s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                case OP_LE -> {
                    for (int k = 0; k < n; k++) {
                        output[outputOffset + k] = (s[l + k] <= s[r + k]) ? 1.0f : 0.0f;
                    }
                }
                // Standard C-style floating-point truthiness: non-zero is true
                case OP_AND -> {
                    for (int k = 0; k < n; k++) {
                        output[outputOffset + k] = (s[l + k] != 0.0 && s[r + k] != 0.0) ? 1.0f : 0.0f;
                    }
                }
                case OP_OR -> {
                    for (int k = 0; k < n; k++) {
                        output[outputOffset + k] = (s[l + k] != 0.0 || s[r + k] != 0.0) ? 1.0f : 0.0f;
                    }
                }
                default ->
                    throw new IllegalArgumentException("Unknown comparison opcode: " + opcode);
            }
        }
    }

// --- Ternary / Branching ---
    record VmaCommand(int aOff, int bOff, int cOff, int destOff) implements VectorCommand, DirectOutputCommand {

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

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            float[] s = ctx.scratch;
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int k = 0, bound = SPECIES.loopBound(n);
            for (; k < bound; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, aOff + k)
                        .fma(FloatVector.fromArray(SPECIES, s, bOff + k),
                                FloatVector.fromArray(SPECIES, s, cOff + k))
                        .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder());
            }
            if (k < n) {
                var mask = SPECIES.indexInRange(k, n);
                FloatVector.fromArray(SPECIES, s, aOff + k, mask)
                        .fma(FloatVector.fromArray(SPECIES, s, bOff + k, mask),
                                FloatVector.fromArray(SPECIES, s, cOff + k, mask))
                        .intoMemorySegment(output, (outputElemOffset + k) * elemBytes, ByteOrder.nativeOrder(), mask);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            float[] s = ctx.scratch;
            int k = 0, bound = SPECIES.loopBound(n);
            for (; k < bound; k += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, s, aOff + k)
                        .fma(FloatVector.fromArray(SPECIES, s, bOff + k),
                                FloatVector.fromArray(SPECIES, s, cOff + k))
                        .intoArray(output, outputOffset + k);
            }
            if (k < n) {
                var mask = SPECIES.indexInRange(k, n);
                FloatVector.fromArray(SPECIES, s, aOff + k, mask)
                        .fma(FloatVector.fromArray(SPECIES, s, bOff + k, mask),
                                FloatVector.fromArray(SPECIES, s, cOff + k, mask))
                        .intoArray(output, outputOffset + k, mask);
            }
        }
    }

    record IfCommand(int condOff, int trueOff, int falseOff, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            float[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                s[destOff + k] = (s[condOff + k] != 0.0) ? s[trueOff + k] : s[falseOff + k];
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            float[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, outputElemOffset + k,
                        (s[condOff + k] != 0.0) ? s[trueOff + k] : s[falseOff + k]);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            float[] s = ctx.scratch;
            for (int k = 0; k < n; k++) {
                output[outputOffset + k] = (s[condOff + k] != 0.0) ? s[trueOff + k] : s[falseOff + k];
            }
        }
    }

// --- Unified Unary Operations (Delegates to VectorMathF) ---
    @FunctionalInterface
    interface UnaryMathOp {

        void apply(int base, int n, float[] scratch);
    }

    record UnaryMathCommand(UnaryMathOp op, int baseOff) implements VectorCommand, OutputInPlaceCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            op.apply(baseOff, n, ctx.scratch);
        }

        @Override
        public int inPlaceBase() {
            return baseOff;
        }

        @Override
        public void applyInPlaceOnOutput(float[] output, int outputOffset, int n) {
            op.apply(outputOffset, n, output);
        }
    }

    // --- Lane-wise op chains (abs / sqrt) ---
    // abs and sqrt each map to ONE hardware instruction per vector
    // (VectorOperators.ABS / VectorOperators.SQRT). Compiled one-op-per-command
    // they are dominated by memory traffic, not math: sqrt(abs(x)) used to cost
    //   copy x->scratch, abs in place, sqrt in place, scratch->output
    // = four full passes over every block, versus ONE pass for a bare sqrt(x)
    // (LoadSqrtCommand). The two commands below keep the value in a vector
    // register across a chain of up to two lane ops and touch memory once:
    //
    //   LoadLaneChainCommand  variable -> [lane ops] -> scratch / output
    //   LaneChainCommand      scratch  -> [lane ops] -> scratch / output
    //
    // Both implement DirectOutputCommand, so when one is the last command of a
    // plan its result goes straight to the output buffer (array or
    // MemorySegment) and the separate scratch-to-output writeback pass disappears
    // too. The lane ops are the same VectorOperators primitives the unfused
    // abs()/sqrt() ops use, so results are bit-identical to the unfused path.
    //
    // A chain is represented by a small SHAPE id, and each shape has its own
    // straight-line loop in runLanes(). Deliberately NOT a loop over an op
    // array: a vector value carried through a variable-length loop is not
    // reliably scalarized by C2 (it boxes a DoubleVector per op, ~24 B/element),
    // and which way it goes depends on what else the JVM has compiled. Chains
    // longer than two ops are simply split across consecutive commands.
    static final int LANE_ABS = 0;
    static final int LANE_SQRT = 1;

    static final int SHAPE_ABS = 0;         // abs(x)
    static final int SHAPE_SQRT = 1;        // sqrt(x)
    static final int SHAPE_ABS_SQRT = 2;    // sqrt(abs(x))
    static final int SHAPE_SQRT_ABS = 3;    // abs(sqrt(x))
    static final int SHAPE_SQRT_SQRT = 4;   // sqrt(sqrt(x))

    private static boolean isLaneOp(int opcode) {
        return opcode == OP_ABS || opcode == OP_SQRT;
    }

    private static int toLaneOp(int opcode) {
        return opcode == OP_ABS ? LANE_ABS : LANE_SQRT;
    }

    /**
     * The shape that results from applying {@code lane} after {@code shape}, or
     * -1 if that would need more than two ops (caller then starts a new command).
     * abs(abs(x)) == abs(x) exactly, so a repeated abs never grows the chain.
     */
    private static int extendShape(int shape, int lane) {
        return switch (shape) {
            case SHAPE_ABS ->
                lane == LANE_ABS ? SHAPE_ABS : SHAPE_ABS_SQRT;
            case SHAPE_SQRT ->
                lane == LANE_ABS ? SHAPE_SQRT_ABS : SHAPE_SQRT_SQRT;
            default ->
                -1;
        };
    }

    private static int shapeOf(int lane) {
        return lane == LANE_ABS ? SHAPE_ABS : SHAPE_SQRT;
    }

    /**
     * dst = chain(src) for n elements. Four flavours (array/segment source x
     * array/segment destination); src and dst may be the same array at the same
     * offset (in-place): each vector is fully read before it is written.
     */
    private static void runLanesAA(float[] src, int srcOff, float[] dst, int dstOff, int n, int shape) {
        int k = 0;
        final int limit = SPECIES.loopBound(n);
        switch (shape) {
            case SHAPE_ABS -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.ABS)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = Math.abs(src[srcOff + k]);
                }
            }
            case SHAPE_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = (float) Math.sqrt(src[srcOff + k]);
                }
            }
            case SHAPE_ABS_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.ABS)
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = (float) Math.sqrt(Math.abs(src[srcOff + k]));
                }
            }
            case SHAPE_SQRT_ABS -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.SQRT)
                            .lanewise(VectorOperators.ABS)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = Math.abs((float) Math.sqrt(src[srcOff + k]));
                }
            }
            case SHAPE_SQRT_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.SQRT)
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = (float) Math.sqrt((float) Math.sqrt(src[srcOff + k]));
                }
            }
            default ->
                throw new IllegalStateException("bad lane shape " + shape);
        }
    }

    private static void runLanesAS(float[] src, int srcOff, MemorySegment dst, long dstElem, int n, int shape) {
        final long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
        int k = 0;
        final int limit = SPECIES.loopBound(n);
        switch (shape) {
            case SHAPE_ABS -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.ABS)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, Math.abs(src[srcOff + k]));
                }
            }
            case SHAPE_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.SQRT)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, (float) Math.sqrt(src[srcOff + k]));
                }
            }
            case SHAPE_ABS_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.ABS)
                            .lanewise(VectorOperators.SQRT)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, (float) Math.sqrt(Math.abs(src[srcOff + k])));
                }
            }
            case SHAPE_SQRT_ABS -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.SQRT)
                            .lanewise(VectorOperators.ABS)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, Math.abs((float) Math.sqrt(src[srcOff + k])));
                }
            }
            case SHAPE_SQRT_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromArray(SPECIES, src, srcOff + k)
                            .lanewise(VectorOperators.SQRT)
                            .lanewise(VectorOperators.SQRT)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, (float) Math.sqrt((float) Math.sqrt(src[srcOff + k])));
                }
            }
            default ->
                throw new IllegalStateException("bad lane shape " + shape);
        }
    }

    private static void runLanesSA(MemorySegment src, long srcElem, float[] dst, int dstOff, int n, int shape) {
        final long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
        int k = 0;
        final int limit = SPECIES.loopBound(n);
        switch (shape) {
            case SHAPE_ABS -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.ABS)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = Math.abs(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k));
                }
            }
            case SHAPE_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = (float) Math.sqrt(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k));
                }
            }
            case SHAPE_ABS_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.ABS)
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = (float) Math.sqrt(Math.abs(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k)));
                }
            }
            case SHAPE_SQRT_ABS -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.SQRT)
                            .lanewise(VectorOperators.ABS)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = Math.abs((float) Math.sqrt(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k)));
                }
            }
            case SHAPE_SQRT_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.SQRT)
                            .lanewise(VectorOperators.SQRT)
                            .intoArray(dst, dstOff + k);
                }
                for (; k < n; k++) {
                    dst[dstOff + k] = (float) Math.sqrt((float) Math.sqrt(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k)));
                }
            }
            default ->
                throw new IllegalStateException("bad lane shape " + shape);
        }
    }

    private static void runLanesSS(MemorySegment src, long srcElem, MemorySegment dst, long dstElem, int n, int shape) {
        final long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
        int k = 0;
        final int limit = SPECIES.loopBound(n);
        switch (shape) {
            case SHAPE_ABS -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.ABS)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, Math.abs(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k)));
                }
            }
            case SHAPE_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.SQRT)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, (float) Math.sqrt(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k)));
                }
            }
            case SHAPE_ABS_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.ABS)
                            .lanewise(VectorOperators.SQRT)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, (float) Math.sqrt(Math.abs(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k))));
                }
            }
            case SHAPE_SQRT_ABS -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.SQRT)
                            .lanewise(VectorOperators.ABS)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, Math.abs((float) Math.sqrt(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k))));
                }
            }
            case SHAPE_SQRT_SQRT -> {
                for (; k < limit; k += SPECIES.length()) {
                    FloatVector.fromMemorySegment(SPECIES, src, (srcElem + k) * elemBytes, ByteOrder.nativeOrder())
                            .lanewise(VectorOperators.SQRT)
                            .lanewise(VectorOperators.SQRT)
                            .intoMemorySegment(dst, (dstElem + k) * elemBytes, ByteOrder.nativeOrder());
                }
                for (; k < n; k++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElem + k, (float) Math.sqrt((float) Math.sqrt(src.getAtIndex(ValueLayout.JAVA_FLOAT, srcElem + k))));
                }
            }
            default ->
                throw new IllegalStateException("bad lane shape " + shape);
        }
    }

    record LoadLaneChainCommand(int varSlot, int shape, int destOff) implements VectorCommand, DirectOutputCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            if (ctx.flatVariables != null) {
                runLanesAA(ctx.flatVariables, (varSlot * ctx.dataSize) + ctx.blockStart, ctx.scratch, destOff, n, shape);
            } else if (ctx._2DVariables != null) {
                runLanesAA(ctx._2DVariables[varSlot], ctx.blockStart, ctx.scratch, destOff, n, shape);
            } else if (ctx.flatVariablesSeg != null) {
                runLanesSA(ctx.flatVariablesSeg, ((long) varSlot * ctx.dataSize) + ctx.blockStart, ctx.scratch, destOff, n, shape);
            } else {
                runLanesSA(ctx._2DVariablesSeg[varSlot], ctx.blockStart, ctx.scratch, destOff, n, shape);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            if (ctx.flatVariables != null) {
                runLanesAS(ctx.flatVariables, (varSlot * ctx.dataSize) + ctx.blockStart, output, outputElemOffset, n, shape);
            } else if (ctx._2DVariables != null) {
                runLanesAS(ctx._2DVariables[varSlot], ctx.blockStart, output, outputElemOffset, n, shape);
            } else if (ctx.flatVariablesSeg != null) {
                runLanesSS(ctx.flatVariablesSeg, ((long) varSlot * ctx.dataSize) + ctx.blockStart, output, outputElemOffset, n, shape);
            } else {
                runLanesSS(ctx._2DVariablesSeg[varSlot], ctx.blockStart, output, outputElemOffset, n, shape);
            }
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            if (ctx.flatVariables != null) {
                runLanesAA(ctx.flatVariables, (varSlot * ctx.dataSize) + ctx.blockStart, output, outputOffset, n, shape);
            } else if (ctx._2DVariables != null) {
                runLanesAA(ctx._2DVariables[varSlot], ctx.blockStart, output, outputOffset, n, shape);
            } else if (ctx.flatVariablesSeg != null) {
                runLanesSA(ctx.flatVariablesSeg, ((long) varSlot * ctx.dataSize) + ctx.blockStart, output, outputOffset, n, shape);
            } else {
                runLanesSA(ctx._2DVariablesSeg[varSlot], ctx.blockStart, output, outputOffset, n, shape);
            }
        }
    }

    record LaneChainCommand(int shape, int baseOff) implements VectorCommand, DirectOutputCommand, OutputInPlaceCommand {

        @Override
        public void execute(EvaluationContext ctx, int n) {
            runLanesAA(ctx.scratch, baseOff, ctx.scratch, baseOff, n, shape);
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, MemorySegment output, long outputElemOffset) {
            runLanesAS(ctx.scratch, baseOff, output, outputElemOffset, n, shape);
        }

        @Override
        public void executeToOutput(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            runLanesAA(ctx.scratch, baseOff, output, outputOffset, n, shape);
        }

        @Override
        public int inPlaceBase() {
            return baseOff;
        }

        @Override
        public void applyInPlaceOnOutput(float[] output, int outputOffset, int n) {
            runLanesAA(output, outputOffset, output, outputOffset, n, shape);
        }
    }

    // --- Unary tail on the output array ---
    // A plan that ENDS in in-place unary ops -- e.g. floor(x1 + x2), or
    // abs(floor(x1 * x2)) -- used to cost, per block: producer -> scratch,
    // each unary op in place on scratch, then a scratch-to-output writeback.
    // When the command feeding that tail can write straight to `output` (a
    // DirectOutputCommand), the tail ops can just as well run in place on the
    // OUTPUT block it wrote: producer -> output, unary ops in place on output.
    // Same results, one full read+write pass fewer per block (the writeback).
    //
    // OPT-IN: off by default. Measured gain is real only when the block stays
    // cache-resident (small inputs, roughly 1.1x-2x for cheap unary ops such as
    // floor/ceil); at large sizes the run is memory-bound and the result is
    // neutral to slightly negative (about 0.9x-1.05x), so enabling it blindly is
    // not worthwhile. Turn it on with -Dparserng.simd.outputTail=true and
    // benchmark YOUR workload first. Plans it does not apply to are untouched.
    //
    // CONTRACT this relies on: a UnaryMathOp only reads and writes
    // array[base .. base+n). Every op in resolveUnaryMathOp is elementwise, and
    // UnaryMathOp.apply already takes (base, n, array) rather than a scratch
    // slot, but it is the one assumption the compiler does not enforce -- run
    // UnaryOpContractTest before enabling.
    private static final boolean OUTPUT_TAIL_ENABLED = Boolean.getBoolean("parserng.simd.outputTail");

    interface OutputInPlaceCommand {

        /** Scratch offset this command operates at; only base-0 commands can be the tail (the final result lives at 0). */
        int inPlaceBase();

        /** Apply this command in place to output[outputOffset .. outputOffset+n). */
        void applyInPlaceOnOutput(float[] output, int outputOffset, int n);
    }

    private static final class OutputTail {

        final int runLen;                      // commands [0, runLen) run through execute() as usual
        final DirectOutputCommand producer;    // plan[runLen]: writes its result straight to output
        final OutputInPlaceCommand[] tail;     // plan(runLen, end): run in place on output, in order

        OutputTail(int runLen, DirectOutputCommand producer, OutputInPlaceCommand[] tail) {
            this.runLen = runLen;
            this.producer = producer;
            this.tail = tail;
        }

        void apply(EvaluationContext ctx, int n, float[] output, int outputOffset) {
            producer.executeToOutput(ctx, n, output, outputOffset);
            for (int i = 0; i < tail.length; i++) {
                tail[i].applyInPlaceOnOutput(output, outputOffset, n);
            }
        }
    }

    /**
     * Splits {@code plan} into [normal commands][producer][in-place unary tail]
     * when that shape applies, else returns null (caller keeps the ordinary
     * execute()+writeback path). Only used when the LAST command is not itself a
     * DirectOutputCommand -- that case is already handled by the terminal path.
     *
     * <p>The command immediately before the tail produced the value the first tail
     * op consumes (strict stack machine), and every tail op sits at base 0, so the
     * producer's result is at scratch 0 -- exactly what executeToOutput writes.
     */
    private static OutputTail analyzeOutputTail(VectorCommand[] plan) {
        if (!OUTPUT_TAIL_ENABLED) {
            return null;
        }
        int i = plan.length;
        while (i > 0 && plan[i - 1] instanceof OutputInPlaceCommand c && c.inPlaceBase() == 0) {
            i--;
        }
        if (i == plan.length || i == 0 || !(plan[i - 1] instanceof DirectOutputCommand producer)) {
            return null;
        }
        OutputInPlaceCommand[] tail = new OutputInPlaceCommand[plan.length - i];
        for (int j = 0; j < tail.length; j++) {
            tail[j] = (OutputInPlaceCommand) plan[i + j];
        }
        return new OutputTail(i - 1, producer, tail);
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

                    VectorCommand fused = tryFuseLoadLoad(plan, opcode, lOff, rOff, destOff);
                    if (fused != null) {
                        // Both LoadCommands are dead after this point: codegen is
                        // a strict LIFO stack machine, so lOff/rOff cannot be
                        // referenced by anything else once this op consumes them.
                        // Removing them skips materializing both operands into
                        // ctx.scratch - the fused command reads straight from
                        // flatVariables/_2DVariables (or their MemorySegment
                        // equivalents) instead.
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

                    // Cheap lane-wise ops (abs, sqrt) collapse into one command --
                    // see tryFuseLaneChain(). Everything else falls through to the
                    // original table below, untouched.
                    VectorCommand laneFused = tryFuseLaneChain(plan, opcode, baseOff);
                    if (laneFused != null) {
                        plan.add(laneFused);
                        continue;
                    }

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
                        case OP_CEIL ->
                            VectorMathF::ceil;
                        case OP_ROUND ->
                            VectorMathF::round;
                        case OP_FLOOR ->
                            VectorMathF::floor;

                        case OP_FACT ->
                            VectorMathF::fact;

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

    /**
     * Peephole fusion: when both operands of a binary arithmetic op are plain
     * variable loads - i.e. the last two entries in the plan are LoadCommands
     * feeding directly into this op and nothing else - collapse them into a
     * single fused command that reads straight from the source (flatVariables /
     * _2DVariables, or their MemorySegment equivalents) instead of
     * round-tripping both operands through ctx.scratch. Returns null (no
     * fusion) for anything that doesn't match that exact shape, including
     * OP_REM (not vectorized regardless) and OP_POW (routes through
     * VectorMathF.executePowerBlended, not a plain lane op).
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
     * Peephole for the cheap lane-wise unary ops ({@code abs}, {@code sqrt}):
     * folds them into one command so a chain costs one pass over memory
     * instead of one pass per op. See the "Lane-wise op chains" comment above
     * {@link LoadLaneChainCommand}.
     *
     * <p>The top of the virtual stack is always produced by the LAST plan
     * entry (strict stack machine), so only the tail of {@code plan} is ever
     * inspected. Cases, in order:
     * <ul>
     * <li>tail is a {@link LoadLaneChainCommand} / {@link LaneChainCommand} at
     * {@code baseOff} with room for another op: extend its chain;</li>
     * <li>tail is a bare {@link LoadCommand}: start a
     * {@link LoadLaneChainCommand};</li>
     * <li>otherwise (e.g. {@code sqrt(x+1)}, or a chain that is already two ops
     * long) the op becomes an in-place {@link LaneChainCommand}.</li>
     * </ul>
     * Returns {@code null} (leaving {@code plan} untouched) when {@code opcode}
     * is not a lane op,.
     */
    private static VectorCommand tryFuseLaneChain(List<VectorCommand> plan, int opcode, int baseOff) {
        if (!isLaneOp(opcode)) {
            return null;
        }
        final int lane = toLaneOp(opcode);
        final int size = plan.size();
        if (size >= 1) {
            VectorCommand last = plan.get(size - 1);
            if (last instanceof LoadLaneChainCommand c && c.destOff() == baseOff) {
                int ext = extendShape(c.shape(), lane);
                if (ext >= 0) {
                    plan.remove(size - 1);
                    return new LoadLaneChainCommand(c.varSlot(), ext, baseOff);
                }
            } else if (last instanceof LaneChainCommand c && c.baseOff() == baseOff) {
                int ext = extendShape(c.shape(), lane);
                if (ext >= 0) {
                    plan.remove(size - 1);
                    return new LaneChainCommand(ext, baseOff);
                }
            } else if (last instanceof LoadCommand c && c.destOff() == baseOff) {
                plan.remove(size - 1);
                return new LoadLaneChainCommand(c.slotIdx(), shapeOf(lane), baseOff);
            }
        }
        return new LaneChainCommand(shapeOf(lane), baseOff);
    }

    public final class SIMDVectorCompositeExpression extends BatchedVectorCompositeExpression implements AutoCloseable {

        private static final Cleaner SYSTEM_CLEANER = Cleaner.create();

        // Bounded spin budget used by the spin-then-park hybrid wait below, both
        // for a worker waiting on new dispatch and for the master fanning-in on
        // worker completion. Avoids the syscall cost of park()/unpark() on the
        // (extremely common) case where the wait is only a handful of nanoseconds,
        // while still falling back to a real park so we don't burn a core forever.
        private static final int SPIN_LIMIT = 4_000;

        // ---- opt-in timing probes: -Dparserng.diag=true (JIT folds the checks away when false) ----
        private static final boolean DIAG = Boolean.getBoolean("parserng.diag");
        private static final java.util.concurrent.atomic.LongAdder D_PICKUP = new java.util.concurrent.atomic.LongAdder();
        private static final java.util.concurrent.atomic.LongAdder D_BUSY = new java.util.concurrent.atomic.LongAdder();
        private static final java.util.concurrent.atomic.LongAdder D_WAIT = new java.util.concurrent.atomic.LongAdder();
        private static final java.util.concurrent.atomic.LongAdder D_TASKS = new java.util.concurrent.atomic.LongAdder();

        /** Average per worker task, in microseconds; resets the counters. */
        public static String diagReportAndReset() {
            long t = Math.max(1, D_TASKS.sumThenReset());
            double pickup = D_PICKUP.sumThenReset() / 1000.0 / t;
            double busy = D_BUSY.sumThenReset() / 1000.0 / t;
            double wait = D_WAIT.sumThenReset() / 1000.0 / t;
            return String.format("  [diag] per worker task: dispatch->pickup %.1f us | worker busy %.1f us | master wait %.1f us (tasks=%d)", pickup, busy, wait, t);
        }

        private final int NUM_WORKERS;
        private final WorkerThread[] workerPool;
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

                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i] = new WorkerThread(i, executionPlan, stackDepth, blockSize);
                }

                for (int i = 0; i < NUM_WORKERS; i++) {
                    workerPool[i].start();
                }

                this.cleanable = SYSTEM_CLEANER.register(this, new ThreadPoolShutdownAction(workerPool));
            } else {
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

        /**
         * Spin-then-park hybrid used by the master to fan-in on a single
         * worker's completion. Replaces the old design where every worker
         * decremented one shared AtomicInteger latch and the master parked on
         * that - under fan-in, all NUM_WORKERS threads were hammering the same
         * cache line on every batch. Each WorkerThread now owns its own `done`
         * flag (padded away from its neighbors, see WorkerThread), so this is a
         * plain, uncontended volatile read per worker.
         */
        private static void awaitDone(WorkerThread worker) {
            final long w0 = DIAG ? System.nanoTime() : 0L;
            awaitDone0(worker);
            if (DIAG) { D_WAIT.add(System.nanoTime() - w0); }
        }

        private static void awaitDone0(WorkerThread worker) {
            int spins = 0;
            while (!worker.done) {
                if (spins < SPIN_LIMIT) {
                    Thread.onSpinWait();
                    spins++;
                } else {
                    LockSupport.park();
                }
            }
        }

        private static final class WorkerThread extends Thread {

            private final int workerId;
            private final EvaluationContext evalContext;
            private final VectorCommand[] executionPlan;
            private final int blockSize;

            private volatile boolean isRunning = true;

            // --- Padding: isolates the hot dispatch/completion fields below from
            // whatever lands adjacent to this WorkerThread instance on the heap
            // (e.g. the neighboring WorkerThread in workerPool[]). Without this,
            // one worker's dispatch writes and another worker's completion flag
            // can share a cache line and false-share on every batch, even though
            // the two workers never touch each other's actual data. 8 longs = 64
            // bytes on each side of the hot block, i.e. a full cache line of
            // padding front and back. ---
            private long fp0, fp1, fp2, fp3, fp4, fp5, fp6, fp7;

            // taskState: 0 = idle/available, 1 = task dispatched, awaiting pickup.
            private volatile int taskState = 0;
            private long submitNanos;
            // Per-worker completion flag. The master spins/parks on THIS field
            // (via awaitDone) instead of a single shared AtomicInteger latch, so
            // fan-in no longer forces every worker to contend on one cache line.
            private volatile boolean done = true;
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

            private long bp0, bp1, bp2, bp3, bp4, bp5, bp6, bp7;

            public WorkerThread(int workerId, VectorCommand[] executionPlan, int stackDepth, int blockSize) {
                this.workerId = workerId;
                this.executionPlan = executionPlan;
                this.blockSize = blockSize;
                this.evalContext = new EvaluationContext(stackDepth, blockSize);
                this.setDaemon(true);
                this.setName("ParserNG-SIMD-Worker-" + workerId);
            }

            public void submitTask2D(float[][] vars, float[] output, int dataSize, int startIdx, int length) {
                this.vars2D = vars;
                this.vars1D = null;
                this.vars2DSeg = null;
                this.vars1DSeg = null;
                this.output = output;
                this.outputSeg = null;
                this.dataSize = dataSize;
                this.startIdx = startIdx;
                this.length = length;
                this.done = false;
                this.masterThread = Thread.currentThread();
                if (DIAG) { this.submitNanos = System.nanoTime(); }
                this.taskState = 1;
                LockSupport.unpark(this);
            }

            public void submitTask1D(float[] vars, float[] output, int dataSize, int startIdx, int length) {
                this.vars1D = vars;
                this.vars2D = null;
                this.vars2DSeg = null;
                this.vars1DSeg = null;
                this.output = output;
                this.outputSeg = null;
                this.dataSize = dataSize;
                this.startIdx = startIdx;
                this.length = length;
                this.done = false;
                this.masterThread = Thread.currentThread();
                if (DIAG) { this.submitNanos = System.nanoTime(); }
                this.taskState = 1;
                LockSupport.unpark(this);
            }

            public void submitTaskSeg2D(MemorySegment[] vars, MemorySegment output, int dataSize, int startIdx, int length) {
                this.vars2DSeg = vars;
                this.vars1DSeg = null;
                this.vars2D = null;
                this.vars1D = null;
                this.outputSeg = output;
                this.output = null;
                this.dataSize = dataSize;
                this.startIdx = startIdx;
                this.length = length;
                this.done = false;
                this.masterThread = Thread.currentThread();
                if (DIAG) { this.submitNanos = System.nanoTime(); }
                this.taskState = 1;
                LockSupport.unpark(this);
            }

            public void submitTaskSeg1D(MemorySegment vars, MemorySegment output, int dataSize, int startIdx, int length) {
                this.vars1DSeg = vars;
                this.vars2DSeg = null;
                this.vars1D = null;
                this.vars2D = null;
                this.outputSeg = output;
                this.output = null;
                this.dataSize = dataSize;
                this.startIdx = startIdx;
                this.length = length;
                this.done = false;
                this.masterThread = Thread.currentThread();
                if (DIAG) { this.submitNanos = System.nanoTime(); }
                this.taskState = 1;
                LockSupport.unpark(this);
            }

            public void terminate() {
                this.isRunning = false;
                this.interrupt();
            }

            @Override
            public void run() {
                CPUPinner.pinWorker(this.workerId);
                while (isRunning) {
                    int spins = 0;
                    while (taskState == 0 && isRunning) {
                        if (spins < SPIN_LIMIT) {
                            Thread.onSpinWait();
                            spins++;
                        } else {
                            LockSupport.park();
                            if (Thread.interrupted() && !isRunning) {
                                return;
                            }
                        }
                    }
                    if (!isRunning) {
                        return;
                    }

                    // try/finally: guarantees `done` is always raised and task
                    // state always cleared, even if a bad expression or
                    // malformed input throws mid-block. Without this, a single
                    // faulting task would leave the master's awaitDone() loop
                    // spinning/parking forever on a flag that never flips.
                    final long diagT0 = DIAG ? System.nanoTime() : 0L;
                    if (DIAG) { D_PICKUP.add(diagT0 - this.submitNanos); }
                    try {
                        if (vars2D != null) {
                            applyBulkInternal(vars2D, evalContext, executionPlan, blockSize, dataSize, output, startIdx, length);
                        } else if (vars1D != null) {
                            applyBulkInternal(vars1D, evalContext, executionPlan, blockSize, dataSize, output, startIdx, length);
                        } else if (vars2DSeg != null) {
                            applyBulkInternalSeg(vars2DSeg, evalContext, executionPlan, blockSize, dataSize, outputSeg, startIdx, length);
                        } else if (vars1DSeg != null) {
                            applyBulkInternalSeg(vars1DSeg, evalContext, executionPlan, blockSize, dataSize, outputSeg, startIdx, length);
                        }
                    } finally {
                        if (DIAG) { D_BUSY.add(System.nanoTime() - diagT0); D_TASKS.increment(); }
                        this.taskState = 0;
                        this.vars2D = null;
                        this.vars1D = null;
                        this.output = null;
                        this.vars2DSeg = null;
                        this.vars1DSeg = null;
                        this.outputSeg = null;

                        // Plain per-worker volatile write - no unpark() needed since the
                        // master fans-in via awaitDone()'s spin-then-park hybrid, which
                        // simply polls this flag.
                        this.done = true;
                        Thread master = this.masterThread;
                        if (master != null) {
                            LockSupport.unpark(master);
                        }
                    }
                }
            }
        }

        /**
         *
         * @param variables
         * @param output
         */
        public void validate(float[][] variables, float[] output) {
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

        /**
         *
         * @param flatVariables
         * @param output
         */
        public void validate(float[] flatVariables, float[] output) {
            int totalSamples = flatVariables != null && flatVariables.length > 0 && output != null && output.length > 0 ? flatVariables.length : -1;
            int stride = getVarCount();
            if (totalSamples != stride * output.length) {
                throw new IllegalStateException(String.format("array sizes not correct[totalSamples=%d vs computed(var-count*output-array-size)=%d]",
                        totalSamples, stride * output.length));
            }
        }

        /**
         *
         * @param variables
         * @param output
         */
        public void validate(double[][] variables, double[] output) {
            throw new InputMismatchException("double[][] not supported only float[] and float[][], MemorySegment and MemorySegment[]");
        }

        /**
         *
         * @param flatVariables
         * @param output
         */
        public void validate(double[] flatVariables, double[] output) {
            throw new InputMismatchException("double[][] not supported only float[] and float[][], MemorySegment and MemorySegment[]");
        }

        // --- MemorySegment (off-heap / zero-copy) entry points ---
        // Same public contract as the float[] / float[][] variants above,
        // reusing the same masterEvalContext / workerPool / per-worker done-flag machinery.
        /**
         * Validates a single flat off-heap variables segment against its output
         * segment. numSamples for this entry point is derived from output's
         * byte size (see applyBulk(MemorySegment, MemorySegment) below), so
         * variables must hold exactly that many float elements too, or the read
         * past its end would either throw or silently read garbage/adjacent
         * memory depending on how it was allocated.
         *
         * @param variables
         * @param output
         */
        public void validate(MemorySegment variables, MemorySegment output) {
            if (variables == null) {
                throw new InputMismatchException("variables MemorySegment is null");
            }
            if (output == null) {
                throw new InputMismatchException("output MemorySegment is null");
            }
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            if (variables.byteSize() == 0 || output.byteSize() == 0) {
                throw new InputMismatchException("null/empty MemorySegment");
            }
            if (variables.byteSize() % elemBytes != 0) {
                throw new InputMismatchException("variables MemorySegment byte size " + variables.byteSize()
                        + " is not a whole multiple of the float element size (" + elemBytes + " bytes)");
            }
            if (output.byteSize() % elemBytes != 0) {
                throw new InputMismatchException("output MemorySegment byte size " + output.byteSize()
                        + " is not a whole multiple of the float element size (" + elemBytes + " bytes)");
            }
            if (variables.byteSize() != output.byteSize()) {
                throw new InputMismatchException("variables (" + (variables.byteSize() / elemBytes)
                        + " samples) and output (" + (output.byteSize() / elemBytes)
                        + " samples) must hold the same number of elements");
            }
        }

        /**
         * Validates a per-variable off-heap segment array against its output
         * segment. numSamples for this entry point is derived from
         * variables[0]'s byte size (see applyBulk(MemorySegment[],
         * MemorySegment) below), so every other segment in the array - and
         * output - must agree with that sample count, or later variables would
         * be read out of bounds relative to variables[0]'s length.
         *
         * @param variables
         * @param output
         */
        public void validate(MemorySegment[] variables, MemorySegment output) {
            if (variables == null || variables.length == 0) {
                throw new InputMismatchException("null/empty MemorySegment[]");
            }
            if (output == null) {
                throw new InputMismatchException("output MemorySegment is null");
            }
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            if (output.byteSize() == 0) {
                throw new InputMismatchException("null/empty MemorySegment");
            }
            if (output.byteSize() % elemBytes != 0) {
                throw new InputMismatchException("output MemorySegment byte size " + output.byteSize()
                        + " is not a whole multiple of the float element size (" + elemBytes + " bytes)");
            }
            if (variables[0] == null) {
                throw new InputMismatchException("variables[0] is null");
            }
            long expectedByteSize = variables[0].byteSize();
            if (expectedByteSize == 0) {
                throw new InputMismatchException("null/empty MemorySegment");
            }
            if (expectedByteSize % elemBytes != 0) {
                throw new InputMismatchException("variables[0] MemorySegment byte size " + expectedByteSize
                        + " is not a whole multiple of the float element size (" + elemBytes + " bytes)");
            }
            for (int i = 1; i < variables.length; i++) {
                MemorySegment seg = variables[i];
                if (seg == null) {
                    throw new InputMismatchException("variables[" + i + "] is null");
                }
                if (seg.byteSize() != expectedByteSize) {
                    throw new InputMismatchException("variables[" + i + "] has " + (seg.byteSize() / elemBytes)
                            + " samples but variables[0] has " + (expectedByteSize / elemBytes)
                            + " - all variable segments must be the same length");
                }
            }
            if (output.byteSize() != expectedByteSize) {
                throw new InputMismatchException("output has " + (output.byteSize() / elemBytes)
                        + " samples but variables have " + (expectedByteSize / elemBytes) + " samples");
            }
        }

        public void applyBulk(float[][] variables, float[] output) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            int numSamples = variables[0].length;
            applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(float[][] variables, float[] output) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
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

            // Zero-allocation chunk plan: pure local/primitive arithmetic below -
            // no arrays, no boxing. Aligns each participant's boundary to the
            // SIMD lane width and hands any leftover aligned units out one-per-
            // participant instead of dumping the whole remainder on the last one.
            final int participants = NUM_WORKERS + 1; // NUM_WORKERS background threads + the calling (master) thread
            final int align = Math.max(1, SPECIES.length());
            final int totalUnits = (numSamples + align - 1) / align;
            final int baseUnits = totalUnits / participants;
            final int extraUnits = totalUnits % participants;

            int cursor = 0;
            int masterStart = 0, masterLength = 0;
            // Participant 0 is the master's own chunk; workers take 1..NUM_WORKERS.
            for (int i = 0; i < participants; i++) {
                int units = baseUnits + (i < extraUnits ? 1 : 0);
                int len = Math.max(0, Math.min(units * align, numSamples - cursor));
                if (i == 0) {
                    masterStart = cursor;
                    masterLength = len;
                } else {
                    workerPool[i - 1].submitTask2D(variables, output, numSamples, cursor, len);
                }
                cursor += len;
            }

            // The master used to just block here reserving a whole hardware thread
            // for nothing - now it does its own slice of the work inline first.
            if (masterLength > 0) {
                applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, masterStart, masterLength);
            }

            for (int i = 0; i < NUM_WORKERS; i++) {
                awaitDone(workerPool[i]);
            }
        }

        public void applyBulkParallel(float[] flatVariables, float[] output) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
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

            final int participants = NUM_WORKERS + 1;
            final int align = Math.max(1, SPECIES.length());
            final int totalUnits = (numSamples + align - 1) / align;
            final int baseUnits = totalUnits / participants;
            final int extraUnits = totalUnits % participants;

            int cursor = 0;
            int masterStart = 0, masterLength = 0;
            for (int i = 0; i < participants; i++) {
                int units = baseUnits + (i < extraUnits ? 1 : 0);
                int len = Math.max(0, Math.min(units * align, numSamples - cursor));
                if (i == 0) {
                    masterStart = cursor;
                    masterLength = len;
                } else {
                    workerPool[i - 1].submitTask1D(flatVariables, output, numSamples, cursor, len);
                }
                cursor += len;
            }

            if (masterLength > 0) {
                applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, masterStart, masterLength);
            }

            for (int i = 0; i < NUM_WORKERS; i++) {
                awaitDone(workerPool[i]);
            }
        }

        public void applyBulkBatched(float[][] variables, float[] output, int batchSize) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            EvaluationContext ctx = masterEvalContext.get();
            int numSamples = variables[0].length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(variables, ctx, executionPlan, BLOCK_SIZE, numSamples, output, start, length);
            }
        }

        public void applyBulk(float[] flatVariables, float[] output) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, output.length, output, 0, output.length);
        }

        public void applyBulkBatched(float[] flatVariables, float[] output, int batchSize) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            EvaluationContext ctx = masterEvalContext.get();
            int numSamples = output.length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(flatVariables, ctx, executionPlan, BLOCK_SIZE, numSamples, output, start, length);
            }
        }

        // --- MemorySegment (off-heap / zero-copy) entry points ---
        // Same public contract as the float[] / float[][] variants above,
        // reusing the same masterEvalContext / workerPool / per-worker done-flag machinery.
        public void applyBulk(MemorySegment[] variables, MemorySegment output) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            int numSamples = (int) (variables[0].byteSize() / ValueLayout.JAVA_FLOAT.byteSize());
            applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(MemorySegment[] variables, MemorySegment output) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = (int) (variables[0].byteSize() / ValueLayout.JAVA_FLOAT.byteSize());

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            final int participants = NUM_WORKERS + 1;
            final int align = Math.max(1, SPECIES.length());
            final int totalUnits = (numSamples + align - 1) / align;
            final int baseUnits = totalUnits / participants;
            final int extraUnits = totalUnits % participants;

            int cursor = 0;
            int masterStart = 0, masterLength = 0;
            for (int i = 0; i < participants; i++) {
                int units = baseUnits + (i < extraUnits ? 1 : 0);
                int len = Math.max(0, Math.min(units * align, numSamples - cursor));
                if (i == 0) {
                    masterStart = cursor;
                    masterLength = len;
                } else {
                    workerPool[i - 1].submitTaskSeg2D(variables, output, numSamples, cursor, len);
                }
                cursor += len;
            }

            if (masterLength > 0) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, masterStart, masterLength);
            }

            for (int i = 0; i < NUM_WORKERS; i++) {
                awaitDone(workerPool[i]);
            }
        }

        public void applyBulk(MemorySegment variables, MemorySegment output) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            int numSamples = (int) (output.byteSize() / ValueLayout.JAVA_FLOAT.byteSize());
            applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(MemorySegment variables, MemorySegment output) {
            if (varCount == 0) {
                fillOutput((float) SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            if (variables == null || output == null) {
                return;
            }
            int numSamples = (int) (output.byteSize() / ValueLayout.JAVA_FLOAT.byteSize());

            if (NUM_WORKERS <= 0 || numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
                return;
            }

            final int participants = NUM_WORKERS + 1;
            final int align = Math.max(1, SPECIES.length());
            final int totalUnits = (numSamples + align - 1) / align;
            final int baseUnits = totalUnits / participants;
            final int extraUnits = totalUnits % participants;

            int cursor = 0;
            int masterStart = 0, masterLength = 0;
            for (int i = 0; i < participants; i++) {
                int units = baseUnits + (i < extraUnits ? 1 : 0);
                int len = Math.max(0, Math.min(units * align, numSamples - cursor));
                if (i == 0) {
                    masterStart = cursor;
                    masterLength = len;
                } else {
                    workerPool[i - 1].submitTaskSeg1D(variables, output, numSamples, cursor, len);
                }
                cursor += len;
            }

            if (masterLength > 0) {
                applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, masterStart, masterLength);
            }

            for (int i = 0; i < NUM_WORKERS; i++) {
                awaitDone(workerPool[i]);
            }
        }

        // --- Core Internal Hot-Loops with Vectorized Copy Defenses ---
        private static void applyBulkInternal(float[] flatVariables, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, float[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            float[] s = ctx.scratch;
            final int planLen = executionPlan.length;
            // If the last command can write its result straight to `output` (see
            // DirectOutputCommand), run every command before it as usual and skip
            // BOTH its execute() and the scratch-to-output writeback below.
            // Otherwise `terminal` is null and the original path runs unchanged.
            final DirectOutputCommand terminal = (planLen > 0 && executionPlan[planLen - 1] instanceof DirectOutputCommand doc) ? doc : null;
            // Plan ends in in-place unary ops fed by a direct-output producer: the
            // producer writes `output` and the unary tail runs in place on it
            // (see analyzeOutputTail). Only considered when `terminal` is null.
            final OutputTail otail = terminal == null ? analyzeOutputTail(executionPlan) : null;
            final int runLen = terminal != null ? planLen - 1 : (otail != null ? otail.runLen : planLen);
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlock(flatVariables, null, dataSize, blockStart);

                for (int i = 0; i < runLen; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                if (terminal != null) {
                    terminal.executeToOutput(ctx, currentBlockSize, output, blockStart);
                } else if (otail != null) {
                    otail.apply(ctx, currentBlockSize, output, blockStart);
                } else {
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
        }

        private static void applyBulkInternal(float[][] variables, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, float[] output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            float[] s = ctx.scratch;
            final int planLen = executionPlan.length;
            // If the last command can write its result straight to `output` (see
            // DirectOutputCommand), run every command before it as usual and skip
            // BOTH its execute() and the scratch-to-output writeback below.
            // Otherwise `terminal` is null and the original path runs unchanged.
            final DirectOutputCommand terminal = (planLen > 0 && executionPlan[planLen - 1] instanceof DirectOutputCommand doc) ? doc : null;
            // Plan ends in in-place unary ops fed by a direct-output producer: the
            // producer writes `output` and the unary tail runs in place on it
            // (see analyzeOutputTail). Only considered when `terminal` is null.
            final OutputTail otail = terminal == null ? analyzeOutputTail(executionPlan) : null;
            final int runLen = terminal != null ? planLen - 1 : (otail != null ? otail.runLen : planLen);
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlock(null, variables, dataSize, blockStart);

                for (int i = 0; i < runLen; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                if (terminal != null) {
                    terminal.executeToOutput(ctx, currentBlockSize, output, blockStart);
                } else if (otail != null) {
                    otail.apply(ctx, currentBlockSize, output, blockStart);
                } else {
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
        }

        // --- MemorySegment Core Internal Hot-Loops (mirrors the float[] / float[][] loops above) ---
        private static void applyBulkInternalSeg(MemorySegment flatVariablesSeg, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, MemorySegment output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            final long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            float[] s = ctx.scratch;
            final int planLen = executionPlan.length;
            // If the last command can write its result straight to `output` (see
            // DirectOutputCommand), run every command before it as usual and skip
            // BOTH its execute() and the scratch-to-output writeback below.
            // Otherwise `terminal` is null and the original path runs unchanged.
            final DirectOutputCommand terminal = (planLen > 0 && executionPlan[planLen - 1] instanceof DirectOutputCommand doc) ? doc : null;
            final int runLen = terminal != null ? planLen - 1 : planLen;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlockSeg(flatVariablesSeg, null, dataSize, blockStart);

                for (int i = 0; i < runLen; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                if (terminal != null) {
                    terminal.executeToOutput(ctx, currentBlockSize, output, blockStart);
                } else {
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
        }

        private static void applyBulkInternalSeg(MemorySegment[] variablesSeg, EvaluationContext ctx, VectorCommand[] executionPlan, int blockSize, int dataSize, MemorySegment output, int startIdx, int length) {
            final int endIdx = startIdx + length;
            final long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            float[] s = ctx.scratch;
            final int planLen = executionPlan.length;
            // If the last command can write its result straight to `output` (see
            // DirectOutputCommand), run every command before it as usual and skip
            // BOTH its execute() and the scratch-to-output writeback below.
            // Otherwise `terminal` is null and the original path runs unchanged.
            final DirectOutputCommand terminal = (planLen > 0 && executionPlan[planLen - 1] instanceof DirectOutputCommand doc) ? doc : null;
            final int runLen = terminal != null ? planLen - 1 : planLen;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += blockSize) {
                final int currentBlockSize = Math.min(blockSize, endIdx - blockStart);
                ctx.initForBlockSeg(null, variablesSeg, dataSize, blockStart);

                for (int i = 0; i < runLen; i++) {
                    executionPlan[i].execute(ctx, currentBlockSize);
                }

                if (terminal != null) {
                    terminal.executeToOutput(ctx, currentBlockSize, output, blockStart);
                } else {
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

        // Note: fillOutput(float, float[]) lives on the shared base class
        // (BatchedVectorCompositeExpression) and isn't in this file, so this
        // is declared here directly rather than as a sibling overload next
        // to it. Same vectorized-broadcast + scalar-tail shape as the
        // applyBulkInternalSeg write-backs below, just filling every slot
        // with one constant instead of the command-plan result.
        protected void fillOutput(float value, MemorySegment out) {
            long elemBytes = ValueLayout.JAVA_FLOAT.byteSize();
            int n = (int) (out.byteSize() / elemBytes);
            FloatVector bcast = FloatVector.broadcast(SPECIES, value);
            int k = 0, limit = SPECIES.loopBound(n);
            for (; k < limit; k += SPECIES.length()) {
                bcast.intoMemorySegment(out, k * elemBytes, ByteOrder.nativeOrder());
            }
            for (; k < n; k++) {
                out.setAtIndex(ValueLayout.JAVA_FLOAT, k, value);
            }
        }
    }

}