package com.github.gbenroscience.simdext.turbo.tools.command;

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
    record LoadLoadAddCommand(int lSlot, int rSlot, int destOff) implements VectorCommand {

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
    }

    record LoadLoadSubCommand(int lSlot, int rSlot, int destOff) implements VectorCommand {

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
    }

    record LoadLoadMulCommand(int lSlot, int rSlot, int destOff) implements VectorCommand {

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
    }

    record LoadLoadDivCommand(int lSlot, int rSlot, int destOff) implements VectorCommand {

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
                s[destOff + k] = (s[condOff + k] != 0.0) ? s[trueOff + k] : s[falseOff + k];
            }
        }
    }

// --- Unified Unary Operations (Delegates to VectorMathF) ---
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

            public WorkerThread(int workerId, AtomicInteger reuseLatch, VectorCommand[] executionPlan, int stackDepth, int blockSize) {
                this.workerId = workerId;
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
        // reusing the same masterEvalContext / workerPool / reuseLatch machinery.

        /**
         * Validates a single flat off-heap variables segment against its
         * output segment. numSamples for this entry point is derived from
         * output's byte size (see applyBulk(MemorySegment, MemorySegment)
         * below), so variables must hold exactly that many float elements
         * too, or the read past its end would either throw or silently read
         * garbage/adjacent memory depending on how it was allocated.
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
         * Validates a per-variable off-heap segment array against its
         * output segment. numSamples for this entry point is derived from
         * variables[0]'s byte size (see applyBulk(MemorySegment[],
         * MemorySegment) below), so every other segment in the array - and
         * output - must agree with that sample count, or later variables
         * would be read out of bounds relative to variables[0]'s length.
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
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            int numSamples = variables[0].length;
            applyBulkInternal(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(float[][] variables, float[] output) {
              if (varCount == 0) {
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
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

        public void applyBulkParallel(float[] flatVariables, float[] output) {
              if (varCount == 0) {
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
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

        public void applyBulkBatched(float[][] variables, float[] output, int batchSize) {
              if (varCount == 0) {
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
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
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            applyBulkInternal(flatVariables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, output.length, output, 0, output.length);
        }

        public void applyBulkBatched(float[] flatVariables, float[] output, int batchSize) {
              if (varCount == 0) {
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
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
        // reusing the same masterEvalContext / workerPool / reuseLatch machinery.
        public void applyBulk(MemorySegment[] variables, MemorySegment output) {
             if (varCount == 0) {
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            int numSamples = (int) (variables[0].byteSize() / ValueLayout.JAVA_FLOAT.byteSize());
            applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(MemorySegment[] variables, MemorySegment output) {
             if (varCount == 0) {
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
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
             if (varCount == 0) {
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
                return;
            }
            int numSamples = (int) (output.byteSize() / ValueLayout.JAVA_FLOAT.byteSize());
            applyBulkInternalSeg(variables, masterEvalContext.get(), executionPlan, BLOCK_SIZE, numSamples, output, 0, numSamples);
        }

        public void applyBulkParallel(MemorySegment variables, MemorySegment output) {
             if (varCount == 0) {
                fillOutput((float)SIMDCommandSegmentF32.this.constantAnswer, output);
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
