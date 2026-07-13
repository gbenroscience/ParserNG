package com.github.gbenroscience.simd.turbo.tools;

import com.github.gbenroscience.simd.turbo.tools.matrix.ParallelTiledEngine;
import java.util.*;
import java.util.concurrent.*;
import jdk.incubator.vector.*;
import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.*;

/**
 *
 * @author GBEMIRO
 */
public final class FlatMatrixF {

    private static final boolean HAS_VECTOR = VF_LEN > 1;

    // GELU constants float
    private static final float GELU_C1 = 0.7978845608028654f;
    private static final float GELU_C2 = 0.044715f;
    private static final FloatVector V_GELU_C1 = FloatVector.broadcast(F_SPECIES, GELU_C1);
    private static final FloatVector V_GELU_C2 = FloatVector.broadcast(F_SPECIES, GELU_C2);
    private static final FloatVector V_HALF = FloatVector.broadcast(F_SPECIES, 0.5f);
    private static final FloatVector V_ONE = FloatVector.broadcast(F_SPECIES, 1.0f);

    // Clamp to avoid exp overflow. exp(-10) = 4.5e-5, exp(10) = 22026 -> 1.0f
    private static final FloatVector MIN_X = FloatVector.broadcast(F_SPECIES, -10.0f);
    private static final FloatVector MAX_X = FloatVector.broadcast(F_SPECIES, 10.0f);
    private static final FloatVector ONE = FloatVector.broadcast(F_SPECIES, 1.0f);
    private static final FloatVector NEG_ONE = FloatVector.broadcast(F_SPECIES, -1.0f);

    // Block sizes for float - NR matches float lanes
    private static final int MC = 64;
    private static final int NC = 256, KC = 256, MR = 4;
    private static final int NR = VF_LEN; // 8 for AVX2, 16 for AVX512

    public final float[] data;
    public final int rows, cols;
    public final int rowStride;
    public final int offset;
    private byte[] byteData; // lazy backing for Q8

    // Reusable buffer for scalar fallback (small, per-thread)
    private static final ThreadLocal<float[]> SCALAR_ROW_BUFFER
            = ThreadLocal.withInitial(() -> new float[MR * NR]);

// Reusable buffers
    private static final ThreadLocal<FloatVector[]> ACC_BUFFER_TL
            = ThreadLocal.withInitial(() -> new FloatVector[MR]);

    public FlatMatrixF(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.rowStride = cols;
        this.offset = 0;
        this.data = new float[rows * cols];
    }

    public FlatMatrixF(int rows, int cols, float[] data) {
        if (data.length != rows * cols) {
            throw new IllegalArgumentException("Data length mismatch");
        }
        this.rows = rows;
        this.cols = cols;
        this.data = data;
        this.rowStride = cols;
        this.offset = 0;
    }

    private FlatMatrixF(float[] data, int rows, int cols, int rowStride, int offset) {
        this.data = data;
        this.rows = rows;
        this.cols = cols;
        this.rowStride = rowStride;
        this.offset = offset;
    }

    public FlatMatrixF view(int rowOff, int colOff, int rows, int cols) {
        if (rowOff < 0 || colOff < 0 || rowOff + rows > this.rows || colOff + cols > this.cols) {
            throw new IllegalArgumentException("View bounds");
        }
        int newOffset = this.offset + rowOff * this.rowStride + colOff;
        return new FlatMatrixF(this.data, rows, cols, this.rowStride, newOffset);
    }

    public float get(int r, int c) {
        return data[offset + r * rowStride + c];
    }

    public void set(int r, int c, float v) {
        data[offset + r * rowStride + c] = v;
    }

    public byte[] asByteArray() {
        if (byteData == null) {
            byteData = new byte[rows * cols];
        }
        return byteData;
    }

    public static FlatMatrixF zeros(int r, int c) {
        return new FlatMatrixF(r, c);
    }

    // ====================== MATMUL ======================
    public static void matmul(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        matmul(A, B, C, null);
    }

    public static FlatMatrixF matmul(FlatMatrixF A, FlatMatrixF B) {
        checkDims(A, B);
        FlatMatrixF C = new FlatMatrixF(A.rows, B.cols);
        matmul(A, B, C, null);
        return C;
    }

    public static FlatMatrixF matmul(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C, ExecutorService executor) {
        checkDims(A, B, C);
        final int M = A.rows, K = A.cols, N = C.cols;
        final long flops = 2L * M * K * N;

        if (executor == null || flops < 200_000) {
            matmulSingleThread(A, B, C);
            return C;
        }

        final int nThreads = Math.min(Runtime.getRuntime().availableProcessors(), Math.max(1, M / 32));
        if (nThreads == 1) {
            matmulSingleThread(A, B, C);
            return C;
        }

        final int rowChunk = (M + nThreads - 1) / nThreads;
        List<Future<?>> futures = new ArrayList<>(nThreads);
        for (int t = 0; t < nThreads; t++) {
            final int rStart = t * rowChunk;
            final int rEnd = Math.min(rStart + rowChunk, M);
            if (rStart >= rEnd) {
                break;
            }
            futures.add(executor.submit(() -> {
                zeroMatrix(C, rStart, rEnd);
                matmulBlockOuter(A, B, C, rStart, rEnd);
            }));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (ExecutionException e) {
                throw new RuntimeException("Parallel matmul failed", e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted during matmul", e);
            }
        }
        return C;
    }

    private static void matmulSingleThread(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        zeroMatrix(C, 0, C.rows);
        matmulBlockOuter(A, B, C, 0, A.rows);
    }

    private static void zeroMatrix(FlatMatrixF C, int rowStart, int rowEnd) {
        if (C.isContiguous()) {
            int start = C.offset + rowStart * C.rowStride;
            int len = (rowEnd - rowStart) * C.rowStride;
            Arrays.fill(C.data, start, start + len, 0.0f);
        } else {
            for (int i = rowStart; i < rowEnd; i++) {
                for (int j = 0; j < C.cols; j++) {
                    C.set(i, j, 0.0f);
                }
            }
        }
    }

    private static void matmulBlockOuter(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C, int rStart, int rEnd) {
        final int K = A.cols, N = B.cols;
        for (int j0 = 0; j0 < N; j0 += NC) {
            int j1 = Math.min(j0 + NC, N);
            for (int k0 = 0; k0 < K; k0 += KC) {
                int k1 = Math.min(k0 + KC, K);
                for (int i = rStart; i < rEnd; i += MR) {
                    int i1 = Math.min(i + MR, rEnd);
                    for (int j = j0; j < j1; j += NR) {
                        int j1b = Math.min(j + NR, j1);
                        matmulMicrokernel(A, B, C, i, i1, j, j1b, k0, k1);
                    }
                }
            }
        }
    }

    private static void matmulMicrokernelOld(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C,
            int i0, int i1, int j0, int j1, int k0, int k1) {
        final int rows = i1 - i0;
        final int cols = j1 - j0;

        if (rows != MR || cols != NR || !A.isContiguous() || !B.isContiguous() || !C.isContiguous()) {
            scalarMicrokernel(A, B, C, i0, i1, j0, j1, k0, k1);
            return;
        }

        // === NO ALLOCATION - fully unrolled ===
        FloatVector acc0, acc1, acc2, acc3;

        if (k0 == 0) {
            acc0 = acc1 = acc2 = acc3 = FloatVector.zero(F_SPECIES);
        } else {
            final int baseC = C.offset + i0 * C.rowStride + j0;
            acc0 = FloatVector.fromArray(F_SPECIES, C.data, baseC);
            acc1 = FloatVector.fromArray(F_SPECIES, C.data, baseC + C.rowStride);
            acc2 = FloatVector.fromArray(F_SPECIES, C.data, baseC + 2 * C.rowStride);
            acc3 = FloatVector.fromArray(F_SPECIES, C.data, baseC + 3 * C.rowStride);
        }

        final int baseA0 = A.offset + i0 * A.rowStride;
        final int baseA1 = baseA0 + A.rowStride;
        final int baseA2 = baseA1 + A.rowStride;
        final int baseA3 = baseA2 + A.rowStride;
        final float[] aData = A.data;

        for (int k = k0; k < k1; k++) {
            var bvec = FloatVector.fromArray(F_SPECIES, B.data, B.offset + k * B.rowStride + j0);

            float a0 = aData[baseA0 + k];
            float a1 = aData[baseA1 + k];
            float a2 = aData[baseA2 + k];
            float a3 = aData[baseA3 + k];

            acc0 = acc0.fma(FloatVector.broadcast(F_SPECIES, a0), bvec);
            acc1 = acc1.fma(FloatVector.broadcast(F_SPECIES, a1), bvec);
            acc2 = acc2.fma(FloatVector.broadcast(F_SPECIES, a2), bvec);
            acc3 = acc3.fma(FloatVector.broadcast(F_SPECIES, a3), bvec);
        }

        final int baseC = C.offset + i0 * C.rowStride + j0;
        acc0.intoArray(C.data, baseC);
        acc1.intoArray(C.data, baseC + C.rowStride);
        acc2.intoArray(C.data, baseC + 2 * C.rowStride);
        acc3.intoArray(C.data, baseC + 3 * C.rowStride);
    }

    private static void scalarMicrokernelOld(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C,
            int i0, int i1, int j0, int j1, int k0, int k1) {

        final int rows = i1 - i0;
        final int cols = j1 - j0;
        final int blockSize = rows * cols;

        // 1. Reusable thread-local buffer extraction
        float[] cvals = SCALAR_ROW_BUFFER.get();
        if (cvals.length < blockSize) {
            cvals = new float[blockSize];
            SCALAR_ROW_BUFFER.set(cvals);
        }

        final int cBase = C.offset + i0 * C.rowStride + j0;
        final float[] cData = C.data;
        final float[] aData = A.data;
        final float[] bData = B.data;

        // ====================== LOAD ======================
        // Unconditionally fast: individual rows are always continuous memory spans
        int destIdx = 0;
        for (int r = 0; r < rows; r++) {
            System.arraycopy(cData, cBase + r * C.rowStride, cvals, destIdx, cols);
            destIdx += cols;
        }

        // ====================== ACCUMULATE ======================
        final int aOffsetBase = A.offset + i0 * A.rowStride;

        for (int k = k0; k < k1; k++) {
            final int bRowOffset = B.offset + k * B.rowStride + j0;

            for (int r = 0; r < rows; r++) {
                final float a = aData[aOffsetBase + r * A.rowStride + k];
                final int cIdxBase = r * cols;

                // Stride-1 execution layout -> Perfect target for HotSpot loop unrolling/SIMD
                for (int c = 0; c < cols; c++) {
                    cvals[cIdxBase + c] += a * bData[bRowOffset + c];
                }
            }
        }

        // ====================== STORE ======================
        int srcIdx = 0;
        for (int r = 0; r < rows; r++) {
            System.arraycopy(cvals, srcIdx, cData, cBase + r * C.rowStride, cols);
            srcIdx += cols;
        }
    }

    // ====================== ELEMENTWISE ======================
    /**
     * Slower than Bert, but MORE accurate GELU in-place using the Sigmoid-based
     * exact identity math. Matches standard Tanh approximation curves
     * perfectly.
     */
    public void geluInPlaceSigmoid() {
        int len = rows * cols;
        int idx = 0;

        if (HAS_VECTOR && isContiguous()) {
            // 2 * 0.79788456f = 1.59576912f
            final FloatVector V_C1 = FloatVector.broadcast(F_SPECIES, 1.59576912f);
            final FloatVector V_C2 = FloatVector.broadcast(F_SPECIES, 0.044715f);
            final FloatVector V_ONE = FloatVector.broadcast(F_SPECIES, 1.0f);

            for (; idx < F_SPECIES.loopBound(len); idx += VF_LEN) {
                var x = FloatVector.fromArray(F_SPECIES, this.data, this.offset + idx);

                var x3 = x.mul(x).mul(x);
                var z = x.add(x3.mul(V_C2)).mul(V_C1);

                // Compute the denominator: 1 + e^(-z)
                var denominator = V_ONE.add(fastExp(z.neg()));

                // Divide x directly by the denominator to get x * sigmoid(z)
                var result = x.div(denominator);

                result.intoArray(this.data, this.offset + idx);
            }
        }

        // Scalar fallback matching the exact Tanh/Sigmoid curve behavior
        for (; idx < len; idx++) {
            float x = this.data[this.offset + idx];
            float x3 = x * x * x;
            float t = (float) Math.tanh(0.79788456f * (x + 0.044715f * x3));
            this.data[this.offset + idx] = 0.5f * x * (1.0f + t);
        }
    }

    /**
     * Faster than sigmoid, but less accurate GELU in-place using the fast
     * BERT/OpenAI 1.702 * x approximation. Eliminates cubic calculations
     * entirely for maximum throughput.
     */
    public void geluInPlaceBert() {
        int len = rows * cols;
        int idx = 0;

        if (HAS_VECTOR && isContiguous()) {
            final FloatVector V_BERT_C = FloatVector.broadcast(F_SPECIES, 1.702f);
            final FloatVector V_ONE = FloatVector.broadcast(F_SPECIES, 1.0f);

            for (; idx < F_SPECIES.loopBound(len); idx += VF_LEN) {
                var x = FloatVector.fromArray(F_SPECIES, this.data, this.offset + idx);

                // z = 1.702 * x
                var z = x.mul(V_BERT_C);

                // sigmoid(z) = 1 / (1 + fastExp(-z))
                var denominator = V_ONE.add(fastExp(z.neg()));

                // x * sigmoid(z)
                var result = x.div(denominator);
                result.intoArray(this.data, this.offset + idx);
            }
        }

        // Scalar fallback matching the exact 1.702 sigmoid calculation
        for (; idx < len; idx++) {
            float x = this.data[this.offset + idx];
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-1.702f * x));
            this.data[this.offset + idx] = x * sigmoid;
        }
    }

    /**
     * Slower than Bert, but MORE accurate GeGLU in-place using the
     * Sigmoid-based exact identity math. Computes: output = value *
     * GELU_sigmoid(gate)
     *
     * @param gate The gating matrix (inputs[1])
     */
    public void gegluInPlaceSigmoid(FlatMatrixF gate) {
        if (rows != gate.rows || cols != gate.cols) {
            throw new IllegalArgumentException("Shape mismatch for GeGLU");
        }

        int len = rows * cols;
        int idx = 0;

        if (HAS_VECTOR && isContiguous() && gate.isContiguous()) {
            // 2 * 0.79788456f = 1.59576912f
            final FloatVector V_C1 = FloatVector.broadcast(F_SPECIES, 1.59576912f);
            final FloatVector V_C2 = FloatVector.broadcast(F_SPECIES, 0.044715f);
            final FloatVector V_ONE = FloatVector.broadcast(F_SPECIES, 1.0f);

            for (; idx < F_SPECIES.loopBound(len); idx += VF_LEN) {
                var value = FloatVector.fromArray(F_SPECIES, this.data, this.offset + idx);
                var g = FloatVector.fromArray(F_SPECIES, gate.data, gate.offset + idx);

                // z = 1.59576912 * (g + 0.044715 * g^3)
                var g3 = g.mul(g).mul(g);
                var z = g.add(g3.mul(V_C2)).mul(V_C1);

                // GELU(gate) = g / (1 + fastExp(-z))
                var denominator = V_ONE.add(fastExp(z.neg()));
                var geluGate = g.div(denominator);

                // value * GELU(gate)
                value.mul(geluGate).intoArray(this.data, this.offset + idx);
            }
        }

        // Scalar fallback matching the exact Tanh/Sigmoid curve behavior
        for (; idx < len; idx++) {
            float value = this.data[this.offset + idx];
            float g = gate.data[gate.offset + idx];

            float g3 = g * g * g;
            float t = (float) Math.tanh(0.79788456f * (g + 0.044715f * g3));
            float geluGate = 0.5f * g * (1.0f + t);

            this.data[this.offset + idx] = value * geluGate;
        }
    }

    /**
     * Faster than sigmoid, but less accurate GeGLU in-place using the fast
     * BERT/OpenAI 1.702 * g approximation. Computes: output = value *
     * GELU_bert(gate)
     *
     * @param gate The gating matrix (inputs[1])
     */
    public void gegluInPlaceBert(FlatMatrixF gate) {
        if (rows != gate.rows || cols != gate.cols) {
            throw new IllegalArgumentException("Shape mismatch for GeGLU");
        }

        int len = rows * cols;
        int idx = 0;

        if (HAS_VECTOR && isContiguous() && gate.isContiguous()) {
            final FloatVector V_BERT_C = FloatVector.broadcast(F_SPECIES, 1.702f);
            final FloatVector V_ONE = FloatVector.broadcast(F_SPECIES, 1.0f);

            for (; idx < F_SPECIES.loopBound(len); idx += VF_LEN) {
                var value = FloatVector.fromArray(F_SPECIES, this.data, this.offset + idx);
                var g = FloatVector.fromArray(F_SPECIES, gate.data, gate.offset + idx);

                // z = 1.702 * g
                var z = g.mul(V_BERT_C);

                // GELU(gate) = g / (1 + fastExp(-z))
                var denominator = V_ONE.add(fastExp(z.neg()));
                var geluGate = g.div(denominator);

                // value * GELU(gate)
                value.mul(geluGate).intoArray(this.data, this.offset + idx);
            }
        }

        // Scalar fallback matching the exact 1.702 sigmoid calculation
        for (; idx < len; idx++) {
            float value = this.data[this.offset + idx];
            float g = gate.data[gate.offset + idx];

            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-1.702f * g));
            float geluGate = g * sigmoid;

            this.data[this.offset + idx] = value * geluGate;
        }
    }

    public void gegluInPlace(FlatMatrixF gate) {
        if (rows != gate.rows || cols != gate.cols) {
            throw new IllegalArgumentException("Shape mismatch for GeGLU");
        }

        int len = rows * cols;
        int idx = 0;

        if (HAS_VECTOR && isContiguous() && gate.isContiguous()) {
            for (; idx < F_SPECIES.loopBound(len); idx += VF_LEN) {
                var value = FloatVector.fromArray(F_SPECIES, this.data, this.offset + idx);
                var g = FloatVector.fromArray(F_SPECIES, gate.data, gate.offset + idx);

                // GELU(gate)
                var g2 = g.mul(g);
                var g3 = g2.mul(g);
                var inner = g.add(g3.mul(V_GELU_C2)).mul(V_GELU_C1);
                var t = inner.lanewise(VectorOperators.TANH);
                var geluGate = g.mul(V_HALF).mul(t.add(V_ONE));

                // value * GELU(gate)
                value.mul(geluGate).intoArray(this.data, this.offset + idx);
            }
        }

        // Scalar fallback
        for (; idx < len; idx++) {
            float value = this.data[this.offset + idx];
            float g = gate.data[gate.offset + idx];

            float g3 = g * g * g;
            float t = (float) Math.tanh(GELU_C1 * (g + GELU_C2 * g3));
            float geluGate = 0.5f * g * (1.0f + t);

            this.data[this.offset + idx] = value * geluGate;
        }
    }

    /**
     * Resolves the flat array index for row-major layout with implicit unit
     * column stride. Highly optimized for aggressive HotSpot JIT compiler
     * inlining (< 35 bytes of bytecode).
     */
    public final int index(int r, int c) {
        return this.offset + (r * this.rowStride) + c;
    }

    public void geluInPlace(ExecutorService pool, int availableCores) {
        int len = this.rows * this.cols;

        // Path A: Globally Contiguous & Vector Supported
        if (HAS_VECTOR && this.isContiguous()) {
            ParallelTiledEngine.geluInPlaceParallel(pool, availableCores, this.data, this.offset, len);
        } else {
            // Path B: Strided/Sub-matrix view fallback.
            // Cache object fields into local registers to optimize loop invariant code motion (LCM)
            final int rStride = this.rowStride;
            final int off = this.offset;
            final int width = this.cols;
            final float[] d = this.data;

            for (int r = 0; r < this.rows; r++) {
                int startIdx = off + (r * rStride);
                int endIdx = startIdx + width;

                // Blazing fast sequential pointer loop over the row segment
                for (int idx = startIdx; idx < endIdx; idx++) {
                    float x = d[idx];

                    // Pure deterministic minimax scalar approximation
                    float inner = 0.79788456f * (x + 0.044715f * x * x * x);
                    float x2 = inner * inner;
                    float num = inner + x2 * inner * 0.1056f;
                    float den = 1.0f + x2 * 0.4324f;
                    float t = num / den;

                    if (inner > 3.0f) {
                        t = 1.0f;
                    } else if (inner < -3.0f) {
                        t = -1.0f;
                    }

                    d[idx] = 0.5f * x * (1.0f + t);
                }
            }
        }
    }

    /**
     * Checks if the matrix elements are structurally sequential from the
     * starting offset to the end of the data block.
     */
    public final boolean isContiguous() {
        return this.rowStride == this.cols;
    }

    /**
     * Checks if the starting memory pointer aligns perfectly with the SIMD
     * hardware register width to maximize native instruction efficiency.
     */
    public final boolean isVectorAligned() {
        return (this.offset % VF_LEN) == 0;
    }

    public void siluInPlace() {
        int len = rows * cols, idx = 0;
        if (HAS_VECTOR && isContiguous()) {
            for (; idx < F_SPECIES.loopBound(len); idx += VF_LEN) {
                var v = FloatVector.fromArray(F_SPECIES, data, offset + idx);
                var sigmoid = V_ONE.div(V_ONE.add(v.neg().lanewise(VectorOperators.EXP)));
                v.mul(sigmoid).intoArray(data, offset + idx);
            }
        }
        for (; idx < len; idx++) {
            float x = data[offset + idx];
            data[offset + idx] = x / (1.0f + (float) Math.exp(-x));
        }
    }

    public void reluInPlace() {
        int len = rows * cols, idx = 0;
        if (HAS_VECTOR && isContiguous()) {
            var zero = FloatVector.zero(F_SPECIES);
            for (; idx < F_SPECIES.loopBound(len); idx += VF_LEN) {
                FloatVector.fromArray(F_SPECIES, data, offset + idx).max(zero).intoArray(data, offset + idx);
            }
        }
        for (; idx < len; idx++) {
            data[offset + idx] = Math.max(0.0f, data[offset + idx]);
        }
    }

    public void softmaxRowsInPlace() {
        if (cols == 0) {
            return;
        }
        float[] row = new float[cols];
        for (int i = 0; i < rows; i++) {
            int base = offset + i * rowStride;
            System.arraycopy(data, base, row, 0, cols);
            softmaxInPlace(row);
            System.arraycopy(row, 0, data, base, cols);
        }
    }

    public static void softmaxInPlace(float[] x) {
        if (x.length == 0) {
            return;
        }
        float max = Float.NEGATIVE_INFINITY;
        int i = 0;
        if (HAS_VECTOR) {
            var vmax = FloatVector.broadcast(F_SPECIES, max);
            for (; i < F_SPECIES.loopBound(x.length); i += VF_LEN) {
                vmax = vmax.max(FloatVector.fromArray(F_SPECIES, x, i));
            }
            max = vmax.reduceLanes(VectorOperators.MAX);
        }
        for (; i < x.length; i++) {
            max = Math.max(max, x[i]);
        }

        float sum = 0.0f;
        i = 0;
        if (HAS_VECTOR) {
            var vmax = FloatVector.broadcast(F_SPECIES, max);
            var vsum = FloatVector.zero(F_SPECIES);
            for (; i < F_SPECIES.loopBound(x.length); i += VF_LEN) {
                var ev = FloatVector.fromArray(F_SPECIES, x, i).sub(vmax).lanewise(VectorOperators.EXP);
                ev.intoArray(x, i);
                vsum = vsum.add(ev);
            }
            sum = vsum.reduceLanes(VectorOperators.ADD);
        }
        for (; i < x.length; i++) {
            x[i] = (float) Math.exp(x[i] - max);
            sum += x[i];
        }

        final float invSum = 1.0f / sum;
        i = 0;
        if (HAS_VECTOR) {
            var vinv = FloatVector.broadcast(F_SPECIES, invSum);
            for (; i < F_SPECIES.loopBound(x.length); i += VF_LEN) {
                FloatVector.fromArray(F_SPECIES, x, i).mul(vinv).intoArray(x, i);
            }
        }
        for (; i < x.length; i++) {
            x[i] *= invSum;
        }
    }

    public static void add(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        int n = A.rows * A.cols;
        if (A.rows != B.rows || A.cols != B.cols || C.rows != A.rows || C.cols != A.cols) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        int i = 0;
        if (HAS_VECTOR && A.isContiguous() && B.isContiguous() && C.isContiguous()) {
            for (; i < F_SPECIES.loopBound(n); i += VF_LEN) {
                var va  = FloatVector.fromArray(F_SPECIES, A.data, A.offset + i);
                var vb = FloatVector.fromArray(F_SPECIES, B.data, B.offset + i);
                va.add(vb).intoArray(C.data, C.offset + i);
            }
        }
        for (; i < n; i++) {
            C.data[C.offset + i] = A.data[A.offset + i] + B.data[B.offset + i];
        }
    }

    public static void mul(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        int n = A.rows * A.cols;
        int i = 0;
        if (HAS_VECTOR && A.isContiguous() && B.isContiguous() && C.isContiguous()) {
            for (; i < F_SPECIES.loopBound(n); i += VF_LEN) {
                var va  = FloatVector.fromArray(F_SPECIES, A.data, A.offset + i);
                var vb = FloatVector.fromArray(F_SPECIES, B.data, B.offset + i);
                va.mul(vb).intoArray(C.data, C.offset + i);
            }
        }
        for (; i < n; i++) {
            C.data[C.offset + i] = A.data[A.offset + i] * B.data[B.offset + i];
        }
    }

    // ====================== FUSED KERNELS ======================
    public static void matmulAddBiasGelu(FlatMatrixF A, FlatMatrixF B, FlatMatrixF bias, FlatMatrixF C) {
        checkDims(A, B, C);
        if (bias.rows != 1 || bias.cols != B.cols) {
            throw new IllegalArgumentException("Bias must be 1xN");
        }
        final int M = A.rows, N = B.cols, K = A.cols;
        for (int i = 0; i < M; i++) {
            int j = 0;
            if (HAS_VECTOR && A.isContiguous() && B.isContiguous() && C.isContiguous() && bias.isContiguous()) {
                for (; j < F_SPECIES.loopBound(N); j += VF_LEN) {
                    var sum = FloatVector.fromArray(F_SPECIES, bias.data, bias.offset + j);
                    for (int k = 0; k < K; k++) {
                        var av = FloatVector.broadcast(F_SPECIES, A.get(i, k));
                        var bv = FloatVector.fromArray(F_SPECIES, B.data, B.offset + k * B.rowStride + j);
                        sum = av.fma(bv, sum);
                    }
                    var x2 = sum.mul(sum);
                    var inner = sum.add(x2.mul(sum).mul(V_GELU_C2)).mul(V_GELU_C1);
                    var t = inner.lanewise(VectorOperators.TANH);
                    sum.mul(V_HALF).mul(t.add(V_ONE)).intoArray(C.data, C.offset + i * C.rowStride + j);
                }
            }
            for (; j < N; j++) {
                float sum = bias.get(0, j);
                for (int k = 0; k < K; k++) {
                    sum += A.get(i, k) * B.get(k, j);
                }
                float x3 = sum * sum * sum;
                float t = (float) Math.tanh(GELU_C1 * (sum + GELU_C2 * x3));
                C.set(i, j, 0.5f * sum * (1.0f + t));
            }
        }
    }

    public static void matmulAddBiasRelu(FlatMatrixF a, FlatMatrixF b, FlatMatrixF bias, FlatMatrixF out) {
        matmul(a, b, out);
        add(out, bias, out);
        out.reluInPlace();
    }

    public static void matmulAddBiasSilu(FlatMatrixF a, FlatMatrixF b, FlatMatrixF bias, FlatMatrixF out) {
        matmul(a, b, out);
        add(out, bias, out);
        out.siluInPlace();
    }

    // ====================== FUSED MATMUL + alpha * sin(C) ======================
    /**
     * FUSED MATMUL + alpha * sin(C) C = A * B + alpha * sin(C)
     */
    public static void matmulAddSin(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C, float alpha) {
        checkDims(A, B, C);
        final int M = A.rows, N = B.cols, K = A.cols;
        final long flops = 2L * M * K * N;

        if (!HAS_VECTOR || flops < 16_384
                || !A.isContiguous() || !B.isContiguous() || !C.isContiguous()) {
            matmulAddSinScalar(A, B, C, alpha);
            return;
        }
        matmulAddSinSimd(A, B, C, alpha);
    }

    private static void matmulAddSinSimd(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C, float alpha) {
        final int M = A.rows, N = B.cols, K = A.cols;
        final FloatVector vAlpha = FloatVector.broadcast(F_SPECIES, alpha);
        final int loopBoundN = F_SPECIES.loopBound(N);

        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;

            int j = 0;
            for (; j < loopBoundN; j += VF_LEN) {
                var acc = FloatVector.zero(F_SPECIES);
                for (int k = 0; k < K; k++) {
                    var aVec = FloatVector.broadcast(F_SPECIES, A.data[rowA + k]);
                    var bVec = FloatVector.fromArray(F_SPECIES, B.data, B.offset + k * B.rowStride + j);
                    acc = aVec.fma(bVec, acc);
                }
                // Read old C for sin, then compute new value
                var cOld = FloatVector.fromArray(F_SPECIES, C.data, rowC + j);
                var sinC = cOld.lanewise(VectorOperators.SIN);
                acc.add(sinC.mul(vAlpha)).intoArray(C.data, rowC + j);
            }

            // Scalar tail
            for (; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A.data[rowA + k] * B.data[B.offset + k * B.rowStride + j];
                }
                final int idx = rowC + j;
                C.data[idx] = sum + alpha * (float) Math.sin(C.data[idx]);
            }
        }
    }

    private static void matmulAddSinScalar(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C, float alpha) {
        final int M = A.rows, N = B.cols, K = A.cols;
        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A.data[rowA + k] * B.data[B.offset + k * B.rowStride + j];
                }
                final int idx = rowC + j;
                C.data[idx] = sum + alpha * (float) Math.sin(C.data[idx]);
            }
        }
    }

// ====================== MATMUL IN-PLACE C += A * B ======================
    /**
     * MATMUL IN-PLACE (C += A * B) Caller must zero C if pure matmul is desired
     */
    public static void matmulInPlaceOld(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        checkDims(A, B, C);
        final int M = A.rows, N = B.cols, K = A.cols;
        final long flops = 2L * M * K * N;

        if (!HAS_VECTOR || flops < 16_384
                || !A.isContiguous() || !B.isContiguous() || !C.isContiguous()) {
            matmulInPlaceScalar(A, B, C);
            return;
        }
        matmulSimdBroadcast(A, B, C);
    }

    public static void matmulSimdBroadcast(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        checkDims(A, B, C);
        final int M = A.rows, N = B.cols, K = A.cols;
        final int loopBoundN = F_SPECIES.loopBound(N);

        // Note: Does NOT zero C — caller is responsible
        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;

            for (int k = 0; k < K; k++) {
                var aVec = FloatVector.broadcast(F_SPECIES, A.data[rowA + k]);
                final int rowB = B.offset + k * B.rowStride;
                int j = 0;
                for (; j < loopBoundN; j += VF_LEN) {
                    var bVec = FloatVector.fromArray(F_SPECIES, B.data, rowB + j);
                    var cVec = FloatVector.fromArray(F_SPECIES, C.data, rowC + j);
                    aVec.fma(bVec, cVec).intoArray(C.data, rowC + j);
                }
                // Scalar tail
                for (; j < N; j++) {
                    C.data[rowC + j] += A.data[rowA + k] * B.data[rowB + j];
                }
            }
        }
    }

    private static void matmulInPlaceScalar(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        final int M = A.rows, N = B.cols, K = A.cols;
        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;
            for (int k = 0; k < K; k++) {
                float a = A.data[rowA + k];
                final int rowB = B.offset + k * B.rowStride;
                for (int j = 0; j < N; j++) {
                    C.data[rowC + j] += a * B.data[rowB + j];
                }
            }
        }
    }

    public static void randomFill(FlatMatrixF A) {
        randomFill(A, 101);
    }//end method randomFill

    public static void randomFill(FlatMatrixF A, int max) {
        ThreadLocalRandom ran = ThreadLocalRandom.current();
        for (int i = 0; i < A.data.length; i++) {
            A.data[i] = 1 + ran.nextFloat(max);
        }
    }//end method randomFill

    public static void swiGLUOld(FlatMatrixF gate, FlatMatrixF up, FlatMatrixF out) {
        if (gate != out) {
            System.arraycopy(gate.data, gate.offset, out.data, out.offset, gate.rows * gate.cols);
        }
        out.siluInPlace();
        mul(out, up, out);
    }

    /**
     * out[i] = a[i] * b[i] for all elements. O(n) Handles aliasing: a==out,
     * b==out, or a==b are all safe.
     *
     * @throws IllegalArgumentException if sizes/offsets don't match
     * @param a
     * @param b
     * @param out
     */
    public static void mulElementwise(FlatMatrixF a, FlatMatrixF b, FlatMatrixF out) {
        final int n = a.rows * a.cols;
        if (n != b.rows * b.cols || n != out.rows * out.cols) {
            throw new IllegalArgumentException(
                    "Shape mismatch: a=%dx%d b=%dx%d out=%dx%d".formatted(
                            a.rows, a.cols, b.rows, b.cols, out.rows, out.cols));
        }

        final float[] ad = a.data;
        final float[] bd = b.data;
        final float[] od = out.data;
        final int ao = a.offset;
        final int bo = b.offset;
        final int oo = out.offset;
        final int vl = F_SPECIES.length();
        int i = 0;
        int upperBound = n - vl;

        // Vectorized main loop
        for (; i <= upperBound; i += vl) {
            FloatVector va  = FloatVector.fromArray(F_SPECIES, ad, ao + i);
            FloatVector vb = FloatVector.fromArray(F_SPECIES, bd, bo + i);
            va.mul(vb).intoArray(od, oo + i);
        }

        // Masked tail: handles n % vl!= 0 without branches in loop
        int remaining = n - i;
        if (remaining > 0) {
            var m = F_SPECIES.indexInRange(0, remaining);
            FloatVector va  = FloatVector.fromArray(F_SPECIES, ad, ao + i, m);
            FloatVector vb = FloatVector.fromArray(F_SPECIES, bd, bo + i, m);
            va.mul(vb).intoArray(od, oo + i, m);
        }
    }

    public static void swiGLU2point85ns(FlatMatrixF gate, FlatMatrixF up, FlatMatrixF out) {
        final int n = gate.rows * gate.cols;
        if (gate.rows != up.rows || gate.cols != up.cols) {
            throw new IllegalArgumentException("gate/up shape mismatch");
        }

        // 1. out = gate
        if (gate != out) {
            System.arraycopy(gate.data, gate.offset, out.data, out.offset, n);
        }

        // 2. out = silu(out) = x * sigmoid(x)
        final float[] od = out.data;
        final int oo = out.offset;
        final int vl = F_SPECIES.length();
        int i = 0;
        int upperBound = n - vl;

        for (; i <= upperBound; i += vl) {
            FloatVector x = FloatVector.fromArray(F_SPECIES, od, oo + i);
            FloatVector sig = sigmoid(x); // use your fast approx
            x.mul(sig).intoArray(od, oo + i);
        }
        // tail
        int remaining = n - i;
        if (remaining > 0) {
            var m = F_SPECIES.indexInRange(0, remaining);
            FloatVector x = FloatVector.fromArray(F_SPECIES, od, oo + i, m);
            FloatVector sig = sigmoid(x);
            x.mul(sig).intoArray(od, oo + i, m);
        }

        // 3. out = out * up -> elementwise, not GEMM
        mulElementwise(out, up, out);
    }

    // Flip the order here: 'up' (value path) comes first, matching 'x' in swiglu(x, y)
    /**
     *
     * @param up
     * @param gate
     * @param out
     */
    public static void swiGLU(FlatMatrixF up, FlatMatrixF gate, FlatMatrixF out) {
        final int n = gate.rows * gate.cols;

        // 1. Comprehensive Shape Validation
        if (gate.rows != up.rows || gate.cols != up.cols
                || gate.rows != out.rows || gate.cols != out.cols) {
            throw new IllegalArgumentException(
                    "Shape mismatch: gate=%dx%d, up=%dx%d, out=%dx%d".formatted(
                            gate.rows, gate.cols, up.rows, up.cols, out.rows, out.cols));
        }

        // 2. Memory Layout Validation
        if (!gate.isContiguous() || !up.isContiguous() || !out.isContiguous()) {
            throw new IllegalArgumentException("Vectorized swiGLU requires contiguous matrices.");
        }

        final float[] gd = gate.data;
        final float[] ud = up.data;
        final float[] od = out.data;
        final int go = gate.offset;
        final int uo = up.offset;
        final int oo = out.offset;

        final int vl = F_SPECIES.length();
        final int limit = F_SPECIES.loopBound(n);
        int i = 0;

        // 3. Fused Loop: Read gate/up once, compute, write once
        for (; i < limit; i += vl) {
            FloatVector g = FloatVector.fromArray(F_SPECIES, gd, go + i);
            FloatVector u = FloatVector.fromArray(F_SPECIES, ud, uo + i);

            // Perfectly matches: value * swish(gate)
            FloatVector sig = sigmoid(g);
            u.mul(g.mul(sig)).intoArray(od, oo + i);
        }

        // 4. Masked tail
        int remaining = n - i;
        if (remaining > 0) {
            var m = F_SPECIES.indexInRange(0, remaining);
            FloatVector g = FloatVector.fromArray(F_SPECIES, gd, go + i, m);
            FloatVector u = FloatVector.fromArray(F_SPECIES, ud, uo + i, m);

            FloatVector sig = sigmoid(g);
            u.mul(g.mul(sig)).intoArray(od, oo + i, m);
        }
    }

    public static void rmsNorm(FlatMatrixF x, FlatMatrixF weight, FlatMatrixF out, float eps) {
        int len = x.rows * x.cols;
        float sumSq = 0.0f;
        for (int i = 0; i < len; i++) {
            float v = x.data[x.offset + i];
            sumSq += v * v;
        }
        float invRms = 1.0f / (float) Math.sqrt(sumSq / len + eps);
        for (int i = 0; i < len; i++) {
            out.data[out.offset + i] = x.data[x.offset + i] * invRms * weight.data[weight.offset + i];
        }
    }

    public static void layerNorm(FlatMatrixF x, FlatMatrixF gamma, FlatMatrixF beta, FlatMatrixF out, float eps) {
        int len = x.rows * x.cols;
        float mean = 0.0f;
        for (int i = 0; i < len; i++) {
            mean += x.data[x.offset + i];
        }
        mean /= len;
        float var = 0.0f;
        for (int i = 0; i < len; i++) {
            float d = x.data[x.offset + i] - mean;
            var += d * d;
        }
        var /= len;
        float invStd = 1.0f / (float) Math.sqrt(var + eps);
        for (int i = 0; i < len; i++) {
            float n = (x.data[x.offset + i] - mean) * invStd;
            out.data[out.offset + i] = n * gamma.data[gamma.offset + i] + beta.data[beta.offset + i];
        }
    }

    private static void checkDims(FlatMatrixF A, FlatMatrixF B) {
        if (A.cols != B.rows) {
            throw new IllegalArgumentException("A.cols!= B.rows: " + A.cols + "!= " + B.rows);
        }
    }

    private static void checkDims(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        checkDims(A, B);
        if (C.rows != A.rows || C.cols != B.cols) {
            throw new IllegalArgumentException("C dims mismatch");
        }
    }

    /////////////////////////////////////MHA-ATTENTION////////////////////////////////////
    ///
    ///
    
    
    // ====================== MHA ATTENTION ======================
/**
 * Multi-head attention: output = softmax(Q * K^T / sqrt(d_k)) * V
 * Q: [seq_len, d_k], K: [seq_len, d_k], V: [seq_len, d_v], output: [seq_len, d_v]
 * scale should be 1.0f / sqrt(d_k)
 */
public static FlatMatrixF mhaAttention(FlatMatrixF Q, FlatMatrixF K, FlatMatrixF V,
            FlatMatrixF output, float scale) {
        return mhaAttention(Q, K, V, output, scale, null);
    }

    public static FlatMatrixF mhaAttention(FlatMatrixF Q, FlatMatrixF K, FlatMatrixF V,
            FlatMatrixF output, float scale, ExecutorService executor) {
        final int seqLen = Q.rows;
        final int d_k = Q.cols;
        final int d_v = V.cols;

        if (K.rows != seqLen || V.rows != seqLen || output.rows != seqLen || output.cols != d_v) {
            throw new IllegalArgumentException("MHA shape mismatch");
        }

        if (executor == null || seqLen < 32) {
            mhaAttentionSingleThread(Q, K, V, output, scale);
        } else {
            mhaAttentionParallel(Q, K, V, output, scale, executor);
        }
        return output;
    }

    /**
     *
     * @return a string representation of the matrix in rows and columns.
     */
    @Override
    public String toString() {
        String output = "\n";
        String appender = "";
        for (int row = 0; row < rows; row++) {

            for (int column = 0; column < cols; column++) {

                if (column < cols) {
                    appender += String.format("%7s%3s", data[row * cols + column], ",");
                }
                if (column == cols - 1) {
                    appender = appender.substring(0, appender.length() - 1);
                    appender += "          \n";
                }
                output += appender;
                appender = "";
            }
        }

        return output;
    }//end method toString

    private static void mhaAttentionSingleThread(FlatMatrixF Q, FlatMatrixF K, FlatMatrixF V,
            FlatMatrixF out, float scale) {
        final int seqLen = Q.rows;
        float[] scores = new float[seqLen];
        for (int i = 0; i < seqLen; i++) {
            qkT_row(Q, K, i, scores, scale);
            softmaxInPlace(scores);
            weightedSum(scores, V, out, i);
        }
    }

    private static void mhaAttentionParallel(FlatMatrixF Q, FlatMatrixF K, FlatMatrixF V,
            FlatMatrixF out, float scale, ExecutorService executor) {
        final int seqLen = Q.rows;
        final int nThreads = Math.min(Runtime.getRuntime().availableProcessors(), Math.max(1, seqLen / 8));
        final int chunk = (seqLen + nThreads - 1) / nThreads;
        List<Future<?>> futures = new ArrayList<>(nThreads);

        for (int t = 0; t < nThreads; t++) {
            final int start = t * chunk, end = Math.min(start + chunk, seqLen);
            if (start >= end) {
                break;
            }
            futures.add(executor.submit(() -> {
                float[] scores = new float[seqLen];
                for (int i = start; i < end; i++) {
                    qkT_row(Q, K, i, scores, scale);
                    softmaxInPlace(scores);
                    weightedSum(scores, V, out, i);
                }
            }));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (ExecutionException e) {
                throw new RuntimeException("MHA failed", e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted", e);
            }
        }
    }

// Q[i] @ K^T * scale
    private static void qkT_row(FlatMatrixF Q, FlatMatrixF K, int i, float[] scores, float scale) {
        final int d_k = Q.cols;
        final int seqLen = K.rows;
        for (int j = 0; j < seqLen; j++) {
            float dot = 0.0f;
            int k = 0;
            if (HAS_VECTOR && Q.isContiguous() && K.isContiguous()) {
                var sum = FloatVector.zero(F_SPECIES);
                int qOff = Q.offset + i * Q.rowStride;
                int kOff = K.offset + j * K.rowStride;
                for (; k < F_SPECIES.loopBound(d_k); k += VF_LEN) {
                    var qv = FloatVector.fromArray(F_SPECIES, Q.data, qOff + k);
                    var kv = FloatVector.fromArray(F_SPECIES, K.data, kOff + k);
                    sum = qv.fma(kv, sum);
                }
                dot = sum.reduceLanes(VectorOperators.ADD);
            }
            for (; k < d_k; k++) {
                dot += Q.get(i, k) * K.get(j, k);
            }
            scores[j] = dot * scale;
        }
    }

// output[i] = sum_j scores[j] * V[j]
    private static void weightedSum(float[] scores, FlatMatrixF V, FlatMatrixF out, int i) {
        final int d_v = V.cols;
        final int seqLen = V.rows;

        // Zero output row
        Arrays.fill(out.data, out.offset + i * out.rowStride, out.offset + i * out.rowStride + d_v, 0.0f);

        for (int j = 0; j < seqLen; j++) {
            float w = scores[j];
            if (w == 0.0f) {
                continue;
            }
            int k = 0;
            if (HAS_VECTOR && V.isContiguous() && out.isContiguous()) {
                var vw = FloatVector.broadcast(F_SPECIES, w);
                int vOff = V.offset + j * V.rowStride;
                int oOff = out.offset + i * out.rowStride;
                for (; k < F_SPECIES.loopBound(d_v); k += VF_LEN) {
                    var vv = FloatVector.fromArray(F_SPECIES, V.data, vOff + k);
                    var vo = FloatVector.fromArray(F_SPECIES, out.data, oOff + k);
                    vw.fma(vv, vo).intoArray(out.data, oOff + k);
                }
            }
            for (; k < d_v; k++) {
                out.set(i, k, out.get(i, k) + w * V.get(j, k));
            }
        }
    }

    /**
     * Fast exp approx. Max error ~2e-3 on [-10, 10] Based on Pade approx:
     * exp(x) = 2^(x * log2e)
     *
     * @param x
     */
    public static FloatVector fastExp(FloatVector x) {
        // Clamp input to avoid float underflow/overflow bounds [-88.0f, 88.0f]
        x = x.lanewise(VectorOperators.MAX, -88.0f)
                .lanewise(VectorOperators.MIN, 88.0f);

        final FloatVector LOG2E = FloatVector.broadcast(F_SPECIES, 1.44269504f);
        final FloatVector LN2 = FloatVector.broadcast(F_SPECIES, 0.69314718f);
        final FloatVector ONE = FloatVector.broadcast(F_SPECIES, 1.0f);

        // x * log2(e)
        var fx = x.mul(LOG2E);

        // STEP 1: Perform manual floor via F2I truncation toward zero
        IntVector ni = (IntVector) fx.convertShape(VectorOperators.F2I, I_SPECIES, 0);
        FloatVector ni_f = (FloatVector) ni.convertShape(VectorOperators.I2F, F_SPECIES, 0);

        // For negative values, truncation goes the wrong way. 
        // If fx < ni_f, we must subtract 1 from the integer vector.
        var needsDec = fx.compare(VectorOperators.LT, ni_f);
        var oneInt = IntVector.broadcast(I_SPECIES, 1);

        // CORRECT FIX: Cast the Float mask to an Integer mask matching I_SPECIES
        VectorMask<Integer> intMask = needsDec.cast(I_SPECIES);
        ni = ni.sub(oneInt, intMask);

        // STEP 2: Recompute the corrected float representation of the floor
        ni_f = (FloatVector) ni.convertShape(VectorOperators.I2F, F_SPECIES, 0);

        // Fractional part is now guaranteed to be in [0, 1)
        var f = fx.sub(ni_f);

        // STEP 3: 4th-order Polynomial approximation for 2^f
        var y = f.mul(LN2);
        var y2 = y.mul(y);
        var y3 = y2.mul(y);
        var y4 = y3.mul(y);

        var poly = ONE.add(y)
                .add(y2.mul(0.5f))
                .add(y3.mul(0.16666667f))
                .add(y4.mul(0.041666668f));

        // STEP 4: 2^ni via raw bit manipulation: (ni + 127) << 23
        var biased = ni.add(127);
        var bits = biased.lanewise(VectorOperators.LSHL, 23);

        // Pure bit-cast from integer registers to float registers
        FloatVector pow2n = bits.reinterpretAsFloats();

        return poly.mul(pow2n);
    }

    /**
     * Fast sigmoid: 1 / (1 + exp(-x)) Max error ~1e-3. ~3x faster than Math.exp
     * scalar. Uses expm1 + rational approx to stay stable.
     *
     * @param x
     * @return
     */
    public static FloatVector sigmoid(FloatVector x) {
        final FloatVector ONE = FloatVector.broadcast(F_SPECIES, 1.0f);
        var negX = x.neg();
        return ONE.div(ONE.add(fastExp(negX)));
    }


    //////////////////////////////MHA-ATTENTION-DONE/////////////////////////////////////////////
    
    
    
    
    
    
    
    
    

  

    /**
     * MATMUL IN-PLACE: C += A * B
     * Implements a cache-blocked microkernel architecture to maintain steady data throughput.
     */
    public static void matmulInPlace(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        checkDims(A, B, C);

        // If data layouts are non-contiguous or dimensions are trivial, execute flat scalar path
        if (A.rows < MR || B.cols < NR || !A.isContiguous() || !B.isContiguous() || !C.isContiguous()) {
            matmulInPlaceScalarFallback(A, B, C);
            return;
        }
        
        final int KC = 128;

        final int M = A.rows;
        final int K = A.cols;
        final int N = B.cols;

        // 3-Tier Cache Blocking Macro Loops (NC -> KC -> MC)
        for (int jc = 0; jc < N; jc += NC) {
            int jEnd = Math.min(jc + NC, N);

            for (int kc = 0; kc < K; kc += KC) {
                int kEnd = Math.min(kc + KC, K);

                for (int ic = 0; ic < M; ic += MC) {
                    int iEnd = Math.min(ic + MC, M);

                    // Execute register-blocked microkernels inside the cache tiles
                    for (int i = ic; i < iEnd; i += MR) {
                        int rBlock = Math.min(MR, iEnd - i);

                        for (int j = jc; j < jEnd; j += NR) {
                            int cBlock = Math.min(NR, jEnd - j);

                            matmulMicrokernel(A, B, C, i, i + rBlock, j, j + cBlock, kc, kEnd);
                        }
                    }
                }
            }
        }
    }

    /**
     * Core Register-Hoisted Vector Microkernel
     */
    private static void matmulMicrokernel(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C,
                                          int i0, int i1, int j0, int j1, int k0, int k1) {
        final int rows = i1 - i0;
        final int cols = j1 - j0;

       
        // Trap partial edge fragments and redirect to high-performance scalar fallback
        if (rows != MR || cols != NR) {
            scalarMicrokernel(A, B, C, i0, i1, j0, j1, k0, k1);
            return;
        }

        // Hoist accumulation target directly into CPU SIMD registers
        final int baseC = C.offset + i0 * C.rowStride + j0;
        FloatVector acc0 = FloatVector.fromArray(F_SPECIES, C.data, baseC);
        FloatVector acc1 = FloatVector.fromArray(F_SPECIES, C.data, baseC + C.rowStride);
        FloatVector acc2 = FloatVector.fromArray(F_SPECIES, C.data, baseC + 2 * C.rowStride);
        FloatVector acc3 = FloatVector.fromArray(F_SPECIES, C.data, baseC + 3 * C.rowStride);

        final int baseA0 = A.offset + i0 * A.rowStride;
        final int baseA1 = baseA0 + A.rowStride;
        final int baseA2 = baseA1 + A.rowStride;
        final int baseA3 = baseA2 + A.rowStride;
        final float[] aData = A.data;
        final float[] bData = B.data;

        // Tight inner loop: Interleaves FMA operations completely within registers
        for (int k = k0; k < k1; k++) {
            var bVec = FloatVector.fromArray(F_SPECIES, bData, B.offset + k * B.rowStride + j0);

            acc0 = acc0.fma(FloatVector.broadcast(F_SPECIES, aData[baseA0 + k]), bVec);
            acc1 = acc1.fma(FloatVector.broadcast(F_SPECIES, aData[baseA1 + k]), bVec);
            acc2 = acc2.fma(FloatVector.broadcast(F_SPECIES, aData[baseA2 + k]), bVec);
            acc3 = acc3.fma(FloatVector.broadcast(F_SPECIES, aData[baseA3 + k]), bVec);
        }

        // Flush register states to main memory structure exactly once at completion
        acc0.intoArray(C.data, baseC);
        acc1.intoArray(C.data, baseC + C.rowStride);
        acc2.intoArray(C.data, baseC + 2 * C.rowStride);
        acc3.intoArray(C.data, baseC + 3 * C.rowStride);
    }

    /**
     * Stride-1 Optimized Scalar Edge Fallback Block
     */
    private static void scalarMicrokernel(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C,
                                          int i0, int i1, int j0, int j1, int k0, int k1) {
        final int rows = i1 - i0;
        final int cols = j1 - j0;
        final int blockSize = rows * cols;

       
        float[] cvals = SCALAR_ROW_BUFFER.get();
        if (cvals.length < blockSize) {
            cvals = new float[blockSize];
            SCALAR_ROW_BUFFER.set(cvals);
        }

        final int cBase = C.offset + i0 * C.rowStride + j0;
        final float[] cData = C.data;
        final float[] aData = A.data;
        final float[] bData = B.data;

        int destIdx = 0;
        for (int r = 0; r < rows; r++) {
            System.arraycopy(cData, cBase + r * C.rowStride, cvals, destIdx, cols);
            destIdx += cols;
        }

        final int aOffsetBase = A.offset + i0 * A.rowStride;
        for (int k = k0; k < k1; k++) {
            final int bRowOffset = B.offset + k * B.rowStride + j0;
            for (int r = 0; r < rows; r++) {
                final float a = aData[aOffsetBase + r * A.rowStride + k];
                final int cIdxBase = r * cols;

                for (int c = 0; c < cols; c++) {
                    cvals[cIdxBase + c] += a * bData[bRowOffset + c];
                }
            }
        }

        int srcIdx = 0;
        for (int r = 0; r < rows; r++) {
            System.arraycopy(cvals, srcIdx, cData, cBase + r * C.rowStride, cols);
            srcIdx += cols;
        }
    }

    /**
     * General un-blocked matrix multiplier fallback for trivial tasks or unaligned layouts.
     */
    private static void matmulInPlaceScalarFallback(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        final int M = A.rows;
        final int K = A.cols;
        final int N = B.cols;

        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float aVal = A.get(i, k);
                for (int j = 0; j < N; j++) {
                    C.set(i, j, C.get(i, j) + aVal * B.get(k, j));
                }
            }
        }
    }    
    
    
    
    
    
    
    
    
    
    
    
}
