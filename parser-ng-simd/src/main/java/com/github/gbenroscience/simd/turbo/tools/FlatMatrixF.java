package com.github.gbenroscience.simd.turbo.tools;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;



import java.util.*;
import java.util.concurrent.*;
import jdk.incubator.vector.*;
/**
 *
 * @author GBEMIRO
 */ 
public final class FlatMatrixF {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VLEN = F_SPECIES.length();
    private static final boolean HAS_VECTOR = VLEN > 1;

    // GELU constants float
    private static final float GELU_C1 = 0.7978845608028654f;
    private static final float GELU_C2 = 0.044715f;
    private static final FloatVector V_GELU_C1 = FloatVector.broadcast(F_SPECIES, GELU_C1);
    private static final FloatVector V_GELU_C2 = FloatVector.broadcast(F_SPECIES, GELU_C2);
    private static final FloatVector V_HALF = FloatVector.broadcast(F_SPECIES, 0.5f);
    private static final FloatVector V_ONE = FloatVector.broadcast(F_SPECIES, 1.0f);

    // Block sizes for float - NR matches float lanes
    private static final int NC = 256, KC = 256, MR = 4;
    private static final int NR = F_SPECIES.length(); // 8 for AVX2, 16 for AVX512

    public final float[] data;
    public final int rows, cols;
    public final int rowStride;
    public final int offset;
    private byte[] byteData; // lazy backing for Q8

    public FlatMatrixF(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.rowStride = cols;
        this.offset = 0;
        this.data = new float[rows * cols];
    }

    public FlatMatrixF(int rows, int cols, float[] data) {
        if (data.length!= rows * cols) {
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

    public boolean isContiguous() {
        return rowStride == cols && offset % VLEN == 0;
    }

    public byte[] asByteArray() {
        if (byteData == null) byteData = new byte[rows * cols];
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
            if (rStart >= rEnd) break;
            futures.add(executor.submit(() -> {
                zeroMatrix(C, rStart, rEnd);
                matmulBlockOuter(A, B, C, rStart, rEnd);
            }));
        }
        for (Future<?> f : futures) {
            try { f.get(); }
            catch (ExecutionException e) { throw new RuntimeException("Parallel matmul failed", e.getCause()); }
            catch (InterruptedException e) {
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
            for (int i = rowStart; i < rowEnd; i++)
                for (int j = 0; j < C.cols; j++)
                    C.set(i, j, 0.0f);
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

    private static void matmulMicrokernel(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C,
            int i0, int i1, int j0, int j1, int k0, int k1) {
        final int rows = i1 - i0;
        final int cols = j1 - j0;

        if (rows!= MR || cols!= NR ||!A.isContiguous() ||!B.isContiguous() ||!C.isContiguous()) {
            scalarMicrokernel(A, B, C, i0, i1, j0, j1, k0, k1);
            return;
        }

        FloatVector[] acc = new FloatVector[MR];
        if (k0 == 0) {
            for (int r = 0; r < MR; r++) acc[r] = FloatVector.zero(F_SPECIES);
        } else {
            for (int r = 0; r < MR; r++)
                acc[r] = FloatVector.fromArray(F_SPECIES, C.data, C.offset + (i0 + r) * C.rowStride + j0);
        }

        final int baseA0 = A.offset + i0 * A.rowStride;
        final int baseA1 = baseA0 + A.rowStride;
        final int baseA2 = baseA1 + A.rowStride;
        final int baseA3 = baseA2 + A.rowStride;
        final float[] aData = A.data;

        for (int k = k0; k < k1; k++) {
            var bvec = FloatVector.fromArray(F_SPECIES, B.data, B.offset + k * B.rowStride + j0);
            acc[0] = acc[0].fma(FloatVector.broadcast(F_SPECIES, aData[baseA0 + k]), bvec);
            acc[1] = acc[1].fma(FloatVector.broadcast(F_SPECIES, aData[baseA1 + k]), bvec);
            acc[2] = acc[2].fma(FloatVector.broadcast(F_SPECIES, aData[baseA2 + k]), bvec);
            acc[3] = acc[3].fma(FloatVector.broadcast(F_SPECIES, aData[baseA3 + k]), bvec);
        }

        for (int r = 0; r < MR; r++)
            acc[r].intoArray(C.data, C.offset + (i0 + r) * C.rowStride + j0);
    }

    private static void scalarMicrokernel(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C,
            int i0, int i1, int j0, int j1, int k0, int k1) {
        final int rows = i1 - i0;
        final int cols = j1 - j0;
        float[] cvals = new float[rows * cols];

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                cvals[r * cols + c] = C.get(i0 + r, j0 + c);

        for (int k = k0; k < k1; k++)
            for (int r = 0; r < rows; r++) {
                float a = A.get(i0 + r, k);
                for (int c = 0; c < cols; c++)
                    cvals[r * cols + c] += a * B.get(k, j0 + c);
            }

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                C.set(i0 + r, j0 + c, cvals[r * cols + c]);
    }

    // ====================== ELEMENTWISE ======================
    public void geluInPlace() {
        int len = rows * cols, idx = 0;
        if (HAS_VECTOR && isContiguous()) {
            for (; idx < F_SPECIES.loopBound(len); idx += VLEN) {
                var v = FloatVector.fromArray(F_SPECIES, data, offset + idx);
                var x2 = v.mul(v);
                var inner = v.add(x2.mul(v).mul(V_GELU_C2)).mul(V_GELU_C1);
                var t = inner.lanewise(VectorOperators.TANH);
                v.mul(V_HALF).mul(t.add(V_ONE)).intoArray(data, offset + idx);
            }
        }
        for (; idx < len; idx++) {
            float x = data[offset + idx];
            float x3 = x * x * x;
            float t = (float) Math.tanh(GELU_C1 * (x + GELU_C2 * x3));
            data[offset + idx] = 0.5f * x * (1.0f + t);
        }
    }

    public void siluInPlace() {
        int len = rows * cols, idx = 0;
        if (HAS_VECTOR && isContiguous()) {
            for (; idx < F_SPECIES.loopBound(len); idx += VLEN) {
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
            for (; idx < F_SPECIES.loopBound(len); idx += VLEN)
                FloatVector.fromArray(F_SPECIES, data, offset + idx).max(zero).intoArray(data, offset + idx);
        }
        for (; idx < len; idx++) data[offset + idx] = Math.max(0.0f, data[offset + idx]);
    }

    public void softmaxRowsInPlace() {
        if (cols == 0) return;
        float[] row = new float[cols];
        for (int i = 0; i < rows; i++) {
            int base = offset + i * rowStride;
            System.arraycopy(data, base, row, 0, cols);
            softmaxInPlace(row);
            System.arraycopy(row, 0, data, base, cols);
        }
    }

    public static void softmaxInPlace(float[] x) {
        if (x.length == 0) return;
        float max = Float.NEGATIVE_INFINITY;
        int i = 0;
        if (HAS_VECTOR) {
            var vmax = FloatVector.broadcast(F_SPECIES, max);
            for (; i < F_SPECIES.loopBound(x.length); i += VLEN)
                vmax = vmax.max(FloatVector.fromArray(F_SPECIES, x, i));
            max = vmax.reduceLanes(VectorOperators.MAX);
        }
        for (; i < x.length; i++) max = Math.max(max, x[i]);

        float sum = 0.0f;
        i = 0;
        if (HAS_VECTOR) {
            var vmax = FloatVector.broadcast(F_SPECIES, max);
            var vsum = FloatVector.zero(F_SPECIES);
            for (; i < F_SPECIES.loopBound(x.length); i += VLEN) {
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
            for (; i < F_SPECIES.loopBound(x.length); i += VLEN)
                FloatVector.fromArray(F_SPECIES, x, i).mul(vinv).intoArray(x, i);
        }
        for (; i < x.length; i++) x[i] *= invSum;
    }

    public static void add(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        int n = A.rows * A.cols;
        if (A.rows!= B.rows || A.cols!= B.cols || C.rows!= A.rows || C.cols!= A.cols)
            throw new IllegalArgumentException("Shape mismatch");
        int i = 0;
        if (HAS_VECTOR && A.isContiguous() && B.isContiguous() && C.isContiguous()) {
            for (; i < F_SPECIES.loopBound(n); i += VLEN) {
                var va = FloatVector.fromArray(F_SPECIES, A.data, A.offset + i);
                var vb = FloatVector.fromArray(F_SPECIES, B.data, B.offset + i);
                va.add(vb).intoArray(C.data, C.offset + i);
            }
        }
        for (; i < n; i++) C.data[C.offset + i] = A.data[A.offset + i] + B.data[B.offset + i];
    }

    public static void mul(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        int n = A.rows * A.cols;
        int i = 0;
        if (HAS_VECTOR && A.isContiguous() && B.isContiguous() && C.isContiguous()) {
            for (; i < F_SPECIES.loopBound(n); i += VLEN) {
                var va = FloatVector.fromArray(F_SPECIES, A.data, A.offset + i);
                var vb = FloatVector.fromArray(F_SPECIES, B.data, B.offset + i);
                va.mul(vb).intoArray(C.data, C.offset + i);
            }
        }
        for (; i < n; i++) C.data[C.offset + i] = A.data[A.offset + i] * B.data[B.offset + i];
    }

    // ====================== FUSED KERNELS ======================
    public static void matmulAddBiasGelu(FlatMatrixF A, FlatMatrixF B, FlatMatrixF bias, FlatMatrixF C) {
        checkDims(A, B, C);
        if (bias.rows!= 1 || bias.cols!= B.cols) throw new IllegalArgumentException("Bias must be 1xN");
        final int M = A.rows, N = B.cols, K = A.cols;
        for (int i = 0; i < M; i++) {
            int j = 0;
            if (HAS_VECTOR && A.isContiguous() && B.isContiguous() && C.isContiguous() && bias.isContiguous()) {
                for (; j < F_SPECIES.loopBound(N); j += VLEN) {
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
                for (int k = 0; k < K; k++) sum += A.get(i, k) * B.get(k, j);
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
 * FUSED MATMUL + alpha * sin(C)
 * C = A * B + alpha * sin(C)
 */
public static void matmulAddSin(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C, float alpha) {
    checkDims(A, B, C);
    final int M = A.rows, N = B.cols, K = A.cols;
    final long flops = 2L * M * K * N;

    if (!HAS_VECTOR || flops < 16_384 ||
       !A.isContiguous() ||!B.isContiguous() ||!C.isContiguous()) {
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
        for (; j < loopBoundN; j += VLEN) {
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
 * MATMUL IN-PLACE (C += A * B)
 * Caller must zero C if pure matmul is desired
 */
public static void matmulInPlace(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
    checkDims(A, B, C);
    final int M = A.rows, N = B.cols, K = A.cols;
    final long flops = 2L * M * K * N;

    if (!HAS_VECTOR || flops < 16_384 ||
       !A.isContiguous() ||!B.isContiguous() ||!C.isContiguous()) {
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
            for (; j < loopBoundN; j += VLEN) {
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

    public static void swiGLU(FlatMatrixF gate, FlatMatrixF up, FlatMatrixF out) {
        if (gate!= out) System.arraycopy(gate.data, gate.offset, out.data, out.offset, gate.rows * gate.cols);
        out.siluInPlace();
        mul(out, up, out);
    }

    public static void rmsNorm(FlatMatrixF x, FlatMatrixF weight, FlatMatrixF out, float eps) {
        int len = x.rows * x.cols;
        float sumSq = 0.0f;
        for (int i = 0; i < len; i++) {
            float v = x.data[x.offset + i];
            sumSq += v * v;
        }
        float invRms = 1.0f / (float) Math.sqrt(sumSq / len + eps);
        for (int i = 0; i < len; i++)
            out.data[out.offset + i] = x.data[x.offset + i] * invRms * weight.data[weight.offset + i];
    }

    public static void layerNorm(FlatMatrixF x, FlatMatrixF gamma, FlatMatrixF beta, FlatMatrixF out, float eps) {
        int len = x.rows * x.cols;
        float mean = 0.0f;
        for (int i = 0; i < len; i++) mean += x.data[x.offset + i];
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
        if (A.cols!= B.rows) throw new IllegalArgumentException("A.cols!= B.rows: " + A.cols + "!= " + B.rows);
    }

    private static void checkDims(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        checkDims(A, B);
        if (C.rows!= A.rows || C.cols!= B.cols) throw new IllegalArgumentException("C dims mismatch");
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

    if (K.rows!= seqLen || V.rows!= seqLen || output.rows!= seqLen || output.cols!= d_v) {
        throw new IllegalArgumentException("MHA shape mismatch");
    }

    if (executor == null || seqLen < 32) {
        mhaAttentionSingleThread(Q, K, V, output, scale);
    } else {
        mhaAttentionParallel(Q, K, V, output, scale, executor);
    }
    return output;
}

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
        if (start >= end) break;
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
        try { f.get(); }
        catch (ExecutionException e) { throw new RuntimeException("MHA failed", e.getCause()); }
        catch (InterruptedException e) {
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
            for (; k < F_SPECIES.loopBound(d_k); k += VLEN) {
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
        if (w == 0.0f) continue;
        int k = 0;
        if (HAS_VECTOR && V.isContiguous() && out.isContiguous()) {
            var vw = FloatVector.broadcast(F_SPECIES, w);
            int vOff = V.offset + j * V.rowStride;
            int oOff = out.offset + i * out.rowStride;
            for (; k < F_SPECIES.loopBound(d_v); k += VLEN) {
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
//////////////////////////////MHA-ATTENTION-DONE/////////////////////////////////////////////
}