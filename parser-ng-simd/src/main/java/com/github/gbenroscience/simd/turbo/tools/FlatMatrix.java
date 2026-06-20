package com.github.gbenroscience.simd.turbo.tools;

import java.util.*;
import java.util.concurrent.*;
import jdk.incubator.vector.*;

public class FlatMatrix {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int VLEN = SPECIES.length();
    private static final boolean HAS_VECTOR = VLEN > 1;

    // GELU constants
    private static final double GELU_C1 = 0.7978845608028654;
    private static final double GELU_C2 = 0.044715;
    private static final DoubleVector V_GELU_C1 = DoubleVector.broadcast(SPECIES, GELU_C1);
    private static final DoubleVector V_GELU_C2 = DoubleVector.broadcast(SPECIES, GELU_C2);
    private static final DoubleVector V_HALF = DoubleVector.broadcast(SPECIES, 0.5);
    private static final DoubleVector V_ONE = DoubleVector.broadcast(SPECIES, 1.0);

    // Block sizes for cache-friendly tiling
    private static final int NC = 256, KC = 256, MR = 4;
    private static final int NR = SPECIES.length();

    public final double[] data;
    public final int rows, cols;
    public final int rowStride;
    public final int offset;

    private byte[] byteData; // Lazy backing for Q8 quantization

    public FlatMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.rowStride = cols;
        this.offset = 0;
        this.data = new double[rows * cols];
    }

    public FlatMatrix(int rows, int cols, double[] data) {
        if (data.length != rows * cols) {
            throw new IllegalArgumentException("Data length mismatch");
        }
        this.rows = rows;
        this.cols = cols;
        this.data = data;
        this.rowStride = cols;
        this.offset = 0;
    }

    private FlatMatrix(double[] data, int rows, int cols, int rowStride, int offset) {
        this.data = data;
        this.rows = rows;
        this.cols = cols;
        this.rowStride = rowStride;
        this.offset = offset;
    }

    public FlatMatrix view(int rowOff, int colOff, int rows, int cols) {
        if (rowOff < 0 || colOff < 0 || rowOff + rows > this.rows || colOff + cols > this.cols) {
            throw new IllegalArgumentException("View bounds overflow");
        }
        int newOffset = this.offset + rowOff * this.rowStride + colOff;
        return new FlatMatrix(this.data, rows, cols, this.rowStride, newOffset);
    }

    public double get(int r, int c) {
        return data[offset + r * rowStride + c];
    }

    public void set(int r, int c, double v) {
        data[offset + r * rowStride + c] = v;
    }

    public boolean isContiguous() {
        return rowStride == cols && offset % VLEN == 0;
    }

    public static FlatMatrix zeros(int r, int c) {
        return new FlatMatrix(r, c);
    }

    // ====================== MATMUL ENTRYPOINTS ======================
    private static void matmulSingleThread(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        zeroMatrix(C, 0, C.rows);
        matmulBlockOuter(A, B, C, 0, A.rows);
    }

    public static FlatMatrix matmul(FlatMatrix A, FlatMatrix B, FlatMatrix C, ExecutorService executor) {
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
            try {
                f.get();
            } catch (ExecutionException e) {
                throw new RuntimeException("Parallel matmul execution error", e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Matmul thread interrupted", e);
            }
        }
        return C;
    }

    public static FlatMatrix matmul(FlatMatrix A, FlatMatrix B) {
        checkDims(A, B);
        FlatMatrix C = new FlatMatrix(A.rows, B.cols);
        matmulSingleThread(A, B, C);
        return C;
    }

    public static void matmul(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        matmul(A, B, C, null);
    }

    public static FlatMatrix matmul(FlatMatrix A, FlatMatrix B, ExecutorService executor) {
        checkDims(A, B);
        FlatMatrix C = new FlatMatrix(A.rows, B.cols);
        return matmul(A, B, C, executor);
    }

    private static void zeroMatrix(FlatMatrix C, int rowStart, int rowEnd) {
        if (C.isContiguous()) {
            int start = C.offset + rowStart * C.rowStride;
            int len = (rowEnd - rowStart) * C.rowStride;
            Arrays.fill(C.data, start, start + len, 0.0);
        } else {
            for (int i = rowStart; i < rowEnd; i++) {
                int start = C.offset + i * C.rowStride;
                Arrays.fill(C.data, start, start + C.cols, 0.0);
            }
        }
    }

    private static void matmulBlockOuter(FlatMatrix A, FlatMatrix B, FlatMatrix C, int rStart, int rEnd) {
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

    private static void matmulMicrokernel(FlatMatrix A, FlatMatrix B, FlatMatrix C,
                                          int i0, int i1, int j0, int j1, int k0, int k1) {
        final int rows = i1 - i0;
        final int cols = j1 - j0;

        if (rows != MR || cols != NR || !A.isContiguous() || !B.isContiguous() || !C.isContiguous()) {
            scalarMicrokernel(A, B, C, i0, i1, j0, j1, k0, k1);
            return;
        }

        DoubleVector[] acc = new DoubleVector[MR];

        if (k0 == 0) {
            for (int r = 0; r < MR; r++) {
                acc[r] = DoubleVector.zero(SPECIES);
            }
        } else {
            for (int r = 0; r < MR; r++) {
                acc[r] = DoubleVector.fromArray(SPECIES, C.data, C.offset + (i0 + r) * C.rowStride + j0);
            }
        }

        final int baseA0 = A.offset + i0 * A.rowStride;
        final int baseA1 = baseA0 + A.rowStride;
        final int baseA2 = baseA1 + A.rowStride;
        final int baseA3 = baseA2 + A.rowStride;
        final double[] aData = A.data;

        for (int k = k0; k < k1; k++) {
            var bvec = DoubleVector.fromArray(SPECIES, B.data, B.offset + k * B.rowStride + j0);

            acc[0] = acc[0].fma(DoubleVector.broadcast(SPECIES, aData[baseA0 + k]), bvec);
            acc[1] = acc[1].fma(DoubleVector.broadcast(SPECIES, aData[baseA1 + k]), bvec);
            acc[2] = acc[2].fma(DoubleVector.broadcast(SPECIES, aData[baseA2 + k]), bvec);
            acc[3] = acc[3].fma(DoubleVector.broadcast(SPECIES, aData[baseA3 + k]), bvec);
        }

        for (int r = 0; r < MR; r++) {
            acc[r].intoArray(C.data, C.offset + (i0 + r) * C.rowStride + j0);
        }
    }

    private static void scalarMicrokernel(FlatMatrix A, FlatMatrix B, FlatMatrix C,
                                          int i0, int i1, int j0, int j1, int k0, int k1) {
        final int rows = i1 - i0;
        final int cols = j1 - j0;

        for (int k = k0; k < k1; k++) {
            for (int r = 0; r < rows; r++) {
                double a = A.get(i0 + r, k);
                int cRowOff = C.offset + (i0 + r) * C.rowStride + j0;
                int bRowOff = B.offset + k * B.rowStride + j0;
                for (int c = 0; c < cols; c++) {
                    C.data[cRowOff + c] += a * B.data[bRowOff + c];
                }
            }
        }
    }

    // ====================== SOFTMAX ======================
    public void softmaxRowsInPlace() {
        if (cols == 0) return;
        double[] rowScratchBuffer = new double[cols];
        for (int i = 0; i < rows; i++) {
            int rowStartIdx = offset + i * rowStride;
            System.arraycopy(data, rowStartIdx, rowScratchBuffer, 0, cols);
            softmaxInPlace(rowScratchBuffer);
            System.arraycopy(rowScratchBuffer, 0, data, rowStartIdx, cols);
        }
    }

    public static void softmaxInPlace(double[] x) {
        if (x.length == 0) return;

        double max = Double.NEGATIVE_INFINITY;
        int i = 0;
        if (HAS_VECTOR) {
            var vmax = DoubleVector.broadcast(SPECIES, max);
            for (; i < SPECIES.loopBound(x.length); i += VLEN) {
                vmax = vmax.max(DoubleVector.fromArray(SPECIES, x, i));
            }
            max = vmax.reduceLanes(VectorOperators.MAX);
        }
        for (; i < x.length; i++) {
            max = Math.max(max, x[i]);
        }

        double sum = 0.0;
        i = 0;
        if (HAS_VECTOR) {
            var vmax = DoubleVector.broadcast(SPECIES, max);
            var vsum = DoubleVector.zero(SPECIES);
            for (; i < SPECIES.loopBound(x.length); i += VLEN) {
                var ev = DoubleVector.fromArray(SPECIES, x, i).sub(vmax).lanewise(VectorOperators.EXP);
                ev.intoArray(x, i);
                vsum = vsum.add(ev);
            }
            sum = vsum.reduceLanes(VectorOperators.ADD);
        }
        for (; i < x.length; i++) {
            x[i] = Math.exp(x[i] - max);
            sum += x[i];
        }

        final double invSum = 1.0 / sum;
        i = 0;
        if (HAS_VECTOR) {
            var vinv = DoubleVector.broadcast(SPECIES, invSum);
            for (; i < SPECIES.loopBound(x.length); i += VLEN) {
                DoubleVector.fromArray(SPECIES, x, i).mul(vinv).intoArray(x, i);
            }
        }
        for (; i < x.length; i++) {
            x[i] *= invSum;
        }
    }

    // ====================== ELEMENTWISE (STRIDE-AWARE VECTORIZATION) ======================
    public void geluInPlace() {
        if (isContiguous()) {
            int idx = 0;
            final int totalElements = rows * cols;
            if (HAS_VECTOR) {
                for (; idx < SPECIES.loopBound(totalElements); idx += VLEN) {
                    var v = DoubleVector.fromArray(SPECIES, data, offset + idx);
                    VectorMath.geluVector(v).intoArray(data, offset + idx);
                }
            }
            for (; idx < totalElements; idx++) {
                double x = data[offset + idx];
                double x3 = x * x * x;
                double t = Math.tanh(GELU_C1 * (x + GELU_C2 * x3));
                data[offset + idx] = 0.5 * x * (1.0 + t);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                int c = 0;
                if (HAS_VECTOR) {
                    for (; c < SPECIES.loopBound(cols); c += VLEN) {
                        var v = DoubleVector.fromArray(SPECIES, data, rowStart + c);
                        VectorMath.geluVector(v).intoArray(data, rowStart + c);
                    }
                }
                for (; c < cols; c++) {
                    double x = data[rowStart + c];
                    double x3 = x * x * x;
                    double t = Math.tanh(GELU_C1 * (x + GELU_C2 * x3));
                    data[rowStart + c] = 0.5 * x * (1.0 + t);
                }
            }
        }
    }

    public void reluInPlace() {
        var zero = DoubleVector.zero(SPECIES);
        if (isContiguous()) {
            int idx = 0;
            final int totalElements = rows * cols;
            if (HAS_VECTOR) {
                for (; idx < SPECIES.loopBound(totalElements); idx += VLEN) {
                    DoubleVector.fromArray(SPECIES, data, offset + idx).max(zero).intoArray(data, offset + idx);
                }
            }
            for (; idx < totalElements; idx++) {
                data[offset + idx] = Math.max(0, data[offset + idx]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                int c = 0;
                if (HAS_VECTOR) {
                    for (; c < SPECIES.loopBound(cols); c += VLEN) {
                        DoubleVector.fromArray(SPECIES, data, rowStart + c).max(zero).intoArray(data, rowStart + c);
                    }
                }
                for (; c < cols; c++) {
                    data[rowStart + c] = Math.max(0, data[rowStart + c]);
                }
            }
        }
    }

    public void tanhInPlace() {
        if (isContiguous()) {
            int idx = 0;
            final int totalElements = rows * cols;
            if (HAS_VECTOR) {
                for (; idx < SPECIES.loopBound(totalElements); idx += VLEN) {
                    DoubleVector.fromArray(SPECIES, data, offset + idx).lanewise(VectorOperators.TANH).intoArray(data, offset + idx);
                }
            }
            for (; idx < totalElements; idx++) {
                data[offset + idx] = Math.tanh(data[offset + idx]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                int c = 0;
                if (HAS_VECTOR) {
                    for (; c < SPECIES.loopBound(cols); c += VLEN) {
                        DoubleVector.fromArray(SPECIES, data, rowStart + c).lanewise(VectorOperators.TANH).intoArray(data, rowStart + c);
                    }
                }
                for (; c < cols; c++) {
                    data[rowStart + c] = Math.tanh(data[rowStart + c]);
                }
            }
        }
    }

    public void expInPlace() {
        if (isContiguous()) {
            int idx = 0;
            final int totalElements = rows * cols;
            if (HAS_VECTOR) {
                for (; idx < SPECIES.loopBound(totalElements); idx += VLEN) {
                    DoubleVector.fromArray(SPECIES, data, offset + idx).lanewise(VectorOperators.EXP).intoArray(data, offset + idx);
                }
            }
            for (; idx < totalElements; idx++) {
                data[offset + idx] = Math.exp(data[offset + idx]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                int c = 0;
                if (HAS_VECTOR) {
                    for (; c < SPECIES.loopBound(cols); c += VLEN) {
                        DoubleVector.fromArray(SPECIES, data, rowStart + c).lanewise(VectorOperators.EXP).intoArray(data, rowStart + c);
                    }
                }
                for (; c < cols; c++) {
                    data[rowStart + c] = Math.exp(data[rowStart + c]);
                }
            }
        }
    }

    public void logInPlace() {
        if (isContiguous()) {
            int idx = 0;
            final int totalElements = rows * cols;
            if (HAS_VECTOR) {
                for (; idx < SPECIES.loopBound(totalElements); idx += VLEN) {
                    DoubleVector.fromArray(SPECIES, data, offset + idx).lanewise(VectorOperators.LOG).intoArray(data, offset + idx);
                }
            }
            for (; idx < totalElements; idx++) {
                data[offset + idx] = Math.log(data[offset + idx]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                int c = 0;
                if (HAS_VECTOR) {
                    for (; c < SPECIES.loopBound(cols); c += VLEN) {
                        DoubleVector.fromArray(SPECIES, data, rowStart + c).lanewise(VectorOperators.LOG).intoArray(data, rowStart + c);
                    }
                }
                for (; c < cols; c++) {
                    data[rowStart + c] = Math.log(data[rowStart + c]);
                }
            }
        }
    }

    public void sinInPlace() {
        if (isContiguous()) {
            int idx = 0;
            final int totalElements = rows * cols;
            if (HAS_VECTOR) {
                for (; idx < SPECIES.loopBound(totalElements); idx += VLEN) {
                    DoubleVector.fromArray(SPECIES, data, offset + idx).lanewise(VectorOperators.SIN).intoArray(data, offset + idx);
                }
            }
            for (; idx < totalElements; idx++) {
                data[offset + idx] = Math.sin(data[offset + idx]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                int c = 0;
                if (HAS_VECTOR) {
                    for (; c < SPECIES.loopBound(cols); c += VLEN) {
                        DoubleVector.fromArray(SPECIES, data, rowStart + c).lanewise(VectorOperators.SIN).intoArray(data, rowStart + c);
                    }
                }
                for (; c < cols; c++) {
                    data[rowStart + c] = Math.sin(data[rowStart + c]);
                }
            }
        }
    }

    public static void add(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        if (A.rows != B.rows || A.cols != B.cols || C.rows != A.rows || C.cols != A.cols) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        
        if (A.isContiguous() && B.isContiguous() && C.isContiguous()) {
            int i = 0;
            int n = A.rows * A.cols;
            if (HAS_VECTOR) {
                for (; i < SPECIES.loopBound(n); i += VLEN) {
                    var va = DoubleVector.fromArray(SPECIES, A.data, A.offset + i);
                    var vb = DoubleVector.fromArray(SPECIES, B.data, B.offset + i);
                    va.add(vb).intoArray(C.data, C.offset + i);
                }
            }
            for (; i < n; i++) {
                C.data[C.offset + i] = A.data[A.offset + i] + B.data[B.offset + i];
            }
        } else {
            for (int r = 0; r < A.rows; r++) {
                int aStart = A.offset + r * A.rowStride;
                int bStart = B.offset + r * B.rowStride;
                int cStart = C.offset + r * C.rowStride;
                int c = 0;
                if (HAS_VECTOR) {
                    for (; c < SPECIES.loopBound(A.cols); c += VLEN) {
                        var va = DoubleVector.fromArray(SPECIES, A.data, aStart + c);
                        var vb = DoubleVector.fromArray(SPECIES, B.data, bStart + c);
                        va.add(vb).intoArray(C.data, cStart + c);
                    }
                }
                for (; c < A.cols; c++) {
                    C.data[cStart + c] = A.data[aStart + c] + B.data[bStart + c];
                }
            }
        }
    }

    public FlatMatrix transpose() {
        FlatMatrix T = new FlatMatrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T.set(j, i, get(i, j));
            }
        }
        return T;
    }

    // ====================== FUSED MATMUL + BIAS + GELU ======================
    public static void matmulAddBiasGelu(FlatMatrix A, FlatMatrix B, FlatMatrix bias, FlatMatrix C) {
        checkDims(A, B, C);
        if (bias.rows != 1 || bias.cols != B.cols) {
            throw new IllegalArgumentException("Bias layout must be 1xN");
        }

        final int M = A.rows, N = B.cols, K = A.cols;
        for (int i = 0; i < M; i++) {
            int j = 0;
            if (HAS_VECTOR && A.isContiguous() && B.isContiguous() && C.isContiguous() && bias.isContiguous()) {
                for (; j < SPECIES.loopBound(N); j += VLEN) {
                    var sum = DoubleVector.fromArray(SPECIES, bias.data, bias.offset + j);
                    for (int k = 0; k < K; k++) {
                        var av = DoubleVector.broadcast(SPECIES, A.get(i, k));
                        var bv = DoubleVector.fromArray(SPECIES, B.data, B.offset + k * B.rowStride + j);
                        sum = av.fma(bv, sum);
                    }
                    VectorMath.geluVector(sum).intoArray(C.data, C.offset + i * C.rowStride + j);
                }
            }
            for (; j < N; j++) {
                double sum = bias.get(0, j);
                for (int k = 0; k < K; k++) {
                    sum += A.get(i, k) * B.get(k, j);
                }
                double x3 = sum * sum * sum;
                double t = Math.tanh(GELU_C1 * (sum + GELU_C2 * x3));
                C.set(i, j, 0.5 * sum * (1.0 + t));
            }
        }
    }

    // ====================== ATTENTION MECHANISM ======================
    public static FlatMatrix attention(FlatMatrix Q, FlatMatrix K, FlatMatrix V, FlatMatrix output) {
        return attention(Q, K, V, output, null);
    }

    public static FlatMatrix attention(FlatMatrix Q, FlatMatrix K, FlatMatrix V,
                                       FlatMatrix output, ExecutorService executor) {
        final int seqLen = Q.rows;
        final double scale = 1.0 / Math.sqrt(Q.cols);
        if (executor == null || seqLen < 32) {
            attentionSingleThread(Q, K, V, output, scale);
        } else {
            attentionParallel(Q, K, V, output, scale, executor);
        }
        return output;
    }

    private static void attentionSingleThread(FlatMatrix Q, FlatMatrix K, FlatMatrix V, FlatMatrix out, double scale) {
        double[] scores = new double[K.rows]; // Should be sized to match Key-sequence rows
        for (int i = 0; i < Q.rows; i++) {
            qkT_row(Q, K, i, scores, scale);
            softmaxInPlace(scores);
            weightedSum(scores, V, out, i);
        }
    }

    private static void attentionParallel(FlatMatrix Q, FlatMatrix K, FlatMatrix V, FlatMatrix out,
                                          double scale, ExecutorService executor) {
        final int seqLen = Q.rows;
        final int nThreads = Math.min(Runtime.getRuntime().availableProcessors(), Math.max(1, seqLen / 8));
        final int chunk = (seqLen + nThreads - 1) / nThreads;
        List<Future<?>> futures = new ArrayList<>(nThreads);
        for (int t = 0; t < nThreads; t++) {
            final int start = t * chunk, end = Math.min(start + chunk, seqLen);
            futures.add(executor.submit(() -> {
                double[] scores = new double[K.rows];
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
                throw new RuntimeException("Parallel attention failed", e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Attention interrupted", e);
            }
        }
    }

    private static void qkT_row(FlatMatrix Q, FlatMatrix K, int i, double[] scores, double scale) {
        final int d_k = Q.cols;
        for (int j = 0; j < K.rows; j++) {
            double dot = 0.0;
            int k = 0;
            if (HAS_VECTOR && Q.isContiguous() && K.isContiguous()) {
                var sum = DoubleVector.zero(SPECIES);
                int qOff = Q.offset + i * Q.rowStride;
                int kOff = K.offset + j * K.rowStride;
                for (; k < SPECIES.loopBound(d_k); k += VLEN) {
                    var qv = DoubleVector.fromArray(SPECIES, Q.data, qOff + k);
                    var kv = DoubleVector.fromArray(SPECIES, K.data, kOff + k);
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

    private static void weightedSum(double[] scores, FlatMatrix V, FlatMatrix out, int i) {
        // Zero out row C target range to prepare for safe summation accumulation
        int oOff = out.offset + i * out.rowStride;
        Arrays.fill(out.data, oOff, oOff + V.cols, 0.0);

        for (int j = 0; j < V.rows; j++) {
            double w = scores[j];
            if (w == 0.0) continue;
            
            int k = 0;
            int vOff = V.offset + j * V.rowStride;
            if (HAS_VECTOR && V.isContiguous() && out.isContiguous()) {
                var vw = DoubleVector.broadcast(SPECIES, w);
                for (; k < SPECIES.loopBound(V.cols); k += VLEN) {
                    var vv = DoubleVector.fromArray(SPECIES, V.data, vOff + k);
                    var vo = DoubleVector.fromArray(SPECIES, out.data, oOff + k);
                    // FIXED: Correct FMA mapping executes: w * V + out
                    vw.fma(vv, vo).intoArray(out.data, oOff + k);
                }
            }
            for (; k < V.cols; k++) {
                out.data[oOff + k] += w * V.data[vOff + k];
            }
        }
    }

    private static void checkDims(FlatMatrix A, FlatMatrix B) {
        if (A.cols != B.rows) {
            throw new IllegalArgumentException("Dimension mismatch: A.cols != B.rows: " + A.cols + " != " + B.rows);
        }
    }

    private static void checkDims(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        checkDims(A, B);
        if (C.rows != A.rows || C.cols != B.cols) {
            throw new IllegalArgumentException("C layout dimension mismatch");
        }
    }

    // ====================== FUSED MATMUL + ALPHA * SIN(C) ======================
    public static void matmulAddSin(FlatMatrix A, FlatMatrix B, FlatMatrix C, double alpha) {
        checkDims(A, B, C);
        final int M = A.rows, N = B.cols, K = A.cols;
        final long flops = 2L * M * K * N;

        if (!HAS_VECTOR || flops < 16_384 || !A.isContiguous() || !B.isContiguous() || !C.isContiguous()) {
            matmulAddSinScalar(A, B, C, alpha);
            return;
        }
        matmulAddSinSimd(A, B, C, alpha);
    }

    private static void matmulAddSinSimd(FlatMatrix A, FlatMatrix B, FlatMatrix C, double alpha) {
        final int M = A.rows, N = B.cols, K = A.cols;
        final DoubleVector vAlpha = DoubleVector.broadcast(SPECIES, alpha);
        final int loopBoundN = SPECIES.loopBound(N);

        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;

            int j = 0;
            for (; j < loopBoundN; j += VLEN) {
                var acc = DoubleVector.zero(SPECIES);
                for (int k = 0; k < K; k++) {
                    var aVec = DoubleVector.broadcast(SPECIES, A.data[rowA + k]);
                    var bVec = DoubleVector.fromArray(SPECIES, B.data, B.offset + k * B.rowStride + j);
                    acc = aVec.fma(bVec, acc);
                }
                var cOld = DoubleVector.fromArray(SPECIES, C.data, rowC + j);
                var sinC = cOld.lanewise(VectorOperators.SIN);
                acc.add(sinC.mul(vAlpha)).intoArray(C.data, rowC + j);
            }

            for (; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += A.data[rowA + k] * B.data[B.offset + k * B.rowStride + j];
                }
                final int idx = rowC + j;
                C.data[idx] = sum + alpha * Math.sin(C.data[idx]);
            }
        }
    }

    private static void matmulAddSinScalar(FlatMatrix A, FlatMatrix B, FlatMatrix C, double alpha) {
        final int M = A.rows, N = B.cols, K = A.cols;
        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += A.data[rowA + k] * B.data[B.offset + k * B.rowStride + j];
                }
                final int idx = rowC + j;
                C.data[idx] = sum + alpha * Math.sin(C.data[idx]);
            }
        }
    }

    // ====================== IN-PLACE MATMUL OVERLAYS ======================
    public static void matmulInPlace(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        checkDims(A, B, C);
        final int M = A.rows, N = B.cols, K = A.cols;
        final long flops = 2L * M * K * N;

        if (!HAS_VECTOR || flops < 16_384 || !A.isContiguous() || !B.isContiguous() || !C.isContiguous()) {
            matmulInPlaceScalar(A, B, C);
            return;
        }
        matmulSimdBroadcast(A, B, C);
    }

    public static void matmulSimdBroadcast(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        checkDims(A, B, C);
        final int M = A.rows, N = B.cols, K = A.cols;
        final int loopBoundN = SPECIES.loopBound(N);

        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;

            for (int k = 0; k < K; k++) {
                var aVec = DoubleVector.broadcast(SPECIES, A.data[rowA + k]);
                final int rowB = B.offset + k * B.rowStride;
                int j = 0;
                for (; j < loopBoundN; j += VLEN) {
                    var bVec = DoubleVector.fromArray(SPECIES, B.data, rowB + j);
                    var cVec = DoubleVector.fromArray(SPECIES, C.data, rowC + j);
                    aVec.fma(bVec, cVec).intoArray(C.data, rowC + j);
                }
                for (; j < N; j++) {
                    C.data[rowC + j] += A.data[rowA + k] * B.data[rowB + j];
                }
            }
        }
    }

    private static void matmulInPlaceScalar(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        final int M = A.rows, N = B.cols, K = A.cols;
        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;
            for (int k = 0; k < K; k++) {
                double a = A.data[rowA + k];
                final int rowB = B.offset + k * B.rowStride;
                for (int j = 0; j < N; j++) {
                    C.data[rowC + j] += a * B.data[rowB + j];
                }
            }
        }
    }

    // ====================== NESTED VECTOR MATH VECTOR EXTENSIONS ======================
    public static final class VectorMath {

        private static final boolean HAS_TANH = hasTanhOp();

        private static boolean hasTanhOp() {
            if (Runtime.version().feature() < 22) return false;
            try {
                var a = VectorOperators.TANH;
                return true;
            } catch (Throwable t) {
                return false;
            }
        }

        public static DoubleVector geluVector(DoubleVector x) {
            return HAS_TANH ? geluVectorExact(x) : geluVectorCompat(x);
        }

        private static DoubleVector geluVectorExact(DoubleVector x) {
            var x2 = x.mul(x);
            var inner = x.add(x2.mul(x).mul(V_GELU_C2)).mul(V_GELU_C1);
            var t = inner.lanewise(VectorOperators.TANH);
            return x.mul(V_HALF).mul(t.add(V_ONE));
        }

        public static DoubleVector geluVectorCompat(DoubleVector x) {
            var x3 = x.mul(x).mul(x);
            var inner = x3.fma(V_GELU_C2, x).mul(V_GELU_C1);
            var i2 = inner.mul(inner);
            
            // Fast rational Padé approximation step optimized using FMA mechanics
            var num = inner.fma(i2, inner.mul(27.0));
            var den = i2.fma(DoubleVector.broadcast(SPECIES, 9.0), DoubleVector.broadcast(SPECIES, 27.0));
            var t = num.div(den);
            
            return x.mul(V_HALF).mul(t.add(V_ONE));
        }
    }
   public static void randomFill(FlatMatrix A) {
        ThreadLocalRandom ran = ThreadLocalRandom.current();
        for (int i = 0; i < A.data.length; i++) {
            A.data[i] = 1 + ran.nextDouble(101);
        }
    }//end method randomFill
   
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
    // ====================== QUANTIZATION WRAPPERS ======================
    public byte[] asByteArray() {
        if (byteData == null) {
            byteData = new byte[rows * cols];
        }
        return byteData;
    }

    public static FlatMatrix wrapBytes(byte[] bytes, int rows, int cols) {
        FlatMatrix m = new FlatMatrix(null, rows, cols, cols, 0) {
            @Override
            public double get(int r, int c) {
                throw new UnsupportedOperationException("Use asByteArray() direct kernel access for Q8 variants");
            }

            @Override
            public void set(int r, int c, double v) {
                throw new UnsupportedOperationException("Use asByteArray() direct kernel access for Q8 variants");
            }
        };
        m.byteData = bytes;
        return m;
    }
}