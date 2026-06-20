package com.github.gbenroscience.parser.turbo.tools.vector.matrix;

import java.util.*;
import java.util.concurrent.*;

/**
 * High-Performance Flat Float Matrix Engine
 * Optimized for HotSpot C2 Auto-Vectorization (SLP).
 * * @author GBEMIRO
 */ 
public final class FlatMatrixF {

    // GELU constants float
    private static final float GELU_C1 = 0.7978845608028654f;
    private static final float GELU_C2 = 0.044715f;

    // Block sizes for float - NR configured to standard 256-bit hardware vector width (8 floats)
    private static final int NC = 256, KC = 256, MR = 4;
    private static final int NR = 8; 

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

    public boolean isContiguous() {
        return rowStride == cols && offset % 8 == 0;
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

        if (rows != MR || cols != NR || !A.isContiguous() || !B.isContiguous() || !C.isContiguous()) {
            scalarMicrokernel(A, B, C, i0, i1, j0, j1, k0, k1);
            return;
        }

        // Stack-allocated accumulator chunk encourages direct register pinning
        float[] acc = new float[MR * NR];
        if (k0 != 0) {
            for (int r = 0; r < MR; r++) {
                System.arraycopy(C.data, C.offset + (i0 + r) * C.rowStride + j0, acc, r * NR, NR);
            }
        }

        final int baseA0 = A.offset + i0 * A.rowStride;
        final int baseA1 = baseA0 + A.rowStride;
        final int baseA2 = baseA1 + A.rowStride;
        final int baseA3 = baseA2 + A.rowStride;
        final float[] aData = A.data;
        final float[] bData = B.data;

        for (int k = k0; k < k1; k++) {
            final int rowB = B.offset + k * B.rowStride + j0;
            float a0 = aData[baseA0 + k];
            float a1 = aData[baseA1 + k];
            float a2 = aData[baseA2 + k];
            float a3 = aData[baseA3 + k];

            // Isolated linear loops with clear unit strides fully guarantee C2 auto-vectorization
            for (int j = 0; j < NR; j++) acc[0 * NR + j] += a0 * bData[rowB + j];
            for (int j = 0; j < NR; j++) acc[1 * NR + j] += a1 * bData[rowB + j];
            for (int j = 0; j < NR; j++) acc[2 * NR + j] += a2 * bData[rowB + j];
            for (int j = 0; j < NR; j++) acc[3 * NR + j] += a3 * bData[rowB + j];
        }

        for (int r = 0; r < MR; r++) {
            System.arraycopy(acc, r * NR, C.data, C.offset + (i0 + r) * C.rowStride + j0, NR);
        }
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
        int len = rows * cols;
        for (int idx = 0; idx < len; idx++) {
            float x = data[offset + idx];
            float x3 = x * x * x;
            float t = (float) Math.tanh(GELU_C1 * (x + GELU_C2 * x3));
            data[offset + idx] = 0.5f * x * (1.0f + t);
        }
    }

    public void siluInPlace() {
        int len = rows * cols;
        for (int idx = 0; idx < len; idx++) {
            float x = data[offset + idx];
            data[offset + idx] = x / (1.0f + (float) Math.exp(-x));
        }
    }

    public void reluInPlace() {
        int len = rows * cols;
        for (int idx = 0; idx < len; idx++) {
            data[offset + idx] = Math.max(0.0f, data[offset + idx]);
        }
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
        
        // 1. Find Max
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < x.length; i++) {
            max = Math.max(max, x[i]);
        }

        // 2. Exponentiate (Separated from reduction loop to maximize vector pipe processing)
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) Math.exp(x[i] - max);
        }

        // 3. Compute Sum
        float sum = 0.0f;
        for (int i = 0; i < x.length; i++) {
            sum += x[i];
        }

        // 4. Normalize
        final float invSum = 1.0f / sum;
        for (int i = 0; i < x.length; i++) {
            x[i] *= invSum;
        }
    }

    public static void add(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        int n = A.rows * A.cols;
        if (A.rows != B.rows || A.cols != B.cols || C.rows != A.rows || C.cols != A.cols)
            throw new IllegalArgumentException("Shape mismatch");
        
        for (int i = 0; i < n; i++) {
            C.data[C.offset + i] = A.data[A.offset + i] + B.data[B.offset + i];
        }
    }

    public static void mul(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        int n = A.rows * A.cols;
        for (int i = 0; i < n; i++) {
            C.data[C.offset + i] = A.data[A.offset + i] * B.data[B.offset + i];
        }
    }

    // ====================== FUSED KERNELS ======================
    public static void matmulAddBiasGelu(FlatMatrixF A, FlatMatrixF B, FlatMatrixF bias, FlatMatrixF C) {
        checkDims(A, B, C);
        if (bias.rows != 1 || bias.cols != B.cols) throw new IllegalArgumentException("Bias must be 1xN");
        final int M = A.rows, N = B.cols, K = A.cols;
        
        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;

            // Seed target matrix line with bias data directly
            System.arraycopy(bias.data, bias.offset, C.data, rowC, N);

            // Compute GEMM with strict unit-stride inner loops
            for (int k = 0; k < K; k++) {
                float aVal = A.data[rowA + k];
                final int rowB = B.offset + k * B.rowStride;
                for (int j = 0; j < N; j++) {
                    C.data[rowC + j] += aVal * B.data[rowB + j];
                }
            }

            // High-throughput execution loop for activations
            for (int j = 0; j < N; j++) {
                float sum = C.data[rowC + j];
                float x3 = sum * sum * sum;
                float t = (float) Math.tanh(GELU_C1 * (sum + GELU_C2 * x3));
                C.data[rowC + j] = 0.5f * sum * (1.0f + t);
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
    
    /**
     * FUSED MATMUL + alpha * sin(C)
     * C = A * B + alpha * sin(C)
     */
    public static void matmulAddSin(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C, float alpha) {
        checkDims(A, B, C);
        final int M = A.rows, N = B.cols, K = A.cols;

        // Unified temporary structure avoids heap reallocation during looping
        float[] tempSum = new float[N];

        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;

            Arrays.fill(tempSum, 0.0f);

            for (int k = 0; k < K; k++) {
                float aVal = A.data[rowA + k];
                final int rowB = B.offset + k * B.rowStride;
                for (int j = 0; j < N; j++) {
                    tempSum[j] += aVal * B.data[rowB + j];
                }
            }

            for (int j = 0; j < N; j++) {
                int idx = rowC + j;
                C.data[idx] = tempSum[j] + alpha * (float) Math.sin(C.data[idx]);
            }
        }
    }

    /**
     * MATMUL IN-PLACE (C += A * B)
     */
    public static void matmulInPlace(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        checkDims(A, B, C);
        final int M = A.rows, N = B.cols, K = A.cols;
        
        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;

            for (int k = 0; k < K; k++) {
                float aVal = A.data[rowA + k];
                final int rowB = B.offset + k * B.rowStride;
                for (int j = 0; j < N; j++) {
                    C.data[rowC + j] += aVal * B.data[rowB + j];
                }
            }
        }
    }

    public static void swiGLU(FlatMatrixF gate, FlatMatrixF up, FlatMatrixF out) {
        if (gate != out) System.arraycopy(gate.data, gate.offset, out.data, out.offset, gate.rows * gate.cols);
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
        if (A.cols != B.rows) throw new IllegalArgumentException("A.cols != B.rows: " + A.cols + " != " + B.rows);
    }

    private static void checkDims(FlatMatrixF A, FlatMatrixF B, FlatMatrixF C) {
        checkDims(A, B);
        if (C.rows != A.rows || C.cols != B.cols) throw new IllegalArgumentException("C dims mismatch");
    }
    
    // ====================== MHA ATTENTION ======================
    /**
     * Multi-head attention: output = softmax(Q * K^T / sqrt(d_k)) * V
     */
    public static FlatMatrixF mhaAttention(FlatMatrixF Q, FlatMatrixF K, FlatMatrixF V,
                                           FlatMatrixF output, float scale) {
        return mhaAttention(Q, K, V, output, scale, null);
    }

    public static FlatMatrixF mhaAttention(FlatMatrixF Q, FlatMatrixF K, FlatMatrixF V,
                                           FlatMatrixF output, float scale, ExecutorService executor) {
        final int seqLen = Q.rows;
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
  public static void randomFill(FlatMatrixF A) {
        ThreadLocalRandom ran = ThreadLocalRandom.current();
        for (int i = 0; i < A.data.length; i++) {
            A.data[i] = 1 + ran.nextFloat()*101;
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
    // Q[i] @ K^T * scale
    private static void qkT_row(FlatMatrixF Q, FlatMatrixF K, int i, float[] scores, float scale) {
        final int d_k = Q.cols;
        final int seqLen = K.rows;
        final int qOff = Q.offset + i * Q.rowStride;
        
        for (int j = 0; j < seqLen; j++) {
            float dot = 0.0f;
            final int kOff = K.offset + j * K.rowStride;
            
            // Clean unit-stride inner loop auto-vectorizes into rapid FMA reductions
            for (int k = 0; k < d_k; k++) {
                dot += Q.data[qOff + k] * K.data[kOff + k];
            }
            scores[j] = dot * scale;
        }
    }

    // output[i] = sum_j scores[j] * V[j]
    private static void weightedSum(float[] scores, FlatMatrixF V, FlatMatrixF out, int i) {
        final int d_v = V.cols;
        final int seqLen = V.rows;
        final int oOff = out.offset + i * out.rowStride;

        Arrays.fill(out.data, oOff, oOff + d_v, 0.0f);

        for (int j = 0; j < seqLen; j++) {
            float w = scores[j];
            if (w == 0.0f) continue;
            
            final int vOff = V.offset + j * V.rowStride;
            for (int k = 0; k < d_v; k++) {
                out.data[oOff + k] += w * V.data[vOff + k];
            }
        }
    }
}