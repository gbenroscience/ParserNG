package com.github.gbenroscience.parser.turbo.tools.vector.matrix;


import java.util.*;
import java.util.concurrent.*;

/**
 * High-Performance Mechanically Sympathetic Flat Matrix (Non-Vector API Variant).
 * * Bypasses explicit incubator vector instructions while structurally maximizing 
 * HotSpot JIT compiler loop auto-vectorization (SLP) via linear stride-1 alignment.
 * * @author GBEMIRO
 */
public class FlatMatrix {

    // GELU constants
    private static final double GELU_C1 = 0.7978845608028654;
    private static final double GELU_C2 = 0.044715;

    // Block sizes for cache-friendly tiling
    private static final int NC = 256, KC = 256, MR = 4;
    private static final int NR = 64; // Aligned block column factor for spatial L1/L2 data caching

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
        return rowStride == cols;
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
        final double[] cData = C.data;
        if (C.isContiguous()) {
            int start = C.offset + rowStart * C.rowStride;
            int len = (rowEnd - rowStart) * C.rowStride;
            Arrays.fill(cData, start, start + len, 0.0);
        } else {
            for (int i = rowStart; i < rowEnd; i++) {
                int start = C.offset + i * C.rowStride;
                Arrays.fill(cData, start, start + C.cols, 0.0);
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

        if (rows != MR || !A.isContiguous() || !B.isContiguous() || !C.isContiguous()) {
            scalarMicrokernel(A, B, C, i0, i1, j0, j1, k0, k1);
            return;
        }

        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;

        final int baseA0 = A.offset + i0 * A.rowStride;
        final int baseA1 = baseA0 + A.rowStride;
        final int baseA2 = baseA1 + A.rowStride;
        final int baseA3 = baseA2 + A.rowStride;

        final int baseC0 = C.offset + i0 * C.rowStride + j0;
        final int baseC1 = baseC0 + C.rowStride;
        final int baseC2 = baseC1 + C.rowStride;
        final int baseC3 = baseC2 + C.rowStride;

        final int innerCount = j1 - j0;

        for (int k = k0; k < k1; k++) {
            final double a0 = aData[baseA0 + k];
            final double a1 = aData[baseA1 + k];
            final double a2 = aData[baseA2 + k];
            final double a3 = aData[baseA3 + k];

            final int rowB = B.offset + k * B.rowStride + j0;

            // Strict Stride-1 loop. The C2 Compiler natively targets this loop 
            // for hardware vector parsing and automatic unrolling execution steps.
            for (int j = 0; j < innerCount; j++) {
                final double bVal = bData[rowB + j];
                cData[baseC0 + j] += a0 * bVal;
                cData[baseC1 + j] += a1 * bVal;
                cData[baseC2 + j] += a2 * bVal;
                cData[baseC3 + j] += a3 * bVal;
            }
        }
    }

    private static void scalarMicrokernel(FlatMatrix A, FlatMatrix B, FlatMatrix C,
                                          int i0, int i1, int j0, int j1, int k0, int k1) {
        final int rows = i1 - i0;
        final int cols = j1 - j0;
        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;

        for (int k = k0; k < k1; k++) {
            for (int r = 0; r < rows; r++) {
                double a = aData[A.offset + (i0 + r) * A.rowStride + k];
                int cRowOff = C.offset + (i0 + r) * C.rowStride + j0;
                int bRowOff = B.offset + k * B.rowStride + j0;
                for (int c = 0; c < cols; c++) {
                    cData[cRowOff + c] += a * bData[bRowOff + c];
                }
            }
        }
    }

    // ====================== SOFTMAX ======================
    public void softmaxRowsInPlace() {
        if (cols == 0) return;
        final double[] sData = this.data;
        double[] rowScratchBuffer = new double[cols];
        for (int i = 0; i < rows; i++) {
            int rowStartIdx = offset + i * rowStride;
            System.arraycopy(sData, rowStartIdx, rowScratchBuffer, 0, cols);
            softmaxInPlace(rowScratchBuffer);
            System.arraycopy(rowScratchBuffer, 0, sData, rowStartIdx, cols);
        }
    }

    public static void softmaxInPlace(double[] x) {
        final int len = x.length;
        if (len == 0) return;

        // Step 1: Compute Maximum scalar element
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < len; i++) {
            if (x[i] > max) {
                max = x[i];
            }
        }

        // Step 2: Compute exponent streams and tracking sum
        double sum = 0.0;
        for (int i = 0; i < len; i++) {
            double ev = Math.exp(x[i] - max);
            x[i] = ev;
            sum += ev;
        }

        // Step 3: Fast scalar division multiplication pass
        final double invSum = 1.0 / sum;
        for (int i = 0; i < len; i++) {
            x[i] *= invSum;
        }
    }

    // ====================== ELEMENTWISE ACTIVATIONS ======================
    public void geluInPlace() {
        final double[] sData = this.data;
        if (isContiguous()) {
            final int totalElements = rows * cols;
            final int start = this.offset;
            for (int idx = 0; idx < totalElements; idx++) {
                double x = sData[start + idx];
                double x3 = x * x * x;
                double t = Math.tanh(GELU_C1 * (x + GELU_C2 * x3));
                sData[start + idx] = 0.5 * x * (1.0 + t);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                for (int c = 0; c < cols; c++) {
                    double x = sData[rowStart + c];
                    double x3 = x * x * x;
                    double t = Math.tanh(GELU_C1 * (x + GELU_C2 * x3));
                    sData[rowStart + c] = 0.5 * x * (1.0 + t);
                }
            }
        }
    }

    public void reluInPlace() {
        final double[] sData = this.data;
        if (isContiguous()) {
            final int totalElements = rows * cols;
            final int start = this.offset;
            for (int idx = 0; idx < totalElements; idx++) {
                double v = sData[start + idx];
                sData[start + idx] = v < 0.0 ? 0.0 : v;
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                for (int c = 0; c < cols; c++) {
                    double v = sData[rowStart + c];
                    sData[rowStart + c] = v < 0.0 ? 0.0 : v;
                }
            }
        }
    }

    public void tanhInPlace() {
        final double[] sData = this.data;
        if (isContiguous()) {
            final int totalElements = rows * cols;
            final int start = this.offset;
            for (int idx = 0; idx < totalElements; idx++) {
                sData[start + idx] = Math.tanh(sData[start + idx]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                for (int c = 0; c < cols; c++) {
                    sData[rowStart + c] = Math.tanh(sData[rowStart + c]);
                }
            }
        }
    }

    public void expInPlace() {
        final double[] sData = this.data;
        if (isContiguous()) {
            final int totalElements = rows * cols;
            final int start = this.offset;
            for (int idx = 0; idx < totalElements; idx++) {
                sData[start + idx] = Math.exp(sData[start + idx]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                for (int c = 0; c < cols; c++) {
                    sData[rowStart + c] = Math.exp(sData[rowStart + c]);
                }
            }
        }
    }

    public void logInPlace() {
        final double[] sData = this.data;
        if (isContiguous()) {
            final int totalElements = rows * cols;
            final int start = this.offset;
            for (int idx = 0; idx < totalElements; idx++) {
                sData[start + idx] = Math.log(sData[start + idx]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                for (int c = 0; c < cols; c++) {
                    sData[rowStart + c] = Math.log(sData[rowStart + c]);
                }
            }
        }
    }

    public void sinInPlace() {
        final double[] sData = this.data;
        if (isContiguous()) {
            final int totalElements = rows * cols;
            final int start = this.offset;
            for (int idx = 0; idx < totalElements; idx++) {
                sData[start + idx] = Math.sin(sData[start + idx]);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * rowStride;
                for (int c = 0; c < cols; c++) {
                    sData[rowStart + c] = Math.sin(sData[rowStart + c]);
                }
            }
        }
    }

    public static void add(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        if (A.rows != B.rows || A.cols != B.cols || C.rows != A.rows || C.cols != A.cols) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        
        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;

        if (A.isContiguous() && B.isContiguous() && C.isContiguous()) {
            final int n = A.rows * A.cols;
            final int aOff = A.offset;
            final int bOff = B.offset;
            final int cOff = C.offset;
            for (int i = 0; i < n; i++) {
                cData[cOff + i] = aData[aOff + i] + bData[bOff + i];
            }
        } else {
            for (int r = 0; r < A.rows; r++) {
                int aStart = A.offset + r * A.rowStride;
                int bStart = B.offset + r * B.rowStride;
                int cStart = C.offset + r * C.rowStride;
                for (int c = 0; c < A.cols; c++) {
                    cData[cStart + c] = aData[aStart + c] + bData[bStart + c];
                }
            }
        }
    }

    /**
     * Cache-conscious Tiled Matrix Transposition
     */
    public FlatMatrix transpose() {
        FlatMatrix T = new FlatMatrix(cols, rows);
        final int r = this.rows;
        final int c = this.cols;
        final int BLOCK = 64; // Prevents TLB thrashing on large layouts

        for (int i = 0; i < r; i += BLOCK) {
            int iEnd = Math.min(i + BLOCK, r);
            for (int j = 0; j < c; j += BLOCK) {
                int jEnd = Math.min(j + BLOCK, c);
                for (int ii = i; ii < iEnd; ii++) {
                    for (int jj = j; jj < jEnd; jj++) {
                        T.set(jj, ii, this.get(ii, jj));
                    }
                }
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
        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;
        final double[] biasData = bias.data;
        final int bOffset = B.offset;

        for (int i = 0; i < M; i++) {
            final int cRowOff = C.offset + i * C.rowStride;
            
            // Phase 1: Initialize row C space using clean vector stride reads from bias
            final int biasOff = bias.offset;
            for (int j = 0; j < N; j++) {
                cData[cRowOff + j] = biasData[biasOff + j];
            }

            // Phase 2: Accumulate multi-element combinations over IKJ loop strides
            for (int k = 0; k < K; k++) {
                final double av = A.get(i, k);
                final int bRowOff = bOffset + k * B.rowStride;
                for (int j = 0; j < N; j++) {
                    cData[cRowOff + j] += av * bData[bRowOff + j];
                }
            }

            // Phase 3: Run safe localized elementwise execution over the finished columns
            for (int j = 0; j < N; j++) {
                double sum = cData[cRowOff + j];
                double x3 = sum * sum * sum;
                double t = Math.tanh(GELU_C1 * (sum + GELU_C2 * x3));
                cData[cRowOff + j] = 0.5 * sum * (1.0 + t);
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
        double[] scores = new double[K.rows];
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
        final double[] qData = Q.data;
        final double[] kData = K.data;
        final int qOff = Q.offset + i * Q.rowStride;
        final int kRows = K.rows;
        final int kStride = K.rowStride;
        final int kOffset = K.offset;

        for (int j = 0; j < kRows; j++) {
            double dot = 0.0;
            int kOff = kOffset + j * kStride;
            for (int k = 0; k < d_k; k++) {
                dot += qData[qOff + k] * kData[kOff + k];
            }
            scores[j] = dot * scale;
        }
    }

    private static void weightedSum(double[] scores, FlatMatrix V, FlatMatrix out, int i) {
        final int oOff = out.offset + i * out.rowStride;
        final double[] oData = out.data;
        final double[] vData = V.data;
        final int vRows = V.rows;
        final int vCols = V.cols;
        final int vStride = V.rowStride;
        final int vOffset = V.offset;

        Arrays.fill(oData, oOff, oOff + vCols, 0.0);

        for (int j = 0; j < vRows; j++) {
            final double w = scores[j];
            if (w == 0.0) continue;
            
            final int vOff = vOffset + j * vStride;
            for (int k = 0; k < vCols; k++) {
                oData[oOff + k] += w * vData[vOff + k];
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
        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;
        final int bOffset = B.offset;

        // Thread-local array simulation buffer to avoid constant loop thrashing
        double[] rowBuffer = new double[N];

        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;

            Arrays.fill(rowBuffer, 0.0);

            // IKJ streaming structure matches optimal auto-vectorization bounds
            for (int k = 0; k < K; k++) {
                final double aVal = aData[rowA + k];
                final int rowB = bOffset + k * B.rowStride;
                for (int j = 0; j < N; j++) {
                    rowBuffer[j] += aVal * bData[rowB + j];
                }
            }

            for (int j = 0; j < N; j++) {
                final int idx = rowC + j;
                cData[idx] = rowBuffer[j] + alpha * Math.sin(cData[idx]);
            }
        }
    }

    // ====================== IN-PLACE MATMUL OVERLAYS ======================
    public static void matmulInPlace(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        checkDims(A, B, C);
        final int M = A.rows, N = B.cols, K = A.cols;
        final double[] aData = A.data;
        final double[] bData = B.data;
        final double[] cData = C.data;
        final int bOffset = B.offset;

        for (int i = 0; i < M; i++) {
            final int rowA = A.offset + i * A.rowStride;
            final int rowC = C.offset + i * C.rowStride;

            for (int k = 0; k < K; k++) {
                final double aVal = aData[rowA + k];
                final int rowB = bOffset + k * B.rowStride;
                for (int j = 0; j < N; j++) {
                    cData[rowC + j] += aVal * bData[rowB + j];
                }
            }
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
    @Deprecated
    public static void matmulSimdBroadcast(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        matmulInPlace(A, B, C); // Vectorization is naturally generated via matmulInPlace's sequential loops
    }

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