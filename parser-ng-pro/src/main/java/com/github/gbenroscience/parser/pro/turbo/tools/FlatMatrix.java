package com.github.gbenroscience.parser.pro.turbo.tools;

/**
 *
 * @author GBEMIRO
 */
import jdk.incubator.vector.*;

public final class FlatMatrix {
    
    
    
        private static final VectorSpecies<Double> VS = DoubleVector.SPECIES_PREFERRED;
    private static final int VLEN = VS.length();
    private static final boolean HAS_VECTOR = VS.length() > 1;

    public final double[] data;
    public final int rows, cols;

    public FlatMatrix(int rows, int cols) {
        this(rows, cols, new double[rows * cols]);
    }

    public FlatMatrix(int rows, int cols, double[] data) {
        if (data.length!= rows * cols) throw new IllegalArgumentException();
        this.rows = rows; this.cols = cols; this.data = data;
    }
    
     

    // ThreadLocal scratch to stay zero-alloc
    private static final ThreadLocal<double[]> SCRATCH =
        ThreadLocal.withInitial(() -> new double[1 << 20]); // 8MB



    public static FlatMatrix zeros(int r, int c) { return new FlatMatrix(r, c); }

    // C = A * B
    public static FlatMatrix matmul(FlatMatrix A, FlatMatrix B) {
        if (A.cols!= B.rows) throw new IllegalArgumentException("Shape mismatch");
        FlatMatrix C = new FlatMatrix(A.rows, B.cols);
        matmul(A, B, C);
        return C;
    }

    public static void matmul(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        final int M = A.rows, N = B.cols, K = A.cols;
        if (C.rows!= M || C.cols!= N || A.cols!= B.rows)
            throw new IllegalArgumentException();

        // Heuristic: scalar for tiny matrices due to setup cost
        if (!HAS_VECTOR || M * N * K < 32_768) {
            matmulScalar(A.data, B.data, C.data, M, N, K);
            return;
        }

        // Fast path: B transposed gives 1.3x speedup, but costs alloc.
        // Use scratch if B is reused, otherwise do broadcast version.
        matmulSimdBroadcast(A.data, B.data, C.data, M, N, K);
    }

    // C = A * B + alpha * sin(C) - example fused op
    public static void matmulAddSin(FlatMatrix A, FlatMatrix B, FlatMatrix C, double alpha) {
        final int M = A.rows, N = B.cols, K = A.cols;
        if (!HAS_VECTOR || M * N * K < 32_768) {
            matmulAddSinScalar(A.data, B.data, C.data, M, N, K, alpha);
            return;
        }
        matmulAddSinSimd(A.data, B.data, C.data, M, N, K, alpha);
    }

    // Element-wise: A = sin(A)
    public void sinInPlace() {
        if (!HAS_VECTOR) {
            for (int i = 0; i < data.length; i++) data[i] = Math.sin(data[i]);
            return;
        }
        int i = 0;
        for (; i < VS.loopBound(data.length); i += VLEN) {
            var v = DoubleVector.fromArray(VS, data, i);
            v.lanewise(VectorOperators.SIN).intoArray(data, i); // JDK 22+
        }
        for (; i < data.length; i++) data[i] = Math.sin(data[i]); // tail
    }

    // Element-wise: C = A + B
    public static void add(FlatMatrix A, FlatMatrix B, FlatMatrix C) {
        int n = A.data.length;
        if (!HAS_VECTOR) {
            for (int i = 0; i < n; i++) C.data[i] = A.data[i] + B.data[i];
            return;
        }
        int i = 0;
        for (; i < VS.loopBound(n); i += VLEN) {
            var va = DoubleVector.fromArray(VS, A.data, i);
            var vb = DoubleVector.fromArray(VS, B.data, i);
            va.add(vb).intoArray(C.data, i);
        }
        for (; i < n; i++) C.data[i] = A.data[i] + B.data[i];
    }

    // --- Private kernels ---

    private static void matmulScalar(double[] A, double[] B, double[] C, int M, int N, int K) {
        java.util.Arrays.fill(C, 0);
        for (int i = 0; i < M; i++) {
            int rowA = i * K;
            int rowC = i * N;
            for (int k = 0; k < K; k++) {
                double aik = A[rowA + k];
                int rowB = k * N;
                for (int j = 0; j < N; j++) {
                    C[rowC + j] += aik * B[rowB + j];
                }
            }
        }
    }

    private static void matmulSimdBroadcast(double[] A, double[] B, double[] C, int M, int N, int K) {
        java.util.Arrays.fill(C, 0);
        for (int i = 0; i < M; i++) {
            int rowA = i * K;
            int rowC = i * N;
            for (int k = 0; k < K; k++) {
                var a = DoubleVector.broadcast(VS, A[rowA + k]);
                int rowB = k * N;
                int j = 0;
                for (; j < VS.loopBound(N); j += VLEN) {
                    var b = DoubleVector.fromArray(VS, B, rowB + j);
                    var c = DoubleVector.fromArray(VS, C, rowC + j);
                    a.fma(b, c).intoArray(C, rowC + j);
                }
                // Tail scalar
                for (; j < N; j++) {
                    C[rowC + j] += A[rowA + k] * B[rowB + j];
                }
            }
        }
    }

    private static void matmulAddSinSimd(double[] A, double[] B, double[] C,
                                         int M, int N, int K, double alpha) {
        for (int i = 0; i < M; i++) {
            int rowA = i * K;
            int rowC = i * N;
            for (int j = 0; j < N; j += VLEN) {
                var acc = DoubleVector.zero(VS);
                for (int k = 0; k < K; k++) {
                    var a = DoubleVector.broadcast(VS, A[rowA + k]);
                    var b = DoubleVector.fromArray(VS, B, k * N + j);
                    acc = a.fma(b, acc);
                }
                var c = DoubleVector.fromArray(VS, C, rowC + j);
                var sinC = c.lanewise(VectorOperators.SIN); // JDK 22+
                acc.add(sinC.mul(alpha)).intoArray(C, rowC + j);
            }
        }
    }

    private static void matmulAddSinScalar(double[] A, double[] B, double[] C,
                                           int M, int N, int K, double alpha) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum + alpha * Math.sin(C[i * N + j]);
            }
        }
    }

    // Utility: transpose B for faster matmul if B is reused
    public FlatMatrix transpose() {
        FlatMatrix T = new FlatMatrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T.data[j * rows + i] = data[i * cols + j];
            }
        }
        return T;
    }
    
      public void expInPlace() {
        int i = 0;
        if (HAS_VECTOR) {
            for (; i < VS.loopBound(data.length); i += VLEN) {
                var v = DoubleVector.fromArray(VS, data, i);
                v.lanewise(VectorOperators.EXP).intoArray(data, i);
            }
        }
        for (; i < data.length; i++) data[i] = Math.exp(data[i]);
    }

    public void logInPlace() {
        int i = 0;
        if (HAS_VECTOR) {
            for (; i < VS.loopBound(data.length); i += VLEN) {
                var v = DoubleVector.fromArray(VS, data, i);
                v.lanewise(VectorOperators.LOG).intoArray(data, i);
            }
        }
        for (; i < data.length; i++) data[i] = Math.log(data[i]);
    }

    public void tanhInPlace() {
        int i = 0;
        if (HAS_VECTOR) {
            for (; i < VS.loopBound(data.length); i += VLEN) {
                var v = DoubleVector.fromArray(VS, data, i);
                v.lanewise(VectorOperators.TANH).intoArray(data, i);
            }
        }
        for (; i < data.length; i++) data[i] = Math.tanh(data[i]);
    }

    public void reluInPlace() {
        int i = 0;
        if (HAS_VECTOR) {
            var zero = DoubleVector.zero(VS);
            for (; i < VS.loopBound(data.length); i += VLEN) {
                var v = DoubleVector.fromArray(VS, data, i);
                v.max(zero).intoArray(data, i);
            }
        }
        for (; i < data.length; i++) data[i] = Math.max(0, data[i]);
    }

    // GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // Approximation used by BERT/GPT. Vectorizes perfectly.
    public void geluInPlace() {
        final double SQRT_2_OVER_PI = 0.7978845608028654;
        final double COEFF = 0.044715;
        int i = 0;
        if (HAS_VECTOR) {
            var c1 = DoubleVector.broadcast(VS, SQRT_2_OVER_PI);
            var c2 = DoubleVector.broadcast(VS, COEFF);
            var half = DoubleVector.broadcast(VS, 0.5);
            for (; i < VS.loopBound(data.length); i += VLEN) {
                var x = DoubleVector.fromArray(VS, data, i);
                var x3 = x.mul(x).mul(x);
                var inner = x.add(x3.mul(c2)).mul(c1);
                var t = inner.lanewise(VectorOperators.TANH);
                x.mul(half).mul(t.add(1.0)).intoArray(data, i);
            }
        }
        for (; i < data.length; i++) {
            double x = data[i];
            double x3 = x * x * x;
            double t = Math.tanh(SQRT_2_OVER_PI * (x + COEFF * x3));
            data[i] = 0.5 * x * (1.0 + t);
        }
    }

    // --- Fused ops: matmul + activation. This is where you kill PyTorch CPU ---

    // C = GELU(A * B + bias)
    public static void matmulAddBiasGelu(FlatMatrix A, FlatMatrix B, FlatMatrix bias, FlatMatrix C) {
        final int M = A.rows, N = B.cols, K = A.cols;
        if (bias.rows!= 1 || bias.cols!= N) throw new IllegalArgumentException("Bias must be 1xN");
        if (!HAS_VECTOR || M * N * K < 32_768) {
            matmulAddBiasGeluScalar(A.data, B.data, bias.data, C.data, M, N, K);
            return;
        }
        matmulAddBiasGeluSimd(A.data, B.data, bias.data, C.data, M, N, K);
    }

    private static void matmulAddBiasGeluSimd(double[] A, double[] B, double[] bias, double[] C,
                                              int M, int N, int K) {
        final double SQRT_2_OVER_PI = 0.7978845608028654;
        final double COEFF = 0.044715;
        var c1 = DoubleVector.broadcast(VS, SQRT_2_OVER_PI);
        var c2 = DoubleVector.broadcast(VS, COEFF);
        var half = DoubleVector.broadcast(VS, 0.5);
        var one = DoubleVector.broadcast(VS, 1.0);

        for (int i = 0; i < M; i++) {
            int rowA = i * K;
            int rowC = i * N;
            for (int j = 0; j < N; j += VLEN) {
                var acc = DoubleVector.fromArray(VS, bias, j); // start with bias
                for (int k = 0; k < K; k++) {
                    var a = DoubleVector.broadcast(VS, A[rowA + k]);
                    var b = DoubleVector.fromArray(VS, B, k * N + j);
                    acc = a.fma(b, acc);
                }
                // GELU on acc
                var x = acc;
                var x3 = x.mul(x).mul(x);
                var inner = x.add(x3.mul(c2)).mul(c1);
                var t = inner.lanewise(VectorOperators.TANH);
                x.mul(half).mul(t.add(one)).intoArray(C, rowC + j);
            }
        }
    }

    private static void matmulAddBiasGeluScalar(double[] A, double[] B, double[] bias, double[] C,
                                                int M, int N, int K) {
        final double SQRT_2_OVER_PI = 0.7978845608028654;
        final double COEFF = 0.044715;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = bias[j];
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                double x3 = sum * sum * sum;
                double t = Math.tanh(SQRT_2_OVER_PI * (sum + COEFF * x3));
                C[i * N + j] = 0.5 * sum * (1.0 + t);
            }
        }
    }

    // Softmax rows in-place: C[i] = exp(C[i] - max) / sum(exp(C[i] - max))
    public void softmaxRowsInPlace() {
        for (int i = 0; i < rows; i++) {
            int off = i * cols;
            // 1. Find max for numerical stability
            double max = data[off];
            for (int j = 1; j < cols; j++) max = Math.max(max, data[off + j]);
            // 2. exp and sum
            double sum = 0;
            int j = 0;
            if (HAS_VECTOR) {
                var vmax = DoubleVector.broadcast(VS, max);
                var vsum = DoubleVector.zero(VS);
                for (; j < VS.loopBound(cols); j += VLEN) {
                    var v = DoubleVector.fromArray(VS, data, off + j);
                    var e = v.sub(vmax).lanewise(VectorOperators.EXP);
                    e.intoArray(data, off + j);
                    vsum = vsum.add(e);
                }
                sum += vsum.reduceLanes(VectorOperators.ADD);
            }
            for (; j < cols; j++) {
                data[off + j] = Math.exp(data[off + j] - max);
                sum += data[off + j];
            }
            // 3. Normalize
            double invSum = 1.0 / sum;
            j = 0;
            if (HAS_VECTOR) {
                var vinv = DoubleVector.broadcast(VS, invSum);
                for (; j < VS.loopBound(cols); j += VLEN) {
                    var v = DoubleVector.fromArray(VS, data, off + j);
                    v.mul(vinv).intoArray(data, off + j);
                }
            }
            for (; j < cols; j++) data[off + j] *= invSum;
        }
    }
}
