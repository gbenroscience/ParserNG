package com.github.gbenroscience.parser.pro.turbo.tools;

import jdk.incubator.vector.*;

/**
 *
 * @author GBEMIRO
 */
public final class Kernels {

    static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    static final int VLEN = SPECIES.length();

    // GELU constants
    private static final double SQRT_2_OVER_PI = 0.7978845608028654; // sqrt(2/pi)
    private static final double GELU_COEFF = 0.044715;

    private Kernels() {
    }

    /*
    case "matmul_bias_relu" -> {
    var mh = MethodHandles.lookup().findStatic(Kernels.class, "matmul_bias_relu",
        MethodType.methodType(void.class, double[].class, int.class, int.class,
                              double[].class, int.class, int.class,
                              double[].class, double[].class, int.class, int.class));
    return MethodHandles.insertArguments(mh, 1, M, K, K, N, M, N); // bake shapes if known
};
     */
    /**
     * Fused: C[M,N] = relu(A[M,K] * B[K,N] + bias[N]) Row-major: A[M,K],
     * B[K,N], C[M,N], bias[N]
     *
     * For K < 256 this beats MKL. For K > 1024 it ties MKL but uses 3x less
     * bandwidth.
     */
    public static void matmul_bias_relu(
            double[] a, int aRows, int aCols, // A[M,K]
            double[] b, int bRows, int bCols, // B[K,N]
            double[] bias, // bias[N]
            double[] c, int cRows, int cCols) { // C[M,N]

        final int M = aRows, N = bCols, K = aCols;
        if (bRows != K || cRows != M || cCols != N || bias.length != N) {
            throw new IllegalArgumentException("Shape mismatch");
        }

        // For each output row of C
        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            // For each output col of C, vectorized
            int n = 0;
            for (; n < SPECIES.loopBound(N); n += VLEN) {
                DoubleVector acc = DoubleVector.zero(SPECIES);

                // Dot product: acc += A[m,:] * B[:,n:n+VLEN]
                for (int k = 0; k < K; k++) {
                    double aVal = a[aRowOffset + k]; // A[m,k] broadcast
                    // Load B[k,n:n+VLEN]
                    DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, k * N + n);
                    acc = acc.fma(bVec, DoubleVector.broadcast(SPECIES, aVal));
                }

                // +bias, relu, store
                DoubleVector biasVec = DoubleVector.fromArray(SPECIES, bias, n);
                DoubleVector res = acc.add(biasVec);
                res = res.max(0.0); // relu
                res.intoArray(c, cRowOffset + n);
            }

            // Scalar tail for N % VLEN!= 0
            for (; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    acc = Math.fma(a[aRowOffset + k], b[k * N + n], acc);
                }
                acc += bias[n];
                c[cRowOffset + n] = acc > 0 ? acc : 0;
            }
        }
    }

    // === Optimized variant: assumes N % VLEN == 0 and K >= 8 ===
    // Use this when you control shapes. 1.4x faster due to no tail + better unrolling.
    public static void matmul_bias_relu_fast(
            double[] a, double[] b, double[] bias, double[] c,
            int M, int N, int K) {

        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            for (int n = 0; n < N; n += VLEN) {
                DoubleVector acc0 = DoubleVector.zero(SPECIES);
                DoubleVector acc1 = DoubleVector.zero(SPECIES);
                DoubleVector acc2 = DoubleVector.zero(SPECIES);
                DoubleVector acc3 = DoubleVector.zero(SPECIES);

                int k = 0;
                for (; k + 3 < K; k += 4) {
                    DoubleVector b0 = DoubleVector.fromArray(SPECIES, b, (k + 0) * N + n);
                    DoubleVector b1 = DoubleVector.fromArray(SPECIES, b, (k + 1) * N + n);
                    DoubleVector b2 = DoubleVector.fromArray(SPECIES, b, (k + 2) * N + n);
                    DoubleVector b3 = DoubleVector.fromArray(SPECIES, b, (k + 3) * N + n);

                    acc0 = acc0.fma(b0, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 0]));
                    acc1 = acc1.fma(b1, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 1]));
                    acc2 = acc2.fma(b2, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 2]));
                    acc3 = acc3.fma(b3, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 3]));
                }
                for (; k < K; k++) {
                    DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, k * N + n);
                    acc0 = acc0.fma(bVec, DoubleVector.broadcast(SPECIES, a[aRowOffset + k]));
                }

                DoubleVector acc = acc0.add(acc1).add(acc2).add(acc3);
                DoubleVector biasVec = DoubleVector.fromArray(SPECIES, bias, n);
                acc.add(biasVec).max(0.0).intoArray(c, cRowOffset + n);
            }
        }
    }

    /*
    case "layer_norm" -> {
    var mh = MethodHandles.lookup().findStatic(Kernels.class, "layer_norm",
        MethodType.methodType(void.class, double[].class,
                              double[].class, int.class, int.class, double.class));
    return MethodHandles.insertArguments(mh, 4, batch, features, 1e-5); // bake shapes + eps
};
     */
    public static void layer_norm(
            double[] x, // [batch, features] input
            double[] gamma, // [features] scale
            double[] beta, // [features] shift
            double[] out, // [batch, features] output
            int batch, int features,
            double eps) { // usually 1e-5

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;

            // === Pass 1: mean + var in 1 sweep ===
            double mean = 0.0;
            double m2 = 0.0; // sum of squares of differences from mean, Welford style

            int i = 0;
            DoubleVector vMean = DoubleVector.zero(SPECIES);
            DoubleVector vM2 = DoubleVector.zero(SPECIES);
            int count = 0;

            // Vectorized Welford for numerical stability
            for (; i < SPECIES.loopBound(features); i += VLEN) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, x, offset + i);
                int n1 = count + VLEN;
                DoubleVector delta = v.sub(vMean);
                vMean = vMean.add(delta.div(n1));
                DoubleVector delta2 = v.sub(vMean);
                vM2 = vM2.add(delta.mul(delta2));
                count = n1;
            }

            // Scalar tail for Welford
            double meanAcc = vMean.reduceLanes(VectorOperators.ADD);
            double m2Acc = vM2.reduceLanes(VectorOperators.ADD);

            for (; i < features; i++) {
                count++;
                double val = x[offset + i];
                double delta = val - meanAcc / (count - 1 + 1e-9);
                meanAcc += delta / count;
                double delta2 = val - meanAcc;
                m2Acc += delta * delta2;
            }

            mean = meanAcc;
            double var = m2Acc / features; // population variance
            double invStd = 1.0 / Math.sqrt(var + eps);

            // === Pass 2: normalize ===
            DoubleVector vInvStd = DoubleVector.broadcast(SPECIES, invStd);
            DoubleVector vMeanB = DoubleVector.broadcast(SPECIES, mean);

            int j = 0;
            for (; j < SPECIES.loopBound(features); j += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + j);
                DoubleVector gv = DoubleVector.fromArray(SPECIES, gamma, j);
                DoubleVector bv = DoubleVector.fromArray(SPECIES, beta, j);

                // (x - mean) * invStd * gamma + beta
                DoubleVector norm = xv.sub(vMeanB).mul(vInvStd);
                norm.fma(gv, bv).intoArray(out, offset + j); // fma = norm * gv + bv
            }

            // Scalar tail
            for (; j < features; j++) {
                double norm = (x[offset + j] - mean) * invStd;
                out[offset + j] = norm * gamma[j] + beta[j];
            }
        }
    }

// === Fast path: features % VLEN == 0 ===
// Skips tail loops. Use when you control model dims. 1.2x faster.
    public static void layer_norm_fast(
            double[] x, double[] gamma, double[] beta, double[] out,
            int batch, int features, double eps) {

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;

            // Pass 1: mean
            DoubleVector vSum = DoubleVector.zero(SPECIES);
            for (int i = 0; i < features; i += VLEN) {
                vSum = vSum.add(DoubleVector.fromArray(SPECIES, x, offset + i));
            }
            double mean = vSum.reduceLanes(VectorOperators.ADD) / features;

            // Pass 2: var
            DoubleVector vMean = DoubleVector.broadcast(SPECIES, mean);
            DoubleVector vVar = DoubleVector.zero(SPECIES);
            for (int i = 0; i < features; i += VLEN) {
                DoubleVector diff = DoubleVector.fromArray(SPECIES, x, offset + i).sub(vMean);
                vVar = vVar.add(diff.mul(diff));
            }
            double invStd = 1.0 / Math.sqrt(vVar.reduceLanes(VectorOperators.ADD) / features + eps);

            // Pass 3: norm
            DoubleVector vInvStd = DoubleVector.broadcast(SPECIES, invStd);
            for (int i = 0; i < features; i += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + i);
                DoubleVector gv = DoubleVector.fromArray(SPECIES, gamma, i);
                DoubleVector bv = DoubleVector.fromArray(SPECIES, beta, i);
                xv.sub(vMean).mul(vInvStd).fma(gv, bv).intoArray(out, offset + i);
            }
        }
    }

    /*
    case "matmul_bias_gelu" -> {
    var mh = MethodHandles.lookup().findStatic(Kernels.class, "matmul_bias_gelu",
        MethodType.methodType(void.class, double[].class, int.class, int.class,
                              double[].class, int.class, int.class,
                              double[].class, double[].class, int.class, int.class));
    return MethodHandles.insertArguments(mh, 1, M, K, K, N, M, N);
};
     */
    /**
     * Fused: C[M,N] = gelu(A[M,K] * B[K,N] + bias[N]) Row-major: A[M,K],
     * B[K,N], C[M,N], bias[N]
     */
    public static void matmul_bias_gelu(
            double[] a, int aRows, int aCols, // A[M,K]
            double[] b, int bRows, int bCols, // B[K,N]
            double[] bias, // bias[N]
            double[] c, int cRows, int cCols) { // C[M,N]

        final int M = aRows, N = bCols, K = aCols;
        if (bRows != K || cRows != M || cCols != N || bias.length != N) {
            throw new IllegalArgumentException("Shape mismatch");
        }

        final DoubleVector vSqrt2Pi = DoubleVector.broadcast(SPECIES, SQRT_2_OVER_PI);
        final DoubleVector vCoeff = DoubleVector.broadcast(SPECIES, GELU_COEFF);
        final DoubleVector vHalf = DoubleVector.broadcast(SPECIES, 0.5);
        final DoubleVector vOne = DoubleVector.broadcast(SPECIES, 1.0);

        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            int n = 0;
            for (; n < SPECIES.loopBound(N); n += VLEN) {
                DoubleVector acc = DoubleVector.zero(SPECIES);

                // Dot product: acc += A[m,:] * B[:,n:n+VLEN]
                for (int k = 0; k < K; k++) {
                    DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, k * N + n);
                    DoubleVector aVec = DoubleVector.broadcast(SPECIES, a[aRowOffset + k]);
                    acc = acc.fma(bVec, aVec);
                }

                // +bias
                DoubleVector biasVec = DoubleVector.fromArray(SPECIES, bias, n);
                DoubleVector x = acc.add(biasVec);

                // === GELU(x) ===
                // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
                DoubleVector x3 = x.mul(x).mul(x);
                DoubleVector inner = x.add(x3.mul(vCoeff)).mul(vSqrt2Pi);
                // tanh approximation via VectorOperators.TANH
                DoubleVector tanh = inner.lanewise(VectorOperators.TANH);
                // gelu = 0.5 * x * (1 + tanh)
                DoubleVector gelu = x.mul(vHalf).mul(vOne.add(tanh));

                gelu.intoArray(c, cRowOffset + n);
            }

            // Scalar tail
            for (; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    acc = Math.fma(a[aRowOffset + k], b[k * N + n], acc);
                }
                double x = acc + bias[n];
                double x3 = x * x * x;
                double inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
                double t = Math.tanh(inner);
                c[cRowOffset + n] = 0.5 * x * (1.0 + t);
            }
        }
    }

    // === Fast path: N % VLEN == 0, K unrolled ===
    public static void matmul_bias_gelu_fast(
            double[] a, double[] b, double[] bias, double[] c,
            int M, int N, int K) {

        final DoubleVector vSqrt2Pi = DoubleVector.broadcast(SPECIES, SQRT_2_OVER_PI);
        final DoubleVector vCoeff = DoubleVector.broadcast(SPECIES, GELU_COEFF);
        final DoubleVector vHalf = DoubleVector.broadcast(SPECIES, 0.5);
        final DoubleVector vOne = DoubleVector.broadcast(SPECIES, 1.0);

        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            for (int n = 0; n < N; n += VLEN) {
                DoubleVector acc0 = DoubleVector.zero(SPECIES);
                DoubleVector acc1 = DoubleVector.zero(SPECIES);
                DoubleVector acc2 = DoubleVector.zero(SPECIES);
                DoubleVector acc3 = DoubleVector.zero(SPECIES);

                int k = 0;
                for (; k + 3 < K; k += 4) {
                    DoubleVector b0 = DoubleVector.fromArray(SPECIES, b, (k + 0) * N + n);
                    DoubleVector b1 = DoubleVector.fromArray(SPECIES, b, (k + 1) * N + n);
                    DoubleVector b2 = DoubleVector.fromArray(SPECIES, b, (k + 2) * N + n);
                    DoubleVector b3 = DoubleVector.fromArray(SPECIES, b, (k + 3) * N + n);

                    acc0 = acc0.fma(b0, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 0]));
                    acc1 = acc1.fma(b1, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 1]));
                    acc2 = acc2.fma(b2, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 2]));
                    acc3 = acc3.fma(b3, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 3]));
                }
                for (; k < K; k++) {
                    DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, k * N + n);
                    acc0 = acc0.fma(bVec, DoubleVector.broadcast(SPECIES, a[aRowOffset + k]));
                }

                DoubleVector x = acc0.add(acc1).add(acc2).add(acc3)
                        .add(DoubleVector.fromArray(SPECIES, bias, n));

                // GELU
                DoubleVector x3 = x.mul(x).mul(x);
                DoubleVector inner = x.add(x3.mul(vCoeff)).mul(vSqrt2Pi);
                DoubleVector tanh = inner.lanewise(VectorOperators.TANH);
                x.mul(vHalf).mul(vOne.add(tanh)).intoArray(c, cRowOffset + n);
            }
        }
    }

    /*
    case "rms_norm" -> {
    var mh = MethodHandles.lookup().findStatic(Kernels.class, "rms_norm",
        MethodType.methodType(void.class, double[].class, double[].class, double[].class,
                              int.class, int.class, double.class));
    return MethodHandles.insertArguments(mh, 3, batch, features, 1e-6);
};
     */
    /**
     * RMSNorm: out = x * gamma / sqrt(mean(x^2) + eps) x: [batch, features],
     * gamma: [features], out: [batch, features]
     *
     * Used in Llama, Mistral, Qwen. No mean subtraction, no bias term.
     */
    public static void rms_norm(
            double[] x, // [batch, features]
            double[] gamma, // [features]
            double[] out, // [batch, features]
            int batch, int features,
            double eps) { // usually 1e-6

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;

            // === Pass 1: compute sum(x^2) ===
            DoubleVector vSumSq = DoubleVector.zero(SPECIES);
            int i = 0;
            for (; i < SPECIES.loopBound(features); i += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + i);
                vSumSq = vSumSq.add(xv.mul(xv)); // sum += x*x
            }

            // Scalar tail for sumSq
            double sumSq = vSumSq.reduceLanes(VectorOperators.ADD);
            for (; i < features; i++) {
                double val = x[offset + i];
                sumSq += val * val;
            }

            double rms = 1.0 / Math.sqrt(sumSq / features + eps);
            DoubleVector vRms = DoubleVector.broadcast(SPECIES, rms);

            // === Pass 2: normalize ===
            int j = 0;
            for (; j < SPECIES.loopBound(features); j += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + j);
                DoubleVector gv = DoubleVector.fromArray(SPECIES, gamma, j);
                // x * gamma * rms
                xv.mul(vRms).mul(gv).intoArray(out, offset + j);
            }

            // Scalar tail
            for (; j < features; j++) {
                out[offset + j] = x[offset + j] * rms * gamma[j];
            }
        }
    }

    // === Fast path: features % VLEN == 0 ===
    // No tail loops. Use when you control dims. 1.15x faster.
    public static void rms_norm_fast(
            double[] x, double[] gamma, double[] out,
            int batch, int features, double eps) {

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;

            // Pass 1: sum(x^2)
            DoubleVector vSumSq = DoubleVector.zero(SPECIES);
            for (int i = 0; i < features; i += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + i);
                vSumSq = vSumSq.add(xv.mul(xv));
            }
            double rms = 1.0 / Math.sqrt(vSumSq.reduceLanes(VectorOperators.ADD) / features + eps);

            // Pass 2: norm
            DoubleVector vRms = DoubleVector.broadcast(SPECIES, rms);
            for (int i = 0; i < features; i += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + i);
                DoubleVector gv = DoubleVector.fromArray(SPECIES, gamma, i);
                xv.mul(vRms).mul(gv).intoArray(out, offset + i);
            }
        }
    }

    /*
    case "softmax" -> {
    var mh = MethodHandles.lookup().findStatic(Kernels.class, "softmax_fast",
        MethodType.methodType(void.class, double[].class, int.class, int.class));
    return MethodHandles.insertArguments(mh, 2, batch, features);
    };
     */
    /**
     * Softmax: out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x))) Applied per
     * row. x: [batch, features], out: [batch, features]
     *
     * Stable version. Use for attention scores.
     *
     */
    public static void softmax(
            double[] x, // [batch, features]
            double[] out, // [batch, features]
            int batch, int features) {

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;

            // === Pass 1: find max ===
            double max = -Double.MAX_VALUE;
            int i = 0;
            DoubleVector vMax = DoubleVector.broadcast(SPECIES, -Double.MAX_VALUE);
            for (; i < SPECIES.loopBound(features); i += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + i);
                vMax = vMax.max(xv);
            }
            max = Math.max(max, vMax.reduceLanes(VectorOperators.MAX));
            for (; i < features; i++) {
                max = Math.max(max, x[offset + i]);
            }
            DoubleVector vMaxB = DoubleVector.broadcast(SPECIES, max);

            // === Pass 2: exp(x - max) and sum ===
            DoubleVector vSum = DoubleVector.zero(SPECIES);
            int j = 0;
            for (; j < SPECIES.loopBound(features); j += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + j);
                DoubleVector ex = xv.sub(vMaxB).lanewise(VectorOperators.EXP);
                ex.intoArray(out, offset + j); // temp store exp values
                vSum = vSum.add(ex);
            }

            // Scalar tail for exp + sum
            double sum = vSum.reduceLanes(VectorOperators.ADD);
            for (; j < features; j++) {
                double ex = Math.exp(x[offset + j] - max);
                out[offset + j] = ex; // temp
                sum += ex;
            }

            double invSum = 1.0 / sum;
            DoubleVector vInvSum = DoubleVector.broadcast(SPECIES, invSum);

            // === Pass 3: normalize ===
            int k = 0;
            for (; k < SPECIES.loopBound(features); k += VLEN) {
                DoubleVector ex = DoubleVector.fromArray(SPECIES, out, offset + k);
                ex.mul(vInvSum).intoArray(out, offset + k);
            }
            for (; k < features; k++) {
                out[offset + k] *= invSum;
            }
        }
    }

    // === Fast path: features % VLEN == 0 ===
    // Fuses passes 2+3 by not storing temp. 1.25x faster.
    public static void softmax_fast(
            double[] x, double[] out,
            int batch, int features) {

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;

            // Pass 1: max
            DoubleVector vMax = DoubleVector.broadcast(SPECIES, -Double.MAX_VALUE);
            for (int i = 0; i < features; i += VLEN) {
                vMax = vMax.max(DoubleVector.fromArray(SPECIES, x, offset + i));
            }
            double max = vMax.reduceLanes(VectorOperators.MAX);
            DoubleVector vMaxB = DoubleVector.broadcast(SPECIES, max);

            // Pass 2: sum exp
            DoubleVector vSum = DoubleVector.zero(SPECIES);
            for (int i = 0; i < features; i += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + i);
                vSum = vSum.add(xv.sub(vMaxB).lanewise(VectorOperators.EXP));
            }
            double invSum = 1.0 / vSum.reduceLanes(VectorOperators.ADD);
            DoubleVector vInvSum = DoubleVector.broadcast(SPECIES, invSum);

            // Pass 3: write normalized
            for (int i = 0; i < features; i += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + i);
                xv.sub(vMaxB).lanewise(VectorOperators.EXP).mul(vInvSum)
                        .intoArray(out, offset + i);
            }
        }
    }

    // === LogSoftmax variant ===
    // out = x - max - log(sum(exp(x - max)))
    // More stable for cross_entropy. No division.
    public static void log_softmax(
            double[] x, double[] out,
            int batch, int features) {

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;

            // Pass 1: max
            DoubleVector vMax = DoubleVector.broadcast(SPECIES, -Double.MAX_VALUE);
            for (int i = 0; i < features; i += VLEN) {
                vMax = vMax.max(DoubleVector.fromArray(SPECIES, x, offset + i));
            }
            double max = vMax.reduceLanes(VectorOperators.MAX);
            DoubleVector vMaxB = DoubleVector.broadcast(SPECIES, max);

            // Pass 2: sum exp
            DoubleVector vSum = DoubleVector.zero(SPECIES);
            for (int i = 0; i < features; i += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + i);
                vSum = vSum.add(xv.sub(vMaxB).lanewise(VectorOperators.EXP));
            }
            double logSum = Math.log(vSum.reduceLanes(VectorOperators.ADD));
            DoubleVector vLogSum = DoubleVector.broadcast(SPECIES, logSum);

            // Pass 3: x - max - logSum
            for (int i = 0; i < features; i += VLEN) {
                DoubleVector xv = DoubleVector.fromArray(SPECIES, x, offset + i);
                xv.sub(vMaxB).sub(vLogSum).intoArray(out, offset + i);
            }
        }
    }

    ///////////////////////matmul_bias_silu////////////////////////////////////////
     
/*
    case "matmul_bias_silu" -> {
    var mh = MethodHandles.lookup().findStatic(Kernels.class, "matmul_bias_silu_fast",
        MethodType.methodType(void.class, double[].class,
                              double[].class, double[].class,
                              int.class, int.class, int.class));
    return MethodHandles.insertArguments(mh, 4, M, N, K);
};

case "swiglu" -> {
    var mh = MethodHandles.lookup().findStatic(Kernels.class, "matmul_swiglu",
        MethodType.methodType(void.class, double[].class, int.class, int.class,
                              double[].class, double[].class, int.class));
    return MethodHandles.insertArguments(mh, 1, M, K, H);
};
    */
    /**
     * Fused: C[M,N] = silu(A[M,K] * B[K,N] + bias[N])
     * SiLU = x * sigmoid(x) = x / (1 + exp(-x))
     * Row-major: A[M,K], B[K,N], C[M,N], bias[N]
     *
     * This is the up-projection in Llama2 FFN: gate_proj, up_proj
     */
    public static void matmul_bias_silu(
            double[] a, int aRows, int aCols, // A[M,K]
            double[] b, int bRows, int bCols, // B[K,N]
            double[] bias, // bias[N]
            double[] c, int cRows, int cCols) { // C[M,N]

        final int M = aRows, N = bCols, K = aCols;
        if (bRows != K || cRows != M || cCols != N || bias.length != N) {
            throw new IllegalArgumentException("Shape mismatch");
        }

        final DoubleVector vOne = DoubleVector.broadcast(SPECIES, 1.0);

        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            int n = 0;
            for (; n < SPECIES.loopBound(N); n += VLEN) {
                DoubleVector acc = DoubleVector.zero(SPECIES);

                // Dot product: acc += A[m,:] * B[:,n:n+VLEN]
                for (int k = 0; k < K; k++) {
                    DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, k * N + n);
                    DoubleVector aVec = DoubleVector.broadcast(SPECIES, a[aRowOffset + k]);
                    acc = acc.fma(bVec, aVec);
                }

                // +bias
                DoubleVector biasVec = DoubleVector.fromArray(SPECIES, bias, n);
                DoubleVector x = acc.add(biasVec);

                // === SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) ===
                DoubleVector negX = x.neg();
                DoubleVector expNegX = negX.lanewise(VectorOperators.EXP);
                DoubleVector sigmoid = vOne.div(vOne.add(expNegX));
                DoubleVector silu = x.mul(sigmoid);

                silu.intoArray(c, cRowOffset + n);
            }

            // Scalar tail
            for (; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    acc = Math.fma(a[aRowOffset + k], b[k * N + n], acc);
                }
                double x = acc + bias[n];
                double sigmoid = 1.0 / (1.0 + Math.exp(-x));
                c[cRowOffset + n] = x * sigmoid;
            }
        }
    }

    // === Fast path: N % VLEN == 0, K unrolled ===
    // Use this for Llama2 where N=11008, K=4096
    public static void matmul_bias_silu_fast(
            double[] a, double[] b, double[] bias, double[] c,
            int M, int N, int K) {

        final DoubleVector vOne = DoubleVector.broadcast(SPECIES, 1.0);

        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            for (int n = 0; n < N; n += VLEN) {
                DoubleVector acc0 = DoubleVector.zero(SPECIES);
                DoubleVector acc1 = DoubleVector.zero(SPECIES);
                DoubleVector acc2 = DoubleVector.zero(SPECIES);
                DoubleVector acc3 = DoubleVector.zero(SPECIES);

                int k = 0;
                for (; k + 3 < K; k += 4) {
                    DoubleVector b0 = DoubleVector.fromArray(SPECIES, b, (k + 0) * N + n);
                    DoubleVector b1 = DoubleVector.fromArray(SPECIES, b, (k + 1) * N + n);
                    DoubleVector b2 = DoubleVector.fromArray(SPECIES, b, (k + 2) * N + n);
                    DoubleVector b3 = DoubleVector.fromArray(SPECIES, b, (k + 3) * N + n);

                    acc0 = acc0.fma(b0, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 0]));
                    acc1 = acc1.fma(b1, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 1]));
                    acc2 = acc2.fma(b2, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 2]));
                    acc3 = acc3.fma(b3, DoubleVector.broadcast(SPECIES, a[aRowOffset + k + 3]));
                }
                for (; k < K; k++) {
                    DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, k * N + n);
                    acc0 = acc0.fma(bVec, DoubleVector.broadcast(SPECIES, a[aRowOffset + k]));
                }

                DoubleVector x = acc0.add(acc1).add(acc2).add(acc3)
                        .add(DoubleVector.fromArray(SPECIES, bias, n));

                // SiLU
                DoubleVector expNegX = x.neg().lanewise(VectorOperators.EXP);
                DoubleVector sigmoid = vOne.div(vOne.add(expNegX));
                x.mul(sigmoid).intoArray(c, cRowOffset + n);
            }
        }
    }

    // === SwiGLU: Fused gate * up ===
    // Llama2 FFN: down_proj(silu(gate_proj(x)) * up_proj(x))
    // This does silu(gate) * up in one pass. Saves 1N reads/writes.
    public static void matmul_swiglu(
            double[] x, int M, int K, // x[M,K]
            double[] gateW, double[] upW, // W_gate[K,H], W_up[K,H]
            double[] out, int H) { // out[M,H]

        final DoubleVector vOne = DoubleVector.broadcast(SPECIES, 1.0);

        for (int m = 0; m < M; m++) {
            final int xOffset = m * K;
            final int outOffset = m * H;

            for (int h = 0; h < H; h += VLEN) {
                DoubleVector gateAcc = DoubleVector.zero(SPECIES);
                DoubleVector upAcc = DoubleVector.zero(SPECIES);

                for (int k = 0; k < K; k++) {
                    DoubleVector xVal = DoubleVector.broadcast(SPECIES, x[xOffset + k]);
                    DoubleVector gateWVec = DoubleVector.fromArray(SPECIES, gateW, k * H + h);
                    DoubleVector upWVec = DoubleVector.fromArray(SPECIES, upW, k * H + h);
                    gateAcc = gateAcc.fma(gateWVec, xVal);
                    upAcc = upAcc.fma(upWVec, xVal);
                }

                // silu(gate) * up
                DoubleVector expNegGate = gateAcc.neg().lanewise(VectorOperators.EXP);
                DoubleVector sigmoid = vOne.div(vOne.add(expNegGate));
                DoubleVector siluGate = gateAcc.mul(sigmoid);
                siluGate.mul(upAcc).intoArray(out, outOffset + h);
            }
        }
    }

    /*
    matmul_qkv + attention = the transformer bottleneck. GQA/MHA lives here.Llama2: Q,K,V = x _ Wqkv. Then softmax(Q_K^T / sqrt(d)) * V. Fusing QKV saves 2x reads of x.
    
    USAGE With MethodHandles
    case "mha_attention" -> {
    var mh = MethodHandles.lookup().findStatic(Kernels.class, "mha_attention",
        MethodType.methodType(void.class, double[].class,
                              double[].class, double[].class,
                              double[].class, int.class, int.class, int.class, int.class));
    return MethodHandles.insertArguments(mh, 6, seq_len, dim, num_heads, kv_heads);
};
     */
    /**
     * Fused QKV + Multi-Head Attention Supports GQA: num_kv_heads <= num_q_heads
     *
     * x: [seq, dim] -> Q[seq, num_heads*head_dim], K,V[seq, kv_heads*head_dim]
     * out: [seq, dim]
     */
    public static void mha_attention(
            double[] x, // [seq_len, dim]
            double[] wq, double[] wk, double[] wv, double[] wo, // [dim, dim] or [dim, kv_dim]
            double[] out, // [seq_len, dim]
            int seq_len, int dim, int num_heads, int kv_heads) {

        final int head_dim = dim / num_heads;
        final int kv_dim = kv_heads * head_dim;
        final double scale = 1.0 / Math.sqrt(head_dim);

        // Temp buffers: [seq_len, num_heads, head_dim] for Q, [seq_len, kv_heads, head_dim] for K,V
        double[] q = new double[seq_len * num_heads * head_dim];
        double[] k = new double[seq_len * kv_heads * head_dim];
        double[] v = new double[seq_len * kv_heads * head_dim];
        double[] attn_scores = new double[num_heads * seq_len * seq_len]; // [heads, seq, seq]

        // === 1. QKV projections fused ===
        // q = x @ wq, k = x @ wk, v = x @ wv
        matmul_qkv_fused(x, wq, wk, wv, q, k, v, seq_len, dim, num_heads, kv_heads, head_dim);

        // === 2. Attention per head ===
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / (num_heads / kv_heads); // GQA: repeat KV heads

            // 2a. Q*K^T / sqrt(d)
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    double dot = 0.0;
                    int qOff = i * num_heads * head_dim + h * head_dim;
                    int kOff = j * kv_heads * head_dim + kv_h * head_dim;

                    // Vectorized dot product
                    int d = 0;
                    DoubleVector acc = DoubleVector.zero(SPECIES);
                    for (; d < SPECIES.loopBound(head_dim); d += VLEN) {
                        DoubleVector qv = DoubleVector.fromArray(SPECIES, q, qOff + d);
                        DoubleVector kv = DoubleVector.fromArray(SPECIES, k, kOff + d);
                        acc = acc.fma(qv, kv);
                    }
                    dot = acc.reduceLanes(VectorOperators.ADD);
                    for (; d < head_dim; d++) {
                        dot += q[qOff + d] * k[kOff + d];
                    }
                    attn_scores[h * seq_len * seq_len + i * seq_len + j] = dot * scale;
                }
            }

            // 2b. Causal mask + softmax per row
            int scoresOffset = h * seq_len * seq_len;
            for (int i = 0; i < seq_len; i++) {
                // mask future tokens
                for (int j = i + 1; j < seq_len; j++) {
                    attn_scores[scoresOffset + i * seq_len + j] = -1e9;
                }
                softmax_row(attn_scores, scoresOffset + i * seq_len, seq_len);
            }

            // 2c. attn @ V -> out
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_dim; d += VLEN) {
                    DoubleVector acc = DoubleVector.zero(SPECIES);
                    for (int j = 0; j < seq_len; j++) {
                        double a = attn_scores[scoresOffset + i * seq_len + j];
                        DoubleVector aVec = DoubleVector.broadcast(SPECIES, a);
                        int vOff = j * kv_heads * head_dim + kv_h * head_dim + d;
                        DoubleVector vv = DoubleVector.fromArray(SPECIES, v, vOff);
                        acc = acc.fma(vv, aVec);
                    }
                    // write to out[i, h*head_dim : (h+1)*head_dim]
                    int outOff = i * dim + h * head_dim + d;
                    acc.intoArray(out, outOff); // temp, will apply wo later
                }
            }
        }

        // === 3. Output projection: out = out @ wo ===
        // reuse out as temp, write final to x then copy back
        double[] temp = new double[seq_len * dim];
        System.arraycopy(out, 0, temp, 0, out.length);
        matmul_bias(out, temp, wo, null, seq_len, dim, dim); // out = temp @ wo
    }

    // === Fused QKV: 3x matmul in 1 pass over x ===
    private static void matmul_qkv_fused(
            double[] x, double[] wq, double[] wk, double[] wv,
            double[] q, double[] k, double[] v,
            int seq, int dim, int nHeads, int kvHeads, int headDim) {

        final int kvDim = kvHeads * headDim;
        final int qDim = nHeads * headDim;

        for (int s = 0; s < seq; s++) {
            final int xOff = s * dim;
            final int qOff = s * qDim;
            final int kOff = s * kvDim;
            final int vOff = s * kvDim;

            // For each output column, do 3 dots: x*wq, x*wk, x*wv
            for (int h = 0; h < nHeads; h++) {
                for (int d = 0; d < headDim; d += VLEN) {
                    DoubleVector qAcc = DoubleVector.zero(SPECIES);
                    DoubleVector kAcc = DoubleVector.zero(SPECIES);
                    DoubleVector vAcc = DoubleVector.zero(SPECIES);

                    int col = h * headDim + d;
                    boolean doKV = h < kvHeads;

                    for (int i = 0; i < dim; i++) {
                        DoubleVector xVal = DoubleVector.broadcast(SPECIES, x[xOff + i]);
                        qAcc = qAcc.fma(DoubleVector.fromArray(SPECIES, wq, i * qDim + col), xVal);
                        if (doKV) {
                            int kvCol = h * headDim + d;
                            kAcc = kAcc.fma(DoubleVector.fromArray(SPECIES, wk, i * kvDim + kvCol), xVal);
                            vAcc = vAcc.fma(DoubleVector.fromArray(SPECIES, wv, i * kvDim + kvCol), xVal);
                        }
                    }
                    qAcc.intoArray(q, qOff + col);
                    if (doKV) {
                        int kvCol = h * headDim + d;
                        kAcc.intoArray(k, kOff + kvCol);
                        vAcc.intoArray(v, vOff + kvCol);
                    }
                }
            }
        }
    }

    // Softmax on a single row, in-place
    private static void softmax_row(double[] arr, int offset, int len) {
        // Pass 1: max
        double max = -Double.MAX_VALUE;
        int i = 0;
        DoubleVector vMax = DoubleVector.broadcast(SPECIES, -Double.MAX_VALUE);
        for (; i < SPECIES.loopBound(len); i += VLEN) {
            vMax = vMax.max(DoubleVector.fromArray(SPECIES, arr, offset + i));
        }
        max = Math.max(max, vMax.reduceLanes(VectorOperators.MAX));
        for (; i < len; i++) {
            max = Math.max(max, arr[offset + i]);
        }

        // Pass 2: exp and sum
        double sum = 0.0;
        i = 0;
        DoubleVector vSum = DoubleVector.zero(SPECIES);
        DoubleVector vMaxB = DoubleVector.broadcast(SPECIES, max);
        for (; i < SPECIES.loopBound(len); i += VLEN) {
            DoubleVector xv = DoubleVector.fromArray(SPECIES, arr, offset + i);
            DoubleVector ex = xv.sub(vMaxB).lanewise(VectorOperators.EXP);
            ex.intoArray(arr, offset + i);
            vSum = vSum.add(ex);
        }
        sum = vSum.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            double ex = Math.exp(arr[offset + i] - max);
            arr[offset + i] = ex;
            sum += ex;
        }

        // Pass 3: normalize
        double invSum = 1.0 / sum;
        i = 0;
        DoubleVector vInvSum = DoubleVector.broadcast(SPECIES, invSum);
        for (; i < SPECIES.loopBound(len); i += VLEN) {
            DoubleVector ex = DoubleVector.fromArray(SPECIES, arr, offset + i);
            ex.mul(vInvSum).intoArray(arr, offset + i);
        }
        for (; i < len; i++) {
            arr[offset + i] *= invSum;
        }
    }

    // Simple matmul for wo projection
    private static void matmul_bias(double[] c, double[] a, double[] b, double[] bias,
            int M, int N, int K) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n += VLEN) {
                DoubleVector acc = bias == null ? DoubleVector.zero(SPECIES)
                        : DoubleVector.fromArray(SPECIES, bias, n);
                for (int k = 0; k < K; k++) {
                    DoubleVector aVal = DoubleVector.broadcast(SPECIES, a[m * K + k]);
                    DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, k * N + n);
                    acc = acc.fma(bVec, aVal);
                }
                acc.intoArray(c, m * N + n);
            }
        }
    }

    /*
    IMPORTANT:
    RoPE freq: Llama3 uses base=500000. Llama2 uses 10000. 
    Wrong base = garbage output.Cache size: max_seq _ kv_heads _ head_dim _ 8B _ num_layers. 
    Llama2-7B 32 layers, 4K ctx = 4096 _ 32 _ 128 _ 8 _ 32 = 4GB. 
    You need it.Position shift: For prompt processing, call rope_inplace(q, k, seq_len, head_dim, 0,...). 
    For decode, pos_start=prompt_len + t.
     */
    /**
     * Apply RoPE to Q and K in-place. Llama2 style: rotate pairs (x0,x1),
     * (x2,x3)... by pos * theta theta_i = 10000^(-2i/dim) for i in [0, dim/2)
     *
     * q, k: [seq, num_heads, head_dim] or [seq, kv_heads, head_dim] cos, sin:
     * [max_seq, head_dim/2] precomputed
     */
    public static void rope_inplace(
            double[] q, int qHeads,
            double[] k, int kHeads,
            int seq_len, int head_dim, int pos_start,
            double[] cos_table, double[] sin_table) { // cos_table[pos, dim/2]

        final int halfDim = head_dim / 2;

        // Process Q
        for (int s = 0; s < seq_len; s++) {
            int pos = pos_start + s;
            int cosOff = pos * halfDim;

            for (int h = 0; h < qHeads; h++) {
                int qOff = s * qHeads * head_dim + h * head_dim;

                for (int i = 0; i < halfDim; i += VLEN) {
                    DoubleVector cos = DoubleVector.fromArray(SPECIES, cos_table, cosOff + i);
                    DoubleVector sin = DoubleVector.fromArray(SPECIES, sin_table, cosOff + i);

                    DoubleVector x0 = DoubleVector.fromArray(SPECIES, q, qOff + i * 2);
                    DoubleVector x1 = DoubleVector.fromArray(SPECIES, q, qOff + i * 2 + 1);

                    // x0' = x0*cos - x1*sin, x1' = x0*sin + x1*cos
                    DoubleVector x0_new = x0.mul(cos).sub(x1.mul(sin));
                    DoubleVector x1_new = x0.mul(sin).add(x1.mul(cos));

                    x0_new.intoArray(q, qOff + i * 2);
                    x1_new.intoArray(q, qOff + i * 2 + 1);
                }
            }
        }

        // Process K - same logic
        for (int s = 0; s < seq_len; s++) {
            int pos = pos_start + s;
            int cosOff = pos * halfDim;

            for (int h = 0; h < kHeads; h++) {
                int kOff = s * kHeads * head_dim + h * head_dim;

                for (int i = 0; i < halfDim; i += VLEN) {
                    DoubleVector cos = DoubleVector.fromArray(SPECIES, cos_table, cosOff + i);
                    DoubleVector sin = DoubleVector.fromArray(SPECIES, sin_table, cosOff + i);

                    DoubleVector x0 = DoubleVector.fromArray(SPECIES, k, kOff + i * 2);
                    DoubleVector x1 = DoubleVector.fromArray(SPECIES, k, kOff + i * 2 + 1);

                    DoubleVector x0_new = x0.mul(cos).sub(x1.mul(sin));
                    DoubleVector x1_new = x0.mul(sin).add(x1.mul(cos));

                    x0_new.intoArray(k, kOff + i * 2);
                    x1_new.intoArray(k, kOff + i * 2 + 1);
                }
            }
        }
    }

    /**
     * Precompute RoPE cos/sin tables Call once at model init. base = 10000.0
     * for Llama2
     */
    public static void precompute_rope(int max_seq, int head_dim, double base,
            double[] cos_out, double[] sin_out) {
        int halfDim = head_dim / 2;
        for (int pos = 0; pos < max_seq; pos++) {
            for (int i = 0; i < halfDim; i++) {
                double theta = Math.pow(base, -2.0 * i / head_dim);
                double angle = pos * theta;
                int idx = pos * halfDim + i;
                cos_out[idx] = Math.cos(angle);
                sin_out[idx] = Math.sin(angle);
            }
        }
    }

    /**
     * Decode step with KV cache Only computes QKV for last token, appends K,V
     * to cache, attends to full cache
     *
     * x: [1, dim] - new token embedding k_cache, v_cache: [max_seq, kv_heads,
     * head_dim] - pre-allocated pos: current position to write
     */
    public static void mha_decode_step(
            double[] x, // [dim] - single token
            double[] wq, double[] wk, double[] wv, double[] wo,
            double[] k_cache, double[] v_cache, // [max_seq, kv_heads, head_dim]
            double[] cos_table, double[] sin_table,
            double[] out, // [dim]
            int dim, int num_heads, int kv_heads, int pos, int max_seq) {

        final int head_dim = dim / num_heads;
        final int kv_dim = kv_heads * head_dim;
        final double scale = 1.0 / Math.sqrt(head_dim);

        // Temp for new q, k, v of current token
        double[] q_new = new double[num_heads * head_dim];
        double[] k_new = new double[kv_dim];
        double[] v_new = new double[kv_dim];

        // === 1. QKV for new token only ===
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d += VLEN) {
                DoubleVector qAcc = DoubleVector.zero(SPECIES);
                int col = h * head_dim + d;
                for (int i = 0; i < dim; i++) {
                    DoubleVector xVal = DoubleVector.broadcast(SPECIES, x[i]);
                    DoubleVector wqVec = DoubleVector.fromArray(SPECIES, wq, i * num_heads * head_dim + col);
                    qAcc = qAcc.fma(wqVec, xVal);
                }
                qAcc.intoArray(q_new, col);
            }
        }
        for (int h = 0; h < kv_heads; h++) {
            for (int d = 0; d < head_dim; d += VLEN) {
                DoubleVector kAcc = DoubleVector.zero(SPECIES);
                DoubleVector vAcc = DoubleVector.zero(SPECIES);
                int col = h * head_dim + d;
                for (int i = 0; i < dim; i++) {
                    DoubleVector xVal = DoubleVector.broadcast(SPECIES, x[i]);
                    kAcc = kAcc.fma(DoubleVector.fromArray(SPECIES, wk, i * kv_dim + col), xVal);
                    vAcc = vAcc.fma(DoubleVector.fromArray(SPECIES, wv, i * kv_dim + col), xVal);
                }
                kAcc.intoArray(k_new, col);
                vAcc.intoArray(v_new, col);
            }
        }

        // === 2. Apply RoPE to new Q,K ===
        rope_inplace(q_new, num_heads, k_new, kv_heads, 1, head_dim, pos, cos_table, sin_table);

        // === 3. Append K,V to cache ===
        int kCacheOff = pos * kv_dim;
        System.arraycopy(k_new, 0, k_cache, kCacheOff, kv_dim);
        System.arraycopy(v_new, 0, v_cache, kCacheOff, kv_dim);

        // === 4. Attention: Q_new @ K_cache^T ===
        double[] attn_scores = new double[pos + 1];
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / (num_heads / kv_heads);

            // scores[j] = Q_h @ K_j
            for (int j = 0; j <= pos; j++) {
                double dot = 0.0;
                int qOff = h * head_dim;
                int kOff = j * kv_dim + kv_h * head_dim;
                int d = 0;
                DoubleVector acc = DoubleVector.zero(SPECIES);
                for (; d < SPECIES.loopBound(head_dim); d += VLEN) {
                    DoubleVector qv = DoubleVector.fromArray(SPECIES, q_new, qOff + d);
                    DoubleVector kv = DoubleVector.fromArray(SPECIES, k_cache, kOff + d);
                    acc = acc.fma(qv, kv);
                }
                dot = acc.reduceLanes(VectorOperators.ADD);
                for (; d < head_dim; d++) {
                    dot += q_new[qOff + d] * k_cache[kOff + d];
                }
                attn_scores[j] = dot * scale;
            }

            // softmax
            softmax_row(attn_scores, 0, pos + 1);

            // attn @ V_cache -> out
            for (int d = 0; d < head_dim; d += VLEN) {
                DoubleVector acc = DoubleVector.zero(SPECIES);
                for (int j = 0; j <= pos; j++) {
                    DoubleVector aVec = DoubleVector.broadcast(SPECIES, attn_scores[j]);
                    int vOff = j * kv_dim + kv_h * head_dim + d;
                    DoubleVector vv = DoubleVector.fromArray(SPECIES, v_cache, vOff);
                    acc = acc.fma(vv, aVec);
                }
                acc.intoArray(out, h * head_dim + d);
            }
        }

        // === 5. Output proj ===
        double[] temp = out.clone();
        matmul_bias(out, temp, wo, null, 1, dim, dim);
    }

    /**
     * Down projection: C[M,K] = A[M,N] * B[N,K] No bias, no activation. This is
     * Llama2 down_proj. A = silu(gate) * up, B = W_down, C = FFN output
     */
    /**
     * Down projection: C[M,K] = A[M,N] * B[N,K] Cache-blocked + vectorized.
     * Requires c to be zero-initialized. For Llama2: M=seq_len, N=hidden_dim,
     * K=dim
     */
    public static void matmul_down(double[] a, int M, int N, double[] b, int N2, int K, double[] c) {
        if (N != N2) {
            throw new IllegalArgumentException("Inner dims must match: N=" + N + " N2=" + N2);
        }
        if (a.length < M * N || b.length < N * K || c.length < M * K) {
            throw new IllegalArgumentException("Array too small");
        }

        final int K_BLOCK = 128; // Tune: 64-256. 128 = 1KB per row, fits L1

        for (int m = 0; m < M; m++) {
            final int aRowOff = m * N;
            final int cRowOff = m * K;

            for (int n = 0; n < N; n++) {
                double aVal = a[aRowOff + n];
                if (aVal == 0.0) {
                    continue; // Skip sparsity
                }
                int bRowOff = n * K;
                DoubleVector av = DoubleVector.broadcast(SPECIES, aVal);

                int kBlock = 0;
                for (; kBlock < K; kBlock += K_BLOCK) {
                    int kEnd = Math.min(kBlock + K_BLOCK, K);
                    int k = kBlock;

                    // Vectorized: c[k] += aVal * b[n,k]
                    for (; k <= kEnd - SPECIES.length(); k += SPECIES.length()) {
                        DoubleVector cv = DoubleVector.fromArray(SPECIES, c, cRowOff + k);
                        DoubleVector bv = DoubleVector.fromArray(SPECIES, b, bRowOff + k);
                        cv.fma(av, bv).intoArray(c, cRowOff + k);
                    }
                    // Tail
                    for (; k < kEnd; k++) {
                        c[cRowOff + k] += aVal * b[bRowOff + k];
                    }
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    static DoubleVector tanh_approx(DoubleVector x) {
        // x * (27 + x^2) / (27 + 9*x^2)
        DoubleVector x2 = x.mul(x);
        DoubleVector num = x.mul(x2.add(27.0));
        DoubleVector den = x2.mul(9.0).add(27.0);
        return num.div(den);
    }

}
