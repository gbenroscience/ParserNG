package com.github.gbenroscience.simd.turbo.tools.matrix;

import java.util.Arrays;

/**
 * Clean mechanical sympathetic cross-platform mathematical compute kernels.
 * Highly optimized for Android ART and desktop/server JVMs.
 * * @author GBEMIRO
 */
public final class Kernels {

    // Constants
    private static final double SQRT_2_OVER_PI = 0.7978845608028654; // sqrt(2/pi)
    private static final double GELU_COEFF = 0.044715;

    private Kernels() {
    }

    /**
     * Highly optimized, mechanically sympathetic Multi-Head Attention Decode Step.
     * Eliminates Vector API dependencies and arranges loops to guarantee C2 auto-vectorization.
     * * @param x            [dim] - Single input token vector
     * @param wq           [dim * dim] - Query weight matrix (row-major)
     * @param wk           [dim * kv_dim] - Key weight matrix (row-major)
     * @param wv           [dim * kv_dim] - Value weight matrix (row-major)
     * @param wo           [dim * dim] - Output projection weight matrix (row-major)
     * @param k_cache      [max_seq * kv_dim] - Preallocated key cache matrix
     * @param v_cache      [max_seq * kv_dim] - Preallocated value cache matrix
     * @param cos_table    [max_seq * (head_dim / 2)] - RoPE cos lookup table
     * @param sin_table    [max_seq * (head_dim / 2)] - RoPE sin lookup table
     * @param out          [dim] - Output destination vector
     * @param dim          Hidden layer dimensionality
     * @param num_heads    Number of Query heads
     * @param kv_heads     Number of Key/Value heads (supports MHA and GQA)
     * @param pos          Current token position index in the sequence
     * @param max_seq      Maximum permitted sequence length context
     * @param q_new        Scratch buffer of size [dim] to eliminate allocations
     * @param k_new        Scratch buffer of size [kv_heads * head_dim] to eliminate allocations
     * @param v_new        Scratch buffer of size [kv_heads * head_dim] to eliminate allocations
     * @param attn_scores  Scratch buffer of size [max_seq] to eliminate allocations
     * @param proj_temp    Scratch buffer of size [dim] to eliminate allocations
     */
    public static void mha_decode_step(
            double[] x, 
            double[] wq, double[] wk, double[] wv, double[] wo,
            double[] k_cache, double[] v_cache, 
            double[] cos_table, double[] sin_table,
            double[] out, 
            int dim, int num_heads, int kv_heads, int pos, int max_seq,
            double[] q_new, double[] k_new, double[] v_new, 
            double[] attn_scores, double[] proj_temp) {

        final int head_dim = dim / num_heads;
        final int kv_dim = kv_heads * head_dim;
        final double scale = 1.0 / Math.sqrt(head_dim);

        // === 1. QKV Matrix-Vector Projections (Outer-Product Loop Transformation) ===
        // By zeroing out buffers and transforming to an outer-product style traversal,
        // we sweep through the weight matrices sequentially, triggering blazing-fast SIMD streaming.
        Arrays.fill(q_new, 0, dim, 0.0);
        for (int i = 0; i < dim; i++) {
            final double xVal = x[i];
            final int wqOff = i * dim;
            int c = 0;
            for (; c <= dim - 8; c += 8) {
                q_new[c]     += xVal * wq[wqOff + c];
                q_new[c + 1] += xVal * wq[wqOff + c + 1];
                q_new[c + 2] += xVal * wq[wqOff + c + 2];
                q_new[c + 3] += xVal * wq[wqOff + c + 3];
                q_new[c + 4] += xVal * wq[wqOff + c + 4];
                q_new[c + 5] += xVal * wq[wqOff + c + 5];
                q_new[c + 6] += xVal * wq[wqOff + c + 6];
                q_new[c + 7] += xVal * wq[wqOff + c + 7];
            }
            for (; c < dim; c++) {
                q_new[c] += xVal * wq[wqOff + c];
            }
        }

        Arrays.fill(k_new, 0, kv_dim, 0.0);
        Arrays.fill(v_new, 0, kv_dim, 0.0);
        for (int i = 0; i < dim; i++) {
            final double xVal = x[i];
            final int wkOff = i * kv_dim;
            int c = 0;
            for (; c <= kv_dim - 8; c += 8) {
                k_new[c]     += xVal * wk[wkOff + c];
                k_new[c + 1] += xVal * wk[wkOff + c + 1];
                k_new[c + 2] += xVal * wk[wkOff + c + 2];
                k_new[c + 3] += xVal * wk[wkOff + c + 3];
                k_new[c + 4] += xVal * wk[wkOff + c + 4];
                k_new[c + 5] += xVal * wk[wkOff + c + 5];
                k_new[c + 6] += xVal * wk[wkOff + c + 6];
                k_new[c + 7] += xVal * wk[wkOff + c + 7];

                v_new[c]     += xVal * wv[wkOff + c];
                v_new[c + 1] += xVal * wv[wkOff + c + 1];
                v_new[c + 2] += xVal * wv[wkOff + c + 2];
                v_new[c + 3] += xVal * wv[wkOff + c + 3];
                v_new[c + 4] += xVal * wv[wkOff + c + 4];
                v_new[c + 5] += xVal * wv[wkOff + c + 5];
                v_new[c + 6] += xVal * wv[wkOff + c + 6];
                v_new[c + 7] += xVal * wv[wkOff + c + 7];
            }
            for (; c < kv_dim; c++) {
                k_new[c] += xVal * wk[wkOff + c];
                v_new[c] += xVal * wv[wkOff + c];
            }
        }

        // === 2. Apply RoPE to new Q and K vectors ===
        rope_inplace(q_new, num_heads, k_new, kv_heads, head_dim, pos, cos_table, sin_table);

        // === 3. Append Key and Value states to the preallocated rolling Cache ===
        final int cacheOff = pos * kv_dim;
        System.arraycopy(k_new, 0, k_cache, cacheOff, kv_dim);
        System.arraycopy(v_new, 0, v_cache, cacheOff, kv_dim);

        // === 4. Attention Score Calculation & Context Assembly Block ===
        final int kvGroupSize = num_heads / kv_heads;

        for (int h = 0; h < num_heads; h++) {
            final int kv_h = h / kvGroupSize;
            final int qOff = h * head_dim;

            // Dot Product: Q_new @ K_cache^T for positions [0 ... pos]
            // Unrolled loop bounds break dependency chains to enforce maximized ILP.
            for (int j = 0; j <= pos; j++) {
                final int kOff = j * kv_dim + kv_h * head_dim;
                double dot = 0.0;
                int d = 0;
                for (; d <= head_dim - 8; d += 8) {
                    dot += q_new[qOff + d]     * k_cache[kOff + d]
                         + q_new[qOff + d + 1] * k_cache[kOff + d + 1]
                         + q_new[qOff + d + 2] * k_cache[kOff + d + 2]
                         + q_new[qOff + d + 3] * k_cache[kOff + d + 3]
                         + q_new[qOff + d + 4] * k_cache[kOff + d + 4]
                         + q_new[qOff + d + 5] * k_cache[kOff + d + 5]
                         + q_new[qOff + d + 6] * k_cache[kOff + d + 6]
                         + q_new[qOff + d + 7] * k_cache[kOff + d + 7];
                }
                for (; d < head_dim; d++) {
                    dot += q_new[qOff + d] * k_cache[kOff + d];
                }
                attn_scores[j] = dot * scale;
            }

            // High-throughput local Row Softmax execution
            softmax_row(attn_scores, 0, pos + 1);

            // Context Assembly: attn_scores @ V_cache -> out
            // CRITICAL LOOP INVERSION: Swapping j and d loops converts high-stride jumps into 
            // perfect linear hardware prefetches, streaming v_cache smoothly through CPU registers.
            final int outOff = h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out[outOff + d] = 0.0;
            }

            for (int j = 0; j <= pos; j++) {
                final double scoreVal = attn_scores[j];
                final int vOff = j * kv_dim + kv_h * head_dim;
                int d = 0;
                for (; d <= head_dim - 8; d += 8) {
                    out[outOff + d]     += scoreVal * v_cache[vOff + d];
                    out[outOff + d + 1] += scoreVal * v_cache[vOff + d + 1];
                    out[outOff + d + 2] += scoreVal * v_cache[vOff + d + 2];
                    out[outOff + d + 3] += scoreVal * v_cache[vOff + d + 3];
                    out[outOff + d + 4] += scoreVal * v_cache[vOff + d + 4];
                    out[outOff + d + 5] += scoreVal * v_cache[vOff + d + 5];
                    out[outOff + d + 6] += scoreVal * v_cache[vOff + d + 6];
                    out[outOff + d + 7] += scoreVal * v_cache[vOff + d + 7];
                }
                for (; d < head_dim; d++) {
                    out[outOff + d] += scoreVal * v_cache[vOff + d];
                }
            }
        }

        // === 5. Output Projection Matrix Multiplication ===
        System.arraycopy(out, 0, proj_temp, 0, dim);
        matmul_bias(out, proj_temp, wo, null, 1, dim, dim);
    }

    /**
     * Ultra-fast In-place Rotary Position Embeddings (RoPE) abstraction.
     * Supports standard interleaved/split pair systems.
     */
    public static void rope_inplace(
            double[] q, int qHeads, double[] k, int kHeads, 
            int head_dim, int pos, double[] cos_table, double[] sin_table) {
        
        final int halfDim = head_dim / 2;
        final int cosOff = pos * halfDim;

        // Apply RoPE rotation vectors to Query states
        for (int h = 0; h < qHeads; h++) {
            final int qOff = h * head_dim;
            int i = 0;
            for (; i <= halfDim - 4; i += 4) {
                int i0 = i, i1 = i + 1, i2 = i + 2, i3 = i + 3;
                int idx0_0 = qOff + i0 * 2; int idx1_0 = idx0_0 + 1;
                int idx0_1 = qOff + i1 * 2; int idx1_1 = idx0_1 + 1;
                int idx0_2 = qOff + i2 * 2; int idx1_2 = idx0_2 + 1;
                int idx0_3 = qOff + i3 * 2; int idx1_3 = idx0_3 + 1;

                double c0 = cos_table[cosOff + i0], s0 = sin_table[cosOff + i0];
                double c1 = cos_table[cosOff + i1], s1 = sin_table[cosOff + i1];
                double c2 = cos_table[cosOff + i2], s2 = sin_table[cosOff + i2];
                double c3 = cos_table[cosOff + i3], s3 = sin_table[cosOff + i3];

                double q0_0 = q[idx0_0], q1_0 = q[idx1_0];
                double q0_1 = q[idx0_1], q1_1 = q[idx1_1];
                double q0_2 = q[idx0_2], q1_2 = q[idx1_2];
                double q0_3 = q[idx0_3], q1_3 = q[idx1_3];

                q[idx0_0] = q0_0 * c0 - q1_0 * s0; q[idx1_0] = q0_0 * s0 + q1_0 * c0;
                q[idx0_1] = q0_1 * c1 - q1_1 * s1; q[idx1_1] = q0_1 * s1 + q1_1 * c1;
                q[idx0_2] = q0_2 * c2 - q1_2 * s2; q[idx1_2] = q0_2 * s2 + q1_2 * c2;
                q[idx0_3] = q0_3 * c3 - q1_3 * s3; q[idx1_3] = q0_3 * s3 + q1_3 * c3;
            }
            for (; i < halfDim; i++) {
                int idx0 = qOff + i * 2;
                int idx1 = idx0 + 1;
                double c = cos_table[cosOff + i];
                double s = sin_table[cosOff + i];
                double q0 = q[idx0];
                double q1 = q[idx1];
                q[idx0] = q0 * c - q1 * s;
                q[idx1] = q0 * s + q1 * c;
            }
        }

        // Apply RoPE rotation vectors to Key states
        for (int h = 0; h < kHeads; h++) {
            final int kOff = h * head_dim;
            int i = 0;
            for (; i <= halfDim - 4; i += 4) {
                int i0 = i, i1 = i + 1, i2 = i + 2, i3 = i + 3;
                int idx0_0 = kOff + i0 * 2; int idx1_0 = idx0_0 + 1;
                int idx0_1 = kOff + i1 * 2; int idx1_1 = idx0_1 + 1;
                int idx0_2 = kOff + i2 * 2; int idx1_2 = idx0_2 + 1;
                int idx0_3 = kOff + i3 * 2; int idx1_3 = idx0_3 + 1;

                double c0 = cos_table[cosOff + i0], s0 = sin_table[cosOff + i0];
                double c1 = cos_table[cosOff + i1], s1 = sin_table[cosOff + i1];
                double c2 = cos_table[cosOff + i2], s2 = sin_table[cosOff + i2];
                double c3 = cos_table[cosOff + i3], s3 = sin_table[cosOff + i3];

                double k0_0 = k[idx0_0], k1_0 = k[idx1_0];
                double k0_1 = k[idx0_1], k1_1 = k[idx1_1];
                double k0_2 = k[idx0_2], k1_2 = k[idx1_2];
                double k0_3 = k[idx0_3], k1_3 = k[idx1_3];

                k[idx0_0] = k0_0 * c0 - k1_0 * s0; k[idx1_0] = k0_0 * s0 + k1_0 * c0;
                k[idx0_1] = k0_1 * c1 - k1_1 * s1; k[idx1_1] = k0_1 * s1 + k1_1 * c1;
                k[idx0_2] = k0_2 * c2 - k1_2 * s2; k[idx1_2] = k0_2 * s2 + k1_2 * c2;
                k[idx0_3] = k0_3 * c3 - k1_3 * s3; k[idx1_3] = k0_3 * s3 + k1_3 * c3;
            }
            for (; i < halfDim; i++) {
                int idx0 = kOff + i * 2;
                int idx1 = idx0 + 1;
                double c = cos_table[cosOff + i];
                double s = sin_table[cosOff + i];
                double k0 = k[idx0];
                double k1 = k[idx1];
                k[idx0] = k0 * c - k1 * s;
                k[idx1] = k0 * s + k1 * c;
            }
        }
    }

    /**
     * Low-level numerical Stable Softmax across an isolated array row segment.
     */
    public static void softmax_row(double[] arr, int offset, int len) {
        double max = arr[offset];
        for (int i = 1; i < len; i++) {
            if (arr[offset + i] > max) {
                max = arr[offset + i];
            }
        }
        double sum = 0.0;
        for (int i = 0; i < len; i++) {
            double ex = Math.exp(arr[offset + i] - max);
            arr[offset + i] = ex;
            sum += ex;
        }
        final double invSum = 1.0 / (sum + 1e-15);
        int i = 0;
        for (; i <= len - 4; i += 4) {
            arr[offset + i]     *= invSum;
            arr[offset + i + 1] *= invSum;
            arr[offset + i + 2] *= invSum;
            arr[offset + i + 3] *= invSum;
        }
        for (; i < len; i++) {
            arr[offset + i] *= invSum;
        }
    }

    /**
     * Standard backward-compatible signature. Allocates small arrays internally 
     * if passing scratchpads down isn't viable.
     */
    public static void mha_decode_step(
            double[] x, 
            double[] wq, double[] wk, double[] wv, double[] wo,
            double[] k_cache, double[] v_cache, 
            double[] cos_table, double[] sin_table,
            double[] out, 
            int dim, int num_heads, int kv_heads, int pos, int max_seq) {
        
        double[] q_new = new double[dim];
        double[] k_new = new double[kv_heads * (dim / num_heads)];
        double[] v_new = new double[k_new.length];
        double[] attn_scores = new double[pos + 1];
        double[] proj_temp = new double[dim];

        mha_decode_step(x, wq, wk, wv, wo, k_cache, v_cache, cos_table, sin_table, 
                        out, dim, num_heads, kv_heads, pos, max_seq, 
                        q_new, k_new, v_new, attn_scores, proj_temp);
    }

   /**
    * Fused: C[M,N] = relu(A[M,K] * B[K,N] + bias[N])
    *  Row-major: A[M,K], B[K,N], C[M,N], bias[N]
    * @param a
    * @param aRows
    * @param aCols
    * @param b
    * @param bRows
    * @param bCols
    * @param bias
    * @param c
    * @param cRows
    * @param cCols 
    */
    public static void matmul_bias_relu(
            double[] a, int aRows, int aCols, // A[M,K]
            double[] b, int bRows, int bCols, // B[K,N]
            double[] bias,                    // bias[N]
            double[] c, int cRows, int cCols) { // C[M,N]

        final int M = aRows, N = bCols, K = aCols;
        if (bRows != K || cRows != M || cCols != N || bias.length != N) {
            throw new IllegalArgumentException("Shape mismatch");
        }

        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            int n = 0;
            // 4x Loop Unrolling to maximize pipeline throughput and trigger SIMD hardware lanes
            for (; n <= N - 4; n += 4) {
                double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
                for (int k = 0; k < K; k++) {
                    double aVal = a[aRowOffset + k];
                    int bIdx = k * N + n;
                    acc0 += aVal * b[bIdx];
                    acc1 += aVal * b[bIdx + 1];
                    acc2 += aVal * b[bIdx + 2];
                    acc3 += aVal * b[bIdx + 3];
                }

                double res0 = acc0 + bias[n];
                double res1 = acc1 + bias[n + 1];
                double res2 = acc2 + bias[n + 2];
                double res3 = acc3 + bias[n + 3];

                c[cRowOffset + n]     = res0 > 0 ? res0 : 0;
                c[cRowOffset + n + 1] = res1 > 0 ? res1 : 0;
                c[cRowOffset + n + 2] = res2 > 0 ? res2 : 0;
                c[cRowOffset + n + 3] = res3 > 0 ? res3 : 0;
            }

            // Scalar tail
            for (; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOffset + k] * b[k * N + n];
                }
                acc += bias[n];
                c[cRowOffset + n] = acc > 0 ? acc : 0;
            }
        }
    }

    /**
     * Optimized variant: assumes N % 4 == 0 and K >= 4
     * Unrolls both column and dot-product depth loops to maximize out-of-order execution pipelines.
     */
    public static void matmul_bias_relu_fast(
            double[] a, double[] b, double[] bias, double[] c,
            int M, int N, int K) {

        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            int n = 0;
            for (; n <= N - 4; n += 4) {
                double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
                int k = 0;
                for (; k <= K - 4; k += 4) {
                    double a0 = a[aRowOffset + k];
                    double a1 = a[aRowOffset + k + 1];
                    double a2 = a[aRowOffset + k + 2];
                    double a3 = a[aRowOffset + k + 3];

                    int bOff0 = k * N + n;
                    int bOff1 = (k + 1) * N + n;
                    int bOff2 = (k + 2) * N + n;
                    int bOff3 = (k + 3) * N + n;
                    acc0 += a0 * b[bOff0]     + a1 * b[bOff1]     + a2 * b[bOff2]     + a3 * b[bOff3];
                    acc1 += a0 * b[bOff0 + 1] + a1 * b[bOff1 + 1] + a2 * b[bOff2 + 1] + a3 * b[bOff3 + 1];
                    acc2 += a0 * b[bOff0 + 2] + a1 * b[bOff1 + 2] + a2 * b[bOff2 + 2] + a3 * b[bOff3 + 2];
                    acc3 += a0 * b[bOff0 + 3] + a1 * b[bOff1 + 3] + a2 * b[bOff2 + 3] + a3 * b[bOff3 + 3];
                }

                for (; k < K; k++) {
                    double aVal = a[aRowOffset + k];
                    int bOff = k * N + n;
                    acc0 += aVal * b[bOff];
                    acc1 += aVal * b[bOff + 1];
                    acc2 += aVal * b[bOff + 2];
                    acc3 += aVal * b[bOff + 3];
                }

                double r0 = acc0 + bias[n];
                double r1 = acc1 + bias[n + 1];
                double r2 = acc2 + bias[n + 2];
                double r3 = acc3 + bias[n + 3];

                c[cRowOffset + n]     = r0 > 0 ? r0 : 0;
                c[cRowOffset + n + 1] = r1 > 0 ? r1 : 0;
                c[cRowOffset + n + 2] = r2 > 0 ? r2 : 0;
                c[cRowOffset + n + 3] = r3 > 0 ? r3 : 0;
            }

            for (; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOffset + k] * b[k * N + n];
                }
                double r = acc + bias[n];
                c[cRowOffset + n] = r > 0 ? r : 0;
            }
        }
    }

    /**
     * Layer Normalization: clean, sequential multi-pass execution loop structure
     * which allows efficient vectorization over hidden dimensions.
     */
    public static void layer_norm(
            double[] x, double[] gamma, double[] beta, double[] out,
            int batch, int features, double eps) {

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;
            double sum = 0.0;
            int i = 0;
            for (; i <= features - 4; i += 4) {
                sum += x[offset + i] + x[offset + i + 1] + x[offset + i + 2] + x[offset + i + 3];
            }
            for (; i < features; i++) {
                sum += x[offset + i];
            }
            double mean = sum / features;
            double varSum = 0.0;
            i = 0;
            for (; i <= features - 4; i += 4) {
                double d0 = x[offset + i] - mean;
                double d1 = x[offset + i + 1] - mean;
                double d2 = x[offset + i + 2] - mean;
                double d3 = x[offset + i + 3] - mean;
                varSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            }
            for (; i < features; i++) {
                double d = x[offset + i] - mean;
                varSum += d * d;
            }
            
            double invStd = 1.0 / Math.sqrt(varSum / features + eps);
            int j = 0;
            for (; j <= features - 4; j += 4) {
                int idx0 = offset + j;
                out[idx0]     = (x[idx0] - mean) * invStd * gamma[j] + beta[j];
                out[idx0 + 1] = (x[idx0 + 1] - mean) * invStd * gamma[j + 1] + beta[j + 1];
                out[idx0 + 2] = (x[idx0 + 2] - mean) * invStd * gamma[j + 2] + beta[j + 2];
                out[idx0 + 3] = (x[idx0 + 3] - mean) * invStd * gamma[j + 3] + beta[j + 3];
            }
            for (; j < features; j++) {
                out[offset + j] = (x[offset + j] - mean) * invStd * gamma[j] + beta[j];
            }
        }
    }

    public static void layer_norm_fast(
            double[] x, double[] gamma, double[] beta, double[] out,
            int batch, int features, double eps) {
        for (int b = 0; b < batch; b++) {
            final int offset = b * features;
            double sum = 0.0;
            for (int i = 0; i < features; i += 4) {
                sum += x[offset + i] + x[offset + i + 1] + x[offset + i + 2] + x[offset + i + 3];
            }
            double mean = sum / features;
            double varSum = 0.0;
            for (int i = 0; i < features; i += 4) {
                double d0 = x[offset + i] - mean;
                double d1 = x[offset + i + 1] - mean;
                double d2 = x[offset + i + 2] - mean;
                double d3 = x[offset + i + 3] - mean;
                varSum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            }
            double invStd = 1.0 / Math.sqrt(varSum / features + eps);
            for (int i = 0; i < features; i += 4) {
                int idx = offset + i;
                out[idx]     = (x[idx] - mean) * invStd * gamma[i] + beta[i];
                out[idx + 1] = (x[idx + 1] - mean) * invStd * gamma[i + 1] + beta[i + 1];
                out[idx + 2] = (x[idx + 2] - mean) * invStd * gamma[i + 2] + beta[i + 2];
                out[idx + 3] = (x[idx + 3] - mean) * invStd * gamma[i + 3] + beta[i + 3];
            }
        }
    }

    /**
     * Fused Matrix Multiplication + Bias + Approximate GELU non-linear formulation
     */
    public static void matmul_bias_gelu(
            double[] a, int aRows, int aCols,
            double[] b, int bRows, int bCols,
            double[] bias,
            double[] c, int cRows, int cCols) {
        final int M = aRows, N = bCols, K = aCols;
        if (bRows != K || cRows != M || cCols != N || bias.length != N) {
            throw new IllegalArgumentException("Shape mismatch");
        }

        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            int n = 0;
            for (; n <= N - 4; n += 4) {
                double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
                for (int k = 0; k < K; k++) {
                    double aVal = a[aRowOffset + k];
                    int bIdx = k * N + n;
                    acc0 += aVal * b[bIdx];
                    acc1 += aVal * b[bIdx + 1];
                    acc2 += aVal * b[bIdx + 2];
                    acc3 += aVal * b[bIdx + 3];
                }

                double x0 = acc0 + bias[n];
                double x1 = acc1 + bias[n + 1];
                double x2 = acc2 + bias[n + 2];
                double x3 = acc3 + bias[n + 3];

                c[cRowOffset + n]     = 0.5 * x0 * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x0 + GELU_COEFF * x0 * x0 * x0)));
                c[cRowOffset + n + 1] = 0.5 * x1 * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x1 + GELU_COEFF * x1 * x1 * x1)));
                c[cRowOffset + n + 2] = 0.5 * x2 * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x2 + GELU_COEFF * x2 * x2 * x2)));
                c[cRowOffset + n + 3] = 0.5 * x3 * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x3 + GELU_COEFF * x3 * x3 * x3)));
            }

            for (; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOffset + k] * b[k * N + n];
                }
                double x = acc + bias[n];
                c[cRowOffset + n] = 0.5 * x * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)));
            }
        }
    }

    public static void matmul_bias_gelu_fast(
            double[] a, double[] b, double[] bias, double[] c,
            int M, int N, int K) {
        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            int n = 0;
            for (; n <= N - 4; n += 4) {
                double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
                int k = 0;
                for (; k <= K - 4; k += 4) {
                    double a0 = a[aRowOffset + k];
                    double a1 = a[aRowOffset + k + 1];
                    double a2 = a[aRowOffset + k + 2];
                    double a3 = a[aRowOffset + k + 3];

                    int bOff0 = k * N + n;
                    int bOff1 = (k + 1) * N + n;
                    int bOff2 = (k + 2) * N + n;
                    int bOff3 = (k + 3) * N + n;
                    acc0 += a0 * b[bOff0]     + a1 * b[bOff1]     + a2 * b[bOff2]     + a3 * b[bOff3];
                    acc1 += a0 * b[bOff0 + 1] + a1 * b[bOff1 + 1] + a2 * b[bOff2 + 1] + a3 * b[bOff3 + 1];
                    acc2 += a0 * b[bOff0 + 2] + a1 * b[bOff1 + 2] + a2 * b[bOff2 + 2] + a3 * b[bOff3 + 2];
                    acc3 += a0 * b[bOff0 + 3] + a1 * b[bOff1 + 3] + a2 * b[bOff2 + 3] + a3 * b[bOff3 + 3];
                }
                for (; k < K; k++) {
                    double aVal = a[aRowOffset + k];
                    int bOff = k * N + n;
                    acc0 += aVal * b[bOff];
                    acc1 += aVal * b[bOff + 1];
                    acc2 += aVal * b[bOff + 2];
                    acc3 += aVal * b[bOff + 3];
                }

                double x0 = acc0 + bias[n];
                double x1 = acc1 + bias[n + 1];
                double x2 = acc2 + bias[n + 2];
                double x3 = acc3 + bias[n + 3];

                c[cRowOffset + n]     = 0.5 * x0 * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x0 + GELU_COEFF * x0 * x0 * x0)));
                c[cRowOffset + n + 1] = 0.5 * x1 * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x1 + GELU_COEFF * x1 * x1 * x1)));
                c[cRowOffset + n + 2] = 0.5 * x2 * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x2 + GELU_COEFF * x2 * x2 * x2)));
                c[cRowOffset + n + 3] = 0.5 * x3 * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x3 + GELU_COEFF * x3 * x3 * x3)));
            }
            for (; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOffset + k] * b[k * N + n];
                }
                double x = acc + bias[n];
                c[cRowOffset + n] = 0.5 * x * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)));
            }
        }
    }

    /**
     * Root Mean Square Normalization (Llama/Mistral/Qwen modern style)
     */
    public static void rms_norm(
            double[] x, double[] gamma, double[] out,
            int batch, int features, double eps) {

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;

            double sumSq = 0.0;
            int i = 0;
            for (; i <= features - 4; i += 4) {
                double v0 = x[offset + i];
                double v1 = x[offset + i + 1];
                double v2 = x[offset + i + 2];
                double v3 = x[offset + i + 3];
                sumSq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }
            for (; i < features; i++) {
                double val = x[offset + i];
                sumSq += val * val;
            }

            double rms = 1.0 / Math.sqrt(sumSq / features + eps);
            int j = 0;
            for (; j <= features - 4; j += 4) {
                int idx = offset + j;
                out[idx]     = x[idx]     * rms * gamma[j];
                out[idx + 1] = x[idx + 1] * rms * gamma[j + 1];
                out[idx + 2] = x[idx + 2] * rms * gamma[j + 2];
                out[idx + 3] = x[idx + 3] * rms * gamma[j + 3];
            }
            for (; j < features; j++) {
                out[offset + j] = x[offset + j] * rms * gamma[j];
            }
        }
    }

    public static void rms_norm_fast(
            double[] x, double[] gamma, double[] out,
            int batch, int features, double eps) {
        for (int b = 0; b < batch; b++) {
            final int offset = b * features;
            double sumSq = 0.0;
            for (int i = 0; i < features; i += 4) {
                double v0 = x[offset + i];
                double v1 = x[offset + i + 1];
                double v2 = x[offset + i + 2];
                double v3 = x[offset + i + 3];
                sumSq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }
            double rms = 1.0 / Math.sqrt(sumSq / features + eps);
            for (int i = 0; i < features; i += 4) {
                int idx = offset + i;
                out[idx]     = x[idx]     * rms * gamma[i];
                out[idx + 1] = x[idx + 1] * rms * gamma[i + 1];
                out[idx + 2] = x[idx + 2] * rms * gamma[i + 2];
                out[idx + 3] = x[idx + 3] * rms * gamma[i + 3];
            }
        }
    }

    /**
     * Stable Online Softmax (Per-Row Reduction Platform Vectorized)
     */
    public static void softmax(double[] x, double[] out, int batch, int features) {
        for (int b = 0; b < batch; b++) {
            final int offset = b * features;
            // Pass 1: find max
            double max = -Double.MAX_VALUE;
            int i = 0;
            for (; i <= features - 4; i += 4) {
                double m01 = Math.max(x[offset + i], x[offset + i + 1]);
                double m23 = Math.max(x[offset + i + 2], x[offset + i + 3]);
                max = Math.max(max, Math.max(m01, m23));
            }
            for (; i < features; i++) {
                max = Math.max(max, x[offset + i]);
            }

            // Pass 2: exp(x - max) and accumulation
            double sum = 0.0;
            int j = 0;
            for (; j <= features - 4; j += 4) {
                double e0 = Math.exp(x[offset + j] - max);
                double e1 = Math.exp(x[offset + j + 1] - max);
                double e2 = Math.exp(x[offset + j + 2] - max);
                double e3 = Math.exp(x[offset + j + 3] - max);
                out[offset + j]     = e0;
                out[offset + j + 1] = e1;
                out[offset + j + 2] = e2;
                out[offset + j + 3] = e3;
                sum += e0 + e1 + e2 + e3;
            }
            for (; j < features; j++) {
                double ex = Math.exp(x[offset + j] - max);
                out[offset + j] = ex;
                sum += ex;
            }

            // Pass 3: write normalized outputs
            double invSum = 1.0 / sum;
            int k = 0;
            for (; k <= features - 4; k += 4) {
                out[offset + k]     *= invSum;
                out[offset + k + 1] *= invSum;
                out[offset + k + 2] *= invSum;
                out[offset + k + 3] *= invSum;
            }
            for (; k < features; k++) {
                out[offset + k] *= invSum;
            }
        }
    }

    public static void softmax_fast(double[] x, double[] out, int batch, int features) {
        for (int b = 0; b < batch; b++) {
            final int offset = b * features;
            double max = -Double.MAX_VALUE;
            for (int i = 0; i < features; i += 4) {
                double m01 = Math.max(x[offset + i], x[offset + i + 1]);
                double m23 = Math.max(x[offset + i + 2], x[offset + i + 3]);
                max = Math.max(max, Math.max(m01, m23));
            }

            double sum = 0.0;
            for (int i = 0; i < features; i += 4) {
                sum += Math.exp(x[offset + i] - max)
                     + Math.exp(x[offset + i + 1] - max)
                     + Math.exp(x[offset + i + 2] - max)
                     + Math.exp(x[offset + i + 3] - max);
            }
            double invSum = 1.0 / sum;
            for (int i = 0; i < features; i += 4) {
                int idx = offset + i;
                out[idx]     = Math.exp(x[idx] - max) * invSum;
                out[idx + 1] = Math.exp(x[idx + 1] - max) * invSum;
                out[idx + 2] = Math.exp(x[idx + 2] - max) * invSum;
                out[idx + 3] = Math.exp(x[idx + 3] - max) * invSum;
            }
        }
    }

    /**
     * Stable Vectorized Log-Softmax implementation
     */
    public static void log_softmax(double[] x, double[] out, int batch, int features) {
        for (int b = 0; b < batch; b++) {
            final int offset = b * features;
            double max = -Double.MAX_VALUE;
            int i = 0;
            for (; i <= features - 4; i += 4) {
                double m01 = Math.max(x[offset + i], x[offset + i + 1]);
                double m23 = Math.max(x[offset + i + 2], x[offset + i + 3]);
                max = Math.max(max, Math.max(m01, m23));
            }
            for (; i < features; i++) {
                max = Math.max(max, x[offset + i]);
            }

            double sum = 0.0;
            int j = 0;
            for (; j <= features - 4; j += 4) {
                sum += Math.exp(x[offset + j] - max) 
                     + Math.exp(x[offset + j + 1] - max) 
                     + Math.exp(x[offset + j + 2] - max) 
                     + Math.exp(x[offset + j + 3] - max);
            }
            for (; j < features; j++) {
                sum += Math.exp(x[offset + j] - max);
            }
            double logSum = Math.log(sum);

            int k = 0;
            for (; k <= features - 4; k += 4) {
                out[offset + k]     = x[offset + k] - max - logSum;
                out[offset + k + 1] = x[offset + k + 1] - max - logSum;
                out[offset + k + 2] = x[offset + k + 2] - max - logSum;
                out[offset + k + 3] = x[offset + k + 3] - max - logSum;
            }
            for (; k < features; k++) {
                out[offset + k] = x[offset + k] - max - logSum;
            }
        }
    }

    /**
     * Fused: C[M,N] = silu(A[M,K] * B[K,N] + bias[N])
     */
    public static void matmul_bias_silu(
            double[] a, int aRows, int aCols,
            double[] b, int bRows, int bCols,
            double[] bias,
            double[] c, int cRows, int cCols) {
        final int M = aRows, N = bCols, K = aCols;
        if (bRows != K || cRows != M || cCols != N || bias.length != N) {
            throw new IllegalArgumentException("Shape mismatch");
        }

        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            int n = 0;
            for (; n <= N - 4; n += 4) {
                double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
                for (int k = 0; k < K; k++) {
                    double aVal = a[aRowOffset + k];
                    int bIdx = k * N + n;
                    acc0 += aVal * b[bIdx];
                    acc1 += aVal * b[bIdx + 1];
                    acc2 += aVal * b[bIdx + 2];
                    acc3 += aVal * b[bIdx + 3];
                }

                double x0 = acc0 + bias[n];
                double x1 = acc1 + bias[n + 1];
                double x2 = acc2 + bias[n + 2];
                double x3 = acc3 + bias[n + 3];

                c[cRowOffset + n]     = x0 / (1.0 + Math.exp(-x0));
                c[cRowOffset + n + 1] = x1 / (1.0 + Math.exp(-x1));
                c[cRowOffset + n + 2] = x2 / (1.0 + Math.exp(-x2));
                c[cRowOffset + n + 3] = x3 / (1.0 + Math.exp(-x3));
            }

            for (; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOffset + k] * b[k * N + n];
                }
                double x = acc + bias[n];
                c[cRowOffset + n] = x / (1.0 + Math.exp(-x));
            }
        }
    }

    public static void matmul_bias_silu_fast(
            double[] a, double[] b, double[] bias, double[] c,
            int M, int N, int K) {
        for (int m = 0; m < M; m++) {
            final int aRowOffset = m * K;
            final int cRowOffset = m * N;

            int n = 0;
            for (; n <= N - 4; n += 4) {
                double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
                int k = 0;
                for (; k <= K - 4; k += 4) {
                    double a0 = a[aRowOffset + k];
                    double a1 = a[aRowOffset + k + 1];
                    double a2 = a[aRowOffset + k + 2];
                    double a3 = a[aRowOffset + k + 3];

                    int bOff0 = k * N + n;
                    int bOff1 = (k + 1) * N + n;
                    int bOff2 = (k + 2) * N + n;
                    int bOff3 = (k + 3) * N + n;
                    acc0 += a0 * b[bOff0]     + a1 * b[bOff1]     + a2 * b[bOff2]     + a3 * b[bOff3];
                    acc1 += a0 * b[bOff0 + 1] + a1 * b[bOff1 + 1] + a2 * b[bOff2 + 1] + a3 * b[bOff3 + 1];
                    acc2 += a0 * b[bOff0 + 2] + a1 * b[bOff1 + 2] + a2 * b[bOff2 + 2] + a3 * b[bOff3 + 2];
                    acc3 += a0 * b[bOff0 + 3] + a1 * b[bOff1 + 3] + a2 * b[bOff2 + 3] + a3 * b[bOff3 + 3];
                }
                for (; k < K; k++) {
                    double aVal = a[aRowOffset + k];
                    int bOff = k * N + n;
                    acc0 += aVal * b[bOff];
                    acc1 += aVal * b[bOff + 1];
                    acc2 += aVal * b[bOff + 2];
                    acc3 += aVal * b[bOff + 3];
                }

                double x0 = acc0 + bias[n];
                double x1 = acc1 + bias[n + 1];
                double x2 = acc2 + bias[n + 2];
                double x3 = acc3 + bias[n + 3];

                c[cRowOffset + n]     = x0 / (1.0 + Math.exp(-x0));
                c[cRowOffset + n + 1] = x1 / (1.0 + Math.exp(-x1));
                c[cRowOffset + n + 2] = x2 / (1.0 + Math.exp(-x2));
                c[cRowOffset + n + 3] = x3 / (1.0 + Math.exp(-x3));
            }
            for (; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOffset + k] * b[k * N + n];
                }
                double x = acc + bias[n];
                c[cRowOffset + n] = x / (1.0 + Math.exp(-x));
            }
        }
    }

    /**
     * SwiGLU non-linear activation matrix compute layer
     */
    public static void matmul_swiglu(
            double[] x, double[] wGate, double[] wUp, double[] out,
            int rows, int K, int H) {
        for (int r = 0; r < rows; r++) {
            final int xOff = r * K;
            final int outOff = r * H;

            int h = 0;
            for (; h <= H - 4; h += 4) {
                double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;
                double u0 = 0.0, u1 = 0.0, u2 = 0.0, u3 = 0.0;
                for (int k = 0; k < K; k++) {
                    double xv = x[xOff + k];
                    int bIdx = k * H + h;
                    g0 += xv * wGate[bIdx];
                    g1 += xv * wGate[bIdx + 1];
                    g2 += xv * wGate[bIdx + 2];
                    g3 += xv * wGate[bIdx + 3];

                    u0 += xv * wUp[bIdx];
                    u1 += xv * wUp[bIdx + 1];
                    u2 += xv * wUp[bIdx + 2];
                    u3 += xv * wUp[bIdx + 3];
                }

                double silu0 = g0 / (1.0 + Math.exp(-g0));
                double silu1 = g1 / (1.0 + Math.exp(-g1));
                double silu2 = g2 / (1.0 + Math.exp(-g2));
                double silu3 = g3 / (1.0 + Math.exp(-g3));

                out[outOff + h]     = silu0 * u0;
                out[outOff + h + 1] = silu1 * u1;
                out[outOff + h + 2] = silu2 * u2;
                out[outOff + h + 3] = silu3 * u3;
            }

            for (; h < H; h++) {
                double g = 0.0;
                double u = 0.0;
                for (int k = 0; k < K; k++) {
                    double xv = x[xOff + k];
                    g += xv * wGate[k * H + h];
                    u += xv * wUp[k * H + h];
                }
                double silu = g / (1.0 + Math.exp(-g));
                out[outOff + h] = silu * u;
            }
        }
    }

    /**
     * Rotary Position Embedding (RoPE) In-place transformation pipeline
     */
    public static void rope_inplace(
            double[] q, int qHeads, double[] k, int kHeads, int seq_len, int head_dim,
            int pos_start, double[] cos_table, double[] sin_table) {
        final int halfDim = head_dim / 2;
        for (int s = 0; s < seq_len; s++) {
            int pos = pos_start + s;
            int cosOff = pos * halfDim;

            // Apply RoPE on Query states
            for (int h = 0; h < qHeads; h++) {
                int qOff = s * qHeads * head_dim + h * head_dim;
                int i = 0;
                for (; i <= halfDim - 2; i += 2) {
                    int idx0_0 = qOff + i * 2;
                    int idx0_1 = idx0_0 + 1;
                    int idx1_0 = qOff + (i + 1) * 2;
                    int idx1_1 = idx1_0 + 1;

                    double c0 = cos_table[cosOff + i];
                    double s0 = sin_table[cosOff + i];
                    double c1 = cos_table[cosOff + i + 1];
                    double s1 = sin_table[cosOff + i + 1];

                    double q_x0_0 = q[idx0_0];
                    double q_x0_1 = q[idx0_1];
                    q[idx0_0] = q_x0_0 * c0 - q_x0_1 * s0;
                    q[idx0_1] = q_x0_0 * s0 + q_x0_1 * c0;

                    double q_x1_0 = q[idx1_0];
                    double q_x1_1 = q[idx1_1];
                    q[idx1_0] = q_x1_0 * c1 - q_x1_1 * s1;
                    q[idx1_1] = q_x1_0 * s1 + q_x1_1 * c1;
                }
                for (; i < halfDim; i++) {
                    int idx0 = qOff + i * 2;
                    int idx1 = idx0 + 1;
                    double c = cos_table[cosOff + i];
                    double s_val = sin_table[cosOff + i];
                    double q_x0 = q[idx0];
                    double q_x1 = q[idx1];
                    q[idx0] = q_x0 * c - q_x1 * s_val;
                    q[idx1] = q_x0 * s_val + q_x1 * c;
                }
            }

            // Apply RoPE on Key states
            for (int h = 0; h < kHeads; h++) {
                int kOff = s * kHeads * head_dim + h * head_dim;
                int i = 0;
                for (; i <= halfDim - 2; i += 2) {
                    int idx0_0 = kOff + i * 2;
                    int idx0_1 = idx0_0 + 1;
                    int idx1_0 = kOff + (i + 1) * 2;
                    int idx1_1 = idx1_0 + 1;

                    double c0 = cos_table[cosOff + i];
                    double s0 = sin_table[cosOff + i];
                    double c1 = cos_table[cosOff + i + 1];
                    double s1 = sin_table[cosOff + i + 1];

                    double k_x0_0 = k[idx0_0];
                    double k_x0_1 = k[idx0_1];
                    k[idx0_0] = k_x0_0 * c0 - k_x0_1 * s0;
                    k[idx0_1] = k_x0_0 * s0 + k_x0_1 * c0;

                    double k_x1_0 = k[idx1_0];
                    double k_x1_1 = k[idx1_1];
                    k[idx1_0] = k_x1_0 * c1 - k_x1_1 * s1;
                    k[idx1_1] = k_x1_0 * s1 + k_x1_1 * c1;
                }
                for (; i < halfDim; i++) {
                    int idx0 = kOff + i * 2;
                    int idx1 = idx0 + 1;
                    double c = cos_table[cosOff + i];
                    double s_val = sin_table[cosOff + i];
                    double k_x0 = k[idx0];
                    double k_x1 = k[idx1];
                    k[idx0] = k_x0 * c - k_x1 * s_val;
                    k[idx1] = k_x0 * s_val + k_x1 * c;
                }
            }
        }
    }

    /**
     * Inference Decode Token Step Kernel using preallocated Key/Value Cache matrices
     */
    public static void decode_step(
            double[] x, double[] wq, double[] wk, double[] wv, double[] wo,
            double[] k_cache, double[] v_cache, int pos, int max_seq, int dim,
            int num_heads, int kv_heads, int head_dim, double[] cos_table, double[] sin_table,
            double[] out) {
        int kv_dim = kv_heads * head_dim;
        double scale = 1.0 / Math.sqrt(head_dim);
        double[] q_new = new double[num_heads * head_dim];
        double[] k_new = new double[kv_dim];
        double[] v_new = new double[kv_dim];

        for (int h = 0; h < num_heads; h++) {
            boolean doKV = h < kv_heads;
            for (int d = 0; d < head_dim; d++) {
                int col = h * head_dim + d;
                double qAcc = 0.0;
                double kAcc = 0.0;
                double vAcc = 0.0;
                for (int i = 0; i < dim; i++) {
                    double xv = x[i];
                    qAcc += xv * wq[i * num_heads * head_dim + col];
                    if (doKV) {
                        kAcc += xv * wk[i * kv_dim + col];
                        vAcc += xv * wv[i * kv_dim + col];
                    }
                }
                q_new[col] = qAcc;
                if (doKV) {
                    k_new[col] = kAcc;
                    v_new[col] = vAcc;
                }
            }
        }

        // Apply RoPE on q_new and k_new for single token step at position 'pos'
        final int halfDim = head_dim / 2;
        int cosOff = pos * halfDim;
        for (int h = 0; h < num_heads; h++) {
            int qOff = h * head_dim;
            for (int i = 0; i < halfDim; i++) {
                int idx0 = qOff + i * 2;
                int idx1 = idx0 + 1;
                double c = cos_table[cosOff + i];
                double s_val = sin_table[cosOff + i];
                double qx0 = q_new[idx0];
                double qx1 = q_new[idx1];
                q_new[idx0] = qx0 * c - qx1 * s_val;
                q_new[idx1] = qx0 * s_val + qx1 * c;
            }
        }
        for (int h = 0; h < kv_heads; h++) {
            int kOff = h * head_dim;
            for (int i = 0; i < halfDim; i++) {
                int idx0 = kOff + i * 2;
                int idx1 = idx0 + 1;
                double c = cos_table[cosOff + i];
                double s_val = sin_table[cosOff + i];
                double kx0 = k_new[idx0];
                double kx1 = k_new[idx1];
                k_new[idx0] = kx0 * c - kx1 * s_val;
                k_new[idx1] = kx0 * s_val + kx1 * c;
            }
        }

        // Save new Key and Value states into Cache boundaries
        int cache_offset = pos * kv_dim;
        System.arraycopy(k_new, 0, k_cache, cache_offset, kv_dim);
        System.arraycopy(v_new, 0, v_cache, cache_offset, kv_dim);

        // Grouped Query Attention (GQA) implementation
        double[] attn_out = new double[num_heads * head_dim];
        double[] score_buf = new double[pos + 1];

        int kv_group_size = num_heads / kv_heads;

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / kv_group_size;
            double max_score = -Double.MAX_VALUE;

            // Step 1: Query x Key table scanning up to current pos
            for (int t = 0; t <= pos; t++) {
                double score = 0.0;
                for (int d = 0; d < head_dim; d++) {
                    score += q_new[h * head_dim + d] * k_cache[t * kv_dim + kv_h * head_dim + d];
                }
                score *= scale;
                score_buf[t] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            // Step 2: Stable single-pass softmax reduction layer
            double sum_exp = 0.0;
            for (int t = 0; t <= pos; t++) {
                double exp_val = Math.exp(score_buf[t] - max_score);
                score_buf[t] = exp_val;
                sum_exp += exp_val;
            }
            double inv_sum = 1.0 / (sum_exp + 1e-15);
            for (int t = 0; t <= pos; t++) {
                score_buf[t] *= inv_sum;
            }

            // Step 3: Accumulate values weighted sum
            for (int d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (int t = 0; t <= pos; t++) {
                    acc += score_buf[t] * v_cache[t * kv_dim + kv_h * head_dim + d];
                }
                attn_out[h * head_dim + d] = acc;
            }
        }

        // Project outputs directly through Wo
        matmul_bias(out, attn_out, wo, null, 1, dim, num_heads * head_dim);
    }

    
 
    /**
     * General Matrix Multiplication with optional bias layout support helper
     */
    public static void matmul_bias(double[] c, double[] a, double[] b, double[] bias, int M, int N, int K) {
        for (int m = 0; m < M; m++) {
            int aRowOff = m * K;
            int cRowOff = m * N;
            int n = 0;
            for (; n <= N - 4; n += 4) {
                double acc0 = (bias == null) ? 0.0 : bias[n];
                double acc1 = (bias == null) ? 0.0 : bias[n + 1];
                double acc2 = (bias == null) ? 0.0 : bias[n + 2];
                double acc3 = (bias == null) ? 0.0 : bias[n + 3];
                for (int k = 0; k < K; k++) {
                    double aVal = a[aRowOff + k];
                    int bIdx = k * N + n;
                    acc0 += aVal * b[bIdx];
                    acc1 += aVal * b[bIdx + 1];
                    acc2 += aVal * b[bIdx + 2];
                    acc3 += aVal * b[bIdx + 3];
                }
                c[cRowOff + n]     = acc0;
                c[cRowOff + n + 1] = acc1;
                c[cRowOff + n + 2] = acc2;
                c[cRowOff + n + 3] = acc3;
            }
            for (; n < N; n++) {
                double acc = (bias == null) ? 0.0 : bias[n];
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOff + k] * b[k * N + n];
                }
                c[cRowOff + n] = acc;
            }
        }
    }
}