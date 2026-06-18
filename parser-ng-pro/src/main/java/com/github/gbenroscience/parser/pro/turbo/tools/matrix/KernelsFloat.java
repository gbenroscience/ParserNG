package com.github.gbenroscience.parser.pro.turbo.tools.matrix;

import java.util.Arrays;

/**
 * High-Performance Pure Java Linear Algebra and Quantization Kernels (FP32 +
 * Q8_0). Designed to maximize HotSpot C2 / ART auto-vectorization and cache
 * locality.
 *
 * @author GBEMIRO
 */
public final class KernelsFloat {

    private KernelsFloat() {
    }

    // ===================== RoPE =====================
    /**
     * Precompute RoPE cos/sin tables in FP32
     */
    public static void precompute_rope_f32(int max_seq, int head_dim, float base,
            float[] cos_out, float[] sin_out) {
        int halfDim = head_dim / 2;
        for (int pos = 0; pos < max_seq; pos++) {
            for (int i = 0; i < halfDim; i++) {
                double theta = Math.pow(base, -2.0 * i / head_dim);
                double angle = pos * theta;
                int idx = pos * halfDim + i;
                cos_out[idx] = (float) Math.cos(angle);
                sin_out[idx] = (float) Math.sin(angle);
            }
        }
    }

    /**
     * This computes the static rotary frequency lookup tables for standard interleaved RoPE embeddings over the maximum sequence window.
     * Precomputes the static RoPE frequencies for lookup. Layout: flat arrays
     * of size [max_seq * (head_dim / 2)]
     */
    public static void precompute_rope(int max_seq, int head_dim, float base, float[] cos_table, float[] sin_table) {
        final int halfDim = head_dim / 2;
        for (int pos = 0; pos < max_seq; pos++) {
            final int stepOff = pos * halfDim;
            for (int i = 0; i < halfDim; i++) {
                double theta = Math.pow(base, -2.0 * i / head_dim);
                double angle = pos * theta;
                int idx = stepOff + i;
                cos_table[idx] = (float) Math.cos(angle);
                sin_table[idx] = (float) Math.sin(angle);
            }
        }
    }

    /**
     * Applies Split-Half Rotary Position Embeddings (GPT-NeoX style) in-place.
     * Pairs element i with element (i + head_dim / 2).
     */
    public static void apply_rope_f32_split(
            float[] q_f32_split, int qHeads,
            float[] k_f32_split, int kHeads,
            int head_dim, int pos,
            float[] cos_table, float[] sin_table) {

        final int halfDim = head_dim / 2;
        final int cosOff = pos * halfDim;

        // 1. Process Query Heads
        for (int h = 0; h < qHeads; h++) {
            final int headOff = h * head_dim;
            int i = 0;

            // Unroll 4x to maximize pipeline performance
            for (; i <= halfDim - 4; i += 4) {
                int cIdx = cosOff + i;
                int i0 = headOff + i;
                int i1 = i0 + halfDim;

                // Cache inputs
                float c0 = cos_table[cIdx];
                float s0 = sin_table[cIdx];
                float q0_v0 = q_f32_split[i0];
                float q0_v1 = q_f32_split[i1];

                float c1 = cos_table[cIdx + 1];
                float s1 = sin_table[cIdx + 1];
                float q1_v0 = q_f32_split[i0 + 1];
                float q1_v1 = q_f32_split[i1 + 1];

                float c2 = cos_table[cIdx + 2];
                float s2 = sin_table[cIdx + 2];
                float q2_v0 = q_f32_split[i0 + 2];
                float q2_v1 = q_f32_split[i1 + 2];

                float c3 = cos_table[cIdx + 3];
                float s3 = sin_table[cIdx + 3];
                float q3_v0 = q_f32_split[i0 + 3];
                float q3_v1 = q_f32_split[i1 + 3];

                // Apply rotation and write back
                q_f32_split[i0] = q0_v0 * c0 - q0_v1 * s0;
                q_f32_split[i1] = q0_v0 * s0 + q0_v1 * c0;

                q_f32_split[i0 + 1] = q1_v0 * c1 - q1_v1 * s1;
                q_f32_split[i1 + 1] = q1_v0 * s1 + q1_v1 * c1;

                q_f32_split[i0 + 2] = q2_v0 * c2 - q2_v1 * s2;
                q_f32_split[i1 + 2] = q2_v0 * s2 + q2_v1 * c2;

                q_f32_split[i0 + 3] = q3_v0 * c3 - q3_v1 * s3;
                q_f32_split[i1 + 3] = q3_v0 * s3 + q3_v1 * c3;
            }

            // Scalar clean-up loop for any remaining dimensions
            for (; i < halfDim; i++) {
                int i0 = headOff + i;
                int i1 = i0 + halfDim;
                float c = cos_table[cosOff + i];
                float s = sin_table[cosOff + i];
                float v0 = q_f32_split[i0];
                float v1 = q_f32_split[i1];

                q_f32_split[i0] = v0 * c - v1 * s;
                q_f32_split[i1] = v0 * s + v1 * c;
            }
        }

        // 2. Process Key Heads
        for (int h = 0; h < kHeads; h++) {
            final int headOff = h * head_dim;
            int i = 0;

            // Unroll 4x
            for (; i <= halfDim - 4; i += 4) {
                int cIdx = cosOff + i;
                int i0 = headOff + i;
                int i1 = i0 + halfDim;

                float c0 = cos_table[cIdx];
                float s0 = sin_table[cIdx];
                float k0_v0 = k_f32_split[i0];
                float k0_v1 = k_f32_split[i1];

                float c1 = cos_table[cIdx + 1];
                float s1 = sin_table[cIdx + 1];
                float k1_v0 = k_f32_split[i0 + 1];
                float k1_v1 = k_f32_split[i1 + 1];

                float c2 = cos_table[cIdx + 2];
                float s2 = sin_table[cIdx + 2];
                float k2_v0 = k_f32_split[i0 + 2];
                float k2_v1 = k_f32_split[i1 + 2];

                float c3 = cos_table[cIdx + 3];
                float s3 = sin_table[cIdx + 3];
                float k3_v0 = k_f32_split[i0 + 3];
                float k3_v1 = k_f32_split[i1 + 3];

                k_f32_split[i0] = k0_v0 * c0 - k0_v1 * s0;
                k_f32_split[i1] = k0_v0 * s0 + k0_v1 * c0;

                k_f32_split[i0 + 1] = k1_v0 * c1 - k1_v1 * s1;
                k_f32_split[i1 + 1] = k1_v0 * s1 + k1_v1 * c1;

                k_f32_split[i0 + 2] = k2_v0 * c2 - k2_v1 * s2;
                k_f32_split[i1 + 2] = k2_v0 * s2 + k2_v1 * c2;

                k_f32_split[i0 + 3] = k3_v0 * c3 - k3_v1 * s3;
                k_f32_split[i1 + 3] = k3_v0 * s3 + k3_v1 * c3;
            }

            for (; i < halfDim; i++) {
                int i0 = headOff + i;
                int i1 = i0 + halfDim;
                float c = cos_table[cosOff + i];
                float s = sin_table[cosOff + i];
                float v0 = k_f32_split[i0];
                float v1 = k_f32_split[i1];

                k_f32_split[i0] = v0 * c - v1 * s;
                k_f32_split[i1] = v0 * s + v1 * c;
            }
        }
    }

    /**
     * Standard interleaved RoPE
     */
    public static void rope_inplace(float[] q, int qHeads, float[] k, int kHeads,
            int head_dim, int pos, float[] cos_table, float[] sin_table) {
        final int halfDim = head_dim / 2;
        final int cosOff = pos * halfDim;

        applyRope(q, qHeads, head_dim, cos_table, sin_table, cosOff);
        applyRope(k, kHeads, head_dim, cos_table, sin_table, cosOff);
    }

    private static void applyRope(float[] buf, int heads, int head_dim,
            float[] cos, float[] sin, int cosOff) {
        final int halfDim = head_dim / 2;
        for (int h = 0; h < heads; h++) {
            final int off = h * head_dim;
            int i = 0;
            for (; i <= halfDim - 4; i += 4) {
                applyRopeUnrolled(buf, off, i, cos, sin, cosOff);
            }
            for (; i < halfDim; i++) {
                int idx0 = off + i * 2;
                int idx1 = idx0 + 1;
                float c = cos[cosOff + i];
                float s = sin[cosOff + i];
                float x0 = buf[idx0], x1 = buf[idx1];
                buf[idx0] = x0 * c - x1 * s;
                buf[idx1] = x0 * s + x1 * c;
            }
        }
    }

    private static void applyRopeUnrolled(float[] buf, int off, int i,
            float[] cos, float[] sin, int cosOff) {
        int i0 = i, i1 = i + 1, i2 = i + 2, i3 = i + 3;
        int idx0_0 = off + i0 * 2, idx1_0 = idx0_0 + 1;
        int idx0_1 = off + i1 * 2, idx1_1 = idx0_1 + 1;
        int idx0_2 = off + i2 * 2, idx1_2 = idx0_2 + 1;
        int idx0_3 = off + i3 * 2, idx1_3 = idx0_3 + 1;

        float c0 = cos[cosOff + i0], s0 = sin[cosOff + i0];
        float c1 = cos[cosOff + i1], s1 = sin[cosOff + i1];
        float c2 = cos[cosOff + i2], s2 = sin[cosOff + i2];
        float c3 = cos[cosOff + i3], s3 = sin[cosOff + i3];

        float q0_0 = buf[idx0_0], q1_0 = buf[idx1_0];
        float q0_1 = buf[idx0_1], q1_1 = buf[idx1_1];
        float q0_2 = buf[idx0_2], q1_2 = buf[idx1_2];
        float q0_3 = buf[idx0_3], q1_3 = buf[idx1_3];

        buf[idx0_0] = q0_0 * c0 - q1_0 * s0;
        buf[idx1_0] = q0_0 * s0 + q1_0 * c0;
        buf[idx0_1] = q0_1 * c1 - q1_1 * s1;
        buf[idx1_1] = q0_1 * s1 + q1_1 * c1;
        buf[idx0_2] = q0_2 * c2 - q1_2 * s2;
        buf[idx1_2] = q0_2 * s2 + q1_2 * c2;
        buf[idx0_3] = q0_3 * c3 - q1_3 * s3;
        buf[idx1_3] = q0_3 * s3 + q1_3 * c3;
    }

    /**
     * Split layout RoPE for workspace buffers
     */
    public static void rope_inplace_split_ws(float[] buf, int qOff, int qHeads, int kOff, int kHeads,
            int head_dim, int pos, float[] cos_table, float[] sin_table) {
        final int halfDim = head_dim >> 1;
        final int cosOff = pos * halfDim;
        for (int h = 0; h < qHeads; h++) {
            int evenOff = qOff + h * head_dim;
            int oddOff = evenOff + halfDim;
            for (int i = 0; i < halfDim; i++) {
                float c = cos_table[cosOff + i];
                float s = sin_table[cosOff + i];
                float x0 = buf[evenOff + i];
                float x1 = buf[oddOff + i];
                buf[evenOff + i] = x0 * c - x1 * s;
                buf[oddOff + i] = x0 * s + x1 * c;
            }
        }
        for (int h = 0; h < kHeads; h++) {
            int evenOff = kOff + h * head_dim;
            int oddOff = evenOff + halfDim;
            for (int i = 0; i < halfDim; i++) {
                float c = cos_table[cosOff + i];
                float s = sin_table[cosOff + i];
                float x0 = buf[evenOff + i];
                float x1 = buf[oddOff + i];
                buf[evenOff + i] = x0 * c - x1 * s;
                buf[oddOff + i] = x0 * s + x1 * c;
            }
        }
    }

    // ===================== Normalization & Softmax =====================    
    public static void rms_norm(float[] x, float[] gamma, float[] out, int M, int N, float eps) {
        for (int m = 0; m < M; m++) {
            final int off = m * N;
            float ss = 0.0f;
            int n = 0;
            for (; n <= N - 4; n += 4) {
                float v0 = x[off + n];
                float v1 = x[off + n + 1];
                float v2 = x[off + n + 2];
                float v3 = x[off + n + 3];
                ss += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }
            for (; n < N; n++) {
                ss += x[off + n] * x[off + n];
            }
            float scale = 1.0f / (float) Math.sqrt(ss / N + eps);

            n = 0;
            for (; n <= N - 4; n += 4) {
                out[off + n] = x[off + n] * scale * gamma[n];
                out[off + n + 1] = x[off + n + 1] * scale * gamma[n + 1];
                out[off + n + 2] = x[off + n + 2] * scale * gamma[n + 2];
                out[off + n + 3] = x[off + n + 3] * scale * gamma[n + 3];
            }
            for (; n < N; n++) {
                out[off + n] = x[off + n] * scale * gamma[n];
            }
        }
    }

    /**
     * Fast Root Mean Square Normalization (float32 variant optimized for
     * unrolled performance). Assumes features dimension is a multiple of 4.
     */
    public static void rms_norm_fast(
            float[] x, float[] gamma, float[] out,
            int batch, int features, float eps) {

        for (int b = 0; b < batch; b++) {
            final int offset = b * features;

            // Pass 1: Accumulate squared sums with 4x unrolling
            float sumSq = 0.0f;
            for (int i = 0; i < features; i += 4) {
                float v0 = x[offset + i];
                float v1 = x[offset + i + 1];
                float v2 = x[offset + i + 2];
                float v3 = x[offset + i + 3];
                sumSq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }

            // Calculate scale factor
            float rms = (float) (1.0 / Math.sqrt(sumSq / features + eps));

            // Pass 2: Multiply by scaling factors and scale with gamma weights
            for (int i = 0; i < features; i += 4) {
                int idx = offset + i;
                out[idx] = x[idx] * rms * gamma[i];
                out[idx + 1] = x[idx + 1] * rms * gamma[i + 1];
                out[idx + 2] = x[idx + 2] * rms * gamma[i + 2];
                out[idx + 3] = x[idx + 3] * rms * gamma[i + 3];
            }
        }
    }

    public static void softmax_row(float[] arr, int offset, int len) {
        if (len <= 0) {
            return;
        }

        // Max reduction with 4-way unroll for ILP
        float max = arr[offset];
        int i = 1;
        for (; i <= len - 4; i += 4) {
            float m0 = arr[offset + i];
            float m1 = arr[offset + i + 1];
            float m2 = arr[offset + i + 2];
            float m3 = arr[offset + i + 3];
            if (m0 > max) {
                max = m0;
            }
            if (m1 > max) {
                max = m1;
            }
            if (m2 > max) {
                max = m2;
            }
            if (m3 > max) {
                max = m3;
            }
        }
        for (; i < len; i++) {
            if (arr[offset + i] > max) {
                max = arr[offset + i];
            }
        }

        // Exp + sum with 4-way unroll
        float sum = 0.0f;
        i = 0;
        for (; i <= len - 4; i += 4) {
            float e0 = (float) Math.exp(arr[offset + i] - max);
            float e1 = (float) Math.exp(arr[offset + i + 1] - max);
            float e2 = (float) Math.exp(arr[offset + i + 2] - max);
            float e3 = (float) Math.exp(arr[offset + i + 3] - max);
            arr[offset + i] = e0;
            arr[offset + i + 1] = e1;
            arr[offset + i + 2] = e2;
            arr[offset + i + 3] = e3;
            sum += e0 + e1 + e2 + e3;
        }
        for (; i < len; i++) {
            float e = (float) Math.exp(arr[offset + i] - max);
            arr[offset + i] = e;
            sum += e;
        }

        // Normalize with 4-way unroll
        float invSum = 1.0f / (sum + 1e-7f);
        i = 0;
        for (; i <= len - 4; i += 4) {
            arr[offset + i] *= invSum;
            arr[offset + i + 1] *= invSum;
            arr[offset + i + 2] *= invSum;
            arr[offset + i + 3] *= invSum;
        }
        for (; i < len; i++) {
            arr[offset + i] *= invSum;
        }
    }

    // ===================== Matmul Helpers =====================
    public static void matmul_f32_1xN_axpy(float[] buf, float[] b, float[] out,
            int cOff, int K, int N, int aOff) {
        Arrays.fill(out, cOff, cOff + N, 0.0f);
        for (int k = 0; k < K; k++) {
            float ak = buf[aOff + k];
            if (ak == 0.0f) {
                continue;
            }
            int bRowOff = k * N;
            int n = 0;
            for (; n <= N - 4; n += 4) {
                out[cOff + n] += ak * b[bRowOff + n];
                out[cOff + n + 1] += ak * b[bRowOff + n + 1];
                out[cOff + n + 2] += ak * b[bRowOff + n + 2];
                out[cOff + n + 3] += ak * b[bRowOff + n + 3];
            }
            for (; n < N; n++) {
                out[cOff + n] += ak * b[bRowOff + n];
            }
        }
    }

    public static void matmul_swiglu(float[] x, int M, int K,
            float[] w_gate, float[] w_up,
            float[] out, int N) {
        for (int m = 0; m < M; m++) {
            final int xOff = m * K;
            final int outOff = m * N;
            int n = 0;
            for (; n <= N - 4; n += 4) {
                float g0 = 0f, g1 = 0f, g2 = 0f, g3 = 0f;
                float u0 = 0f, u1 = 0f, u2 = 0f, u3 = 0f;
                for (int k = 0; k < K; k++) {
                    float xv = x[xOff + k];
                    int idx = k * N + n;
                    g0 += xv * w_gate[idx];
                    g1 += xv * w_gate[idx + 1];
                    g2 += xv * w_gate[idx + 2];
                    g3 += xv * w_gate[idx + 3];
                    u0 += xv * w_up[idx];
                    u1 += xv * w_up[idx + 1];
                    u2 += xv * w_up[idx + 2];
                    u3 += xv * w_up[idx + 3];
                }
                out[outOff + n] = silu(g0) * u0;
                out[outOff + n + 1] = silu(g1) * u1;
                out[outOff + n + 2] = silu(g2) * u2;
                out[outOff + n + 3] = silu(g3) * u3;
            }
            for (; n < N; n++) {
                float g = 0f, u = 0f;
                for (int k = 0; k < K; k++) {
                    float xv = x[xOff + k];
                    g += xv * w_gate[k * N + n];
                    u += xv * w_up[k * N + n];
                }
                out[outOff + n] = silu(g) * u;
            }
        }
    }

    private static float silu(float x) {
        return x / (1.0f + (float) Math.exp(-x));
    }

    // ===================== Q8_0 GEMM =====================
    public static void gemm_batch_q8_f32(float[] a, byte[] b_q8, float[] b_scale, float[] c,
            int cOff, int M, int K, int N) {
        final int GROUP_SIZE = 32;
        Arrays.fill(c, cOff, cOff + M * N, 0.0f);

        for (int m = 0; m < M; m++) {
            final int aRowOff = m * K;
            final int cRowOff = cOff + m * N;
            for (int k = 0; k < K; k += GROUP_SIZE) {
                final int kLen = Math.min(GROUP_SIZE, K - k);
                float ak_max = 0.0f;
                for (int s = 0; s < kLen; s++) {
                    float val = Math.abs(a[aRowOff + k + s]);
                    if (val > ak_max) {
                        ak_max = val;
                    }
                }
                if (ak_max == 0.0f) {
                    continue;
                }

                final int scaleRowOff = (k / GROUP_SIZE) * N;
                for (int s = 0; s < kLen; s++) {
                    final float av = a[aRowOff + k + s];
                    if (av == 0.0f) {
                        continue;
                    }
                    final int bRowOff = (k + s) * N;
                    int n = 0;
                    for (; n <= N - 4; n += 4) {
                        c[cRowOff + n] += av * (b_q8[bRowOff + n] * b_scale[scaleRowOff + n]);
                        c[cRowOff + n + 1] += av * (b_q8[bRowOff + n + 1] * b_scale[scaleRowOff + n + 1]);
                        c[cRowOff + n + 2] += av * (b_q8[bRowOff + n + 2] * b_scale[scaleRowOff + n + 2]);
                        c[cRowOff + n + 3] += av * (b_q8[bRowOff + n + 3] * b_scale[scaleRowOff + n + 3]);
                    }
                    for (; n < N; n++) {
                        c[cRowOff + n] += av * (b_q8[bRowOff + n] * b_scale[scaleRowOff + n]);
                    }
                }
            }
        }
    }

    /**
     * This method encapsulates the multi-head/grouped-query attention (MHA/GQA) workflow for a single-token decoding iteration.
     * Cache Layout Assumptions:
     *
     * wq, wk, wv, and wo are in standard row-major layout.
     *
     * k_cache and v_cache feature a flat sequential token-major layout of shape
     * [max_seq, kv_heads * head_dim]. Executes a single-token fused
     * Multi-Head/Grouped-Query Attention decoding step. Overwrites the output
     * buffer in-place if specified.
     */
    public static void mha_decode_step(
            float[] x, float[] wq, float[] wk, float[] wv, float[] wo,
            float[] k_cache, float[] v_cache,
            float[] cos_table, float[] sin_table,
            float[] out,
            int dim, int num_heads, int kv_heads, int pos, int max_seq) {

        final int head_dim = dim / num_heads;
        final int kv_dim = kv_heads * head_dim;
        final int kv_mul = num_heads / kv_heads; // Head grouping factor for GQA/MQA
        final float sqrtHeadDim = (float) Math.sqrt(head_dim);

        // 1. Project input activations to Q, K, V (Row-Major Streaming)
        float[] q = new float[dim];
        float[] k = new float[kv_dim];
        float[] v = new float[kv_dim];

        for (int j = 0; j < dim; j++) {
            float xVal = x[j];
            int wOff = j * dim;
            int i = 0;
            for (; i <= dim - 4; i += 4) {
                q[i] += xVal * wq[wOff + i];
                q[i + 1] += xVal * wq[wOff + i + 1];
                q[i + 2] += xVal * wq[wOff + i + 2];
                q[i + 3] += xVal * wq[wOff + i + 3];
            }
            for (; i < dim; i++) {
                q[i] += xVal * wq[wOff + i];
            }
        }

        for (int j = 0; j < dim; j++) {
            float xVal = x[j];
            int wOff = j * kv_dim;
            int i = 0;
            for (; i <= kv_dim - 4; i += 4) {
                k[i] += xVal * wk[wOff + i];
                k[i + 1] += xVal * wk[wOff + i + 1];
                k[i + 2] += xVal * wk[wOff + i + 2];
                k[i + 3] += xVal * wk[wOff + i + 3];

                v[i] += xVal * wv[wOff + i];
                v[i + 1] += xVal * wv[wOff + i + 1];
                v[i + 2] += xVal * wv[wOff + i + 2];
                v[i + 3] += xVal * wv[wOff + i + 3];
            }
            for (; i < kv_dim; i++) {
                k[i] += xVal * wk[wOff + i];
                v[i] += xVal * wv[wOff + i];
            }
        }

        // 2. Apply Standard Interleaved RoPE to current Q and K tokens
        final int halfDim = head_dim / 2;
        final int cosOff = pos * halfDim;

        for (int h = 0; h < num_heads; h++) {
            int headOff = h * head_dim;
            for (int i = 0; i < head_dim; i += 2) {
                int cIdx = cosOff + (i / 2);
                float c = cos_table[cIdx];
                float s = sin_table[cIdx];
                float q0 = q[headOff + i];
                float q1 = q[headOff + i + 1];
                q[headOff + i] = q0 * c - q1 * s;
                q[headOff + i + 1] = q0 * s + q1 * c;
            }
        }

        for (int h = 0; h < kv_heads; h++) {
            int headOff = h * head_dim;
            for (int i = 0; i < head_dim; i += 2) {
                int cIdx = cosOff + (i / 2);
                float c = cos_table[cIdx];
                float s = sin_table[cIdx];
                float k0 = k[headOff + i];
                float k1 = k[headOff + i + 1];
                k[headOff + i] = k0 * c - k1 * s;
                k[headOff + i + 1] = k0 * s + k1 * c;
            }
        }

        // 3. Register current state into Key/Value Cache
        int cacheRowOff = pos * kv_dim;
        System.arraycopy(k, 0, k_cache, cacheRowOff, kv_dim);
        System.arraycopy(v, 0, v_cache, cacheRowOff, kv_dim);

        // 4. Compute Multi-Head Attention Attention scores and values matching context history
        float[] xb = new float[dim];
        float[] att = new float[pos + 1]; // Score buffer for current trajectory sequence

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / kv_mul;
            int qOff = h * head_dim;
            int kvHeadOff = kv_h * head_dim;

            // Query dot product against Key cache history
            float maxScore = -Float.MAX_VALUE;
            for (int t = 0; t <= pos; t++) {
                int tKOff = t * kv_dim + kvHeadOff;
                float score = 0.0f;
                int i = 0;
                for (; i <= head_dim - 4; i += 4) {
                    score += q[qOff + i] * k_cache[tKOff + i]
                            + q[qOff + i + 1] * k_cache[tKOff + i + 1]
                            + q[qOff + i + 2] * k_cache[tKOff + i + 2]
                            + q[qOff + i + 3] * k_cache[tKOff + i + 3];
                }
                for (; i < head_dim; i++) {
                    score += q[qOff + i] * k_cache[tKOff + i];
                }
                score /= sqrtHeadDim;
                att[t] = score;
                if (score > maxScore) {
                    maxScore = score;
                }
            }

            // Inline Softmax Normalized Exponential
            float sumExp = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float expVal = (float) Math.exp(att[t] - maxScore);
                att[t] = expVal;
                sumExp += expVal;
            }
            for (int t = 0; t <= pos; t++) {
                att[t] /= sumExp;
            }

            // Weighted Value Vector synthesis
            int xbOff = h * head_dim;
            for (int t = 0; t <= pos; t++) {
                float a = att[t];
                int tVOff = t * kv_dim + kvHeadOff;
                int i = 0;
                for (; i <= head_dim - 4; i += 4) {
                    xb[xbOff + i] += a * v_cache[tVOff + i];
                    xb[xbOff + i + 1] += a * v_cache[tVOff + i + 1];
                    xb[xbOff + i + 2] += a * v_cache[tVOff + i + 2];
                    xb[xbOff + i + 3] += a * v_cache[tVOff + i + 3];
                }
                for (; i < head_dim; i++) {
                    xb[xbOff + i] += a * v_cache[tVOff + i];
                }
            }
        }

        // 5. Output Projection (xb * wo) safely clearing output destinations
        for (int i = 0; i < dim; i++) {
            out[i] = 0.0f;
        }

        for (int j = 0; j < dim; j++) {
            float xbVal = xb[j];
            int wOff = j * dim;
            int i = 0;
            for (; i <= dim - 4; i += 4) {
                out[i] += xbVal * wo[wOff + i];
                out[i + 1] += xbVal * wo[wOff + i + 1];
                out[i + 2] += xbVal * wo[wOff + i + 2];
                out[i + 3] += xbVal * wo[wOff + i + 3];
            }
            for (; i < dim; i++) {
                out[i] += xbVal * wo[wOff + i];
            }
        }
    }

    // ===================== MHA Decode Step (Main Production Path) =====================
    /**
     * Zero-allocation decode step using Q8_0 weights for QKV projections.
     */
    public static void mha_decode_step_q8(
            float[] x,
            byte[] wq_q8, float[] wq_scale,
            byte[] wk_q8, float[] wk_scale,
            byte[] wv_q8, float[] wv_scale,
            float[] wo, // FP32 output projection
            float[] k_cache, float[] v_cache,
            float[] cos_table, float[] sin_table,
            float[] out,
            int dim, int num_heads, int kv_heads, int pos,
            float[] q_new, float[] k_new, float[] v_new,
            float[] attn_scores, float[] proj_temp) {

        final int head_dim = dim / num_heads;
        final int kv_dim = kv_heads * head_dim;
        final float scale = 1.0f / (float) Math.sqrt(head_dim);

        // QKV projections using fast Q8 GEMM
        gemm_batch_q8_f32(x, wq_q8, wq_scale, q_new, 0, 1, dim, num_heads * head_dim);
        gemm_batch_q8_f32(x, wk_q8, wk_scale, k_new, 0, 1, dim, kv_dim);
        gemm_batch_q8_f32(x, wv_q8, wv_scale, v_new, 0, 1, dim, kv_dim);

        rope_inplace(q_new, num_heads, k_new, kv_heads, head_dim, pos, cos_table, sin_table);

        int cacheOff = pos * kv_dim;
        System.arraycopy(k_new, 0, k_cache, cacheOff, kv_dim);
        System.arraycopy(v_new, 0, v_cache, cacheOff, kv_dim);

        int kvGroupSize = num_heads / kv_heads;
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / kvGroupSize;
            int qOff = h * head_dim;

            for (int j = 0; j <= pos; j++) {
                int kOff = j * kv_dim + kv_h * head_dim;
                float dot = dot_product_f32(q_new, qOff, k_cache, kOff, head_dim);
                attn_scores[j] = dot * scale;
            }

            softmax_row(attn_scores, 0, pos + 1);

            int outOff = h * head_dim;
            Arrays.fill(out, outOff, outOff + head_dim, 0.0f);

            for (int j = 0; j <= pos; j++) {
                float s = attn_scores[j];
                if (s < 1e-8f) {
                    continue;
                }
                int vOff = j * kv_dim + kv_h * head_dim;
                accumulate_v_f32(out, outOff, v_cache, vOff, head_dim, s);
            }
        }

        System.arraycopy(out, 0, proj_temp, 0, dim);
        matmul_f32_1xN_axpy(proj_temp, wo, out, 0, dim, dim, 0);
    }

    // ===================== Utils =====================
    public static void accumulate_v_f32(float[] dst, int dOff, float[] src, int sOff, int len, float scale) {
        int i = 0;
        for (; i <= len - 4; i += 4) {
            dst[dOff + i] += src[sOff + i] * scale;
            dst[dOff + i + 1] += src[sOff + i + 1] * scale;
            dst[dOff + i + 2] += src[sOff + i + 2] * scale;
            dst[dOff + i + 3] += src[sOff + i + 3] * scale;
        }
        for (; i < len; i++) {
            dst[dOff + i] += src[sOff + i] * scale;
        }
    }

    private static float dot_product_f32(float[] a, int aOff, float[] b, int bOff, int len) {
        float sum0 = 0f, sum1 = 0f, sum2 = 0f, sum3 = 0f;
        int i = 0;
        for (; i <= len - 4; i += 4) {
            sum0 += a[aOff + i] * b[bOff + i];
            sum1 += a[aOff + i + 1] * b[bOff + i + 1];
            sum2 += a[aOff + i + 2] * b[bOff + i + 2];
            sum3 += a[aOff + i + 3] * b[bOff + i + 3];
        }
        for (; i < len; i++) {
            sum0 += a[aOff + i] * b[bOff + i];
        }
        return sum0 + sum1 + sum2 + sum3;
    }

    // Convenience overload with allocations (for easy testing)
    public static void mha_decode_step_q8(
            float[] x,
            byte[] wq_q8, float[] wq_scale,
            byte[] wk_q8, float[] wk_scale,
            byte[] wv_q8, float[] wv_scale,
            float[] wo,
            float[] k_cache, float[] v_cache,
            float[] cos_table, float[] sin_table,
            float[] out,
            int dim, int num_heads, int kv_heads, int pos) {

        int head_dim = dim / num_heads;
        int kv_dim = kv_heads * head_dim;

        float[] q_new = new float[num_heads * head_dim];
        float[] k_new = new float[kv_dim];
        float[] v_new = new float[kv_dim];
        float[] attn_scores = new float[pos + 1];
        float[] proj_temp = new float[dim];

        mha_decode_step_q8(x, wq_q8, wq_scale, wk_q8, wk_scale, wv_q8, wv_scale, wo,
                k_cache, v_cache, cos_table, sin_table, out, dim, num_heads, kv_heads, pos,
                q_new, k_new, v_new, attn_scores, proj_temp);
    }

    /**
     * Matrix multiplication with optional bias fused: C[M,N] = A[M,K] * B[K,N]
     * + bias[N] Optimized for row-major layout with 4x unrolling across the
     * columns.
     */
    public static void matmul_bias_f32(float[] c, float[] a, float[] b, float[] bias, int M, int N, int K) {
        for (int m = 0; m < M; m++) {
            final int aRowOff = m * K;
            final int cRowOff = m * N;
            int n = 0;

            // 4x unrolling along the N dimension (columns)
            for (; n <= N - 4; n += 4) {
                float acc0 = (bias == null) ? 0.0f : bias[n];
                float acc1 = (bias == null) ? 0.0f : bias[n + 1];
                float acc2 = (bias == null) ? 0.0f : bias[n + 2];
                float acc3 = (bias == null) ? 0.0f : bias[n + 3];

                for (int k = 0; k < K; k++) {
                    float aVal = a[aRowOff + k];
                    int bIdx = k * N + n;
                    acc0 += aVal * b[bIdx];
                    acc1 += aVal * b[bIdx + 1];
                    acc2 += aVal * b[bIdx + 2];
                    acc3 += aVal * b[bIdx + 3];
                }

                c[cRowOff + n] = acc0;
                c[cRowOff + n + 1] = acc1;
                c[cRowOff + n + 2] = acc2;
                c[cRowOff + n + 3] = acc3;
            }

            // Tail loop for remaining columns
            for (; n < N; n++) {
                float acc = (bias == null) ? 0.0f : bias[n];
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOff + k] * b[k * N + n];
                }
                c[cRowOff + n] = acc;
            }
        }
    }

    /**
     * Standard matrix multiplication: C[aRows, bCols] = A[aRows, aCols] *
     * B[bRows, bCols] Designed to perform cleanly on single-token inferences
     * (aRows = 1) and batch sizes.
     */
    public static void matmul_down(float[] a, int aRows, int aCols, float[] b, int bRows, int bCols, float[] c) {
        final int M = aRows;
        final int K = aCols; // explicitly pairs with bRows
        final int N = bCols;

        for (int m = 0; m < M; m++) {
            final int aRowOff = m * K;
            final int cRowOff = m * N;
            int n = 0;

            // 4x unrolling along the N dimension
            for (; n <= N - 4; n += 4) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;

                for (int k = 0; k < K; k++) {
                    float aVal = a[aRowOff + k];
                    int bIdx = k * N + n;
                    acc0 += aVal * b[bIdx];
                    acc1 += aVal * b[bIdx + 1];
                    acc2 += aVal * b[bIdx + 2];
                    acc3 += aVal * b[bIdx + 3];
                }

                c[cRowOff + n] = acc0;
                c[cRowOff + n + 1] = acc1;
                c[cRowOff + n + 2] = acc2;
                c[cRowOff + n + 3] = acc3;
            }

            // Tail loop for remaining columns
            for (; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOff + k] * b[k * N + n];
                }
                c[cRowOff + n] = acc;
            }
        }
    }
}
