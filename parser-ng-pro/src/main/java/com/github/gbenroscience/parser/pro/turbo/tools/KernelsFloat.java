package com.github.gbenroscience.parser.pro.turbo.tools;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 *
 * @author GBEMIRO
 */
public final class KernelsFloat {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VLEN = F_SPECIES.length(); // 8 on AVX2, 16 on AVX-512

    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_PREFERRED;

    public static void apply_rope_f32_split(
            float[] q_f32_split, int qHeads,
            float[] k_f32_split, int kHeads,
            int head_dim, int pos,
            float[] cos_table, float[] sin_table) {

        if ((head_dim & 1) != 0) {
            throw new IllegalArgumentException("head_dim must be even, got " + head_dim);
        }

        final int halfDim = head_dim >>> 1;
        final int csOff = pos * halfDim;

        // 1. Rotate Queries
        for (int h = 0; h < qHeads; h++) {
            int headOff = h * head_dim;
            int x0Off = headOff;
            int x1Off = headOff + halfDim;
            rotate_head_split(q_f32_split, x0Off, x1Off, halfDim, cos_table, sin_table, csOff);
        }

        // 2. Rotate Keys
        for (int h = 0; h < kHeads; h++) {
            int headOff = h * head_dim;
            int x0Off = headOff;
            int x1Off = headOff + halfDim;
            rotate_head_split(k_f32_split, x0Off, x1Off, halfDim, cos_table, sin_table, csOff);
        }
    }

    private static void rotate_head_split(
            float[] buf, int x0Off, int x1Off, int halfDim,
            float[] cos_t, float[] sin_t, int csOff) {

        int i = 0;
        int step = F_SPECIES.length();

        for (; i < F_SPECIES.loopBound(halfDim); i += step) {
            FloatVector x0 = FloatVector.fromArray(F_SPECIES, buf, x0Off + i);
            FloatVector x1 = FloatVector.fromArray(F_SPECIES, buf, x1Off + i);
            FloatVector cos = FloatVector.fromArray(F_SPECIES, cos_t, csOff + i);
            FloatVector sin = FloatVector.fromArray(F_SPECIES, sin_t, csOff + i);

            FloatVector x0_new = x0.mul(cos).sub(x1.mul(sin));
            FloatVector x1_new = x0.mul(sin).add(x1.mul(cos));

            x0_new.intoArray(buf, x0Off + i);
            x1_new.intoArray(buf, x1Off + i);
        }

        for (; i < halfDim; i++) {
            float x0 = buf[x0Off + i];
            float x1 = buf[x1Off + i];
            float cos = cos_t[csOff + i];
            float sin = sin_t[csOff + i];

            buf[x0Off + i] = x0 * cos - x1 * sin;
            buf[x1Off + i] = x0 * sin + x1 * cos;
        }
    }

    /**
     * Precompute RoPE cos/sin tables in FP32 base = 10000.0f for Llama2,
     * 500000.0f for Llama3
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
     * FP32 Matmul + bias: C[M,N] = A[M,K] * B[K,N] + bias[N] Used for
     * O-projection and lm_head. Bias can be null. Handles N not divisible by
     * VLEN.
     */
    public static void matmul_bias_f32(
            float[] c, float[] a, float[] b, float[] bias,
            int M, int N, int K) {

        final int VLEN = F_SPECIES.length();

        for (int m = 0; m < M; m++) {
            final int aRowOff = m * K;
            final int cRowOff = m * N;

            int n = 0;
            // Vectorized loop
            for (; n < F_SPECIES.loopBound(N); n += VLEN) {
                FloatVector acc = bias == null
                        ? FloatVector.zero(F_SPECIES)
                        : FloatVector.fromArray(F_SPECIES, bias, n);

                for (int k = 0; k < K; k++) {
                    FloatVector aVal = FloatVector.broadcast(F_SPECIES, a[aRowOff + k]);
                    FloatVector bVec = FloatVector.fromArray(F_SPECIES, b, k * N + n);
                    acc = acc.fma(bVec, aVal);
                }
                acc.intoArray(c, cRowOff + n);
            }

            // Scalar tail for N % VLEN!= 0
            for (; n < N; n++) {
                float acc = bias == null ? 0.0f : bias[n];
                for (int k = 0; k < K; k++) {
                    acc += a[aRowOff + k] * b[k * N + n];
                }
                c[cRowOff + n] = acc;
            }
        }
    }

    /**
     * RMSNorm FP32: out = x * gamma / sqrt(mean(x^2) + eps) Used for attn_norm
     * and ffn_norm. Weights stay FP32. Handles features not divisible by VLEN.
     *
     * @param x
     * @param gamma
     * @param out
     * @param batch
     * @param features
     * @param eps
     *
     */
    public static void rms_norm_fast(
            float[] x, float[] gamma, float[] out,
            int batch, int features, float eps) {

        final int VLEN = F_SPECIES.length();

        for (int b = 0; b < batch; b++) {
            final int off = b * features;

            // 1. Compute sum(x^2)
            float sumSq = 0.0f;
            int i = 0;
            FloatVector vSumSq = FloatVector.zero(F_SPECIES);
            for (; i < F_SPECIES.loopBound(features); i += VLEN) {
                FloatVector xv = FloatVector.fromArray(F_SPECIES, x, off + i);
                vSumSq = vSumSq.fma(xv, xv); // sum += x*x
            }
            sumSq = vSumSq.reduceLanes(VectorOperators.ADD);
            // Scalar tail
            for (; i < features; i++) {
                float v = x[off + i];
                sumSq += v * v;
            }

            // 2. rsqrt: 1 / sqrt(mean + eps)
            float rms = 1.0f / (float) Math.sqrt(sumSq / features + eps);
            FloatVector vRms = FloatVector.broadcast(F_SPECIES, rms);

            // 3. out = x * gamma * rms
            i = 0;
            for (; i < F_SPECIES.loopBound(features); i += VLEN) {
                FloatVector xv = FloatVector.fromArray(F_SPECIES, x, off + i);
                FloatVector gv = FloatVector.fromArray(F_SPECIES, gamma, i);
                xv.mul(vRms).mul(gv).intoArray(out, off + i);
            }
            // Scalar tail
            for (; i < features; i++) {
                out[off + i] = x[off + i] * rms * gamma[i];
            }
        }
    }

    private static void softmax_row_f32(float[] arr, int offset, int len) {
        // Pass 1: max
        float max = -Float.MAX_VALUE;
        int i = 0;
        FloatVector vMax = FloatVector.broadcast(F_SPECIES, -Float.MAX_VALUE);
        for (; i < F_SPECIES.loopBound(len); i += F_SPECIES.length()) {
            vMax = vMax.max(FloatVector.fromArray(F_SPECIES, arr, offset + i));
        }
        max = Math.max(max, vMax.reduceLanes(VectorOperators.MAX));
        for (; i < len; i++) {
            max = Math.max(max, arr[offset + i]);
        }

        // Pass 2: exp(x - max) and sum
        float sum = 0.0f;
        FloatVector vMaxB = FloatVector.broadcast(F_SPECIES, max);
        FloatVector vSum = FloatVector.zero(F_SPECIES);
        i = 0;
        for (; i < F_SPECIES.loopBound(len); i += F_SPECIES.length()) {
            FloatVector xv = FloatVector.fromArray(F_SPECIES, arr, offset + i);
            FloatVector ex = xv.sub(vMaxB).lanewise(VectorOperators.EXP);
            ex.intoArray(arr, offset + i); // store back
            vSum = vSum.add(ex);
        }
        sum = vSum.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            float ex = (float) Math.exp(arr[offset + i] - max);
            arr[offset + i] = ex;
            sum += ex;
        }

        // Pass 3: normalize
        float invSum = 1.0f / sum;
        FloatVector vInvSum = FloatVector.broadcast(F_SPECIES, invSum);
        i = 0;
        for (; i < F_SPECIES.loopBound(len); i += F_SPECIES.length()) {
            FloatVector ex = FloatVector.fromArray(F_SPECIES, arr, offset + i);
            ex.mul(vInvSum).intoArray(arr, offset + i);
        }
        for (; i < len; i++) {
            arr[offset + i] *= invSum;
        }
    }
 // RoPE for workspace slices: split layout [x0,x2...][x1,x3...]
    public static void rope_inplace_split_ws(float[] buf, int qOff, int qHeads, int kOff, int kHeads,
                                             int head_dim, int pos, float[] cos_table, float[] sin_table) {
        final int halfDim = head_dim >> 1;
        if ((head_dim & 1)!= 0) throw new IllegalArgumentException("head_dim must be even");
        int cosOff = pos * halfDim;

        // Q
        for (int h = 0; h < qHeads; h++) {
            int headOff = qOff + h * head_dim;
            int evenOff = headOff;
            int oddOff = headOff + halfDim;
            int i = 0;
            for (; i < F_SPECIES.loopBound(halfDim); i += F_SPECIES.length()) {
                FloatVector cos = FloatVector.fromArray(F_SPECIES, cos_table, cosOff + i);
                FloatVector sin = FloatVector.fromArray(F_SPECIES, sin_table, cosOff + i);
                FloatVector x0 = FloatVector.fromArray(F_SPECIES, buf, evenOff + i);
                FloatVector x1 = FloatVector.fromArray(F_SPECIES, buf, oddOff + i);
                x0.mul(cos).sub(x1.mul(sin)).intoArray(buf, evenOff + i);
                x0.mul(sin).add(x1.mul(cos)).intoArray(buf, oddOff + i);
            }
            for (; i < halfDim; i++) {
                float c = cos_table[cosOff + i], s = sin_table[cosOff + i];
                float x0 = buf[evenOff + i], x1 = buf[oddOff + i];
                buf[evenOff + i] = x0 * c - x1 * s;
                buf[oddOff + i] = x0 * s + x1 * c;
            }
        }

        // K - same logic
        for (int h = 0; h < kHeads; h++) {
            int headOff = kOff + h * head_dim;
            int evenOff = headOff;
            int oddOff = headOff + halfDim;
            int i = 0;
            for (; i < F_SPECIES.loopBound(halfDim); i += F_SPECIES.length()) {
                FloatVector cos = FloatVector.fromArray(F_SPECIES, cos_table, cosOff + i);
                FloatVector sin = FloatVector.fromArray(F_SPECIES, sin_table, cosOff + i);
                FloatVector x0 = FloatVector.fromArray(F_SPECIES, buf, evenOff + i);
                FloatVector x1 = FloatVector.fromArray(F_SPECIES, buf, oddOff + i);
                x0.mul(cos).sub(x1.mul(sin)).intoArray(buf, evenOff + i);
                x0.mul(sin).add(x1.mul(cos)).intoArray(buf, oddOff + i);
            }
            for (; i < halfDim; i++) {
                float c = cos_table[cosOff + i], s = sin_table[cosOff + i];
                float x0 = buf[evenOff + i], x1 = buf[oddOff + i];
                buf[evenOff + i] = x0 * c - x1 * s;
                buf[oddOff + i] = x0 * s + x1 * c;
            }
        }
    }

    // Overload: out = A[1,K] @ B[K,N], reading A from buf[aOff], writing C to out[cOff]
    public static void matmul_f32_1xN_axpy(float[] buf, float[] b, float[] out, int cOff, int K, int N, int aOff) {
        java.util.Arrays.fill(out, cOff, cOff + N, 0.0f);
        for (int k = 0; k < K; k++) {
            float ak = buf[aOff + k];
            if (ak == 0.0f) continue;
            FloatVector av = FloatVector.broadcast(F_SPECIES, ak);
            int bRowOff = k * N;
            int n = 0;
            for (; n <= N - 4 * F_SPECIES.length(); n += 4 * F_SPECIES.length()) {
                FloatVector c0 = FloatVector.fromArray(F_SPECIES, out, cOff + n);
                FloatVector c1 = FloatVector.fromArray(F_SPECIES, out, cOff + n + F_SPECIES.length());
                FloatVector c2 = FloatVector.fromArray(F_SPECIES, out, cOff + n + 2 * F_SPECIES.length());
                FloatVector c3 = FloatVector.fromArray(F_SPECIES, out, cOff + n + 3 * F_SPECIES.length());
                FloatVector b0 = FloatVector.fromArray(F_SPECIES, b, bRowOff + n);
                FloatVector b1 = FloatVector.fromArray(F_SPECIES, b, bRowOff + n + F_SPECIES.length());
                FloatVector b2 = FloatVector.fromArray(F_SPECIES, b, bRowOff + n + 2 * F_SPECIES.length());
                FloatVector b3 = FloatVector.fromArray(F_SPECIES, b, bRowOff + n + 3 * F_SPECIES.length());
                c0.fma(av, b0).intoArray(out, cOff + n);
                c1.fma(av, b1).intoArray(out, cOff + n + F_SPECIES.length());
                c2.fma(av, b2).intoArray(out, cOff + n + 2 * F_SPECIES.length());
                c3.fma(av, b3).intoArray(out, cOff + n + 3 * F_SPECIES.length());
            }
            for (; n < N; n += F_SPECIES.length()) {
                VectorMask<Float> mask = F_SPECIES.indexInRange(n, N);
                FloatVector cv = FloatVector.fromArray(F_SPECIES, out, cOff + n, mask);
                FloatVector bv = FloatVector.fromArray(F_SPECIES, b, bRowOff + n, mask);
                cv.fma(av, bv).intoArray(out, cOff + n, mask);
            }
        }
    }
    
    // temp[d] += scale * v[d], vectorized
public static void accumulate_v_f32(float[] dst, int dOff, float[] src, int sOff, int len, float scale) {
    FloatVector sv = FloatVector.broadcast(F_SPECIES, scale);
    int i = 0;
    for (; i < F_SPECIES.loopBound(len); i += F_SPECIES.length()) {
        FloatVector dv = FloatVector.fromArray(F_SPECIES, dst, dOff + i);
        FloatVector vv = FloatVector.fromArray(F_SPECIES, src, sOff + i);
        FloatVector res = dv.fma(vv, sv); // dv + vv * scale
        res.intoArray(dst, dOff + i);
    }
    for (; i < len; i++) {
        dst[dOff + i] += src[sOff + i] * scale;
    }
}

    // C[1,N] = A[1,K] @ B[K,N] -> AXPY: C[n] += A[k] * B[k,n]
    private static void matmul_f32_1xN_axpy(float[] a, float[] b, float[] c, int cOff, int K, int N) {
        // Zero c first - caller responsibility or do it here
        java.util.Arrays.fill(c, cOff, cOff + N, 0.0f);

        for (int k = 0; k < K; k++) {
            float ak = a[k]; // broadcast scalar
            if (ak == 0.0f) {
                continue; // Sparsity: SwiGLU output ~50% zeros
            }
            FloatVector av = FloatVector.broadcast(F_SPECIES, ak);
            int bRowOff = k * N;

            int n = 0;
            // Unroll by 4 for ILP - hides FMA latency
            for (; n <= N - 4 * F_SPECIES.length(); n += 4 * F_SPECIES.length()) {
                FloatVector c0 = FloatVector.fromArray(F_SPECIES, c, cOff + n);
                FloatVector c1 = FloatVector.fromArray(F_SPECIES, c, cOff + n + F_SPECIES.length());
                FloatVector c2 = FloatVector.fromArray(F_SPECIES, c, cOff + n + 2 * F_SPECIES.length());
                FloatVector c3 = FloatVector.fromArray(F_SPECIES, c, cOff + n + 3 * F_SPECIES.length());

                FloatVector b0 = FloatVector.fromArray(F_SPECIES, b, bRowOff + n);
                FloatVector b1 = FloatVector.fromArray(F_SPECIES, b, bRowOff + n + F_SPECIES.length());
                FloatVector b2 = FloatVector.fromArray(F_SPECIES, b, bRowOff + n + 2 * F_SPECIES.length());
                FloatVector b3 = FloatVector.fromArray(F_SPECIES, b, bRowOff + n + 3 * F_SPECIES.length());

                c0.fma(av, b0).intoArray(c, cOff + n);
                c1.fma(av, b1).intoArray(c, cOff + n + F_SPECIES.length());
                c2.fma(av, b2).intoArray(c, cOff + n + 2 * F_SPECIES.length());
                c3.fma(av, b3).intoArray(c, cOff + n + 3 * F_SPECIES.length());
            }
            // Tail
            for (; n < N; n += F_SPECIES.length()) {
                VectorMask<Float> mask = F_SPECIES.indexInRange(n, N);
                FloatVector cv = FloatVector.fromArray(F_SPECIES, c, cOff + n, mask);
                FloatVector bv = FloatVector.fromArray(F_SPECIES, b, bRowOff + n, mask);
                cv.fma(av, bv).intoArray(c, cOff + n, mask);
            }
        }
    }

    private static float dot_product_f32_unrolled(float[] a, int aOff, float[] b, int bOff, int len) {
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        FloatVector acc2 = FloatVector.zero(F_SPECIES);
        FloatVector acc3 = FloatVector.zero(F_SPECIES);

        int i = 0;
        for (; i <= len - 4 * F_SPECIES.length(); i += 4 * F_SPECIES.length()) {
            FloatVector av0 = FloatVector.fromArray(F_SPECIES, a, aOff + i);
            FloatVector bv0 = FloatVector.fromArray(F_SPECIES, b, bOff + i);
            acc0 = acc0.fma(av0, bv0);

            FloatVector av1 = FloatVector.fromArray(F_SPECIES, a, aOff + i + F_SPECIES.length());
            FloatVector bv1 = FloatVector.fromArray(F_SPECIES, b, bOff + i + F_SPECIES.length());
            acc1 = acc1.fma(av1, bv1);

            FloatVector av2 = FloatVector.fromArray(F_SPECIES, a, aOff + i + 2 * F_SPECIES.length());
            FloatVector bv2 = FloatVector.fromArray(F_SPECIES, b, bOff + i + 2 * F_SPECIES.length());
            acc2 = acc2.fma(av2, bv2);

            FloatVector av3 = FloatVector.fromArray(F_SPECIES, a, aOff + i + 3 * F_SPECIES.length());
            FloatVector bv3 = FloatVector.fromArray(F_SPECIES, b, bOff + i + 3 * F_SPECIES.length());
            acc3 = acc3.fma(av3, bv3);
        }
        FloatVector acc = acc0.add(acc1).add(acc2).add(acc3);

        // Tail with mask
        VectorMask<Float> mask = F_SPECIES.indexInRange(i, len);
        FloatVector av = FloatVector.fromArray(F_SPECIES, a, aOff + i, mask);
        FloatVector bv = FloatVector.fromArray(F_SPECIES, b, bOff + i, mask);
        acc = acc.add(av.mul(bv));

        return acc.reduceLanes(VectorOperators.ADD);
    }

    /**
     * MHA decode with Q8_0 KV cache. 4x memory reduction vs FP32. Cache layout:
     * k_cache_q8[seq * kv_heads * head_dim], k_scale_cache[seq * kv_heads]
     */
    public static void mha_decode_step_q8(
            float[] x, // [dim]
            byte[] wq_q8, float[] wq_scale, // Q8_0 weights [dim, num_heads*head_dim]
            byte[] wk_q8, float[] wk_scale,
            byte[] wv_q8, float[] wv_scale,
            float[] wo, // FP32 output proj [dim, dim]
            byte[] k_cache_q8, float[] k_scale_cache, // [max_seq, kv_heads, head_dim] Q8 + scales
            byte[] v_cache_q8, float[] v_scale_cache,
            float[] cos_table, float[] sin_table,
            float[] out, // [dim]
            float[] workspace, // pre-allocated: qLen + 2*kv_dim + dim + max_seq
            int dim, int num_heads, int kv_heads, int pos, int head_dim) {

        final int kv_dim = kv_heads * head_dim;
        final int qLen = num_heads * head_dim;
        final float scale = 1.0f / (float) Math.sqrt(head_dim);

        // Workspace layout: [q | k | v | temp | attn]
        int qOff = 0;
        int kOff = qOff + qLen;
        int vOff = kOff + kv_dim;
        int tempOff = vOff + kv_dim;
        int attnOff = tempOff + dim;

        // 1. QKV projections: dequant weights + AXPY GEMV
        gemv_q8_f32(x, wq_q8, wq_scale, workspace, qOff, dim, qLen);
        gemv_q8_f32(x, wk_q8, wk_scale, workspace, kOff, dim, kv_dim);
        gemv_q8_f32(x, wv_q8, wv_scale, workspace, vOff, dim, kv_dim);

        // 2. RoPE in-place on workspace Q,K
        rope_inplace_split_ws(workspace, qOff, num_heads, kOff, kv_heads, head_dim, pos, cos_table, sin_table);

        // 3. Quantize K,V to cache with per-head scales
        int kCacheBase = pos * kv_dim;
        int scaleBase = pos * kv_heads;
        for (int h = 0; h < kv_heads; h++) {
            int kHeadOff = kOff + h * head_dim;
            int vHeadOff = vOff + h * head_dim;
            int cacheHeadOff = kCacheBase + h * head_dim;

            // K
            float kmax = KernelsInt8.absmax(workspace, kHeadOff, head_dim);
            float kscale = kmax / 127.0f;
            k_scale_cache[scaleBase + h] = kscale;
            KernelsInt8.quantize_f32_to_i8(workspace, kHeadOff, head_dim, k_cache_q8, cacheHeadOff, kscale);

            // V
            float vmax = KernelsInt8.absmax(workspace, vHeadOff, head_dim);
            float vscale = vmax / 127.0f;
            v_scale_cache[scaleBase + h] = vscale;
            KernelsInt8.quantize_f32_to_i8(workspace, vHeadOff, head_dim, v_cache_q8, cacheHeadOff, vscale);
        }

        // 4. Attention: dequant K,V on the fly
        float[] kDequantBuf = new float[head_dim]; // Stack alloc, small
        float[] vDequantBuf = new float[head_dim];

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / (num_heads / kv_heads);
            int qHeadOff = qOff + h * head_dim;

            // Scores: Q_h @ K_j^T
            for (int j = 0; j <= pos; j++) {
                float kscale = k_scale_cache[j * kv_heads + kv_h];
                int kCacheOff = j * kv_dim + kv_h * head_dim;
                KernelsInt8.dequant_i8_to_f32(k_cache_q8, kCacheOff, head_dim, kDequantBuf, 0, kscale);

                float dot = dot_product_f32_unrolled(workspace, qHeadOff, kDequantBuf, 0, head_dim);
                workspace[attnOff + j] = dot * scale;
            }
            softmax_row_f32(workspace, attnOff, pos + 1);

            // Aggregate V: temp_h = sum_j attn[j] * V_j
            int tempHeadOff = tempOff + h * head_dim;
            java.util.Arrays.fill(workspace, tempHeadOff, tempHeadOff + head_dim, 0.0f);

            for (int j = 0; j <= pos; j++) {
                float s = workspace[attnOff + j];
                if (s < 1e-8f) {
                    continue;
                }

                float vscale = v_scale_cache[j * kv_heads + kv_h];
                int vCacheOff = j * kv_dim + kv_h * head_dim;
                KernelsInt8.dequant_i8_to_f32(v_cache_q8, vCacheOff, head_dim, vDequantBuf, 0, vscale);

                accumulate_v_f32(workspace, tempHeadOff, vDequantBuf, 0, head_dim, s);
            }
        }

        // 5. O-proj: out = temp @ wo, FP32
        matmul_f32_1xN_axpy(workspace, wo, out, 0, dim, dim, tempOff);
    }

// Q8_0 GEMV: c[1,N] += a[1,K] @ B_q8[K,N]
    private static void gemv_q8_f32(float[] a, byte[] b_q8, float[] b_scale, float[] c, int cOff, int K, int N) {
        java.util.Arrays.fill(c, cOff, cOff + N, 0.0f);
        final int GROUP_SIZE = 32; // Q8_0 block size

        for (int k = 0; k < K; k += GROUP_SIZE) {
            int kLen = Math.min(GROUP_SIZE, K - k);
            float ak_max = KernelsInt8.absmax(a, k, kLen);
            if (ak_max == 0.0f) {
                continue;
            }

            FloatVector av = FloatVector.zero(F_SPECIES);
            for (int i = 0; i < kLen; i++) {
                av = av.withLane(i, a[k + i]);
            }

            for (int n = 0; n < N; n += F_SPECIES.length()) {
                FloatVector acc = FloatVector.fromArray(F_SPECIES, c, cOff + n);
                float bscale = b_scale[(k / GROUP_SIZE) * N + n];
                float scale = (ak_max / 127.0f) * bscale;

                // Dequant B block and FMA
                for (int i = 0; i < kLen; i++) {
                    byte bval = b_q8[(k + i) * N + n]; // scalar for now, vectorize if needed
                    acc = acc.add(bval * a[k + i] * scale);
                }
                acc.intoArray(c, cOff + n);
            }
        }
    }

}
