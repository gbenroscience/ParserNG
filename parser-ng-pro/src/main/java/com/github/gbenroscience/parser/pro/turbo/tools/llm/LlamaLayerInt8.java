package com.github.gbenroscience.parser.pro.turbo.tools.llm;
 

import com.github.gbenroscience.parser.pro.turbo.tools.matrix.KernelsFloat;
import com.github.gbenroscience.parser.pro.turbo.tools.matrix.KernelsInt8;

/**
 * Highly optimized quantized INT8 Llama inference layer engineered with strict mechanical sympathy.
 * Eliminates garbage collection allocation pressure and uses manual batching to maximize JIT auto-vectorization.
 * * @author GBEMIRO
 */
public final class LlamaLayerInt8 {

    // ThreadLocal scratchpad to eliminate per-token transient array allocations during forward pass execution
    private static final ThreadLocal<Scratchpad> SCRATCH = ThreadLocal.withInitial(Scratchpad::new);

    private static final class Scratchpad {
        float[] temp_f32 = new float[4096];
        float[] swiglu_out_f32 = new float[11008];
        byte[] x_q8 = new byte[4096];
        byte[] q_new_q8 = new byte[4096];
        byte[] k_new_q8 = new byte[4096];
        byte[] v_new_q8 = new byte[4096];
        float[] q_scale = new float[4096];
        float[] k_scale = new float[4096];
        float[] v_scale = new float[4096];
        float[] q_f32 = new float[4096];
        float[] k_f32 = new float[4096];
        float[] v_f32 = new float[4096];
        float[] attn_scores = new float[4096];
        float[] k_temp = new float[128];
        float[] temp_clone = new float[4096];
        
        byte[] gate_q8 = new byte[11008];
        byte[] up_q8 = new byte[11008];
        float[] gate_scale = new float[11008];
        float[] up_scale = new float[11008];
        byte[] a_q8 = new byte[11008];

        void ensureCapacity(int dim, int hidden, int max_seq, int head_dim) {
            if (temp_f32.length < dim) temp_f32 = new float[dim];
            if (swiglu_out_f32.length < hidden) swiglu_out_f32 = new float[hidden];
            if (x_q8.length < dim) x_q8 = new byte[dim];
            if (q_new_q8.length < dim) q_new_q8 = new byte[dim];
            int kv_dim = dim; 
            if (k_new_q8.length < kv_dim) k_new_q8 = new byte[kv_dim];
            if (v_new_q8.length < kv_dim) v_new_q8 = new byte[kv_dim];
            if (q_scale.length < dim) q_scale = new float[dim];
            if (k_scale.length < kv_dim) k_scale = new float[kv_dim];
            if (v_scale.length < kv_dim) v_scale = new float[kv_dim];
            if (q_f32.length < dim) q_f32 = new float[dim];
            if (k_f32.length < kv_dim) k_f32 = new float[kv_dim];
            if (v_f32.length < kv_dim) v_f32 = new float[kv_dim];
            if (attn_scores.length < max_seq) attn_scores = new float[max_seq];
            if (k_temp.length < head_dim) k_temp = new float[head_dim];
            if (temp_clone.length < dim) temp_clone = new float[dim];
            if (gate_q8.length < hidden) gate_q8 = new byte[hidden];
            if (up_q8.length < hidden) up_q8 = new byte[hidden];
            if (gate_scale.length < hidden) gate_scale = new float[hidden];
            if (up_scale.length < hidden) up_scale = new float[hidden];
            if (a_q8.length < hidden) a_q8 = new byte[hidden];
        }
    }

    // Model config - same as FP32
    public static final class Config {
        public int dim = 4096;
        public int hidden_dim = 11008;
        public int num_heads = 32;
        public int kv_heads = 32; // GQA: 8 for Llama2-70B
        public int head_dim = 128;
        public int max_seq = 4096;
        public float norm_eps = 1e-5f; // FP32 eps
        public float rope_theta = 10000.0f;
    }

    // INT8 Weights for one layer
    public static final class WeightsQ8 {
        // Attention: weights are INT8 + per-channel scale
        public byte[] wq_q8, wk_q8, wv_q8; // [dim, dim] or [dim, kv_dim]
        public float[] wq_scale, wk_scale, wv_scale; // [dim] or [kv_dim]
        public float[] wo; // [dim, dim] - keep O proj FP32, it's small
        public float[] attn_norm_gamma; // [dim] FP32

        // FFN: INT8
        public byte[] w_gate_q8, w_up_q8; // [dim, hidden]
        public float[] w_gate_scale, w_up_scale; // [hidden]
        public byte[] w_down_q8; // [hidden, dim]
        public float[] w_down_scale; // [dim]
        public float[] ffn_norm_gamma; // [dim] FP32
    }

    // Per-sequence state with INT8 cache
    public static final class StateQ8 {
        public byte[] k_cache_q8, v_cache_q8; // [max_seq, kv_heads, head_dim]
        public float[] k_scale_cache, v_scale_cache; // [max_seq, kv_heads]
        public float[] cos_table, sin_table; // [max_seq, head_dim/2] FP32
        public int pos = 0;

        public StateQ8(Config cfg) {
            int kv_dim = cfg.kv_heads * cfg.head_dim;
            k_cache_q8 = new byte[cfg.max_seq * kv_dim];
            v_cache_q8 = new byte[cfg.max_seq * kv_dim];
            k_scale_cache = new float[cfg.max_seq * cfg.kv_heads];
            v_scale_cache = new float[cfg.max_seq * cfg.kv_heads];
            cos_table = new float[cfg.max_seq * cfg.head_dim / 2];
            sin_table = new float[cfg.max_seq * cfg.head_dim / 2];
            KernelsFloat.precompute_rope_f32(cfg.max_seq, cfg.head_dim, cfg.rope_theta, cos_table, sin_table);
        }
    }

    /**
     * Run one INT8 decoder layer for a single token x_f32: [dim] - input FP32, modified in-place
     */
    public static void forward_layer_q8(
            float[] x_f32, // [dim] - in/out FP32
            WeightsQ8 w,
            StateQ8 s,
            Config cfg) {

        final int dim = cfg.dim;
        final int hidden = cfg.hidden_dim;
        
        Scratchpad sp = SCRATCH.get();
        sp.ensureCapacity(dim, hidden, cfg.max_seq, cfg.head_dim);
        float[] temp_f32 = sp.temp_f32;

        // === 1. Attention block ===
        // x = x + attn(rms_norm(x))
        // 1a. RMSNorm FP32 - norm weights stay FP32
        KernelsFloat.rms_norm_fast(x_f32, w.attn_norm_gamma, temp_f32, 1, dim, cfg.norm_eps);

        // 1b. INT8 Attention: quantize -> qkv -> rope -> cache -> attn -> dequant
        mha_decode_step_q8(
                temp_f32, // input FP32
                w.wq_q8, w.wq_scale, w.wk_q8, w.wk_scale, w.wv_q8, w.wv_scale, w.wo,
                s.k_cache_q8, s.k_scale_cache, s.v_cache_q8, s.v_scale_cache,
                s.cos_table, s.sin_table,
                temp_f32, // output FP32, reuse buffer
                dim, cfg.num_heads, cfg.kv_heads, s.pos, cfg.max_seq
        );

        // 1c. Residual: x += attn_out FP32
        // Spatial unrolling loop by 8 to eliminate execution hazards and maximize pipeline capacity
        int i = 0;
        for (; i <= dim - 8; i += 8) {
            x_f32[i]     += temp_f32[i];
            x_f32[i + 1] += temp_f32[i + 1];
            x_f32[i + 2] += temp_f32[i + 2];
            x_f32[i + 3] += temp_f32[i + 3];
            x_f32[i + 4] += temp_f32[i + 4];
            x_f32[i + 5] += temp_f32[i + 5];
            x_f32[i + 6] += temp_f32[i + 6];
            x_f32[i + 7] += temp_f32[i + 7];
        }
        for (; i < dim; i++) {
            x_f32[i] += temp_f32[i];
        }

        // === 2. FFN block ===
        // x = x + ffn(rms_norm(x))
        // 2a. RMSNorm FP32
        KernelsFloat.rms_norm_fast(x_f32, w.ffn_norm_gamma, temp_f32, 1, dim, cfg.norm_eps);

        // 2b. INT8 SwiGLU: gate = silu(x @ W_gate_q8) * (x @ W_up_q8)
        float[] swiglu_out_f32 = sp.swiglu_out_f32;
        matmul_swiglu_q8(
                temp_f32, 1, dim,
                w.w_gate_q8, w.w_gate_scale,
                w.w_up_q8, w.w_up_scale,
                swiglu_out_f32, hidden
        );

        // 2c. INT8 Down proj: ffn_out = swiglu_out @ W_down_q8
        matmul_down_q8(
                swiglu_out_f32, 1, hidden,
                w.w_down_q8, w.w_down_scale,
                temp_f32, dim // output to temp_f32
        );

        // 2d. Residual: x += ffn_out
        i = 0;
        for (; i <= dim - 8; i += 8) {
            x_f32[i]     += temp_f32[i];
            x_f32[i + 1] += temp_f32[i + 1];
            x_f32[i + 2] += temp_f32[i + 2];
            x_f32[i + 3] += temp_f32[i + 3];
            x_f32[i + 4] += temp_f32[i + 4];
            x_f32[i + 5] += temp_f32[i + 5];
            x_f32[i + 6] += temp_f32[i + 6];
            x_f32[i + 7] += temp_f32[i + 7];
        }
        for (; i < dim; i++) {
            x_f32[i] += temp_f32[i];
        }
    }

    /**
     * INT8 MHA decode step
     */
    private static void mha_decode_step_q8(
            float[] x_f32, byte[] wq_q8, float[] wq_scale, byte[] wk_q8, float[] wk_scale,
            byte[] wv_q8, float[] wv_scale, float[] wo,
            byte[] k_cache_q8, float[] k_scale_cache, byte[] v_cache_q8, float[] v_scale_cache,
            float[] cos_table, float[] sin_table, float[] out_f32,
            int dim, int num_heads, int kv_heads, int pos, int max_seq) {

        final int head_dim = dim / num_heads;
        final int kv_dim = kv_heads * head_dim;
        final float rsqrt_d = 1.0f / (float) Math.sqrt(head_dim);

        Scratchpad sp = SCRATCH.get();

        // 1. Quantize input
        float x_absmax = KernelsInt8.absmax(x_f32, dim);
        float x_scale = x_absmax / 127.0f;
        byte[] x_q8 = sp.x_q8;
        KernelsInt8.quantize_i8(x_f32, x_q8, dim, x_scale);

        byte[] q_new_q8 = sp.q_new_q8;
        byte[] k_new_q8 = sp.k_new_q8;
        byte[] v_new_q8 = sp.v_new_q8;
        float[] q_scale = sp.q_scale;
        float[] k_scale = sp.k_scale;
        float[] v_scale = sp.v_scale;

        // 2. QKV proj INT8 with manual matrix batching loops
        matmul_q8_1xN(x_q8, x_scale, wq_q8, wq_scale, q_new_q8, q_scale, num_heads * head_dim, dim);
        matmul_q8_1xN(x_q8, x_scale, wk_q8, wk_scale, k_new_q8, k_scale, kv_dim, dim);
        matmul_q8_1xN(x_q8, x_scale, wv_q8, wv_scale, v_new_q8, v_scale, kv_dim, dim);

        // 3. Dequant Q,K,V to FP32 for RoPE + attention
        float[] q_f32 = sp.q_f32;
        float[] k_f32 = sp.k_f32;
        float[] v_f32 = sp.v_f32;
        for (int i = 0; i < num_heads * head_dim; i++) q_f32[i] = q_new_q8[i] * q_scale[i];
        for (int i = 0; i < kv_dim; i++) {
            k_f32[i] = k_new_q8[i] * k_scale[i];
            v_f32[i] = v_new_q8[i] * v_scale[i];
        }

        // 4. FP32 RoPE - use split layout version
        KernelsFloat.apply_rope_f32_split(
            q_f32, num_heads, k_f32, kv_heads,
            head_dim, pos, cos_table, sin_table
        );

        // 5. Requantize K,V for cache with single scale per head (zero-allocation slicing)
        for (int h = 0; h < kv_heads; h++) {
            int hOff = h * head_dim;
            float k_h_absmax = absmax_slice(k_f32, hOff, head_dim);
            float v_h_absmax = absmax_slice(v_f32, hOff, head_dim);
            float k_h_scale = k_h_absmax / 127.0f;
            float v_h_scale = v_h_absmax / 127.0f;

            quantize_i8_slice(k_f32, hOff, k_new_q8, hOff, head_dim, k_h_scale);
            quantize_i8_slice(v_f32, hOff, v_new_q8, hOff, head_dim, v_h_scale);

            System.arraycopy(k_new_q8, hOff, k_cache_q8, pos * kv_dim + hOff, head_dim);
            System.arraycopy(v_new_q8, hOff, v_cache_q8, pos * kv_dim + hOff, head_dim);
            k_scale_cache[pos * kv_heads + h] = k_h_scale;
            v_scale_cache[pos * kv_heads + h] = v_h_scale;
        }

        // 6. Attention in FP32 - dequant K,V from cache on the fly
        float[] attn_scores = sp.attn_scores;
        float[] k_temp = sp.k_temp;

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / (num_heads / kv_heads);
            int qOff = h * head_dim;

            for (int j = 0; j <= pos; j++) {
                // Dequant K from cache inside an unrolled loop
                int kCacheOff = j * kv_dim + kv_h * head_dim;
                float k_h_scale = k_scale_cache[j * kv_heads + kv_h];
                
                int d = 0;
                for (; d <= head_dim - 8; d += 8) {
                    k_temp[d]     = k_cache_q8[kCacheOff + d] * k_h_scale;
                    k_temp[d + 1] = k_cache_q8[kCacheOff + d + 1] * k_h_scale;
                    k_temp[d + 2] = k_cache_q8[kCacheOff + d + 2] * k_h_scale;
                    k_temp[d + 3] = k_cache_q8[kCacheOff + d + 3] * k_h_scale;
                    k_temp[d + 4] = k_cache_q8[kCacheOff + d + 4] * k_h_scale;
                    k_temp[d + 5] = k_cache_q8[kCacheOff + d + 5] * k_h_scale;
                    k_temp[d + 6] = k_cache_q8[kCacheOff + d + 6] * k_h_scale;
                    k_temp[d + 7] = k_cache_q8[kCacheOff + d + 7] * k_h_scale;
                }
                for (; d < head_dim; d++) {
                    k_temp[d] = k_cache_q8[kCacheOff + d] * k_h_scale;
                }

                // Unrolled dot product evaluation with 8 parallel accumulators for maximum ILP
                float dot = 0.0f;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
                d = 0;
                for (; d <= head_dim - 8; d += 8) {
                    acc0 += q_f32[qOff + d]     * k_temp[d];
                    acc1 += q_f32[qOff + d + 1] * k_temp[d + 1];
                    acc2 += q_f32[qOff + d + 2] * k_temp[d + 2];
                    acc3 += q_f32[qOff + d + 3] * k_temp[d + 3];
                    acc4 += q_f32[qOff + d + 4] * k_temp[d + 4];
                    acc5 += q_f32[qOff + d + 5] * k_temp[d + 5];
                    acc6 += q_f32[qOff + d + 6] * k_temp[d + 6];
                    acc7 += q_f32[qOff + d + 7] * k_temp[d + 7];
                }
                dot = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
                for (; d < head_dim; d++) {
                    dot += q_f32[qOff + d] * k_temp[d];
                }

                attn_scores[j] = dot * rsqrt_d;
            }

            softmax_row_f32(attn_scores, 0, pos + 1);

            int outOff = h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int j = 0; j <= pos; j++) {
                    int vCacheOff = j * kv_dim + kv_h * head_dim + d;
                    float v_val = v_cache_q8[vCacheOff] * v_scale_cache[j * kv_heads + kv_h];
                    acc += attn_scores[j] * v_val;
                }
                out_f32[outOff + d] = acc;
            }
        }

        // 7. O proj FP32
        float[] temp_clone = sp.temp_clone;
        System.arraycopy(out_f32, 0, temp_clone, 0, dim);
        KernelsFloat.matmul_bias_f32(out_f32, temp_clone, wo, null, 1, dim, dim);
    }

    /**
     * INT8 SwiGLU: out = silu(x @ W_gate_q8) * (x @ W_up_q8)
     */
    private static void matmul_swiglu_q8(
            float[] x_f32, int M, int K,
            byte[] w_gate_q8, float[] w_gate_scale,
            byte[] w_up_q8, float[] w_up_scale,
            float[] out_f32, int H) {

        Scratchpad sp = SCRATCH.get();
        float x_absmax = KernelsInt8.absmax(x_f32, K);
        float x_scale = x_absmax / 127.0f;
        byte[] x_q8 = sp.x_q8;
        KernelsInt8.quantize_i8(x_f32, x_q8, K, x_scale);

        byte[] gate_q8 = sp.gate_q8;
        byte[] up_q8 = sp.up_q8;
        float[] gate_scale = sp.gate_scale;
        float[] up_scale = sp.up_scale;

        matmul_q8_1xN(x_q8, x_scale, w_gate_q8, w_gate_scale, gate_q8, gate_scale, H, K);
        matmul_q8_1xN(x_q8, x_scale, w_up_q8, w_up_scale, up_q8, up_scale, H, K);

        // Dequant + SiLU activation + Multiplication batched in 4-way registers
        int h = 0;
        for (; h <= H - 4; h += 4) {
            float g0 = gate_q8[h] * gate_scale[h];
            float g1 = gate_q8[h + 1] * gate_scale[h + 1];
            float g2 = gate_q8[h + 2] * gate_scale[h + 2];
            float g3 = gate_q8[h + 3] * gate_scale[h + 3];

            float u0 = up_q8[h] * up_scale[h];
            float u1 = up_q8[h + 1] * up_scale[h + 1];
            float u2 = up_q8[h + 2] * up_scale[h + 2];
            float u3 = up_q8[h + 3] * up_scale[h + 3];

            float sig0 = 1.0f / (1.0f + (float) Math.exp(-Math.max(g0, -88f)));
            float sig1 = 1.0f / (1.0f + (float) Math.exp(-Math.max(g1, -88f)));
            float sig2 = 1.0f / (1.0f + (float) Math.exp(-Math.max(g2, -88f)));
            float sig3 = 1.0f / (1.0f + (float) Math.exp(-Math.max(g3, -88f)));

            out_f32[h]     = g0 * sig0 * u0;
            out_f32[h + 1] = g1 * sig1 * u1;
            out_f32[h + 2] = g2 * sig2 * u2;
            out_f32[h + 3] = g3 * sig3 * u3;
        }
        for (; h < H; h++) {
            float gate = gate_q8[h] * gate_scale[h];
            float up = up_q8[h] * up_scale[h];
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-Math.max(gate, -88f)));
            out_f32[h] = gate * sigmoid * up;
        }
    }

    /**
     * INT8 Down proj matrix-vector multiplication with manual horizontal batch tiling (4x unrolled)
     */
    private static void matmul_down_q8(
            float[] a_f32, int M, int N,
            byte[] b_q8, float[] b_scale,
            float[] c_f32, int K) {

        Scratchpad sp = SCRATCH.get();
        float a_absmax = KernelsInt8.absmax(a_f32, N);
        float a_scale = a_absmax / 127.0f;
        byte[] a_q8 = sp.a_q8;
        KernelsInt8.quantize_i8(a_f32, a_q8, N, a_scale);

        int k = 0;
        for (; k <= K - 4; k += 4) {
            int acc0 = 0;
            int acc1 = 0;
            int acc2 = 0;
            int acc3 = 0;
            for (int n = 0; n < N; n++) {
                byte av = a_q8[n];
                int bIdx = n * K + k;
                acc0 += av * b_q8[bIdx];
                acc1 += av * b_q8[bIdx + 1];
                acc2 += av * b_q8[bIdx + 2];
                acc3 += av * b_q8[bIdx + 3];
            }
            c_f32[k]     = acc0 * a_scale * b_scale[k];
            c_f32[k + 1] = acc1 * a_scale * b_scale[k + 1];
            c_f32[k + 2] = acc2 * a_scale * b_scale[k + 2];
            c_f32[k + 3] = acc3 * a_scale * b_scale[k + 3];
        }
        for (; k < K; k++) {
            int acc = 0;
            for (int n = 0; n < N; n++) {
                acc += a_q8[n] * b_q8[n * K + k];
            }
            c_f32[k] = acc * a_scale * b_scale[k];
        }
    }

    /**
     * INT8 matmul helper: 1xK @ KxN -> 1xN 
     * Batches contiguous cache lines horizontally to guarantee perfect sequential streaming layouts.
     */
    private static void matmul_q8_1xN(
            byte[] a_q8, float a_scale,
            byte[] b_q8, float[] b_scale,
            byte[] c_q8, float[] c_scale,
            int N, int K) {

        int n = 0;
        for (; n <= N - 4; n += 4) {
            int acc0 = 0;
            int acc1 = 0;
            int acc2 = 0;
            int acc3 = 0;
            for (int i = 0; i < K; i++) {
                byte av = a_q8[i];
                int bIdx = i * N + n;
                acc0 += av * b_q8[bIdx];
                acc1 += av * b_q8[bIdx + 1];
                acc2 += av * b_q8[bIdx + 2];
                acc3 += av * b_q8[bIdx + 3];
            }
            
            // Channel 0
            {
                float c_val = acc0 * a_scale * b_scale[n];
                c_scale[n] = Math.abs(c_val) / 127.0f;
                c_q8[n] = c_scale[n] == 0.0f ? 0 : (byte) Math.max(-127, Math.min(127, Math.round(c_val / c_scale[n])));
            }
            // Channel 1
            {
                float c_val = acc1 * a_scale * b_scale[n + 1];
                c_scale[n + 1] = Math.abs(c_val) / 127.0f;
                c_q8[n + 1] = c_scale[n + 1] == 0.0f ? 0 : (byte) Math.max(-127, Math.min(127, Math.round(c_val / c_scale[n + 1])));
            }
            // Channel 2
            {
                float c_val = acc2 * a_scale * b_scale[n + 2];
                c_scale[n + 2] = Math.abs(c_val) / 127.0f;
                c_q8[n + 2] = c_scale[n + 2] == 0.0f ? 0 : (byte) Math.max(-127, Math.min(127, Math.round(c_val / c_scale[n + 2])));
            }
            // Channel 3
            {
                float c_val = acc3 * a_scale * b_scale[n + 3];
                c_scale[n + 3] = Math.abs(c_val) / 127.0f;
                c_q8[n + 3] = c_scale[n + 3] == 0.0f ? 0 : (byte) Math.max(-127, Math.min(127, Math.round(c_val / c_scale[n + 3])));
            }
        }
        for (; n < N; n++) {
            int acc = 0;
            for (int i = 0; i < K; i++) {
                acc += a_q8[i] * b_q8[i * N + n];
            }
            float c_val = acc * a_scale * b_scale[n];
            c_scale[n] = Math.abs(c_val) / 127.0f;
            c_q8[n] = c_scale[n] == 0.0f ? 0 : (byte) Math.max(-127, Math.min(127, Math.round(c_val / c_scale[n])));
        }
    }

    private static float absmax_slice(float[] arr, int offset, int len) {
        float max = 0.0f;
        for (int i = 0; i < len; i++) {
            float val = Math.abs(arr[offset + i]);
            if (val > max) max = val;
        }
        return max;
    }

    private static void quantize_i8_slice(float[] src, int srcOff, byte[] dst, int dstOff, int len, float scale) {
        float invScale = scale == 0.0f ? 0.0f : 1.0f / scale;
        for (int i = 0; i < len; i++) {
            float val = src[srcOff + i] * invScale;
            dst[dstOff + i] = (byte) Math.max(-127, Math.min(127, Math.round(val)));
        }
    }

    private static float avg(float[] arr, int len) {
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += arr[i];
        }
        return sum / len;
    }

    private static void softmax_row_f32(float[] arr, int offset, int len) {
        float max = -Float.MAX_VALUE;
        for (int i = 0; i < len; i++) {
            max = Math.max(max, arr[offset + i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            arr[offset + i] = (float) Math.exp(Math.max(arr[offset + i] - max, -88f));
            sum += arr[offset + i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < len; i++) {
            arr[offset + i] *= invSum;
        }
    }
}