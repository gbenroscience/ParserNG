package com.github.gbenroscience.simd.turbo.tools;

import jdk.incubator.vector.*;
import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.*;

/**
 *
 * @author GBEMIRO
 */
public final class LlamaLayerInt8 {


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
     * Run one INT8 decoder layer for a single token x_f32: [dim] - input FP32,
     * modified in-place
     */
    public static void forward_layer_q8(
            float[] x_f32, // [dim] - in/out FP32
            WeightsQ8 w,
            StateQ8 s,
            Config cfg) {

        final int dim = cfg.dim;
        final int hidden = cfg.hidden_dim;
        float[] temp_f32 = new float[dim];

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
        for (int i = 0; i < dim; i += VF_LEN) {
            FloatVector xv = FloatVector.fromArray(F_SPECIES, x_f32, i);
            FloatVector tv = FloatVector.fromArray(F_SPECIES, temp_f32, i);
            xv.add(tv).intoArray(x_f32, i);
        }

        // === 2. FFN block ===
        // x = x + ffn(rms_norm(x))
        // 2a. RMSNorm FP32
        KernelsFloat.rms_norm_fast(x_f32, w.ffn_norm_gamma, temp_f32, 1, dim, cfg.norm_eps);

        // 2b. INT8 SwiGLU: gate = silu(x @ W_gate_q8) * (x @ W_up_q8)
        float[] swiglu_out_f32 = new float[hidden];
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
        for (int i = 0; i < dim; i += VF_LEN) {
            FloatVector xv = FloatVector.fromArray(F_SPECIES, x_f32, i);
            FloatVector tv = FloatVector.fromArray(F_SPECIES, temp_f32, i);
            xv.add(tv).intoArray(x_f32, i);
        }
    }

    /**
     * INT8 MHA decode step - from previous message, integrated here
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

    // 1. Quantize input
    float x_absmax = KernelsInt8.absmax(x_f32, dim);
    float x_scale = x_absmax / 127.0f;
    byte[] x_q8 = new byte[dim];
    KernelsInt8.quantize_i8(x_f32, x_q8, dim, x_scale);

    byte[] q_new_q8 = new byte[num_heads * head_dim];
    byte[] k_new_q8 = new byte[kv_dim];
    byte[] v_new_q8 = new byte[kv_dim];
    float[] q_scale = new float[num_heads * head_dim];
    float[] k_scale = new float[kv_dim];
    float[] v_scale = new float[kv_dim];

    // 2. QKV proj INT8
    matmul_q8_1xN(x_q8, x_scale, wq_q8, wq_scale, q_new_q8, q_scale, num_heads * head_dim, dim);
    matmul_q8_1xN(x_q8, x_scale, wk_q8, wk_scale, k_new_q8, k_scale, kv_dim, dim);
    matmul_q8_1xN(x_q8, x_scale, wv_q8, wv_scale, v_new_q8, v_scale, kv_dim, dim);

    // 3. Dequant Q,K,V to FP32 for RoPE + attention
    float[] q_f32 = new float[num_heads * head_dim];
    float[] k_f32 = new float[kv_dim];
    float[] v_f32 = new float[kv_dim];
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

    // 5. Requantize K,V for cache with single scale per head
    for (int h = 0; h < kv_heads; h++) {
        int hOff = h * head_dim;
        float k_h_absmax = KernelsInt8.absmax(k_f32, head_dim);
        float v_h_absmax = KernelsInt8.absmax(v_f32, head_dim);
        float k_h_scale = k_h_absmax / 127.0f;
        float v_h_scale = v_h_absmax / 127.0f;

        KernelsInt8.quantize_i8(k_f32, k_new_q8, head_dim, k_h_scale);
        KernelsInt8.quantize_i8(v_f32, v_new_q8, head_dim, v_h_scale);

        System.arraycopy(k_new_q8, 0, k_cache_q8, pos * kv_dim + hOff, head_dim);
        System.arraycopy(v_new_q8, 0, v_cache_q8, pos * kv_dim + hOff, head_dim);
        k_scale_cache[pos * kv_heads + h] = k_h_scale;
        v_scale_cache[pos * kv_heads + h] = v_h_scale;
    }

    // 6. Attention in FP32 - dequant K,V from cache on the fly
    float[] attn_scores = new float[pos + 1];
    float[] k_temp = new float[head_dim];
    float[] v_temp = new float[head_dim];

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / (num_heads / kv_heads);
        int qOff = h * head_dim;

        for (int j = 0; j <= pos; j++) {
            // Dequant K from cache
            int kCacheOff = j * kv_dim + kv_h * head_dim;
            float k_h_scale = k_scale_cache[j * kv_heads + kv_h];
            for (int d = 0; d < head_dim; d++) {
                k_temp[d] = k_cache_q8[kCacheOff + d] * k_h_scale;
            }

            float dot = 0.0f;
            int d = 0;
            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (; d < F_SPECIES.loopBound(head_dim); d += VF_LEN) {
                FloatVector qv = FloatVector.fromArray(F_SPECIES, q_f32, qOff + d);
                FloatVector kv = FloatVector.fromArray(F_SPECIES, k_temp, d);
                acc = acc.fma(qv, kv);
            }
            dot = acc.reduceLanes(VectorOperators.ADD);
            for (; d < head_dim; d++) dot += q_f32[qOff + d] * k_temp[d];

            attn_scores[j] = dot * rsqrt_d;
        }

        softmax_row_f32(attn_scores, 0, pos + 1);

        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int j = 0; j <= pos; j++) {
                int vCacheOff = j * kv_dim + kv_h * head_dim + d;
                float v_val = v_cache_q8[vCacheOff] * v_scale_cache[j * kv_heads + kv_h];
                acc += attn_scores[j] * v_val;
            }
            out_f32[qOff + d] = acc;
        }
    }

    // 7. O proj FP32
    float[] temp = out_f32.clone();
    KernelsFloat.matmul_bias_f32(out_f32, temp, wo, null, 1, dim, dim);
}
    /**
     * INT8 SwiGLU: out = silu(x @ W_gate_q8) * (x @ W_up_q8)
     */
    private static void matmul_swiglu_q8(
            float[] x_f32, int M, int K,
            byte[] w_gate_q8, float[] w_gate_scale,
            byte[] w_up_q8, float[] w_up_scale,
            float[] out_f32, int H) {

        float x_absmax = KernelsInt8.absmax(x_f32, K);
        float x_scale = x_absmax / 127.0f;
        byte[] x_q8 = new byte[K];
        KernelsInt8.quantize_i8(x_f32, x_q8, K, x_scale);

        byte[] gate_q8 = new byte[H];
        byte[] up_q8 = new byte[H];
        float[] gate_scale = new float[H];
        float[] up_scale = new float[H];

        matmul_q8_1xN(x_q8, x_scale, w_gate_q8, w_gate_scale, gate_q8, gate_scale, H, K);
        matmul_q8_1xN(x_q8, x_scale, w_up_q8, w_up_scale, up_q8, up_scale, H, K);

        // Dequant + SiLU + mul in FP32
        for (int h = 0; h < H; h++) {
            float gate = gate_q8[h] * gate_scale[h];
            float up = up_q8[h] * up_scale[h];
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-Math.max(gate, -88f)));
            out_f32[h] = gate * sigmoid * up;
        }
    }

    /**
     * INT8 Down proj: C_f32 = A_f32 @ B_q8
     */
    private static void matmul_down_q8(
            float[] a_f32, int M, int N,
            byte[] b_q8, float[] b_scale,
            float[] c_f32, int K) {

        float a_absmax = KernelsInt8.absmax(a_f32, N);
        float a_scale = a_absmax / 127.0f;
        byte[] a_q8 = new byte[N];
        KernelsInt8.quantize_i8(a_f32, a_q8, N, a_scale);

        for (int k = 0; k < K; k++) {
            int acc = 0;
            for (int n = 0; n < N; n++) {
                acc += a_q8[n] * b_q8[n * K + k];
            }
            c_f32[k] = acc * a_scale * b_scale[k];
        }
    }

      // INT8 matmul helper: 1xK @ KxN -> 1xN
    private static void matmul_q8_1xN(
            byte[] a_q8, float a_scale,
            byte[] b_q8, float[] b_scale,
            byte[] c_q8, float[] c_scale,
            int N, int K) {

        for (int n = 0; n < N; n++) {
            int acc = 0;
            int i = 0;
            IntVector vAcc = IntVector.zero(I_SPECIES);

            // Vectorized dot: i8 -> i16 -> i32
            for (; i < B_SPECIES.loopBound(K); i += B_SPECIES.length()) {
                ByteVector av = ByteVector.fromArray(B_SPECIES, a_q8, i);
                ByteVector bv = ByteVector.fromArray(B_SPECIES, b_q8, i * N + n);

                // Step 1: i8 -> i16. Need 2 ShortVectors per ByteVector
                ShortVector av_s0 = (ShortVector) av.convertShape(VectorOperators.B2S, S_SPECIES, 0);
                ShortVector av_s1 = (ShortVector) av.convertShape(VectorOperators.B2S, S_SPECIES, 1);
                ShortVector bv_s0 = (ShortVector) bv.convertShape(VectorOperators.B2S, S_SPECIES, 0);
                ShortVector bv_s1 = (ShortVector) bv.convertShape(VectorOperators.B2S, S_SPECIES, 1);

                // Step 2: i16 -> i32. Need 2 IntVectors per ShortVector
                IntVector av_i0 = (IntVector) av_s0.convertShape(VectorOperators.S2I, I_SPECIES, 0);
                IntVector av_i1 = (IntVector) av_s0.convertShape(VectorOperators.S2I, I_SPECIES, 1);
                IntVector bv_i0 = (IntVector) bv_s0.convertShape(VectorOperators.S2I, I_SPECIES, 0);
                IntVector bv_i1 = (IntVector) bv_s0.convertShape(VectorOperators.S2I, I_SPECIES, 1);

                vAcc = vAcc.add(av_i0.mul(bv_i0));
                vAcc = vAcc.add(av_i1.mul(bv_i1));

                // Handle upper half if B_SPECIES.length() * 2 == S_SPECIES.length()
                if (S_SPECIES.length() == B_SPECIES.length() * 2) {
                    IntVector av_i2 = (IntVector) av_s1.convertShape(VectorOperators.S2I, I_SPECIES, 0);
                    IntVector av_i3 = (IntVector) av_s1.convertShape(VectorOperators.S2I, I_SPECIES, 1);
                    IntVector bv_i2 = (IntVector) bv_s1.convertShape(VectorOperators.S2I, I_SPECIES, 0);
                    IntVector bv_i3 = (IntVector) bv_s1.convertShape(VectorOperators.S2I, I_SPECIES, 1);
                    vAcc = vAcc.add(av_i2.mul(bv_i2));
                    vAcc = vAcc.add(av_i3.mul(bv_i3));
                }
            }
            acc = vAcc.reduceLanes(VectorOperators.ADD);
            for (; i < K; i++) {
                acc += a_q8[i] * b_q8[i * N + n];
            }

            float c_val = acc * a_scale * b_scale[n];
            c_scale[n] = Math.abs(c_val) / 127.0f;
            // Avoid divide-by-zero if c_val == 0
            c_q8[n] = c_scale[n] == 0 ? 0 : (byte) Math.max(-127, Math.min(127, Math.round(c_val / c_scale[n])));
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

    /*
    END TO END USAGE
    // 1. Load INT8 weights once
Config cfg = new Config();
WeightsQ8[] layers = loadWeightsQ8("llama2-7b-q8_0.bin", 32);
float[] final_norm = loadFinalNorm(); // FP32
float[] lm_head = loadLmHead(); // FP32 [dim, vocab]

// 2. Init INT8 cache per layer
StateQ8[] states = new StateQ8[32];
for (int i = 0; i < 32; i++) states[i] = new StateQ8(cfg);

// 3. Token loop
float[] embed_table = loadEmbeddings(); // FP32 [vocab, dim]
float[] x = new float[cfg.dim];
int[] prompt = tokenize("The meaning of life is");

for (int t = 0; t < prompt.length + 50; t++) {
    int tok = t < prompt.length? prompt[t] : next_tok;
    System.arraycopy(embed_table, tok * cfg.dim, x, 0, cfg.dim);

    // Run all layers
    for (int l = 0; l < 32; l++) {
        LlamaLayerInt8.forward_layer_q8(x, layers[l], states[l], cfg);
    }

    // Final norm + lm_head
    float[] normed = new float[cfg.dim];
    KernelsFloat.rms_norm_fast(x, final_norm, normed, 1, cfg.dim, cfg.norm_eps);

    float[] logits = new float[32000];
    KernelsFloat.matmul_bias_f32(logits, normed, lm_head, null, 1, 32000, cfg.dim);

    int next_tok = argmax(logits);
    if (t >= prompt.length) System.out.print(detokenize(next_tok) + " ");

    for (StateQ8 s : states) s.pos++;
}
     */
}
