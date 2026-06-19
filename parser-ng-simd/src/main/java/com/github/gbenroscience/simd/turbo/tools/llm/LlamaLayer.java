package com.github.gbenroscience.simd.turbo.tools.llm;

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;
import com.github.gbenroscience.simd.turbo.tools.matrix.KernelsFloat;

/**
 * Highly optimized Llama inference layer engineered with strict mechanical
 * sympathy. Eliminates garbage collection allocation pressure and maximizes JIT
 * auto-vectorization.
 *
 * * @author GBEMIRO
 */
public final class LlamaLayer {

    // ThreadLocal scratchpad to eliminate per-token transient array allocations in generate_token
    private static final ThreadLocal<Scratchpad> SCRATCH = ThreadLocal.withInitial(Scratchpad::new);

    private static final class Scratchpad {

        float[] x = new float[4096];
        float[] normed = new float[4096];
        float[] logits = new float[32000];

        void ensureCapacity(int dim, int vocabSize) {
            if (x.length < dim) {
                x = new float[dim];
            }
            if (normed.length < dim) {
                normed = new float[dim];
            }
            if (logits.length < vocabSize) {
                logits = new float[vocabSize];
            }
        }
    }

    // Model config
    public static final class Config {

        public int n_layers = 32;       // Added: standard Llama-2-7B / Llama-3-8B fallback
        public int dim = 4096;
        public int hidden_dim = 11008; // FFN hidden
        public int num_heads = 32;
        public int kv_heads = 32;      // GQA: set < num_heads for Llama2-70B
        public int head_dim = 128;     // dim / num_heads
        public int max_seq = 4096;
        public float norm_eps = 1e-6f;
        public float rope_theta = 10000.0f;

        /**
         * Dynamically instantiates a Config class from raw GGUF metadata. This
         * avoids magic numbers and auto-adapts to variations (e.g., Mistral,
         * Llama-3).
         */
        public static Config fromGGUF(GGUFLoader.GGUFFile gguf) {
            Config cfg = new Config();
            var meta = gguf.metadata; // Assuming your GGUFLoader exposes a map or lookup for KV metadata

            cfg.n_layers = getIntMeta(meta, "llama.block_count", 32);
            cfg.dim = getIntMeta(meta, "llama.embedding_length", 4096);
            cfg.hidden_dim = getIntMeta(meta, "llama.feed_forward_length", 11008);
            cfg.num_heads = getIntMeta(meta, "llama.attention.head_count", 32);
            cfg.kv_heads = getIntMeta(meta, "llama.attention.head_count_kv", cfg.num_heads);
            cfg.max_seq = getIntMeta(meta, "llama.context_length", 4096);

            cfg.head_dim = cfg.dim / cfg.num_heads;
            cfg.norm_eps = getFloatMeta(meta, "llama.attention.layer_norm_rms_epsilon", 1e-6f);
            cfg.rope_theta = getFloatMeta(meta, "llama.rope.freq_base", 10000.0f);

            return cfg;
        }

        private static int getIntMeta(java.util.Map<String, Object> map, String key, int fallback) {
            Object val = map.get(key);
            return (val instanceof Number) ? ((Number) val).intValue() : fallback;
        }

        private static float getFloatMeta(java.util.Map<String, Object> map, String key, float fallback) {
            Object val = map.get(key);
            return (val instanceof Number) ? ((Number) val).floatValue() : fallback;
        }
    }

    // Weights for one layer
    public static final class Weights {

        public float[] wq, wk, wv, wo; // [dim, dim] or [dim, kv_dim]
        public float[] attn_norm_gamma; // [dim]
        public float[] w_gate, w_up, w_down; // [dim, hidden], [dim, hidden], [hidden, dim]
        public float[] ffn_norm_gamma; // [dim]

        public Weights(float[] wq, float[] wk, float[] wv, float[] wo) {
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
        }

        public Weights(float[] wq, float[] wk, float[] wv, float[] wo, float[] attn_norm_gamma, float[] w_gate, float[] w_up, float[] w_down, float[] ffn_norm_gamma) {
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
            this.attn_norm_gamma = attn_norm_gamma;
            this.w_gate = w_gate;
            this.w_up = w_up;
            this.w_down = w_down;
            this.ffn_norm_gamma = ffn_norm_gamma;
        }

    }

    // Per-sequence state
    public static final class State {

        public float[] k_cache, v_cache; // [max_seq, kv_heads, head_dim]
        public float[] cos_table, sin_table; // [max_seq, head_dim/2]
        public int pos = 0; // current position in sequence

        // Mechanically sympathetic scratch buffers to eliminate per-layer transient allocations
        public final float[] temp;
        public final float[] temp_hidden;

        public State(Config cfg) {
            int kv_dim = cfg.kv_heads * cfg.head_dim;
            k_cache = new float[cfg.max_seq * kv_dim];
            v_cache = new float[cfg.max_seq * kv_dim];
            cos_table = new float[cfg.max_seq * cfg.head_dim / 2];
            sin_table = new float[cfg.max_seq * cfg.head_dim / 2];
            temp = new float[cfg.dim];
            temp_hidden = new float[cfg.hidden_dim];
            KernelsFloat.precompute_rope(cfg.max_seq, cfg.head_dim, cfg.rope_theta, cos_table, sin_table);
        }
    }

    /**
     * Run one decoder layer for a single token x: [dim] - input token
     * embedding, modified in-place
     */
    public static void forward_layer(
            float[] x, // [dim] - in/out
            Weights w,
            State s,
            Config cfg) {

        final int dim = cfg.dim;
        final int hidden = cfg.hidden_dim;

        // Reusing pre-allocated layer state arrays to completely avoid GC load during inference
        float[] temp = s.temp;
        float[] temp_hidden = s.temp_hidden;

        // === 1. Attention block ===
        // x = x + attn(rms_norm(x))
        // 1a. RMSNorm
        KernelsFloat.rms_norm(x, w.attn_norm_gamma, temp, 1, dim, cfg.norm_eps);

        // 1b. QKV + RoPE + Attention + O_proj, writes to temp
        KernelsFloat.mha_decode_step(
                temp, // input after norm
                w.wq, w.wk, w.wv, w.wo,
                s.k_cache, s.v_cache,
                s.cos_table, s.sin_table,
                temp, // output overwrites temp
                dim, cfg.num_heads, cfg.kv_heads, s.pos, cfg.max_seq
        );

        // 1c. Residual: x += attn_out
        // Unrolled by 8 to facilitate maximum Instruction-Level Parallelism (ILP) and JIT SIMD generation
        int i = 0;
        for (; i <= dim - 8; i += 8) {
            x[i] += temp[i];
            x[i + 1] += temp[i + 1];
            x[i + 2] += temp[i + 2];
            x[i + 3] += temp[i + 3];
            x[i + 4] += temp[i + 4];
            x[i + 5] += temp[i + 5];
            x[i + 6] += temp[i + 6];
            x[i + 7] += temp[i + 7];
        }
        for (; i < dim; i++) {
            x[i] += temp[i];
        }

        // === 2. FFN block ===
        // x = x + ffn(rms_norm(x))
        // 2a. RMSNorm
        KernelsFloat.rms_norm(x, w.ffn_norm_gamma, temp, 1, dim, cfg.norm_eps);

        // 2b. SwiGLU: gate = silu(x @ W_gate) * (x @ W_up)
        KernelsFloat.matmul_swiglu(
                temp, 1, dim, // x[1,dim]
                w.w_gate, w.w_up, // W_gate[dim,hidden], W_up[dim,hidden]
                temp_hidden, hidden // out[1,hidden]
        );

        // 2c. Down proj: ffn_out = gate * W_down
        KernelsFloat.matmul_down(
                temp_hidden, 1, hidden, // [1,hidden]
                w.w_down, hidden, dim, // [hidden,dim]
                temp // [1,dim]
        );

        // 2d. Residual: x += ffn_out
        i = 0;
        for (; i <= dim - 8; i += 8) {
            x[i] += temp[i];
            x[i + 1] += temp[i + 1];
            x[i + 2] += temp[i + 2];
            x[i + 3] += temp[i + 3];
            x[i + 4] += temp[i + 4];
            x[i + 5] += temp[i + 5];
            x[i + 6] += temp[i + 6];
            x[i + 7] += temp[i + 7];
        }
        for (; i < dim; i++) {
            x[i] += temp[i];
        }
    }

    /**
     * Run full model: embed -> N layers -> norm -> lm_head
     */
    public static int generate_token(
            float[] token_embedding, // [dim] - current token
            Weights[] layers, // array of layer weights
            float[] final_norm_gamma, // [dim]
            float[] lm_head, // [dim, vocab_size]
            State[] states, // per-layer KV cache
            Config cfg,
            int vocab_size) {

        Scratchpad sp = SCRATCH.get();
        sp.ensureCapacity(cfg.dim, vocab_size);

        float[] x = sp.x;
        System.arraycopy(token_embedding, 0, x, 0, cfg.dim);

        // Run all decoder layers
        for (int l = 0; l < layers.length; l++) {
            forward_layer(x, layers[l], states[l], cfg);
        }

        // Final norm
        float[] normed = sp.normed;
        KernelsFloat.rms_norm(x, final_norm_gamma, normed, 1, cfg.dim, cfg.norm_eps);

        // LM head: logits = normed @ lm_head
        float[] logits = sp.logits;
        KernelsFloat.matmul_down(normed, 1, cfg.dim, lm_head, cfg.dim, vocab_size, logits);

        // Greedy decode: argmax with manual 4x loop unrolling to optimize pipeline efficiency
        int max_idx = 0;
        float max_val = logits[0];

        int i = 1;
        for (; i <= vocab_size - 4; i += 4) {
            float v0 = logits[i];
            float v1 = logits[i + 1];
            float v2 = logits[i + 2];
            float v3 = logits[i + 3];

            if (v0 > max_val) {
                max_val = v0;
                max_idx = i;
            }
            if (v1 > max_val) {
                max_val = v1;
                max_idx = i + 1;
            }
            if (v2 > max_val) {
                max_val = v2;
                max_idx = i + 2;
            }
            if (v3 > max_val) {
                max_val = v3;
                max_idx = i + 3;
            }
        }
        for (; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }

        // Advance all layer positions
        for (int l = 0; l < states.length; l++) {
            states[l].pos++;
        }

        return max_idx;
    }
}
