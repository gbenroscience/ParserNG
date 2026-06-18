package com.github.gbenroscience.parser.pro.turbo.tools;

import jdk.incubator.vector.DoubleVector;

/**
 *
 * @author GBEMIRO
 */
public final class LlamaLayer {

    // Model config
    static final class Config {
        int dim = 4096;
        int hidden_dim = 11008; // FFN hidden
        int num_heads = 32;
        int kv_heads = 32; // GQA: set < num_heads for Llama2-70B
        int head_dim = 128; // dim / num_heads
        int max_seq = 4096;
        double norm_eps = 1e-6;
        double rope_theta = 10000.0;
    }

    // Weights for one layer
    static final class Weights {
        double[] wq, wk, wv, wo; // [dim, dim] or [dim, kv_dim]
        double[] attn_norm_gamma; // [dim]
        double[] w_gate, w_up, w_down; // [dim, hidden], [dim, hidden], [hidden, dim]
        double[] ffn_norm_gamma; // [dim]
    }

    // Per-sequence state
    static final class State {
        double[] k_cache, v_cache; // [max_seq, kv_heads, head_dim]
        double[] cos_table, sin_table; // [max_seq, head_dim/2]
        int pos = 0; // current position in sequence

        State(Config cfg) {
            int kv_dim = cfg.kv_heads * cfg.head_dim;
            k_cache = new double[cfg.max_seq * kv_dim];
            v_cache = new double[cfg.max_seq * kv_dim];
            cos_table = new double[cfg.max_seq * cfg.head_dim / 2];
            sin_table = new double[cfg.max_seq * cfg.head_dim / 2];
            Kernels.precompute_rope(cfg.max_seq, cfg.head_dim, cfg.rope_theta, cos_table, sin_table);
        }
    }

    /**
     * Run one decoder layer for a single token
     * x: [dim] - input token embedding, modified in-place
     */
    public static void forward_layer(
            double[] x, // [dim] - in/out
            Weights w,
            State s,
            Config cfg) {

        final int dim = cfg.dim;
        final int hidden = cfg.hidden_dim;
        double[] temp = new double[dim];
        double[] temp_hidden = new double[hidden];

        // === 1. Attention block ===
        // x = x + attn(rms_norm(x))

        // 1a. RMSNorm
        Kernels.rms_norm(x, w.attn_norm_gamma, temp, 1, dim, cfg.norm_eps);

        // 1b. QKV + RoPE + Attention + O_proj, writes to temp
        Kernels.mha_decode_step(
            temp, // input after norm
            w.wq, w.wk, w.wv, w.wo,
            s.k_cache, s.v_cache,
            s.cos_table, s.sin_table,
            temp, // output overwrites temp
            dim, cfg.num_heads, cfg.kv_heads, s.pos, cfg.max_seq
        );

        // 1c. Residual: x += attn_out
        for (int i = 0; i < dim; i += Kernels.VLEN) {
            DoubleVector xv = DoubleVector.fromArray(Kernels.SPECIES, x, i);
            DoubleVector tv = DoubleVector.fromArray(Kernels.SPECIES, temp, i);
            xv.add(tv).intoArray(x, i);
        }

        // === 2. FFN block ===
        // x = x + ffn(rms_norm(x))

        // 2a. RMSNorm
        Kernels.rms_norm(x, w.ffn_norm_gamma, temp, 1, dim, cfg.norm_eps);

        // 2b. SwiGLU: gate = silu(x @ W_gate) * (x @ W_up)
        Kernels.matmul_swiglu(
            temp, 1, dim, // x[1,dim]
            w.w_gate, w.w_up, // W_gate[dim,hidden], W_up[dim,hidden]
            temp_hidden, hidden // out[1,hidden]
        );

        // 2c. Down proj: ffn_out = gate * W_down
        Kernels.matmul_down(
            temp_hidden, 1, hidden, // [1,hidden]
            w.w_down, hidden, dim, // [hidden,dim]
            temp // [1,dim]
        );

        // 2d. Residual: x += ffn_out
        for (int i = 0; i < dim; i += Kernels.VLEN) {
            DoubleVector xv = DoubleVector.fromArray(Kernels.SPECIES, x, i);
            DoubleVector tv = DoubleVector.fromArray(Kernels.SPECIES, temp, i);
            xv.add(tv).intoArray(x, i);
        }
    }

    /**
     * Run full model: embed -> N layers -> norm -> lm_head
     */
    public static int generate_token(
            double[] token_embedding, // [dim] - current token
            Weights[] layers, // array of layer weights
            double[] final_norm_gamma, // [dim]
            double[] lm_head, // [dim, vocab_size]
            State[] states, // per-layer KV cache
            Config cfg,
            int vocab_size) {

        double[] x = token_embedding.clone();

        // Run all decoder layers
        for (int l = 0; l < layers.length; l++) {
            forward_layer(x, layers[l], states[l], cfg);
        }

        // Final norm
        double[] normed = new double[cfg.dim];
        Kernels.rms_norm(x, final_norm_gamma, normed, 1, cfg.dim, cfg.norm_eps);

        // LM head: logits = normed @ lm_head
        double[] logits = new double[vocab_size];
        Kernels.matmul_down(normed, 1, cfg.dim, lm_head, cfg.dim, vocab_size, logits);

        // Greedy decode: argmax
        int max_idx = 0;
        double max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }

        // Advance all layer positions
        for (State s : states) s.pos++;

        return max_idx;
    }
    
    
/*
     public static void endToEndLlamaUsage(){
        // 1. Load config + weights once
Config cfg = new Config(); // set dim=4096 for Llama2-7B
Weights[] layers = loadWeights("llama2-7b.bin", 32); // 32 layers
double[] final_norm = loadFinalNorm();
double[] lm_head = loadLmHead(); // [4096, 32000]

// 2. Init state per layer
LlamaLayer.State[] states = new LlamaLayer.State[32];
for (int i = 0; i < 32; i++) states[i] = new LlamaLayer.State(cfg);

// 3. Token loop
int[] prompt_tokens = tokenize("The capital of France is");
double[] embedding_table = loadEmbeddings(); // [vocab, dim]

double[] x = new double[cfg.dim];
for (int t = 0; t < prompt_tokens.length + 50; t++) {
    int token_id = t < prompt_tokens.length? prompt_tokens[t] : next_token;
    System.arraycopy(embedding_table, token_id * cfg.dim, x, 0, cfg.dim);

    int next_token = LlamaLayer.generate_token(
        x, layers, final_norm, lm_head, states, cfg, 32000
    );

    if (t >= prompt_tokens.length) {
        System.out.print(detokenize(next_token) + " ");
    }
}
    }
*/
    
}