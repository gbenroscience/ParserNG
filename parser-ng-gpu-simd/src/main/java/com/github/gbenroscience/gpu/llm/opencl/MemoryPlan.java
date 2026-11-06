package com.github.gbenroscience.gpu.llm.opencl;

import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Model-free device-memory plan: allocates exactly ONE transformer layer (weights + its state) plus the once-only
 * buffers through the real engine classes -- so the sizes are what the engine really asks the driver for, not a
 * re-derivation -- and extrapolates to {@code layers}. Only per-layer categories (the layer weights and the KV cache)
 * scale with the layer count; the RoPE tables and decode/prefill scratch are shared by all layers and counted once.
 * The result can be compared with the "device memory live" figure the engine prints for a real model of the same shape.
 */
final class MemoryPlan {

    private MemoryPlan() {
    }

    static long run(GpuContext ctx, LlamaLayer.Config cfg, int vocab, boolean headQ8) throws Throwable {
        return run(ctx, cfg, vocab, headQ8, false);
    }

    /** @param woQ8 whether wo is kept as Q8_0 (true) or dequantised to F32 (false) */
    static long run(GpuContext ctx, LlamaLayer.Config cfg, int vocab, boolean headQ8, boolean woQ8) throws Throwable {
        final int dim = cfg.dim;
        final int qd = cfg.num_heads * cfg.head_dim;
        final int kvd = cfg.kv_heads * cfg.head_dim;
        final int hidden = cfg.hidden_dim;
        // inputs are always the GGUF (AoS) bytes; the weights class repacks them when the context uses SoA
        byte[] wq = new byte[(int) Q8Layout.weightBytes(false, qd, dim)];
        byte[] wk = new byte[(int) Q8Layout.weightBytes(false, kvd, dim)];
        byte[] wg = new byte[(int) Q8Layout.weightBytes(false, hidden, dim)];
        byte[] wd = new byte[(int) Q8Layout.weightBytes(false, dim, hidden)];
        byte[] woBytes = woQ8 ? new byte[(int) Q8Layout.weightBytes(false, dim, qd)] : null;
        float[] wo = woQ8 ? null : new float[dim * qd];
        float[] norm = new float[dim];
        LlamaLayer.GpuWeights w = null;
        LlamaLayer.GpuState st = null;
        MemorySegment head = null, finalNorm = null, xs = null, xb = null, lr = null;
        LlamaLayer.LogitsScratch ls = null;
        long predicted = 0;
        try {
            long before = ctx.allocations.liveBytes();
            w = new LlamaLayer.GpuWeights(ctx, dim, qd, kvd, hidden, wq, wk, wk, woBytes, wo, wg, wg, wd, norm, norm);
            st = new LlamaLayer.GpuState(ctx, cfg);

            ctx.allocCategory = "final norm gamma";
            finalNorm = LlamaLayer.allocFloats(ctx, dim);
            ctx.allocCategory = "lm head";
            head = LlamaLayer.allocBytes(ctx, headQ8 ? Q8Layout.weightBytes(ctx.q8Soa, vocab, dim) : (long) vocab * dim * 4L);
            ctx.allocCategory = "engine scratch";
            xs = LlamaLayer.allocFloats(ctx, dim);
            xb = cfg.max_prefill_batch > 0 ? LlamaLayer.allocFloats(ctx, (long) cfg.max_prefill_batch * dim) : null;
            lr = LlamaLayer.allocFloats(ctx, dim);
            ls = new LlamaLayer.LogitsScratch(ctx, cfg, vocab);

            SortedMap<String, long[]> all = ctx.allocations.byCategory();
            Map<String, Double> scaled = new TreeMap<>();
            double grand = 0;
            for (Map.Entry<String, long[]> e : all.entrySet()) {
                boolean perLayer = e.getKey().startsWith("weights:") || e.getKey().equals("state: KV cache");
                double v = e.getValue()[0] * (perLayer ? (double) cfg.n_layers : 1.0);
                scaled.put(e.getKey(), v);
                grand += v;
            }
            System.out.printf("%n==================== device memory plan: %d layers, dim %d, hidden %d, heads %d/%d (head_dim %d), vocab %d, context %d, prefill batch %d, LM head %s, Q8_0 layout %s, wo %s ====================%n",
                    cfg.n_layers, dim, hidden, cfg.num_heads, cfg.kv_heads, cfg.head_dim, vocab, cfg.max_seq, cfg.max_prefill_batch,
                    headQ8 ? "Q8_0" : "F32", ctx.q8Soa ? "SoA" : "AoS", woQ8 ? "Q8_0" : "F32");
            System.out.printf("%-46s %12s %14s %10s%n", "category", "MB each", "MB total", "share");
            for (Map.Entry<String, Double> e : scaled.entrySet()) {
                boolean perLayer = e.getKey().startsWith("weights:") || e.getKey().equals("state: KV cache");
                System.out.printf("%-46s %12.1f %14.1f %9.1f%%%s%n", e.getKey(), e.getValue() / (perLayer ? cfg.n_layers : 1) / 1e6,
                        e.getValue() / 1e6, 100 * e.getValue() / grand, perLayer ? "  (x layers)" : "  (once)");
            }
            System.out.printf("%-46s %12s %14.1f%n", "TOTAL predicted device memory", "", grand / 1e6);
            predicted = Math.round(grand);
            System.out.printf("(one layer of weights + KV = %.1f MB; host side, not counted above: Q8_0 token-embedding bytes kept for row lookup = %.1f MB)%n",
                    (ctx.allocations.liveBytes() - before) / 1e6, (double) Q8Layout.weightBytes(false, vocab, dim) / 1e6);
        } finally {
            if (ls != null) {
                ls.free(ctx);
            }
            for (MemorySegment m : new MemorySegment[]{head, finalNorm, xs, xb, lr}) {
                LlamaLayer.freeQuietly(ctx, m);
            }
            if (st != null) {
                st.close();
            }
            if (w != null) {
                w.close();
            }
        }
        return predicted;
    }
}
