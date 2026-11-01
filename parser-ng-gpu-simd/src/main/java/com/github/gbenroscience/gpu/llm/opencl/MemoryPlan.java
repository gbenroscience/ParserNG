package com.github.gbenroscience.gpu.llm.opencl;

import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Model-free device-memory plan: allocates exactly ONE transformer layer (weights + per-layer state) plus the
 * once-only buffers through the real engine classes -- so the sizes are what the engine really asks the driver for,
 * not a re-derivation -- and extrapolates to {@code layers}. The result can be compared with the "device memory
 * live" figure the engine prints for a real model of the same shape.
 */
final class MemoryPlan {

    private MemoryPlan() {
    }

    /** @return the predicted total device bytes for the whole model (one real layer x layers + once-only buffers) */
    static long run(GpuContext ctx, LlamaLayer.Config cfg, int vocab, boolean headQ8) throws Throwable {
        final int dim = cfg.dim;
        final int qd = cfg.num_heads * cfg.head_dim;
        final int kvd = cfg.kv_heads * cfg.head_dim;
        final int hidden = cfg.hidden_dim;
        long q8 = 34L;
        byte[] wq = new byte[(int) ((long) qd * (dim / 32) * q8)];
        byte[] wk = new byte[(int) ((long) kvd * (dim / 32) * q8)];
        byte[] wg = new byte[(int) ((long) hidden * (dim / 32) * q8)];
        byte[] wd = new byte[(int) ((long) dim * (hidden / 32) * q8)];
        float[] wo = new float[dim * qd];
        float[] norm = new float[dim];
        LlamaLayer.GpuWeights w = null;
        LlamaLayer.GpuState st = null;
        MemorySegment head = null, finalNorm = null, xs = null, xb = null, lr = null;
        LlamaLayer.LogitsScratch ls = null;
        long predicted = 0;
        try {
            long before = ctx.allocations.liveBytes();
            w = new LlamaLayer.GpuWeights(ctx, wq, wk, wk, wo, wg, wg, wd, norm, norm);
            st = new LlamaLayer.GpuState(ctx, cfg);
            SortedMap<String, long[]> layerCats = new TreeMap<>(ctx.allocations.byCategory());
            long layerBytes = ctx.allocations.liveBytes() - before;

            ctx.allocCategory = "final norm gamma";
            finalNorm = LlamaLayer.allocFloats(ctx, dim);
            ctx.allocCategory = "lm head";
            head = LlamaLayer.allocBytes(ctx, headQ8 ? (long) vocab * (dim / 32) * q8 : (long) vocab * dim * 4L);
            ctx.allocCategory = "engine scratch";
            xs = LlamaLayer.allocFloats(ctx, dim);
            xb = cfg.max_prefill_batch > 0 ? LlamaLayer.allocFloats(ctx, (long) cfg.max_prefill_batch * dim) : null;
            lr = LlamaLayer.allocFloats(ctx, dim);
            ls = new LlamaLayer.LogitsScratch(ctx, cfg, vocab);
            long once = ctx.allocations.liveBytes() - before - layerBytes;

            System.out.printf("%n==================== device memory plan: %d layers, dim %d, hidden %d, heads %d/%d (head_dim %d), vocab %d, context %d, prefill batch %d, LM head %s ====================%n",
                    cfg.n_layers, dim, hidden, cfg.num_heads, cfg.kv_heads, cfg.head_dim, vocab, cfg.max_seq, cfg.max_prefill_batch, headQ8 ? "Q8_0" : "F32");
            System.out.printf("%-46s %12s %14s %10s%n", "category (one real layer allocated, scaled)", "MB / layer", "MB x layers", "share");
            double total = 0;
            Map<String, Double> scaled = new TreeMap<>();
            for (Map.Entry<String, long[]> e : layerCats.entrySet()) {
                if (e.getKey().startsWith("state:") || e.getKey().startsWith("weights:")) {
                    scaled.put(e.getKey(), e.getValue()[0] * (double) cfg.n_layers);
                }
            }
            double onceTotal = once;
            double grand = onceTotal;
            for (double v : scaled.values()) {
                grand += v;
            }
            for (Map.Entry<String, Double> e : scaled.entrySet()) {
                System.out.printf("%-46s %12.1f %14.1f %9.1f%%%n", e.getKey(), e.getValue() / cfg.n_layers / 1e6, e.getValue() / 1e6, 100 * e.getValue() / grand);
                total += e.getValue();
            }
            SortedMap<String, long[]> all = ctx.allocations.byCategory();
            for (String cat : new String[]{"lm head", "final norm gamma", "engine scratch", "logits scratch"}) {
                long[] v = all.get(cat);
                if (v != null) {
                    System.out.printf("%-46s %12s %14.1f %9.1f%%%n", cat + " (once)", "-", v[0] / 1e6, 100.0 * v[0] / grand);
                }
            }
            System.out.printf("%-46s %12.1f %14.1f%n", "TOTAL predicted device memory", layerBytes / 1e6, grand / 1e6);
            predicted = Math.round(grand);
            System.out.printf("(host side, not counted above: Q8_0 token-embedding bytes kept for row lookup = %.1f MB for this vocab/dim)%n", (double) vocab * (dim / 32) * q8 / 1e6);
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
