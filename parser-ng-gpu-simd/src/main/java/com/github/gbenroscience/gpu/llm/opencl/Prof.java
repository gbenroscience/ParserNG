package com.github.gbenroscience.gpu.llm.opencl;

/**
 * Opt-in stage profiler for the decode path ({@code -Dllm.profile=true}).
 * It inserts a clFinish after every stage so the host can attribute wall time to
 * the stage, which SERIALISES the pipeline: use it to find where time goes, not
 * to measure throughput (use {@code -Dllm.timing=true} for that; it adds no syncs).
 */
final class Prof {

    static final boolean ON = Boolean.getBoolean("llm.profile");

    static final int NORM = 0, QKV = 1, ROPE_CACHE = 2, ATTENTION = 3, WO = 4, FFN_UP = 5, FFN_DOWN = 6, LOGITS = 7;
    private static final String[] NAMES = {"rmsnorm+quantize (x2/layer)", "q/k/v projections", "rope + kv-cache write",
        "attention (all heads)", "wo projection + residual", "ffn gate/up/swiglu/quantize", "ffn down + residual", "final norm + lm head + readback"};
    private static final long[] NS = new long[NAMES.length];
    private static long last;

    private Prof() {
    }

    static void start(GpuContext ctx) throws Throwable {
        if (ON) {
            LlamaLayer.finish(ctx);
            last = System.nanoTime();
        }
    }

    static void mark(GpuContext ctx, int stage) throws Throwable {
        if (ON) {
            LlamaLayer.finish(ctx);
            long now = System.nanoTime();
            NS[stage] += now - last;
            last = now;
        }
    }

    static void reset() {
        java.util.Arrays.fill(NS, 0L);
    }

    static String report() {
        long total = 0;
        for (long v : NS) {
            total += v;
        }
        StringBuilder sb = new StringBuilder("stage profile (serialised, clFinish between stages):\n");
        for (int i = 0; i < NAMES.length; i++) {
            sb.append(String.format("  %-34s %9.1f ms  %5.1f%%%n", NAMES[i], NS[i] / 1e6, total == 0 ? 0.0 : 100.0 * NS[i] / total));
        }
        return sb.append(String.format("  %-34s %9.1f ms%n", "total", total / 1e6)).toString();
    }
}
