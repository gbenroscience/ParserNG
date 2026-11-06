package com.github.gbenroscience.gpu.llm.opencl;

/**
 * Runtime switches between the optimised kernels (default) and the original generic implementations, so any
 * optimisation can be A/B-tested on a given device without rebuilding:
 * <ul>
 *   <li>{@code -Dllm.gemv=legacy}      one work-item per output row (original q8_0_gemv_plain / f32_gemv)</li>
 *   <li>{@code -Dllm.attention=legacy} three launches per head per layer (attn_scores / softmax / attn_weighted_sum)</li>
 *   <li>{@code -Dllm.rmsnorm=legacy}   two kernels plus a blocking host read-back per norm</li>
 *   <li>{@code -Dllm.layout=aos|soa|auto} device layout of every Q8_0 tensor: the GGUF 34-byte blocks (aos), the coalesced
 *       structure-of-arrays form (soa), or let the engine time both on the device at load (auto, the default). Any
 *       explicit {@code -Dllm.gemv} other than {@code auto} implies aos, because those kernels read AoS.</li>
 *   <li>{@code -Dllm.wo=f32|q8|auto} the attention output projection: dequantised to F32 (f32) or kept as Q8_0 with the
 *       activation quantised like every other Q8_0 projection (q8; auto = q8 whenever the file stores wo as Q8_0).</li>
 * </ul>
 * Read on every call so tests can flip them.
 */
final class Variants {

    private Variants() {
    }

    private static boolean legacy(String key) {
        return "legacy".equalsIgnoreCase(System.getProperty(key));
    }

    static boolean legacyGemv() {
        return legacy("llm.gemv");
    }

    /** Forced decode GEMV variant from -Dllm.gemv=wg|vloadf|u32, or -1 (not forced: use the context's autotuned choice). */
    static int forcedGemvVariant() {
        String v = System.getProperty("llm.gemv");
        if (v == null) {
            return -1;
        }
        switch (v.toLowerCase()) {
            case "wg":
                return 0;
            case "vloadf":
                return 1;
            case "u32":
                return 2;
            default:
                return -1; // "legacy" and "auto" are handled elsewhere
        }
    }

    /** True unless a forced variant (or "legacy") was requested: the engine may autotune the decode GEMV at load. */
    static boolean autotuneGemv() {
        String v = System.getProperty("llm.gemv");
        return v == null || v.equalsIgnoreCase("auto");
    }

    /** True unless -Dllm.gemm is set to legacy or rows: the engine then times legacy vs row-wise GEMMs at load and keeps the faster. */
    static boolean autotuneGemm() {
        String v = System.getProperty("llm.gemm");
        return v == null || v.equalsIgnoreCase("auto");
    }

    /** -Dllm.gemm=rows forces the row-wise GEMMs. */
    static boolean forcedRowsGemm() {
        return "rows".equalsIgnoreCase(System.getProperty("llm.gemm"));
    }

    /** -Dllm.gemm=legacy selects the original one-thread-per-output prefill GEMMs. */
    static boolean legacyGemm() {
        return legacy("llm.gemm");
    }

    /** -Dllm.prefill=legacy restores the original prefill synchronisation: per-layer positions upload, per-layer clFinish (and, with -Dllm.rmsnorm=legacy, the host-round-trip norms). */
    static boolean legacyPrefill() {
        return legacy("llm.prefill");
    }

    static boolean legacyAttention() {
        return legacy("llm.attention");
    }

    static boolean legacyRmsnorm() {
        return legacy("llm.rmsnorm");
    }

    /**
     * -Dllm.flushEvery=N: during a decode step, hand the queue to the device after every N-th layer so it starts
     * executing while the host is still enqueueing the rest (0 disables; default 4). Many drivers batch commands until
     * a flush or a blocking call, which leaves the device idle for the whole enqueue time (7-9 ms per token measured on
     * an HD 520). Results are identical either way.
     */
    static int flushEvery() {
        String v = System.getProperty("llm.flushEvery");
        if (v == null || v.isBlank()) {
            return 4;
        }
        try {
            return Math.max(0, Integer.parseInt(v.trim()));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("-Dllm.flushEvery must be a non-negative integer, got \"" + v + "\"");
        }
    }

    /** Layout request: "aos", "soa" or "auto". Anything else is rejected loudly rather than silently ignored. */
    static String layoutRequest() {
        String v = System.getProperty("llm.layout");
        if (v == null || v.isBlank() || v.equalsIgnoreCase("auto")) {
            // a forced AoS GEMV kernel (or legacy) can only read the AoS layout
            String g = System.getProperty("llm.gemv");
            return (g != null && !g.equalsIgnoreCase("auto")) ? "aos" : "auto";
        }
        switch (v.toLowerCase()) {
            case "aos":
            case "soa":
                return v.toLowerCase();
            default:
                throw new IllegalArgumentException("-Dllm.layout must be aos, soa or auto, got \"" + v + "\"");
        }
    }

    /** -Dllm.wo: "f32", "q8" or "auto" (default). */
    static String woRequest() {
        String v = System.getProperty("llm.wo");
        if (v == null || v.isBlank() || v.equalsIgnoreCase("auto")) {
            return "auto";
        }
        switch (v.toLowerCase()) {
            case "f32":
            case "q8":
                return v.toLowerCase();
            default:
                throw new IllegalArgumentException("-Dllm.wo must be f32, q8 or auto, got \"" + v + "\"");
        }
    }

    static String describe() {
        return "kernel variants: gemv=" + (legacyGemv() ? "legacy" : forcedGemvVariant() >= 0 ? "forced " + System.getProperty("llm.gemv") : "autotuned at load (wg|vloadf|u32)")
                + ", gemm=" + (legacyGemm() ? "legacy per-token" : forcedRowsGemm() ? "forced row-wise" : "autotuned at load (legacy|row-wise)")
                + ", attention=" + (legacyAttention() ? "legacy per-head" : "fused all-heads")
                + ", rmsnorm=" + (legacyRmsnorm() ? "legacy 2-kernel+readback" : "fused single launch")
                + ", layout=" + layoutRequest() + ", wo=" + woRequest() + ", flushEvery=" + flushEvery()
                + ", programCache=" + (ProgramCache.enabled() ? "on" : "off");
    }
}
