package com.github.gbenroscience.gpu.llm.opencl;

/**
 * Runtime switches between the optimised kernels (default) and the original generic implementations, so any
 * optimisation can be A/B-tested on a given device without rebuilding:
 * <ul>
 *   <li>{@code -Dllm.gemv=legacy}      one work-item per output row (original q8_0_gemv_plain / f32_gemv)</li>
 *   <li>{@code -Dllm.attention=legacy} three launches per head per layer (attn_scores / softmax / attn_weighted_sum)</li>
 *   <li>{@code -Dllm.rmsnorm=legacy}   two kernels plus a blocking host read-back per norm</li>
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

    static boolean legacyAttention() {
        return legacy("llm.attention");
    }

    static boolean legacyRmsnorm() {
        return legacy("llm.rmsnorm");
    }

    static String describe() {
        return "kernel variants: gemv=" + (legacyGemv() ? "legacy" : "workgroup-per-row")
                + ", attention=" + (legacyAttention() ? "legacy per-head" : "fused all-heads")
                + ", rmsnorm=" + (legacyRmsnorm() ? "legacy 2-kernel+readback" : "fused single launch");
    }
}
