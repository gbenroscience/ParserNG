package com.github.gbenroscience.gpu.llm;

/**
 * Backend-agnostic contract for a single loaded Llama model bound to a GPU,
 * ready to generate. The LLM counterpart of
 * {@link com.github.gbenroscience.gpu.GpuCompositeExpression} -- same
 * relationship to its two implementations
 * ({@code com.github.gbenroscience.gpu.opencl.llm.OpenClLlamaEngine} and
 * {@code com.github.gbenroscience.gpu.cuda.llm.CudaLlamaEngine}) that
 * GpuCompositeExpression has to OpenClCompositeExpression/CudaCompositeExpression:
 * code written against this interface doesn't need to know or care which
 * backend actually loaded the model.
 *
 * Extends AutoCloseable because every implementation owns a whole model's
 * worth of device memory (weights, KV cache, scratch buffers, the compiled
 * kernel program) -- always use try-with-resources.
 */
public interface LlamaGpuEngine extends AutoCloseable {

    /** Runs the full prompt through the model, then samples up to cfg.maxNewTokens new tokens. Returns the DETOKENIZED generated continuation only (not the echoed prompt). */
    String generate(String prompt, LlamaGenerationConfig cfg) throws Throwable;

    /**
     * Same as {@link #generate(String, LlamaGenerationConfig)} but hands the text to {@code listener} as it is produced.
     * This default implementation is for backends without incremental output: it generates everything and then delivers
     * the whole text in one piece. The OpenCL engine overrides it with true token-by-token streaming.
     */
    default String generate(String prompt, LlamaGenerationConfig cfg, TokenListener listener) throws Throwable {
        String text = generate(prompt, cfg);
        boolean go = text.isEmpty() || listener.onText(text);
        listener.onComplete(go ? StopReason.UNKNOWN : StopReason.CANCELLED, -1);
        return text;
    }

    /**
     * Which GPU this instance is bound to -- e.g.
     * "[platform 0: Intel(R) OpenCL Graphics] [device 0: Intel(R) Corporation Intel(R) Iris(R) Xe Graphics]"
     * on OpenCL, "[device 0: NVIDIA GeForce RTX 4090 (compute capability 8.9)]"
     * on CUDA. Fixed at construction time for this instance's whole lifetime.
     */
    String getDeviceDescription();

    /**
     * Narrows AutoCloseable.close()'s `throws Exception` to no checked
     * exception -- both implementations do best-effort cleanup internally
     * and never throw from close(), so callers shouldn't have to handle
     * one just because the interface theoretically allows it.
     */
    @Override
    void close();
}