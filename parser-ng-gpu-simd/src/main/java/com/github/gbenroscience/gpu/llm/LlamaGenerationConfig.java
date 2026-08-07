package com.github.gbenroscience.gpu.llm;

/**
 * Backend-agnostic counterpart of
 * {@code LlamaOpenCLEngine.GenerationConfig}/{@code LlamaCudaEngine.GenerationConfig}
 * (themselves each paired with their own {@code Sampler.Config}) -- one flat
 * shape, same field names/defaults as both, so code written against
 * {@link LlamaGpuBridge} never needs to import a backend-specific config
 * class. {@link com.github.gbenroscience.gpu.opencl.llm.OpenClLlamaEngine}/
 * {@link com.github.gbenroscience.gpu.cuda.llm.CudaLlamaEngine} translate
 * this into their own backend's nested types internally.
 *
 * If you need a sampler/generation option that exists on one backend's
 * Sampler.Config but isn't listed here, go to that backend's engine
 * directly ({@code LlamaOpenCLEngine}/{@code LlamaCudaEngine}) rather than
 * through the bridge -- same "escape hatch" relationship
 * {@link com.github.gbenroscience.gpu.GpuExpressionBridge} has with the
 * backend-specific expression bridges it wraps.
 */
public final class LlamaGenerationConfig {

    public int maxNewTokens = 256;
    /** -1 means "use the tokenizer's own EOS id". */
    public int eosTokenId = -1;

    public float temperature = 0.8f;
    /** 0 disables top-k filtering. */
    public int topK = 40;
    /** 1.0 disables top-p filtering. */
    public float topP = 0.95f;
    /** 1.0 disables the repetition penalty. */
    public float repetitionPenalty = 1.1f;
    public int repetitionPenaltyWindow = 64;
    public long seed = System.nanoTime();
}