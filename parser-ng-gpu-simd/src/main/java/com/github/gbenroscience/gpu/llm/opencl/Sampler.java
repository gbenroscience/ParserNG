package com.github.gbenroscience.gpu.llm.opencl;

import java.util.List;

/**
 * Public sampler of the opencl backend. It carries NO logic of its own: it
 * delegates to {@link com.github.gbenroscience.gpu.llm.core.Sampler}, the single
 * implementation shared by CUDA, OpenCL and Metal, so the three backends cannot
 * sample differently. See that class for the exact pipeline (temperature ->
 * top-k -> renormalise -> top-p -> sample), NaN/Inf handling and the
 * validation rules for {@link Config}.
 */
public final class Sampler {

    /** Same fields and validation as {@link com.github.gbenroscience.gpu.llm.core.Sampler.Config}. */
    public static final class Config extends com.github.gbenroscience.gpu.llm.core.Sampler.Config {
    }

    private final com.github.gbenroscience.gpu.llm.core.Sampler delegate;

    public Sampler(Config cfg) {
        this.delegate = new com.github.gbenroscience.gpu.llm.core.Sampler(cfg);
    }

    /** @throws IllegalStateException if the logits contain NaN or +Inf */
    public int sample(float[] logits, List<Integer> history) {
        return delegate.sample(logits, history);
    }
}
