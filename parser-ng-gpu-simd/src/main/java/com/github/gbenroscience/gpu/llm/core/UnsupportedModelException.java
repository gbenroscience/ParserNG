package com.github.gbenroscience.gpu.llm.core;

/** Thrown before any device allocation when a GGUF file cannot be run correctly by this runtime. */
public final class UnsupportedModelException extends IllegalArgumentException {

    public UnsupportedModelException(String message) {
        super(message);
    }
}
