package com.github.gbenroscience.gpu.llm;

/** Why a generation ended; reported to {@link TokenListener#onComplete}. */
public enum StopReason {
    /** The model produced an end-of-turn / end-of-sequence token. */
    STOP_TOKEN,
    /** {@code maxNewTokens} tokens were generated. */
    MAX_TOKENS,
    /** {@link TokenListener#onText} returned false. */
    CANCELLED,
    /** The backend does not report a reason (the non-streaming fallback of {@link LlamaGpuEngine}). */
    UNKNOWN
}
