package com.github.gbenroscience.gpu.llm;

/**
 * Receives generated text as it is produced, instead of only when the whole reply is finished.
 *
 * <pre>{@code
 * engine.generate(prompt, cfg, piece -> { System.out.print(piece); System.out.flush(); return true; });
 * }</pre>
 *
 * <h2>Contract</h2>
 * <ul>
 *   <li><b>Pieces</b>: {@link #onText} is called with each newly decoded piece of text, never empty, in order. A piece is
 *       usually one token, but a token that ends in the middle of a multi-byte UTF-8 character (common for emoji and
 *       non-Latin scripts with byte-level BPE) is held back until the character is complete, so no piece ever contains
 *       a half character; the pieces joined are exactly the text the non-streaming call returns.</li>
 *   <li><b>Thread</b>: always on the thread that called generate/chat, synchronously, between tokens. The engine is
 *       single-flight: calling the same engine from inside a callback throws {@link IllegalStateException}. Hand work
 *       to another thread (a queue, {@code SwingUtilities.invokeLater}) if the callback would be slow.</li>
 *   <li><b>Timing</b>: the callback runs while the device computes the next token (the next step is already submitted),
 *       so a callback faster than one token (about 90-130 ms on the test GPUs) costs no throughput; a slower one delays
 *       the next token by the difference.</li>
 *   <li><b>Cancel</b>: return {@code false} to stop. The engine drains the device and returns the text delivered so far;
 *       it is immediately usable for another call.</li>
 *   <li><b>Errors</b>: an exception thrown by the callback aborts the generation and is rethrown unchanged from
 *       generate/chat (after the device has been drained); the engine stays usable.</li>
 * </ul>
 */
@FunctionalInterface
public interface TokenListener {

    /**
     * @param piece newly decoded text, never empty
     * @return true to continue, false to stop the generation
     */
    boolean onText(String piece);

    /**
     * Called once when the generation ends normally or by cancellation, after the last {@link #onText} (not called if
     * generate/chat throws).
     *
     * @param reason          why it ended
     * @param generatedTokens number of tokens generated, or -1 if the backend does not report it
     */
    default void onComplete(StopReason reason, int generatedTokens) {
    }
}
