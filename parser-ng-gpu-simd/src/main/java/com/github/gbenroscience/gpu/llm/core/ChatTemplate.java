package com.github.gbenroscience.gpu.llm.core;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Chat framing for instruct models, built at the TOKEN-ID level: the structural tokens ({@code <|start_header_id|>},
 * {@code <|eot_id|>}, ...) are looked up in the vocabulary and inserted as single ids, never run through the BPE
 * regex (which would split them into ordinary pieces); only the message text is encoded.
 *
 * <p>Why this exists: an instruct model that is fed a bare string such as {@code "2+2"} treats it as text to
 * continue. It then never emits its end-of-turn token, so generation only stops at {@code maxNewTokens}
 * (255 tokens of "2+2=4" repeated in a user's log). With the template the model answers and ends its turn.
 *
 * <p>Only the Llama-3 header format is implemented ({@link #supportsLlama3}); other families are refused rather than
 * guessed. Format (Llama 3.x instruct):
 * <pre>
 * &lt;|begin_of_text|&gt;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;\n\n{system}&lt;|eot_id|&gt;
 * &lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;\n\n{user}&lt;|eot_id|&gt;
 * &lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;\n\n
 * </pre>
 */
public final class ChatTemplate {

    public static final String BEGIN_OF_TEXT = "<|begin_of_text|>";
    public static final String START_HEADER = "<|start_header_id|>";
    public static final String END_HEADER = "<|end_header_id|>";
    public static final String EOT = "<|eot_id|>";
    public static final String EOM = "<|eom_id|>";
    public static final String END_OF_TEXT = "<|end_of_text|>";

    private ChatTemplate() {
    }

    /** What the template needs from a tokenizer. */
    public interface Encoder {

        /** Encodes ordinary text WITHOUT a BOS token and without interpreting special tokens. */
        int[] encodeText(String text);

        /** Vocabulary id of an exact token string, or -1 if the vocabulary has no such token. */
        int tokenId(String piece);

        /** Beginning-of-sequence id from the model metadata, or -1. */
        int bosId();

        /** End-of-sequence id from the model metadata, or -1. */
        int eosId();
    }

    public record Message(String role, String content) {

        public Message {
            if (role == null || content == null) {
                throw new IllegalArgumentException("role and content must not be null");
            }
            if (!role.equals("system") && !role.equals("user") && !role.equals("assistant")) {
                throw new IllegalArgumentException("role must be system, user or assistant, got \"" + role + "\"");
            }
        }
    }

    /** True when the vocabulary carries the Llama-3 header tokens (a Llama-3 family instruct model). */
    public static boolean supportsLlama3(Encoder enc) {
        return enc.tokenId(START_HEADER) >= 0 && enc.tokenId(END_HEADER) >= 0 && enc.tokenId(EOT) >= 0;
    }

    /**
     * Token ids of the conversation followed by an open assistant header, ready to feed to the model.
     * The BOS token is added once at the start (taken from the vocabulary's {@code <|begin_of_text|>} when the metadata has none).
     * A trailing assistant message (a prefill) is left open: no end-of-turn token is appended after it.
     */
    public static int[] llama3(Encoder enc, List<Message> messages) {
        if (!supportsLlama3(enc)) {
            throw new UnsupportedOperationException("this tokenizer's vocabulary has no " + START_HEADER + " / " + END_HEADER + " / " + EOT
                    + " tokens: the Llama-3 chat template does not apply to this model");
        }
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("at least one message is required");
        }
        final int start = enc.tokenId(START_HEADER);
        final int end = enc.tokenId(END_HEADER);
        final int eot = enc.tokenId(EOT);
        int bos = enc.bosId() >= 0 ? enc.bosId() : enc.tokenId(BEGIN_OF_TEXT);
        List<Integer> ids = new ArrayList<>();
        if (bos >= 0) {
            ids.add(bos);
        }
        // "\n\n" is encoded on its own so it stays ONE token, as in the reference tokenizer, instead of being split by the
        // whitespace rule of the GPT-2 pre-tokenizer when glued to the first word of the message.
        final int[] blank = enc.encodeText("\n\n");
        for (int i = 0; i < messages.size(); i++) {
            Message m = messages.get(i);
            ids.add(start);
            addAll(ids, enc.encodeText(m.role()));
            ids.add(end);
            addAll(ids, blank);
            addAll(ids, enc.encodeText(m.content()));
            boolean openAssistant = i == messages.size() - 1 && m.role().equals("assistant");
            if (!openAssistant) {
                ids.add(eot);
            }
        }
        if (!messages.get(messages.size() - 1).role().equals("assistant")) {
            ids.add(start);
            addAll(ids, enc.encodeText("assistant"));
            ids.add(end);
            addAll(ids, blank);
        }
        int[] out = new int[ids.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ids.get(i);
        }
        return out;
    }

    /**
     * Every token id that ends generation: the metadata EOS plus whichever of {@code <|eot_id|>}, {@code <|eom_id|>},
     * {@code <|end_of_text|>} exist in the vocabulary (Llama-3 GGUFs differ in which one they list as EOS).
     */
    public static int[] stopIds(Encoder enc) {
        Set<Integer> s = new LinkedHashSet<>();
        if (enc.eosId() >= 0) {
            s.add(enc.eosId());
        }
        for (String piece : new String[]{EOT, EOM, END_OF_TEXT}) {
            int id = enc.tokenId(piece);
            if (id >= 0) {
                s.add(id);
            }
        }
        int[] out = new int[s.size()];
        int i = 0;
        for (int v : s) {
            out[i++] = v;
        }
        return out;
    }

    private static void addAll(List<Integer> to, int[] ids) {
        for (int id : ids) {
            to.add(id);
        }
    }
}
