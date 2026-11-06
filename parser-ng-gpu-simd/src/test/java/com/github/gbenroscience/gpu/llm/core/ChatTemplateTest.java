package com.github.gbenroscience.gpu.llm.core;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ChatTemplateTest {

    /** Ordinary text encodes to one id per char (its code point); specials live at 1000+. */
    private static final class Fake implements ChatTemplate.Encoder {

        final Map<String, Integer> specials;
        final int bos, eos;

        Fake(Map<String, Integer> specials, int bos, int eos) {
            this.specials = specials;
            this.bos = bos;
            this.eos = eos;
        }

        @Override
        public int[] encodeText(String text) {
            return text.chars().toArray();
        }

        @Override
        public int tokenId(String piece) {
            return specials.getOrDefault(piece, -1);
        }

        @Override
        public int bosId() {
            return bos;
        }

        @Override
        public int eosId() {
            return eos;
        }
    }

    private static Fake llama3() {
        return new Fake(Map.of(ChatTemplate.BEGIN_OF_TEXT, 1000, ChatTemplate.START_HEADER, 1001, ChatTemplate.END_HEADER, 1002,
                ChatTemplate.EOT, 1003, ChatTemplate.EOM, 1004, ChatTemplate.END_OF_TEXT, 1005), 1000, 1005);
    }

    private static int[] seq(Object... parts) {
        List<Integer> l = new ArrayList<>();
        for (Object p : parts) {
            if (p instanceof Integer i) {
                l.add(i);
            } else {
                ((String) p).chars().forEach(l::add);
            }
        }
        return l.stream().mapToInt(Integer::intValue).toArray();
    }

    @Test
    void userTurnIsFramedWithHeadersAndOpenAssistant() {
        int[] got = ChatTemplate.llama3(llama3(), List.of(new ChatTemplate.Message("user", "2+2")));
        int[] want = seq(1000, 1001, "user", 1002, "\n\n", "2+2", 1003, 1001, "assistant", 1002, "\n\n");
        assertArrayEquals(want, got);
    }

    @Test
    void systemUserAssistantConversation() {
        int[] got = ChatTemplate.llama3(llama3(), List.of(
                new ChatTemplate.Message("system", "be brief"),
                new ChatTemplate.Message("user", "hi"),
                new ChatTemplate.Message("assistant", "hello"),
                new ChatTemplate.Message("user", "bye")));
        int[] want = seq(1000,
                1001, "system", 1002, "\n\n", "be brief", 1003,
                1001, "user", 1002, "\n\n", "hi", 1003,
                1001, "assistant", 1002, "\n\n", "hello", 1003,
                1001, "user", 1002, "\n\n", "bye", 1003,
                1001, "assistant", 1002, "\n\n");
        assertArrayEquals(want, got);
    }

    @Test
    void trailingAssistantMessageIsAPrefillLeftOpen() {
        int[] got = ChatTemplate.llama3(llama3(), List.of(
                new ChatTemplate.Message("user", "q"), new ChatTemplate.Message("assistant", "The answer is")));
        int[] want = seq(1000, 1001, "user", 1002, "\n\n", "q", 1003, 1001, "assistant", 1002, "\n\n", "The answer is");
        assertArrayEquals(want, got, "no end-of-turn after a prefill, and no second assistant header");
    }

    @Test
    void bosFallsBackToTheVocabularyTokenWhenMetadataHasNone() {
        Fake noBosMeta = new Fake(llama3().specials, -1, -1);
        assertEquals(1000, ChatTemplate.llama3(noBosMeta, List.of(new ChatTemplate.Message("user", "x")))[0]);
    }

    @Test
    void stopIdsCoverEveryEndOfGenerationTokenWithoutDuplicates() {
        assertArrayEquals(new int[]{1005, 1003, 1004}, ChatTemplate.stopIds(llama3()), "metadata eos first, then eot, eom (end_of_text already listed)");
        Fake eotMeta = new Fake(llama3().specials, 1000, 1003);
        assertArrayEquals(new int[]{1003, 1004, 1005}, ChatTemplate.stopIds(eotMeta));
        Fake none = new Fake(Map.of(), -1, -1);
        assertEquals(0, ChatTemplate.stopIds(none).length);
    }

    @Test
    void nonLlama3VocabularyIsRefusedNotGuessed() {
        Fake other = new Fake(Map.of("<s>", 1), 1, 2);
        assertFalse(ChatTemplate.supportsLlama3(other));
        assertThrows(UnsupportedOperationException.class, () -> ChatTemplate.llama3(other, List.of(new ChatTemplate.Message("user", "x"))));
    }

    @Test
    void invalidInputIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new ChatTemplate.Message("tool", "x"));
        assertThrows(IllegalArgumentException.class, () -> new ChatTemplate.Message("user", null));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplate.llama3(llama3(), List.of()));
    }
}
