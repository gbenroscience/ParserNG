package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.LlamaGenerationConfig;
import com.github.gbenroscience.gpu.llm.LlamaGpuEngine;
import com.github.gbenroscience.gpu.llm.StopReason;
import com.github.gbenroscience.gpu.llm.TokenListener;
import com.github.gbenroscience.gpu.llm.core.ChatTemplate;
import com.github.gbenroscience.gpu.llm.core.TinyModel;
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Streaming output: the incremental UTF-8 decoder, and generate/chat with a {@link TokenListener}. */
class StreamingTest {

    private static BpeTokenizer tokenizer;
    private static File plainModel, chatModel;

    @BeforeAll
    static void models() throws Exception {
        TinyModel plain = new TinyModel();
        plain.q8Wo = true;
        plain.build(41);
        plainModel = Files.createTempFile("stream-plain", ".gguf").toFile();
        plainModel.deleteOnExit();
        plain.write(plainModel);
        tokenizer = BpeTokenizer.fromGguf(GGUFLoader.load(plainModel));

        TinyModel chat = new TinyModel();
        chat.llama3Specials = true;
        chat.build(42);
        chatModel = Files.createTempFile("stream-chat", ".gguf").toFile();
        chatModel.deleteOnExit();
        chat.write(chatModel);
    }

    private static String probeSkip() {
        try (GpuContext c = new GpuContext()) {
            return null;
        } catch (Throwable t) {
            return "no usable OpenCL device: " + t;
        }
    }

    // ------------------------------------------------------------------ incremental decoder (no device needed)

    /** In the tiny vocabulary id b is the single-byte token for byte b, so a byte array is its own id sequence. */
    private static int[] idsOf(byte[] bytes) {
        int[] ids = new int[bytes.length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = bytes[i] & 0xFF;
        }
        return ids;
    }

    private static String streamAll(int[] ids, List<String> piecesOut) {
        BpeTokenizer.StreamDecoder d = tokenizer.newStreamDecoder();
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String p = d.push(id);
            piecesOut.add(p);
            sb.append(p);
        }
        String tail = d.finish();
        piecesOut.add(tail);
        sb.append(tail);
        return sb.toString();
    }

    @Test
    void validTextIsDeliveredWholeCharactersOnly() {
        String text = "héllo wörld 日本語 😀 ok";
        int[] ids = idsOf(text.getBytes(StandardCharsets.UTF_8));
        List<String> pieces = new ArrayList<>();
        String joined = streamAll(ids, pieces);
        assertEquals(text, joined);
        assertEquals(tokenizer.decode(ids), joined);
        for (String p : pieces) {
            assertFalse(p.contains("\uFFFD"), "a piece must never contain half a character: " + p);
        }
        // the 4-byte emoji comes out only when its last byte arrives
        BpeTokenizer.StreamDecoder d = tokenizer.newStreamDecoder();
        int[] emoji = idsOf("😀".getBytes(StandardCharsets.UTF_8));
        assertEquals("", d.push(emoji[0]));
        assertEquals("", d.push(emoji[1]));
        assertEquals("", d.push(emoji[2]));
        assertEquals("😀", d.push(emoji[3]));
        assertEquals("", d.finish());
    }

    @Test
    void streamingEqualsOneShotDecodeOnRandomAndMalformedStreams() {
        Random r = new Random(99);
        byte[][] fragments = {
            "a".getBytes(StandardCharsets.UTF_8), "é".getBytes(StandardCharsets.UTF_8), "日".getBytes(StandardCharsets.UTF_8),
            "😀".getBytes(StandardCharsets.UTF_8), {(byte) 0x80}, {(byte) 0xC0}, {(byte) 0xE6, (byte) 0x97}, {(byte) 0xF0, (byte) 0x9F},
            {(byte) 0xFF}, {(byte) 0xF5, (byte) 0x80}, {(byte) 0xED, (byte) 0xA0, (byte) 0x80}, {0x00}, {0x7F}};
        for (int iter = 0; iter < 20000; iter++) {
            List<Integer> ids = new ArrayList<>();
            int n = r.nextInt(30);
            for (int i = 0; i < n; i++) {
                int kind = r.nextInt(10);
                if (kind == 0) {
                    ids.add(256 + r.nextInt(2));          // BOS / EOS markers are skipped by decode
                } else if (kind == 1) {
                    ids.add(r.nextBoolean() ? -1 : 100000); // out of range
                } else if (kind < 6) {
                    ids.add(r.nextInt(256));              // arbitrary byte
                } else {
                    for (byte b : fragments[r.nextInt(fragments.length)]) {
                        ids.add(b & 0xFF);
                    }
                }
            }
            int[] arr = ids.stream().mapToInt(Integer::intValue).toArray();
            String joined = streamAll(arr, new ArrayList<>());
            assertEquals(tokenizer.decode(arr), joined, "stream " + ids);
        }
    }

    @Test
    void completePrefixLengthHoldsBackOnlyCompletableTails() {
        assertEquals(0, BpeTokenizer.completePrefixLength(new byte[]{(byte) 0xE6, (byte) 0x97}, 2));
        assertEquals(1, BpeTokenizer.completePrefixLength(new byte[]{0x41, (byte) 0xE6, (byte) 0x97}, 3));
        assertEquals(3, BpeTokenizer.completePrefixLength(new byte[]{(byte) 0xE6, (byte) 0x97, (byte) 0xA5}, 3));
        assertEquals(0, BpeTokenizer.completePrefixLength(new byte[]{(byte) 0xF0, (byte) 0x9F, (byte) 0x98}, 3));
        assertEquals(0, BpeTokenizer.completePrefixLength(new byte[]{(byte) 0xC3}, 1));
        assertEquals(1, BpeTokenizer.completePrefixLength(new byte[]{(byte) 0xF5}, 1), "F5 can never start a sequence");
        assertEquals(2, BpeTokenizer.completePrefixLength(new byte[]{(byte) 0x80, (byte) 0x80}, 2), "stray continuation bytes");
        assertEquals(1, BpeTokenizer.completePrefixLength(new byte[]{0x41}, 1));
        assertEquals(0, BpeTokenizer.completePrefixLength(new byte[0], 0));
    }

    // ------------------------------------------------------------------ engine streaming (needs a device)

    private static LlamaGenerationConfig greedy(int maxNew) {
        LlamaGenerationConfig g = new LlamaGenerationConfig();
        g.temperature = 0f;
        g.repetitionPenalty = 1f;
        g.maxNewTokens = maxNew;
        g.eosTokenId = -1;
        return g;
    }

    private static final class Recorder implements TokenListener {
        final List<String> pieces = new ArrayList<>();
        final Thread caller = Thread.currentThread();
        StopReason reason;
        int completeCalls, tokens = -2;
        int cancelAfter = Integer.MAX_VALUE;

        @Override
        public boolean onText(String piece) {
            assertSame(caller, Thread.currentThread(), "callbacks run on the calling thread");
            assertFalse(piece.isEmpty(), "pieces are never empty");
            pieces.add(piece);
            return pieces.size() < cancelAfter;
        }

        @Override
        public void onComplete(StopReason r, int generatedTokens) {
            reason = r;
            tokens = generatedTokens;
            completeCalls++;
        }

        String joined() {
            return String.join("", pieces);
        }
    }

    @Test
    void streamedTextEqualsTheNonStreamedResult() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(plainModel, new GpuContext())) {
            String plain = eng.generate("hello there, this is a test", greedy(10));
            Recorder rec = new Recorder();
            String streamed = eng.generate("hello there, this is a test", greedy(10), rec);
            assertEquals(plain, streamed, "the return value is unchanged by listening");
            assertEquals(plain, rec.joined(), "the pieces joined are the text");
            assertFalse(rec.pieces.isEmpty());
            assertEquals(1, rec.completeCalls);
            assertEquals(StopReason.MAX_TOKENS, rec.reason);
            assertEquals(10, rec.tokens);
            LlamaOpenCLEngine.RunStats st = eng.lastRunStats();
            assertEquals(10, st.generatedTokens);
            assertEquals(10, st.generatedIds.length);
            assertTrue(st.firstPieceMs >= 0 && st.firstPieceMs <= st.prefillMs + st.decodeMs + 50, "first piece time recorded");
            // a null listener is the plain call
            assertEquals(plain, eng.generate("hello there, this is a test", greedy(10), null));
        }
    }

    @Test
    void aStopTokenEndsTheStreamAndIsReported() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(plainModel, new GpuContext())) {
            eng.generate("hello there, this is a test", greedy(8));
            int[] ids = eng.lastRunStats().generatedIds;
            int stopId = ids[3];
            int firstAt = 0;
            while (ids[firstAt] != stopId) {
                firstAt++;
            }
            LlamaGenerationConfig g = greedy(8);
            g.eosTokenId = stopId;
            Recorder rec = new Recorder();
            String text = eng.generate("hello there, this is a test", g, rec);
            assertEquals(StopReason.STOP_TOKEN, rec.reason);
            assertEquals(firstAt, rec.tokens);
            assertEquals(tokenizer.decode(java.util.Arrays.copyOf(ids, firstAt)), text);
            assertEquals(text, rec.joined(), "a character cut off by the stop token is still delivered");
        }
    }

    @Test
    void cancellingStopsPromptlyReturnsWhatWasDeliveredAndLeavesTheEngineUsable() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(plainModel, new GpuContext())) {
            String reference = eng.generate("hello there, this is a test", greedy(12));
            Recorder rec = new Recorder();
            rec.cancelAfter = 1;
            String text = eng.generate("hello there, this is a test", greedy(12), rec);
            assertEquals(StopReason.CANCELLED, rec.reason);
            assertEquals(1, rec.completeCalls);
            assertEquals(1, rec.pieces.size());
            assertEquals(rec.joined(), text, "cancelled: exactly the delivered text");
            assertTrue(rec.tokens < 12, "stopped well before the limit, generated " + rec.tokens);
            assertEquals(reference, eng.generate("hello there, this is a test", greedy(12)), "the engine is reusable and its state was reset");
        }
    }

    @Test
    void anExceptionInTheCallbackPropagatesUnchangedAndTheEngineSurvives() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(plainModel, new GpuContext())) {
            String reference = eng.generate("hello there, this is a test", greedy(8));
            final IllegalArgumentException boom = new IllegalArgumentException("boom");
            Throwable thrown = assertThrows(Throwable.class, () -> eng.generate("hello there, this is a test", greedy(8), piece -> {
                throw boom;
            }));
            assertSame(boom, thrown);
            assertEquals(reference, eng.generate("hello there, this is a test", greedy(8)));
        }
    }

    @Test
    void callingTheEngineFromInsideACallbackIsRefused() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(plainModel, new GpuContext())) {
            String reference = eng.generate("hello there, this is a test", greedy(6));
            Throwable thrown = assertThrows(IllegalStateException.class, () -> eng.generate("x", greedy(6), piece -> {
                try {
                    eng.generate("y", greedy(2));
                } catch (Throwable t) {
                    throw t instanceof RuntimeException re ? re : new RuntimeException(t);
                }
                return true;
            }));
            assertTrue(thrown.getMessage().contains("streaming callback"), thrown.getMessage());
            assertEquals(reference, eng.generate("hello there, this is a test", greedy(6)));
        }
    }

    @Test
    void zeroNewTokensCompletesWithoutText() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(plainModel, new GpuContext())) {
            Recorder rec = new Recorder();
            assertEquals("", eng.generate("hi", greedy(0), rec));
            assertTrue(rec.pieces.isEmpty());
            assertEquals(1, rec.completeCalls);
            assertEquals(StopReason.MAX_TOKENS, rec.reason);
            assertEquals(0, rec.tokens);
        }
    }

    @Test
    void streamingChatEqualsPlainChat() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(chatModel, new GpuContext())) {
            assertTrue(eng.supportsChat());
            LlamaGenerationConfig g = greedy(8); // the tiny model's context is 64 positions and the template alone is about 40
            String plain = eng.chat("s", "hi", g);
            Recorder rec = new Recorder();
            String streamed = eng.chat("s", "hi", g, rec);
            assertEquals(plain, streamed);
            assertEquals(plain, rec.joined());
            Recorder rec2 = new Recorder();
            LlamaGenerationConfig g6 = greedy(6);
            List<ChatTemplate.Message> convo = List.of(new ChatTemplate.Message("user", "hi"), new ChatTemplate.Message("assistant", "ok"),
                    new ChatTemplate.Message("user", "by"));
            String plainConvo = eng.chat(convo, g6);
            assertEquals(plainConvo, eng.chat(convo, g6, rec2));
            assertEquals(plainConvo, rec2.joined());
        }
    }

    @Test
    void backendsWithoutStreamingGetTheDefaultSinglePiece() throws Throwable {
        LlamaGpuEngine fake = new LlamaGpuEngine() {
            @Override
            public String generate(String prompt, LlamaGenerationConfig cfg) {
                return "whole answer";
            }

            @Override
            public String getDeviceDescription() {
                return "fake";
            }

            @Override
            public void close() {
            }
        };
        Recorder rec = new Recorder();
        assertEquals("whole answer", fake.generate("p", greedy(1), rec));
        assertEquals(List.of("whole answer"), rec.pieces);
        assertEquals(StopReason.UNKNOWN, rec.reason);
        assertEquals(-1, rec.tokens);
        Recorder cancel = new Recorder();
        cancel.cancelAfter = 1;
        fake.generate("p", greedy(1), cancel);
        assertEquals(StopReason.CANCELLED, cancel.reason);
    }
}
