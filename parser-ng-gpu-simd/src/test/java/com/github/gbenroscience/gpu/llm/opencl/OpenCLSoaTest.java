package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.core.CpuReference;
import com.github.gbenroscience.gpu.llm.core.TinyModel;
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Conformance of the structure-of-arrays Q8_0 layout: the activation quantiser is byte-exact against its host twin,
 * every lanes-per-row GEMV variant and the prefill GEMM reproduce the CPU reference on awkward shapes, the repack
 * upload is byte-identical to the host repack, scratch sharing frees exactly once, and the layout selection honours
 * its switches. Runs on whatever OpenCL device is selected (-Dopencl.device.type=CPU for PoCL).
 */
class OpenCLSoaTest {

    private static GpuContext ctx;
    private static String skip;

    @BeforeAll
    static void open() {
        try {
            ctx = new GpuContext();
            if (!ctx.soaAvailable) {
                skip = "SoA kernels did not compile on this device";
            }
            ctx.q8Soa = true;
        } catch (Throwable t) {
            skip = "no usable OpenCL device: " + t;
        }
    }

    @AfterAll
    static void close() {
        if (ctx != null) {
            ctx.close();
        }
    }

    private static float[] gaussian(Random r, int n, float scale) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = scale * (float) r.nextGaussian();
        }
        return x;
    }

    /** Device bytes via the float read-back (every layout size here is a multiple of 4). */
    private static byte[] downloadBytes(MemorySegment mem, long bytes) throws Throwable {
        assertEquals(0, bytes % 4);
        float[] f = LlamaLayer.downloadFloats(ctx, mem, (int) (bytes / 4));
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate((int) bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float v : f) {
            bb.putInt(Float.floatToRawIntBits(v));
        }
        return bb.array();
    }

    private static void assertClose(float[] ref, float[] got, double rel, String what) {
        double scale = 1e-6;
        for (float v : ref) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < ref.length; i++) {
            assertFalse(Float.isNaN(got[i]), what + ": NaN at " + i);
            assertTrue(Math.abs(ref[i] - got[i]) <= rel * scale, what + ": element " + i + " expected " + ref[i] + " got " + got[i] + " (scale " + scale + ")");
        }
    }

    @Test
    void activationQuantizerIsByteExactAgainstItsHostTwin() throws Throwable {
        assumeTrue(skip == null, skip);
        Random r = new Random(1);
        for (int len : new int[]{32, 64, 96, 2048, 32 * 17}) {
            float[] x = gaussian(r, len, 3f);
            for (int i = 0; i < 32; i++) {
                x[i] = 0f; // an all-zero block: scale 0
            }
            if (len > 64) {
                x[40] = 1e6f; // wide dynamic range inside a block
                x[41] = 1e-3f;
            }
            MemorySegment dx = LlamaLayer.uploadFloats(ctx, x);
            MemorySegment dq = LlamaLayer.allocBytes(ctx, Q8Layout.activationBytes(true, len));
            try {
                LlamaLayer.quantizeActivationQ8_0(ctx, dx, dq, len);
                byte[] got = downloadBytes(dq, Q8Layout.activationBytes(true, len));
                assertArrayEquals(Q8Layout.packActivationSoa(x), got, "len " + len);
            } finally {
                LlamaLayer.freeQuietly(ctx, dx);
                LlamaLayer.freeQuietly(ctx, dq);
            }
        }
    }

    @Test
    void repackUploadIsByteIdenticalToHostRepack() throws Throwable {
        assumeTrue(skip == null, skip);
        Random r = new Random(2);
        for (int[] nk : new int[][]{{2, 32}, {6, 96}, {38, 2048}}) {
            byte[] aos = GemvSweep.fakeQ8((long) nk[0] * (nk[1] / 32), r);
            MemorySegment dev = Q8Layout.uploadWeight(ctx, aos, nk[0], nk[1]);
            try {
                assertArrayEquals(Q8Layout.repackWeightsSoa(aos, nk[0], nk[1]), downloadBytes(dev, Q8Layout.weightBytes(true, nk[0], nk[1])),
                        "N=" + nk[0] + " K=" + nk[1]);
            } finally {
                LlamaLayer.freeQuietly(ctx, dev);
            }
        }
        assertThrows(IllegalArgumentException.class, () -> Q8Layout.repackWeightsSoa(new byte[34], 1, 33));
        assertThrows(IllegalArgumentException.class, () -> Q8Layout.repackWeightsSoa(new byte[35], 1, 32));
    }

    @Test
    void everyLanesPerRowGemvMatchesCpuReference() throws Throwable {
        assumeTrue(skip == null, skip);
        Random r = new Random(3);
        // K = 32 (a single block), odd block counts, ragged N (not a multiple of rows-per-work-group), and a long row
        int[][] shapes = {{1, 32}, {3, 64}, {130, 96}, {257, 2048}, {64, 32}, {65, 8192}, {1000, 160}, {7, 32 * 5}};
        for (int[] s : shapes) {
            final int N = s[0], K = s[1];
            byte[] aos = GemvSweep.fakeQ8((long) N * (K / 32), r);
            float[] x = gaussian(r, K, 1f);
            float[] ref = CpuReference.q8Gemv(CpuReference.quantizeQ8_0(x), aos, N, K);
            MemorySegment dw = LlamaLayer.uploadBytes(ctx, Q8Layout.repackWeightsSoa(aos, N, K));
            MemorySegment dxf = LlamaLayer.uploadFloats(ctx, x);
            MemorySegment dx = LlamaLayer.allocBytes(ctx, Q8Layout.activationBytes(true, K));
            MemorySegment dout = LlamaLayer.allocFloats(ctx, N);
            try {
                LlamaLayer.quantizeActivationQ8_0(ctx, dxf, dx, K);
                for (int lpr : KernelSourceSoa.GEMV_LPR) {
                    ctx.soaLprByK.put(K, lpr);
                    LlamaLayer.q8_0GemvPlain(ctx, dx, dw, dout, N, K);
                    assertClose(ref, LlamaLayer.downloadFloats(ctx, dout, N), 2e-5, "gemv N=" + N + " K=" + K + " lpr=" + lpr);
                }
            } finally {
                ctx.soaLprByK.remove(K);
                for (MemorySegment m : new MemorySegment[]{dw, dxf, dx, dout}) {
                    LlamaLayer.freeQuietly(ctx, m);
                }
            }
        }
    }

    @Test
    void prefillGemmMatchesCpuReferenceTokenByToken() throws Throwable {
        assumeTrue(skip == null, skip);
        Random r = new Random(4);
        // T around the 16-token chunk, ragged N, odd block counts
        int[][] shapes = {{1, 1, 32}, {5, 7, 64}, {16, 4, 96}, {17, 9, 2048}, {40, 130, 160}, {33, 5, 32 * 7}};
        for (int[] s : shapes) {
            final int T = s[0], N = s[1], K = s[2];
            byte[] aos = GemvSweep.fakeQ8((long) N * (K / 32), r);
            float[] x = gaussian(r, T * K, 1f);
            MemorySegment dw = LlamaLayer.uploadBytes(ctx, Q8Layout.repackWeightsSoa(aos, N, K));
            MemorySegment dxf = LlamaLayer.uploadFloats(ctx, x);
            MemorySegment dx = LlamaLayer.allocBytes(ctx, Q8Layout.activationBytes(true, (long) T * K));
            MemorySegment dout = LlamaLayer.allocFloats(ctx, (long) T * N);
            try {
                LlamaLayer.quantizeActivationQ8_0(ctx, dxf, dx, T * K);
                LlamaLayer.q8_0GemmTiled(ctx, dx, dw, dout, T, N, K);
                float[] got = LlamaLayer.downloadFloats(ctx, dout, T * N);
                for (int t = 0; t < T; t++) {
                    float[] ref = CpuReference.q8Gemv(CpuReference.quantizeQ8_0(java.util.Arrays.copyOfRange(x, t * K, (t + 1) * K)), aos, N, K);
                    assertClose(ref, java.util.Arrays.copyOfRange(got, t * N, (t + 1) * N), 2e-5, "gemm T=" + T + " N=" + N + " K=" + K + " token " + t);
                }
            } finally {
                for (MemorySegment m : new MemorySegment[]{dw, dxf, dx, dout}) {
                    LlamaLayer.freeQuietly(ctx, m);
                }
            }
        }
    }

    private static LlamaLayer.Config config(TinyModel m) throws Exception {
        File f = Files.createTempFile("tiny", ".gguf").toFile();
        f.deleteOnExit();
        m.write(f);
        return LlamaLayer.Config.fromGguf(GGUFLoader.load(f));
    }

    @Test
    void sharedScratchIsAllocatedOnceAndFreedExactlyOnce() throws Throwable {
        assumeTrue(skip == null, skip);
        LlamaLayer.Config cfg = config(new TinyModel().build(9));
        GpuContext c = new GpuContext();
        try {
            long base = c.allocations.liveBytes();
            LlamaLayer.GpuState a = new LlamaLayer.GpuState(c, cfg);
            long afterFirst = c.allocations.liveBytes() - base;
            long kv = c.allocations.byCategory().get("state: KV cache")[0];
            LlamaLayer.GpuState b = new LlamaLayer.GpuState(c, cfg, a);
            LlamaLayer.GpuState d = new LlamaLayer.GpuState(c, cfg, a);
            assertEquals(afterFirst + 2 * kv, c.allocations.liveBytes() - base, "each further layer must add only its own KV cache");
            assertSame(a.xNorm, b.xNorm);
            assertSame(a.cos_table, d.cos_table);
            assertNotSame(a.k_cache_f32, b.k_cache_f32, "KV caches are per layer");

            a.close(); // the layer that allocated the scratch goes first: siblings must keep using it
            assertEquals(afterFirst - kv + 2 * kv, c.allocations.liveBytes() - base);
            a.close(); // idempotent
            b.close();
            d.close();
            assertEquals(0, c.allocations.liveBytes() - base, "the last reference frees the shared scratch");
        } finally {
            c.close();
        }
    }

    @Test
    void sharingScratchBetweenDifferentShapesIsRejected() throws Throwable {
        assumeTrue(skip == null, skip);
        LlamaLayer.Config cfg = config(new TinyModel().build(10));
        LlamaLayer.Config other = config(new TinyModel().build(10));
        other.max_prefill_batch = cfg.max_prefill_batch + 1;
        GpuContext c = new GpuContext();
        try {
            LlamaLayer.GpuState a = new LlamaLayer.GpuState(c, cfg);
            assertThrows(IllegalArgumentException.class, () -> new LlamaLayer.GpuState(c, other, a));
            a.close();
            assertEquals(0, c.allocations.liveBytes());
        } finally {
            c.close();
        }
    }

    @Test
    void layoutSelectionHonoursItsSwitches() throws Throwable {
        assumeTrue(skip == null, skip);
        LlamaLayer.Config cfg = config(new TinyModel().build(11));
        String old = System.getProperty("llm.layout");
        try {
            System.setProperty("llm.layout", "aos");
            GpuContext a = new GpuContext();
            try {
                GemvAutotune.selectLayout(a, cfg);
                assertFalse(a.q8Soa);
            } finally {
                a.close();
            }
            System.setProperty("llm.layout", "soa");
            GpuContext s = new GpuContext();
            try {
                GemvAutotune.selectLayout(s, cfg);
                assertTrue(s.q8Soa, "forced SoA");
                for (int k : new int[]{cfg.dim, cfg.hidden_dim}) {
                    assertTrue(s.soaLprByK.containsKey(k), "a lanes-per-row choice for K=" + k);
                }
            } finally {
                s.close();
            }
            System.setProperty("llm.layout", "bogus");
            assertThrows(IllegalArgumentException.class, Variants::layoutRequest);
        } finally {
            if (old == null) {
                System.clearProperty("llm.layout");
            } else {
                System.setProperty("llm.layout", old);
            }
        }
    }

    @Test
    void engineGeneratesTheSameTextInBothLayouts() throws Throwable {
        assumeTrue(skip == null, skip);
        TinyModel m = new TinyModel();
        m.q8Wo = true;
        m.build(12);
        File f = Files.createTempFile("tiny", ".gguf").toFile();
        f.deleteOnExit();
        m.write(f);
        String old = System.getProperty("llm.layout");
        String[] text = new String[2];
        int i = 0;
        try {
            for (String layout : new String[]{"aos", "soa"}) {
                System.setProperty("llm.layout", layout);
                GpuContext c = new GpuContext();
                try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(f, c)) {
                    assertEquals(layout.equals("soa"), c.q8Soa);
                    com.github.gbenroscience.gpu.llm.LlamaGenerationConfig g = new com.github.gbenroscience.gpu.llm.LlamaGenerationConfig();
                    g.temperature = 0f;
                    g.repetitionPenalty = 1f;
                    g.maxNewTokens = 8;
                    g.eosTokenId = -1;
                    text[i++] = eng.generate("hello there, this is a test", g);
                }
            }
        } finally {
            if (old == null) {
                System.clearProperty("llm.layout");
            } else {
                System.setProperty("llm.layout", old);
            }
        }
        assertEquals(text[0], text[1], "greedy text must not depend on the Q8_0 layout");
    }

    @Test
    void chatReplyMatchesReferenceOverTheTemplatedPromptAndStopsOnEndOfTurn() throws Throwable {
        assumeTrue(skip == null, skip);
        TinyModel m = new TinyModel();
        m.llama3Specials = true;
        m.build(13);
        File f = Files.createTempFile("tiny", ".gguf").toFile();
        f.deleteOnExit();
        m.write(f);
        GGUFLoader.GGUFFile g = GGUFLoader.load(f);
        BpeTokenizer tok = BpeTokenizer.fromGguf(g);
        var msgs = java.util.List.of(new com.github.gbenroscience.gpu.llm.core.ChatTemplate.Message("system", "s"),
                new com.github.gbenroscience.gpu.llm.core.ChatTemplate.Message("user", "hello"));
        int[] prompt = com.github.gbenroscience.gpu.llm.core.ChatTemplate.llama3(tok, msgs);
        assertEquals(256, prompt[0], "BOS from the metadata first");
        assertEquals(java.util.Arrays.stream(prompt).filter(id -> id == 258).count(), 3, "three headers: system, user, open assistant");

        var spec = com.github.gbenroscience.gpu.llm.core.ModelSpec.fromGguf(g);
        var ref = new com.github.gbenroscience.gpu.llm.core.CpuReferenceModel(spec, m);
        float[] logits = null;
        for (int i = 0; i < prompt.length; i++) {
            logits = ref.forward(prompt[i], i);
        }
        int[] gen = new int[4];
        int pos = prompt.length;
        for (int i = 0; i < gen.length; i++) {
            int best = 0;
            for (int t = 1; t < logits.length; t++) {
                if (logits[t] > logits[best]) {
                    best = t;
                }
            }
            gen[i] = best;
            if (i + 1 < gen.length) {
                logits = ref.forward(best, pos++);
            }
        }
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(f, new GpuContext())) {
            assertTrue(eng.supportsChat());
            com.github.gbenroscience.gpu.llm.LlamaGenerationConfig c = new com.github.gbenroscience.gpu.llm.LlamaGenerationConfig();
            c.temperature = 0f;
            c.repetitionPenalty = 1f;
            c.maxNewTokens = 4;
            c.eosTokenId = -1;
            assertEquals(tok.decode(gen), eng.chat(msgs, c), "device chat reply must equal the CPU reference's greedy continuation of the templated prompt");
            // making the first greedy token an end-of-turn id ends the turn immediately
            c.eosTokenId = gen[0];
            assertEquals("", eng.chat(msgs, c));
        }
        // a model without the Llama-3 header tokens refuses chat instead of guessing a format
        TinyModel plain = new TinyModel().build(14);
        File pf = Files.createTempFile("tiny", ".gguf").toFile();
        pf.deleteOnExit();
        plain.write(pf);
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(pf, new GpuContext())) {
            assertFalse(eng.supportsChat());
            assertThrows(UnsupportedOperationException.class, () -> eng.chat(null, "hi", new com.github.gbenroscience.gpu.llm.LlamaGenerationConfig()));
        }
    }
}
