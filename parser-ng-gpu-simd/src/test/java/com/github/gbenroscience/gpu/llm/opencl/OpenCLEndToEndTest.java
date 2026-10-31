package com.github.gbenroscience.gpu.llm.opencl;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.github.gbenroscience.gpu.llm.LlamaGenerationConfig;
import com.github.gbenroscience.gpu.llm.core.*;
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;
import java.io.File;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * End-to-end checks of the OpenCL engine against {@link CpuReferenceModel}.
 * Runs on whatever OpenCL device is visible (printed); SKIPPED when none.
 * A PoCL/CPU device proves the OpenCL kernels and host logic, NOT GPU hardware.
 */
class OpenCLEndToEndTest {

    /** Documented tolerance for logits. Measured worst case on PoCL/CPU over 153 vectors: abs 5.7e-6, rel 6.8e-4 (near-zero logits). Leaves ~17x headroom for GPU fast-math/FMA differences. */
    static final double ABS_TOL = 1e-4;
    static final double REL_TOL = 1e-3;

    private static GpuContext shared;
    private static double worstAbs, worstRel, worstRms;
    private static int comparedVectors;
    private static String skip;

    @BeforeAll
    static void open() {
        try {
            shared = new GpuContext();
            System.out.println("[OpenCLEndToEndTest] device = " + shared.selectedDeviceDescription);
        } catch (Throwable t) {
            skip = "no usable OpenCL device: " + t;
        }
    }

    @AfterAll
    static void close() {
        System.out.printf("[OpenCLEndToEndTest] %d logit vectors compared; worst maxAbs=%.3e worst maxRel=%.3e worst rms=%.3e%n", comparedVectors, worstAbs, worstRel, worstRms);
        if (shared != null) {
            shared.close();
        }
    }

    private static TinyModel tiny(long seed) {
        return new TinyModel().build(seed);
    }

    private static File write(TinyModel m) throws Exception {
        File f = Files.createTempFile("tiny", ".gguf").toFile();
        f.deleteOnExit();
        return m.write(f);
    }

    /** Drives the real engine layers exactly like LlamaOpenCLEngine does, returning logits after every fed token. */
    private static float[][] deviceLogits(GpuContext ctx, GGUFLoader.GGUFFile g, LlamaLayer.Config cfg, TinyModel m, int[] ids, int prefillTokens) throws Throwable {
        LlamaLayer.GpuWeights[] w = new LlamaLayer.GpuWeights[cfg.n_layers];
        LlamaLayer.GpuState[] s = new LlamaLayer.GpuState[cfg.n_layers];
        MemorySegment norm = null, head = null, x = null, xb = null;
        try {
            for (int l = 0; l < cfg.n_layers; l++) {
                w[l] = LlamaLayer.GpuWeights.fromGguf(ctx, g, "blk." + l + ".");
                s[l] = new LlamaLayer.GpuState(ctx, cfg);
            }
            norm = LlamaLayer.uploadFloats(ctx, m.truth.get("output_norm.weight"));
            head = LlamaLayer.uploadFloats(ctx, m.truth.get("output.weight"));
            EmbeddingTable emb = new EmbeddingTable(m.truth.get("token_embd.weight"), m.vocab, m.dim);
            x = LlamaLayer.allocFloats(ctx, m.dim);
            float[][] out = new float[ids.length][];

            int pre = Math.min(prefillTokens, ids.length);
            if (pre > 0) {
                xb = LlamaLayer.allocFloats(ctx, (long) pre * m.dim);
                emb.embedRows(java.util.Arrays.copyOf(ids, pre), xb, ctx);
                for (int l = 0; l < cfg.n_layers; l++) {
                    LlamaLayer.prefill_layer(xb, w[l], s[l], cfg, ctx, 0, pre);
                }
                // The batched path yields hidden states for all `pre` rows; logits of every prefill row are
                // checked by re-reading each row (this validates per-row causal attention, not just the last).
                float[] all = LlamaLayer.downloadFloats(ctx, xb, pre * m.dim);
                for (int r = 0; r < pre; r++) {
                    MemorySegment row = LlamaLayer.uploadFloats(ctx, java.util.Arrays.copyOfRange(all, r * m.dim, (r + 1) * m.dim));
                    try {
                        out[r] = LlamaLayer.finalLogits(row, norm, head, cfg, ctx, m.vocab);
                    } finally {
                        LlamaLayer.freeQuietly(ctx, row);
                    }
                }
            }
            for (int i = pre; i < ids.length; i++) {
                emb.embedRow(ids[i], x, ctx);
                for (int l = 0; l < cfg.n_layers; l++) {
                    LlamaLayer.forward_layer(x, w[l], s[l], cfg, ctx);
                }
                out[i] = LlamaLayer.finalLogits(x, norm, head, cfg, ctx, m.vocab);
            }
            return out;
        } finally {
            for (LlamaLayer.GpuState st : s) {
                if (st != null) {
                    st.close();
                }
            }
            for (LlamaLayer.GpuWeights wt : w) {
                if (wt != null) {
                    wt.close();
                }
            }
            LlamaLayer.freeQuietly(ctx, norm);
            LlamaLayer.freeQuietly(ctx, head);
            LlamaLayer.freeQuietly(ctx, x);
            LlamaLayer.freeQuietly(ctx, xb);
        }
    }

    private void compareAllPaths(TinyModel m, int promptLen, int extra) throws Throwable {
        assumeTrue(shared != null, skip);
        File f = write(m);
        GGUFLoader.GGUFFile g = GGUFLoader.load(f);
        ModelSpec spec = ModelSpec.fromGguf(g);
        LlamaLayer.Config cfg = LlamaLayer.Config.fromGguf(g);
        int n = promptLen + extra;
        int[] ids = new int[n];
        java.util.Random r = new java.util.Random(77 + n);
        for (int i = 0; i < n; i++) {
            ids[i] = r.nextInt(256);
        }
        CpuReferenceModel ref = new CpuReferenceModel(spec, m);
        float[][] expect = new float[n][];
        for (int i = 0; i < n; i++) {
            expect[i] = ref.forward(ids[i], i);
        }
        // decode-only, full prefill, and a chunk boundary in the middle of the prompt
        int[] prefillOptions = {0, promptLen, Math.max(1, promptLen / 2)};
        for (int pre : prefillOptions) {
            cfg.max_prefill_batch = Math.max(pre, 1);
            float[][] got = deviceLogits(shared, g, cfg, m, ids, pre);
            for (int i = 0; i < n; i++) {
                ErrorStats st = ErrorStats.compare(got[i], expect[i], ABS_TOL, REL_TOL);
                worstAbs = Math.max(worstAbs, st.maxAbsError);
                worstRel = Math.max(worstRel, st.maxRelError);
                worstRms = Math.max(worstRms, st.rmsError);
                comparedVectors++;
                assertTrue(st.passed(), "promptLen=" + promptLen + " prefill=" + pre + " position " + i + ": " + st);
            }
        }
    }

    @Test
    void logitsMatchReferenceForGqaModelAcrossDecodeAndPrefillPaths() throws Throwable {
        compareAllPaths(tiny(1), 1, 3);
        compareAllPaths(tiny(2), 7, 4);
        compareAllPaths(tiny(3), 12, 3);
    }

    @Test
    void logitsMatchReferenceForMhaAndOddShapes() throws Throwable {
        TinyModel mha = new TinyModel();
        mha.heads = 4;
        mha.kvHeads = 4; // plain multi-head
        mha.build(4);
        compareAllPaths(mha, 6, 3);
        TinyModel odd = new TinyModel();
        odd.dim = 96; // head_dim 24 (not a power of two), hidden 160
        odd.hidden = 160;
        odd.heads = 4;
        odd.kvHeads = 1; // multi-query
        odd.layers = 3;
        odd.build(5);
        compareAllPaths(odd, 9, 3);
    }

    @Test
    void engineGenerateIsDeterministicRepeatableAndMatchesReferenceGreedy() throws Throwable {
        assumeTrue(skip == null, skip);
        TinyModel m = tiny(6);
        File f = write(m);
        GpuContext ctx = new GpuContext();
        OpenCLLlamaEngine eng = new OpenCLLlamaEngine(f, ctx);
        try {
            LlamaGenerationConfig c = new LlamaGenerationConfig();
            c.temperature = 0f; // greedy
            c.repetitionPenalty = 1f;
            c.maxNewTokens = 6;
            c.eosTokenId = -1;
            String a = eng.generate("hello there", c);
            String b = eng.generate("hello there", c);
            assertEquals(a, b, "a second generate() on the same engine must start a fresh sequence");

            // independent greedy reference over the same prompt ids
            ModelSpec spec = ModelSpec.fromGguf(GGUFLoader.load(f));
            BpeTokenizer tok = BpeTokenizer.fromGguf(GGUFLoader.load(f));
            int[] prompt = tok.encode("hello there", true);
            CpuReferenceModel ref = new CpuReferenceModel(spec, m);
            float[] logits = null;
            for (int i = 0; i < prompt.length; i++) {
                logits = ref.forward(prompt[i], i);
            }
            int[] gen = new int[c.maxNewTokens];
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
            assertEquals(tok.decode(gen), a, "device greedy generation must equal the CPU reference's greedy generation");

            // context overflow is refused up front, before touching the device
            LlamaGenerationConfig big = new LlamaGenerationConfig();
            big.maxNewTokens = 1000;
            assertThrows(IllegalArgumentException.class, () -> eng.generate("hello", big));
        } finally {
            eng.close();
            eng.close(); // idempotent
        }
        assertEquals(0, ctx.sweptAtClose, "normal close must release every buffer explicitly");
        assertThrows(IllegalStateException.class, () -> eng.generate("x", new LlamaGenerationConfig()));
    }

    @Test
    void failedLoadReleasesEverythingAcquiredSoFar() throws Throwable {
        assumeTrue(skip == null, skip);
        TinyModel m = tiny(7);
        m.writer.metadata.put("llama.context_length", Integer.MAX_VALUE); // layer 0 weights upload fine, then the KV cache allocation fails
        File f = write(m);
        GpuContext ctx = new GpuContext();
        Throwable t = assertThrows(Throwable.class, () -> new OpenCLLlamaEngine(f, ctx));
        System.out.println("[failedLoad] expected failure: " + t);
        assertEquals(0, ctx.sweptAtClose, "every buffer acquired before the failure must have been released explicitly");
        assertEquals(0, ctx.allocations.size());
        ctx.close(); // already closed by the failed constructor: must be a no-op
    }

    @Test
    void unsupportedModelFailsBeforeAnyDeviceWork() throws Throwable {
        assumeTrue(skip == null, skip);
        TinyModel m = tiny(8);
        m.writer.metadata.put("general.architecture", "qwen2");
        File f = write(m);
        GpuContext ctx = new GpuContext();
        try {
            assertThrows(UnsupportedModelException.class, () -> new OpenCLLlamaEngine(f, ctx));
            assertEquals(-1, ctx.sweptAtClose, "the caller keeps ownership of the context when validation fails");
            assertEquals(0, ctx.allocations.size());
        } finally {
            ctx.close();
        }
    }

    @Test
    void tokenizerRoundTripAndByteLevelEncoding() throws Throwable {
        TinyModel m = tiny(9);
        BpeTokenizer tok = BpeTokenizer.fromGguf(GGUFLoader.load(write(m)));
        int[] ids = tok.encode("h\u00e9!", true);
        // BOS, 'h', then U+00E9 as its two UTF-8 bytes C3 A9, then '!'
        assertArrayEquals(new int[]{256, 'h', 0xC3, 0xA9, '!'}, ids);
        assertEquals("h\u00e9!", tok.decode(ids));
    }
}
