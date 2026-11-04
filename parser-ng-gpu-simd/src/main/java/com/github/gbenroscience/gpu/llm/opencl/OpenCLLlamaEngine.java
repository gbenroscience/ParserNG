package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.LlamaGenerationConfig;
import com.github.gbenroscience.gpu.llm.LlamaGpuEngine;
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.io.File;
import java.lang.foreign.MemorySegment;
import java.util.NoSuchElementException;

/**
 * OpenCL implementation of {@link LlamaGpuEngine} -- same loading sequence
 * as {@link OpenCLDemo}'s {@code main()} (GGUF -> config -> GpuContext ->
 * tokenizer/embeddings -> per-layer weights/state -> final norm/LM head ->
 * {@link LlamaOpenCLEngine}), just packaged as a constructor + fields
 * instead of a linear main() method, so
 * {@code com.github.gbenroscience.gpu.llm.LlamaGpuBridge} can build one
 * without knowing any of these steps individually.
 *
 * Device selection happens exactly as it does for every other OpenCL
 * consumer in this codebase -- see {@link OpenCLDeviceSelector} (this package's own)'s
 * javadoc -- since {@link GpuContext}'s constructor resolves through that
 * selector. Call {@code OpenCLDeviceSelector.selectDevice(...)} before
 * constructing this (or before {@code LlamaGpuBridge.load(...)}) to target
 * a specific vendor/device.
 */
public final class OpenCLLlamaEngine implements LlamaGpuEngine {

    private final GpuContext ctx;
    private final LlamaLayer.GpuWeights[] layers;
    private final LlamaLayer.GpuState[] states;
    private final MemorySegment finalNormGammaDevice;
    private final MemorySegment lmHeadDevice;
    private final boolean lmHeadQ8;
    private final LlamaOpenCLEngine engine;
    private final String deviceDescription;
    private boolean closed;

    /** Everything that needs no device: parsing, full model validation, tokenizer, embeddings. */
    private static final class Prepared {
        final GGUFLoader.GGUFFile gguf;
        final LlamaLayer.Config cfg;
        final BpeTokenizer tokenizer;
        final EmbeddingTable embeddings;

        Prepared(File ggufPath) throws Throwable {
            long t = LoadTimer.start();
            this.gguf = GGUFLoader.load(ggufPath);
            LoadTimer.stop("host: GGUF parse (header, metadata, tensor table)", t);
            t = LoadTimer.start();
            this.cfg = LlamaLayer.Config.fromGguf(gguf); // throws UnsupportedModelException before any device work
            LoadTimer.stop("host: model validation (ModelSpec)", t);
            // The KV cache, RoPE tables and score buffers are allocated for max_seq positions in EVERY layer.
            // Models advertise very long contexts (Llama-3.2: 131072), which would reserve gigabytes of device
            // memory up front; cap it (override with -Dllm.maxContext=N). generate() refuses longer sequences.
            final int cap = Integer.getInteger("llm.maxContext", 8192);
            if (cap <= 0) {
                throw new IllegalArgumentException("llm.maxContext must be > 0, got " + cap);
            }
            if (cfg.max_seq > cap) {
                LoadTimer.note("context length capped from " + cfg.max_seq + " to " + cap + " positions (-Dllm.maxContext=N to change; KV cache memory scales linearly with it)");
                cfg.max_seq = cap;
            }
            t = LoadTimer.start();
            this.tokenizer = BpeTokenizer.fromGguf(gguf);
            LoadTimer.stop("host: tokenizer build", t);
            t = LoadTimer.start();
            this.embeddings = EmbeddingTable.fromGguf(gguf, tokenizer.vocabSize(), cfg.dim);
            LoadTimer.stop("host: embedding table (Q8_0 kept quantised, rows dequantised on demand)", t);
        }
    }

    /** Validates the model first (no device touched, so a bad file fails fast), then acquires the device. */
    public OpenCLLlamaEngine(File ggufPath) throws Throwable {
        this(new Prepared(ggufPath), new GpuContext());
    }

    /**
     * Same, on a caller-supplied context (used by tests). Ownership of {@code context} transfers to
     * this engine once the model has been validated: it is closed if loading fails and by {@link #close()}.
     * If validation itself fails the caller keeps the context.
     */
    OpenCLLlamaEngine(File ggufPath, GpuContext context) throws Throwable {
        this(new Prepared(ggufPath), context);
    }

    /**
     * Transactional: device acquisition happens inside one try block; if ANY step throws, every
     * resource acquired so far is released in reverse order (cleanup failures are attached to the
     * original exception as suppressed) and the original exception is rethrown.
     */
    private OpenCLLlamaEngine(Prepared prep, GpuContext c) throws Throwable {
        LlamaLayer.Config cfg = prep.cfg;
        LlamaLayer.GpuWeights[] w = new LlamaLayer.GpuWeights[cfg.n_layers];
        LlamaLayer.GpuState[] st = new LlamaLayer.GpuState[cfg.n_layers];
        MemorySegment norm = null;
        MemorySegment head = null;
        boolean headQ8 = false;
        LlamaOpenCLEngine eng = null;
        try {
            long weightBytes = 0, stateBytes = 0;
            for (int i = 0; i < cfg.n_layers; i++) {
                String prefix = "blk." + i + ".";
                long b0 = c.allocations.liveBytes();
                w[i] = LlamaLayer.GpuWeights.fromGguf(c, prep.gguf, prefix);
                long b1 = c.allocations.liveBytes();
                long t = LoadTimer.start();
                st[i] = new LlamaLayer.GpuState(c, cfg);
                LoadTimer.stop("layers: state allocation (KV cache, RoPE tables, scratch)", t);
                weightBytes += b1 - b0;
                stateBytes += c.allocations.liveBytes() - b1;
            }
            long bHead0 = c.allocations.liveBytes();
            c.allocCategory = "final norm gamma";
            norm = loadAndUploadFloat(c, prep.gguf, "output_norm.weight");
            c.allocCategory = "lm head";
            LmHead lh = uploadLmHead(c, prep.gguf);
            head = lh.device;
            headQ8 = lh.q8;
            long headBytes = c.allocations.liveBytes() - bHead0;
            if (Variants.autotuneGemv() && !Variants.legacyGemv()) {
                long tTune = LoadTimer.start();
                c.allocCategory = "autotune scratch (freed)";
                GemvAutotune.run(c, cfg.dim, Math.min(cfg.hidden_dim, 4096));
                LoadTimer.stop("opencl: decode GEMV autotune", tTune);
            }
            if (Variants.autotuneGemm() && !Variants.legacyGemm() && !Variants.forcedRowsGemm()) {
                long tTune = LoadTimer.start();
                c.allocCategory = "autotune scratch (freed)";
                GemvAutotune.runGemm(c, cfg.dim, Math.min(cfg.hidden_dim, 2048));
                LoadTimer.stop("opencl: prefill GEMM autotune", tTune);
            }
            long kvPerLayer = 2L * cfg.max_seq * cfg.kv_heads * cfg.head_dim * 4L;
            StringBuilder cat = new StringBuilder("device memory by category (live buffers, all layers):\n");
            for (java.util.Map.Entry<String, long[]> e : c.allocations.byCategory().entrySet()) {
                cat.append(String.format("  %-44s %9.1f MB  (%d buffers, %4.1f%%)%n", e.getKey(), e.getValue()[0] / 1e6, e.getValue()[1],
                        100.0 * e.getValue()[0] / Math.max(c.allocations.liveBytes(), 1)));
            }
            LoadTimer.note(cat.toString().stripTrailing());
            LoadTimer.note(String.format("device memory after load: %.1f MB live (peak %.1f MB) = layer weights %.1f MB + per-layer state %.1f MB "
                    + "(of which KV cache %.1f MB: %.1f MB/layer at %d positions) + final norm/LM head %.1f MB (%s)",
                    c.allocations.liveBytes() / 1e6, c.allocations.peakBytes() / 1e6, weightBytes / 1e6, stateBytes / 1e6,
                    kvPerLayer * cfg.n_layers / 1e6, kvPerLayer / 1e6, cfg.max_seq, headBytes / 1e6,
                    headQ8 ? "LM head kept as Q8_0" : "LM head dequantised to F32"));
            eng = new LlamaOpenCLEngine(c, cfg, w, st, prep.embeddings, prep.tokenizer, norm, head, headQ8, prep.tokenizer.vocabSize());
        } catch (Throwable failure) {
            Throwable cleanup = teardown(c, eng, st, w, norm, head);
            if (cleanup != null) {
                failure.addSuppressed(cleanup);
            }
            throw failure;
        }
        this.ctx = c;
        this.deviceDescription = c.selectedDeviceDescription;
        this.layers = w;
        this.states = st;
        this.finalNormGammaDevice = norm;
        this.lmHeadDevice = head;
        this.lmHeadQ8 = headQ8;
        this.engine = eng;
    }

    /** Releases in reverse order of acquisition. Never throws; returns the first failure (others suppressed) or null. */
    private static Throwable teardown(GpuContext ctx, LlamaOpenCLEngine engine, LlamaLayer.GpuState[] states,
            LlamaLayer.GpuWeights[] layers, MemorySegment norm, MemorySegment head) {
        Throwable[] first = new Throwable[1];
        if (engine != null) {
            step(first, engine::close);
        }
        for (LlamaLayer.GpuState s : states) {
            if (s != null) {
                step(first, s::close);
            }
        }
        for (LlamaLayer.GpuWeights w : layers) {
            if (w != null) {
                step(first, w::close);
            }
        }
        LlamaLayer.freeQuietly(ctx, norm);
        LlamaLayer.freeQuietly(ctx, head);
        step(first, ctx::close);
        return first[0];
    }

    private static void step(Throwable[] first, Runnable r) {
        try {
            r.run();
        } catch (Throwable t) {
            if (first[0] == null) {
                first[0] = t;
            } else {
                first[0].addSuppressed(t);
            }
        }
    }

    @Override
    public synchronized String generate(String prompt, LlamaGenerationConfig cfg) throws Throwable {
        if (closed) {
            throw new IllegalStateException("OpenCLLlamaEngine is closed");
        }
        LlamaOpenCLEngine.GenerationConfig genCfg = new LlamaOpenCLEngine.GenerationConfig();
        genCfg.maxNewTokens = cfg.maxNewTokens;
        genCfg.eosTokenId = cfg.eosTokenId;
        genCfg.sampler.temperature = cfg.temperature;
        genCfg.sampler.topK = cfg.topK;
        genCfg.sampler.topP = cfg.topP;
        genCfg.sampler.repetitionPenalty = cfg.repetitionPenalty;
        genCfg.sampler.repetitionPenaltyWindow = cfg.repetitionPenaltyWindow;
        genCfg.sampler.seed = cfg.seed;
        return engine.generate(prompt, genCfg);
    }

    /** Wall-clock timings of the most recent {@link #generate} call. */
    public LlamaOpenCLEngine.RunStats lastRunStats() {
        return engine.lastRunStats();
    }

    @Override
    public String getDeviceDescription() {
        return deviceDescription;
    }

    /** Idempotent. Throws IllegalStateException (first failure as cause, others suppressed) if any resource failed to release. */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        Throwable failure = teardown(ctx, engine, states, layers, finalNormGammaDevice, lmHeadDevice);
        if (failure != null) {
            throw new IllegalStateException("failure while releasing GPU resources of OpenCLLlamaEngine", failure);
        }
    }

    private static MemorySegment loadAndUploadFloat(GpuContext ctx, GGUFLoader.GGUFFile gguf, String tensorName) throws Throwable {
        GGUFLoader.Tensor t = gguf.tensors.get(tensorName);
        if (t == null) {
            throw new NoSuchElementException("Required GGUF tensor missing: " + tensorName);
        }
        final long tRead = LoadTimer.start();
        float[] data = switch (t.type) {
            case 0 -> t.loadFloat();
            case 8 -> t.loadQ8_0AsFloat();
            default -> throw new IllegalArgumentException(
                    "Tensor '" + tensorName + "' has GGML type " + t.type + " -- only F32 (0) and Q8_0 (8) are handled here.");
        };
        LoadTimer.stop("final norm / LM head: read + dequantize to float[]", tRead);
        final long tUp = LoadTimer.start();
        MemorySegment dev = LlamaLayer.uploadFloats(ctx, data);
        LoadTimer.stop("final norm / LM head: device upload", tUp);
        return dev;
    }

    /** Device copy of the output projection and its format. */
    static final class LmHead {

        final MemorySegment device;
        final boolean q8;

        LmHead(MemorySegment device, boolean q8) {
            this.device = device;
            this.q8 = q8;
        }
    }

    /**
     * Uploads the LM head. A Q8_0 head (the tied token embedding of a Q8_0 model, or output.weight) is kept as Q8_0
     * blocks -- a quarter of the bytes read per token compared with dequantising to F32 -- unless
     * {@code -Dllm.lmhead=f32} asks for the old F32 form. Any other tensor type goes through the F32 path.
     */
    static LmHead uploadLmHead(GpuContext ctx, GGUFLoader.GGUFFile gguf) throws Throwable {
        String name = findLmHeadTensorName(gguf);
        GGUFLoader.Tensor t = gguf.tensors.get(name);
        if (t != null && t.type == 8 && !"f32".equalsIgnoreCase(System.getProperty("llm.lmhead"))) {
            long t0 = LoadTimer.start();
            byte[] q8 = t.loadQ8_0();
            LoadTimer.stop("final norm / LM head: read Q8_0 tensor", t0);
            long t1 = LoadTimer.start();
            MemorySegment dev = LlamaLayer.uploadBytes(ctx, q8);
            LoadTimer.stop("final norm / LM head: device upload", t1);
            return new LmHead(dev, true);
        }
        return new LmHead(loadAndUploadFloat(ctx, gguf, name), false);
    }

    private static String findLmHeadTensorName(GGUFLoader.GGUFFile gguf) {
        if (gguf.tensors.containsKey("output.weight")) {
            return "output.weight";
        }
        if (gguf.tensors.containsKey("token_embd.weight")) {
            return "token_embd.weight"; // tied embeddings
        }
        throw new NoSuchElementException(
                "Neither \"output.weight\" nor \"token_embd.weight\" (tied-embedding fallback) found in GGUF tensors");
    }
}