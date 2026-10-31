package com.github.gbenroscience.gpu.llm.metal;

import com.github.gbenroscience.gpu.llm.LlamaGenerationConfig;
import com.github.gbenroscience.gpu.llm.LlamaGpuEngine;
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.io.File;
import java.util.NoSuchElementException;

/**
 * Metal implementation of {@link LlamaGpuEngine} -- same loading sequence
 * as {@link MetalDemo}'s {@code main()}, packaged as a constructor +
 * fields instead of a linear main() method, so
 * {@code com.github.gbenroscience.gpu.llm.LlamaGpuBridge} can build one
 * without knowing any of these steps individually. Direct counterpart of
 * {@code com.github.gbenroscience.gpu.llm.cuda.CudaLlamaEngine} /
 * {@code com.github.gbenroscience.gpu.llm.opencl.OpenCLLlamaEngine} --
 * identical shape, {@link MetalBuffer} values in place of {@code long}
 * CUdeviceptr / {@code MemorySegment} cl_mem handles.
 *
 * Device selection: {@link GpuContext}'s constructor resolves through
 * {@link MetalDeviceSelector} -- call
 * {@code MetalDeviceSelector.selectDevice(int)} before constructing this
 * (or before {@code LlamaGpuBridge.load(...)}) to target a specific
 * Metal device index when more than one is installed (e.g. a Mac with
 * both an integrated and a discrete/eGPU -- though see
 * {@code MetalDeviceSelector.resolve()}'s javadoc for why this port
 * requires the SELECTED device specifically to have unified memory).
 */
public final class MetalLlamaEngine implements LlamaGpuEngine {

    private final GpuContext ctx;
    private final LlamaLayer.GpuWeights[] layers;
    private final LlamaLayer.GpuState[] states;
    private final MetalBuffer finalNormGammaDevice;
    private final MetalBuffer lmHeadDevice;
    private final LlamaMetalEngine engine;
    private final String deviceDescription;
    private boolean closed;

    /** Everything that needs no device: parsing, full model validation, tokenizer, embeddings. */
    private static final class Prepared {
        final GGUFLoader.GGUFFile gguf;
        final LlamaLayer.Config cfg;
        final BpeTokenizer tokenizer;
        final EmbeddingTable embeddings;

        Prepared(File ggufPath) throws Throwable {
            this.gguf = GGUFLoader.load(ggufPath);
            this.cfg = LlamaLayer.Config.fromGguf(gguf); // throws UnsupportedModelException before any device work
            this.tokenizer = BpeTokenizer.fromGguf(gguf);
            this.embeddings = EmbeddingTable.fromGguf(gguf, tokenizer.vocabSize(), cfg.dim);
        }
    }

    /** Validates the model first (no device touched, so a bad file fails fast), then acquires the device. */
    public MetalLlamaEngine(File ggufPath) throws Throwable {
        this(new Prepared(ggufPath), new GpuContext());
    }

    /**
     * Same, on a caller-supplied context (used by tests). Ownership of {@code context} transfers to
     * this engine once the model has been validated: it is closed if loading fails and by {@link #close()}.
     * If validation itself fails the caller keeps the context.
     */
    MetalLlamaEngine(File ggufPath, GpuContext context) throws Throwable {
        this(new Prepared(ggufPath), context);
    }

    /**
     * Transactional: device acquisition happens inside one try block; if ANY step throws, every
     * resource acquired so far is released in reverse order (cleanup failures are attached to the
     * original exception as suppressed) and the original exception is rethrown.
     */
    private MetalLlamaEngine(Prepared prep, GpuContext c) throws Throwable {
        LlamaLayer.Config cfg = prep.cfg;
        LlamaLayer.GpuWeights[] w = new LlamaLayer.GpuWeights[cfg.n_layers];
        LlamaLayer.GpuState[] st = new LlamaLayer.GpuState[cfg.n_layers];
        MetalBuffer norm = null;
        MetalBuffer head = null;
        LlamaMetalEngine eng = null;
        try {
            for (int i = 0; i < cfg.n_layers; i++) {
                String prefix = "blk." + i + ".";
                w[i] = LlamaLayer.GpuWeights.fromGguf(c, prep.gguf, prefix);
                st[i] = new LlamaLayer.GpuState(c, cfg);
            }
            norm = loadAndUploadFloat(c, prep.gguf, "output_norm.weight");
            head = loadAndUploadFloat(c, prep.gguf, findLmHeadTensorName(prep.gguf));
            eng = new LlamaMetalEngine(c, cfg, w, st, prep.embeddings, prep.tokenizer, norm, head, prep.tokenizer.vocabSize());
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
        this.engine = eng;
    }

    /** Releases in reverse order of acquisition. Never throws; returns the first failure (others suppressed) or null. */
    private static Throwable teardown(GpuContext ctx, LlamaMetalEngine engine, LlamaLayer.GpuState[] states,
            LlamaLayer.GpuWeights[] layers, MetalBuffer norm, MetalBuffer head) {
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
            throw new IllegalStateException("MetalLlamaEngine is closed");
        }
        LlamaMetalEngine.GenerationConfig genCfg = new LlamaMetalEngine.GenerationConfig();
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
            throw new IllegalStateException("failure while releasing GPU resources of MetalLlamaEngine", failure);
        }
    }

    private static MetalBuffer loadAndUploadFloat(GpuContext ctx, GGUFLoader.GGUFFile gguf, String tensorName) throws Throwable {
        GGUFLoader.Tensor t = gguf.tensors.get(tensorName);
        if (t == null) {
            throw new NoSuchElementException("Required GGUF tensor missing: " + tensorName);
        }
        float[] data = switch (t.type) {
            case 0 -> t.loadFloat();
            case 8 -> t.loadQ8_0AsFloat();
            default -> throw new IllegalArgumentException(
                    "Tensor '" + tensorName + "' has GGML type " + t.type + " -- only F32 (0) and Q8_0 (8) are handled here.");
        };
        return LlamaLayer.uploadFloats(ctx, data);
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