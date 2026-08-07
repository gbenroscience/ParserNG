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
 * consumer in this codebase -- see {@link com.github.gbenroscience.gpu.opencl.OpenClDeviceSelector}'s
 * javadoc -- since {@link GpuContext}'s constructor resolves through that
 * selector. Call {@code OpenClDeviceSelector.selectDevice(...)} before
 * constructing this (or before {@code LlamaGpuBridge.load(...)}) to target
 * a specific vendor/device.
 */
public final class OpenCLLlamaEngine implements LlamaGpuEngine {

    private final GpuContext ctx;
    private final LlamaLayer.GpuWeights[] layers;
    private final LlamaLayer.GpuState[] states;
    private final MemorySegment finalNormGammaDevice;
    private final MemorySegment lmHeadDevice;
    private final LlamaOpenCLEngine engine;
    private final String deviceDescription;

    public OpenCLLlamaEngine(File ggufPath) throws Throwable {
        GGUFLoader.GGUFFile gguf = GGUFLoader.load(ggufPath);
        LlamaLayer.Config cfg = LlamaLayer.Config.fromGguf(gguf);

        this.ctx = new GpuContext();
        this.deviceDescription = ctx.selectedDeviceDescription;

        BpeTokenizer tokenizer = BpeTokenizer.fromGguf(gguf);
        EmbeddingTable embeddings = EmbeddingTable.fromGguf(gguf, tokenizer.vocabSize(), cfg.dim);

        this.layers = new LlamaLayer.GpuWeights[cfg.n_layers];
        this.states = new LlamaLayer.GpuState[cfg.n_layers];
        for (int i = 0; i < cfg.n_layers; i++) {
            String prefix = "blk." + i + ".";
            layers[i] = LlamaLayer.GpuWeights.fromGguf(ctx, gguf, prefix);
            states[i] = new LlamaLayer.GpuState(ctx, cfg);
        }

        this.finalNormGammaDevice = loadAndUploadFloat(ctx, gguf, "output_norm.weight");
        this.lmHeadDevice = loadAndUploadFloat(ctx, gguf, findLmHeadTensorName(gguf));

        this.engine = new LlamaOpenCLEngine(
                ctx, cfg, layers, states, embeddings, tokenizer,
                finalNormGammaDevice, lmHeadDevice, tokenizer.vocabSize());
    }

    @Override
    public String generate(String prompt, LlamaGenerationConfig cfg) throws Throwable {
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

    @Override
    public String getDeviceDescription() {
        return deviceDescription;
    }

    @Override
    public void close() {
        // Reverse order of construction -- same rationale as OpenCLDemo's
        // finally block: per-layer state/weights, then the shared
        // final-norm/LM-head buffers, then the engine's own scratch, then
        // the context last (everything else's device memory lives inside it).
        engine.close();
        for (LlamaLayer.GpuState s : states) {
            s.close();
        }
        for (LlamaLayer.GpuWeights w : layers) {
            w.close();
        }
        LlamaLayer.freeQuietly(ctx, finalNormGammaDevice);
        LlamaLayer.freeQuietly(ctx, lmHeadDevice);
        ctx.close();
    }

    private static MemorySegment loadAndUploadFloat(GpuContext ctx, GGUFLoader.GGUFFile gguf, String tensorName) throws Throwable {
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