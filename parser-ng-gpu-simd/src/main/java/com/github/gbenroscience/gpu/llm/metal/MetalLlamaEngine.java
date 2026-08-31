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

    public MetalLlamaEngine(File ggufPath) throws Throwable {
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

        this.engine = new LlamaMetalEngine(
                ctx, cfg, layers, states, embeddings, tokenizer,
                finalNormGammaDevice, lmHeadDevice, tokenizer.vocabSize());
    }

    @Override
    public String generate(String prompt, LlamaGenerationConfig cfg) throws Throwable {
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

    @Override
    public void close() {
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