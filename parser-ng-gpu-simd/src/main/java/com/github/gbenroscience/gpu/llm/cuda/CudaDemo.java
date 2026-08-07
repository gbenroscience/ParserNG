package com.github.gbenroscience.gpu.llm.cuda;

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.io.File;
import java.util.NoSuchElementException;

/**
 * End-to-end wiring example: GGUF file on disk -> tokenized prompt ->
 * generated text. This is the "how do I actually run this" answer for
 * every other file in this package -- nothing here is new machinery,
 * it's just the order everything else needs to be called in.
 *
 * Run with (adjust the classpath/module path for your build):
 *   java -cp <your-classpath> com.github.gbenroscience.gpu.cuda.LlamaCudaDemo /path/to/model.gguf "Your prompt here"
 *
 * REQUIRES, and will fail loudly and immediately if missing:
 *   - An NVIDIA GPU with a working driver (CudaBindings loads libcuda.so.1/nvcuda.dll)
 *   - The CUDA Toolkit installed for NVRTC (NvrtcBindings loads libnvrtc.so/nvrtc64_*.dll
 *     -- the driver alone does NOT ship this; see NvrtcBindings' javadoc)
 *   - A GGUF file using a byte-level BPE tokenizer (tokenizer.ggml.model == "gpt2") --
 *     see BpeTokenizer's class javadoc for why SentencePiece-unigram ("llama") files
 *     aren't supported yet
 *
 * UNVERIFIED, same standing caveat as every file in this package: no
 * CUDA GPU was available while writing this, so this exact call sequence
 * has been traced against every method's actual signature and contract
 * but has not been run.
 */
public final class CudaDemo {

    private CudaDemo() {
    }

    public static void main(String[] args) throws Throwable {
        if (args.length < 2) {
            System.err.println("Usage: CudaDemo <path-to-model.gguf> <prompt>");
            System.exit(1);
        }
        File ggufPath = new File(args[0]);
        String prompt = args[1];

        // === 1. Load the GGUF file (tensors + metadata, both off-heap-mapped) ===
        GGUFLoader.GGUFFile gguf = GGUFLoader.load(ggufPath);

        // === 2. Architecture config, read straight from GGUF metadata ===
        LlamaLayer.Config cfg = LlamaLayer.Config.fromGguf(gguf);
        // cfg.activationType stays at its SWIGLU default here -- override
        // manually first if your model's architecture actually uses
        // GeLU/GeGLU (GGUF has no metadata key for this; see
        // Config.fromGguf's javadoc). E.g.:
        //   cfg.activationType = LlamaLayer.ActivationType.GELU;

        // === 3. Bootstrap CUDA: device selection, NVRTC compile, module load ===
        GpuContext ctx = new GpuContext();

        // === 4. Tokenizer + embedding table, both from GGUF metadata/tensors ===
        BpeTokenizer tokenizer = BpeTokenizer.fromGguf(gguf);
        EmbeddingTable embeddings = EmbeddingTable.fromGguf(gguf, tokenizer.vocabSize(), cfg.dim);

        // === 5. Per-layer weights + per-layer KV-cache/scratch state ===
        LlamaLayer.GpuWeights[] layers = new LlamaLayer.GpuWeights[cfg.n_layers];
        LlamaLayer.GpuState[] states = new LlamaLayer.GpuState[cfg.n_layers];
        for (int i = 0; i < cfg.n_layers; i++) {
            String prefix = "blk." + i + ".";
            layers[i] = LlamaLayer.GpuWeights.fromGguf(ctx, gguf, prefix);
            states[i] = new LlamaLayer.GpuState(ctx, cfg);
        }

        // === 6. Final norm + LM head -- not part of any per-layer GpuWeights,
        // uploaded directly here the same way GpuWeights does internally. ===
        long finalNormGammaDevice = loadAndUploadFloat(ctx, gguf, "output_norm.weight");
        long lmHeadDevice = loadAndUploadFloat(ctx, gguf, findLmHeadTensorName(gguf));

        // === 7. The engine ties it all together ===
        try (LlamaCudaEngine engine = new LlamaCudaEngine(
                ctx, cfg, layers, states, embeddings, tokenizer,
                finalNormGammaDevice, lmHeadDevice, tokenizer.vocabSize())) {

            LlamaCudaEngine.GenerationConfig genCfg = new LlamaCudaEngine.GenerationConfig();
            genCfg.maxNewTokens = 256;
            genCfg.sampler.temperature = 0.8f;
            genCfg.sampler.topK = 40;
            genCfg.sampler.topP = 0.95f;
            genCfg.sampler.repetitionPenalty = 1.1f;
            // genCfg.sampler.temperature = 0f; // uncomment for deterministic greedy decoding instead

            String completion = engine.generate(prompt, genCfg);
            System.out.println(completion);

        } finally {
            // Reverse order of construction: per-layer state/weights, then
            // the shared final-norm/LM-head buffers, then the context last
            // (everything else's device memory lives inside it).
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
    }

    /**
     * Loads a GGUF tensor as float[] (auto-detecting F32 vs Q8_0, same as
     * GpuWeights/EmbeddingTable's internal loadAsFloat -- duplicated here
     * since this tensor doesn't belong to either of those classes) and
     * uploads it to a fresh device buffer.
     */
    private static long loadAndUploadFloat(GpuContext ctx, GGUFLoader.GGUFFile gguf, String tensorName) throws Throwable {
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

    /**
     * Most GGUF files carry a dedicated "output.weight" LM head tensor.
     * Models trained with tied input/output embeddings often omit it
     * entirely and expect the caller to reuse "token_embd.weight" for
     * both -- this checks for the dedicated tensor first and falls back
     * to the embedding table's tensor name if it's absent, rather than
     * failing outright on every tied-embedding model.
     */
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