package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.io.File;
import java.lang.foreign.MemorySegment;
import java.util.NoSuchElementException;

/**
 * End-to-end OpenCL GGUF inference demo.
 *
 * Usage:
 *   OpenCLDemo <model.gguf> <prompt...>
 *
 * The prompt is passed as plain text. This demo does not synthesize a Llama-3
 * chat template: the current BpeTokenizer does not yet implement Llama-3
 * special-token parsing / pre-tokenization, so this class must not pretend
 * that adding the visible template strings would make them valid token IDs.
 */
public final class OpenCLDemo {

    private static final String DEFAULT_MODEL =
            "C:\\Users\\GBEMIRO\\Documents\\NetBeansProjects\\ai-models\\llama-3.2-1b-instruct-q8_0.gguf";
    private static final String DEFAULT_PROMPT = "Define turbulent flow in flid mechanics";

    private OpenCLDemo() {
    }

    public static void main(String[] args) throws Throwable {
        if (args.length == 0) {
            args = new String[]{DEFAULT_MODEL, DEFAULT_PROMPT};
        }
        if (args.length < 2) {
            System.err.println("Usage: OpenCLDemo <path-to-model.gguf> <prompt...>");
            System.exit(1);
            return;
        }

        File ggufPath = new File(args[0]);
        if (!ggufPath.isFile()) {
            throw new IllegalArgumentException("GGUF model file does not exist: " + ggufPath.getAbsolutePath());
        }
        String prompt = String.join(" ", java.util.Arrays.copyOfRange(args, 1, args.length));
        System.out.println("Model: " + ggufPath.getAbsolutePath());
        System.out.println("Prompt: " + prompt);
        System.out.println("Loading GGUF model...");

        GGUFLoader.GGUFFile gguf = GGUFLoader.load(ggufPath);
        LlamaLayer.Config cfg = LlamaLayer.Config.fromGguf(gguf);
        System.out.printf("Model config: dim=%d, layers=%d, heads=%d, maxSeq=%d%n",
                cfg.dim, cfg.n_layers, cfg.num_heads, cfg.max_seq);

        // Select the requested device before constructing GpuContext.
        OpenCLDeviceSelector.selectDevice(OpenCLDeviceSelector.GpuVendor.INTEL);
        GpuContext ctx = new GpuContext();

        LlamaLayer.GpuWeights[] layers = null;
        LlamaLayer.GpuState[] states = null;
        MemorySegment finalNormGammaDevice = MemorySegment.NULL;
        MemorySegment lmHeadDevice = MemorySegment.NULL;

        try {
            BpeTokenizer tokenizer = BpeTokenizer.fromGguf(gguf);
            int[] promptIds = tokenizer.encode(prompt, true);
            System.out.printf("Tokenizer: vocab=%d, BOS=%d, EOS=%d, promptTokens=%d%n",
                    tokenizer.vocabSize(), tokenizer.getBosId(), tokenizer.getEosId(), promptIds.length);
            if (promptIds.length == 0) {
                throw new IllegalStateException("Prompt tokenized to zero tokens");
            }

            EmbeddingTable embeddings = EmbeddingTable.fromGguf(gguf, tokenizer.vocabSize(), cfg.dim);
            layers = new LlamaLayer.GpuWeights[cfg.n_layers];
            states = new LlamaLayer.GpuState[cfg.n_layers];
            for (int i = 0; i < cfg.n_layers; i++) {
                String prefix = "blk." + i + ".";
                layers[i] = LlamaLayer.GpuWeights.fromGguf(ctx, gguf, prefix);
                states[i] = new LlamaLayer.GpuState(ctx, cfg);
                System.out.printf("Uploaded layer %d/%d%n", i + 1, cfg.n_layers);
            }

            finalNormGammaDevice = loadAndUploadFloat(ctx, gguf, "output_norm.weight");
            lmHeadDevice = loadAndUploadFloat(ctx, gguf, findLmHeadTensorName(gguf));

            try (LlamaOpenCLEngine engine = new LlamaOpenCLEngine(
                    ctx, cfg, layers, states, embeddings, tokenizer,
                    finalNormGammaDevice, lmHeadDevice, tokenizer.vocabSize())) {

                LlamaOpenCLEngine.GenerationConfig genCfg = new LlamaOpenCLEngine.GenerationConfig();
                // One token is only a smoke test and commonly produces an
                // apparently empty answer when the first sampled token is EOS.
                genCfg.maxNewTokens = 64;
                genCfg.sampler.temperature = 0.0f; // deterministic greedy decoding
                genCfg.sampler.topK = 1;
                genCfg.sampler.topP = 1.0f;
                genCfg.sampler.repetitionPenalty = 1.0f;

                System.out.println("Running inference...");
                String completion = engine.generate(prompt, genCfg);
                if (completion == null || completion.isBlank()) {
                    System.out.println("Generated text: <empty>");
                    System.out.println("The engine returned no visible text. Check the tokenizer/model compatibility warning below.");
                } else {
                    System.out.println("Generated text:");
                    System.out.println(completion);
                }
            }
        } finally {
            if (states != null) {
                for (LlamaLayer.GpuState state : states) {
                    if (state != null) state.close();
                }
            }
            if (layers != null) {
                for (LlamaLayer.GpuWeights weights : layers) {
                    if (weights != null) weights.close();
                }
            }
            LlamaLayer.freeQuietly(ctx, finalNormGammaDevice);
            LlamaLayer.freeQuietly(ctx, lmHeadDevice);
            ctx.close();
        }

        System.out.println();
        System.out.println("Compatibility note: this demo's current BPE implementation is not yet a verified Llama-3.2 tokenizer.");
        System.out.println("Increasing maxNewTokens prevents the one-token smoke-test issue, but it does not fix tokenizer/pre-tokenizer or chat-template incompatibility.");
    }

    private static MemorySegment loadAndUploadFloat(
            GpuContext ctx, GGUFLoader.GGUFFile gguf, String tensorName) throws Throwable {
        GGUFLoader.Tensor tensor = gguf.tensors.get(tensorName);
        if (tensor == null) {
            throw new NoSuchElementException("Required GGUF tensor missing: " + tensorName);
        }
        float[] data = switch (tensor.type) {
            case 0 -> tensor.loadFloat();
            case 8 -> tensor.loadQ8_0AsFloat();
            default -> throw new IllegalArgumentException(
                    "Tensor '" + tensorName + "' has GGML type " + tensor.type
                    + " -- only F32 (0) and Q8_0 (8) are handled here.");
        };
        return LlamaLayer.uploadFloats(ctx, data);
    }

    private static String findLmHeadTensorName(GGUFLoader.GGUFFile gguf) {
        if (gguf.tensors.containsKey("output.weight")) return "output.weight";
        if (gguf.tensors.containsKey("token_embd.weight")) return "token_embd.weight";
        throw new NoSuchElementException(
                "Neither output.weight nor token_embd.weight was found in the GGUF tensors");
    }
}
