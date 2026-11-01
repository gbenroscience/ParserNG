package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.LlamaGenerationConfig;
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.io.File;

/**
 * End-to-end OpenCL GGUF inference demo.
 *
 * Usage:
 *   OpenCLDemo [model.gguf] [prompt...]
 *
 * System properties:
 *   -Dopencl.vendor=INTEL|NVIDIA|AMD|...   device to use (default INTEL; use -Dopencl.device.type=CPU for a CPU device)
 *   -Dllm.timing=true                      (on by default here) time-to-first-token and decode tokens/s
 *   -Dllm.profile=true                     per-stage time breakdown; serialises the pipeline, so use it to
 *                                          find hotspots, not to measure speed
 *   -Dllm.printTokens=true                 print the prompt token ids, for diffing against `llama-tokenize`
 *
 * The prompt is passed as plain text. This demo does not apply a chat template, and the current
 * BpeTokenizer uses the GPT-2 pre-tokenizer regex rather than the one Llama-3 models declare in
 * tokenizer.ggml.pre, so prompts containing unusual whitespace, contractions or long digit runs may
 * tokenize differently from llama.cpp. Treat output quality accordingly.
 */
public final class OpenCLDemo {

    private static final String DEFAULT_MODEL =
            "C:\\Users\\GBEMIRO\\Documents\\NetBeansProjects\\ai-models\\llama-3.2-1b-instruct-q8_0.gguf";
    //private static final String DEFAULT_PROMPT = "What is 23*34? Answer briefly.";
    private static final String DEFAULT_PROMPT = "Explain photosynthesis";

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
        System.setProperty("llm.timing", System.getProperty("llm.timing", "true"));

        File ggufPath = new File(args[0]);
        if (!ggufPath.isFile()) {
            throw new IllegalArgumentException("GGUF model file does not exist: " + ggufPath.getAbsolutePath());
        }
        String prompt = String.join(" ", java.util.Arrays.copyOfRange(args, 1, args.length));
        System.out.println("Model: " + ggufPath.getAbsolutePath());
        System.out.println("Prompt: " + prompt);

        // Device selection: honoured by GpuContext, which the engine creates after the model is validated.
        if (System.getProperty("opencl.device.type") == null) {
            OpenCLDeviceSelector.selectDevice(OpenCLDeviceSelector.GpuVendor.valueOf(
                    System.getProperty("opencl.vendor", "INTEL").toUpperCase()));
        }

        if (Boolean.getBoolean("llm.printTokens")) {
            BpeTokenizer tok = BpeTokenizer.fromGguf(GGUFLoader.load(ggufPath));
            System.out.println("Prompt token ids: " + java.util.Arrays.toString(tok.encode(prompt, true)));
        }

        System.out.println("Loading and validating model, then uploading to the device...");
        long t0 = System.nanoTime();
        // Validates architecture / tensor shapes / types before touching the device; if loading fails
        // partway, everything already uploaded is released and the original exception is rethrown.
        try (OpenCLLlamaEngine engine = new OpenCLLlamaEngine(ggufPath)) {
            System.out.printf("Load + upload: %.1f s on %s%n", (System.nanoTime() - t0) / 1e9, engine.getDeviceDescription());

            LlamaGenerationConfig cfg = new LlamaGenerationConfig();
            cfg.maxNewTokens = 64;
            cfg.temperature = 0.0f; // deterministic greedy decoding
            cfg.topK = 1;
            cfg.topP = 1.0f;
            cfg.repetitionPenalty = 1.0f;

            System.out.println("Running inference...");
            String completion = engine.generate(prompt, cfg);
            if (completion == null || completion.isBlank()) {
                System.out.println("Generated text: <empty>");
            } else {
                System.out.println("Generated text:");
                System.out.println(completion);
            }
        }
    }
}
