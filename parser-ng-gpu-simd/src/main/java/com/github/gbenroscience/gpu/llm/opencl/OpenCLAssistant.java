package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.LlamaGenerationConfig;
import com.github.gbenroscience.gpu.llm.TokenListener;
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.io.File;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Reusable interactive OpenCL GGUF inference engine.
 *
 * The model is loaded and uploaded ONCE. Multiple prompts are processed using
 * the same engine instance.
 *
 * Usage:
 *
 * OpenCLDemo OpenCLDemo model.gguf
 *
 * System properties:
 *
 * -Dopencl.vendor=AMD|INTEL|NVIDIA -Dopencl.device.type=CPU|GPU
 *
 * -Dllm.timing=true -Dllm.profile=false -Dllm.maxContext=8192
 * -Dllm.maxNewTokens=256
 *
 * -Dllm.temperature=0.0 -Dllm.topK=1 -Dllm.topP=1.0 -Dllm.repetitionPenalty=1.0
 *
 * -Dllm.printTokens=false
 *
 * Interactive commands:
 *
 * :quit Exit :exit Exit :help Display commands
 *
 * IMPORTANT:
 *
 * OpenCLLlamaEngine.generate() must reset the sequence position and KV-cache
 * state for every independent prompt. Reusing the engine must not mean
 * continuing the previous conversation unintentionally.
 */
public final class OpenCLAssistant {

    private static final String DEFAULT_MODEL
            = "C:\\Users\\GBEMIRO\\Documents\\NetBeansProjects\\ai-models\\llama-3.2-1b-instruct-q8_0.gguf";

    private OpenCLAssistant() {
    }

    public static void main(String[] args) throws Throwable {

        configureRuntime();

        File model = args.length > 0
                ? new File(args[0])
                : new File(DEFAULT_MODEL);

        if (!model.isFile()) {
            throw new IllegalArgumentException(
                    "GGUF model does not exist: " + model.getAbsolutePath()
            );
        }

        System.out.println("================================================");
        System.out.println("          ParserNG OpenCL LLM Runtime");
        System.out.println("================================================");
        System.out.println("Model: " + model.getAbsolutePath());

        selectDevice();

        /*
         * Load and upload the model only once.
         */
        System.out.println();
        System.out.println("Loading and validating model...");

        long loadStart = System.nanoTime();

        try (OpenCLLlamaEngine engine = new OpenCLLlamaEngine(model)) {
            System.out.print(com.github.gbenroscience.gpu.llm.opencl.LoadTimer.report());
            long loadTime = System.nanoTime() - loadStart;

            System.out.printf(
                    "Model ready in %.2f seconds%n",
                    loadTime / 1_000_000_000.0
            );

            System.out.println("Device: " + engine.getDeviceDescription());

            System.out.println();
            System.out.println("Model is ready. Enter prompts below.");
            System.out.println("Type :help for commands.");
            System.out.println();

            /*
             * Scanner is deliberately passed into the session method.
             * This allows other Java applications to supply their own
             * Scanner without transferring ownership of System.in.
             */
            Scanner scanner = new Scanner(System.in);

            runInteractiveSession(engine, scanner, System.out);

        } catch (Throwable ex) {
            System.err.println();
            System.err.println("OpenCL inference failed:");
            ex.printStackTrace();
            throw ex;
        }

        System.out.println();
        System.out.println("OpenCL engine closed.");
    }

    /**
     * Configure all system properties before OpenCL engine classes initialize.
     */
    private static void configureRuntime() {

        /*
         * Timing is useful during development.
         */
        setDefaultProperty("llm.timing", "true");

        /*
         * Disable event profiling during normal inference.
         *
         * Enable with -Dllm.profile=true when investigating kernel performance.
         */
        setDefaultProperty("llm.profile", "false");

        /*
         * Match the engine's documented default context capacity.
         */
        setDefaultProperty("llm.maxContext", "8192");

        /*
         * Generation settings.
         */
        setDefaultProperty("llm.maxNewTokens", "1024");
        setDefaultProperty("llm.temperature", "0.0");
        setDefaultProperty("llm.topK", "1");
        setDefaultProperty("llm.topP", "1.0");
        setDefaultProperty("llm.repetitionPenalty", "1.0");

        setDefaultProperty("llm.printTokens", "false");
    }

    private static void setDefaultProperty(String name, String value) {
        if (System.getProperty(name) == null) {
            System.setProperty(name, value);
        }
    }

    /**
     * Select the OpenCL device.
     */
    private static void selectDevice() {

        /*
         * Explicit device type takes precedence over vendor selection.
         */
        if (System.getProperty("opencl.device.type") != null) {
            return;
        }

        String vendor = System.getProperty("opencl.vendor", "INTEL")
                .trim()
                .toUpperCase();

        OpenCLDeviceSelector.selectDevice(
                OpenCLDeviceSelector.GpuVendor.valueOf(vendor)
        );
    }

    /**
     * Interactive inference session.
     *
     * The same engine instance processes every prompt.
     *
     * This method does not close the engine or scanner. 
     * @param engine
     * @param scanner
     * @param out
     * @throws Throwable 
     */
    public static void runInteractiveSession(
            OpenCLLlamaEngine engine,
            Scanner scanner,
            PrintStream out
    ) throws Throwable {

        int requestCount = 0;

        while (true) {

            out.print("You > ");
            out.flush();

            if (!scanner.hasNextLine()) {
                break;
            }

            String prompt = scanner.nextLine().trim();

            if (prompt.isEmpty()) {
                continue;
            }

            /*
             * Interactive commands.
             */
            if (prompt.equalsIgnoreCase(":quit")
                    || prompt.equalsIgnoreCase(":exit")) {
                break;
            }

            if (prompt.equalsIgnoreCase(":help")) {
                printHelp(out);
                continue;
            }

            requestCount++;

            out.println();
            out.println("Assistant >");

            long start = System.nanoTime();

            try {

                LlamaGenerationConfig config = createGenerationConfig();

                /*
                 * Reuse the already initialized OpenCL engine.
                 */
                String response = engine.generate(prompt, config, (String piece) -> {
                    System.out.print(piece);
                    System.out.flush();
                    return true;          // return false to stop (a Stop button, Esc, a timeout)

                });

                long elapsed = System.nanoTime() - start;

                if (response == null || response.isBlank()) {
                    out.println("<empty response>");
                } else {
                   // out.println(response);
                    out.println("______________________________________________________".repeat(5));
                }

                out.printf(
                        "%n[request %d completed in %.2f seconds]%n",
                        requestCount,
                        elapsed / 1_000_000_000.0
                );

            } catch (Throwable ex) {

                out.println();
                out.println("[Inference error]");
                ex.printStackTrace(out);

                /*
                 * Keep the interactive session alive.
                 * A failed generation should not automatically terminate
                 * the application.
                 */
            }

            out.println();
        }

        out.printf(
                "Session ended. Total requests: %d%n",
                requestCount
        );
    }

    /**
     * Construct generation configuration.
     *
     * Properties make it possible to tune inference without recompiling.
     */
    private static LlamaGenerationConfig createGenerationConfig() {

        LlamaGenerationConfig config = new LlamaGenerationConfig();

        config.maxNewTokens = Integer.getInteger(
                "llm.maxNewTokens", 1024
        );

        config.temperature = Float.parseFloat(
                System.getProperty("llm.temperature", "0.0")
        );

        config.topK = Integer.getInteger(
                "llm.topK", 1
        );

        config.topP = Float.parseFloat(
                System.getProperty("llm.topP", "1.0")
        );

        config.repetitionPenalty = Float.parseFloat(
                System.getProperty("llm.repetitionPenalty", "1.0")
        );

        return config;
    }

    private static void printHelp(PrintStream out) {

        out.println();
        out.println("Available commands:");
        out.println("  :help    Display this help");
        out.println("  :quit    Exit the application");
        out.println("  :exit    Exit the application");
        out.println();
    }
}
