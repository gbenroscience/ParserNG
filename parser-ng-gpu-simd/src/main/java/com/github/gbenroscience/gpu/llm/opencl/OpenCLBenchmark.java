package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.LlamaGenerationConfig;

import java.io.File;
import java.lang.foreign.MemorySegment;
import java.util.Map;

/**
 * Reproducible end-to-end benchmark of the OpenCL engine.
 *
 * <pre>
 * OpenCLBenchmark &lt;model.gguf&gt; [--prompt "text"] [--max-new N] [--warm-runs R] [--no-profile] [--no-timing] [--bandwidth]
 * OpenCLBenchmark --bandwidth-only          (no model needed: device info + streaming read bandwidth probe)
 * </pre>
 *
 * It runs two separate engines so measurement does not distort itself:
 * <ol>
 *   <li><b>timing run</b> (profiler OFF): model load, cold first generation, then warm generations. These are the
 *       numbers to quote; nothing is synchronised other than what the engine itself needs.</li>
 *   <li><b>profile run</b> (event profiler ON): where the device time goes per kernel / phase / layer, plus kernel
 *       launches, synchronisations, allocations and transfers per token. Event capture adds a little overhead, so
 *       its wall-clock numbers are not quoted as performance.</li>
 * </ol>
 * Greedy decoding with EOS disabled so every run produces exactly the same number of tokens.
 * Other switches: -Dopencl.vendor / -Dopencl.device.type, -Dllm.maxContext=N,
 * -Dllm.gemv=legacy, -Dllm.attention=legacy, -Dllm.rmsnorm=legacy.
 */
public final class OpenCLBenchmark {

    private OpenCLBenchmark() {
    }

    private static LlamaGenerationConfig config(int maxNew) {
        LlamaGenerationConfig c = new LlamaGenerationConfig();
        c.maxNewTokens = maxNew;
        c.temperature = 0f;
        c.topK = 1;
        c.topP = 1f;
        c.repetitionPenalty = 1f;
        c.eosTokenId = -1;
        return c;
    }

    /**
     * Streaming-read bandwidth of the selected device at different per-work-item load widths, to establish what the
     * memory system really delivers and how sensitive it is to load width / alignment (a Q8_0 block's quants start at a
     * 2-byte offset, so vector loads on them are misaligned).
     */
    static void bandwidthProbe(GpuContext ctx) throws Throwable {
        long bytes = Long.getLong("llm.bandwidth.mb", 256L) << 20;
        MemorySegment buf = null;
        for (; bytes >= (16L << 20); bytes >>= 1) {
            try {
                byte[] host = new byte[(int) bytes];
                for (int i = 0; i < host.length; i += 61) {
                    host[i] = (byte) (i * 31 + 7); // touch the pages with non-zero data
                }
                buf = LlamaLayer.uploadBytes(ctx, host);
                break;
            } catch (Throwable t) {
                System.out.println("(could not allocate " + (bytes >> 20) + " MB probe buffer: " + t.getMessage() + ")");
            }
        }
        if (buf == null) {
            System.out.println("bandwidth probe skipped: no buffer could be allocated");
            return;
        }
        final long items = 65536;
        MemorySegment out = LlamaLayer.allocBytes(ctx, items * 4);
        try {
            System.out.printf("%n==================== streaming read bandwidth (%d MB buffer, %d work-items) ====================%n", bytes >> 20, items);
            Object[][] probes = {
                {"1-byte loads (uchar)", ctx.kBwU8, (int) bytes},
                {"4-byte loads (uint)", ctx.kBwU32, (int) (bytes / 4)},
                {"16-byte loads (uint4, aligned)", ctx.kBwU128, (int) (bytes / 16)},
                {"16-byte vload16 at +2 byte offset", ctx.kBwV16Off2, (int) ((bytes - 2) / 16)},};
            for (Object[] pr : probes) {
                MemorySegment k = (MemorySegment) pr[1];
                int n = (Integer) pr[2];
                LlamaLayer.bandwidthRead(ctx, k, buf, out, n, items);
                LlamaLayer.finish(ctx);
                double best = Double.MAX_VALUE;
                for (int r = 0; r < 5; r++) {
                    long t0 = System.nanoTime();
                    LlamaLayer.bandwidthRead(ctx, k, buf, out, n, items);
                    LlamaLayer.finish(ctx);
                    best = Math.min(best, (System.nanoTime() - t0) / 1e9);
                }
                System.out.printf("  %-36s %7.2f GB/s  (best of 5, %.1f ms)%n", pr[0], bytes / 1e9 / best, best * 1e3);
            }
            System.out.println("  Decode needs roughly (bytes read per token) / (this bandwidth) seconds per token;"
                    + " e.g. 1.4 GB/token at 20 GB/s = 70 ms/token = 14 tok/s.");
        } finally {
            LlamaLayer.freeQuietly(ctx, buf);
            LlamaLayer.freeQuietly(ctx, out);
        }
    }

    private static int intArg(String[] a, String key, int def) {
        for (int i = 0; i < a.length - 1; i++) {
            if (a[i].equals(key)) {
                return Integer.parseInt(a[i + 1]);
            }
        }
        return def;
    }

    /** Model-free modes: --gemv-sweep and --memory-plan (Llama-3.2-1B shapes by default; override with --dim --hidden --heads --kv-heads --head-dim --layers --vocab --max-seq). */
    private static void modelFree(String[] a) throws Throwable {
        if (System.getProperty("opencl.device.type") == null && System.getProperty("opencl.vendor") != null) {
            OpenCLDeviceSelector.selectDevice(OpenCLDeviceSelector.GpuVendor.valueOf(System.getProperty("opencl.vendor").toUpperCase()));
        }
        int dim = intArg(a, "--dim", 2048), hidden = intArg(a, "--hidden", 8192), heads = intArg(a, "--heads", 32);
        int kv = intArg(a, "--kv-heads", 8), hd = intArg(a, "--head-dim", 64), layers = intArg(a, "--layers", 16);
        int vocab = intArg(a, "--vocab", 128256), maxSeq = intArg(a, "--max-seq", 8192);
        boolean sweep = false, plan = false, f32head = false;
        for (String s : a) {
            sweep |= s.equals("--gemv-sweep");
            plan |= s.equals("--memory-plan");
            f32head |= s.equals("--f32-head");
        }
        GpuContext ctx = new GpuContext(sweep);
        try {
            System.out.println("==================== hardware ====================\n" + ctx.deviceInfo());
            if (plan) {
                LlamaLayer.Config cfg = new LlamaLayer.Config();
                cfg.n_layers = layers;
                cfg.dim = dim;
                cfg.hidden_dim = hidden;
                cfg.num_heads = heads;
                cfg.kv_heads = kv;
                cfg.head_dim = hd;
                cfg.max_seq = maxSeq;
                MemoryPlan.run(ctx, cfg, vocab, !f32head);
            }
            if (sweep) {
                GemvSweep.run(ctx, dim, hidden, heads, kv, hd, layers, vocab, true);
            }
        } finally {
            ctx.close();
        }
    }

    public static void main(String[] args) throws Throwable {
        for (String s : args) {
            if (s.equals("--gemv-sweep") || s.equals("--memory-plan")) {
                modelFree(args);
                return;
            }
        }
        if (args.length >= 1 && args[0].equals("--bandwidth-only")) {
            if (System.getProperty("opencl.device.type") == null && System.getProperty("opencl.vendor") != null) {
                OpenCLDeviceSelector.selectDevice(OpenCLDeviceSelector.GpuVendor.valueOf(System.getProperty("opencl.vendor").toUpperCase()));
            }
            GpuContext ctx = new GpuContext(false);
            try {
                System.out.println("==================== hardware ====================\n" + ctx.deviceInfo());
                bandwidthProbe(ctx);
            } finally {
                ctx.close();
            }
            return;
        }
        if (args.length < 1) {
            System.err.println("Usage: OpenCLBenchmark <model.gguf> [--prompt text] [--max-new N] [--warm-runs R] [--no-profile] [--no-timing]");
            System.exit(1);
        }
        File model = new File(args[0]);
        String prompt = "What is 23*34? Answer briefly.";
        int maxNew = 16;
        int warmRuns = 1;
        boolean doProfile = true;
        boolean doTiming = true;
        boolean doBandwidth = false;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--prompt" -> prompt = args[++i];
                case "--max-new" -> maxNew = Integer.parseInt(args[++i]);
                case "--warm-runs" -> warmRuns = Integer.parseInt(args[++i]);
                case "--no-profile" -> doProfile = false;
                case "--no-timing" -> doTiming = false;
                case "--bandwidth" -> doBandwidth = true;
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (!model.isFile()) {
            throw new IllegalArgumentException("model file not found: " + model.getAbsolutePath());
        }
        if (System.getProperty("opencl.device.type") == null && System.getProperty("opencl.vendor") != null) {
            OpenCLDeviceSelector.selectDevice(OpenCLDeviceSelector.GpuVendor.valueOf(System.getProperty("opencl.vendor").toUpperCase()));
        }
        System.out.println("model: " + model.getAbsolutePath());
        System.out.println("prompt: " + prompt + "   (greedy, " + maxNew + " new tokens, EOS disabled)");
        System.out.println(Variants.describe());
        System.out.println("JVM: " + System.getProperty("java.vm.name") + " " + System.getProperty("java.version") + ", max heap "
                + Runtime.getRuntime().maxMemory() / (1024 * 1024) + " MB, " + Runtime.getRuntime().availableProcessors() + " CPUs");

        double loadS = Double.NaN;
        LlamaOpenCLEngine.RunStats cold = null;
        LlamaOpenCLEngine.RunStats warm = null;
        if (doTiming) {
            System.setProperty("llm.timing", "false");
            LoadTimer.reset();
            long t0 = System.nanoTime();
            GpuContext ctx = new GpuContext(false);
            System.out.println("\n==================== hardware ====================\n" + ctx.deviceInfo());
            if (doBandwidth) {
                bandwidthProbe(ctx);
            }
            try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(model, ctx)) {
                loadS = (System.nanoTime() - t0) / 1e9;
                System.out.printf("model load + upload: %.1f s%n", loadS);
                System.out.print(LoadTimer.report());
                System.out.println("\n-- cold generation (first use of every kernel) --");
                eng.generate(prompt, config(maxNew));
                cold = eng.lastRunStats();
                for (int r = 0; r < warmRuns; r++) {
                    System.out.println("-- warm generation " + (r + 1) + " --");
                    eng.generate(prompt, config(maxNew));
                    warm = eng.lastRunStats();
                    printRun("warm " + (r + 1), warm);
                }
                printRun("cold", cold);
            }
        }

        long launchesPerToken = -1, syncsPerToken = -1;
        if (doProfile) {
            LoadTimer.reset();
            GpuContext ctx = new GpuContext(true);
            if (!doTiming) {
                System.out.println("\n==================== hardware ====================\n" + ctx.deviceInfo());
            }
            try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(model, ctx)) {
                System.out.println("\n-- profile run: warm-up generation (its report below is discarded) --");
                eng.generate(prompt, config(2)); // warm-up: first use of every kernel
                ctx.prof.flush();
                ctx.prof.resetStats();
                System.out.println("\n-- profile run: MEASURED generation --");
                eng.generate(prompt, config(maxNew));
                LlamaOpenCLEngine.RunStats st = eng.lastRunStats();
                DeviceProfiler p = ctx.prof;
                int fwd = Math.max(st.decodeForwards, 1);
                launchesPerToken = p.counter("decode", DeviceProfiler.LAUNCHES) / fwd;
                syncsPerToken = (p.counter("decode", DeviceProfiler.SYNCS) + p.counter("decode", DeviceProfiler.BLOCKING_READS)
                        + p.counter("decode", DeviceProfiler.BLOCKING_WRITES)) / fwd;
            }
        }

        System.out.println("\n==================== summary (measured on this machine) ====================");
        System.out.printf("| %-24s | %16s | %16s |%n", "Metric", "Cold", "Warm");
        System.out.printf("| %-24s | %16s | %16s |%n", "------------------------", "---------------", "---------------");
        if (cold != null && warm != null) {
            System.out.printf("| %-24s | %14.1f s | %16s |%n", "Model load/upload", loadS, "-");
            System.out.printf("| %-24s | %11.2f tok/s | %11.2f tok/s |%n", "Prefill", cold.promptTokens * 1000.0 / cold.prefillMs, warm.promptTokens * 1000.0 / warm.prefillMs);
            System.out.printf("| %-24s | %11.3f tok/s | %11.3f tok/s |%n", "Decode", cold.decodeForwards * 1000.0 / cold.decodeMs, warm.decodeForwards * 1000.0 / warm.decodeMs);
            System.out.printf("| %-24s | %9.1f ms/tok | %9.1f ms/tok |%n", "Decode latency", cold.decodeMs / cold.decodeForwards, warm.decodeMs / warm.decodeForwards);
        }
        if (launchesPerToken >= 0) {
            System.out.printf("| %-24s | %16d | %16s |%n", "Kernel launches/token", launchesPerToken, "(same)");
            System.out.printf("| %-24s | %16d | %16s |%n", "Synchronizations/token", syncsPerToken, "(same)");
        }
        System.out.println("(Synchronizations = clFinish + blocking reads + blocking writes per decoded token.)");
    }

    private static void printRun(String label, LlamaOpenCLEngine.RunStats s) {
        System.out.printf("[%s] prompt=%d tokens; tokenize %.2f ms; time-to-first-token %.1f ms (%.2f tok/s); decode %d tokens in %.1f ms = %.3f tok/s, %.1f ms/token%n",
                label, s.promptTokens, s.tokenizeMs, s.prefillMs, s.promptTokens * 1000.0 / s.prefillMs, s.decodeForwards, s.decodeMs,
                s.decodeForwards * 1000.0 / s.decodeMs, s.decodeMs / Math.max(s.decodeForwards, 1));
        System.out.printf("[%s] decode host wall per token: embed %.2f ms | enqueue all layers %.2f ms | logits + readback wait %.2f ms | sample %.2f ms%n",
                label, s.embedMs / Math.max(s.decodeForwards, 1), s.layerSubmitMs / Math.max(s.decodeForwards, 1),
                s.logitsWaitMs / Math.max(s.decodeForwards, 1), s.sampleMs / Math.max(s.decodeForwards, 1));
    }
}
