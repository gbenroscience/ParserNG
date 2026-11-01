package com.github.gbenroscience.gpu.llm.opencl;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.time.LocalDateTime;

/**
 * One-click diagnostics: runs every measurement needed to find out where the device time goes and writes the whole
 * output to {@code llm-diagnostics.txt} (in the working directory; override with -Dllm.diagnostics.out=path) as well as to the console, so it can simply be
 * attached to a report. No arguments are required (an IDE "Run File" is enough).
 *
 * <p>Steps (each is independent: a failure in one is reported and the next still runs):
 * <ol>
 *   <li>hardware and driver information plus streaming-read bandwidth at four load widths</li>
 *   <li>device-memory plan of the model shape, checked against what a real load allocates</li>
 *   <li>GEMV sweep: event-timed projection shapes and the LM head, rows-vs-bytes fit</li>
 *   <li>full model run: load breakdown, cold/warm timing, and an event-profiled run with per-operation device time</li>
 * </ol>
 *
 * <pre>
 * OpenCLDiagnostics [model.gguf] [--quick] [--max-new N]
 * </pre>
 * {@code --quick} uses tiny shapes for the model-free steps (for testing the tool itself). Device selection follows
 * {@code -Dopencl.vendor=...} (default INTEL here) or {@code -Dopencl.device.type=CPU}. Starting the JVM with
 * {@code -Xmx4g} additionally lets the sweep time the old F32 LM head.
 */
public final class OpenCLDiagnostics {

    private static final String DEFAULT_MODEL =
            "C:\\Users\\GBEMIRO\\Documents\\NetBeansProjects\\ai-models\\llama-3.2-1b-instruct-q8_0.gguf";

    private OpenCLDiagnostics() {
    }

    private static final class Tee extends OutputStream {

        private final OutputStream a;
        private final OutputStream b;

        Tee(OutputStream a, OutputStream b) {
            this.a = a;
            this.b = b;
        }

        @Override
        public void write(int c) throws java.io.IOException {
            a.write(c);
            b.write(c);
        }

        @Override
        public void write(byte[] buf, int off, int len) throws java.io.IOException {
            a.write(buf, off, len);
            b.write(buf, off, len);
        }

        @Override
        public void flush() throws java.io.IOException {
            a.flush();
            b.flush();
        }
    }

    private static void step(String title, ThrowingRunnable r) {
        System.out.println("\n########## " + title + " ##########");
        long t0 = System.nanoTime();
        try {
            r.run();
        } catch (Throwable t) {
            System.out.println("STEP FAILED: " + t);
            t.printStackTrace(System.out);
        }
        System.out.printf("(step took %.1f s)%n", (System.nanoTime() - t0) / 1e9);
    }

    private interface ThrowingRunnable {

        void run() throws Throwable;
    }

    public static void main(String[] args) throws Throwable {
        String model = DEFAULT_MODEL;
        boolean quick = false;
        String maxNew = "8";
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--quick")) {
                quick = true;
                System.setProperty("llm.bandwidth.mb", "16");
            } else if (args[i].equals("--max-new")) {
                maxNew = args[++i];
            } else {
                model = args[i];
            }
        }
        if (System.getProperty("opencl.device.type") == null && System.getProperty("opencl.vendor") == null) {
            System.setProperty("opencl.vendor", "INTEL");
        }
        File out = new File(System.getProperty("llm.diagnostics.out", "llm-diagnostics.txt"));
        PrintStream console = System.out;
        try (PrintStream file = new PrintStream(new FileOutputStream(out), true, "UTF-8")) {
            System.setOut(new PrintStream(new Tee(console, file), true, "UTF-8"));
            System.out.println("ParserNG OpenCL diagnostics, " + LocalDateTime.now());
            System.out.println("os: " + System.getProperty("os.name") + " " + System.getProperty("os.version") + " " + System.getProperty("os.arch"));
            System.out.println("JVM: " + System.getProperty("java.vm.name") + " " + System.getProperty("java.version") + ", max heap "
                    + Runtime.getRuntime().maxMemory() / (1024 * 1024) + " MB");
            System.out.println(Variants.describe());
            final String[] shape = quick
                    ? new String[]{"--dim", "256", "--hidden", "512", "--heads", "8", "--kv-heads", "4", "--head-dim", "32", "--layers", "2", "--vocab", "2000", "--max-seq", "256"}
                    : new String[0];
            step("1/4 hardware and streaming bandwidth", () -> OpenCLBenchmark.main(new String[]{"--bandwidth-only"}));
            step("2/4 device memory plan", () -> OpenCLBenchmark.main(concat(new String[]{"--memory-plan"}, shape)));
            step("3/4 GEMV sweep (event-timed)", () -> OpenCLBenchmark.main(concat(new String[]{"--gemv-sweep"}, shape)));
            final String m = model;
            final String mn = maxNew;
            if (new File(m).isFile()) {
                step("4/4 full model: load breakdown, timing, per-operation profile", () -> OpenCLBenchmark.main(new String[]{m, "--max-new", mn}));
            } else {
                System.out.println("\n(step 4/4 skipped: model file not found: " + m + " -- pass the path as the first argument)");
            }
            System.out.println("\nAll output was also written to " + out.getAbsolutePath());
        } finally {
            System.setOut(console);
            console.println("diagnostics written to " + out.getAbsolutePath());
        }
    }

    private static String[] concat(String[] a, String[] b) {
        String[] r = java.util.Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }
}
