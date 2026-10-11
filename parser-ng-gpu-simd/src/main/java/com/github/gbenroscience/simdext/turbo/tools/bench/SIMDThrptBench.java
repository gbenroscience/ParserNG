package com.github.gbenroscience.simdext.turbo.tools.bench;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF32;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF64;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandSegmentF32;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandSegmentF64;
import com.github.gbenroscience.simdext.turbo.tools.utils.CPUPinner;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.function.DoubleSupplier;

/**
 * Throughput benchmark for every bulk evaluator path.
 *
 * Run with: java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED ...
 *
 * Variants:
 *  1 Auto-vectorized (double[][])
 *  2 Base SIMD (double[][])
 *  3 SIMD F32 (float[][])
 *  4 SIMD F64 (double[][])
 *  5 SIMD F32 zero-copy, array API (float[][])
 *  6 SIMD F64 zero-copy, array API (double[][])
 *  7 SIMD F64 zero-copy, flat MemorySegment (all variables in one segment)
 *  8 SIMD F32 zero-copy, flat MemorySegment (all variables in one segment)
 *  9 SIMD F64 zero-copy, MemorySegment[] (one segment per variable)
 * 10 SIMD F32 zero-copy, MemorySegment[] (one segment per variable)
 *  0 Run everything and print a summary table
 *
 * Flat MemorySegment layout (confirmed from the evaluator source): variable-major
 * contiguous blocks, element (v, i) lives at byte offset (v * N + i) * elemSize,
 * where N = output sample count. Output element i lives at i * elemSize.
 * MemorySegment[] layout: segment v holds N elements of variable v; sample
 * count is derived from variables[0].byteSize(), and every segment plus the
 * output must have identical length.
 *
 * Note: validate(MemorySegment, MemorySegment) requires input size == output
 * size, which only holds for single-variable expressions, so the flat path is
 * benchmarked without calling validate().
 *
 * @author oluwagbemirojiboye
 */
public class SIMDThrptBench {

    private static final int WARMUP_ITERATIONS = 40;
    private static final int MEASUREMENT_ITERATIONS = 150;
    private static final int TOTAL_ELEMENTS = 1_000_017; // trailing 17 forces lane remainders

    /** Optional -Dbench.workers=N: worker count for variants 3-10 (default: evaluator's own default). */
    private static final boolean DIAG = Boolean.getBoolean("parserng.diag");
    private static java.util.function.Supplier<String> diagHook = null;

    private static final Integer WORKERS = Integer.getInteger("bench.workers");

    public static final int RUN_ALL = 0;
    public static final int EVALUATOR_BASE_AUTO_VECTORIZED = 1;
    public static final int EVALUATOR_BASE_SIMD = 2;
    public static final int EVALUATOR_SIMD_F32 = 3;
    public static final int EVALUATOR_SIMD_F64 = 4;
    public static final int EVALUATOR_SIMD_ZERO_COPY_F32 = 5;
    public static final int EVALUATOR_SIMD_ZERO_COPY_F64 = 6;
    public static final int EVALUATOR_SEGMENT_F64 = 7;
    public static final int EVALUATOR_SEGMENT_F32 = 8;
    public static final int EVALUATOR_SEGMENT_ARRAY_F64 = 9;
    public static final int EVALUATOR_SEGMENT_ARRAY_F32 = 10;

    @FunctionalInterface
    private interface F64Bulk {
        void apply(double[][] in, double[] out) throws Throwable;
    }

    @FunctionalInterface
    private interface F32Bulk {
        void apply(float[][] in, float[] out) throws Throwable;
    }

    @FunctionalInterface
    private interface SegBulk {
        void apply(MemorySegment in, MemorySegment out) throws Throwable;
    }

    @FunctionalInterface
    private interface SegArrBulk {
        void apply(MemorySegment[] in, MemorySegment out) throws Throwable;
    }

    @FunctionalInterface
    private interface Bulk {
        void run() throws Throwable;
    }

    private record Result(String label, double avgMs, double mops) {}

    public static void main(String[] args) throws Throwable {
        Scanner sc = new Scanner(System.in);
        System.out.println("Pin order (logical CPUs): " + java.util.Arrays.toString(CPUPinner.pinOrder())
                + " | pinning=" + CPUPinner.isPinningEnabled()
                + " | workers=" + (WORKERS == null ? "default" : WORKERS));
        String cliExpr = (args.length > 0 && !args[0].isBlank()) ? args[0] : null;
        // One-shot mode: args = <expr> <variant 0-10>. Runs once and exits, so a shell loop
        // can give every variant its own JVM (clean JIT profile, no stale worker pools).
        Integer cliVariant = (args.length > 1) ? Integer.valueOf(args[1].trim()) : null;

        while (true) {
            String rawExpr;
            if (cliExpr != null) {
                rawExpr = cliExpr;
                cliExpr = null; // only use the CLI argument for the first pass
                System.out.println("Using expression from command line argument.");
            } else {
                System.out.println("\nEnter math expression (e.g., sin(sqrt(x^2+y^2))), or blank/q to quit:");
                if (!sc.hasNextLine()) {
                    return;
                }
                rawExpr = sc.nextLine().trim();
            }
            if (rawExpr.isBlank() || rawExpr.equalsIgnoreCase("q")) {
                return;
            }

            MathExpression me;
            try {
                me = new MathExpression(rawExpr);
            } catch (Exception e) {
                System.err.println("Could not parse expression.");
                e.printStackTrace();
                continue;
            }

            int choice;
            if (cliVariant != null) {
                choice = cliVariant;
            } else {
            System.out.println("Select bulk evaluator variant (0 = run all):");
            System.out.println("1============================Auto-Vectorizer (slowest)");
            System.out.println("2============================Base SIMD (Vector API 21)");
            System.out.println("3============================SIMD F32 (arrays)");
            System.out.println("4============================SIMD F64 (arrays)");
            System.out.println("5============================SIMD F32 zero-copy (arrays)");
            System.out.println("6============================SIMD F64 zero-copy (arrays)");
            System.out.println("7============================SIMD F64 zero-copy (flat MemorySegment)");
            System.out.println("8============================SIMD F32 zero-copy (flat MemorySegment)");
            System.out.println("9============================SIMD F64 zero-copy (MemorySegment[])");
            System.out.println("10===========================SIMD F32 zero-copy (MemorySegment[])");
            System.out.println("0============================All of the above + summary");

            try {
                String line = sc.nextLine().trim();
                if (line.equalsIgnoreCase("q")) {
                    continue; // back to the expression prompt
                }
                choice = Integer.parseInt(line);
            } catch (NumberFormatException e) {
                System.err.println("Please enter a number from 0 to 10.");
                continue;
            }
            }

            int varCount = me.getVariables().getKey().length;
            if (varCount == 0) {
                System.out.println("No variables found. Defaulting to 1 dummy variable slot.");
                varCount = 1;
            }
            System.out.println("Detected variables: " + me.getVariables() + " (" + varCount + " variables)");

            List<Result> results = new ArrayList<>();
            try {
                if (choice == RUN_ALL) {
                    for (int v = 1; v <= 10; v++) {
                        runVariant(v, me, varCount, results);
                    }
                } else {
                    runVariant(choice, me, varCount, results);
                }
            } catch (Throwable t) {
                System.err.println("Variant failed. Check syntax/function support or the signature assumptions.");
                t.printStackTrace();
                if (cliVariant != null) {
                    return;
                }
                continue;
            }

            if (results.size() > 2) {
                printSummary(results);
            }
            if (cliVariant != null) {
                return; // one-shot mode
            }
        }
    }

    private static void runVariant(int variant, MathExpression me, int varCount, List<Result> results) throws Throwable {
        switch (variant) {
            case EVALUATOR_BASE_AUTO_VECTORIZED -> {
                var ev = VectorTurboEvaluator.getEvaluator(me);
                benchF64("1 AutoVec F64[]", ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            case EVALUATOR_BASE_SIMD -> {
                var ev = SIMDVectorTurboEvaluator.getEvaluator(me);
                benchF64("2 BaseSIMD F64[]", ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            case EVALUATOR_SIMD_F32 -> {
                var ev = (WORKERS == null) ? SIMDCommandF32.getEvaluator(me) : SIMDCommandF32.getEvaluator(me, WORKERS);
                diagHook = SIMDCommandF32.SIMDVectorCompositeExpression::diagReportAndReset;
                benchF32("3 SIMD F32[]", ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            case EVALUATOR_SIMD_F64 -> {
                var ev = (WORKERS == null) ? SIMDCommandF64.getEvaluator(me) : SIMDCommandF64.getEvaluator(me, WORKERS);
                diagHook = SIMDCommandF64.SIMDVectorCompositeExpression::diagReportAndReset;
                benchF64("4 SIMD F64[]", ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            case EVALUATOR_SIMD_ZERO_COPY_F32 -> {
                var ev = (WORKERS == null) ? SIMDCommandSegmentF32.getEvaluator(me) : SIMDCommandSegmentF32.getEvaluator(me, WORKERS);
                diagHook = SIMDCommandSegmentF32.SIMDVectorCompositeExpression::diagReportAndReset;
                benchF32("5 ZeroCopy F32[]", ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            case EVALUATOR_SIMD_ZERO_COPY_F64 -> {
                var ev = (WORKERS == null) ? SIMDCommandSegmentF64.getEvaluator(me) : SIMDCommandSegmentF64.getEvaluator(me, WORKERS);
                diagHook = SIMDCommandSegmentF64.SIMDVectorCompositeExpression::diagReportAndReset;
                benchF64("6 ZeroCopy F64[]", ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            case EVALUATOR_SEGMENT_F64 -> {
                var ev = (WORKERS == null) ? SIMDCommandSegmentF64.getEvaluator(me) : SIMDCommandSegmentF64.getEvaluator(me, WORKERS);
                diagHook = SIMDCommandSegmentF64.SIMDVectorCompositeExpression::diagReportAndReset;
                benchSegment("7 ZeroCopy F64 flat Segment", true, ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            case EVALUATOR_SEGMENT_F32 -> {
                var ev = (WORKERS == null) ? SIMDCommandSegmentF32.getEvaluator(me) : SIMDCommandSegmentF32.getEvaluator(me, WORKERS);
                diagHook = SIMDCommandSegmentF32.SIMDVectorCompositeExpression::diagReportAndReset;
                benchSegment("8 ZeroCopy F32 flat Segment", false, ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            case EVALUATOR_SEGMENT_ARRAY_F64 -> {
                var ev = (WORKERS == null) ? SIMDCommandSegmentF64.getEvaluator(me) : SIMDCommandSegmentF64.getEvaluator(me, WORKERS);
                diagHook = SIMDCommandSegmentF64.SIMDVectorCompositeExpression::diagReportAndReset;
                benchSegmentArray("9 ZeroCopy F64 Segment[]", true, ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            case EVALUATOR_SEGMENT_ARRAY_F32 -> {
                var ev = (WORKERS == null) ? SIMDCommandSegmentF32.getEvaluator(me) : SIMDCommandSegmentF32.getEvaluator(me, WORKERS);
                diagHook = SIMDCommandSegmentF32.SIMDVectorCompositeExpression::diagReportAndReset;
                benchSegmentArray("10 ZeroCopy F32 Segment[]", false, ev::applyBulk, ev::applyBulkParallel, varCount, results);
                closeQuietly(ev);
            }
            default -> throw new IllegalArgumentException("No matching evaluator exists: " + variant);
        }
    }

    // ------------------------------------------------------------------
    // Data setup + dispatch per memory model
    // ------------------------------------------------------------------
    /**
     * Evaluators own worker threads that are only stopped by close() (or GC via a
     * Cleaner). Leaving earlier variants' pools alive makes later variants compete
     * with stale spinning/parking workers and skews the parallel numbers.
     */
    private static void closeQuietly(Object ev) {
        if (ev instanceof AutoCloseable c) {
            try {
                c.close();
            } catch (Exception ignored) {
            }
        }
    }

    private static void benchF64(String label, F64Bulk seq, F64Bulk par, int varCount, List<Result> results) throws Throwable {
        double[][] in = new double[varCount][TOTAL_ELEMENTS];
        double[] out = new double[TOTAL_ELEMENTS];
        for (int v = 0; v < varCount; v++) {
            for (int i = 0; i < TOTAL_ELEMENTS; i++) {
                in[v][i] = 1.0 + (i * 0.01) + (v * 0.5);
            }
        }
        System.out.println("\n=== " + label + " | elements per variable: " + TOTAL_ELEMENTS);
        results.add(measure(label + " serial", () -> seq.apply(in, out), () -> out[TOTAL_ELEMENTS - 1]));
        results.add(measure(label + " parallel", () -> par.apply(in, out), () -> out[TOTAL_ELEMENTS - 1]));
    }

    private static void benchF32(String label, F32Bulk seq, F32Bulk par, int varCount, List<Result> results) throws Throwable {
        float[][] in = new float[varCount][TOTAL_ELEMENTS];
        float[] out = new float[TOTAL_ELEMENTS];
        for (int v = 0; v < varCount; v++) {
            for (int i = 0; i < TOTAL_ELEMENTS; i++) {
                in[v][i] = 1.0f + (i * 0.01f) + (v * 0.5f);
            }
        }
        System.out.println("\n=== " + label + " | elements per variable: " + TOTAL_ELEMENTS);
        results.add(measure(label + " serial", () -> seq.apply(in, out), () -> out[TOTAL_ELEMENTS - 1]));
        results.add(measure(label + " parallel", () -> par.apply(in, out), () -> out[TOTAL_ELEMENTS - 1]));
    }

    private static void benchSegment(String label, boolean f64, SegBulk seq, SegBulk par, int varCount, List<Result> results) throws Throwable {
        final long n = TOTAL_ELEMENTS;
        final long elemSize = f64 ? Double.BYTES : Float.BYTES;

        // Shared arena for BOTH serial and parallel runs so they hit the same memory.
        try (Arena arena = Arena.ofShared()) {
            MemorySegment in = arena.allocate(varCount * n * elemSize, 64);
            MemorySegment out = arena.allocate(n * elemSize, 64);

            for (int v = 0; v < varCount; v++) {
                for (long i = 0; i < n; i++) {
                    long off = (v * n + i) * elemSize;
                    if (f64) {
                        in.set(ValueLayout.JAVA_DOUBLE, off, 1.0 + (i * 0.01) + (v * 0.5));
                    } else {
                        in.set(ValueLayout.JAVA_FLOAT, off, 1.0f + (i * 0.01f) + (v * 0.5f));
                    }
                }
            }

            DoubleSupplier sink = f64
                    ? () -> out.get(ValueLayout.JAVA_DOUBLE, (n - 1) * elemSize)
                    : () -> out.get(ValueLayout.JAVA_FLOAT, (n - 1) * elemSize);

            System.out.println("\n=== " + label + " | elements per variable: " + n);
            results.add(measure(label + " serial", () -> seq.apply(in, out), sink));
            results.add(measure(label + " parallel", () -> par.apply(in, out), sink));
        }
    }

    private static void benchSegmentArray(String label, boolean f64, SegArrBulk seq, SegArrBulk par, int varCount, List<Result> results) throws Throwable {
        final long n = TOTAL_ELEMENTS;
        final long elemSize = f64 ? Double.BYTES : Float.BYTES;

        try (Arena arena = Arena.ofShared()) {
            MemorySegment[] in = new MemorySegment[varCount];
            for (int v = 0; v < varCount; v++) {
                in[v] = arena.allocate(n * elemSize, 64);
                for (long i = 0; i < n; i++) {
                    if (f64) {
                        in[v].setAtIndex(ValueLayout.JAVA_DOUBLE, i, 1.0 + (i * 0.01) + (v * 0.5));
                    } else {
                        in[v].setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f + (i * 0.01f) + (v * 0.5f));
                    }
                }
            }
            MemorySegment out = arena.allocate(n * elemSize, 64);

            DoubleSupplier sink = f64
                    ? () -> out.getAtIndex(ValueLayout.JAVA_DOUBLE, n - 1)
                    : () -> out.getAtIndex(ValueLayout.JAVA_FLOAT, n - 1);

            System.out.println("\n=== " + label + " | elements per variable: " + n);
            results.add(measure(label + " serial", () -> seq.apply(in, out), sink));
            results.add(measure(label + " parallel", () -> par.apply(in, out), sink));
        }
    }

    // ------------------------------------------------------------------
    // Single measurement routine used by every path
    // ------------------------------------------------------------------
    private static Result measure(String label, Bulk run, DoubleSupplier sink) throws Throwable {
        System.out.println("--- " + label + " ---");
        System.out.print("Warming up JVM JIT compiler layers... ");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            run.run();
        }
        System.out.println("Stable.");
        if (DIAG && diagHook != null) {
            diagHook.get(); // discard warmup counters
        }

        // Several short rounds instead of one long one: a single descheduled worker can wreck one
        // round (we have seen 2-3x outliers), so we report the MEDIAN round and also the best one.
        final int rounds = Math.max(1, Integer.getInteger("bench.rounds", 5));
        final int itersPerRound = Math.max(1, MEASUREMENT_ITERATIONS / rounds);
        double[] roundMs = new double[rounds];
        System.out.print("Collecting performance throughput analytics... ");
        for (int r = 0; r < rounds; r++) {
            long start = System.nanoTime();
            for (int i = 0; i < itersPerRound; i++) {
                run.run();
            }
            roundMs[r] = ((System.nanoTime() - start) / 1_000_000.0) / itersPerRound;
        }
        System.out.println("Done.");

        double[] sorted = roundMs.clone();
        java.util.Arrays.sort(sorted);
        double avgMs = sorted[sorted.length / 2];     // median round
        double bestMs = sorted[0];
        double worstMs = sorted[sorted.length - 1];
        double seconds = (avgMs / 1000.0) * itersPerRound;
        double opsPerSec = ((double) itersPerRound * TOTAL_ELEMENTS) / seconds;
        double batchesPerSec = 1000.0 / avgMs;

        System.out.printf("  Median evaluation latency : %.3f ms per bulk payload (best %.3f, worst %.3f over %d rounds)%n",
                avgMs, bestMs, worstMs, rounds);
        System.out.printf("  Execution frequency       : %.2f full batches/sec%n", batchesPerSec);
        System.out.printf("  Total element throughput  : %,.0f element evaluations/sec (%.2fM ops/sec)%n",
                opsPerSec, opsPerSec / 1_000_000.0);
        // Reading the output keeps the work observable and gives a cheap cross-variant sanity check.
        System.out.printf("  Sink (last element)       : %.6f%n", sink.getAsDouble());
        if (DIAG && diagHook != null && label.endsWith("parallel")) {
            System.out.println(diagHook.get());
        }
        return new Result(label, avgMs, opsPerSec / 1_000_000.0);
    }

    private static void printSummary(List<Result> results) {
        System.out.println("\n================ SUMMARY ================");
        System.out.printf("%-34s %12s %14s%n", "Path", "ms/batch", "M ops/sec");
        for (Result r : results) {
            System.out.printf("%-34s %12.3f %14.2f%n", r.label(), r.avgMs(), r.mops());
        }
    }
}