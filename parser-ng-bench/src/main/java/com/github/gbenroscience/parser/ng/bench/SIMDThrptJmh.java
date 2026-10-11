package com.github.gbenroscience.parser.ng.bench;

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
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import java.util.function.DoubleSupplier;
import java.util.function.Supplier;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.ChainedOptionsBuilder;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * JMH version of {@link SIMDThrptBench}: every bulk evaluation path, serial and parallel.
 *
 * <p>The score is <b>milliseconds per bulk payload</b> (one call over {@code elements} samples per
 * variable). To convert to the throughput the interactive harness prints:
 * {@code M elements/s = elements / 1000 / score_ms}, e.g. 1,000,017 elements at 1.5 ms = 666.7M ops/s.
 *
 * <p>Each parameter combination runs in its own forked JVM, so there is no JIT-profile pollution
 * between variants and no leftover worker threads from earlier evaluators (the evaluator is closed in
 * {@link #tearDown()}).
 *
 * <h2>Parameters</h2>
 * <ul>
 * <li>{@code variant}: 1 auto-vectorized {@code double[][]}, 2 base SIMD {@code double[][]},
 * 3 {@code SIMDCommandF32} arrays, 4 {@code SIMDCommandF64} arrays, 5/6 segment-class array overloads
 * (F32/F64), 7/8 flat {@code MemorySegment} (F64/F32), 9/10 {@code MemorySegment[]} (F64/F32).</li>
 * <li>{@code mode}: {@code serial} ({@code applyBulk}) or {@code parallel} ({@code applyBulkParallel}).</li>
 * <li>{@code expression}: prompted for by {@link #main(String[])}; the defaults below only apply when the
 * class is launched through {@code org.openjdk.jmh.Main}.</li>
 * <li>{@code workers}: 0 = evaluator default; otherwise {@code getEvaluator(me, workers)} for variants 3-10.</li>
 * <li>{@code elements}: samples per variable.</li>
 * </ul>
 *
 * <p><b>Running:</b> use this class's own {@link #main(String[])} (not {@code org.openjdk.jmh.Main}); it
 * includes <i>only</i> this class's benchmark, so other JMH classes in the same directory are never run.
 * It prompts like {@link SIMDThrptBench} (expression, then variant, 0 = all) and runs the chosen variant(s) in
 * both serial and parallel mode:
 * <pre>
 * mvn test-compile exec:exec -Dexec.classpathScope=test -Dexec.executable=java \
 *   -Dexec.args="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -classpath %classpath \
 *   com.github.gbenroscience.simdext.turbo.tools.bench.SIMDThrptJmh"
 * </pre>
 * One-shot (no prompts): pass {@code <expression> <variant 0-10>} as program arguments.
 *
 * <p>Optional system properties on the launching JVM: {@code bench.workers=N}, {@code bench.elements=N},
 * {@code bench.mode=serial|parallel} (default both), {@code parserng.pin=true}, {@code parserng.diag=true}.
 * The module flags and the two {@code parserng.*} properties are forwarded to the forked JVMs explicitly.
 * Each fork prints a {@code [config]} line showing what it saw. {@code -prof gc} is available through
 * {@code org.openjdk.jmh.Main} if you need profilers.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@Threads(1) // the evaluators own their worker threads; never drive one evaluator from several JMH threads
@State(Scope.Benchmark)
public class SIMDThrptJmh {

    @Param({"x^2+1", "0.39894228/x1*exp(-((x2-x3)*(x2-x3))/(2*x1*x1))"})
    public String expression;

    @Param({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"})
    public int variant;

    @Param({"serial", "parallel"})
    public String mode;

    @Param({"1000017"}) // trailing 17 forces vector-lane remainders
    public int elements;

    @Param({"0"})
    public int workers;

    @FunctionalInterface
    private interface Bulk {
        void run() throws Throwable;
    }

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

    private Bulk action;
    private DoubleSupplier sink;
    private Supplier<String> diag;
    private Object evaluator;
    private Arena arena;
    private boolean parallel;

    @Setup(Level.Trial)
    public void setup() throws Throwable {
        parallel = "parallel".equalsIgnoreCase(mode);
        MathExpression me = new MathExpression(expression);
        int varCount = Math.max(1, me.getVariables().getKey().length);

        System.out.printf("[config] variant=%d mode=%s expression=%s vars=%d elements=%d workers=%s pinning=%s pinOrder=%s%n",
                variant, mode, expression, varCount, elements, workers > 0 ? workers : "default",
                CPUPinner.isPinningEnabled(), java.util.Arrays.toString(CPUPinner.pinOrder()));

        switch (variant) {
            case 1 -> {
                var ev = VectorTurboEvaluator.getEvaluator(me);
                evaluator = ev;
                f64(ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            case 2 -> {
                var ev = SIMDVectorTurboEvaluator.getEvaluator(me);
                evaluator = ev;
                f64(ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            case 3 -> {
                var ev = workers > 0 ? SIMDCommandF32.getEvaluator(me, workers) : SIMDCommandF32.getEvaluator(me);
                evaluator = ev;
                diag = SIMDCommandF32.SIMDVectorCompositeExpression::diagReportAndReset;
                f32(ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            case 4 -> {
                var ev = workers > 0 ? SIMDCommandF64.getEvaluator(me, workers) : SIMDCommandF64.getEvaluator(me);
                evaluator = ev;
                diag = SIMDCommandF64.SIMDVectorCompositeExpression::diagReportAndReset;
                f64(ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            case 5 -> {
                var ev = workers > 0 ? SIMDCommandSegmentF32.getEvaluator(me, workers) : SIMDCommandSegmentF32.getEvaluator(me);
                evaluator = ev;
                diag = SIMDCommandSegmentF32.SIMDVectorCompositeExpression::diagReportAndReset;
                f32(ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            case 6 -> {
                var ev = workers > 0 ? SIMDCommandSegmentF64.getEvaluator(me, workers) : SIMDCommandSegmentF64.getEvaluator(me);
                evaluator = ev;
                diag = SIMDCommandSegmentF64.SIMDVectorCompositeExpression::diagReportAndReset;
                f64(ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            case 7 -> {
                var ev = workers > 0 ? SIMDCommandSegmentF64.getEvaluator(me, workers) : SIMDCommandSegmentF64.getEvaluator(me);
                evaluator = ev;
                diag = SIMDCommandSegmentF64.SIMDVectorCompositeExpression::diagReportAndReset;
                flatSegment(true, ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            case 8 -> {
                var ev = workers > 0 ? SIMDCommandSegmentF32.getEvaluator(me, workers) : SIMDCommandSegmentF32.getEvaluator(me);
                evaluator = ev;
                diag = SIMDCommandSegmentF32.SIMDVectorCompositeExpression::diagReportAndReset;
                flatSegment(false, ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            case 9 -> {
                var ev = workers > 0 ? SIMDCommandSegmentF64.getEvaluator(me, workers) : SIMDCommandSegmentF64.getEvaluator(me);
                evaluator = ev;
                diag = SIMDCommandSegmentF64.SIMDVectorCompositeExpression::diagReportAndReset;
                segmentArray(true, ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            case 10 -> {
                var ev = workers > 0 ? SIMDCommandSegmentF32.getEvaluator(me, workers) : SIMDCommandSegmentF32.getEvaluator(me);
                evaluator = ev;
                diag = SIMDCommandSegmentF32.SIMDVectorCompositeExpression::diagReportAndReset;
                segmentArray(false, ev::applyBulk, ev::applyBulkParallel, varCount);
            }
            default ->
                throw new IllegalArgumentException("variant must be 1-10, got " + variant);
        }
    }

    @Benchmark
    public void bulk() throws Throwable {
        action.run();
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        // Cheap cross-variant correctness check: same-precision variants should print the same value.
        if (sink != null) {
            System.out.printf("[check] variant=%d mode=%s last element=%.6f%n", variant, mode, sink.getAsDouble());
        }
        if (parallel && diag != null && Boolean.getBoolean("parserng.diag")) {
            System.out.println(diag.get());
        }
        // Stops the evaluator's worker threads so nothing leaks into later forks' measurements.
        if (evaluator instanceof AutoCloseable c) {
            try {
                c.close();
            } catch (Exception ignored) {
                // best effort
            }
        }
        if (arena != null) {
            arena.close();
            arena = null;
        }
    }

    // ------------------------------------------------------------------ data layouts

    private void f64(F64Bulk seq, F64Bulk par, int varCount) {
        final double[][] in = new double[varCount][elements];
        final double[] out = new double[elements];
        for (int v = 0; v < varCount; v++) {
            for (int i = 0; i < elements; i++) {
                in[v][i] = 1.0 + (i * 0.01) + (v * 0.5);
            }
        }
        final F64Bulk f = parallel ? par : seq;
        final int last = elements - 1;
        action = () -> f.apply(in, out);
        sink = () -> out[last];
    }

    private void f32(F32Bulk seq, F32Bulk par, int varCount) {
        final float[][] in = new float[varCount][elements];
        final float[] out = new float[elements];
        for (int v = 0; v < varCount; v++) {
            for (int i = 0; i < elements; i++) {
                in[v][i] = 1.0f + (i * 0.01f) + (v * 0.5f);
            }
        }
        final F32Bulk f = parallel ? par : seq;
        final int last = elements - 1;
        action = () -> f.apply(in, out);
        sink = () -> out[last];
    }

    /** Flat layout: variable-major blocks, element (v, i) at index v*N + i; output element i at index i. */
    private void flatSegment(boolean f64, SegBulk seq, SegBulk par, int varCount) {
        final long n = elements;
        final long elemSize = f64 ? Double.BYTES : Float.BYTES;
        // Shared arena: worker threads must be able to read the segments in parallel mode.
        arena = Arena.ofShared();
        final MemorySegment in = arena.allocate(varCount * n * elemSize, 64);
        final MemorySegment out = arena.allocate(n * elemSize, 64);
        for (int v = 0; v < varCount; v++) {
            for (long i = 0; i < n; i++) {
                long idx = v * n + i;
                if (f64) {
                    in.setAtIndex(ValueLayout.JAVA_DOUBLE, idx, 1.0 + (i * 0.01) + (v * 0.5));
                } else {
                    in.setAtIndex(ValueLayout.JAVA_FLOAT, idx, 1.0f + (i * 0.01f) + (v * 0.5f));
                }
            }
        }
        final SegBulk f = parallel ? par : seq;
        action = () -> f.apply(in, out);
        sink = f64
                ? () -> out.getAtIndex(ValueLayout.JAVA_DOUBLE, n - 1)
                : () -> out.getAtIndex(ValueLayout.JAVA_FLOAT, n - 1);
    }

    /** One segment per variable (the off-heap analogue of double[][] / float[][]). */
    private void segmentArray(boolean f64, SegArrBulk seq, SegArrBulk par, int varCount) {
        final long n = elements;
        final long elemSize = f64 ? Double.BYTES : Float.BYTES;
        arena = Arena.ofShared();
        final MemorySegment[] in = new MemorySegment[varCount];
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
        final MemorySegment out = arena.allocate(n * elemSize, 64);
        final SegArrBulk f = parallel ? par : seq;
        action = () -> f.apply(in, out);
        sink = f64
                ? () -> out.getAtIndex(ValueLayout.JAVA_DOUBLE, n - 1)
                : () -> out.getAtIndex(ValueLayout.JAVA_FLOAT, n - 1);
    }

    private static final String[] MENU = {
        "1============================Auto-Vectorizer (slowest)",
        "2============================Base SIMD (Vector API 21)",
        "3============================SIMD F32 (arrays)",
        "4============================SIMD F64 (arrays)",
        "5============================SIMD F32 zero-copy (arrays)",
        "6============================SIMD F64 zero-copy (arrays)",
        "7============================SIMD F64 zero-copy (flat MemorySegment)",
        "8============================SIMD F32 zero-copy (flat MemorySegment)",
        "9============================SIMD F64 zero-copy (MemorySegment[])",
        "10===========================SIMD F32 zero-copy (MemorySegment[])",
        "0============================All of the above"
    };

    /**
     * Interactive entry point, mirroring {@link SIMDThrptBench}: prompts for an expression and a variant,
     * then runs ONLY this class's JMH benchmark with those values. Blank or {@code q} at the expression
     * prompt quits; {@code q} at the variant prompt goes back to the expression prompt.
     * <p>Program arguments {@code <expression> <variant>} run once without prompts.
     */
    public static void main(String[] args) throws Exception {
        System.out.println("Pin order (logical CPUs): " + java.util.Arrays.toString(CPUPinner.pinOrder())
                + " | pinning=" + CPUPinner.isPinningEnabled()
                + " | workers=" + (Integer.getInteger("bench.workers") == null ? "default" : Integer.getInteger("bench.workers")));

        String cliExpr = (args.length > 0 && !args[0].isBlank()) ? args[0] : null;
        Integer cliVariant = (args.length > 1) ? Integer.valueOf(args[1].trim()) : null;
        Scanner sc = new Scanner(System.in);

        while (true) {
            String expr;
            if (cliExpr != null) {
                expr = cliExpr;
                cliExpr = null; // command-line expression is used for the first pass only
            } else {
                System.out.println("\nEnter math expression (e.g., sin(sqrt(x^2+y^2))), or blank/q to quit:");
                if (!sc.hasNextLine()) {
                    return;
                }
                expr = sc.nextLine().trim();
            }
            if (expr.isBlank() || expr.equalsIgnoreCase("q")) {
                return;
            }
            // Confirm what was read: some Windows launch paths (e.g. mvn exec:exec) do not echo typed text.
            System.out.println("Expression : " + expr);

            try {
                new MathExpression(expr); // fail fast on a bad expression, before any fork is started
            } catch (Exception e) {
                System.err.println("Could not parse expression.");
                e.printStackTrace();
                if (cliVariant != null) {
                    return;
                }
                continue;
            }

            int variant;
            if (cliVariant != null) {
                variant = cliVariant;
            } else {
                System.out.println("Select bulk evaluator variant (0 = run all):");
                for (String line : MENU) {
                    System.out.println(line);
                }
                if (!sc.hasNextLine()) {
                    return;
                }
                String line = sc.nextLine().trim();
                if (line.equalsIgnoreCase("q")) {
                    continue; // back to the expression prompt
                }
                try {
                    variant = Integer.parseInt(line);
                } catch (NumberFormatException e) {
                    System.err.println("Please enter a number from 0 to 10.");
                    continue;
                }
            }
            if (variant < 0 || variant > 10) {
                System.err.println("Variant must be 0-10, got " + variant);
                if (cliVariant != null) {
                    return;
                }
                continue;
            }
            System.out.println("Variant    : " + (variant == 0 ? "0 (all variants)" : MENU[variant - 1].replaceFirst("^\\d+=+", variant + " ")));
            if (variant == 0) {
                System.out.println("Running 10 variants x 2 modes in separate forks; this takes a few minutes.");
            }

            try {
                runJmh(expr, variant);
            } catch (RunnerException e) {
                System.err.println("JMH run failed: " + e.getMessage());
                e.printStackTrace();
            }
            if (cliVariant != null) {
                return; // one-shot mode
            }
        }
    }

    /** Runs only this class's benchmark, restricted to the chosen expression/variant. */
    private static void runJmh(String expr, int variant) throws RunnerException {
        // Exact match on this class's benchmark method, so other JMH classes in the directory are not run.
        ChainedOptionsBuilder b = new OptionsBuilder()
                .include("^" + Pattern.quote(SIMDThrptJmh.class.getName()) + "\\.bulk$")
                .param("expression", expr);
        if (variant != 0) {
            b.param("variant", Integer.toString(variant));
        }
        String mode = System.getProperty("bench.mode");
        if (mode != null && !mode.isBlank() && !mode.equalsIgnoreCase("both")) {
            b.param("mode", mode.toLowerCase());
        }
        Integer w = Integer.getInteger("bench.workers");
        if (w != null) {
            b.param("workers", Integer.toString(w));
        }
        Integer n = Integer.getInteger("bench.elements");
        if (n != null) {
            b.param("elements", Integer.toString(n));
        }

        // jvmArgsAppend on the Options REPLACES the annotation's value, so repeat the module flags here.
        List<String> jvm = new ArrayList<>(List.of(
                "--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"));
        for (String key : new String[]{"parserng.pin", "parserng.diag"}) {
            String v = System.getProperty(key);
            if (v != null) {
                jvm.add("-D" + key + "=" + v);
            }
        }
        b.jvmArgsAppend(jvm.toArray(new String[0]));

        new Runner(b.build()).run();
    }
}