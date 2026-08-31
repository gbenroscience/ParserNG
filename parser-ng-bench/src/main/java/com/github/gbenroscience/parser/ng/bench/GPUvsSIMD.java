/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.parser.ng.bench;

import com.github.gbenroscience.gpu.evaluator.opencl.OpenClCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.opencl.OpenClExpressionBridge;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandSegmentF64;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

/**
 *
 *
 *
 * ParserNG GPU (OpenCL) vs ParserNG SIMD (CPU) benchmark suite.
 *
 * mvn exec:exec -Dexec.executable="java" -Dexec.args="--add-modules
 * jdk.incubator.vector -classpath %classpath
 * com.github.gbenroscience.parser.ng.bench.GPUvsSIMD"
 *
 *
 * <p>
 * Expressions are a plain array of name/source pairs rather than a hardcoded
 * enum + switch. They live in the {@link #EXPRESSIONS} array (32 entries: the
 * original 30 bandwidth-bound expressions plus 2 compute-heavy ones --
 * GPU_HEAVY_TRANSCENDENTAL and GPU_STRESS -- added specifically to raise
 * arithmetic intensity per element for GPU-vs-SIMD comparisons). This lets the
 * suite grow (or shrink) just by editing that array, and lets {@link #main}
 * pick a subset of expressions to actually run based on interactive input,
 * without touching JMH's parameterization mechanism.
 *
 * <p>
 * Expression selection at runtime: when launched via {@link #main}, the process
 * prints the indexed expression list and reads a line from {@code System.in}
 * via {@link Scanner#nextLine()}. The line may be a single index ({@code "7"})
 * or a comma-separated list of indices ({@code "0,4,17,29"}); every expression
 * named by those indices is run. This only applies when going through
 * {@code main()} -- running the shaded jar directly through JMH's own launcher
 * bypasses the prompt and runs every expression in {@link #EXPRESSIONS}, since
 * {@code exprName}'s {@code @Param} values list all 30 names as the default
 * set.
 *
 * <p>
 * This suite exercises {@code SIMDCommandSegmentF64}'s and
 * {@code OpenClCompositeExpression}'s zero-copy {@code MemorySegment[]}/
 * {@code MemorySegment}-backed bulk evaluation paths directly -- the same
 * {@code applyBulk(MemorySegment[], MemorySegment)} /
 * {@code applyBulkParallel(MemorySegment[], MemorySegment)} shape used
 * elsewhere in this codebase, with input/output backed by a plain
 * {@link Arena} -- no Arrow buffers involved. Both evaluators are compiled
 * from a single shared {@link MathExpression} per trial so their
 * variable-to-slot mapping (from {@link MathExpression#getSlotItems()}) is
 * guaranteed identical, rather than risking two independently-parsed
 * instances disagreeing on slot order.
 *
 * <p>
 * CAVEAT -- unverified API guess: {@code OpenClCompositeExpression}'s exact
 * bulk-apply method name was not available when this file was written.
 * {@link #parserNGGPU} calls {@code gpuEval.apply(MemorySegment[], MemorySegment)},
 * inferred by symmetry with {@code SIMDVectorCompositeExpression}'s contract.
 * If the real method is named differently (e.g. {@code evaluate(...)},
 * {@code execute(...)}, or something requiring an explicit device queue /
 * completion handle rather than blocking synchronously), that single call is
 * the only line that needs to change -- input generation, output buffers, and
 * teardown are otherwise API-agnostic.
 *
 * <p>
 * Primary comparison per expression: ParserNG SIMD (serial and parallel) vs
 * ParserNG GPU (OpenCL), all along the {@code MemorySegment} path.
 */
@State(Scope.Benchmark)
@BenchmarkMode(org.openjdk.jmh.annotations.Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 2)
public class GPUvsSIMD {

    /*
     * -------------------------------------------------------------------------
     * Expression catalog
     * -------------------------------------------------------------------------
     */
    /**
     * One benchmarkable expression: a name paired with its ParserNG source
     * string.
     */
    public static final class ExpressionDef {

        public final String name;
        public final String parserExpr;

        public ExpressionDef(String name, String parserExpr) {
            this.name = name;
            this.parserExpr = parserExpr;
        }
    }

    /**
     * All 30 benchmarkable expressions, in display/index order. Index into this
     * array is exactly the index the user enters at the prompt in
     * {@link #main}.
     */
    public static final ExpressionDef[] EXPRESSIONS = new ExpressionDef[]{
        new ExpressionDef("DISTANCE", "sin(sqrt(x1*x1 + x2*x2 + x3*x3))"),
        new ExpressionDef("POLY", "(x1+x2)*(x1-x2) + x3*x3*x3"),
        new ExpressionDef("LOG_EXP", "ln(x1*x1 + 1) + exp(x2*0.001) - x3"),
        new ExpressionDef("TRIG_CHAIN", "sin(x1)*cos(x2) + tan(x3*0.1) - sin(x1*x2*0.0001) + sqrt(abs(x3)+1)"),
        new ExpressionDef("VARIABLE_POWER", "(x1+11.0)^(x2*0.0001 + 1.0)"),
        new ExpressionDef("SIMPLE_SUM", "x1 + x2 + x3"),
        new ExpressionDef("SIMPLE_PRODUCT", "x1*x2*x3"),
        new ExpressionDef("QUADRATIC", "x1*x1 + x2*x2 + x3*x3"),
        new ExpressionDef("CUBIC", "x1*x1*x1 + x2*x2*x2 + x3*x3*x3"),
        new ExpressionDef("DIVIDE_CHAIN", "(x1+x2) / (x3+1)"),
        new ExpressionDef("ABS_DIFF", "abs(x1-x2) + abs(x2-x3)"),
        new ExpressionDef("SQRT_SUM", "sqrt(x1*x1+x2*x2) + sqrt(x2*x2+x3*x3)"),
        new ExpressionDef("SIN_PRODUCT", "sin(x1)*sin(x2)*sin(x3)"),
        new ExpressionDef("COS_SUM", "cos(x1) + cos(x2) + cos(x3)"),
        new ExpressionDef("TAN_RATIO", "tan(x1*0.01) / (tan(x2*0.01) + 1)"),
        new ExpressionDef("LOG_CHAIN", "ln(x1*x1 + x2*x2 + 1)"),
        new ExpressionDef("EXP_CHAIN", "exp(x1*0.0001) * exp(x2*0.0001)"),
        new ExpressionDef("POWER_SQUARE", "(x1+5.0)^2.0"),
        new ExpressionDef("POWER_CUBE", "(x2+3.0)^3.0"),
        new ExpressionDef("POWER_MIXED", "(x1+2.0)^(x2*0.0001 + 0.5)"),
        new ExpressionDef("MIXED_TRIG_LOG", "sin(x1) * ln(x2*x2+1)"),
        new ExpressionDef("MIXED_EXP_TRIG", "exp(x1*0.0001) * cos(x2)"),
        new ExpressionDef("NESTED_SQRT", "sqrt(sqrt(x1*x1+x2*x2)+1)"),
        new ExpressionDef("DEEP_CHAIN", "sin(cos(x1*0.001)) + tan(x2*0.001)"),
        new ExpressionDef("WEIGHTED_SUM", "0.3*x1 + 0.5*x2 + 0.2*x3"),
        new ExpressionDef("NORMALIZED_DIFF", "(x1-x2) / (abs(x1)+abs(x2)+1)"),
        new ExpressionDef("HARMONIC", "1/(x1*x1+1) + 1/(x2*x2+1)"),
        new ExpressionDef("LOG_RATIO", "ln(x1*x1+1) / ln(x2*x2+2)"),
        new ExpressionDef("TRIG_POLY", "sin(x1)*x2 + cos(x2)*x3 - tan(x3*0.01)*x1"),
        new ExpressionDef("COMPOSITE", "sqrt(abs(x1*x2*x3)) + sin(x1+x2+x3) / exp(0.0001*abs(x3))"),
        // ---- Added: high-arithmetic-intensity expressions for GPU vs SIMD comparison ----
        // The 30 expressions above are all bandwidth-bound: 3 doubles in, 1 double
        // out (32 bytes/element), but only ~5-10 FLOPs of actual work. That FLOP:byte
        // ratio is close to the worst case for a discrete GPU -- the PCIe transfer
        // dominates and SIMD/parallel (which never leave DRAM) win by construction,
        // regardless of how well the GPU dispatch path itself is implemented. These
        // two keep the same 3-input/1-output shape (so the transfer cost per element
        // is identical) but multiply the compute per element by chaining many
        // transcendental calls, which is where a GPU's raw ALU throughput should
        // start to outweigh the transfer overhead. Every argument to ln/sqrt is kept
        // strictly positive and every tan() argument is scaled well away from its
        // asymptotes, so neither should produce NaN/Inf for x1,x2,x3 in [-10, 10].
        new ExpressionDef("GPU_HEAVY_TRANSCENDENTAL",
        "sin(cos(tan(x1*0.001))) + cos(sin(tan(x2*0.001))) + tan(sin(cos(x3*0.001)))"
        + " + exp(sin(x1*0.0001)) + exp(cos(x2*0.0001))"
        + " + ln(abs(x1*x2)+1) + ln(abs(x2*x3)+1) + sqrt(abs(x1*x3)+1)"
        + " + sin(x1*x2*x3*0.000001) + cos(x1+x2+x3) + tan((x1-x2+x3)*0.001)"
        + " + (x1+2.0)^(x2*0.00001 + 0.5) + (x2+3.0)^(x3*0.00001 + 0.5)"),
        // ~2x the transcendental call count of GPU_HEAVY_TRANSCENDENTAL above --
        // use this one if that one still doesn't tip the balance toward the GPU at
        // 8M elements; if even this loses to SIMD, the bottleneck is very likely the
        // per-call transfer/dispatch overhead itself (unpinned host memory, no
        // stream overlap -- see the earlier discussion), not insufficient FLOPs/byte.
        new ExpressionDef("GPU_STRESS",
        "sin(cos(tan(sin(x1*0.001)))) + cos(sin(tan(cos(x2*0.001)))) + tan(sin(cos(tan(x3*0.001))))"
        + " + exp(sin(x1*0.0001)) + exp(cos(x2*0.0001)) + exp(sin(x3*0.0001))"
        + " + ln(abs(x1*x2)+1) + ln(abs(x2*x3)+1) + ln(abs(x1*x3)+1)"
        + " + sqrt(abs(x1*x2)+1) + sqrt(abs(x2*x3)+1) + sqrt(abs(x1*x3)+1)"
        + " + sin(x1*x2*x3*0.000001) + cos(x1*x2*x3*0.000001) + tan((x1-x2+x3)*0.001)"
        + " + (x1+2.0)^(x2*0.00001 + 0.5) + (x2+3.0)^(x3*0.00001 + 0.5) + (x3+4.0)^(x1*0.00001 + 0.5)")
    };

    private static final Map<String, ExpressionDef> EXPRESSIONS_BY_NAME;

    static {
        Map<String, ExpressionDef> byName = new HashMap<>();
        for (ExpressionDef def : EXPRESSIONS) {
            byName.put(def.name, def);
        }
        EXPRESSIONS_BY_NAME = Map.copyOf(byName);
    }

    /*
     * -------------------------------------------------------------------------
     * JMH state
     * -------------------------------------------------------------------------
     */
    // NOTE: trimmed from the original 6-point size sweep to keep total
    // runtime sane now that it's crossed with 30 expressions instead of 5.
    @Param({
        "1024",
        "262144",
        "8388608"
    })
    private int size;

    // Default value set lists every expression by name, in EXPRESSIONS order.
    // When launched through main(), this default is overridden per the
    // indices entered at the prompt; running the shaded jar directly through
    // JMH's own launcher (bypassing main()) runs all 30, same as before.
    @Param({
        "DISTANCE", "POLY", "LOG_EXP", "TRIG_CHAIN", "VARIABLE_POWER",
        "SIMPLE_SUM", "SIMPLE_PRODUCT", "QUADRATIC", "CUBIC", "DIVIDE_CHAIN",
        "ABS_DIFF", "SQRT_SUM", "SIN_PRODUCT", "COS_SUM", "TAN_RATIO",
        "LOG_CHAIN", "EXP_CHAIN", "POWER_SQUARE", "POWER_CUBE", "POWER_MIXED",
        "MIXED_TRIG_LOG", "MIXED_EXP_TRIG", "NESTED_SQRT", "DEEP_CHAIN", "WEIGHTED_SUM",
        "NORMALIZED_DIFF", "HARMONIC", "LOG_RATIO", "TRIG_POLY", "COMPOSITE",
        "GPU_HEAVY_TRANSCENDENTAL", "GPU_STRESS"
    })
    private String exprName;

    private SIMDCommandSegmentF64.SIMDVectorCompositeExpression simdComd;
    private OpenClCompositeExpression gpuEval;

    // Trial-scoped: allocated once per (size, exprName) combination in
    // setupTrial(), closed in teardownTrial(). Deliberately NOT global/
    // never-closing (unlike HotBenchSegment's SEG_DATA_ARENA, which populates
    // a small fixed number of scenarios exactly once for the whole fork) --
    // this suite sweeps both size and exprName as @Param, so a global arena
    // would accumulate a fresh `size`-sized allocation per variable for every
    // trial in the sweep and never release any of it, which at the 8388608
    // (64MB/variable) end of the range would exhaust native memory well
    // before the sweep finished. Data is refreshed in place every iteration
    // (see setupIteration/createInput) without any further allocation, so
    // this stays bounded at "one trial's worth" of native memory at a time.
    private Arena trialArena;
    private MemorySegment[] inputSegments;
    private MemorySegment simdOutput;
    private MemorySegment gpuOutput;

    private MathExpression.Slot[] requiredSlots;
    private int slotCount;

    private final Random random = new Random(42);

    /*
     * -------------------------------------------------------------------------
     * Setup
     * -------------------------------------------------------------------------
     */
    @Setup(Level.Trial)
    public void setupTrial() throws Exception {
        ExpressionDef def = EXPRESSIONS_BY_NAME.get(exprName);
        if (def == null) {
            throw new IllegalStateException("Unknown expression name: " + exprName);
        }

        // Compiled once, shared by both evaluators below, so their slot
        // assignments (used to place each variable's MemorySegment at the
        // right index in inputSegments) are guaranteed to agree -- rather
        // than each independently parsing def.parserExpr and risking two
        // separately-derived slot orderings disagreeing in theory.
        MathExpression me = new MathExpression(def.parserExpr);
        requiredSlots = me.getSlotItems();
        slotCount = me.getRegistry().size();

        try {
            simdComd = SIMDCommandSegmentF64.getEvaluator(me);
            gpuEval = (OpenClCompositeExpression) OpenClExpressionBridge.compile(me);
        } catch (Throwable ex) {
            System.getLogger(GPUvsSIMD.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
            ex.printStackTrace();
        }

        trialArena = Arena.ofShared();
        inputSegments = new MemorySegment[slotCount];
        for (MathExpression.Slot slot : requiredSlots) {
            inputSegments[slot.getSlot()] = trialArena.allocate(ValueLayout.JAVA_DOUBLE, (long) size);
        }
        simdOutput = trialArena.allocate(ValueLayout.JAVA_DOUBLE, (long) size);
        gpuOutput = trialArena.allocate(ValueLayout.JAVA_DOUBLE, (long) size);
    }

    @Setup(Level.Iteration)
    public void setupIteration() {
        createInput();
    }

    /*
     * -------------------------------------------------------------------------
     * Input generation
     * -------------------------------------------------------------------------
     */
    private void createInput() {
        // Uniform in [-10, 10]: matches the range GPU_HEAVY_TRANSCENDENTAL and
        // GPU_STRESS were explicitly designed around (see their comments in
        // EXPRESSIONS above) -- keeps every ln/sqrt argument positive and every
        // tan() argument away from its asymptotes across all 32 expressions.
        // Overwrites the existing trial-scoped segments in place; see the
        // trialArena field comment for why this doesn't allocate per-iteration.
        for (MathExpression.Slot slot : requiredSlots) {
            MemorySegment seg = inputSegments[slot.getSlot()];
            for (int i = 0; i < size; i++) {
                double val = (random.nextDouble() * 20.0) - 10.0;
                seg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
            }
        }
    }

    /*
     * -------------------------------------------------------------------------
     * Benchmarks
     * -------------------------------------------------------------------------
     */
    @Benchmark
    public void parserNGSIMD(Blackhole bh) {
        simdComd.applyBulk(inputSegments, simdOutput);
        bh.consume(simdOutput);
    }

    @Benchmark
    public void parserNGGPU(Blackhole bh) throws Throwable {
        // See the class javadoc's CAVEAT: gpuEval.apply(...)'s exact method
        // name/signature is an inferred guess, not a verified API surface.
        gpuEval.applyBulk(inputSegments, gpuOutput);
        bh.consume(gpuOutput);
    }

    @Benchmark
    public void parserNGParallel(Blackhole bh) {
        simdComd.applyBulkParallel(inputSegments, simdOutput);
        bh.consume(simdOutput);
    }

    /*
     * -------------------------------------------------------------------------
     * Teardown
     * -------------------------------------------------------------------------
     */
    @TearDown(Level.Iteration)
    public void teardownIteration() {
        // Nothing to do: createInput() refreshes data in place within
        // trialArena's already-allocated segments, so there is no
        // per-iteration native allocation to release here.
    }

    @TearDown(Level.Trial)
    public void teardownTrial() {
        if (simdComd != null) {
            simdComd.close();
            simdComd = null;
        }
        if (gpuEval != null) {
            gpuEval.close();
            gpuEval = null;
        }
        if (trialArena != null) {
            trialArena.close();
            trialArena = null;
        }
        inputSegments = null;
        simdOutput = null;
        gpuOutput = null;
    }

    /*
     * -------------------------------------------------------------------------
     * Interactive expression selection
     * -------------------------------------------------------------------------
     */
    /**
     * Prints the indexed expression catalog and reads a comma-separated list of
     * indices from {@code System.in} via {@link Scanner#nextLine()}. Re-prompts
     * on empty input, an out-of-range index, or a non-integer token. Returns
     * the parsed, order-preserving (duplicates allowed) list of indices the
     * user selected.
     */
    private static List<Integer> promptForExpressionIndices(Scanner scanner) {
        System.out.println("Available expressions:");
        for (int i = 0; i < EXPRESSIONS.length; i++) {
            System.out.printf(Locale.ROOT, "  [%2d] %-18s %s%n", i, EXPRESSIONS[i].name, EXPRESSIONS[i].parserExpr);
        }

        while (true) {
            System.out.print("\nEnter index or comma-separated indices to benchmark (e.g. 0,4,17): ");
            String line = scanner.nextLine().trim();

            if (line.isEmpty()) {
                System.out.println("Please enter at least one index.");
                continue;
            }

            String[] tokens = line.split(",");
            List<Integer> parsed = new ArrayList<>();
            boolean valid = true;

            for (String token : tokens) {
                String trimmed = token.trim();
                try {
                    int idx = Integer.parseInt(trimmed);
                    if (idx < 0 || idx >= EXPRESSIONS.length) {
                        System.out.println("Index out of range (0-" + (EXPRESSIONS.length - 1) + "): " + idx);
                        valid = false;
                        break;
                    }
                    parsed.add(idx);
                } catch (NumberFormatException ex) {
                    System.out.println("Not a valid integer: '" + trimmed + "'");
                    valid = false;
                    break;
                }
            }

            if (valid) {
                return parsed;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        List<Integer> selectedIndices = promptForExpressionIndices(scanner);

        String[] selectedNames = new String[selectedIndices.size()];
        for (int i = 0; i < selectedIndices.size(); i++) {
            selectedNames[i] = EXPRESSIONS[selectedIndices.get(i)].name;
        }

        System.out.println("\nRunning JMH for: " + String.join(", ", selectedNames) + "\n");

        Options opt = new OptionsBuilder()
                .include(GPUvsSIMD.class.getSimpleName())
                .addProfiler(org.openjdk.jmh.profile.GCProfiler.class)
                .param("exprName", selectedNames)
                .jvmArgsAppend(
                        "--add-modules=jdk.incubator.vector",
                        "--add-opens=java.base/java.nio=ALL-UNNAMED",
                        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
                )
                .build();
        new Runner(opt).run();
    }
}