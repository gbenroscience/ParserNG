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

import com.github.gbenroscience.arrow.tools.box.ArrowBulkEvaluator;
import com.github.gbenroscience.arrow.tools.box.ArrowGpuBulkEvaluator;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import org.apache.arrow.gandiva.expression.TreeBuilder;
import org.apache.arrow.gandiva.expression.TreeNode;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.types.pojo.ArrowType;
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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.TimeUnit; 

/**
 *
 *
 *
 * ParserNG Arrow vs Apache Arrow Gandiva benchmark suite.
 *
 * mvn exec:exec -Dexec.executable="java" -Dexec.args="--add-modules
 * jdk.incubator.vector -classpath %classpath
 * com.github.gbenroscience.parser.ng.bench.GandivaVsParserNGArrowBenchmark"
 *
 *
 * <p>
 * Expressions are no longer a hardcoded enum + switch. They live in the
 * {@link #EXPRESSIONS} array (30 entries), each pairing a ParserNG source
 * string with a matching Gandiva {@link TreeNode} builder. This lets the suite
 * grow (or shrink) just by editing that array, and lets {@link #main} pick a
 * subset of expressions to actually run based on interactive input, without
 * touching JMH's parameterization mechanism.
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
 * CAVEAT -- please verify before trusting the numbers: the Gandiva function
 * names used below ("add", "subtract", "multiply", "divide", "sqrt", "abs",
 * "sin", "cos", "tan", "log", "exp", "power") are Gandiva's standard math
 * function registry as I understand it, but I have not run this against Gandiva
 * myself to confirm every name resolves -- "log" specifically is assumed to
 * mean natural log (matching ParserNG's "ln"), which is the common Gandiva
 * convention but worth double-checking against Gandiva's actual registry if any
 * expression throws a "function not found" error. "divide" in particular is
 * exercised by several of the new expressions (HARMONIC, LOG_RATIO, TAN_RATIO,
 * NORMALIZED_DIFF, DIVIDE_CHAIN, COMPOSITE) and was not used anywhere in the
 * original five-expression suite, so it's the newest unverified surface here.
 * ParserNG's own function names ("ln", "^", "sin", "cos", "tan", "sqrt", "abs",
 * "exp") are taken from confirmed working usage elsewhere in this codebase, but
 * adjust to whatever ParserNG's actual token names are if any of these don't
 * parse.
 *
 * <p>
 * Primary comparison per expression: ParserNG Arrow SIMD, single-threaded and
 * parallel vs Gandiva Projector
 */
@State(Scope.Benchmark)
@BenchmarkMode(org.openjdk.jmh.annotations.Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 2)
public class ParserNGArrowVSGPUBenchmark {

    /*
     * -------------------------------------------------------------------------
     * Expression catalog
     * -------------------------------------------------------------------------
     */
    /**
     * One benchmarkable expression: a ParserNG source string paired with a
     * Gandiva tree builder.
     */
    public static final class ExpressionDef {

        public final String name;
        public final String parserExpr; 

        public ExpressionDef(String name, String parserExpr) {
            this.name = name;
            this.parserExpr = parserExpr; 
        }
    } 
    // ---- small helpers so the 30 builders below read like the expressions they encode ----
    private static TreeNode add(TreeNode a, TreeNode b, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("add", List.of(a, b), t);
    }

    private static TreeNode sub(TreeNode a, TreeNode b, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("subtract", List.of(a, b), t);
    }

    private static TreeNode mul(TreeNode a, TreeNode b, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("multiply", List.of(a, b), t);
    }

    private static TreeNode div(TreeNode a, TreeNode b, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("divide", List.of(a, b), t);
    }

    private static TreeNode sqrtFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("sqrt", List.of(a), t);
    }

    private static TreeNode absFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("abs", List.of(a), t);
    }

    private static TreeNode sinFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("sin", List.of(a), t);
    }

    private static TreeNode cosFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("cos", List.of(a), t);
    }

    private static TreeNode tanFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("tan", List.of(a), t);
    }

    private static TreeNode lnFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("log", List.of(a), t);
    }

    private static TreeNode expFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("exp", List.of(a), t);
    }

    private static TreeNode pow(TreeNode a, TreeNode b, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("power", List.of(a, b), t);
    }

    private static TreeNode lit(double v) {
        return TreeBuilder.makeLiteral(v);
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
        new ExpressionDef("COMPOSITE", "sqrt(abs(x1*x2*x3)) + sin(x1+x2+x3) / exp(0.0001*abs(x3))")
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
        "NORMALIZED_DIFF", "HARMONIC", "LOG_RATIO", "TRIG_POLY", "COMPOSITE"
    })
    private String exprName;

    private BufferAllocator allocator;

    private Float8Vector x1;
    private Float8Vector x2;
    private Float8Vector x3;

    private Float8Vector parserOutput; 

    private Map<String, Float8Vector> parserColumns;

    private ArrowBulkEvaluator parserSIMDEvaluator;
    private ArrowGpuBulkEvaluator parserGpuEvaluator;
 
 

    /*
     * -------------------------------------------------------------------------
     * Setup
     * -------------------------------------------------------------------------
     */
    @Setup(Level.Trial)
    public void setupTrial() throws Exception {
        allocator = new RootAllocator(Long.MAX_VALUE);

        ExpressionDef def = EXPRESSIONS_BY_NAME.get(exprName);
        if (def == null) {
            throw new IllegalStateException("Unknown expression name: " + exprName);
        }

        try {
            parserSIMDEvaluator = ArrowBulkEvaluator.compile(def.parserExpr);
        } catch (Throwable ex) {
            System.getLogger(ParserNGArrowVSGPUBenchmark.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }

        try {
            parserGpuEvaluator = ArrowGpuBulkEvaluator.compile(def.parserExpr);
        } catch (Throwable ex) {
            System.getLogger(ParserNGArrowVSGPUBenchmark.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }
 
    }

    @Setup(Level.Iteration)
    public void setupIteration() {
        createInput();

        parserOutput = ArrowBulkEvaluator.allocateOutput(
                allocator,
                "parser_ng_result",
                size
        );
 

        parserColumns = Map.of("x1", x1, "x2", x2, "x3", x3);
 
    }

    /*
     * -------------------------------------------------------------------------
     * Input generation
     * -------------------------------------------------------------------------
     */
    private void createInput() {
        x1 = new Float8Vector("x1", allocator);
        x2 = new Float8Vector("x2", allocator);
        x3 = new Float8Vector("x3", allocator);

        x1.allocateNew(size);
        x2.allocateNew(size);
        x3.allocateNew(size);

        for (int i = 0; i < size; i++) {
            double t = i * 0.0001;
            x1.set(i, Math.sin(t) * 10.0);
            x2.set(i, Math.cos(t * 0.7) * 10.0);
            x3.set(i, Math.sin(t * 1.3) * Math.cos(t * 0.31) * 10.0);
        }

        x1.setValueCount(size);
        x2.setValueCount(size);
        x3.setValueCount(size);
    }
 

    /*
     * -------------------------------------------------------------------------
     * Benchmarks
     * -------------------------------------------------------------------------
     */
    @Benchmark
    public void parserNGSIMD(Blackhole bh) {
        parserSIMDEvaluator.evaluate(
                parserColumns,
                parserOutput,
                NullPolicy.IGNORE,
                false
        );
        bh.consume(parserOutput);
    }

    @Benchmark
    public void parserNGGPU(Blackhole bh) {
        parserGpuEvaluator.evaluate(
                parserColumns,
                parserOutput,
                NullPolicy.IGNORE
        );
        bh.consume(parserOutput);
    }
 

    @Benchmark
    public void parserNGParallel(Blackhole bh) {
        parserSIMDEvaluator.evaluate(
                parserColumns,
                parserOutput,
                NullPolicy.IGNORE,
                true
        );
        bh.consume(parserOutput);
    }

 
    /*
     * -------------------------------------------------------------------------
     * Teardown
     * -------------------------------------------------------------------------
     */
    @TearDown(Level.Iteration)
    public void teardownIteration() {
        if (parserOutput != null) {
            parserOutput.close();
            parserOutput = null;
        }
 
        if (x1 != null) {
            x1.close();
            x1 = null;
        }
        if (x2 != null) {
            x2.close();
            x2 = null;
        }
        if (x3 != null) {
            x3.close();
            x3 = null;
        }
    }

    @TearDown(Level.Trial)
    public void teardownTrial() {
        if (parserSIMDEvaluator != null) {
            parserSIMDEvaluator.close();
            parserSIMDEvaluator = null;
        }
        if (allocator != null) {
            allocator.close();
            allocator = null;
        }
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
        // Run correctness across every expression FIRST.
       // runAllCorrectnessChecks();

        Scanner scanner = new Scanner(System.in);
        List<Integer> selectedIndices = promptForExpressionIndices(scanner);

        String[] selectedNames = new String[selectedIndices.size()];
        for (int i = 0; i < selectedIndices.size(); i++) {
            selectedNames[i] = EXPRESSIONS[selectedIndices.get(i)].name;
        }

        System.out.println("\nRunning JMH for: " + String.join(", ", selectedNames) + "\n");

        Options opt = new OptionsBuilder()
                .include(ParserNGArrowVSGPUBenchmark.class.getSimpleName())
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
