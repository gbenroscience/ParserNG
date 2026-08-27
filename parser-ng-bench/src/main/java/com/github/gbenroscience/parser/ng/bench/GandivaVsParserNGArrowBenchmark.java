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
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import org.apache.arrow.gandiva.expression.ExpressionTree;
import org.apache.arrow.gandiva.expression.TreeBuilder;
import org.apache.arrow.gandiva.expression.TreeNode;
import org.apache.arrow.gandiva.evaluator.Projector;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;
import org.apache.arrow.gandiva.exceptions.GandivaException;
import org.apache.arrow.vector.types.FloatingPointPrecision;

/**
 * 
 * 
 * 
 * ParserNG Arrow vs Apache Arrow Gandiva benchmark suite.
 *  
 * mvn exec:exec -Dexec.executable="java" -Dexec.args="--add-modules jdk.incubator.vector -classpath %classpath com.github.gbenroscience.parser.ng.bench.GandivaVsParserNGArrowBenchmark"
 * 
 * 
 * <p>Expressions are no longer a hardcoded enum + switch. They live in the
 * {@link #EXPRESSIONS} array (30 entries), each pairing a ParserNG source
 * string with a matching Gandiva {@link TreeNode} builder. This lets the
 * suite grow (or shrink) just by editing that array, and lets {@link #main}
 * pick a subset of expressions to actually run based on interactive input,
 * without touching JMH's parameterization mechanism.
 *
 * <p>Expression selection at runtime: when launched via {@link #main}, the
 * process prints the indexed expression list and reads a line from
 * {@code System.in} via {@link Scanner#nextLine()}. The line may be a single
 * index ({@code "7"}) or a comma-separated list of indices
 * ({@code "0,4,17,29"}); every expression named by those indices is run.
 * This only applies when going through {@code main()} -- running the shaded
 * jar directly through JMH's own launcher bypasses the prompt and runs every
 * expression in {@link #EXPRESSIONS}, since {@code exprName}'s {@code @Param}
 * values list all 30 names as the default set.
 *
 * <p>CAVEAT -- please verify before trusting the numbers: the Gandiva
 * function names used below ("add", "subtract", "multiply", "divide",
 * "sqrt", "abs", "sin", "cos", "tan", "log", "exp", "power") are Gandiva's
 * standard math function registry as I understand it, but I have not run
 * this against Gandiva myself to confirm every name resolves -- "log"
 * specifically is assumed to mean natural log (matching ParserNG's "ln"),
 * which is the common Gandiva convention but worth double-checking against
 * Gandiva's actual registry if any expression throws a "function not found"
 * error. "divide" in particular is exercised by several of the new
 * expressions (HARMONIC, LOG_RATIO, TAN_RATIO, NORMALIZED_DIFF,
 * DIVIDE_CHAIN, COMPOSITE) and was not used anywhere in the original
 * five-expression suite, so it's the newest unverified surface here.
 * ParserNG's own function names ("ln", "^", "sin", "cos", "tan", "sqrt",
 * "abs", "exp") are taken from confirmed working usage elsewhere in this
 * codebase, but adjust to whatever ParserNG's actual token names are if any
 * of these don't parse.
 *
 * <p>Primary comparison per expression:
 * ParserNG Arrow SIMD, single-threaded and parallel vs Gandiva Projector
 */
@State(Scope.Benchmark)
@BenchmarkMode(org.openjdk.jmh.annotations.Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 2)
public class GandivaVsParserNGArrowBenchmark {

    /*
     * -------------------------------------------------------------------------
     * Expression catalog
     * -------------------------------------------------------------------------
     */

    /** One benchmarkable expression: a ParserNG source string paired with a Gandiva tree builder. */
    public static final class ExpressionDef {

        public final String name;
        public final String parserExpr;
        public final GandivaTreeBuilder gandivaBuilder;

        public ExpressionDef(String name, String parserExpr, GandivaTreeBuilder gandivaBuilder) {
            this.name = name;
            this.parserExpr = parserExpr;
            this.gandivaBuilder = gandivaBuilder;
        }
    }

    @FunctionalInterface
    public interface GandivaTreeBuilder {
        TreeNode build(TreeNode x1, TreeNode x2, TreeNode x3, ArrowType.FloatingPoint doubleType);
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
     * All 30 benchmarkable expressions, in display/index order. Index into
     * this array is exactly the index the user enters at the prompt in
     * {@link #main}.
     */
    public static final ExpressionDef[] EXPRESSIONS = new ExpressionDef[]{
        new ExpressionDef("DISTANCE", "sin(sqrt(x1*x1 + x2*x2 + x3*x3))",
        (x1, x2, x3, t) -> sinFn(sqrtFn(add(add(mul(x1, x1, t), mul(x2, x2, t), t), mul(x3, x3, t), t), t), t)),
        new ExpressionDef("POLY", "(x1+x2)*(x1-x2) + x3*x3*x3",
        (x1, x2, x3, t) -> add(mul(add(x1, x2, t), sub(x1, x2, t), t), mul(mul(x3, x3, t), x3, t), t)),
        new ExpressionDef("LOG_EXP", "ln(x1*x1 + 1) + exp(x2*0.001) - x3",
        (x1, x2, x3, t) -> sub(add(lnFn(add(mul(x1, x1, t), lit(1.0), t), t), expFn(mul(x2, lit(0.001), t), t), t), x3, t)),
        new ExpressionDef("TRIG_CHAIN", "sin(x1)*cos(x2) + tan(x3*0.1) - sin(x1*x2*0.0001) + sqrt(abs(x3)+1)",
        (x1, x2, x3, t) -> add(
        sub(add(mul(sinFn(x1, t), cosFn(x2, t), t), tanFn(mul(x3, lit(0.1), t), t), t),
        sinFn(mul(mul(x1, x2, t), lit(0.0001), t), t), t),
        sqrtFn(add(absFn(x3, t), lit(1.0), t), t), t)),
        new ExpressionDef("VARIABLE_POWER", "(x1+11.0)^(x2*0.0001 + 1.0)",
        (x1, x2, x3, t) -> pow(add(x1, lit(11.0), t), add(mul(x2, lit(0.0001), t), lit(1.0), t), t)),
        new ExpressionDef("SIMPLE_SUM", "x1 + x2 + x3",
        (x1, x2, x3, t) -> add(add(x1, x2, t), x3, t)),
        new ExpressionDef("SIMPLE_PRODUCT", "x1*x2*x3",
        (x1, x2, x3, t) -> mul(mul(x1, x2, t), x3, t)),
        new ExpressionDef("QUADRATIC", "x1*x1 + x2*x2 + x3*x3",
        (x1, x2, x3, t) -> add(add(mul(x1, x1, t), mul(x2, x2, t), t), mul(x3, x3, t), t)),
        new ExpressionDef("CUBIC", "x1*x1*x1 + x2*x2*x2 + x3*x3*x3",
        (x1, x2, x3, t) -> add(add(mul(mul(x1, x1, t), x1, t), mul(mul(x2, x2, t), x2, t), t), mul(mul(x3, x3, t), x3, t), t)),
        new ExpressionDef("DIVIDE_CHAIN", "(x1+x2) / (x3+1)",
        (x1, x2, x3, t) -> div(add(x1, x2, t), add(x3, lit(1.0), t), t)),
        new ExpressionDef("ABS_DIFF", "abs(x1-x2) + abs(x2-x3)",
        (x1, x2, x3, t) -> add(absFn(sub(x1, x2, t), t), absFn(sub(x2, x3, t), t), t)),
        new ExpressionDef("SQRT_SUM", "sqrt(x1*x1+x2*x2) + sqrt(x2*x2+x3*x3)",
        (x1, x2, x3, t) -> add(sqrtFn(add(mul(x1, x1, t), mul(x2, x2, t), t), t), sqrtFn(add(mul(x2, x2, t), mul(x3, x3, t), t), t), t)),
        new ExpressionDef("SIN_PRODUCT", "sin(x1)*sin(x2)*sin(x3)",
        (x1, x2, x3, t) -> mul(mul(sinFn(x1, t), sinFn(x2, t), t), sinFn(x3, t), t)),
        new ExpressionDef("COS_SUM", "cos(x1) + cos(x2) + cos(x3)",
        (x1, x2, x3, t) -> add(add(cosFn(x1, t), cosFn(x2, t), t), cosFn(x3, t), t)),
        new ExpressionDef("TAN_RATIO", "tan(x1*0.01) / (tan(x2*0.01) + 1)",
        (x1, x2, x3, t) -> div(tanFn(mul(x1, lit(0.01), t), t), add(tanFn(mul(x2, lit(0.01), t), t), lit(1.0), t), t)),
        new ExpressionDef("LOG_CHAIN", "ln(x1*x1 + x2*x2 + 1)",
        (x1, x2, x3, t) -> lnFn(add(add(mul(x1, x1, t), mul(x2, x2, t), t), lit(1.0), t), t)),
        new ExpressionDef("EXP_CHAIN", "exp(x1*0.0001) * exp(x2*0.0001)",
        (x1, x2, x3, t) -> mul(expFn(mul(x1, lit(0.0001), t), t), expFn(mul(x2, lit(0.0001), t), t), t)),
        new ExpressionDef("POWER_SQUARE", "(x1+5.0)^2.0",
        (x1, x2, x3, t) -> pow(add(x1, lit(5.0), t), lit(2.0), t)),
        new ExpressionDef("POWER_CUBE", "(x2+3.0)^3.0",
        (x1, x2, x3, t) -> pow(add(x2, lit(3.0), t), lit(3.0), t)),
        new ExpressionDef("POWER_MIXED", "(x1+2.0)^(x2*0.0001 + 0.5)",
        (x1, x2, x3, t) -> pow(add(x1, lit(2.0), t), add(mul(x2, lit(0.0001), t), lit(0.5), t), t)),
        new ExpressionDef("MIXED_TRIG_LOG", "sin(x1) * ln(x2*x2+1)",
        (x1, x2, x3, t) -> mul(sinFn(x1, t), lnFn(add(mul(x2, x2, t), lit(1.0), t), t), t)),
        new ExpressionDef("MIXED_EXP_TRIG", "exp(x1*0.0001) * cos(x2)",
        (x1, x2, x3, t) -> mul(expFn(mul(x1, lit(0.0001), t), t), cosFn(x2, t), t)),
        new ExpressionDef("NESTED_SQRT", "sqrt(sqrt(x1*x1+x2*x2)+1)",
        (x1, x2, x3, t) -> sqrtFn(add(sqrtFn(add(mul(x1, x1, t), mul(x2, x2, t), t), t), lit(1.0), t), t)),
        new ExpressionDef("DEEP_CHAIN", "sin(cos(x1*0.001)) + tan(x2*0.001)",
        (x1, x2, x3, t) -> add(sinFn(cosFn(mul(x1, lit(0.001), t), t), t), tanFn(mul(x2, lit(0.001), t), t), t)),
        new ExpressionDef("WEIGHTED_SUM", "0.3*x1 + 0.5*x2 + 0.2*x3",
        (x1, x2, x3, t) -> add(add(mul(lit(0.3), x1, t), mul(lit(0.5), x2, t), t), mul(lit(0.2), x3, t), t)),
        new ExpressionDef("NORMALIZED_DIFF", "(x1-x2) / (abs(x1)+abs(x2)+1)",
        (x1, x2, x3, t) -> div(sub(x1, x2, t), add(add(absFn(x1, t), absFn(x2, t), t), lit(1.0), t), t)),
        new ExpressionDef("HARMONIC", "1/(x1*x1+1) + 1/(x2*x2+1)",
        (x1, x2, x3, t) -> add(div(lit(1.0), add(mul(x1, x1, t), lit(1.0), t), t), div(lit(1.0), add(mul(x2, x2, t), lit(1.0), t), t), t)),
        new ExpressionDef("LOG_RATIO", "ln(x1*x1+1) / ln(x2*x2+2)",
        (x1, x2, x3, t) -> div(lnFn(add(mul(x1, x1, t), lit(1.0), t), t), lnFn(add(mul(x2, x2, t), lit(2.0), t), t), t)),
        new ExpressionDef("TRIG_POLY", "sin(x1)*x2 + cos(x2)*x3 - tan(x3*0.01)*x1",
        (x1, x2, x3, t) -> sub(add(mul(sinFn(x1, t), x2, t), mul(cosFn(x2, t), x3, t), t), mul(tanFn(mul(x3, lit(0.01), t), t), x1, t), t)),
        new ExpressionDef("COMPOSITE", "sqrt(abs(x1*x2*x3)) + sin(x1+x2+x3) / exp(0.0001*abs(x3))",
        (x1, x2, x3, t) -> add(
        sqrtFn(absFn(mul(mul(x1, x2, t), x3, t), t), t),
        div(sinFn(add(add(x1, x2, t), x3, t), t), expFn(mul(lit(0.0001), absFn(x3, t), t), t), t),
        t))
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
    private Float8Vector gandivaOutput;

    private Map<String, Float8Vector> parserColumns;

    private ArrowBulkEvaluator parserEvaluator;

    private Projector gandivaProjector;

    private List<org.apache.arrow.memory.ArrowBuf> gandivaInputBuffers;
    private List<ValueVector> gandivaOutputVectors;
    private Schema gandivaSchema;

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
            parserEvaluator = ArrowBulkEvaluator.compile(def.parserExpr);
        } catch (Throwable ex) {
            System.getLogger(GandivaVsParserNGArrowBenchmark.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }

        gandivaProjector = buildGandivaProjector(def);
    }

    @Setup(Level.Iteration)
    public void setupIteration() {
        createInput();

        parserOutput = ArrowBulkEvaluator.allocateOutput(
                allocator,
                "parser_ng_result",
                size
        );

        gandivaOutput = new Float8Vector("gandiva_result", allocator);
        gandivaOutput.allocateNew(size);
        gandivaOutput.setValueCount(size);

        parserColumns = Map.of("x1", x1, "x2", x2, "x3", x3);

        gandivaInputBuffers = Arrays.asList(
                x1.getValidityBuffer(),
                x1.getDataBuffer(),
                x2.getValidityBuffer(),
                x2.getDataBuffer(),
                x3.getValidityBuffer(),
                x3.getDataBuffer()
        );

        gandivaOutputVectors = List.of(gandivaOutput);
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
     * Gandiva expression construction
     * -------------------------------------------------------------------------
     */
    private Projector buildGandivaProjector(ExpressionDef def) throws Exception {

        ArrowType.FloatingPoint doubleType = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);

        Field x1Field = new Field("x1", FieldType.nullable(doubleType), null);
        Field x2Field = new Field("x2", FieldType.nullable(doubleType), null);
        Field x3Field = new Field("x3", FieldType.nullable(doubleType), null);
        Field resultField = new Field("result", FieldType.nullable(doubleType), null);

        TreeNode x1Node = TreeBuilder.makeField(x1Field);
        TreeNode x2Node = TreeBuilder.makeField(x2Field);
        TreeNode x3Node = TreeBuilder.makeField(x3Field);

        TreeNode root = def.gandivaBuilder.build(x1Node, x2Node, x3Node, doubleType);

        ExpressionTree expressionTree = TreeBuilder.makeExpression(root, resultField);

        gandivaSchema = new Schema(List.of(x1Field, x2Field, x3Field));

        return Projector.make(gandivaSchema, List.of(expressionTree));
    }

    /*
     * -------------------------------------------------------------------------
     * Benchmarks
     * -------------------------------------------------------------------------
     */
    @Benchmark
    public void parserNG(Blackhole bh) {
        parserEvaluator.evaluate(
                parserColumns,
                parserOutput,
                NullPolicy.IGNORE,
                false
        );
        bh.consume(parserOutput);
    }

    @Benchmark
    public void gandiva(Blackhole bh) throws Exception {
        gandivaProjector.evaluate(
                size,
                gandivaInputBuffers,
                gandivaOutputVectors
        );
        bh.consume(gandivaOutput);
    }

    @Benchmark
    public void parserNGParallel(Blackhole bh) {
        parserEvaluator.evaluate(
                parserColumns,
                parserOutput,
                NullPolicy.IGNORE,
                true
        );
        bh.consume(parserOutput);
    }

    /*
     * -------------------------------------------------------------------------
     * Correctness check
     * -------------------------------------------------------------------------
     */
    private void verifyCorrectness(ExpressionDef def) throws Exception {
        parserEvaluator.evaluate(parserColumns, parserOutput, NullPolicy.IGNORE, false);
        gandivaProjector.evaluate(size, gandivaInputBuffers, gandivaOutputVectors);

        double maxAbs = 0.0;
        double maxRel = 0.0;
        int mismatches = 0;
        int nanMismatches = 0;

        for (int i = 0; i < size; i++) {
            double a = parserOutput.get(i);
            double b = gandivaOutput.get(i);

            boolean aNaN = Double.isNaN(a);
            boolean bNaN = Double.isNaN(b);
            if (aNaN || bNaN) {
                if (aNaN != bNaN) {
                    nanMismatches++;
                }
                continue;
            }

            double abs = Math.abs(a - b);
            double denominator = Math.max(Math.max(Math.abs(a), Math.abs(b)), 1e-15);
            double rel = abs / denominator;

            maxAbs = Math.max(maxAbs, abs);
            maxRel = Math.max(maxRel, rel);

            if (abs > 1e-12 && rel > 1e-12) {
                mismatches++;
            }
        }

        System.out.println("\n============================================================");
        System.out.println("Correctness");
        System.out.println("============================================================");
        System.out.println("Expression name  : " + def.name);
        System.out.println("Expression       : " + def.parserExpr);
        System.out.println("Rows             : " + size);
        System.out.printf(Locale.ROOT, "maxAbs           : %.17g%n", maxAbs);
        System.out.printf(Locale.ROOT, "maxRel           : %.17g%n", maxRel);
        System.out.println("mismatches       : " + mismatches);
        System.out.println("NaN mismatches   : " + nanMismatches
                + " (one side NaN, other not -- always worth investigating)");
        System.out.println("============================================================\n");
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
        if (gandivaOutput != null) {
            gandivaOutput.close();
            gandivaOutput = null;
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
        if (parserEvaluator != null) {
            parserEvaluator.close();
            parserEvaluator = null;
        }
        if (gandivaProjector != null) {
            try {
                gandivaProjector.close();
            } catch (GandivaException ex) {
                /* ignored */ }
            gandivaProjector = null;
        }
        if (allocator != null) {
            allocator.close();
            allocator = null;
        }
    }

    /*
     * -------------------------------------------------------------------------
     * Correctness-check runner -- checks EVERY expression in EXPRESSIONS in
     * one pass, independent of JMH. Run this FIRST, before trusting any
     * throughput number below, via:
     *
     *   java --add-modules=jdk.incubator.vector \
     *        --add-opens=java.base/java.nio=ALL-UNNAMED \
     *        --add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
     *        -Darrow.allocation.manager.type=Unsafe \
     *        -cp target/benchmarks.jar \
     *        com.github.gbenroscience.parser.ng.bench.GandivaVsParserNGArrowBenchmark
     * -------------------------------------------------------------------------
     */
    public static void runAllCorrectnessChecks() throws Exception {
        for (ExpressionDef def : EXPRESSIONS) {
            GandivaVsParserNGArrowBenchmark benchmark = new GandivaVsParserNGArrowBenchmark();
            benchmark.exprName = def.name;
            benchmark.size = 1_000_000;
            benchmark.setupTrial();
            benchmark.setupIteration();

            try {
                benchmark.verifyCorrectness(def);
            } finally {
                benchmark.teardownIteration();
                benchmark.teardownTrial();
            }
        }
    }

    /*
     * -------------------------------------------------------------------------
     * Interactive expression selection
     * -------------------------------------------------------------------------
     */

    /**
     * Prints the indexed expression catalog and reads a comma-separated list
     * of indices from {@code System.in} via {@link Scanner#nextLine()}.
     * Re-prompts on empty input, an out-of-range index, or a non-integer
     * token. Returns the parsed, order-preserving (duplicates allowed) list
     * of indices the user selected.
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
        runAllCorrectnessChecks();

        Scanner scanner = new Scanner(System.in);
        List<Integer> selectedIndices = promptForExpressionIndices(scanner);

        String[] selectedNames = new String[selectedIndices.size()];
        for (int i = 0; i < selectedIndices.size(); i++) {
            selectedNames[i] = EXPRESSIONS[selectedIndices.get(i)].name;
        }

        System.out.println("\nRunning JMH for: " + String.join(", ", selectedNames) + "\n");

        Options opt = new OptionsBuilder()
                .include(GandivaVsParserNGArrowBenchmark.class.getSimpleName())
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