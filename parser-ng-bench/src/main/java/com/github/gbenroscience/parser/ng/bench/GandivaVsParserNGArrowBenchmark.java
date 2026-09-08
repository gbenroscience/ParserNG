/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://apache.org
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
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.runner.options.TimeValue;

@State(Scope.Benchmark)
@BenchmarkMode(org.openjdk.jmh.annotations.Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
// Mirrors the flags in main()'s OptionsBuilder.jvmArgsAppend(...) below, so
// the same JVM tuning applies whether this is launched through main() or
// picked up directly by JMH's own shaded-jar launcher.
@Fork(value = 2, jvmArgsAppend = {
    "--add-modules=jdk.incubator.vector",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "-Darrow.enable_unsafe_memory_access=true",
    "-Darrow.enable_null_check_for_get=false",
    "-Darrow.allocation.manager.type=Unsafe",
    "-Darrow.memory.debug.allocator=false",
    "-Dio.netty.tryReflectionSetAccessible=true",
    "-XX:+UseParallelGC",
    "-XX:+AlwaysPreTouch",
    "-XX:+UseLargePages"
})
public class GandivaVsParserNGArrowBenchmark {

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

    // ---- transcendental / inverse-trig / hyperbolic helpers -----------------
    // STATUS as of the last real run against this codebase:
    //   ParserNG confirmed working: sin, cos, tan, asin, acos, atan, sinh,
    //   cosh, tanh, sqrt, abs, ln, cbrt (none of these appeared as an
    //   unresolved "variable" in ArrowBulkEvaluator's required-variable list
    //   when TRANSCENDENTAL_STACK was run).
    //   ParserNG CONFIRMED NOT SUPPORTED: log10 -- it is not in ParserNG's
    //   registry and gets silently treated as a bare unbound variable
    //   ("Missing Arrow column for variable 'log10'") rather than a parse
    //   error, which is why it slipped past compilation and only failed at
    //   evaluate() time. Every parserExpr string below uses ln(v)/ln(10)
    //   instead of log10(v) for exactly this reason.
    //   Gandiva side: "atan2" is still unverified against Gandiva's function
    //   registry -- none of the expressions below happened to exercise a
    //   Gandiva-only run in isolation yet, so treat ATAN2_ANGLE and
    //   ANGLE_ROUNDTRIP as the two still-unconfirmed entries if either
    //   throws "function signature not supported" at Projector.make() time.
    private static TreeNode asinFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("asin", List.of(a), t);
    }

    private static TreeNode acosFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("acos", List.of(a), t);
    }

    private static TreeNode atanFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("atan", List.of(a), t);
    }

    private static TreeNode atan2Fn(TreeNode a, TreeNode b, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("atan2", List.of(a, b), t);
    }

    private static TreeNode sinhFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("sinh", List.of(a), t);
    }

    private static TreeNode coshFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("cosh", List.of(a), t);
    }

    private static TreeNode tanhFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("tanh", List.of(a), t);
    }

    private static TreeNode log10Fn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("log10", List.of(a), t);
    }

    private static TreeNode cbrtFn(TreeNode a, ArrowType.FloatingPoint t) {
        return TreeBuilder.makeFunction("cbrt", List.of(a), t);
    }
    public static final ExpressionDef[] EXPRESSIONS = new ExpressionDef[]{
        new ExpressionDef("DISTANCE", "sin(sqrt(x1*x1 + x2*x2 + x3*x3))",
        (x1, x2, x3, t) -> sinFn(sqrtFn(add(add(mul(x1, x1, t), mul(x2, x2, t), t), mul(x3, x3, t), t), t), t)),
        new ExpressionDef("POLY", "(x1+x2)*(x1-x2) + x3*x3*x3",
        (x1, x2, x3, t) -> add(mul(add(x1, x2, t), sub(x1, x2, t), t), mul(mul(x3, x3, t), x3, t), t)),
        new ExpressionDef("LOG_EXP", "ln(x1*x1 + 1) + exp(x2*0.001) - x3",
        (x1, x2, x3, t) -> sub(add(lnFn(add(mul(x1, x1, t), lit(1.0), t), t), expFn(mul(x2, lit(0.001), t), t), t), x3, t)),
        new ExpressionDef("TRIG_CHAIN", "sin(x1)*cos(x2) + tan(x3*0.1) - sin(x1*x2*0.0001) + sqrt(abs(x3)+1)",
        (x1, x2, x3, t) -> add(sub(add(mul(sinFn(x1, t), cosFn(x2, t), t), tanFn(mul(x3, lit(0.1), t), t), t), sinFn(mul(mul(x1, x2, t), lit(0.0001), t), t), t), sqrtFn(add(absFn(x3, t), lit(1.0), t), t), t)),
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
        (x1, x2, x3, t) -> add(sqrtFn(absFn(mul(mul(x1, x2, t), x3, t), t), t), div(sinFn(add(add(x1, x2, t), x3, t), t), expFn(mul(lit(0.0001), absFn(x3, t), t), t), t), t)),

        // ---- transcendental category ------------------------------------
        // x1/x2/x3 are sin/cos-derived and range roughly [-10, 10], so asin
        // and acos inputs below are scaled by 0.09 to stay inside [-1, 1].
        new ExpressionDef("INV_TRIG_MIX", "asin(x1*0.09) + acos(x2*0.09) - atan(x3)",
        (x1, x2, x3, t) -> sub(add(asinFn(mul(x1, lit(0.09), t), t), acosFn(mul(x2, lit(0.09), t), t), t), atanFn(x3, t), t)),
        new ExpressionDef("ATAN2_ANGLE", "atan2(x2, x1)",
        (x1, x2, x3, t) -> atan2Fn(x2, x1, t)),
        new ExpressionDef("ANGLE_ROUNDTRIP", "atan2(sin(x1), cos(x1))",
        (x1, x2, x3, t) -> atan2Fn(sinFn(x1, t), cosFn(x1, t), t)),
        new ExpressionDef("HYPERBOLIC_CHAIN", "sinh(x1*0.001) + cosh(x2*0.001) - tanh(x3*0.001)",
        (x1, x2, x3, t) -> sub(add(sinhFn(mul(x1, lit(0.001), t), t), coshFn(mul(x2, lit(0.001), t), t), t), tanhFn(mul(x3, lit(0.001), t), t), t)),
        // CONFIRMED via a real run: ParserNG has no "log10" token -- it gets
        // treated as a bare unbound variable ("Missing Arrow column for
        // variable 'log10'"). Rewritten below as ln(v)/ln(10), which is
        // mathematically identical and safe here since v = x1*x1+1 / x2*x2+2
        // is always >= 1. Gandiva's side is untouched -- it does resolve
        // "log10" as a real function.
        new ExpressionDef("LOG10_SUM", "ln(x1*x1+1)/ln(10) + ln(x2*x2+2)/ln(10)",
        (x1, x2, x3, t) -> add(log10Fn(add(mul(x1, x1, t), lit(1.0), t), t), log10Fn(add(mul(x2, x2, t), lit(2.0), t), t), t)),
        new ExpressionDef("CBRT_SUM", "cbrt(x1) + cbrt(x2) + cbrt(x3)",
        (x1, x2, x3, t) -> add(add(cbrtFn(x1, t), cbrtFn(x2, t), t), cbrtFn(x3, t), t)),
        new ExpressionDef("TRANSCENDENTAL_STACK",
        "sin(x1*0.01)+cos(x2*0.01)+tan(x3*0.01)+asin(x1*0.09)+acos(x2*0.09)+atan(x3)"
        + "+sinh(x1*0.001)+cosh(x2*0.001)+tanh(x3*0.001)+ln(x1*x1+1)/ln(10)+cbrt(x2)+sqrt(abs(x3)+1)",
        (x1, x2, x3, t) -> {
            TreeNode sum = add(sinFn(mul(x1, lit(0.01), t), t), cosFn(mul(x2, lit(0.01), t), t), t);
            sum = add(sum, tanFn(mul(x3, lit(0.01), t), t), t);
            sum = add(sum, asinFn(mul(x1, lit(0.09), t), t), t);
            sum = add(sum, acosFn(mul(x2, lit(0.09), t), t), t);
            sum = add(sum, atanFn(x3, t), t);
            sum = add(sum, sinhFn(mul(x1, lit(0.001), t), t), t);
            sum = add(sum, coshFn(mul(x2, lit(0.001), t), t), t);
            sum = add(sum, tanhFn(mul(x3, lit(0.001), t), t), t);
            sum = add(sum, log10Fn(add(mul(x1, x1, t), lit(1.0), t), t), t);
            sum = add(sum, cbrtFn(x2, t), t);
            sum = add(sum, sqrtFn(add(absFn(x3, t), lit(1.0), t), t), t);
            return sum;
        })
    };

    private static final Map<String, ExpressionDef> EXPRESSIONS_BY_NAME;

    static {
        Map<String, ExpressionDef> byName = new HashMap<>();
        for (ExpressionDef def : EXPRESSIONS) {
            byName.put(def.name, def);
        }
        EXPRESSIONS_BY_NAME = Map.copyOf(byName);
    }
    @Param({
        "1024",
        "262144",
        "8388608"
    })
    private int size;

    @Param({
        "DISTANCE", "POLY", "LOG_EXP", "TRIG_CHAIN", "VARIABLE_POWER",
        "SIMPLE_SUM", "SIMPLE_PRODUCT", "QUADRATIC", "CUBIC", "DIVIDE_CHAIN",
        "ABS_DIFF", "SQRT_SUM", "SIN_PRODUCT", "COS_SUM", "TAN_RATIO",
        "LOG_CHAIN", "EXP_CHAIN", "POWER_SQUARE", "POWER_CUBE", "POWER_MIXED",
        "MIXED_TRIG_LOG", "MIXED_EXP_TRIG", "NESTED_SQRT", "DEEP_CHAIN", "WEIGHTED_SUM",
        "NORMALIZED_DIFF", "HARMONIC", "LOG_RATIO", "TRIG_POLY", "COMPOSITE",
        "INV_TRIG_MIX", "ATAN2_ANGLE", "ANGLE_ROUNDTRIP", "HYPERBOLIC_CHAIN",
        "LOG10_SUM", "CBRT_SUM", "TRANSCENDENTAL_STACK"
    })
    private String exprName;

    private BufferAllocator allocator;
    private Float8Vector x1;
    private Float8Vector x2;
    private Float8Vector x3;
    private Float8Vector parserOutput;
    private Float8Vector gandivaOutput;
    private Map<String, Float8Vector> parserColumns;
    private ArrowBulkEvaluator parserSIMDEvaluator;
    private Projector gandivaProjector;
    private List<org.apache.arrow.memory.ArrowBuf> gandivaInputBuffers;
    private List<ValueVector> gandivaOutputVectors;
    private Schema gandivaSchema;
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
            System.getLogger(GandivaVsParserNGArrowBenchmark.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
            throw new IllegalStateException("Failed to compile ParserNG expression: " + def.parserExpr, ex);
        }

        gandivaProjector = buildGandivaProjector(def);

        x1 = new Float8Vector("x1", allocator);
        x2 = new Float8Vector("x2", allocator);
        x3 = new Float8Vector("x3", allocator);

        x1.allocateNew(size);
        x2.allocateNew(size);
        x3.allocateNew(size);

        parserOutput = ArrowBulkEvaluator.allocateOutput(allocator, "parser_ng_result", size);
        gandivaOutput = new Float8Vector("gandiva_result", allocator);
        gandivaOutput.allocateNew(size);

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

    @Setup(Level.Iteration)
    public void setupIteration() {
        populateInputData();
        gandivaOutput.setValueCount(size);
    }

    private void populateInputData() {
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
    private Projector buildGandivaProjector(ExpressionDef def) throws Exception {
        ArrowType.FloatingPoint doubleType = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);

        // Non-nullable: populateInputData() never writes nulls, so Gandiva can
        // skip generating validity-bitmap checks for every function call in
        // the compiled expression -- this is a real win for chains with many
        // nested function calls (TRANSCENDENTAL_STACK, COMPOSITE, etc.).
        Field x1Field = new Field("x1", FieldType.notNullable(doubleType), null);
        Field x2Field = new Field("x2", FieldType.notNullable(doubleType), null);
        Field x3Field = new Field("x3", FieldType.notNullable(doubleType), null);
        Field resultField = new Field("result", FieldType.notNullable(doubleType), null);

        TreeNode x1Node = TreeBuilder.makeField(x1Field);
        TreeNode x2Node = TreeBuilder.makeField(x2Field);
        TreeNode x3Node = TreeBuilder.makeField(x3Field);

        TreeNode root = def.gandivaBuilder.build(x1Node, x2Node, x3Node, doubleType);
        ExpressionTree expressionTree = TreeBuilder.makeExpression(root, resultField);
        gandivaSchema = new Schema(List.of(x1Field, x2Field, x3Field));

        return Projector.make(gandivaSchema, List.of(expressionTree));
    }
    // NOTE: gandivaInputBuffers (built in setupTrial()) still passes each
    // vector's validity buffer alongside its data buffer even though the
    // fields above are now notNullable -- Arrow vectors always physically
    // carry a validity buffer regardless of the schema-level nullability
    // flag, and Gandiva's evaluate() signature expects the (validity, data)
    // pair per column either way. The notNullable declaration only affects
    // what code Gandiva's LLVM codegen emits (it skips bitmap checks), not
    // which buffers you hand it at evaluate() time -- no other change needed.

    @Benchmark
    public void parserNGSIMD(Blackhole bh) {
        parserSIMDEvaluator.evaluate(parserColumns, parserOutput, NullPolicy.IGNORE, false);
        bh.consume(parserOutput);
    }

    @Benchmark
    public void gandiva(Blackhole bh) throws Exception {
        gandivaProjector.evaluate(size, gandivaInputBuffers, gandivaOutputVectors);
        bh.consume(gandivaOutput);
    }

    @Benchmark
    public void parserNGParallel(Blackhole bh) {
        parserSIMDEvaluator.evaluate(parserColumns, parserOutput, NullPolicy.IGNORE, true);
        bh.consume(parserOutput);
    }
    private void verifyCorrectness(ExpressionDef def) throws Exception {
        parserSIMDEvaluator.evaluate(parserColumns, parserOutput, NullPolicy.IGNORE, false);
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
        System.out.println("NaN mismatches   : " + nanMismatches + " (one side NaN, other not)");
        System.out.println("============================================================\n");
    }

    @TearDown(Level.Iteration)
    public void teardownIteration() {
    }

    @TearDown(Level.Trial)
    public void teardownTrial() {
        if (parserSIMDEvaluator != null) {
            parserSIMDEvaluator.close();
            parserSIMDEvaluator = null;
        }
        if (gandivaProjector != null) {
            try { gandivaProjector.close(); } catch (GandivaException ex) {}
            gandivaProjector = null;
        }
        if (parserOutput != null) { parserOutput.close(); parserOutput = null; }
        if (gandivaOutput != null) { gandivaOutput.close(); gandivaOutput = null; }
        if (x1 != null) { x1.close(); x1 = null; }
        if (x2 != null) { x2.close(); x2 = null; }
        if (x3 != null) { x3.close(); x3 = null; }
        if (allocator != null) {
            allocator.close();
            allocator = null;
        }
    }
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

        public static void runSingleCorrectnessCheck(int exprIndex) throws Exception {
            ExpressionDef ed = EXPRESSIONS[exprIndex];
           GandivaVsParserNGArrowBenchmark benchmark = new GandivaVsParserNGArrowBenchmark();
            benchmark.exprName = ed.name;
            benchmark.size = 1_000_000;
            benchmark.setupTrial();
            benchmark.setupIteration();

            try {
                benchmark.verifyCorrectness(ed);
            } finally {
                benchmark.teardownIteration();
                benchmark.teardownTrial();
            }
    }

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

            if (valid) return parsed;
        }
    }
    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        List<Integer> selectedIndices = promptForExpressionIndices(scanner);

        String[] selectedNames = new String[selectedIndices.size()];
        for (int i = 0; i < selectedIndices.size(); i++) {
            selectedNames[i] = EXPRESSIONS[selectedIndices.get(i)].name;
        }
        System.out.println("TESTING SELECTED EXPRESSIONS FOR CORRECTNESS---");
        for(int idx : selectedIndices){
            runSingleCorrectnessCheck(idx);
        }
        System.out.println("TESTING SELECTED EXPRESSIONS FOR CORRECTNESS---");

        System.out.println("\nRunning JMH for: " + String.join(", ", selectedNames) + "\n");

        Options opt = new OptionsBuilder()
                .include(GandivaVsParserNGArrowBenchmark.class.getSimpleName())
                    .mode(Mode.AverageTime)
                    .timeUnit(TimeUnit.MICROSECONDS)
                    .warmupIterations(5)
                    .warmupTime(TimeValue.milliseconds(1000L))
                    .measurementIterations(8)
                    .measurementTime(TimeValue.milliseconds(1000L))
                .addProfiler(org.openjdk.jmh.profile.GCProfiler.class)
                .param("exprName", selectedNames)
                .jvmArgsAppend(
                        "--add-modules=jdk.incubator.vector",
                        "--add-opens=java.base/java.nio=ALL-UNNAMED",
                        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
                        // Netty (which Arrow's default buffer/allocator machinery
                        // sits on top of) uses reflection to grab direct-buffer
                        // internals for its fast paths; this flag keeps that path
                        // open on JDKs that restrict reflective access by default.
                        "--add-opens=java.base/java.lang=ALL-UNNAMED",
                        // ---- Gandiva / Arrow buffer-access fast paths ----
                        "-Darrow.enable_unsafe_memory_access=true",
                        "-Darrow.enable_null_check_for_get=false",
                        "-Darrow.allocation.manager.type=Unsafe",
                        // Skips Arrow's debug/leak-tracking allocator wrapper,
                        // which otherwise adds bookkeeping to every buffer
                        // alloc/access. CAVEAT: verify this system property name
                        // against your arrow-memory-core version -- it may be
                        // "arrow.memory.debug.allocator" or controlled instead by
                        // the io.netty.util.internal logging/leak-detection level
                        // in older Arrow releases.
                        "-Darrow.memory.debug.allocator=false",
                        "-Dio.netty.tryReflectionSetAccessible=true",
                        // ---- JVM-side throughput tuning ----
                        // Parallel GC favors raw throughput over pause time, which
                        // is what a JMH AverageTime benchmark wants; G1 (the
                        // default) optimizes for pause latency instead.
                        "-XX:+UseParallelGC",
                        // Pre-faults heap pages at JVM startup instead of taking
                        // page faults mid-measurement.
                        "-XX:+AlwaysPreTouch",
                        // Reduces TLB pressure on the large (up to ~64MB per
                        // vector at size=8388608) off-heap buffers. Harmless if
                        // the OS has no huge pages configured -- HotSpot falls
                        // back to normal pages with a warning rather than failing.
                        "-XX:+UseLargePages"
                )
                .build();
        new Runner(opt).run();
    }
}