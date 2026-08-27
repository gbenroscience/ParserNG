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

import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import com.github.gbenroscience.arrow.tools.box.ArrowBulkEvaluator;
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

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.apache.arrow.gandiva.exceptions.GandivaException;
import org.apache.arrow.vector.types.FloatingPointPrecision;

/**
 * Isolated re-run of ONLY the POLY case from GandivaVsParserNGArrowBenchmark's
 * 5-expression suite:
 *
 *     (x1+x2)*(x1-x2) + x3*x3*x3
 *
 * Purpose: the full suite (5 expressions x 3 sizes) showed Gandiva winning
 * on POLY specifically at the two larger sizes -- the one expression with
 * NO transcendental function calls at all -- while ParserNG won everywhere
 * else. That run also had two known confounds: a mid-run sleep interruption
 * and likely thermal throttling over its multi-hour total wall time on
 * mobile Skylake hardware. This class exists to re-check that ONE finding
 * in isolation, on a run short enough to complete without either confound
 * coming into play.
 *
 * Only 2 sizes (default: 262144 and 8388608 -- the two where the reversal
 * showed up), one expression, same JMH rigor (2 forks, 5+8 iterations) as
 * the full suite for a fair apples-to-apples comparison against that run's
 * numbers -- narrowing scope, not measurement methodology, is what makes
 * this fast. 2 sizes x 3 benchmark methods x 2 forks x 13 iterations is
 * ~15x fewer total iterations than the full suite's 30-combination sweep,
 * so this should complete in a few minutes rather than hours.
 *
 * Edit the @Param array below if you want different sizes.
 */
@State(Scope.Benchmark)
@BenchmarkMode(org.openjdk.jmh.annotations.Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 2)
public class GandivaVsParserNGPolyBenchmark {

    private static final String EXPRESSION = "(x1+x2)*(x1-x2) + x3*x3*x3";

    @Param({
        "262144",
        "8388608"
    })
    private int size;

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

        try {
            parserEvaluator = ArrowBulkEvaluator.compile(EXPRESSION);
        } catch (Throwable ex) {
            System.getLogger(GandivaVsParserNGPolyBenchmark.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }

        gandivaProjector = buildGandivaProjector();
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
     * Input generation -- identical formula to the full suite, so results are
     * directly comparable row-for-row.
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
     * Gandiva expression construction: (x1+x2)*(x1-x2) + x3*x3*x3
     * -------------------------------------------------------------------------
     */
    private Projector buildGandivaProjector() throws Exception {

        ArrowType.FloatingPoint doubleType = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);

        Field x1Field = new Field("x1", FieldType.nullable(doubleType), null);
        Field x2Field = new Field("x2", FieldType.nullable(doubleType), null);
        Field x3Field = new Field("x3", FieldType.nullable(doubleType), null);
        Field resultField = new Field("result", FieldType.nullable(doubleType), null);

        TreeNode x1Node = TreeBuilder.makeField(x1Field);
        TreeNode x2Node = TreeBuilder.makeField(x2Field);
        TreeNode x3Node = TreeBuilder.makeField(x3Field);

        TreeNode sumX = TreeBuilder.makeFunction("add", List.of(x1Node, x2Node), doubleType);
        TreeNode diffX = TreeBuilder.makeFunction("subtract", List.of(x1Node, x2Node), doubleType);
        TreeNode diffOfSquares = TreeBuilder.makeFunction("multiply", List.of(sumX, diffX), doubleType);
        TreeNode x3Sq = TreeBuilder.makeFunction("multiply", List.of(x3Node, x3Node), doubleType);
        TreeNode x3Cube = TreeBuilder.makeFunction("multiply", List.of(x3Sq, x3Node), doubleType);
        TreeNode root = TreeBuilder.makeFunction("add", List.of(diffOfSquares, x3Cube), doubleType);

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
    private void verifyCorrectness() throws Exception {
        parserEvaluator.evaluate(parserColumns, parserOutput, NullPolicy.IGNORE, false);
        gandivaProjector.evaluate(size, gandivaInputBuffers, gandivaOutputVectors);

        double maxAbs = 0.0;
        double maxRel = 0.0;
        int mismatches = 0;

        for (int i = 0; i < size; i++) {
            double a = parserOutput.get(i);
            double b = gandivaOutput.get(i);
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
        System.out.println("Correctness -- POLY");
        System.out.println("============================================================");
        System.out.println("Expression : " + EXPRESSION);
        System.out.println("Rows       : " + size);
        System.out.printf(Locale.ROOT, "maxAbs     : %.17g%n", maxAbs);
        System.out.printf(Locale.ROOT, "maxRel     : %.17g%n", maxRel);
        System.out.println("mismatches : " + mismatches);
        System.out.println("============================================================\n");
    }

    /*
     * -------------------------------------------------------------------------
     * Teardown
     * -------------------------------------------------------------------------
     */

    @TearDown(Level.Iteration)
    public void teardownIteration() {
        if (parserOutput != null) { parserOutput.close(); parserOutput = null; }
        if (gandivaOutput != null) { gandivaOutput.close(); gandivaOutput = null; }
        if (x1 != null) { x1.close(); x1 = null; }
        if (x2 != null) { x2.close(); x2 = null; }
        if (x3 != null) { x3.close(); x3 = null; }
    }

    @TearDown(Level.Trial)
    public void teardownTrial() {
        if (parserEvaluator != null) { parserEvaluator.close(); parserEvaluator = null; }
        if (gandivaProjector != null) {
            try { gandivaProjector.close(); } catch (GandivaException ex) { /* ignored */ }
            gandivaProjector = null;
        }
        if (allocator != null) { allocator.close(); allocator = null; }
    }

    /*
     * -------------------------------------------------------------------------
     * Correctness-check runner -- run this FIRST:
     *
     *   java --add-modules=jdk.incubator.vector \
     *        --add-opens=java.base/java.nio=ALL-UNNAMED \
     *        --add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
     *        -Darrow.allocation.manager.type=Unsafe \
     *        -cp target/benchmarks.jar \
     *        com.github.gbenroscience.parser.ng.bench.GandivaVsParserNGPolyBenchmark
     *
     * Then the throughput sweep (fast -- 2 sizes, ~2-5 min total depending
     * on hardware, no sleep/thermal risk over that short a window):
     *
     *   java -jar target/benchmarks.jar GandivaVsParserNGPolyBenchmark \
     *     -jvmArgs "--add-modules=jdk.incubator.vector --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED -Darrow.allocation.manager.type=Unsafe"
     * -------------------------------------------------------------------------
     */
    public static void main(String[] args) throws Exception {
        GandivaVsParserNGPolyBenchmark benchmark = new GandivaVsParserNGPolyBenchmark();
        benchmark.size = 1_000_000;
        benchmark.setupTrial();
        benchmark.setupIteration();

        try {
            benchmark.verifyCorrectness();
        } finally {
            benchmark.teardownIteration();
            benchmark.teardownTrial();
        }
    }
}