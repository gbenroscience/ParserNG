package com.github.gbenroscience.parser.ng.bench;

import com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import com.github.gbenroscience.sqlv1.ArrowQuery;

import org.apache.arrow.gandiva.evaluator.Filter;
import org.apache.arrow.gandiva.evaluator.Projector;
import org.apache.arrow.gandiva.evaluator.SelectionVector;
import org.apache.arrow.gandiva.evaluator.SelectionVectorInt32;
import org.apache.arrow.gandiva.expression.Condition;
import org.apache.arrow.gandiva.expression.ExpressionTree;
import org.apache.arrow.gandiva.expression.TreeBuilder;
import org.apache.arrow.gandiva.expression.TreeNode;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;

import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.infra.BenchmarkParams;

import org.openjdk.jmh.profile.GCProfiler;
import org.openjdk.jmh.results.Result;
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;

/**
 * JMH benchmark comparing parser-ng-sql with Apache Arrow Gandiva.
 *
 * <p>
 * The SQL statements below are the canonical workload (40 cases, indices
 * 0..39). The same workload is hand-translated into Gandiva expression trees on
 * the Gandiva side, since Gandiva is an expression compiler, not a SQL
 * parser.</p>
 *
 * <p>
 * NOTE ON GANDIVA FUNCTION COVERAGE: cases 20-27, 33-39 use the Gandiva
 * functions {@code abs}, {@code ceil}, {@code floor}, {@code round},
 * {@code sign}, {@code power}, {@code log}, {@code exp} and {@code not_equal}.
 * These are standard float8 functions in the Gandiva precompiled function
 * registry, but function availability can differ across Arrow/Gandiva builds
 * &mdash; verify against your installed version before trusting the numbers,
 * and adjust the relevant {@code caseNN} builder if a function is missing.</p>
 *
 * <p>
 * Two ways to run:</p>
 *
 * <pre>
 * 1. Directly with JMH:
 *
 *    java -jar target/benchmarks.jar ParserNGSqlVsGandiva -prof gc
 *
 * 2. Through main():
 *
 *    java ... ParserNGSqlVsGandiva
 *
 *    Select:
 *
 *       7
 *       7,12,19
 *       all
 * </pre>
 *
 * <p>
 * The main() path always enables GCProfiler and, after the run, prints one row
 * per (benchmark, sqlIndex) combination containing the SQL expression,
 * throughput in rows/sec (via {@link OperationsPerInvocation}), and the
 * GCProfiler's {@code gc.alloc.rate} / {@code gc.alloc.rate.norm} secondary
 * results.</p>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(
        iterations = 5,
        time = 2,
        timeUnit = TimeUnit.SECONDS
)
@Measurement(
        iterations = 8,
        time = 3,
        timeUnit = TimeUnit.SECONDS
)
@Fork(3)
@State(Scope.Benchmark)
public class ParserNGSqlVsGandiva {

    // =========================================================================
    // BENCHMARK CONFIGURATION
    // =========================================================================
    private static final int ROW_COUNT = 3_000_000;

    private static final long RANDOM_SEED = 42L;

    private static final int WARMUP_ITERATIONS = 5;

    private static final int MEASUREMENT_ITERATIONS = 8;

    private static final int FORKS = 3;

    private static final int MAX_CASES = 40;

    // =========================================================================
    // SHARED GANDIVA TYPES
    // =========================================================================
    private static final ArrowType DOUBLE
            = new ArrowType.FloatingPoint(
                    FloatingPointPrecision.DOUBLE
            );
    private static final ArrowType BOOL = ArrowType.Bool.INSTANCE;

    /**
     * One output column of a Gandiva projection: the expression tree plus the
     * name it is projected as.
     */
    private record ColumnExpr(TreeNode node, String name) {

    }

    // =========================================================================
    // SQL WORKLOAD (40 cases, indices 0..39)
    // =========================================================================
    private static final List<SqlCase> CASES = List.of(
            new SqlCase(0, "2D magnitude",
                    "SELECT sqrt(x*x + y*y) AS magnitude FROM data",
                    ParserNGSqlVsGandiva::case0),
            new SqlCase(1, "3D magnitude",
                    "SELECT sqrt(x*x + y*y + z*z) AS magnitude3d FROM data",
                    ParserNGSqlVsGandiva::case1),
            new SqlCase(2, "Difference of squares",
                    "SELECT (x + y) * (x - y) AS diff_of_squares FROM data",
                    ParserNGSqlVsGandiva::case2),
            new SqlCase(3, "Trig mix",
                    "SELECT sin(x) * cos(y) + tan(z) AS trig_mix FROM data",
                    ParserNGSqlVsGandiva::case3),
            new SqlCase(4, "Binomial expansion",
                    "SELECT x^2 + y^2 - 2*x*y AS expand FROM data",
                    ParserNGSqlVsGandiva::case4),
            new SqlCase(5, "Five-column mean",
                    "SELECT (x + y + z + a + b) / 5 AS mean5 FROM data",
                    ParserNGSqlVsGandiva::case5),
            new SqlCase(6, "Conditional absolute difference",
                    "SELECT if(x > y, x - y, y - x) AS abs_diff FROM data",
                    ParserNGSqlVsGandiva::case6),
            new SqlCase(7, "Filtered magnitude",
                    "SELECT sqrt(x*x + y*y) AS magnitude FROM data WHERE x > 0 AND y > 0",
                    ParserNGSqlVsGandiva::case7),
            new SqlCase(8, "Cross-term sum with OR filter",
                    "SELECT x*y + y*z + z*a AS cross_sum FROM data WHERE (x > 0 AND y > 0) OR z > 100",
                    ParserNGSqlVsGandiva::case8),
            new SqlCase(9, "Pythagorean identity",
                    "SELECT sin(x)*sin(x) + cos(x)*cos(x) AS trig_identity FROM data",
                    ParserNGSqlVsGandiva::case9),
            new SqlCase(10, "Modulo 7",
                    "SELECT (x + y) % 7 AS mod7 FROM data",
                    ParserNGSqlVsGandiva::case10),
            new SqlCase(11, "Conditional trig",
                    "SELECT if(sin(x) > 0, tan(x), cos(x)) AS conditional_trig FROM data",
                    ParserNGSqlVsGandiva::case11),
            new SqlCase(12, "BETWEEN + NOT IN",
                    "SELECT x, y FROM data WHERE x BETWEEN 100 AND 900 AND y NOT IN (1, 2, 3)",
                    ParserNGSqlVsGandiva::case12),
            new SqlCase(13, "Nested sqrt",
                    "SELECT sqrt(sqrt(x*x + y*y)) AS quad_root FROM data",
                    ParserNGSqlVsGandiva::case13),
            new SqlCase(14, "Sum of squared differences",
                    "SELECT (x-y)*(x-y) + (y-z)*(y-z) AS sumsq_diff FROM data",
                    ParserNGSqlVsGandiva::case14),
            new SqlCase(15, "Range conditional scaling",
                    "SELECT if(x BETWEEN 0 AND 100, x*2, x/2) AS scaled FROM data",
                    ParserNGSqlVsGandiva::case15),
            new SqlCase(16, "Mixed polynomial",
                    "SELECT a*b - x*y + z^2 AS mixed_poly FROM data",
                    ParserNGSqlVsGandiva::case16),
            new SqlCase(17, "WHERE references SELECT alias",
                    "SELECT x, y, z, sqrt(x*x + y*y + z*z) AS magnitude FROM data WHERE magnitude > 50",
                    ParserNGSqlVsGandiva::case17),
            new SqlCase(18, "Large mixed arithmetic",
                    "SELECT (a+b)*sqrt(x*x+y*y) - z^2/(a+2) AS big_mix FROM data",
                    ParserNGSqlVsGandiva::case18),
            new SqlCase(19, "Signed magnitude",
                    "SELECT if((x > 0 AND y > 0) OR z BETWEEN -10 AND 10, sqrt(x*x+y*y+z*z), 0-sqrt(x*x+y*y+z*z)) AS signed_magnitude FROM data",
                    ParserNGSqlVsGandiva::case19),
            new SqlCase(20, "Absolute difference (function)",
                    "SELECT abs(x - y) AS abs_diff FROM data",
                    ParserNGSqlVsGandiva::case20),
            new SqlCase(21, "Five-column total",
                    "SELECT x + y + z + a + b AS total FROM data",
                    ParserNGSqlVsGandiva::case21),
            new SqlCase(22, "Sum of squares via power()",
                    "SELECT power(x, 2) + power(y, 2) AS sum_of_squares FROM data",
                    ParserNGSqlVsGandiva::case22),
            new SqlCase(23, "Filtered log sum",
                    "SELECT log(x) + log(y) AS log_sum FROM data WHERE x > 0 AND y > 0",
                    ParserNGSqlVsGandiva::case23),
            new SqlCase(24, "Scaled exponential",
                    "SELECT exp(x / 100) AS exp_scaled FROM data",
                    ParserNGSqlVsGandiva::case24),
            new SqlCase(25, "Ceil/floor range",
                    "SELECT ceil(x) - floor(y) AS range_diff FROM data",
                    ParserNGSqlVsGandiva::case25),
            new SqlCase(26, "Round",
                    "SELECT round(x) AS rounded FROM data",
                    ParserNGSqlVsGandiva::case26),
            new SqlCase(27, "Sign product",
                    "SELECT sign(x) * sign(y) AS sign_product FROM data",
                    ParserNGSqlVsGandiva::case27),
            new SqlCase(28, "Max of three (nested if)",
                    "SELECT if(x > y, if(x > z, x, z), if(y > z, y, z)) AS max_of_three FROM data",
                    ParserNGSqlVsGandiva::case28),
            new SqlCase(29, "Min of three (nested if)",
                    "SELECT if(x < y, if(x < z, x, z), if(y < z, y, z)) AS min_of_three FROM data",
                    ParserNGSqlVsGandiva::case29),
            new SqlCase(30, "Difference of cubes",
                    "SELECT x*x*x - y*y*y AS diff_cubes FROM data",
                    ParserNGSqlVsGandiva::case30),
            new SqlCase(31, "Four times xy via expansion",
                    "SELECT (x+y)*(x+y) - (x-y)*(x-y) AS four_xy FROM data",
                    ParserNGSqlVsGandiva::case31),
            new SqlCase(32, "Multi-column filter, no computed column",
                    "SELECT x, y, z FROM data WHERE x > y AND y > z",
                    ParserNGSqlVsGandiva::case32),
            new SqlCase(33, "WHERE on a recomputed function expression",
                    "SELECT sqrt(x*x + y*y) AS mag FROM data WHERE sqrt(x*x + y*y) > 100",
                    ParserNGSqlVsGandiva::case33),
            new SqlCase(34, "Quadrant sign (nested if)",
                    "SELECT if(x > 0, if(y > 0, 1, -1), 0) AS quadrant_sign FROM data",
                    ParserNGSqlVsGandiva::case34),
            new SqlCase(35, "Modulo 3",
                    "SELECT a % 3 AS mod3 FROM data",
                    ParserNGSqlVsGandiva::case35),
            new SqlCase(36, "BETWEEN + single-value NOT IN",
                    "SELECT x*y - z*a + b AS combo FROM data WHERE x BETWEEN -100 AND 100 AND y NOT IN (0)",
                    ParserNGSqlVsGandiva::case36),
            new SqlCase(37, "Filtered tangent product",
                    "SELECT tan(x) * tan(y) AS tan_product FROM data WHERE x <> 0 AND y <> 0",
                    ParserNGSqlVsGandiva::case37),
            new SqlCase(38, "5D magnitude",
                    "SELECT sqrt(x*x + y*y + z*z + a*a + b*b) AS mag5d FROM data",
                    ParserNGSqlVsGandiva::case38),
            new SqlCase(39, "Conditional magnitude (avoid sqrt when small)",
                    "SELECT if(x*x + y*y > 10000, sqrt(x*x + y*y), x*x + y*y) AS conditional_magnitude FROM data",
                    ParserNGSqlVsGandiva::case39)
    );

    // =========================================================================
    // JMH PARAMETER
    // =========================================================================
    @Param({
        "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14",
        "15", "16", "17", "18", "19",
        "20", "21", "22", "23", "24",
        "25", "26", "27", "28", "29",
        "30", "31", "32", "33", "34",
        "35", "36", "37", "38", "39"
    })
    public int sqlIndex;

    // =========================================================================
    // PER-BENCHMARK STATE
    // =========================================================================
    private BufferAllocator allocator;
    private VectorSchemaRoot input;
    private VectorSchemaRoot parserNGOutput;
    private ArrowQuery parserQuery;
    private GandivaExecution gandiva;
    private Schema schema;
    private ArrowRecordBatch gandivaRecordBatch;

    // =========================================================================
    // SETUP / TEARDOWN
    // =========================================================================
    @Setup(Level.Trial)
    public void setup() throws Exception {

        allocator = new RootAllocator(Long.MAX_VALUE);
        input = buildRandomBatch(allocator, ROW_COUNT, RANDOM_SEED);
        schema = input.getSchema();

        SqlCase testCase = getCase(sqlIndex);

        // ParserNG-SQL compilation is not measured; force it eagerly here.
        parserQuery = ArrowQuery
                .compile(testCase.sql())
                .withBackend(ArrowExecutionBackend.CPU_SIMD)
                .withNullPolicy(NullPolicy.IGNORE);

        parserNGOutput =  parserQuery.allocateReusableOutput(input, ROW_COUNT);
        parserQuery.execute(input, parserNGOutput);

        // Gandiva compilation is also not measured.
        gandiva = testCase.gandivaFactory().create(schema, allocator, ROW_COUNT);

        VectorUnloader unloader = new VectorUnloader(input);
        gandivaRecordBatch = unloader.getRecordBatch();
    }

    @TearDown(Level.Trial)
    public void tearDown() throws Exception {

        if (gandiva != null) {
            gandiva.close();
        }

        if (gandivaRecordBatch != null) {
            gandivaRecordBatch.close();
        }

        if (parserQuery != null) {
            parserQuery.close();
        }

        if (parserNGOutput != null) {
            parserNGOutput.close();
        }

        if (input != null) {
            input.close();
        }

        if (allocator != null) {
            allocator.close();
        }
    }

    // =========================================================================
    // BENCHMARKS
    //
    // @OperationsPerInvocation(ROW_COUNT) tells JMH that each invocation
    // processes ROW_COUNT input rows, so the Throughput score it reports
    // is directly rows/sec rather than queries/sec (both queries scan the
    // full ROW_COUNT input regardless of filter selectivity, so this is
    // an apples-to-apples input-rows/sec figure for filtered queries too).
    // =========================================================================
    @Benchmark
    @OperationsPerInvocation(ROW_COUNT)
    public void parserNG(Blackhole blackhole) {

        VectorSchemaRoot result = parserQuery.execute(input, parserNGOutput);

        blackhole.consume(result.getRowCount());

        if (!result.getFieldVectors().isEmpty()) {
            blackhole.consume(result.getFieldVectors().get(0));
        }
    }

    @Benchmark
    @OperationsPerInvocation(ROW_COUNT)
    public void gandiva(Blackhole blackhole) {

        try {
            int rowCount;

            if (gandiva.filter != null) {

                gandiva.filter.evaluate(gandivaRecordBatch, gandiva.selectionVector);
                rowCount = gandiva.selectionVector.getRecordCount();

                gandiva.projector.evaluate(
                        gandivaRecordBatch,
                        gandiva.selectionVector,
                        gandiva.outputVectors);

            } else {

                gandiva.projector.evaluate(gandivaRecordBatch, gandiva.outputVectors);
                rowCount = ROW_COUNT;
            }

            blackhole.consume(rowCount);

            for (ValueVector vector : gandiva.outputVectors) {
                blackhole.consume(vector);
            }

        } catch (Exception e) {
            throw new RuntimeException(
                    "Gandiva execution failed for SQL index " + sqlIndex
                    + ": " + getCase(sqlIndex).sql(),
                    e);
        }
    }

    // =========================================================================
    // CASE BUILDERS 0..19 (original workload)
    // =========================================================================
    private static GandivaExecution case0(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode magnitude = fn("sqrt", add(mul(f("x"), f("x")), mul(f("y"), f("y"))), DOUBLE);
        return projection(schema, allocator, rows, magnitude, "magnitude");
    }

    private static GandivaExecution case1(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = fn("sqrt",
                add(add(mul(f("x"), f("x")), mul(f("y"), f("y"))), mul(f("z"), f("z"))), DOUBLE);
        return projection(schema, allocator, rows, expression, "magnitude3d");
    }

    private static GandivaExecution case2(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = mul(add(f("x"), f("y")), sub(f("x"), f("y")));
        return projection(schema, allocator, rows, expression, "diff_of_squares");
    }

    private static GandivaExecution case3(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = add(
                mul(fn("sin", f("x"), DOUBLE), fn("cos", f("y"), DOUBLE)),
                fn("tan", f("z"), DOUBLE));
        return projection(schema, allocator, rows, expression, "trig_mix");
    }

    private static GandivaExecution case4(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = sub(
                add(mul(f("x"), f("x")), mul(f("y"), f("y"))),
                mul(lit(2.0), mul(f("x"), f("y"))));
        return projection(schema, allocator, rows, expression, "expand");
    }

    private static GandivaExecution case5(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = div(
                add(add(add(add(f("x"), f("y")), f("z")), f("a")), f("b")),
                lit(5.0));
        return projection(schema, allocator, rows, expression, "mean5");
    }

    private static GandivaExecution case6(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = TreeBuilder.makeIf(
                gt(f("x"), f("y")), sub(f("x"), f("y")), sub(f("y"), f("x")), DOUBLE);
        return projection(schema, allocator, rows, expression, "abs_diff");
    }

    private static GandivaExecution case7(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode magnitude = fn("sqrt", add(mul(f("x"), f("x")), mul(f("y"), f("y"))), DOUBLE);
        TreeNode condition = TreeBuilder.makeAnd(List.of(gt(f("x"), lit(0.0)), gt(f("y"), lit(0.0))));
        return filteredProjection(schema, allocator, rows, condition, magnitude, "magnitude");
    }

    private static GandivaExecution case8(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = add(add(mul(f("x"), f("y")), mul(f("y"), f("z"))), mul(f("z"), f("a")));
        TreeNode condition = TreeBuilder.makeOr(List.of(
                TreeBuilder.makeAnd(List.of(gt(f("x"), lit(0.0)), gt(f("y"), lit(0.0)))),
                gt(f("z"), lit(100.0))));
        return filteredProjection(schema, allocator, rows, condition, expression, "cross_sum");
    }

    private static GandivaExecution case9(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode sx = fn("sin", f("x"), DOUBLE);
        TreeNode cx = fn("cos", f("x"), DOUBLE);
        TreeNode expression = add(mul(sx, sx), mul(cx, cx));
        return projection(schema, allocator, rows, expression, "trig_identity");
    }

    private static GandivaExecution case10(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = modFn(add(f("x"), f("y")), lit(7.0));
        return projection(schema, allocator, rows, expression, "mod7");
    }

    private static GandivaExecution case11(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode condition = gt(fn("sin", f("x"), DOUBLE), lit(0.0));
        TreeNode expression = TreeBuilder.makeIf(
                condition, fn("tan", f("x"), DOUBLE), fn("cos", f("x"), DOUBLE), DOUBLE);
        return projection(schema, allocator, rows, expression, "conditional_trig");
    }

    private static GandivaExecution case12(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode between = TreeBuilder.makeAnd(List.of(gte(f("x"), lit(100.0)), lte(f("x"), lit(900.0))));
        TreeNode in = TreeBuilder.makeInExpressionDouble(f("y"), new HashSet<>(Set.of(1.0, 2.0, 3.0)));
        TreeNode notIn = fn("not", List.of(in), BOOL);
        TreeNode condition = TreeBuilder.makeAnd(List.of(between, notIn));
        return filteredProjection(schema, allocator, rows, condition,
                List.of(new ColumnExpr(f("x"), "x"), new ColumnExpr(f("y"), "y")));
    }

    private static GandivaExecution case13(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode inner = fn("sqrt", add(mul(f("x"), f("x")), mul(f("y"), f("y"))), DOUBLE);
        TreeNode expression = fn("sqrt", inner, DOUBLE);
        return projection(schema, allocator, rows, expression, "quad_root");
    }

    private static GandivaExecution case14(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode xy = sub(f("x"), f("y"));
        TreeNode yz = sub(f("y"), f("z"));
        TreeNode expression = add(mul(xy, xy), mul(yz, yz));
        return projection(schema, allocator, rows, expression, "sumsq_diff");
    }

    private static GandivaExecution case15(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode condition = TreeBuilder.makeAnd(List.of(gte(f("x"), lit(0.0)), lte(f("x"), lit(100.0))));
        TreeNode expression = TreeBuilder.makeIf(
                condition, mul(f("x"), lit(2.0)), div(f("x"), lit(2.0)), DOUBLE);
        return projection(schema, allocator, rows, expression, "scaled");
    }

    private static GandivaExecution case16(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = add(sub(mul(f("a"), f("b")), mul(f("x"), f("y"))), mul(f("z"), f("z")));
        return projection(schema, allocator, rows, expression, "mixed_poly");
    }

    private static GandivaExecution case17(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode magnitude = fn("sqrt",
                add(add(mul(f("x"), f("x")), mul(f("y"), f("y"))), mul(f("z"), f("z"))), DOUBLE);
        TreeNode condition = gt(magnitude, lit(50.0));
        return filteredProjection(schema, allocator, rows, condition, List.of(
                new ColumnExpr(f("x"), "x"),
                new ColumnExpr(f("y"), "y"),
                new ColumnExpr(f("z"), "z"),
                new ColumnExpr(magnitude, "magnitude")));
    }

    private static GandivaExecution case18(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = sub(
                mul(add(f("a"), f("b")),
                        fn("sqrt", add(mul(f("x"), f("x")), mul(f("y"), f("y"))), DOUBLE)),
                div(mul(f("z"), f("z")), add(f("a"), lit(2.0))));
        return projection(schema, allocator, rows, expression, "big_mix");
    }

    private static GandivaExecution case19(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode magnitude = fn("sqrt",
                add(add(mul(f("x"), f("x")), mul(f("y"), f("y"))), mul(f("z"), f("z"))), DOUBLE);
        TreeNode positiveXY = TreeBuilder.makeAnd(List.of(gt(f("x"), lit(0.0)), gt(f("y"), lit(0.0))));
        TreeNode zBetween = TreeBuilder.makeAnd(List.of(gte(f("z"), lit(-10.0)), lte(f("z"), lit(10.0))));
        TreeNode condition = TreeBuilder.makeOr(List.of(positiveXY, zBetween));
        TreeNode expression = TreeBuilder.makeIf(condition, magnitude, sub(lit(0.0), magnitude), DOUBLE);
        return projection(schema, allocator, rows, expression, "signed_magnitude");
    }

    // =========================================================================
    // CASE BUILDERS 20..39 (extended workload)
    // =========================================================================
    private static GandivaExecution case20(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = absFn(sub(f("x"), f("y")));
        return projection(schema, allocator, rows, expression, "abs_diff");
    }

    private static GandivaExecution case21(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = add(add(add(add(f("x"), f("y")), f("z")), f("a")), f("b"));
        return projection(schema, allocator, rows, expression, "total");
    }

    private static GandivaExecution case22(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = add(powFn(f("x"), lit(2.0)), powFn(f("y"), lit(2.0)));
        return projection(schema, allocator, rows, expression, "sum_of_squares");
    }

    private static GandivaExecution case23(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = add(logFn(f("x")), logFn(f("y")));
        TreeNode condition = TreeBuilder.makeAnd(List.of(gt(f("x"), lit(0.0)), gt(f("y"), lit(0.0))));
        return filteredProjection(schema, allocator, rows, condition, expression, "log_sum");
    }

    private static GandivaExecution case24(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = expFn(div(f("x"), lit(100.0)));
        return projection(schema, allocator, rows, expression, "exp_scaled");
    }

    private static GandivaExecution case25(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = sub(ceilFn(f("x")), floorFn(f("y")));
        return projection(schema, allocator, rows, expression, "range_diff");
    }

    private static GandivaExecution case26(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = roundFn(f("x"));
        return projection(schema, allocator, rows, expression, "rounded");
    }

    private static GandivaExecution case27(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = mul(signFn(f("x")), signFn(f("y")));
        return projection(schema, allocator, rows, expression, "sign_product");
    }

    private static GandivaExecution case28(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode xGtZ = TreeBuilder.makeIf(gt(f("x"), f("z")), f("x"), f("z"), DOUBLE);
        TreeNode yGtZ = TreeBuilder.makeIf(gt(f("y"), f("z")), f("y"), f("z"), DOUBLE);
        TreeNode expression = TreeBuilder.makeIf(gt(f("x"), f("y")), xGtZ, yGtZ, DOUBLE);
        return projection(schema, allocator, rows, expression, "max_of_three");
    }

    private static GandivaExecution case29(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode xLtZ = TreeBuilder.makeIf(lt(f("x"), f("z")), f("x"), f("z"), DOUBLE);
        TreeNode yLtZ = TreeBuilder.makeIf(lt(f("y"), f("z")), f("y"), f("z"), DOUBLE);
        TreeNode expression = TreeBuilder.makeIf(lt(f("x"), f("y")), xLtZ, yLtZ, DOUBLE);
        return projection(schema, allocator, rows, expression, "min_of_three");
    }

    private static GandivaExecution case30(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode x3 = mul(mul(f("x"), f("x")), f("x"));
        TreeNode y3 = mul(mul(f("y"), f("y")), f("y"));
        TreeNode expression = sub(x3, y3);
        return projection(schema, allocator, rows, expression, "diff_cubes");
    }

    private static GandivaExecution case31(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode sum = add(f("x"), f("y"));
        TreeNode diff = sub(f("x"), f("y"));
        TreeNode expression = sub(mul(sum, sum), mul(diff, diff));
        return projection(schema, allocator, rows, expression, "four_xy");
    }

    private static GandivaExecution case32(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode condition = TreeBuilder.makeAnd(List.of(gt(f("x"), f("y")), gt(f("y"), f("z"))));
        return filteredProjection(schema, allocator, rows, condition, List.of(
                new ColumnExpr(f("x"), "x"),
                new ColumnExpr(f("y"), "y"),
                new ColumnExpr(f("z"), "z")));
    }

    private static GandivaExecution case33(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode magnitude = fn("sqrt", add(mul(f("x"), f("x")), mul(f("y"), f("y"))), DOUBLE);
        TreeNode condition = gt(magnitude, lit(100.0));
        return filteredProjection(schema, allocator, rows, condition, magnitude, "mag");
    }

    private static GandivaExecution case34(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode inner = TreeBuilder.makeIf(gt(f("y"), lit(0.0)), lit(1.0), lit(-1.0), DOUBLE);
        TreeNode expression = TreeBuilder.makeIf(gt(f("x"), lit(0.0)), inner, lit(0.0), DOUBLE);
        return projection(schema, allocator, rows, expression, "quadrant_sign");
    }

    private static GandivaExecution case35(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = modFn(f("a"), lit(3.0));
        return projection(schema, allocator, rows, expression, "mod3");
    }

    private static GandivaExecution case36(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode between = TreeBuilder.makeAnd(List.of(gte(f("x"), lit(-100.0)), lte(f("x"), lit(100.0))));
        TreeNode in = TreeBuilder.makeInExpressionDouble(f("y"), new HashSet<>(Set.of(0.0)));
        TreeNode notIn = fn("not", List.of(in), BOOL);
        TreeNode condition = TreeBuilder.makeAnd(List.of(between, notIn));
        TreeNode expression = add(sub(mul(f("x"), f("y")), mul(f("z"), f("a"))), f("b"));
        return filteredProjection(schema, allocator, rows, condition, expression, "combo");
    }

    private static GandivaExecution case37(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode condition = TreeBuilder.makeAnd(List.of(
                notEq(f("x"), lit(0.0)), notEq(f("y"), lit(0.0))));
        TreeNode expression = mul(fn("tan", f("x"), DOUBLE), fn("tan", f("y"), DOUBLE));
        return filteredProjection(schema, allocator, rows, condition, expression, "tan_product");
    }

    private static GandivaExecution case38(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode expression = fn("sqrt", add(add(add(add(
                mul(f("x"), f("x")), mul(f("y"), f("y"))),
                mul(f("z"), f("z"))), mul(f("a"), f("a"))), mul(f("b"), f("b"))), DOUBLE);
        return projection(schema, allocator, rows, expression, "mag5d");
    }

    private static GandivaExecution case39(Schema schema, BufferAllocator allocator, int rows) throws Exception {
        TreeNode sumSquares = add(mul(f("x"), f("x")), mul(f("y"), f("y")));
        TreeNode condition = gt(sumSquares, lit(10000.0));
        TreeNode thenNode = fn("sqrt", sumSquares, DOUBLE);
        TreeNode expression = TreeBuilder.makeIf(condition, thenNode, sumSquares, DOUBLE);
        return projection(schema, allocator, rows, expression, "conditional_magnitude");
    }

    // =========================================================================
    // GENERIC GANDIVA FACTORY HELPERS
    // =========================================================================
    private static GandivaExecution projection(
            Schema schema, BufferAllocator allocator, int rows,
            TreeNode expression, String outputName) throws Exception {
        return projection(schema, allocator, rows, List.of(new ColumnExpr(expression, outputName)));
    }

    private static GandivaExecution projection(
            Schema schema, BufferAllocator allocator, int rows,
            List<ColumnExpr> columns) throws Exception {

        List<ExpressionTree> trees = new ArrayList<>(columns.size());
        List<Field> fields = new ArrayList<>(columns.size());

        for (ColumnExpr column : columns) {
            Field field = resultField(column.name());
            trees.add(TreeBuilder.makeExpression(column.node(), field));
            fields.add(field);
        }

        Projector projector = Projector.make(schema, trees);
        List<ValueVector> outputs = allocateOutputs(allocator, rows, fields);

        return new GandivaExecution(projector, null, null, outputs);
    }

    private static GandivaExecution filteredProjection(
            Schema schema, BufferAllocator allocator, int rows,
            TreeNode conditionNode, TreeNode expression, String outputName) throws Exception {
        return filteredProjection(schema, allocator, rows, conditionNode,
                List.of(new ColumnExpr(expression, outputName)));
    }

    private static GandivaExecution filteredProjection(
            Schema schema, BufferAllocator allocator, int rows,
            TreeNode conditionNode, List<ColumnExpr> columns) throws Exception {

        Condition condition = TreeBuilder.makeCondition(conditionNode);
        Filter filter = Filter.make(schema, condition);

        List<ExpressionTree> trees = new ArrayList<>(columns.size());
        List<Field> fields = new ArrayList<>(columns.size());

        for (ColumnExpr column : columns) {
            Field field = resultField(column.name());
            trees.add(TreeBuilder.makeExpression(column.node(), field));
            fields.add(field);
        }

        // INT32 selection vector is required at this row count (>64Ki rows).
        ArrowBuf selectionBuffer = allocator.buffer((long) rows * Integer.BYTES);
        SelectionVectorInt32 selectionVector = new SelectionVectorInt32(selectionBuffer);

        Projector projector = Projector.make(schema, trees, selectionVector.getType());
        List<ValueVector> outputs = allocateOutputs(allocator, rows, fields);

        return new GandivaExecution(projector, filter, selectionVector, outputs);
    }

    // =========================================================================
    // GANDIVA NODE HELPERS
    // =========================================================================
    private static TreeNode f(String name) {
        return TreeBuilder.makeField(Field.nullable(name, DOUBLE));
    }

    private static TreeNode lit(double value) {
        return TreeBuilder.makeLiteral(value);
    }

    private static TreeNode fn(String name, TreeNode child, ArrowType returnType) {
        return TreeBuilder.makeFunction(name, List.of(child), returnType);
    }

    private static TreeNode fn(String name, List<TreeNode> children, ArrowType returnType) {
        return TreeBuilder.makeFunction(name, children, returnType);
    }

    private static TreeNode add(TreeNode a, TreeNode b) {
        return fn("add", List.of(a, b), DOUBLE);
    }

    private static TreeNode sub(TreeNode a, TreeNode b) {
        return fn("subtract", List.of(a, b), DOUBLE);
    }

    private static TreeNode mul(TreeNode a, TreeNode b) {
        return fn("multiply", List.of(a, b), DOUBLE);
    }

    private static TreeNode div(TreeNode a, TreeNode b) {
        return fn("divide", List.of(a, b), DOUBLE);
    }

    private static TreeNode gt(TreeNode a, TreeNode b) {
        return fn("greater_than", List.of(a, b), BOOL);
    }

    private static TreeNode gte(TreeNode a, TreeNode b) {
        return fn("greater_than_or_equal_to", List.of(a, b), BOOL);
    }

    private static TreeNode lte(TreeNode a, TreeNode b) {
        return fn("less_than_or_equal_to", List.of(a, b), BOOL);
    }

    private static TreeNode lt(TreeNode a, TreeNode b) {
        return fn("less_than", List.of(a, b), BOOL);
    }

    private static TreeNode notEq(TreeNode a, TreeNode b) {
        return fn("not_equal", List.of(a, b), BOOL);
    }

    private static TreeNode absFn(TreeNode a) {
        return fn("abs", a, DOUBLE);
    }

    private static TreeNode ceilFn(TreeNode a) {
        return fn("ceil", a, DOUBLE);
    }

    private static TreeNode floorFn(TreeNode a) {
        return fn("floor", a, DOUBLE);
    }

    private static TreeNode roundFn(TreeNode a) {
        return fn("round", a, DOUBLE);
    }

    private static TreeNode signFn(TreeNode a) {
        return fn("sign", a, DOUBLE);
    }

    private static TreeNode logFn(TreeNode a) {
        return fn("log", a, DOUBLE);
    }

    private static TreeNode expFn(TreeNode a) {
        return fn("exp", a, DOUBLE);
    }

    private static TreeNode powFn(TreeNode a, TreeNode b) {
        return fn("power", List.of(a, b), DOUBLE);
    }

    private static TreeNode modFn(TreeNode a, TreeNode b) {
        return fn("mod", List.of(a, b), DOUBLE);
    }

    // =========================================================================
    // RESULT FIELDS / OUTPUT VECTORS
    // =========================================================================
    private static Field resultField(String name) {
        return Field.nullable(name, DOUBLE);
    }

    private static List<ValueVector> allocateOutputs(
            BufferAllocator allocator, int rows, List<Field> fields) {

        List<ValueVector> outputs = new ArrayList<>(fields.size());

        for (Field field : fields) {
            Float8Vector vector = new Float8Vector(field.getName(), allocator);
            vector.allocateNew(rows);
            vector.setValueCount(rows);
            outputs.add(vector);
        }

        return outputs;
    }

    // =========================================================================
    // DATASET
    // =========================================================================
    private static VectorSchemaRoot buildRandomBatch(
            BufferAllocator allocator, int rowCount, long seed) {

        Random random = new Random(seed);

        Float8Vector x = new Float8Vector("x", allocator);
        Float8Vector y = new Float8Vector("y", allocator);
        Float8Vector z = new Float8Vector("z", allocator);
        Float8Vector a = new Float8Vector("a", allocator);
        Float8Vector b = new Float8Vector("b", allocator);

        x.allocateNew(rowCount);
        y.allocateNew(rowCount);
        z.allocateNew(rowCount);
        a.allocateNew(rowCount);
        b.allocateNew(rowCount);

        for (int i = 0; i < rowCount; i++) {
            x.setSafe(i, random.nextDouble() * 1000.0 - 500.0);
            y.setSafe(i, random.nextDouble() * 1000.0 - 500.0);
            z.setSafe(i, random.nextDouble() * 1000.0 - 500.0);
            a.setSafe(i, random.nextDouble() * 1000.0 - 500.0);
            b.setSafe(i, random.nextDouble() * 1000.0 - 500.0);
        }

        x.setValueCount(rowCount);
        y.setValueCount(rowCount);
        z.setValueCount(rowCount);
        a.setValueCount(rowCount);
        b.setValueCount(rowCount);

        return VectorSchemaRoot.of(x, y, z, a, b);
    }

    // =========================================================================
    // SCANNER / MAIN
    // =========================================================================
    public static void main(String[] args) throws Exception {

        printMenu();

        System.out.println();
        System.out.println("Select SQL index(es), e.g. 7 or 7,12,19.");
        System.out.println("Enter 'all' to run all 40 configured cases.");
        System.out.print("> ");

        Scanner scanner = new Scanner(System.in);
        String selection = scanner.nextLine().trim();
        String[] selected = parseSelection(selection);

        Options options = new OptionsBuilder()
                .include(ParserNGSqlVsGandiva.class.getSimpleName())
                .param("sqlIndex", selected)
                .warmupIterations(WARMUP_ITERATIONS)
                .warmupTime(TimeValue.seconds(2))
                .measurementIterations(MEASUREMENT_ITERATIONS)
                .measurementTime(TimeValue.seconds(3))
                .forks(FORKS)
                .mode(Mode.Throughput)
                .timeUnit(TimeUnit.SECONDS)
                .addProfiler(GCProfiler.class)
                .build();

        Collection<RunResult> results = new Runner(options).run();

        printResults(results);
    }

    private static String[] parseSelection(String selection) {

        if (selection.equalsIgnoreCase("all")) {
            String[] result = new String[MAX_CASES];
            for (int i = 0; i < MAX_CASES; i++) {
                result[i] = Integer.toString(i);
            }
            return result;
        }

        if (selection.isBlank()) {
            throw new IllegalArgumentException("No benchmark indexes supplied.");
        }

        String[] tokens = selection.split(",");
        List<String> indexes = new ArrayList<>();

        for (String token : tokens) {

            String trimmed = token.trim();
            if (trimmed.isEmpty()) {
                continue;
            }

            int index;
            try {
                index = Integer.parseInt(trimmed);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid benchmark index: " + trimmed, e);
            }

            if (index < 0 || index >= MAX_CASES) {
                throw new IllegalArgumentException(
                        "Index " + index + " is outside 0.." + (MAX_CASES - 1));
            }

            indexes.add(Integer.toString(index));
        }

        if (indexes.isEmpty()) {
            throw new IllegalArgumentException("No valid benchmark indexes.");
        }

        return indexes.toArray(String[]::new);
    }

    private static void printMenu() {

        System.out.println();
        System.out.println("==============================================================");
        System.out.println("          ParserNG-SQL vs Gandiva JMH Benchmark");
        System.out.println("==============================================================");
        System.out.printf(Locale.US, "Dataset: %,d rows%n", ROW_COUNT);
        System.out.println("Input columns: x, y, z, a, b (Float64)");
        System.out.println();

        for (SqlCase testCase : CASES) {
            System.out.printf(Locale.US, "[%2d] %-45s %s%n",
                    testCase.index(), testCase.name(), testCase.sql());
        }

        System.out.println();
    }

    /**
     * Prints one row per (benchmark, sqlIndex) result: the SQL expression under
     * test, throughput in rows/sec, and the GCProfiler's allocation-rate
     * secondary results.
     *
     * <p>
     * NOTE: {@code RunResult.getSecondaryResults()} only contains entries when
     * {@code -prof gc} (or {@link GCProfiler} added programmatically, as
     * {@link #main} does) was active for the run; otherwise the gc columns
     * print as {@code n/a}.</p>
     */
    @SuppressWarnings("unchecked")
    private static void printResults(Collection<RunResult> results) {

        System.out.println();
        System.out.println("=================================================================================================================");
        System.out.printf(Locale.US, "%-4s %-10s %-45s %20s %18s %20s%n",
                "Idx", "Engine", "Expression", "Throughput(rows/s)", "gc.alloc.rate", "gc.alloc.rate.norm");
        System.out.println("=================================================================================================================");

        for (RunResult result : results) {

            BenchmarkParams params = result.getParams();
            int idx = Integer.parseInt(params.getParam("sqlIndex"));
            SqlCase testCase = getCase(idx);

            String benchmarkName = params.getBenchmark();
            String shortName = benchmarkName.substring(benchmarkName.lastIndexOf('.') + 1);

            Result<?> primary = result.getPrimaryResult();
            double throughput = primary.getScore();

            Map<String, Result> secondary = result.getSecondaryResults();
            double allocRate = secondary.containsKey("gc.alloc.rate")
                    ? secondary.get("gc.alloc.rate").getScore() : Double.NaN;
            double allocRateNorm = secondary.containsKey("gc.alloc.rate.norm")
                    ? secondary.get("gc.alloc.rate.norm").getScore() : Double.NaN;

            System.out.printf(Locale.US, "%-4d %-10s %-45s %,20.0f %18.4f %20.4f%n",
                    idx, shortName, testCase.name(), throughput, allocRate, allocRateNorm);
        }

        System.out.println("=================================================================================================================");
        System.out.println("gc.alloc.rate is in MB/sec; gc.alloc.rate.norm is bytes allocated per op (per row, given @OperationsPerInvocation).");
    }

    // =========================================================================
    // CASE LOOKUP
    // =========================================================================
    private static SqlCase getCase(int index) {

        if (index < 0 || index >= MAX_CASES) {
            throw new IllegalArgumentException("SQL index out of range: " + index);
        }

        if (index >= CASES.size()) {
            throw new IllegalArgumentException("SQL case " + index + " has not been defined yet.");
        }

        return CASES.get(index);
    }

    // =========================================================================
    // SMALL RESULT CONSUMPTION HELPER
    // =========================================================================
    private static void consume(VectorSchemaRoot result) {
        // Only used for the one-time ParserNG compilation call in setup().
        if (result.getRowCount() < 0) {
            throw new AssertionError();
        }
    }

    // =========================================================================
    // DATA TYPES
    // =========================================================================
    @FunctionalInterface
    private interface GandivaFactory {

        GandivaExecution create(Schema schema, BufferAllocator allocator, int rows) throws Exception;
    }

    private record SqlCase(int index, String name, String sql, GandivaFactory gandivaFactory) {

    }

    private static final class GandivaExecution implements AutoCloseable {

        private final Projector projector;
        private final Filter filter;
        private final SelectionVector selectionVector;
        private final List<ValueVector> outputVectors;

        private GandivaExecution(
                Projector projector, Filter filter,
                SelectionVector selectionVector, List<ValueVector> outputVectors) {
            this.projector = projector;
            this.filter = filter;
            this.selectionVector = selectionVector;
            this.outputVectors = outputVectors;
        }

        @Override
        public void close() throws Exception {

            if (projector != null) {
                projector.close();
            }
            if (filter != null) {
                filter.close();
            }
            // selectionVector's ArrowBuf lifecycle is tied to the allocator,
            // which is closed by the benchmark trial teardown.

            for (ValueVector vector : outputVectors) {
                vector.close();
            }
        }
    }
}
