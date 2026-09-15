package com.github.gbenroscience.sqlv1.demo;

import com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import com.github.gbenroscience.sqlv1.ArrowQuery;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Scanner;

/**
 * A plain, hand-rolled (non-JMH) throughput benchmark for
 * {@code parser-ng-sql}: pick a query from a menu, and it reports how long the
 * query took to compile and how many input rows per second it can evaluate once
 * compiled.
 *
 * <h2>Methodology</h2>
 * <ul>
 * <li><b>One shared, fixed dataset</b> - every expression draws from the same
 * five float64 columns ({@code x, y, z, a, b}), randomly generated once per
 * benchmark run (fixed seed, so numbers are reproducible run to run). This
 * keeps the harness generic: a newly added expression needs no new schema
 * wiring, as long as it only references those five columns.
 * <li><b>Compile time is reported as two parts.</b>
 * {@link ArrowSqlQuery#compile} only parses the SQL text - it's cheap, and
 * doesn't tell you much. The actual ParserNG expression compilation is deferred
 * to the
 * <i>first</i> {@code execute()} call (see {@code ArrowSqlQuery}'s own javadoc:
 * precision, float32 vs. float64, can only be known from a real batch, not from
 * SQL text alone). So this benchmark times {@code compile()} and a single
 * throwaway {@code execute()} against a tiny batch separately, and reports both
 * plus their sum as "compile time" - that sum is the realistic one-time cost
 * before a compiled query is ready for repeated use.
 * <li><b>Warmup, then measurement, on the same compiled query.</b> A handful of
 * untimed warmup calls let the JIT compile hot paths before anything is
 * measured; only the measured iterations count toward the reported rows/second,
 * matching how a compiled {@link ArrowSqlQuery} is actually meant to be used -
 * compiled once, executed many times.
 * <li><b>Rows/second is input rows processed, not output rows returned.</b>
 * A {@code WHERE} clause can shrink a result to a handful of rows; that says
 * nothing about how fast the engine evaluated the full batch. This reports
 * {@code (rowCount * measured iterations) / totalElapsedSeconds}.
 * <li><b>Every result is consumed</b> (summed into {@link #sink}, printed at
 * the end of each run) so the JIT has an observable reason it can't prove away
 * the work - the classic hand-rolled-benchmark discipline a dedicated blackhole
 * gives you for free in JMH.
 * </ul>
 *
 * <p>
 * This is deliberately not JMH: no forked JVMs, no statistical rigor beyond
 * "warm up, then average a batch of timed runs" - good enough for a quick
 * relative sense of throughput, not for publishing a paper.
 */
public final class Benchmark {

    // =========================================================================
    // Tunables
    // =========================================================================
    private static final int DEFAULT_ROW_COUNT = 2_000_000;
    private static final int WARMUP_ITERATIONS = 5;
    private static final int MEASURED_ITERATIONS = 15;
    private static final long RANDOM_SEED = 42L;

    /**
     * Accumulates a touch of every result, purely so the JIT can never prove a
     * benchmarked call's output is unused. Never meaningfully read except to
     * print it.
     */
    private static volatile double sink;

    // =========================================================================
    // The expression registry - EXTEND THIS, nothing else needs to change.
    // =========================================================================
    /**
     * One benchmarkable query: a short label for the menu, and the actual SQL
     * text that gets compiled and timed.
     */
    private record BenchExpression(String label, String sql) {

    }

    /**
     * The list the menu is built from. Add a new {@code new
     * BenchExpression(...)} entry below (or call {@link List#add} on this list
     * before {@link #main} runs, if wiring it in from elsewhere) and it
     * automatically appears in the menu with the next index - nothing else in
     * this file references entries by position or count.
     *
     * <p>
     * New expressions should stick to columns {@code x, y, z, a, b} (all
     * float64) unless {@link #buildRandomBatch} is also extended to generate
     * whatever new column they need.
     */
    private static final List<BenchExpression> EXPRESSIONS = new ArrayList<>(List.of(
            new BenchExpression("2D magnitude",
                    "SELECT sqrt(x*x + y*y) AS magnitude FROM data"),
            new BenchExpression("3D magnitude",
                    "SELECT sqrt(x*x + y*y + z*z) AS magnitude3d FROM data"),
            new BenchExpression("Difference of squares",
                    "SELECT (x + y) * (x - y) AS diff_of_squares FROM data"),
            new BenchExpression("Trig mix: sin*cos + tan",
                    "SELECT sin(x) * cos(y) + tan(z) AS trig_mix FROM data"),
            new BenchExpression("Binomial expansion",
                    "SELECT x^2 + y^2 - 2*x*y AS expand FROM data"),
            new BenchExpression("5-column mean",
                    "SELECT (x + y + z + a + b) / 5 AS mean5 FROM data"),
            new BenchExpression("Conditional absolute difference",
                    "SELECT if(x > y, x - y, y - x) AS abs_diff FROM data"),
            new BenchExpression("Filtered magnitude (WHERE x > 0 AND y > 0)",
                    "SELECT sqrt(x*x + y*y) AS magnitude FROM data WHERE x > 0 AND y > 0"),
            new BenchExpression("Cross-term sum with an OR filter",
                    "SELECT x*y + y*z + z*a AS cross_sum FROM data WHERE (x > 0 AND y > 0) OR z > 100"),
            new BenchExpression("Pythagorean identity",
                    "SELECT sin(x)*sin(x) + cos(x)*cos(x) AS trig_identity FROM data"),
            new BenchExpression("Modulo 7",
                    "SELECT (x + y) % 7 AS mod7 FROM data"),
            new BenchExpression("Conditional trig (embedded comparison in if(...))",
                    "SELECT if(sin(x) > 0, tan(x), cos(x)) AS conditional_trig FROM data"),
            new BenchExpression("BETWEEN + NOT IN filter",
                    "SELECT x, y FROM data WHERE x BETWEEN 100 AND 900 AND y NOT IN (1, 2, 3)"),
            new BenchExpression("Nested sqrt",
                    "SELECT sqrt(sqrt(x*x + y*y)) AS quad_root FROM data"),
            new BenchExpression("Sum of squared differences",
                    "SELECT (x - y)*(x - y) + (y - z)*(y - z) AS sumsq_diff FROM data"),
            new BenchExpression("Range-conditional scaling (BETWEEN inside if(...))",
                    "SELECT if(x BETWEEN 0 AND 100, x*2, x/2) AS scaled FROM data"),
            new BenchExpression("Mixed polynomial",
                    "SELECT a*b - x*y + z^2 AS mixed_poly FROM data"),
            new BenchExpression("WHERE referencing a SELECT alias",
                    "SELECT x, y, z, sqrt(x*x + y*y + z*z) AS magnitude FROM data WHERE magnitude > 50"),
            new BenchExpression("Large mixed arithmetic",
                    "SELECT (a + b) * sqrt(x*x + y*y) - z^2 / (a + 2) AS big_mix FROM data"),
            new BenchExpression("Signed magnitude (AND/OR/BETWEEN inside if(...))",
                    "SELECT if((x > 0 AND y > 0) OR z BETWEEN -10 AND 10, "
                    + "sqrt(x*x + y*y + z*z), 0 - sqrt(x*x + y*y + z*z)) AS signed_magnitude FROM data")
    // Add more here, e.g.:
    // new BenchExpression("My new expression", "SELECT ... FROM data")
    ));

    // =========================================================================
    // Interactive menu
    // =========================================================================
    public static void main(String[] args) throws Throwable {
        System.out.println("parser-ng-sql throughput benchmark");
        System.out.println("(non-JMH: warmup " + WARMUP_ITERATIONS + " iterations, "
                + "then measure " + MEASURED_ITERATIONS + " iterations, over a "
                + String.format(Locale.US, "%,d", DEFAULT_ROW_COUNT) + "-row batch)");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                printMenu();
                System.out.print("Select an expression by index (or 'q' to quit): ");
                String line = scanner.hasNextLine() ? scanner.nextLine().trim() : "q";

                if (line.equalsIgnoreCase("q") || line.equalsIgnoreCase("quit")) {
                    System.out.println("Bye.");
                    return;
                }

                int index;
                try {
                    index = Integer.parseInt(line);
                } catch (NumberFormatException e) {
                    System.out.println("Not a number: \"" + line + "\". Try again.");
                    continue;
                }
                if (index < 0 || index >= EXPRESSIONS.size()) {
                    System.out.println("No expression at index " + index
                            + " - valid range is 0.." + (EXPRESSIONS.size() - 1) + ".");
                    continue;
                }

                runBenchmark(EXPRESSIONS.get(index), DEFAULT_ROW_COUNT, WARMUP_ITERATIONS, MEASURED_ITERATIONS);
            }
        }
    }

    private static void printMenu() {
        System.out.println();
        System.out.println("Available expressions (" + EXPRESSIONS.size() + "):");
        for (int i = 0; i < EXPRESSIONS.size(); i++) {
            BenchExpression e = EXPRESSIONS.get(i);
            System.out.printf(Locale.US, "  [%2d] %-48s %s%n", i, e.label(), e.sql());
        }
    }

    // =========================================================================
    // The benchmark itself
    // =========================================================================
    private static void runBenchmark(
            BenchExpression expr, int rowCount, int warmupIterations, int measuredIterations) throws Throwable {

        System.out.println();
        System.out.println("=".repeat(78));
        System.out.println("Expression: " + expr.sql());
        System.out.println("=".repeat(78));
        System.out.println("Building a " + String.format(Locale.US, "%,d", rowCount)
                + "-row batch (columns x, y, z, a, b)...");

        try (VectorSchemaRoot data = buildRandomBatch(rowCount)) {

            // ---- compile phase: SQL parse, timed on its own ----
            long parseStart = System.nanoTime();
            ArrowQuery query = ArrowQuery.compile(expr.sql()).withBackend(ArrowExecutionBackend.CPU_SIMD).withNullPolicy(NullPolicy.IGNORE);
            long parseElapsedNanos = System.nanoTime() - parseStart;

            try {
                // ---- compile phase continued: force the lazy ParserNG
                // expression compilation via one throwaway execute() against
                // a tiny batch, timed separately from the parse above and
                // from the real measurement below ----
                long firstExecElapsedNanos;
                try (VectorSchemaRoot tinyBatch = buildRandomBatch(8)) {
                    long firstExecStart = System.nanoTime();
                    try (VectorSchemaRoot warm = query.execute(tinyBatch)) {
                        touch(warm);
                    }
                    firstExecElapsedNanos = System.nanoTime() - firstExecStart;
                }

                double parseMs = parseElapsedNanos / 1_000_000.0;
                double firstExecMs = firstExecElapsedNanos / 1_000_000.0;
                double compileMs = parseMs + firstExecMs;

                System.out.printf(Locale.US,
                        "Compile time: %.3f ms (SQL parse) + %.3f ms (first execute, "
                        + "triggers ParserNG expression compilation) = %.3f ms total%n",
                        parseMs, firstExecMs, compileMs);

                // ---- warmup: untimed, lets the JIT compile hot paths ----
                System.out.println("Warming up (" + warmupIterations + " iterations, not timed)...");
                for (int i = 0; i < warmupIterations; i++) {
                    try (VectorSchemaRoot r = query.execute(data)) {
                        touch(r);
                    }
                }

                // ---- measured: this is what gets reported ----
                System.out.println("Measuring (" + measuredIterations + " iterations)...");
                long totalNanos = 0;
                for (int i = 0; i < measuredIterations; i++) {
                    long start = System.nanoTime();
                    try (VectorSchemaRoot r = query.execute(data)) {
                        totalNanos += System.nanoTime() - start;
                        touch(r);
                    }
                }

                double totalSeconds = totalNanos / 1_000_000_000.0;
                long totalRowsProcessed = (long) rowCount * measuredIterations;
                double rowsPerSecond = totalRowsProcessed / totalSeconds;
                double avgMsPerCall = (totalNanos / 1_000_000.0) / measuredIterations;

                System.out.println();
                System.out.println("Results:");
                System.out.println("  Expression:            " + expr.sql());
                System.out.printf(Locale.US, "  Compile time:          %.3f ms%n", compileMs);
                System.out.printf(Locale.US, "  Avg time per execute:  %.3f ms (over %d measured calls)%n",
                        avgMsPerCall, measuredIterations);
                System.out.printf(Locale.US, "  Throughput:            %,.0f rows/second%n", rowsPerSecond);
                System.out.printf(Locale.US, "  (checksum, ignore: %.6f - proves the JIT couldn't skip the work)%n",
                        sink);

            } finally {
                query.close();
            }
        }
    }

    /**
     * Touches just enough of a result to give the JIT an observable reason it
     * can never prove the call's output is dead - see {@link #sink}.
     */
    private static void touch(VectorSchemaRoot result) {
        int rows = result.getRowCount();
        sink += rows;
        if (rows > 0 && !result.getFieldVectors().isEmpty()) {
            Object v = result.getFieldVectors().get(0);
            if (v instanceof Float8Vector f8 && !f8.isNull(0)) {
                sink += f8.get(0);
            }
        }
    }

    // =========================================================================
    // Shared dataset
    // =========================================================================
    /**
     * Five float64 columns ({@code x, y, z, a, b}), uniformly random in [-500,
     * 500), seeded for reproducibility across runs.
     */
    private static VectorSchemaRoot buildRandomBatch(int rowCount) {
        Random random = new Random(RANDOM_SEED);
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);

        Float8Vector xv = new Float8Vector("x", allocator);
        Float8Vector yv = new Float8Vector("y", allocator);
        Float8Vector zv = new Float8Vector("z", allocator);
        Float8Vector av = new Float8Vector("a", allocator);
        Float8Vector bv = new Float8Vector("b", allocator);

        xv.allocateNew(rowCount);
        yv.allocateNew(rowCount);
        zv.allocateNew(rowCount);
        av.allocateNew(rowCount);
        bv.allocateNew(rowCount);

        for (int i = 0; i < rowCount; i++) {
            xv.setSafe(i, (random.nextDouble() * 1000) - 500);
            yv.setSafe(i, (random.nextDouble() * 1000) - 500);
            zv.setSafe(i, (random.nextDouble() * 1000) - 500);
            av.setSafe(i, (random.nextDouble() * 1000) - 500);
            bv.setSafe(i, (random.nextDouble() * 1000) - 500);
        }

        xv.setValueCount(rowCount);
        yv.setValueCount(rowCount);
        zv.setValueCount(rowCount);
        av.setValueCount(rowCount);
        bv.setValueCount(rowCount);

        return VectorSchemaRoot.of(xv, yv, zv, av, bv);
    }

    private Benchmark() {
    }
}
