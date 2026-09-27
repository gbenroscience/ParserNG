package com.github.gbenroscience.sqlv1.parquet.demo;

import com.github.gbenroscience.parser.ng.parquet.util.RandomParquetFiles;
import com.github.gbenroscience.sqlv1.parquet.ParquetSql;
import com.github.gbenroscience.sqlv1.parquet.ParquetSqlPlan;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Runnable walkthrough of {@code parser-ng-sql}'s Parquet bridge ({@code ParquetSql}/
 * {@code ParquetSqlPlan}, package {@code com.github.gbenroscience.sqlv1.parquet}): plain SQL text in,
 * a scan derived and executed against a real file, {@code ArrowQuery} handling everything the
 * Parquet layer can't (arithmetic, aliases, ORDER BY, LIMIT, aggregates) on the other end.
 *
 * <p><b>Belongs in the {@code parser-ng-sql} module</b>, not {@code parser-ng-parquet} —
 * {@code parser-ng-parquet} must never depend on {@code parser-ng-sql} (see both modules' READMEs on
 * the architectural boundary), and this class imports {@code com.github.gbenroscience.sqlv1.parquet.*}
 * and {@code com.github.gbenroscience.sqlv1.ArrowQuery} directly, so it can only compile where those
 * are already on the classpath. It is shipped alongside {@code parser-ng-parquet}'s own examples in
 * this deliverable only because that is where {@code RandomParquetFiles} (the fixture generator both
 * examples share) currently lives; copy this file into {@code parser-ng-sql}'s own test/example
 * source tree, or depend on {@code parser-ng-parquet}'s test-jar for {@code RandomParquetFiles},
 * before building it.
 *
 * <p><b>Not executed</b> — same reason as everything else in this deliverable (no Maven, no
 * dependency jars in the authoring environment) — and this one carries strictly more risk than the
 * parser-ng-parquet-only example: it also assumes {@code ArrowQuery.compile(String)},
 * {@code ArrowQuery.statement()}, and {@code ArrowQuery.execute(VectorSchemaRoot)} exist with those
 * exact names, which was not independently re-verified here beyond matching how
 * {@code ParquetSql}/{@code ParquetSqlPlan} themselves already call them.
 */
public final class ParquetSqlExample {

    public static void main(String[] args) throws IOException {
        Path dir = Files.createTempDirectory("parser-ng-sql-parquet-example");
        Path file = dir.resolve("sales.parquet");

        // SAMPLE_FLAT_SCHEMA has: id (int64), value (double), ratio (float), flag (boolean),
        // name (string), day (date), seen_at (timestamp) -- see RandomParquetFiles' class Javadoc.
        RandomParquetFiles.write(file, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 2_000_000,
                RandomParquetFiles.Config.defaults().seed(99).rowGroupSize(8L * 1024 * 1024));

        try (BufferAllocator alloc = new RootAllocator()) {
            streamingExample(file, alloc);
            globalExample(file, alloc);
            explainOnly(file);
        }
    }

    /**
     * SELECT ... WHERE ... with no GROUP BY/ORDER BY/aggregates: streamable. Each Parquet batch is
     * pushed through ArrowQuery independently and handed to the sink one at a time -- memory stays
     * bounded by batch size regardless of how many rows match.
     */
    private static void streamingExample(Path file, BufferAllocator alloc) {
        section("1. Streaming: SELECT with a WHERE clause, no ORDER BY/GROUP BY");
        String sql = "SELECT id, value, ratio FROM sales WHERE value > 500.0 AND flag = true";
        // The FROM table name ("sales") is ignored by design -- `file` is the actual data source;
        // see ParquetSqlPlan's class Javadoc.
        try (ParquetSqlPlan plan = ParquetSql.plan(file, sql)) {
            System.out.println(plan.explain());
            long[] rowsSeen = {0};
            plan.stream(alloc, batch -> {
                rowsSeen[0] += batch.getRowCount();
                // batch is valid only during this callback -- copy/transfer here if you need to keep it.
            });
            System.out.println("streamed rows=" + rowsSeen[0] + " " + plan.lastMetrics());
        }
    }

    /**
     * ORDER BY / LIMIT force global execution: matching batches are gathered into one root (capped
     * by maxMaterializedRows, so a query that would need too much memory fails loudly instead of
     * running out of it) and ArrowQuery runs once over the combined result.
     */
    private static void globalExample(Path file, BufferAllocator alloc) {
        section("2. Global execution: ORDER BY + LIMIT (not streamable)");
        String sql = "SELECT id, value FROM sales WHERE value > 900.0 ORDER BY value DESC LIMIT 20";
        try (ParquetSqlPlan plan = ParquetSql.plan(file, sql).maxMaterializedRows(5_000_000)) {
            System.out.println(plan.explain());
            try (VectorSchemaRoot result = plan.execute(alloc)) {
                System.out.println("result rows=" + result.getRowCount() + " " + plan.lastMetrics());
            }
        }
    }

    /** Inspecting a plan without running it -- useful for logging/debugging what pushdown actually derived. */
    private static void explainOnly(Path file) {
        section("3. explain() only: what columns/pruning/execution-mode a query resolves to");
        String sql = "SELECT id, value, sqrt(value) AS magnitude FROM sales "
                + "WHERE id BETWEEN 100 AND 200 GROUP BY id HAVING value > 0";
        try (ParquetSqlPlan plan = ParquetSql.plan(file, sql)) {
            // Nothing executed yet -- explain() is derived purely from ScanPlanner.plan(), no I/O.
            System.out.println(plan.explain());
            // Expect: columns=[id, value] (sqrt(value)/magnitude is an ArrowQuery expression, not a
            // file column -- see ColumnRefs, which never counts a function-call identifier as a
            // column reference); pruning derived from "id BETWEEN 100 AND 200" only, since GROUP BY
            // forces materialize mode regardless of what pruning finds; mode=materialize (GROUP BY...).
        }
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("=== " + title + " ===");
    }
}