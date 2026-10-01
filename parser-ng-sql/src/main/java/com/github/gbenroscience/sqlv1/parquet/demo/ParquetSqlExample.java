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
 * Runnable walkthrough of the Parquet bridge: SQL text in, a pruned scan and a vectorized
 * evaluation out. Lives in {@code parser-ng-sql} (parser-ng-parquet never depends on it);
 * {@code RandomParquetFiles} comes from parser-ng-parquet's main sources, so it compiles here.
 *
 * <p>Not executed by the author of this change (no Maven or dependency jars were available);
 * run it with {@code mvn -q exec:java -Dexec.mainClass=...ParquetSqlExample}.
 */
public final class ParquetSqlExample {

    private ParquetSqlExample() { }

    public static void main(String[] args) throws IOException {
        Path dir = Files.createTempDirectory("parser-ng-sql-parquet-example");
        Path file = dir.resolve("sales.parquet");
        // Columns: id (int64), value (double), ratio (float), flag (boolean), name (string), day (date), seen_at (timestamp).
        RandomParquetFiles.write(file, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 2_000_000,
                RandomParquetFiles.Config.defaults().seed(99).rowGroupSize(8L * 1024 * 1024));
        String from = "'" + file + "'";

        try (BufferAllocator alloc = new RootAllocator()) {
            // 1. Streaming. id (int64) and name (string) pass through in their native types;
            //    value is widened to double for the WHERE and the expression.
            try (ParquetSqlPlan plan = ParquetSql.create().plan(
                    "SELECT id, name, value * 2 AS v2 FROM " + from + " WHERE value > 500.0")) {
                System.out.println(plan.explain());
                long[] rows = {0};
                plan.stream(alloc, batch -> rows[0] += batch.getRowCount()); // batch valid only inside the callback
                System.out.println("streamed rows=" + rows[0] + " " + plan.lastMetrics());
            }

            // 2. ORDER BY / LIMIT materialize the rows that survive pruning (capped, fails loudly).
            try (ParquetSqlPlan plan = ParquetSql.create().plan(
                         "SELECT id, value FROM " + from + " ORDER BY value DESC LIMIT 5");
                 VectorSchemaRoot r = plan.execute(alloc)) {
                System.out.println(r.contentToTSVString());
            }

            // 3. COUNT(*) is answered from the footer: no data page is decoded.
            try (VectorSchemaRoot r = ParquetSql.create().execute("SELECT COUNT(*) AS n FROM " + from, alloc)) {
                System.out.println(r.contentToTSVString());
            }

            // 4. EXPLAIN.
            try (VectorSchemaRoot r = ParquetSql.create().execute(
                    "EXPLAIN SELECT id FROM " + from + " WHERE id < 1000", alloc)) {
                System.out.println(r.contentToTSVString());
            }
        }
    }
}
