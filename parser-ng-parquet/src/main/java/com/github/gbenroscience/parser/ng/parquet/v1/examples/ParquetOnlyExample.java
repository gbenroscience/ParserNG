package com.github.gbenroscience.parser.ng.parquet.v1.examples;

import com.github.gbenroscience.parser.ng.parquet.util.RandomParquetFiles;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetBatchReader;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetFileInfo;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;
import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import com.github.gbenroscience.parser.ng.parquet.v1.ScanMetrics;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.StructVector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Runnable walkthrough of {@code parser-ng-parquet} used on its own, with no SQL layer at all: this
 * is the module a hand-written scan, a stream processor, or a different query engine entirely would
 * call directly. Every example below is real, working code against this module's actual public API
 * (not pseudocode) — <b>but has not been executed</b>, for the same reason nothing else in this
 * module has: no Maven, no Parquet/Arrow/Hadoop jars, in the environment this was written in. Run it
 * with:
 *
 * <pre>{@code
 * mvn -pl parser-ng-parquet test-compile
 * java --add-modules jdk.incubator.vector -cp <test-classes>:<classes>:<deps> \
 *      com.github.gbenroscience.parser.ng.parquet.v1.examples.ParquetOnlyExample
 * }</pre>
 */
public final class ParquetOnlyExample {

    public static void main(String[] args) throws IOException {
        Path dir = Files.createTempDirectory("parser-ng-parquet-example");
        Path flatFile = dir.resolve("flat.parquet");
        Path nestedFile = dir.resolve("nested.parquet");

        RandomParquetFiles.write(flatFile, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 50_000,
                RandomParquetFiles.Config.defaults().seed(1));
        RandomParquetFiles.write(nestedFile, RandomParquetFiles.SAMPLE_NESTED_SCHEMA, 20_000,
                RandomParquetFiles.Config.defaults().seed(2));

        try (BufferAllocator alloc = new RootAllocator()) {
            metadataOnly(flatFile);
            fullScan(flatFile, alloc);
            projectedScan(flatFile, alloc);
            pushdownScan(flatFile, alloc);
            parallelScan(flatFile, alloc);
            nestedScan(nestedFile, alloc);
        }
    }

    /** Footer-only inspection: schema, row/row-group counts, per-column-chunk stats. Never decodes a page. */
    private static void metadataOnly(Path file) {
        section("1. Metadata inspection (no data page ever read)");
        ParquetFileInfo info = ParquetFileInfo.read(file);
        System.out.println("rows=" + info.rowCount() + " rowGroups=" + info.rowGroups().size()
                + " topLevelColumns=" + info.topLevelColumns());
        for (ParquetFileInfo.RowGroupInfo rg : info.rowGroups()) {
            System.out.println("  row group " + rg.index() + ": " + rg.rowCount() + " rows, "
                    + rg.totalByteSize() + " bytes");
        }
    }

    /** The simplest possible scan: every column, no predicate, sequential. */
    private static void fullScan(Path file, BufferAllocator alloc) {
        section("2. Full sequential scan");
        long rows = 0;
        ScanMetrics metrics = new ScanMetrics();
        try (ParquetBatchReader r = ParquetScan.scan(file).withMetrics(true).open(alloc)) {
            while (r.next()) {
                rows += r.root().getRowCount(); // r.root() is valid only until the next next()/close()
            }
            metrics = r.metrics();
        }
        System.out.println("rows=" + rows + " " + metrics);
    }

    /** Only requested columns are ever decoded -- unrequested columns' pages are never even read off disk. */
    private static void projectedScan(Path file, BufferAllocator alloc) {
        section("3. Projected scan (2 of 7 columns)");
        try (ParquetBatchReader r = ParquetScan.scan(file).select("id", "value").open(alloc)) {
            if (r.next()) {
                VectorSchemaRoot batch = r.root();
                System.out.println("columns in this batch: " + batch.getSchema().getFields());
            }
        }
    }

    /**
     * Pushdown is PRUNING, never filtering: the caller still re-checks the same condition on every
     * emitted row (see {@link Predicate}'s Javadoc for the exact soundness contract). This example
     * does that re-check explicitly so it's visible, not implicit.
     */
    private static void pushdownScan(Path file, BufferAllocator alloc) {
        section("4. Predicate pushdown (row-group + page pruning), with the mandatory re-check");
        Predicate pred = Predicate.and(Predicate.ge("id", 40_000L), Predicate.lt("id", 40_100L));
        ScanMetrics metrics = new ScanMetrics();
        long matched = 0;
        try (ParquetBatchReader r = ParquetScan.scan(file).pushdown(pred).withMetrics(true).open(alloc)) {
            while (r.next()) {
                VectorSchemaRoot batch = r.root();
                BigIntVector id = (BigIntVector) batch.getVector("id");
                for (int i = 0; i < batch.getRowCount(); i++) {
                    long v = id.get(i);
                    if (v >= 40_000L && v < 40_100L) matched++; // the re-check pushdown does NOT replace
                }
            }
            metrics = r.metrics();
        }
        System.out.println("matched=" + matched + " (expected 100) " + metrics
                + " -- rowGroupsSkipped=" + metrics.rowGroupsSkipped());
    }

    /** Decodes up to 4 row groups concurrently; ordering is preserved regardless (see ParquetBatchReader's Javadoc). */
    private static void parallelScan(Path file, BufferAllocator alloc) {
        section("5. Parallel scan (up to 4 row groups decoded concurrently)");
        long rows = 0;
        ScanMetrics metrics = new ScanMetrics();
        try (ParquetBatchReader r = ParquetScan.scan(file).parallelism(4).withMetrics(true).open(alloc)) {
            while (r.next()) rows += r.root().getRowCount();
            metrics = r.metrics();
        }
        System.out.println("rows=" + rows + " parallelism=" + metrics.parallelism() + " " + metrics);
    }

    /** Struct + LIST columns, decoded via the same native FastColumnCursor engine as flat columns. */
    private static void nestedScan(Path file, BufferAllocator alloc) {
        section("6. Nested scan (struct 'point', LIST 'tags')");
        try (ParquetBatchReader r = ParquetScan.scan(file).open(alloc)) {
            if (r.next()) {
                VectorSchemaRoot batch = r.root();
                StructVector point = (StructVector) batch.getVector("point");
                ListVector tags = (ListVector) batch.getVector("tags");
                for (int i = 0; i < Math.min(5, batch.getRowCount()); i++) {
                    boolean pointNull = point.isNull(i);
                    boolean tagsNull = tags.isNull(i);
                    int tagCount = tagsNull ? 0 : tags.getElementEndIndex(i) - tags.getElementStartIndex(i);
                    System.out.println("row " + i + ": point=" + (pointNull ? "null" : "present")
                            + " tags=" + (tagsNull ? "null" : tagCount + " element(s)"));
                }
            }
        }
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("=== " + title + " ===");
    }
}