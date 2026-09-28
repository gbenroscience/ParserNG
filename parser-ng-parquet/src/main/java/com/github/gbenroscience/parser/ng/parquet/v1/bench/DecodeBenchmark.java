package com.github.gbenroscience.parser.ng.parquet.v1.bench;

import com.github.gbenroscience.parser.ng.parquet.util.RandomParquetFiles;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetBatchReader;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;
import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import com.github.gbenroscience.parser.ng.parquet.v1.ScanMetrics;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Locale;

/**
 * Standalone decode-speed driver: NOT a JUnit test, NOT JMH. Run it directly:
 *
 * <pre>{@code
 * mvn -pl parser-ng-parquet test-compile
 * java --add-modules jdk.incubator.vector \
 *      -cp parser-ng-parquet/target/test-classes:parser-ng-parquet/target/classes:<parquet+arrow+hadoop jars on classpath> \
 *      com.github.gbenroscience.parser.ng.parquet.v1.bench.DecodeBenchmark
 * }</pre>
 *
 * <p><b>Verification status of the byte-counter revision.</b> Compiled and run end to end against real
 * parquet-java 1.13.1 / Arrow 12.0.1 jars (the versions bundled with PySpark 3.5.3 -- the only real jars
 * reachable where this was written), NOT the pom's parquet 1.18.0 / Arrow 19.0.0; two classes absent from
 * 1.13 ({@code LocalInputFile}/{@code LocalOutputFile}) were replaced by minimal local shims for that run.
 * On that stack the counters were cross-checked against the Parquet footer: full-scan decoded bytes equal
 * the footer's uncompressed column-chunk sizes to within page-header overhead (99.96-99.99%), compressed
 * bytes equal the footer's chunk sizes exactly, sequential and parallel scans report identical bytes, and
 * projection and row-group/page pruning shrink them as expected. It has <em>not</em> been run on 1.18.0 /
 * Arrow 19; the sanity check in {@link #run} prints a loud warning if a byte counter comes back zero.
 *
 * <h2>What this measures, and why each scenario is here</h2>
 * <ul>
 *   <li><b>Full flat scan</b> — the baseline: every column, no predicate, sequential. Everything
 *       else is compared against this.</li>
 *   <li><b>Projected flat scan</b> (2 of 7 columns) — isolates projection's effect: columns not
 *       selected are never decoded at all, so this should scale down roughly with column count, not
 *       with row count.</li>
 *   <li><b>Selective predicate scan</b> — a predicate matching ~10% of rows by construction (see
 *       {@link #writeSelectivityFixture}, whose {@code bucket} column is a monotonic function of row
 *       index specifically so min/max statistics can prune). Compares against the full scan to show
 *       row-group and page pruning's effect, not decode speed per se.</li>
 *   <li><b>Sequential vs. parallel(2/4/8)</b> — on the flat fixture; shows whether/how much the
 *       native decode engine's per-row-group cost scales across worker threads on this machine.</li>
 *   <li><b>Nested scan</b> — {@link RandomParquetFiles#SAMPLE_NESTED_SCHEMA}; compares against the
 *       flat full scan at a similar row count to show the {@code LevelWalker}/nested-assembly
 *       overhead relative to the flat path, and exercises {@code FastColumnCursor} against a nested
 *       leaf for the first time in this module's history (see that class's Javadoc on why that
 *       specific inference has not been exercised until this file runs).</li>
 * </ul>
 *
 * <h2>What "MB/s" means here (measured, not estimated)</h2>
 * Earlier revisions divided the fixture's file size on disk by elapsed time. That is wrong for exactly
 * the scenarios that matter: a projected scan never touches most of the file, and a pruned scan skips
 * whole row groups and pages, yet both were credited with the full file. This revision reads the byte
 * counters {@link ScanMetrics} now records inside the decode path and divides each by the same
 * wall-clock time (the median timed run, so rows/s and every MB/s column describe the same run):
 * <ul>
 *   <li><b>decoded MB/s</b> -- {@link ScanMetrics#uncompressedBytesDecoded()}: decompressed page bytes
 *       (dictionary + data pages, incl. rep/def levels) actually loaded by the decode engine. Respects
 *       projection and page pruning exactly. <b>This is the number to quote as decode throughput.</b></li>
 *   <li><b>disk MB/s</b> -- {@link ScanMetrics#compressedBytesRead()}: as-stored (compressed) bytes of the
 *       projected column chunks of the row groups that were read. An upper bound when page-level pruning
 *       skips pages inside a surviving row group. Compare with decoded MB/s for the compression ratio.</li>
 *   <li><b>arrow MB/s</b> -- {@link ScanMetrics#arrowBytesProduced()}: Arrow buffer bytes materialized,
 *       i.e. what a downstream consumer receives.</li>
 * </ul>
 * All rates are <em>warm-cache, end-to-end wall clock</em> (open file, read, decompress, decode, close):
 * the fixtures were just written, so the OS page cache serves the reads. They are not cold-disk numbers.
 * Rows/s is still the best cross-scenario comparison (a nested row and a flat row are different sizes);
 * MB/s is the best cross-<em>engine</em> comparison.
 *
 * <p>Reported per scenario: elapsed wall time, rows/sec, the three MB/s figures above, and the scan's
 * {@link ScanMetrics}.
 *
 * <p>The three fixture files (flat, selectivity, nested -- 5M + 5M + 1M rows total, written by
 * {@link RandomParquetFiles#write} / {@link #writeSelectivityFixture}) live in one temp directory
 * created at the top of {@link #main}, and are deleted, best-effort, in a {@code finally} block
 * around the whole run -- including when fixture generation or a scan throws partway through, so a
 * failed run doesn't leave 5M+5M+1M rows of Parquet fixtures behind either. See
 * {@link #deleteFixtures}.
 */
public final class DecodeBenchmark {

    private static final int WARMUP_ITERS = 2;
    private static final int TIMED_ITERS = 5;

    public static void main(String[] args) throws IOException {
        Path dir = Files.createTempDirectory("parser-ng-parquet-bench");
        System.out.println("Fixtures in " + dir);
        try {
            int flatRows = args.length > 0 ? Integer.parseInt(args[0]) : 5_000_000;
            int nestedRows = args.length > 1 ? Integer.parseInt(args[1]) : 1_000_000;

            Path flatFile = dir.resolve("flat.parquet");
            RandomParquetFiles.write(flatFile, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, flatRows,
                    RandomParquetFiles.Config.defaults().seed(42).rowGroupSize(8L * 1024 * 1024));

            Path selectivityFile = dir.resolve("selectivity.parquet");
            writeSelectivityFixture(selectivityFile, flatRows);

            Path nestedFile = dir.resolve("nested.parquet");
            RandomParquetFiles.write(nestedFile, RandomParquetFiles.SAMPLE_NESTED_SCHEMA, nestedRows,
                    RandomParquetFiles.Config.defaults().seed(7).collectionSize(0, 8).rowGroupSize(8L * 1024 * 1024));

            try (BufferAllocator alloc = new RootAllocator()) {
                printHeader();

                run(alloc, "flat/full-scan/seq", flatFile, s -> s, flatRows);
                run(alloc, "flat/projected(2 of 7)/seq", flatFile, s -> s.select("id", "value"), flatRows);
                run(alloc, "flat/full-scan/parallel(2)", flatFile, s -> s.parallelism(2), flatRows);
                run(alloc, "flat/full-scan/parallel(4)", flatFile, s -> s.parallelism(4), flatRows);
                run(alloc, "flat/full-scan/parallel(8)", flatFile, s -> s.parallelism(8), flatRows);

                // ~10% selectivity by construction -- see writeSelectivityFixture.
                run(alloc, "selective-predicate(~10%)/seq", selectivityFile,
                        s -> s.pushdown(Predicate.ge("bucket", 9L)), flatRows / 10);

                run(alloc, "nested/full-scan/seq", nestedFile, s -> s, nestedRows);
                run(alloc, "nested/full-scan/parallel(4)", nestedFile, s -> s.parallelism(4), nestedRows);
            }
        } finally {
            // Best-effort: a fixture-generation or scan failure above must not leave 5M+5M+1M rows
            // of Parquet fixtures sitting in the OS temp directory. deleteFixtures() logs and swallows
            // its own IOExceptions rather than throwing, so it never masks whatever exception (if any)
            // is already propagating out of the try block above.
            deleteFixtures(dir);
        }
    }

    /** Recursively deletes {@code dir} (and everything under it), best-effort: a failure here is
     *  logged to stderr, never thrown, so cleanup can never turn a successful benchmark run into a
     *  failed process exit, and never replaces/masks a real exception from the run itself when
     *  called from a {@code finally} block. */
    private static void deleteFixtures(Path dir) {
        try (var paths = Files.walk(dir)) {
            paths.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                try {
                    Files.deleteIfExists(p);
                } catch (IOException e) {
                    System.err.println("Warning: failed to delete benchmark fixture " + p + ": " + e);
                }
            });
        } catch (IOException e) {
            System.err.println("Warning: failed to walk benchmark fixture directory " + dir + " for cleanup: " + e);
        }
    }

    /** id: 0..rowCount-1; bucket: id/(rowCount/10), i.e. 10 buckets, monotonic in id -- prunable by stats. */
    private static void writeSelectivityFixture(Path file, int rowCount) throws IOException {
        var schema = org.apache.parquet.schema.MessageTypeParser.parseMessageType(
                "message t { required int64 id; required int64 bucket; }");
        var factory = new org.apache.parquet.example.data.simple.SimpleGroupFactory(schema);
        long perBucket = Math.max(1, rowCount / 10);
        try (var w = org.apache.parquet.hadoop.example.ExampleParquetWriter
                .builder(new org.apache.parquet.io.LocalOutputFile(file))
                .withType(schema)
                .withCompressionCodec(org.apache.parquet.hadoop.metadata.CompressionCodecName.SNAPPY)
                .withRowGroupSize(8L * 1024 * 1024)
                .build()) {
            for (int i = 0; i < rowCount; i++) {
                w.write(factory.newGroup().append("id", (long) i).append("bucket", (long) (i / perBucket)));
            }
        }
    }

    @FunctionalInterface
    private interface Config {
        ParquetScan apply(ParquetScan s);
    }

    private static void run(BufferAllocator alloc, String label, Path file, Config cfg, long expectedRows) throws IOException {
        for (int i = 0; i < WARMUP_ITERS; i++) drain(alloc, cfg.apply(ParquetScan.scan(file)));

        long[] nanos = new long[TIMED_ITERS];
        long[] rowsPerIter = new long[TIMED_ITERS];
        ScanMetrics[] metricsPerIter = new ScanMetrics[TIMED_ITERS];
        for (int i = 0; i < TIMED_ITERS; i++) {
            ParquetScan scan = cfg.apply(ParquetScan.scan(file)).withMetrics(true);
            long t0 = System.nanoTime();
            try (ParquetBatchReader r = scan.open(alloc)) {
                long rows = 0;
                while (r.next()) {
                    VectorSchemaRoot root = r.root();
                    rows += root.getRowCount(); // touch the batch; nothing else to "consume" in a pure decode benchmark
                }
                rowsPerIter[i] = rows;
                // Read while the reader is still open; for parallel scans every worker has finished by the
                // time next() returned false (the window is drained via Future.get()).
                metricsPerIter[i] = r.metrics();
            }
            nanos[i] = System.nanoTime() - t0;
        }

        // Report ONE run end to end: the median-time iteration's wall time, rows and metrics together, so
        // rows/s and every MB/s column come from the same run (bytes are deterministic, time is not).
        Integer[] order = new Integer[TIMED_ITERS];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingLong(i -> nanos[i]));
        int mid = order[TIMED_ITERS / 2];

        double seconds = nanos[mid] / 1e9;
        long rows = rowsPerIter[mid];
        ScanMetrics m = metricsPerIter[mid];
        double rowsPerSec = rows / seconds;
        double decodedMBps = mb(m.uncompressedBytesDecoded()) / seconds;
        double diskMBps = mb(m.compressedBytesRead()) / seconds;
        double arrowMBps = mb(m.arrowBytesProduced()) / seconds;

        System.out.printf(Locale.ROOT, "%-32s %10d rows  %8.3f s  %,12.0f rows/s  %11.1f  %10.1f  %10.1f  %s%n",
                label, rows, seconds, rowsPerSec, decodedMBps, diskMBps, arrowMBps, m);

        if (m.uncompressedBytesDecoded() <= 0 || m.compressedBytesRead() <= 0 || m.arrowBytesProduced() <= 0) {
            System.out.println("  ^ WARNING: a byte counter is zero (decoded=" + m.uncompressedBytesDecoded()
                    + ", disk=" + m.compressedBytesRead() + ", arrow=" + m.arrowBytesProduced()
                    + ") -- the MB/s columns above are NOT valid; byte accounting is not wired as expected");
        }
        if (expectedRows > 0 && Math.abs(rows - expectedRows) > Math.max(1, expectedRows / 100)) {
            System.out.printf(Locale.ROOT, "  ^ WARNING: expected ~%d rows, got %d -- check the fixture/predicate before trusting this number%n",
                    expectedRows, rows);
        }
    }

    private static double mb(long bytes) {
        return bytes / (1024.0 * 1024.0);
    }

    private static void drain(BufferAllocator alloc, ParquetScan scan) {
        try (ParquetBatchReader r = scan.open(alloc)) {
            while (r.next()) {
                r.root().getRowCount();
            }
        }
    }

    private static void printHeader() {
        System.out.printf(Locale.ROOT, "%-32s %10s  %8s  %14s  %11s  %10s  %10s  %s%n",
                "scenario", "rows", "median s", "rows/s", "decodedMB/s", "diskMB/s", "arrowMB/s", "metrics");
    }
}