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
import java.util.ArrayList;
import java.util.List;
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
 * <p><b>This has not been run.</b> The environment this was written in has no Maven and no
 * Parquet/Arrow/Hadoop jars available (network access is restricted to a small allowlist that does
 * not include Maven Central), so there is no way to produce a real number here. Every claim
 * elsewhere in this module about this engine's throughput is architectural reasoning plus two
 * standalone microbenchmarks of individual decode kernels in isolation (see
 * {@code SimdBulkDecode}'s class doc) — neither is a substitute for actually running this class.
 * This file exists so that running it is a five-minute exercise once the module is built somewhere
 * with real dependencies, not a research project.
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
 * <p>Reported per scenario: elapsed wall time, rows/sec, approximate MB/sec (file size on disk /
 * elapsed time -- a rough proxy, not bytes actually decoded, since compression ratio and projection
 * both change how much of the file is actually touched), and the scan's {@link ScanMetrics}.
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
        long fileSize = Files.size(file);

        for (int i = 0; i < WARMUP_ITERS; i++) drain(alloc, cfg.apply(ParquetScan.scan(file)));

        List<Long> nanos = new ArrayList<>();
        ScanMetrics lastMetrics = null;
        long lastRows = 0;
        for (int i = 0; i < TIMED_ITERS; i++) {
            ParquetScan scan = cfg.apply(ParquetScan.scan(file)).withMetrics(true);
            long t0 = System.nanoTime();
            try (ParquetBatchReader r = scan.open(alloc)) {
                long rows = 0;
                while (r.next()) {
                    VectorSchemaRoot root = r.root();
                    rows += root.getRowCount(); // touch the batch; nothing else to "consume" in a pure decode benchmark
                }
                lastRows = rows;
                lastMetrics = r.metrics();
            }
            nanos.add(System.nanoTime() - t0);
        }

        long median = nanos.stream().sorted().toList().get(nanos.size() / 2);
        double seconds = median / 1e9;
        double rowsPerSec = lastRows / seconds;
        double mbPerSec = (fileSize / (1024.0 * 1024.0)) / seconds;

        System.out.printf(Locale.ROOT, "%-32s %10d rows  %8.3f s  %,12.0f rows/s  %8.1f MB/s  %s%n",
                label, lastRows, seconds, rowsPerSec, mbPerSec, lastMetrics);

        if (expectedRows > 0 && Math.abs(lastRows - expectedRows) > Math.max(1, expectedRows / 100)) {
            System.out.printf(Locale.ROOT, "  ^ WARNING: expected ~%d rows, got %d -- check the fixture/predicate before trusting this number%n",
                    expectedRows, lastRows);
        }
    }

    private static void drain(BufferAllocator alloc, ParquetScan scan) {
        try (ParquetBatchReader r = scan.open(alloc)) {
            while (r.next()) {
                r.root().getRowCount();
            }
        }
    }

    private static void printHeader() {
        System.out.printf(Locale.ROOT, "%-32s %10s  %8s  %14s  %9s  %s%n",
                "scenario", "rows", "median s", "rows/s", "MB/s", "metrics");
    }
}