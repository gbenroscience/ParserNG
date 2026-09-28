package com.github.gbenroscience.parser.ng.parquet.v1.bench;

import com.github.gbenroscience.parser.ng.parquet.util.RandomParquetFiles;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetBatchReader;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;
import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import com.github.gbenroscience.parser.ng.parquet.v1.ScanMetrics;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

import org.openjdk.jmh.annotations.AuxCounters;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.results.Result;
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.ChainedOptionsBuilder;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Comparator;
import java.util.Locale;
import java.util.Map;

/**
 * On terminal, run:
 * mvn -Pjmh clean package
 * Followed by:
 * java --add-modules=jdk.incubator.vector -jar target/benchmarks.jar DecodeJmhBenchmark
 * 
 * JMH replacement for {@link DecodeBenchmark}: same eight scenarios, same fixtures (same seeds, same
 * row counts, same row-group size), measured under fork isolation with JIT warmup instead of
 * {@code DecodeBenchmark}'s single-process "2 warmup + 5 timed, take the median" loop.
 *
 * <h2>Fixed in this revision: {@code Mode.AverageTime} + {@code @AuxCounters} do not combine the way
 * a first pass at this class assumed</h2>
 * A real run of the previous revision produced this for the {@code rows} aux counter:
 * <pre>{@code
 * DecodeJmhBenchmark.flatFullScanSeq:rows    avgt   10    ≈ 10??            ms/op
 * DecodeJmhBenchmark.nestedFullScanSeq:rows  avgt   10     0.001 ± 0.001    ms/op
 * }</pre>
 * Neither number is a rate. {@code AuxCounters.Type.OPERATIONS} normalizes its counter the SAME way
 * JMH normalizes the primary metric for whatever {@code @BenchmarkMode} is active. Under
 * {@code AverageTime}, the primary metric is {@code totalTime / totalInvocations} (ms per op) — so
 * the aux counter became {@code totalTime / totalCounterSum}: milliseconds PER ROW, the inverse of
 * "rows per second." The two numbers above are internally consistent with that, not garbage: 760 ms
 * / 1,000,000 rows ≈ 0.00076 ms/row, which rounds to the displayed {@code 0.001}; 1737 ms /
 * 5,000,000 rows ≈ 0.00035 ms/row, small enough that JMH's console formatter switched to scientific
 * notation with a Unicode superscript exponent, which is what actually rendered as {@code ≈ 10??} —
 * a display/encoding artifact of copying that notation out of one particular terminal, layered on
 * top of the real, separate bug (the wrong unit entirely).
 *
 * <p>The fix is {@code Mode.Throughput}, not a formatting tweak: JMH's own documented examples pair
 * {@code @AuxCounters} with {@code Throughput} specifically because that mode's primary metric is
 * {@code totalOps / totalTime} (ops/sec), so by the same normalization rule the aux counter becomes
 * {@code totalCounterSum / totalTime} — rows per second, directly, landing in the
 * hundreds-of-thousands-to-millions range where JMH's formatter never needs scientific notation at
 * all. Byte-valued aux counters ({@code decodedMB} and friends, below) ride the same mechanism for the
 * same reason, giving genuine, correctly-scaled MB/sec columns.
 *
 * <h2>What the MB/sec numbers are: measured bytes, not file size</h2>
 * The previous revision credited every invocation with the fixture's whole on-disk size. That
 * overstated exactly the scenarios you most want to trust: {@link #flatProjectedSeq} decodes 2 of 7
 * columns and {@link #selectivePredicateSeq} skips most row groups and pages, yet both were charged
 * for the full file. This revision enables {@code ScanMetrics} on every scan and feeds JMH's
 * {@code @AuxCounters} from the counters the decode path now records:
 * <ul>
 *   <li>{@code decodedMB} -- {@code ScanMetrics#uncompressedBytesDecoded()}: decompressed page bytes
 *       (dictionary + data pages, incl. rep/def levels) actually loaded by the decode engine. Respects
 *       projection and page pruning exactly. <b>Quote this one as decode throughput (MB/s).</b></li>
 *   <li>{@code compressedMB} -- {@code ScanMetrics#compressedBytesRead()}: as-stored bytes of the
 *       projected column chunks of the row groups read (an upper bound under page-level pruning, since
 *       the footer only knows whole column chunks). Divide {@code decodedMB} by it for the compression
 *       ratio.</li>
 *   <li>{@code arrowMB} -- {@code ScanMetrics#arrowBytesProduced()}: Arrow buffer bytes materialized.</li>
 * </ul>
 * Each is normalized by JMH per second of measured time (see the {@code Mode.Throughput} section above),
 * so the printed score is directly MB/s. These are <b>warm-cache, end-to-end</b> rates: every scan opens
 * the file, reads, decompresses, decodes and closes, and the fixtures were just written so the OS page
 * cache serves the reads. They are not cold-disk numbers.
 *
 * <p><b>Measurement overhead:</b> enabling metrics costs a couple of {@code System.nanoTime()} calls, a
 * handful of {@code LongAdder} increments and one {@code getBufferSize()} pass per batch (default batch:
 * 32,768 rows), i.e. noise next to decoding that many rows. It is applied uniformly to every scenario, so
 * relative comparisons are unaffected; if you want the absolute floor, compare rows/s against a run of
 * the previous revision.
 *
 * <p><b>Verification status:</b> the byte counters this class reads were verified against real
 * parquet-java 1.13.1 / Arrow 12.0.1 jars (see {@link DecodeBenchmark}'s class Javadoc for what was
 * cross-checked, and for the version caveat: not yet run on the pom's 1.18.0 / Arrow 19). This JMH class
 * itself was compiled against hand-written stubs of the JMH API and has <em>not</em> been executed under
 * real JMH, so the {@code @AuxCounters} wiring is verified only by the same reasoning as the previous
 * revision. {@code main}'s summary marks any scenario whose byte counters came back zero as invalid.
 *
 * <h2>Everything else, unchanged from the previous revision</h2>
 * <ul>
 * <li>The fixtures: {@link RandomParquetFiles#SAMPLE_FLAT_SCHEMA} (5,000,000 rows, seed 42), the
 * hand-written {@code id}/{@code bucket} selectivity fixture (~10% match by construction,
 * {@link #writeSelectivityFixture}), and {@link RandomParquetFiles#SAMPLE_NESTED_SCHEMA} (1,000,000
 * rows, seed 7) — built once per fork in {@link #setup()}, not once per invocation.</li>
 * <li>The eight scenarios: flat full scan, flat projected (2 of 7 columns), flat parallel(2/4/8),
 * ~10%-selectivity predicate scan, nested full scan, nested parallel(4).</li>
 * <li>Fork isolation ({@code @Fork(2)}) and real JMH warmup, separated from the measured result —
 * see the previous revision's Javadoc (preserved below) for why this beats {@code DecodeBenchmark}'s
 * single-process approach.</li>
 * </ul>
 *
 * <h2>Fixture cost is excluded from the measurement</h2>
 * Writing three Parquet files (5M + 5M + 1M rows) happens once per fork in {@link #setup()}
 * ({@code @Setup(Level.Trial)}), before any timed iteration. {@link #tearDown()} deletes them again,
 * best-effort, at the end of each fork's trial.
 *
 * <h2>Running it</h2>
 * Two ways, both requiring the {@code jdk.incubator.vector} module (see this module's {@code pom.xml},
 * whose {@code jmh} profile this class's Maven commands below activate):
 * <pre>{@code
 * # Quick, single-JVM-per-run-of-main convenience path (no shading needed). classpathScope=compile
 * # (not the exec-maven-plugin default) so the "provided"-scope jmh-core/jmh-generator-annprocess
 * # dependencies are actually on the classpath -- "provided" is part of Maven's compile classpath,
 * # not its runtime one:
 * mvn -pl parser-ng-parquet -Pjmh compile exec:java \
 *     -Dexec.mainClass=com.github.gbenroscience.parser.ng.parquet.v1.bench.DecodeJmhBenchmark \
 *     -Dexec.classpathScope=compile
 *
 * # Standard JMH packaging (proper multi-fork isolation, recommended for real numbers):
 * mvn -pl parser-ng-parquet -Pjmh clean package
 * java --add-modules=jdk.incubator.vector -jar parser-ng-parquet/target/benchmarks.jar DecodeJmhBenchmark
 * }</pre>
 * The {@code exec:java} path only forwards {@code -wi}/{@code -i}/{@code -f} (see {@link #main});
 * for the full JMH CLI ({@code -r}, {@code -prof}, {@code -rf}, a regex to select a benchmark
 * subset, etc.) use {@code benchmarks.jar}, which gets JMH's real argument parser.
 *
 * <p><b>Expect this to take a few minutes</b>, not seconds: 8 benchmark methods x 2 forks x
 * (2 warmup + 5 measurement) iterations x ~1s target each, plus JVM startup per fork (16 JVM starts
 * total). Pass {@code -wi 1 -i 3 -f 1} for a faster, noisier sanity check while iterating locally.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(java.util.concurrent.TimeUnit.SECONDS)
@Warmup(iterations = 2, time = 1, timeUnit = java.util.concurrent.TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = java.util.concurrent.TimeUnit.SECONDS)
@Fork(value = 2, jvmArgsAppend = {
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "-Dio.netty.tryReflectionSetAccessible=true",
        "-Darrow.allocation.manager.type=Netty",
		"-Dio.netty.noUnsafe=false"
})
@State(Scope.Benchmark)
public class DecodeJmhBenchmark {

    // ------------------------------------------------------------ fixtures (built once per fork)
    Path dir;
    Path flatFile;
    Path selectivityFile;
    Path nestedFile;
    BufferAllocator allocator;

    static final int FLAT_ROWS = 5_000_000;
    static final int NESTED_ROWS = 1_000_000;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        dir = Files.createTempDirectory("parser-ng-parquet-jmh");
        try {
            flatFile = dir.resolve("flat.parquet");
            RandomParquetFiles.write(flatFile, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, FLAT_ROWS,
                    RandomParquetFiles.Config.defaults().seed(42).rowGroupSize(8L * 1024 * 1024));

            selectivityFile = dir.resolve("selectivity.parquet");
            writeSelectivityFixture(selectivityFile, FLAT_ROWS);

            nestedFile = dir.resolve("nested.parquet");
            RandomParquetFiles.write(nestedFile, RandomParquetFiles.SAMPLE_NESTED_SCHEMA, NESTED_ROWS,
                    RandomParquetFiles.Config.defaults().seed(7).collectionSize(0, 8).rowGroupSize(8L * 1024 * 1024));

            allocator = new RootAllocator();
        } catch (IOException | RuntimeException e) {
            // JMH does not call @TearDown for a trial whose @Setup didn't complete, so a failure
            // partway through (e.g. the nested write failing after flat+selectivity already landed
            // on disk) would otherwise leak whatever was already written. Clean up here too, then
            // let the original failure propagate.
            deleteFixtures(dir);
            throw e;
        }
    }

    private static final double BYTES_PER_MB = 1024.0 * 1024.0;

    @TearDown(Level.Trial)
    public void tearDown() {
        allocator.close();
        if (dir != null) {
            deleteFixtures(dir);
        }
    }

    /**
     * Recursively deletes {@code dir} (and everything under it), best-effort: a failure here is
     * logged to stderr, never thrown, so cleanup can never turn a successful trial into a failed
     * one, and never masks a real exception already propagating out of {@link #setup()} or a
     * {@code @Benchmark} method. Mirrors {@code DecodeBenchmark#deleteFixtures} exactly, so both
     * harnesses clean up their 5M+5M+1M-row fixtures the same way.
     */
    private static void deleteFixtures(Path dir) {
        try (var paths = Files.walk(dir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
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

    /**
     * id: 0..rowCount-1; bucket: id/(rowCount/10) -- 10 buckets, monotonic in id, prunable by stats.
     * Copied verbatim from {@code DecodeBenchmark#writeSelectivityFixture} so both harnesses measure
     * the identical fixture.
     */
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

    // ------------------------------------------------------------ rows/sec + MB/sec, via JMH aux counters
    /**
     * Accumulates rows and MEASURED megabytes within one measured iteration; JMH divides each by the
     * iteration's elapsed time under {@code Mode.Throughput} (see this class's Javadoc for why that
     * specific mode is load-bearing here, not incidental) and reports the results as {@code rows/s},
     * {@code decodedMB/s}, {@code compressedMB/s} and {@code arrowMB/s} next to the primary
     * {@code ops/s} score.
     *
     * <p>Deliberately has NO public no-argument methods returning a number: JMH would treat those as
     * additional counters. {@link #add} takes parameters and returns void, so it is not one.
     */
    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.OPERATIONS)
    public static class Counters {

        public long rows;
        public double decodedMB;
        public double compressedMB;
        public double arrowMB;

        @Setup(Level.Iteration)
        public void reset() {
            rows = 0;
            decodedMB = 0;
            compressedMB = 0;
            arrowMB = 0;
        }

        void add(long scanRows, ScanMetrics m) {
            rows += scanRows;
            decodedMB += m.uncompressedBytesDecoded() / BYTES_PER_MB;
            compressedMB += m.compressedBytesRead() / BYTES_PER_MB;
            arrowMB += m.arrowBytesProduced() / BYTES_PER_MB;
        }
    }

    // ------------------------------------------------------------ scenarios
    @Benchmark
    public void flatFullScanSeq(Counters counter, Blackhole bh) throws IOException {
        drain(ParquetScan.scan(flatFile), bh, counter);
    }

    @Benchmark
    public void flatProjectedSeq(Counters counter, Blackhole bh) throws IOException {
        drain(ParquetScan.scan(flatFile).select("id", "value"), bh, counter);
    }

    @Benchmark
    public void flatFullScanParallel2(Counters counter, Blackhole bh) throws IOException {
        drain(ParquetScan.scan(flatFile).parallelism(2), bh, counter);
    }

    @Benchmark
    public void flatFullScanParallel4(Counters counter, Blackhole bh) throws IOException {
        drain(ParquetScan.scan(flatFile).parallelism(4), bh, counter);
    }

    @Benchmark
    public void flatFullScanParallel8(Counters counter, Blackhole bh) throws IOException {
        drain(ParquetScan.scan(flatFile).parallelism(8), bh, counter);
    }

    /**
     * ~10% selectivity by construction -- see {@link #writeSelectivityFixture}. Row count varies
     * slightly around 500,000 depending on where row-group/page boundaries fall relative to the
     * bucket cutoff (pruning here is pushdown-only, so it returns a SUPERSET, not the exact match
     * count); {@link Counters#rows} reports whatever was actually read, so this is measured honestly
     * rather than assumed. Its MB/s is now measured too: only the pages of the row groups/pages that
     * survived pruning are counted, so {@code decodedMB/s} is a real decode rate for this scenario, no
     * longer the inflated "as if the whole file were this selective" figure of earlier revisions.
     */
    @Benchmark
    public void selectivePredicateSeq(Counters counter, Blackhole bh) throws IOException {
        drain(ParquetScan.scan(selectivityFile).pushdown(Predicate.ge("bucket", 9L)), bh, counter);
    }

    @Benchmark
    public void nestedFullScanSeq(Counters counter, Blackhole bh) throws IOException {
        drain(ParquetScan.scan(nestedFile), bh, counter);
    }

    @Benchmark
    public void nestedFullScanParallel4(Counters counter, Blackhole bh) throws IOException {
        drain(ParquetScan.scan(nestedFile).parallelism(4), bh, counter);
    }

    /**
     * Drives one full scan to completion: pull every batch, touch nothing but the row count, then
     * record the scan's rows and measured bytes into {@code counter}.
     * {@link Blackhole#consume} guards that touch against dead-code elimination; the decode work
     * itself (filling the {@code FieldVector}s inside {@code r.next()}) is a side effect of file I/O
     * and of mutating Arrow buffers reachable from the caller, which the JIT cannot eliminate
     * regardless.
     *
     * <p>Metrics are read after the last {@code next()} and before {@code close()}. For parallel scans
     * every worker has finished by then ({@code next()} only returns false after draining every
     * row-group future), so nothing is still being counted.
     */
    private void drain(ParquetScan scan, Blackhole bh, Counters counter) throws IOException {
        long rows = 0;
        try (ParquetBatchReader r = scan.withMetrics(true).open(allocator)) {
            while (r.next()) {
                int n = r.root().getRowCount();
                bh.consume(n);
                rows += n;
            }
            counter.add(rows, r.metrics());
        }
    }

    // ------------------------------------------------------------ standalone entry point
    /**
     * {@code mvn -pl parser-ng-parquet -Pjmh compile exec:java -Dexec.mainClass=...DecodeJmhBenchmark
     *  -Dexec.classpathScope=compile -Dexec.args="-wi 1 -i 3 -f 1"} -- forwards CLI args to JMH's own
     * option parser for {@code -wi}/{@code -i}/{@code -f} (see this class's Javadoc for the
     * {@code -Pjmh} packaging route, which gets the full JMH CLI instead).
     *
     * <p>After the run, prints one summary line per benchmark laying JMH's own aux-counter scores out
     * side by side (rows/sec and the three measured MB/sec figures) next to ops/sec for
     * cross-checking. Nothing is re-derived here: under {@code Throughput}, JMH already divides each
     * counter by measured time (see this class's Javadoc for why {@code AverageTime} +
     * {@code @AuxCounters} was the actual bug two revisions ago).
     *
     * <p>{@code getSecondaryResults()}' key lookup ({@link #scoreOf}) is the one piece of this method not
     * exercised against real JMH; if the summary's columns come out {@code n/a}, look there first. The
     * benchmark run itself and JMH's own raw printed table do not depend on this method at all.
     */
    public static void main(String[] args) throws RunnerException {
        ChainedOptionsBuilder b = new OptionsBuilder().include(DecodeJmhBenchmark.class.getSimpleName());
        applyArgs(b, args);
        Collection<RunResult> results = new Runner(b.build()).run();
        printSummary(results);
    }

    private static void applyArgs(ChainedOptionsBuilder b, String[] args) {
        for (int i = 0; i < args.length - 1; i++) {
            switch (args[i]) {
                case "-wi" -> b.warmupIterations(Integer.parseInt(args[++i]));
                case "-i" -> b.measurementIterations(Integer.parseInt(args[++i]));
                case "-f" -> b.forks(Integer.parseInt(args[++i]));
                default -> {
                    /* unrecognized: ignore rather than fail a convenience entry point */
                }
            }
        }
    }

    private static void printSummary(Collection<RunResult> results) {
        System.out.println();
        System.out.println("=== Summary (all rates per second of measured time) ===");
        System.out.printf(Locale.ROOT, "%-26s %10s %14s %13s %12s %11s%n",
                "benchmark", "ops/sec", "rows/sec", "decodedMB/s", "diskMB/s", "arrowMB/s");
        for (RunResult r : results) {
            String name = r.getParams().getBenchmark();
            String shortName = name.substring(name.lastIndexOf('.') + 1);
            double opsPerSec = r.getPrimaryResult().getScore(); // Mode.Throughput -> already ops/sec
            Map<String, Result> secondary = r.getSecondaryResults();
            Double decoded = scoreOf(secondary, "decodedMB");
            System.out.printf(Locale.ROOT, "%-26s %10.4f %14s %13s %12s %11s%s%n",
                    shortName, opsPerSec,
                    fmt(scoreOf(secondary, "rows")),
                    fmt(decoded),
                    fmt(scoreOf(secondary, "compressedMB")),
                    fmt(scoreOf(secondary, "arrowMB")),
                    (decoded == null || decoded <= 0) ? "   <-- INVALID: byte counters missing/zero" : "");
        }
        System.out.println("decodedMB/s = decompressed page bytes actually decoded (quote this as decode throughput); "
                + "diskMB/s = as-stored bytes of the projected column chunks read (upper bound under page pruning); "
                + "arrowMB/s = Arrow bytes produced. Warm OS page cache, end-to-end per scan.");
    }

    private static String fmt(Double v) {
        return v == null ? "n/a" : String.format(Locale.ROOT, "%,.1f", v);
    }

    private static Double scoreOf(Map<String, Result> secondary, String label) {
        for (Map.Entry<String, Result> e : secondary.entrySet()) {
            if (e.getKey().equals(label) || e.getKey().endsWith(":" + label)) {
                return e.getValue().getScore();
            }
        }
        return null;
    }
}