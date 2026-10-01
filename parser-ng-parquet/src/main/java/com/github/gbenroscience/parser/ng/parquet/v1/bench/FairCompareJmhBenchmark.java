package com.github.gbenroscience.parser.ng.parquet.v1.bench;

import com.github.gbenroscience.parser.ng.parquet.util.RandomParquetFiles;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetBatchReader;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampMilliTZVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.api.ReadSupport;
import org.apache.parquet.hadoop.example.GroupReadSupport;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalInputFile;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;

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
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Comparator;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

/**
 * JMH head-to-head: {@code parser-ng-parquet} vs parquet-java {@code GroupReadSupport}.
 *
 * <p>JMH counterpart of {@link FairCompareBenchmark}. Same fairness rules, measured under fork
 * isolation and real JIT warmup instead of a single-process median-of-N loop.
 *
 * <h2>Fairness rules (unchanged from {@link FairCompareBenchmark})</h2>
 * <ul>
 *   <li>Same fixture file, seed, row-group size, codec for each paired workload.</li>
 *   <li>Both paths materialize every selected column value (sink + {@link Blackhole} prevent DCE).</li>
 *   <li>Warm page cache: fixtures are written once per fork in {@link #setup()}, before any timed
 *       iteration.</li>
 *   <li>parser-ng product = Arrow {@code VectorSchemaRoot} batches.</li>
 *   <li>parquet-java product = boxed {@code Group} rows ({@code GroupReadSupport} via
 *       {@link LocalInputFile}, so Hadoop {@code FileSystem}/UGI is never touched).</li>
 * </ul>
 *
 * <p>This is intentionally <em>columnar Arrow decode</em> vs the <em>typical JVM row-object path</em>,
 * not two equivalent columnar engines.
 *
 * <h2>Why {@code Mode.Throughput} + {@code @AuxCounters}</h2>
 * Same reasoning as {@link DecodeJmhBenchmark}: under {@code Throughput}, an
 * {@code AuxCounters.Type.OPERATIONS} counter is normalized as
 * {@code totalCounterSum / totalTime}, so {@code rows} prints directly as rows/sec. Do not switch
 * this class to {@code AverageTime} without also changing how the aux counter is interpreted.
 *
 * <h2>Workloads (six benchmarks, three files)</h2>
 * <ul>
 *   <li>{@code flat7colUncPng} / {@code flat7colUncGroup} — 7-column mix, uncompressed, 15% nulls.</li>
 *   <li>{@code flat7colSnappyPng} / {@code flat7colSnappyGroup} — same schema, SNAPPY.</li>
 *   <li>{@code singleInt64UncPng} / {@code singleInt64UncGroup} — one required INT64, PLAIN, no nulls.</li>
 * </ul>
 *
 * <h2>Running</h2>
 * <pre>{@code
 * mvn -Pjmh clean package
 * java --add-modules=jdk.incubator.vector \
 *      --enable-native-access=ALL-UNNAMED \
 *      --add-opens=java.base/java.nio=ALL-UNNAMED \
 *      -Dio.netty.tryReflectionSetAccessible=true \
 *      -Dio.netty.noUnsafe=false \
 *      -Darrow.allocation.manager.type=Netty \
 *      -jar target/benchmarks.jar FairCompareJmhBenchmark
 * }</pre>
 *
 * <p>Fork JVM flags are also declared in {@code @Fork(jvmArgsAppend=...)} so forked measurement
 * JVMs get the Vector API / Netty / Arrow opens. Launcher-only flags are not enough.
 *
 * <p>Default fixture sizes match a full-scale laptop run (5M / 10M). On memory-constrained hosts,
 * lower {@link #FLAT_ROWS} / {@link #SINGLE_ROWS} before packaging, or pass JMH
 * {@code -jvmArgsPrepend -Xmx...} as appropriate.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 2, jvmArgsAppend = {
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "-Dio.netty.tryReflectionSetAccessible=true",
        "-Darrow.allocation.manager.type=Netty",
        "-Dio.netty.noUnsafe=false"
})
@State(Scope.Benchmark)
public class FairCompareJmhBenchmark {

    // ------------------------------------------------------------ sizing
    /**
     * Rows for the 7-column fixtures. 5_000_000 matches the full-scale fair-compare run on a
     * 16 GB Skylake laptop; drop to 500_000 on ~1–2 GiB guests.
     */
    static final int FLAT_ROWS = 5_000_000;

    /**
     * Rows for the single-INT64 fixture. 10_000_000 matches the full-scale fair-compare run.
     */
    static final int SINGLE_ROWS = 10_000_000;

    private static final long ROW_GROUP_BYTES = 128L << 20;

    private static final MessageType SINGLE_INT64_SCHEMA = MessageTypeParser.parseMessageType(
            "message single { required int64 id; }");

    // ------------------------------------------------------------ fixtures (once per fork)
    Path dir;
    Path flatUnc;
    Path flatSnappy;
    Path singleUnc;
    BufferAllocator allocator;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        dir = Files.createTempDirectory("fair-compare-jmh");
        try {
            flatUnc = dir.resolve("flat-7col-unc.parquet");
            flatSnappy = dir.resolve("flat-7col-snappy.parquet");
            singleUnc = dir.resolve("single-int64-unc.parquet");

            RandomParquetFiles.write(flatUnc, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, FLAT_ROWS,
                    RandomParquetFiles.Config.defaults()
                            .seed(42)
                            .nullProbability(0.15)
                            .compression(CompressionCodecName.UNCOMPRESSED)
                            .rowGroupSize(ROW_GROUP_BYTES));

            RandomParquetFiles.write(flatSnappy, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, FLAT_ROWS,
                    RandomParquetFiles.Config.defaults()
                            .seed(42)
                            .nullProbability(0.15)
                            .compression(CompressionCodecName.SNAPPY)
                            .rowGroupSize(ROW_GROUP_BYTES));

            RandomParquetFiles.write(singleUnc, SINGLE_INT64_SCHEMA, SINGLE_ROWS,
                    RandomParquetFiles.Config.defaults()
                            .seed(7)
                            .nullProbability(0.0)
                            .compression(CompressionCodecName.UNCOMPRESSED)
                            .rowGroupSize(ROW_GROUP_BYTES));

            allocator = new RootAllocator();
        } catch (IOException | RuntimeException e) {
            deleteFixtures(dir);
            throw e;
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (allocator != null) {
            allocator.close();
        }
        if (dir != null) {
            deleteFixtures(dir);
        }
    }

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

    // ------------------------------------------------------------ rows/sec via JMH aux counters
    /**
     * Under {@code Mode.Throughput}, {@code rows} is reported as rows/sec (total rows summed across
     * invocations in the iteration, divided by iteration wall time). No public no-arg numeric
     * getters: JMH would treat those as extra counters.
     */
    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.OPERATIONS)
    public static class Counters {

        public long rows;

        @Setup(Level.Iteration)
        public void reset() {
            rows = 0;
        }

        void add(long scanRows) {
            rows += scanRows;
        }
    }

    // ------------------------------------------------------------ benchmarks: flat 7-col UNC

    @Benchmark
    public void flat7colUncPng(Counters counter, Blackhole bh) throws IOException {
        drainParserNg(flatUnc, true, counter, bh);
    }

    @Benchmark
    public void flat7colUncGroup(Counters counter, Blackhole bh) throws IOException {
        drainGroup(flatUnc, true, counter, bh);
    }

    // ------------------------------------------------------------ benchmarks: flat 7-col SNAPPY

    @Benchmark
    public void flat7colSnappyPng(Counters counter, Blackhole bh) throws IOException {
        drainParserNg(flatSnappy, true, counter, bh);
    }

    @Benchmark
    public void flat7colSnappyGroup(Counters counter, Blackhole bh) throws IOException {
        drainGroup(flatSnappy, true, counter, bh);
    }

    // ------------------------------------------------------------ benchmarks: single INT64 UNC

    @Benchmark
    public void singleInt64UncPng(Counters counter, Blackhole bh) throws IOException {
        drainParserNg(singleUnc, false, counter, bh);
    }

    @Benchmark
    public void singleInt64UncGroup(Counters counter, Blackhole bh) throws IOException {
        drainGroup(singleUnc, false, counter, bh);
    }

    // ------------------------------------------------------------ parser-ng drain

    /**
     * Full sequential scan; materialize every selected value into {@code sink}, publish row count
     * to {@code counter} and {@code bh}.
     */
    private void drainParserNg(Path file, boolean multiCol, Counters counter, Blackhole bh)
            throws IOException {
        long rows = 0;
        long sink = 0;
        try (ParquetBatchReader r = ParquetScan.scan(file).open(allocator)) {
            while (r.next()) {
                VectorSchemaRoot root = r.root();
                int n = root.getRowCount();
                rows += n;
                if (multiCol) {
                    sink += touchAllFlatColumns(root, n);
                } else {
                    BigIntVector id = (BigIntVector) root.getVector("id");
                    for (int i = 0; i < n; i++) {
                        sink += id.get(i);
                    }
                }
            }
        }
        counter.add(rows);
        bh.consume(rows);
        bh.consume(sink);
    }

    private static long touchAllFlatColumns(VectorSchemaRoot root, int n) {
        long sink = 0;
        BigIntVector idV = (BigIntVector) root.getVector("id");
        Float8Vector valueV = (Float8Vector) root.getVector("value");
        Float4Vector ratioV = (Float4Vector) root.getVector("ratio");
        BitVector flagV = (BitVector) root.getVector("flag");
        VarCharVector nameV = (VarCharVector) root.getVector("name");
        FieldVector day = root.getVector("day");
        FieldVector seenAt = root.getVector("seen_at");

        for (int i = 0; i < n; i++) {
            sink += idV.get(i);
            if (!valueV.isNull(i)) {
                sink += Double.doubleToRawLongBits(valueV.get(i));
            }
            if (!ratioV.isNull(i)) {
                sink += Float.floatToRawIntBits(ratioV.get(i));
            }
            if (!flagV.isNull(i)) {
                sink += flagV.get(i);
            }
            if (!nameV.isNull(i)) {
                sink += nameV.get(i).length;
            }
            if (!day.isNull(i)) {
                if (day instanceof DateDayVector d) {
                    sink += d.get(i);
                } else if (day instanceof IntVector iv) {
                    sink += iv.get(i);
                }
            }
            if (!seenAt.isNull(i)) {
                if (seenAt instanceof TimeStampMilliTZVector t) {
                    sink += t.get(i);
                } else if (seenAt instanceof BigIntVector b) {
                    sink += b.get(i);
                }
            }
        }
        return sink;
    }

    // ------------------------------------------------------------ parquet-java Group drain

    /**
     * Opens a Group reader through {@link LocalInputFile} so measurement never enters Hadoop
     * {@code FileSystem} / UGI (avoids JDK 24+ {@code Subject.getSubject} failures).
     */
    private static ParquetReader<Group> openGroupReader(Path file) throws IOException {
        LocalInputFile input = new LocalInputFile(file.toAbsolutePath());
        return new ParquetReader.Builder<Group>(input) {
            @Override
            protected ReadSupport<Group> getReadSupport() {
                return new GroupReadSupport();
            }
        }.build();
    }

    private void drainGroup(Path file, boolean multiCol, Counters counter, Blackhole bh)
            throws IOException {
        long rows = 0;
        long sink = 0;
        try (ParquetReader<Group> reader = openGroupReader(file)) {
            Group g;
            while ((g = reader.read()) != null) {
                rows++;
                if (multiCol) {
                    sink += touchGroupAllFields(g);
                } else {
                    sink += g.getLong("id", 0);
                }
            }
        }
        counter.add(rows);
        bh.consume(rows);
        bh.consume(sink);
    }

    private static long touchGroupAllFields(Group g) {
        long sink = 0;
        sink += g.getLong("id", 0);
        if (g.getFieldRepetitionCount("value") > 0) {
            sink += Double.doubleToRawLongBits(g.getDouble("value", 0));
        }
        if (g.getFieldRepetitionCount("ratio") > 0) {
            sink += Float.floatToRawIntBits(g.getFloat("ratio", 0));
        }
        sink += g.getBoolean("flag", 0) ? 1 : 0;
        if (g.getFieldRepetitionCount("name") > 0) {
            Binary b = g.getBinary("name", 0);
            sink += b.length();
        }
        if (g.getFieldRepetitionCount("day") > 0) {
            sink += g.getInteger("day", 0);
        }
        if (g.getFieldRepetitionCount("seen_at") > 0) {
            sink += g.getLong("seen_at", 0);
        }
        return sink;
    }

    // ------------------------------------------------------------ standalone entry + summary
    /**
     * Runs only this class's benchmarks and prints a compact paired summary (png vs Group rows/s
     * and speedup) after JMH finishes. Extra CLI args are forwarded to JMH
     * ({@code -f 1}, {@code -wi 1}, etc.).
     */
    /**
     * Convenience entry that runs this class only and prints a paired summary.
     * For full JMH CLI control ({@code -f 1 -wi 1 -i 3 -r 1s}, profilers, etc.) prefer:
     * <pre>{@code
     * java ... -jar target/benchmarks.jar FairCompareJmhBenchmark -f 1 -wi 1 -i 3
     * }</pre>
     * (that path goes through {@code org.openjdk.jmh.Main} and does not print the summary).
     */
    public static void main(String[] args) throws RunnerException {
        if (args.length > 0) {
            System.err.println("Note: extra args are ignored by this main(); use the shaded jar "
                    + "entry (org.openjdk.jmh.Main) to pass JMH flags, e.g.:");
            System.err.println("  java -jar target/benchmarks.jar FairCompareJmhBenchmark -f 1 -wi 1 -i 3");
        }
        Collection<RunResult> results = new Runner(new OptionsBuilder()
                .include(FairCompareJmhBenchmark.class.getSimpleName())
                .shouldFailOnError(true)
                .build()).run();
        printPairedSummary(results);
    }

    /**
     * Pairs {@code *Png} / {@code *Group} scores by shared workload prefix and prints rows/s
     * speedup. Primary JMH score is ops/s (one full scan per op); aux {@code rows} is rows/s.
     */
    private static void printPairedSummary(Collection<RunResult> results) {
        System.out.println();
        System.out.println("=== FairCompare paired summary (aux rows/s) ===");
        System.out.printf(Locale.ROOT, "%-22s  %14s  %14s  %10s%n",
                "workload", "parser-ng", "Group", "speedup");
        System.out.println("-".repeat(66));

        String[] workloads = {"flat7colUnc", "flat7colSnappy", "singleInt64Unc"};
        for (String w : workloads) {
            Double png = findRowsPerSec(results, w + "Png");
            Double group = findRowsPerSec(results, w + "Group");
            if (png == null && group == null) {
                continue;
            }
            String pngS = png == null ? "n/a" : String.format(Locale.ROOT, "%,.0f", png);
            String groupS = group == null ? "n/a" : String.format(Locale.ROOT, "%,.0f", group);
            String speedup = "n/a";
            if (png != null && group != null && group > 0.0) {
                speedup = String.format(Locale.ROOT, "%.2fx", png / group);
            }
            System.out.printf(Locale.ROOT, "%-22s  %14s  %14s  %10s%n", w, pngS, groupS, speedup);
        }
        System.out.println();
        System.out.println("Primary JMH score is scans/sec (ops/s). Aux 'rows' is rows/sec.");
        System.out.println("Speedup = parser-ng rows/s ÷ Group rows/s.");
    }

    private static Double findRowsPerSec(Collection<RunResult> results, String methodSuffix) {
        for (RunResult rr : results) {
            String label = rr.getParams().getBenchmark();
            if (label == null || !label.endsWith("." + methodSuffix)) {
                continue;
            }
            var secondary = rr.getSecondaryResults();
            if (secondary != null) {
                for (var e : secondary.entrySet()) {
                    String key = e.getKey();
                    // JMH labels aux counters like "rows" or "·rows" depending on version/format
                    if (key != null && key.toLowerCase(Locale.ROOT).contains("rows")) {
                        return e.getValue().getScore();
                    }
                }
            }
            return null;
        }
        return null;
    }
}