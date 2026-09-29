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

import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Locale;

/**
 * Fair head-to-head: parser-ng-parquet vs parquet-java {@code Group} reader.
 *
 * <p><b>Fairness rules</b>
 * <ul>
 *   <li>Same fixture file, seed, row-group size, codec.</li>
 *   <li>Both paths materialize every selected column value (sink prevents DCE).</li>
 *   <li>Warm page cache; fixture written before any timed run.</li>
 *   <li>2 warmup + 5 timed iterations; report the median timed run.</li>
 *   <li>parser-ng product = Arrow {@code VectorSchemaRoot} batches.</li>
 *   <li>parquet-java product = boxed {@code Group} rows ({@code GroupReadSupport}).</li>
 * </ul>
 *
 * <p>This is intentionally "columnar Arrow decode" vs "typical JVM row-object path",
 * which is the structural claim in the blog — not two equivalent columnar engines.
 */
public final class FairCompareBenchmark {

    private static final int WARMUP = 2;
    private static final int TIMED = 5;
    private static final int FLAT_ROWS = 5_000_000;
    private static final int SINGLE_ROWS = 10_000_000;

    private static final MessageType SINGLE_INT64_SCHEMA = MessageTypeParser.parseMessageType(
            "message single { required int64 id; }");

    public static void main(String[] args) throws Exception {
        int flatRows = args.length > 0 ? Integer.parseInt(args[0]) : FLAT_ROWS;
        int singleRows = args.length > 1 ? Integer.parseInt(args[1]) : SINGLE_ROWS;

        java.nio.file.Path dir = Files.createTempDirectory("fair-compare");
        System.out.println("Fixtures in " + dir);
        System.out.printf(Locale.ROOT, "Machine: %d cores, Java %s%n",
                Runtime.getRuntime().availableProcessors(),
                System.getProperty("java.version"));
        System.out.println();
        System.out.println("Methodology: same file; materialize all selected values;");
        System.out.println("  warm cache; " + WARMUP + " warmup + " + TIMED + " timed; median.");
        System.out.println("parser-ng product     = Arrow VectorSchemaRoot batches");
        System.out.println("parquet-java product  = boxed Group rows (GroupReadSupport)");
        System.out.println();

        try {
            java.nio.file.Path flatUnc = dir.resolve("flat-7col-unc.parquet");
            java.nio.file.Path flatSnappy = dir.resolve("flat-7col-snappy.parquet");
            java.nio.file.Path singleUnc = dir.resolve("single-int64-unc.parquet");

            System.out.println("Writing fixtures...");
            RandomParquetFiles.write(flatUnc, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, flatRows,
                    RandomParquetFiles.Config.defaults()
                            .seed(42)
                            .nullProbability(0.15)
                            .compression(CompressionCodecName.UNCOMPRESSED)
                            .rowGroupSize(128L << 20));
            RandomParquetFiles.write(flatSnappy, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, flatRows,
                    RandomParquetFiles.Config.defaults()
                            .seed(42)
                            .nullProbability(0.15)
                            .compression(CompressionCodecName.SNAPPY)
                            .rowGroupSize(128L << 20));
            RandomParquetFiles.write(singleUnc, SINGLE_INT64_SCHEMA, singleRows,
                    RandomParquetFiles.Config.defaults()
                            .seed(7)
                            .nullProbability(0.0)
                            .compression(CompressionCodecName.UNCOMPRESSED)
                            .rowGroupSize(128L << 20));
            System.out.printf(Locale.ROOT, "  flat-7col UNC    %s (%d rows)%n", sizeOf(flatUnc), flatRows);
            System.out.printf(Locale.ROOT, "  flat-7col SNAPPY %s (%d rows)%n", sizeOf(flatSnappy), flatRows);
            System.out.printf(Locale.ROOT, "  single-int64 UNC %s (%d rows)%n", sizeOf(singleUnc), singleRows);
            System.out.println();

            printHeader();
            runPair("flat-7col UNC", flatUnc, flatRows, true);
            runPair("flat-7col SNAPPY", flatSnappy, flatRows, true);
            runPair("single-int64 UNC", singleUnc, singleRows, false);
        } finally {
            deleteRecursive(dir);
        }
    }

    private static void runPair(String label, java.nio.file.Path file, int expectedRows, boolean multiCol)
            throws Exception {
        Result png = time(() -> drainParserNg(file, multiCol));
        Result pj = time(() -> drainParquetJavaGroup(file, multiCol));

        if (png.rows != expectedRows || pj.rows != expectedRows) {
            System.out.printf(Locale.ROOT,
                    "  WARNING: row counts png=%d pj=%d expected=%d%n",
                    png.rows, pj.rows, expectedRows);
        }

        double speedup = pj.medianSec > 0 ? pj.medianSec / png.medianSec : Double.NaN;
        System.out.printf(Locale.ROOT,
                "%-18s  %-22s  %10d  %8.3f  %12s  %10s%n",
                label, "parser-ng", png.rows, png.medianSec,
                formatRate(png.rows, png.medianSec), "");
        System.out.printf(Locale.ROOT,
                "%-18s  %-22s  %10d  %8.3f  %12s  %9.2fx%n",
                label, "parquet-java Group", pj.rows, pj.medianSec,
                formatRate(pj.rows, pj.medianSec), speedup);
        System.out.println();
    }

    private static void printHeader() {
        System.out.printf(Locale.ROOT,
                "%-18s  %-22s  %10s  %8s  %12s  %10s%n",
                "workload", "engine", "rows", "median s", "rows/s", "vs png");
        System.out.println("-".repeat(90));
    }

    // ---- parser-ng ------------------------------------------------------------

    private static long drainParserNg(java.nio.file.Path file, boolean multiCol) throws IOException {
        long rows = 0;
        long sink = 0;
        try (BufferAllocator alloc = new RootAllocator();
             ParquetBatchReader r = ParquetScan.scan(file).open(alloc)) {
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
        blackhole(sink);
        return rows;
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

    // ---- parquet-java Group ---------------------------------------------------

    /**
     * Build a Group reader via LocalInputFile so we never touch Hadoop FileSystem
     * (avoids JDK 24+ {@code Subject.getSubject} breakage in UGI).
     */
    private static ParquetReader<Group> openGroupReader(java.nio.file.Path file) throws IOException {
        LocalInputFile input = new LocalInputFile(file.toAbsolutePath());
        return new ParquetReader.Builder<Group>(input) {
            @Override
            protected ReadSupport<Group> getReadSupport() {
                return new GroupReadSupport();
            }
        }.build();
    }

    private static long drainParquetJavaGroup(java.nio.file.Path file, boolean multiCol) throws IOException {
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
        blackhole(sink);
        return rows;
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

    // ---- timing / util --------------------------------------------------------

    @FunctionalInterface
    private interface Work {
        long run() throws Exception;
    }

    private static final class Result {
        final long rows;
        final double medianSec;

        Result(long rows, double medianSec) {
            this.rows = rows;
            this.medianSec = medianSec;
        }
    }

    private static Result time(Work work) throws Exception {
        long rows = 0;
        for (int i = 0; i < WARMUP; i++) {
            rows = work.run();
        }
        double[] secs = new double[TIMED];
        for (int i = 0; i < TIMED; i++) {
            long t0 = System.nanoTime();
            rows = work.run();
            long t1 = System.nanoTime();
            secs[i] = (t1 - t0) / 1_000_000_000.0;
        }
        Arrays.sort(secs);
        return new Result(rows, secs[TIMED / 2]);
    }

    private static void blackhole(long sink) {
        if (sink == Long.MIN_VALUE) {
            System.out.print("");
        }
    }

    private static String formatRate(long rows, double sec) {
        if (sec <= 0) {
            return "n/a";
        }
        return String.format(Locale.ROOT, "%,.0f", rows / sec);
    }

    private static String sizeOf(java.nio.file.Path p) throws IOException {
        long b = Files.size(p);
        if (b < 1024) {
            return b + " B";
        }
        if (b < 1024 * 1024) {
            return String.format(Locale.ROOT, "%.1f KiB", b / 1024.0);
        }
        return String.format(Locale.ROOT, "%.1f MiB", b / (1024.0 * 1024.0));
    }

    private static void deleteRecursive(java.nio.file.Path dir) {
        try {
            if (!Files.exists(dir)) {
                return;
            }
            try (var walk = Files.walk(dir)) {
                walk.sorted(Comparator.reverseOrder()).forEach(p -> {
                    try {
                        Files.deleteIfExists(p);
                    } catch (IOException ignored) {
                    }
                });
            }
        } catch (IOException ignored) {
        }
    }
}