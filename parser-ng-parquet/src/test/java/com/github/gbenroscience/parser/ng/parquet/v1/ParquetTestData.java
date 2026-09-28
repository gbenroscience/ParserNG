package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Deterministic Parquet fixtures for the scan tests. Every value is a pure function of the row index, so tests
 * compare scan output against an in-memory oracle ({@link #rows(int)}) instead of hard-coded expectations, and
 * {@code id} is the row index (sorted), which gives row-group statistics something to prune on.
 */
final class ParquetTestData {

    private ParquetTestData() { }

    /** Rows in the shared flat fixture. */
    static final int N = 5_000;

    /** Small row groups so the shared fixture spans many of them (pruning / parallelism / batching need that). */
    static final long SMALL_ROW_GROUP_BYTES = 8 * 1024;

    static final MessageType FLAT_SCHEMA = MessageTypeParser.parseMessageType(
            "message flat {"
            + " required int64 id;"
            + " optional int32 grp;"
            + " optional binary name (STRING);"
            + " required double score;"
            + " required boolean flag;"
            + " optional int32 day (DATE);"
            + " optional int64 ts (TIMESTAMP(MILLIS,true));"
            + " optional float ratio;"
            + " required int64 lo;"
            + " required int64 hi;"
            + "}");

    static final List<String> FLAT_COLUMNS = List.of("id", "grp", "name", "score", "flag", "day", "ts", "ratio", "lo", "hi");

    static final String[] NAMES = {"alpha", "Beta", "gamma", "alphabet", "delta_1", "deltaX1"};

    static final long TS_BASE = 1_700_000_000_000L;
    static final int DAY_BASE = 18_000;

    /** One logical row of the flat fixture. Nullable columns use boxed types; null means "not written". */
    record Row(long id, Integer grp, String name, double score, boolean flag,
               Integer day, Long ts, Float ratio, long lo, long hi) { }

    static Row row(int i) {
        return new Row(
                i,
                i % 13 == 0 ? null : Integer.valueOf(i % 10),
                i % 11 == 0 ? null : NAMES[i % NAMES.length],
                i * 0.5,
                i % 2 == 0,
                i % 17 == 0 ? null : Integer.valueOf(DAY_BASE + i % 30),
                i % 23 == 0 ? null : Long.valueOf(TS_BASE + i * 1000L),
                i % 19 == 0 ? null : Float.valueOf((float) (i % 5)),
                i,
                (i * 37L) % 1000);
    }

    static List<Row> rows(int n) {
        List<Row> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(row(i));
        return out;
    }

    static final List<Row> ROWS = rows(N);

    // ------------------------------------------------------------------ writing

    static Path writeFlat(Path dir, String fileName, int n, long rowGroupBytes, boolean dictionary, int pageBytes) throws IOException {
        Path file = dir.resolve(fileName);
        SimpleGroupFactory f = new SimpleGroupFactory(FLAT_SCHEMA);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file))
                .withType(FLAT_SCHEMA)
                .withCompressionCodec(CompressionCodecName.UNCOMPRESSED)
                .withRowGroupSize(rowGroupBytes)
                .withPageSize(pageBytes)
                .withDictionaryEncoding(dictionary)
                .build()) {
            for (Row r : rows(n)) {
                Group g = f.newGroup();
                g.append("id", r.id());
                if (r.grp() != null) g.append("grp", r.grp().intValue());
                if (r.name() != null) g.append("name", r.name());
                g.append("score", r.score());
                g.append("flag", r.flag());
                if (r.day() != null) g.append("day", r.day().intValue());
                if (r.ts() != null) g.append("ts", r.ts().longValue());
                if (r.ratio() != null) g.append("ratio", r.ratio().floatValue());
                g.append("lo", r.lo());
                g.append("hi", r.hi());
                w.write(g);
            }
        }
        return file;
    }

    /** The default shared fixture: N rows, many small row groups, dictionary encoding on. */
    static Path writeFlat(Path dir, String fileName) throws IOException {
        Path file = writeFlat(dir, fileName, N, SMALL_ROW_GROUP_BYTES, true, 1024 * 1024);
        int groups = ParquetFileInfo.read(file).rowGroups().size();
        assertTrue(groups >= 3, "fixture must span several row groups for pruning/parallel tests to mean anything, got " + groups);
        return file;
    }

    // ------------------------------------------------------------------ reading

    /** Scans {@code scan} to the end and returns the {@code id} column ("id" must be projected). */
    static List<Long> ids(ParquetScan scan) {
        try (BufferAllocator alloc = new RootAllocator();
             ParquetBatchReader r = scan.open(alloc)) {
            List<Long> out = new ArrayList<>();
            while (r.next()) {
                VectorSchemaRoot b = r.root();
                BigIntVector v = (BigIntVector) b.getVector("id");
                for (int i = 0; i < b.getRowCount(); i++) out.add(v.get(i));
            }
            return out;
        }
    }

    static List<Long> expectedIds(java.util.function.Predicate<Row> oracle) {
        List<Long> out = new ArrayList<>();
        for (Row r : ROWS) if (oracle.test(r)) out.add(r.id());
        return out;
    }

    static String utf8(VarCharVector v, int i) {
        return new String(v.get(i), StandardCharsets.UTF_8);
    }

    /**
     * Scans everything the scan yields and asserts every column of every row equals {@code expected}, in order.
     * The scan must project all of {@link #FLAT_COLUMNS} and must not filter.
     */
    static void assertScanEqualsRows(ParquetScan scan, List<Row> expected) {
        try (BufferAllocator alloc = new RootAllocator();
             ParquetBatchReader r = scan.open(alloc)) {
            int idx = 0;
            while (r.next()) {
                VectorSchemaRoot b = r.root();
                BigIntVector id = (BigIntVector) b.getVector("id");
                IntVector grp = (IntVector) b.getVector("grp");
                VarCharVector name = (VarCharVector) b.getVector("name");
                Float8Vector score = (Float8Vector) b.getVector("score");
                BitVector flag = (BitVector) b.getVector("flag");
                DateDayVector day = (DateDayVector) b.getVector("day");
                TimeStampVector ts = (TimeStampVector) b.getVector("ts");
                Float4Vector ratio = (Float4Vector) b.getVector("ratio");
                BigIntVector lo = (BigIntVector) b.getVector("lo");
                BigIntVector hi = (BigIntVector) b.getVector("hi");
                for (int k = 0; k < b.getRowCount(); k++, idx++) {
                    assertTrue(idx < expected.size(), "scan produced more rows than expected");
                    Row e = expected.get(idx);
                    String at = "row " + idx;
                    assertEquals(e.id(), id.get(k), at + " id");
                    if (e.grp() == null) assertTrue(grp.isNull(k), at + " grp should be null");
                    else { assertFalse(grp.isNull(k), at + " grp should be set"); assertEquals(e.grp().intValue(), grp.get(k), at + " grp"); }
                    if (e.name() == null) assertTrue(name.isNull(k), at + " name should be null");
                    else { assertFalse(name.isNull(k), at + " name should be set"); assertEquals(e.name(), utf8(name, k), at + " name"); }
                    assertEquals(e.score(), score.get(k), 0.0, at + " score");
                    assertEquals(e.flag() ? 1 : 0, flag.get(k), at + " flag");
                    if (e.day() == null) assertTrue(day.isNull(k), at + " day should be null");
                    else { assertFalse(day.isNull(k), at + " day should be set"); assertEquals(e.day().intValue(), day.get(k), at + " day"); }
                    if (e.ts() == null) assertTrue(ts.isNull(k), at + " ts should be null");
                    else { assertFalse(ts.isNull(k), at + " ts should be set"); assertEquals(e.ts().longValue(), ts.get(k), at + " ts"); }
                    if (e.ratio() == null) assertTrue(ratio.isNull(k), at + " ratio should be null");
                    else { assertFalse(ratio.isNull(k), at + " ratio should be set"); assertEquals(e.ratio().floatValue(), ratio.get(k), 0.0f, at + " ratio"); }
                    assertEquals(e.lo(), lo.get(k), at + " lo");
                    assertEquals(e.hi(), hi.get(k), at + " hi");
                }
            }
            assertEquals(expected.size(), idx, "row count");
        }
    }
}