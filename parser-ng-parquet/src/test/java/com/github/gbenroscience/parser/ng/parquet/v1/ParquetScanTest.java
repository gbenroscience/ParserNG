package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import static com.github.gbenroscience.parser.ng.parquet.v1.ParquetTestData.*;
import static org.junit.jupiter.api.Assertions.*;

/** End-to-end tests of {@link ParquetScan} and {@link ParquetBatchReader} against real Parquet files. */
class ParquetScanTest {

    @TempDir
    static Path dir;

    static Path flat;

    @BeforeAll
    static void writeFixture() throws IOException {
        flat = ParquetTestData.writeFlat(dir, "flat.parquet");
    }

    private static List<Long> allIds() {
        return LongStream.range(0, N).boxed().collect(Collectors.toList());
    }

    // ================================================================ full-scan correctness

    @Test
    void fullScanReturnsEveryValueOfEveryColumnIncludingNulls() {
        assertScanEqualsRows(ParquetScan.scan(flat), ROWS);
    }

    @Test
    void smallBatchesDoNotChangeTheData() {
        assertScanEqualsRows(ParquetScan.scan(flat).batchSize(97), ROWS);
    }

    @Test
    void parallelScanReturnsTheSameRowsInTheSameOrder() {
        assertScanEqualsRows(ParquetScan.scan(flat).parallelism(4), ROWS);
    }

    @Test
    void plainEncodedFileDecodesIdentically() throws IOException {
        Path plain = ParquetTestData.writeFlat(dir, "plain.parquet", 2_000, SMALL_ROW_GROUP_BYTES, false, 1024 * 1024);
        assertScanEqualsRows(ParquetScan.scan(plain), ParquetTestData.rows(2_000));
    }

    @Test
    void manySmallPagesDecodeIdentically() throws IOException {
        Path paged = ParquetTestData.writeFlat(dir, "paged.parquet", 2_000, 64 * 1024, true, 512);
        assertScanEqualsRows(ParquetScan.scan(paged).batchSize(300), ParquetTestData.rows(2_000));
    }

    @Test
    void emptyFileYieldsNoBatches() throws IOException {
        Path empty = ParquetTestData.writeFlat(dir, "empty.parquet", 0, SMALL_ROW_GROUP_BYTES, true, 1024);
        assertEquals(List.of(), ids(ParquetScan.scan(empty)));
    }

    @Test
    void singleRowFile() throws IOException {
        Path one = ParquetTestData.writeFlat(dir, "one.parquet", 1, SMALL_ROW_GROUP_BYTES, true, 1024);
        assertScanEqualsRows(ParquetScan.scan(one), ParquetTestData.rows(1));
    }

    // ================================================================ schema mapping

    @Test
    void arrowSchemaMirrorsTheParquetSchema() {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(flat).open(a)) {
            Schema s = r.schema();
            assertEquals(FLAT_COLUMNS, s.getFields().stream().map(Field::getName).toList());

            assertEquals(new ArrowType.Int(64, true), s.findField("id").getType());
            assertFalse(s.findField("id").isNullable(), "REQUIRED column");
            assertTrue(s.findField("grp").isNullable(), "OPTIONAL column");
            assertEquals(new ArrowType.Int(32, true), s.findField("grp").getType());
            assertEquals(ArrowType.Utf8.INSTANCE, s.findField("name").getType());
            assertEquals(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE), s.findField("score").getType());
            assertEquals(ArrowType.Bool.INSTANCE, s.findField("flag").getType());
            assertEquals(new ArrowType.Date(DateUnit.DAY), s.findField("day").getType());
            assertEquals(new ArrowType.Timestamp(TimeUnit.MILLISECOND, "UTC"), s.findField("ts").getType());
            assertEquals(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE), s.findField("ratio").getType());
        }
    }

    // ================================================================ projection

    @Test
    void selectProjectsOnlyTheRequestedColumnsInTheRequestedOrder() {
        try (BufferAllocator a = new RootAllocator();
             ParquetBatchReader r = ParquetScan.scan(flat).select("name", "id").open(a)) {
            assertEquals(List.of("name", "id"), r.schema().getFields().stream().map(Field::getName).toList());
            assertTrue(r.next());
            assertEquals(2, r.root().getFieldVectors().size());
        }
    }

    @Test
    void selectingTheSameColumnTwiceProjectsItOnce() {
        try (BufferAllocator a = new RootAllocator();
             ParquetBatchReader r = ParquetScan.scan(flat).select("id", "id").open(a)) {
            assertEquals(1, r.schema().getFields().size());
        }
        assertEquals(allIds(), ids(ParquetScan.scan(flat).select("id", "id")));
    }

    @Test
    void selectUnknownColumnFailsNamingTheColumnAndTheAlternatives() {
        try (BufferAllocator a = new RootAllocator()) {
            ParquetScanException e = assertThrows(ParquetScanException.class,
                    () -> ParquetScan.scan(flat).select("id", "nope").open(a));
            assertEquals("nope", e.column());
            assertTrue(e.getMessage().contains("available"), e.getMessage());
            assertTrue(e.getMessage().contains("grp"), "lists the real columns: " + e.getMessage());
        }
    }

    @Test
    void projectionDecodesFewerBytesThanAFullScan() {
        long full = decodedBytes(ParquetScan.scan(flat).withMetrics(true));
        long one = decodedBytes(ParquetScan.scan(flat).select("id").withMetrics(true));
        assertTrue(one > 0);
        assertTrue(one < full, "projected " + one + " should be below full " + full);
    }

    private static long decodedBytes(ParquetScan scan) {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = scan.open(a)) {
            while (r.next()) { /* drain */ }
            return r.metrics().uncompressedBytesDecoded();
        }
    }

    // ================================================================ batching

    @Test
    void batchSizeBoundsEveryBatchAndOrderIsPreserved() {
        List<Long> ids = new ArrayList<>();
        int batches = 0;
        try (BufferAllocator a = new RootAllocator();
             ParquetBatchReader r = ParquetScan.scan(flat).select("id").batchSize(100).open(a)) {
            while (r.next()) {
                batches++;
                VectorSchemaRoot b = r.root();
                assertTrue(b.getRowCount() >= 1 && b.getRowCount() <= 100, "rows=" + b.getRowCount());
                BigIntVector v = (BigIntVector) b.getVector("id");
                for (int i = 0; i < b.getRowCount(); i++) ids.add(v.get(i));
            }
        }
        assertEquals(allIds(), ids);
        assertTrue(batches >= N / 100, "batches=" + batches);
    }

    @Test
    void defaultBatchSizeConstantIsTheDocumentedOne() {
        assertEquals(32_768, ParquetScan.DEFAULT_BATCH_SIZE);
        assertEquals(1, ParquetScan.DEFAULT_PARALLELISM);
    }

    @Test
    void forEachBatchVisitsEveryRowAndClosesTheReader() {
        AtomicInteger rows = new AtomicInteger();
        try (BufferAllocator a = new RootAllocator()) {
            ParquetScan.scan(flat).select("id").forEachBatch(a, b -> rows.addAndGet(b.getRowCount()));
            assertEquals(0L, a.getAllocatedMemory(), "forEachBatch must release everything it allocated");
        }
        assertEquals(N, rows.get());
    }

    // ================================================================ parallelism

    @Test
    void parallelismLargerThanTheRowGroupCountStillWorks() {
        assertEquals(allIds(), ids(ParquetScan.scan(flat).select("id").parallelism(64)));
    }

    @Test
    void parallelAndSequentialAgreeUnderPushdown() {
        ParquetScan base = ParquetScan.scan(flat).select("id").pushdown(Predicate.ge("id", 2_500L));
        assertEquals(ids(base), ids(base.parallelism(3)));
    }

    // ================================================================ pruning (pushdown only)

    @Test
    void pushdownPrunesRowGroupsButNeverDropsAMatchingRow() {
        try (BufferAllocator a = new RootAllocator();
             ParquetBatchReader r = ParquetScan.scan(flat).select("id").pushdown(Predicate.gt("id", 4_000L))
                     .withMetrics(true).open(a)) {
            List<Long> got = new ArrayList<>();
            while (r.next()) {
                BigIntVector v = (BigIntVector) r.root().getVector("id");
                for (int i = 0; i < r.root().getRowCount(); i++) got.add(v.get(i));
            }
            ScanMetrics m = r.metrics();
            assertTrue(m.rowGroupsSkipped() > 0, "sorted ids: the early row groups cannot contain id > 4000");
            assertEquals(m.rowGroupsAfterPruning(), m.rowGroupsRead());
            assertTrue(got.containsAll(expectedIds(row -> row.id() > 4_000L)), "pruning must yield a superset");
            assertTrue(got.size() < N, "and it must actually have pruned something");
        }
    }

    @Test
    void pushdownThatMatchesNothingSkipsEveryRowGroup() {
        try (BufferAllocator a = new RootAllocator();
             ParquetBatchReader r = ParquetScan.scan(flat).select("id").pushdown(Predicate.gt("id", 1_000_000L))
                     .withMetrics(true).open(a)) {
            assertFalse(r.next());
            assertEquals(r.metrics().rowGroupsInFile(), r.metrics().rowGroupsSkipped());
            assertEquals(0, r.metrics().rowsRead());
        }
    }

    @Test
    void pushdownOnlyIsInertForPredicatesItCannotPush() {
        assertEquals(allIds(), ids(ParquetScan.scan(flat).select("id", "name").pushdown(Predicate.like("name", "alpha%"))));
        assertEquals(allIds(), ids(ParquetScan.scan(flat).select("id").pushdown(Predicate.custom("unregistered"))));
        assertEquals(allIds(), ids(ParquetScan.scan(flat).select("id", "lo", "hi").pushdown(Predicate.colLt("lo", "hi"))));
    }

    @Test
    void pushdownWithMistypedLiteralPrunesNothingInsteadOfFailing() {
        assertEquals(allIds(), ids(ParquetScan.scan(flat).select("id").pushdown(Predicate.eq("id", "not-a-number"))));
    }

    @Test
    void laterPushdownReplacesTheEarlierOne() {
        ParquetScan s = ParquetScan.scan(flat).select("id").pushdown(Predicate.gt("id", 1_000_000L)).pushdown(Predicate.ge("id", 0L));
        assertEquals(N, ids(s.exactFilter()).size());
    }

    // ================================================================ metrics

    @Test
    void metricsAreAbsentUnlessRequested() {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(flat).open(a)) {
            assertNull(r.metrics());
        }
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(flat).withMetrics(true).withMetrics(false).open(a)) {
            assertNull(r.metrics());
        }
    }

    @Test
    void metricsDescribeAFullSequentialScan() {
        ParquetFileInfo info = ParquetFileInfo.read(flat);
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(flat).batchSize(500).withMetrics(true).open(a)) {
            int batches = 0;
            while (r.next()) batches++;
            ScanMetrics m = r.metrics();
            assertNotNull(m);
            assertEquals(N, m.rowsRead());
            assertEquals(batches, m.batches());
            assertEquals(info.rowGroups().size(), m.rowGroupsInFile());
            assertEquals(0, m.rowGroupsSkipped());
            assertEquals(info.rowGroups().size(), m.rowGroupsRead());
            assertEquals(1, m.parallelism());
            assertTrue(m.decodeNanos() > 0);
            assertTrue(m.compressedBytesRead() > 0);
            assertTrue(m.uncompressedBytesDecoded() > 0);
            assertTrue(m.arrowBytesProduced() > 0);
        }
    }

    @Test
    void metricsReportTheParallelismUsed() {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(flat).parallelism(3).withMetrics(true).open(a)) {
            while (r.next()) { /* drain */ }
            assertEquals(3, r.metrics().parallelism());
            assertEquals(N, r.metrics().rowsRead());
        }
    }

    // ================================================================ detach / ownership

    @Test
    void detachHandsOverTheBatchAndTheReaderContinuesFromTheNextOne() {
        try (BufferAllocator alloc = new RootAllocator()) {
            VectorSchemaRoot detached;
            int firstRows;
            try (ParquetBatchReader r = ParquetScan.scan(flat).select("id").batchSize(1_000).open(alloc)) {
                assertTrue(r.next());
                detached = r.detach();
                firstRows = detached.getRowCount();
                assertTrue(firstRows > 0);
                assertTrue(r.next(), "reader continues after a detach");
                BigIntVector next = (BigIntVector) r.root().getVector("id");
                assertEquals(firstRows, next.get(0), "the next batch starts where the detached one ended");
            }
            try (VectorSchemaRoot d = detached) {
                BigIntVector v = (BigIntVector) d.getVector("id");
                assertEquals(firstRows, d.getRowCount());
                for (int i = 0; i < firstRows; i++) assertEquals(i, v.get(i), "detached data survives reader close");
            }
            assertEquals(0L, alloc.getAllocatedMemory());
        }
    }

    @Test
    void closingTheReaderReleasesAllArrowMemory() {
        for (int parallelism : new int[]{1, 3}) {
            try (BufferAllocator alloc = new RootAllocator()) {
                ParquetBatchReader r = ParquetScan.scan(flat).parallelism(parallelism).open(alloc);
                while (r.next()) { /* drain */ }
                r.close();
                assertEquals(0L, alloc.getAllocatedMemory(), "parallelism=" + parallelism);
            }
        }
    }

    @Test
    void closingMidScanReleasesAllArrowMemory() {
        try (BufferAllocator alloc = new RootAllocator()) {
            ParquetBatchReader r = ParquetScan.scan(flat).batchSize(50).open(alloc);
            assertTrue(r.next());
            r.close();
            assertEquals(0L, alloc.getAllocatedMemory());
        }
    }

    @Test
    void closedReaderRejectsUseAndCloseIsIdempotent() {
        try (BufferAllocator alloc = new RootAllocator()) {
            ParquetBatchReader r = ParquetScan.scan(flat).open(alloc);
            r.close();
            r.close();
            assertThrows(IllegalStateException.class, r::next);
            assertThrows(IllegalStateException.class, r::root);
            assertThrows(IllegalStateException.class, r::detach);
        }
    }

    @Test
    void endOfDataIsStable() {
        try (BufferAllocator alloc = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(flat).select("id").open(alloc)) {
            while (r.next()) { /* drain */ }
            assertFalse(r.next());
            assertFalse(r.next());
        }
    }

    // ================================================================ guards & failures

    @Test
    void maxRowGroupBytesRejectsOversizedRowGroups() {
        try (BufferAllocator alloc = new RootAllocator()) {
            ParquetScanException e = assertThrows(ParquetScanException.class, () -> {
                try (ParquetBatchReader r = ParquetScan.scan(flat).maxRowGroupBytes(1).open(alloc)) {
                    r.next();
                }
            });
            assertTrue(e.getMessage().contains("above the configured limit"), e.getMessage());
            assertEquals(flat, e.file());
        }
    }

    @Test
    void generousMaxRowGroupBytesIsHarmless() {
        assertEquals(N, ids(ParquetScan.scan(flat).select("id").maxRowGroupBytes(Long.MAX_VALUE)).size());
    }

    @Test
    void missingFileFailsWithContext() {
        Path missing = dir.resolve("does-not-exist.parquet");
        try (BufferAllocator alloc = new RootAllocator()) {
            ParquetScanException e = assertThrows(ParquetScanException.class, () -> ParquetScan.scan(missing).open(alloc));
            assertEquals(missing, e.file());
            assertNotNull(e.getCause());
        }
    }

    @Test
    void nonParquetFileFailsWithContext() throws IOException {
        Path junk = dir.resolve("junk.parquet");
        Files.writeString(junk, "this is definitely not parquet, just some text padding padding padding");
        try (BufferAllocator alloc = new RootAllocator()) {
            ParquetScanException e = assertThrows(ParquetScanException.class, () -> ParquetScan.scan(junk).open(alloc));
            assertEquals(junk, e.file());
        }
    }

    @Test
    void truncatedFileFailsWithContext() throws IOException {
        byte[] all = Files.readAllBytes(flat);
        Path cut = dir.resolve("truncated.parquet");
        Files.write(cut, java.util.Arrays.copyOf(all, all.length / 2));
        try (BufferAllocator alloc = new RootAllocator()) {
            assertThrows(ParquetScanException.class, () -> ParquetScan.scan(cut).open(alloc));
        }
    }

    @Test
    void nullAllocatorIsRejected() {
        assertThrows(NullPointerException.class, () -> ParquetScan.scan(flat).open(null));
    }

    // ================================================================ builder contract

    @Test
    void builderRejectsInvalidArguments() {
        assertThrows(NullPointerException.class, () -> ParquetScan.scan(null));
        ParquetScan s = ParquetScan.scan(flat);
        assertThrows(IllegalArgumentException.class, s::select);
        assertThrows(IllegalArgumentException.class, () -> s.select((String[]) null));
        assertThrows(IllegalArgumentException.class, () -> s.batchSize(0));
        assertThrows(IllegalArgumentException.class, () -> s.batchSize(-1));
        assertThrows(IllegalArgumentException.class, () -> s.parallelism(0));
        assertThrows(IllegalArgumentException.class, () -> s.maxRowGroupBytes(0));
        assertThrows(IllegalArgumentException.class, () -> s.withCustomPredicate(null, (b, i) -> true));
        assertThrows(IllegalArgumentException.class, () -> s.withCustomPredicate("", (b, i) -> true));
        assertThrows(NullPointerException.class, () -> s.withCustomPredicate("x", null));
    }

    @Test
    void exactFilterWithoutAPredicateIsAnIllegalState() {
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> ParquetScan.scan(flat).exactFilter());
        assertTrue(e.getMessage().contains("pushdown"), e.getMessage());
    }

    @Test
    void scansAreImmutableSoDerivedScansNeverAffectTheirParent() {
        ParquetScan base = ParquetScan.scan(flat);
        ParquetScan projected = base.select("id").batchSize(10).parallelism(2).pushdown(Predicate.gt("id", 10L));
        assertNotSame(base, projected);
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = base.open(a)) {
            assertEquals(FLAT_COLUMNS.size(), r.schema().getFields().size(), "base scan must still project every column");
            assertTrue(r.next());
            assertTrue(r.root().getRowCount() > 10);
        }
    }
}