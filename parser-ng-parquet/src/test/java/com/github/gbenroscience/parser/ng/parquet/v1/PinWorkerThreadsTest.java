package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static com.github.gbenroscience.parser.ng.parquet.v1.ParquetTestData.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * {@code pinWorkerThreads} is opt-in and best-effort: it must be off by default, never change results, never
 * touch the caller's thread, and report honestly how many workers it managed to pin.
 */
class PinWorkerThreadsTest {

    @TempDir
    static Path dir;
    static Path flat;

    @BeforeAll
    static void write() throws IOException {
        flat = ParquetTestData.writeFlat(dir, "flat.parquet");
    }

    private static long pinnedAfterDraining(ParquetScan scan) {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = scan.withMetrics(true).open(a)) {
            while (r.next()) { /* drain */ }
            return r.metrics().workersPinned();
        }
    }

    @Test
    void offByDefault() {
        assertFalse(ParquetScan.scan(flat).pinsWorkerThreads());
        assertEquals(0, pinnedAfterDraining(ParquetScan.scan(flat).parallelism(3)));
    }

    @Test
    void theFlagSurvivesEveryOtherBuilderCall() {
        ParquetScan s = ParquetScan.scan(flat).pinWorkerThreads(true);
        assertTrue(s.pinsWorkerThreads());
        assertTrue(s.select("id").pinsWorkerThreads());
        assertTrue(s.batchSize(100).pinsWorkerThreads());
        assertTrue(s.parallelism(2).pinsWorkerThreads());
        assertTrue(s.withMetrics(true).pinsWorkerThreads());
        assertTrue(s.maxRowGroupBytes(1L << 20).pinsWorkerThreads());
        assertTrue(s.pushdown(Predicate.gt("id", 1L)).pinsWorkerThreads());
        assertTrue(s.pushdown(Predicate.gt("id", 1L)).exactFilter().pinsWorkerThreads());
        assertTrue(s.withCustomPredicate("x", (batch, row) -> true).pinsWorkerThreads());
        assertFalse(s.pinWorkerThreads(false).pinsWorkerThreads());
    }

    @Test
    void pinnedParallelScanReturnsExactlyTheSameRows() {
        assertScanEqualsRows(ParquetScan.scan(flat).parallelism(3).pinWorkerThreads(true), ROWS);
    }

    @Test
    void pinnedParallelExactFilterMatchesUnpinned() {
        Predicate p = Predicate.and(Predicate.ge("id", 1_000L), Predicate.lt("grp", 5));
        assertEquals(
                ids(ParquetScan.scan(flat).select("id", "grp").pushdown(p).exactFilter().parallelism(3)),
                ids(ParquetScan.scan(flat).select("id", "grp").pushdown(p).exactFilter().parallelism(3).pinWorkerThreads(true)));
    }

    @Test
    void neverReportsMoreWorkersPinnedThanWorkersExist() {
        long pinned = pinnedAfterDraining(ParquetScan.scan(flat).parallelism(3).pinWorkerThreads(true));
        assertTrue(pinned >= 0 && pinned <= 3, "pinned=" + pinned);
    }

    @Test
    void sequentialScanIgnoresTheFlagAndNeverPinsTheCallersThread() {
        assertEquals(0, pinnedAfterDraining(ParquetScan.scan(flat).parallelism(1).pinWorkerThreads(true)));
        assertScanEqualsRows(ParquetScan.scan(flat).pinWorkerThreads(true), ROWS);
    }
}
