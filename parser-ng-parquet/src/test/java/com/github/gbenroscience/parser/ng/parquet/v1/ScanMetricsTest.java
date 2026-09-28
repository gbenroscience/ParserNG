
package com.github.gbenroscience.parser.ng.parquet.v1;

import org.junit.jupiter.api.Test;

import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

/** ScanMetrics' counters are package-private so the decode path can update them; tests live in the same package. */
class ScanMetricsTest {

    @Test
    void freshInstanceIsAllZeroAndSequential() {
        ScanMetrics m = new ScanMetrics();
        assertEquals(0, m.rowGroupsInFile());
        assertEquals(0, m.rowGroupsAfterPruning());
        assertEquals(0, m.rowGroupsSkipped());
        assertEquals(0, m.rowGroupsRead());
        assertEquals(0, m.rowsRead());
        assertEquals(0, m.batches());
        assertEquals(0, m.decodeNanos());
        assertEquals(0, m.compressedBytesRead());
        assertEquals(0, m.uncompressedBytesDecoded());
        assertEquals(0, m.arrowBytesProduced());
        assertEquals(1, m.parallelism());
    }

    @Test
    void skippedIsFileMinusSurvivors() {
        ScanMetrics m = new ScanMetrics();
        m.rowGroupsInFile = 10;
        m.rowGroupsAfterPruning = 4;
        assertEquals(6, m.rowGroupsSkipped());
    }

    @Test
    void countersAccumulate() {
        ScanMetrics m = new ScanMetrics();
        m.rowsRead.add(100);
        m.rowsRead.add(23);
        m.batches.increment();
        m.rowGroupsRead.increment();
        m.decodeNanos.add(5_000_000L);
        m.compressedBytesRead.add(1024);
        m.uncompressedBytesDecoded.add(4096);
        m.arrowBytesProduced.add(8192);
        assertEquals(123, m.rowsRead());
        assertEquals(1, m.batches());
        assertEquals(1, m.rowGroupsRead());
        assertEquals(5_000_000L, m.decodeNanos());
        assertEquals(1024, m.compressedBytesRead());
        assertEquals(4096, m.uncompressedBytesDecoded());
        assertEquals(8192, m.arrowBytesProduced());
    }

    @Test
    void countersAreSafeUnderConcurrentUpdates() throws Exception {
        ScanMetrics m = new ScanMetrics();
        Thread[] ts = new Thread[8];
        for (int i = 0; i < ts.length; i++) {
            ts[i] = new Thread(() -> { for (int k = 0; k < 10_000; k++) m.rowsRead.increment(); });
            ts[i].start();
        }
        for (Thread t : ts) t.join();
        assertEquals(80_000, m.rowsRead());
    }

    @Test
    void toStringIsLocaleIndependentAndListsTheHeadlineFields() {
        Locale old = Locale.getDefault();
        try {
            Locale.setDefault(Locale.GERMANY); // a default-locale format would print "1,0"
            ScanMetrics m = new ScanMetrics();
            m.rowGroupsInFile = 5;
            m.rowGroupsAfterPruning = 3;
            m.rowsRead.add(42);
            m.uncompressedBytesDecoded.add(1024 * 1024);
            String s = m.toString();
            assertTrue(s.startsWith("ScanMetrics{"), s);
            assertTrue(s.contains("rowGroupsInFile=5"), s);
            assertTrue(s.contains("skipped=2"), s);
            assertTrue(s.contains("rowsRead=42"), s);
            assertTrue(s.contains("decodedMB=1.0"), s);
        } finally {
            Locale.setDefault(old);
        }
    }
}