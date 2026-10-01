package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Pinning is best-effort: these tests pin down the ordering rule, the bounds, and what must never throw. */
class CpuPinnerTest {

    @Test
    void interleavePutsOneCpuPerPhysicalCoreFirst() {
        assertArrayEquals(new int[] {0, 2, 1, 3}, CpuPinner.interleave(new int[][] {{0, 1}, {2, 3}}));
        assertArrayEquals(new int[] {0, 3, 1, 2}, CpuPinner.interleave(new int[][] {{0, 1, 2}, {3}}));
        assertArrayEquals(new int[] {5, 7, 6}, CpuPinner.interleave(new int[][] {{5, 6}, {7}}));
        assertArrayEquals(new int[] {0, 1, 2}, CpuPinner.interleave(new int[][] {{0}, {1}, {2}}));
        assertArrayEquals(new int[0], CpuPinner.interleave(new int[0][]));
    }

    @Test
    void detectedTopologyIsNonEmptyWithNoDuplicateOrNegativeCpus() {
        int[][] groups = CpuPinner.detectPhysicalCoreGroups();
        assertTrue(groups.length > 0);
        Set<Integer> seen = new HashSet<>();
        for (int[] g : groups) {
            assertTrue(g.length > 0, "an empty core group");
            for (int cpu : g) {
                assertTrue(cpu >= 0, "negative cpu " + cpu);
                assertTrue(seen.add(cpu), "cpu " + cpu + " listed twice");
            }
        }
    }

    @Test
    void pinOrderIsAPermutationOfEveryDetectedCpuAndLeadsWithDistinctCores() {
        int[][] groups = CpuPinner.detectPhysicalCoreGroups();
        int[] order = CpuPinner.pinOrder();
        int total = 0;
        for (int[] g : groups) total += g.length;
        assertEquals(total, order.length);
        assertEquals(total, java.util.Arrays.stream(order).distinct().count());

        Set<Integer> firstOfEachCore = new HashSet<>();
        for (int[] g : groups) firstOfEachCore.add(g[0]);
        for (int i = 0; i < groups.length; i++) {
            assertTrue(firstOfEachCore.contains(order[i]),
                    "position " + i + " should be the first CPU of some physical core, was " + order[i]);
        }
    }

    @Test
    void outOfRangeIndicesAreRefusedRatherThanWrappingOntoAnotherCpu() {
        assertFalse(CpuPinner.pinCurrentThread(-1));
        assertFalse(CpuPinner.pinCurrentThread(1024)); // beyond a glibc cpu_set_t, and beyond a Windows group
        assertFalse(CpuPinner.pinCurrentThread(64) && System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win"),
                "Windows group-0 masks hold 64 CPUs; 1L << 64 would wrap onto CPU 0");
    }

    /** Runs on its own thread so a successful pin never leaks onto the JUnit worker. */
    @Test
    void pinningTheCurrentThreadNeverThrowsAndOnLinuxTakesEffect() throws Exception {
        int cpu = CpuPinner.pinOrder()[0];
        AtomicReference<Boolean> pinned = new AtomicReference<>();
        AtomicReference<String> allowed = new AtomicReference<>();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread t = new Thread(() -> {
            try {
                pinned.set(CpuPinner.pinCurrentThread(cpu));
                allowed.set(allowedCpusOfThisThread());
            } catch (Throwable e) {
                failure.set(e);
            }
        });
        t.start();
        t.join();
        assertNull(failure.get(), () -> "pinning must never throw: " + failure.get());
        assertNotNull(pinned.get());

        // If the OS accepted the request on Linux, the kernel must now list exactly that one CPU.
        if (pinned.get() && allowed.get() != null) {
            assertEquals(String.valueOf(cpu), allowed.get());
        }
    }

    @Test
    void unsupportedPlatformsReportFalseNotAnException() {
        assumeTrue(!CpuPinner.isSupported(), "only meaningful where no pinning hook could be bound");
        assertFalse(CpuPinner.pinCurrentThread(0));
    }

    /** {@code Cpus_allowed_list} of the calling thread from procfs, or null when not on Linux / unreadable. */
    private static String allowedCpusOfThisThread() {
        Path status = Path.of("/proc/thread-self/status");
        if (!Files.isReadable(status)) return null;
        try {
            for (String line : Files.readAllLines(status)) {
                if (line.startsWith("Cpus_allowed_list:")) return line.substring("Cpus_allowed_list:".length()).trim();
            }
        } catch (IOException e) {
            return null;
        }
        return null;
    }
}
