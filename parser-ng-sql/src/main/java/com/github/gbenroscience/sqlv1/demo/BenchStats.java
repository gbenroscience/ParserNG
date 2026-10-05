package com.github.gbenroscience.sqlv1.demo;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.List;

/**
 * Timing, allocation and GC bookkeeping shared by the benchmarks in this package.
 *
 * <p>Ideas borrowed from the other benchmarks: reps are repeated and the <b>median</b> is reported with min and
 * max ({@link ExecBench}); the work is calibrated so a rep is long enough to time and the result is always consumed
 * so the JIT cannot drop it ({@link Benchmark}); fixtures are written before any timing and the page cache is warm
 * (the parser-ng-parquet fair comparison). Added here: allocation per call (bytes allocated by the calling thread)
 * and GC time, since the goal is "as zero-alloc as possible".
 */
final class BenchStats {

    private BenchStats() { }

    /** One timed piece of work; returns a number that depends on the work (the answer or a checksum). */
    @FunctionalInterface
    interface Work {
        double run() throws Exception;
    }

    /** The summary of one measured work item. */
    static final class Result {
        final double medianMs, minMs, maxMs;
        /** Bytes allocated by the calling thread per call (worker threads of a parallel scan are not included). */
        final long allocBytesPerCall;
        final double gcMsPerCall;
        final double answer;
        final int callsPerRep;

        Result(double medianMs, double minMs, double maxMs, long allocBytesPerCall, double gcMsPerCall, double answer, int callsPerRep) {
            this.medianMs = medianMs;
            this.minMs = minMs;
            this.maxMs = maxMs;
            this.allocBytesPerCall = allocBytesPerCall;
            this.gcMsPerCall = gcMsPerCall;
            this.answer = answer;
            this.callsPerRep = callsPerRep;
        }
    }

    /** Accumulates a touch of every answer so a call's result can never be proven unused. */
    private static volatile double sink;

    static double sink() {
        return sink;
    }

    /**
     * Warms up, calibrates how many calls make a rep of at least {@code minRepMs}, then times {@code reps} reps and
     * reports the median per call.
     *
     * @param warmup  untimed calls first (JIT, class loading, caches)
     * @param reps    timed reps, an odd number gives a true median
     * @param minRepMs a rep runs enough calls to take at least this long (microsecond workloads run many)
     */
    static Result measure(Work w, int warmup, int reps, double minRepMs) throws Exception {
        double answer = 0;
        for (int i = 0; i < warmup; i++) {
            answer = w.run();
            sink += answer;
        }
        // calibrate
        long t0 = System.nanoTime();
        answer = w.run();
        sink += answer;
        double oneMs = Math.max((System.nanoTime() - t0) / 1e6, 1e-3);
        int calls = (int) Math.max(1, Math.min(100_000, Math.ceil(minRepMs / oneMs)));

        double[] ms = new double[reps];
        long allocBefore = allocatedBytes();
        long gcBefore = gcMillis();
        long calledTotal = 0;
        for (int r = 0; r < reps; r++) {
            long s = System.nanoTime();
            for (int i = 0; i < calls; i++) {
                answer = w.run();
                sink += answer;
            }
            ms[r] = (System.nanoTime() - s) / 1e6 / calls;
            calledTotal += calls;
        }
        long alloc = (allocatedBytes() - allocBefore) / Math.max(1, calledTotal);
        double gc = (gcMillis() - gcBefore) / (double) Math.max(1, calledTotal);
        double[] sorted = ms.clone();
        Arrays.sort(sorted);
        return new Result(sorted[sorted.length / 2], sorted[0], sorted[sorted.length - 1], alloc, gc, answer, calls);
    }

    /** Bytes allocated so far by the calling thread, or -1 if the JVM cannot say. */
    static long allocatedBytes() {
        try {
            java.lang.management.ThreadMXBean b = ManagementFactory.getThreadMXBean();
            if (b instanceof com.sun.management.ThreadMXBean sb && sb.isThreadAllocatedMemorySupported()) {
                return sb.getThreadAllocatedBytes(Thread.currentThread().getId());
            }
        } catch (RuntimeException | LinkageError unsupported) {
            // fall through
        }
        return -1;
    }

    /** Total GC time of all collectors so far, in milliseconds. */
    static long gcMillis() {
        long t = 0;
        List<GarbageCollectorMXBean> beans = ManagementFactory.getGarbageCollectorMXBeans();
        for (GarbageCollectorMXBean b : beans) if (b.getCollectionTime() > 0) t += b.getCollectionTime();
        return t;
    }

    static String human(long bytes) {
        if (bytes < 0) return "n/a";
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KiB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024) return String.format("%.1f MiB", bytes / (1024.0 * 1024.0));
        return String.format("%.2f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
}
