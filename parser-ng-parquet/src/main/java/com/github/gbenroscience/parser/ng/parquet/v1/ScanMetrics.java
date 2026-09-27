package com.github.gbenroscience.parser.ng.parquet.v1;

import java.util.concurrent.atomic.LongAdder;

/**
 * Optional counters for one scan. Safe to update from decoder threads (LongAdder); read them after (or
 * between) {@code next()} calls. When metrics are disabled the reader holds no instance and pays only a
 * null check per batch.
 *
 * <p>Not reported (parquet-java does not expose them cheaply): bytes read/skipped, page counts, separate
 * I/O time. {@code decodeNanos} is summed over all decoder threads (so it can exceed wall time when
 * parallel) and covers decode + Arrow materialization together because they are one fused loop.
 */
public final class ScanMetrics {
    long rowGroupsInFile, rowGroupsAfterPruning;
    int parallelism = 1;
    final LongAdder rowGroupsRead = new LongAdder();
    final LongAdder rowsRead = new LongAdder();
    final LongAdder batches = new LongAdder();
    final LongAdder decodeNanos = new LongAdder();

    public long rowGroupsInFile() { return rowGroupsInFile; }
    public long rowGroupsAfterPruning() { return rowGroupsAfterPruning; }
    public long rowGroupsSkipped() { return rowGroupsInFile - rowGroupsAfterPruning; }
    public long rowGroupsRead() { return rowGroupsRead.sum(); }
    public long rowsRead() { return rowsRead.sum(); }
    public long batches() { return batches.sum(); }
    /** Decode time summed across all decoder threads. */
    public long decodeNanos() { return decodeNanos.sum(); }
    /** Number of decoder threads actually used (1 = sequential). */
    public int parallelism() { return parallelism; }

    @Override
    public String toString() {
        return "ScanMetrics{rowGroupsInFile=" + rowGroupsInFile + ", skipped=" + rowGroupsSkipped()
                + ", read=" + rowGroupsRead() + ", rowsRead=" + rowsRead() + ", batches=" + batches()
                + ", parallelism=" + parallelism + ", decodeMs(sum)=" + decodeNanos() / 1_000_000 + '}';
    }
}
