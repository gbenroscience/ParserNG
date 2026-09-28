package com.github.gbenroscience.parser.ng.parquet.v1;

import java.util.concurrent.atomic.LongAdder;

/**
 * Optional counters for one scan. Safe to update from decoder threads (LongAdder); read them after (or
 * between) {@code next()} calls. When metrics are disabled the reader holds no instance and pays only a
 * null check per batch.
 *
 * <h2>Byte counters: three different, honestly-defined quantities</h2>
 * "MB/s" is only meaningful once you say <em>which bytes</em>. Earlier revisions exposed none, so
 * benchmarks fell back to on-disk file size divided by time -- which over-states projected and
 * predicate-pruned scans (bytes never touched were still in the numerator). These three are measured
 * inside the decode path and each answers a different question:
 * <ul>
 *   <li>{@link #uncompressedBytesDecoded()} -- <b>the headline decode-throughput numerator.</b> Bytes of
 *       Parquet page payload (dictionary pages + data pages, including repetition/definition level
 *       streams) actually handed to the decode engine, after decompression. Counted when a page is loaded,
 *       so projection and page-level (column-index) pruning are respected exactly: a column that is not
 *       selected, or a page that is skipped, contributes nothing.</li>
 *   <li>{@link #compressedBytesRead()} -- on-disk (compressed, as-stored) size of the projected column
 *       chunks of every row group that was decoded, taken from the footer. This is what the storage layer had
 *       to deliver. It is an <em>upper bound</em> when page-level pruning skips pages inside a surviving row
 *       group (the footer only knows whole column chunks). Divide by wall time for "I/O + decompress +
 *       decode" throughput; compare against {@code uncompressedBytesDecoded} for the effective
 *       compression ratio.</li>
 *   <li>{@link #arrowBytesProduced()} -- bytes of Arrow buffers materialized (sum of
 *       {@code FieldVector#getBufferSize()} per batch, measured outside the {@link #decodeNanos()} window).
 *       This is the output-side rate a downstream consumer sees.</li>
 * </ul>
 * {@code rowsRead}/{@code batches} are counted per emitted decode batch, not per pruned row.
 *
 * <p>Not reported (parquet-java does not expose them cheaply): page counts and separate I/O time.
 * {@code decodeNanos} is summed over all decoder threads (so it can exceed wall time when parallel) and
 * covers decode + Arrow materialization together because they are one fused loop. It is <em>not</em> a
 * clean kernel timer: pages are decompressed lazily inside parquet-java's {@code PageReader.readPage()},
 * which runs inside that loop for every page except each column's first (loaded eagerly in
 * {@code RowGroupDecoder#begin}, outside the timer), and the file read itself is outside it entirely.
 * So do not divide byte counters by {@code decodeNanos} and call the result throughput; divide by wall
 * clock around the scan, which is what both bundled benchmarks do.
 */
public final class ScanMetrics {
    long rowGroupsInFile, rowGroupsAfterPruning;
    int parallelism = 1;
    final LongAdder rowGroupsRead = new LongAdder();
    final LongAdder rowsRead = new LongAdder();
    final LongAdder batches = new LongAdder();
    final LongAdder decodeNanos = new LongAdder();
    final LongAdder compressedBytesRead = new LongAdder();
    final LongAdder uncompressedBytesDecoded = new LongAdder();
    final LongAdder arrowBytesProduced = new LongAdder();

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

    /** On-disk (compressed) bytes of the projected column chunks of every row group read. Upper bound under page pruning. */
    public long compressedBytesRead() { return compressedBytesRead.sum(); }
    /** Decompressed page bytes (dictionary + data pages incl. levels) actually loaded by the decode engine. See class docs. */
    public long uncompressedBytesDecoded() { return uncompressedBytesDecoded.sum(); }
    /** Arrow buffer bytes materialized across all emitted batches. */
    public long arrowBytesProduced() { return arrowBytesProduced.sum(); }

    @Override
    public String toString() {
        return "ScanMetrics{rowGroupsInFile=" + rowGroupsInFile + ", skipped=" + rowGroupsSkipped()
                + ", read=" + rowGroupsRead() + ", rowsRead=" + rowsRead() + ", batches=" + batches()
                + ", parallelism=" + parallelism + ", decodeMs(sum)=" + decodeNanos() / 1_000_000
                + ", compressedMB=" + mb(compressedBytesRead()) + ", decodedMB=" + mb(uncompressedBytesDecoded())
                + ", arrowMB=" + mb(arrowBytesProduced()) + '}';
    }

    private static String mb(long bytes) {
        return String.format(java.util.Locale.ROOT, "%.1f", bytes / (1024.0 * 1024.0));
    }
}