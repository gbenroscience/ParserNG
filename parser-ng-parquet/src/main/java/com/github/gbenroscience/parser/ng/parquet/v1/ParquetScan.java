package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

/**
 * Immutable scan specification for one Parquet file: projection + pruning predicate + batch size +
 * parallelism. SQL-agnostic: a planner (e.g. parser-ng-sql-parquet) builds one of these from its own plan.
 *
 * <pre>{@code
 * try (BufferAllocator alloc = new RootAllocator();
 *      ParquetBatchReader r = ParquetScan.scan(path)
 *              .select("x", "y")
 *              .pushdown(Predicate.gt("x", 100))   // pruning hint: result is a SUPERSET
 *              .parallelism(4)                     // decode up to 4 row groups concurrently
 *              .open(alloc)) {
 *     while (r.next()) {
 *         VectorSchemaRoot batch = r.root();       // valid until the next next()/close()
 *         // ... hand to parser-ng-arrow, which applies the exact predicate
 *     }
 * }
 * }</pre>
 *
 * <p>The spec deliberately calls the predicate method {@code pushdown}, not {@code filter}: it
 * prunes, it does not filter. See {@link Predicate}.
 *
 * <p>Reads only — this module no longer includes a writer.
 */
public final class ParquetScan {

    public static final int DEFAULT_BATCH_SIZE = 32_768;
    public static final long DEFAULT_MAX_ROW_GROUP_BYTES = 2L << 30; // 2 GiB uncompressed, untrusted-input guard
    /** Sequential by default: predictable, zero-steady-state-allocation. Opt into parallel with {@link #parallelism}. */
    public static final int DEFAULT_PARALLELISM = 1;

    private final Path file;
    private final List<String> columns; // null = all
    private final Predicate predicate;  // null = none
    private final int batchSize;
    private final boolean metrics;
    private final long maxRowGroupBytes;
    private final int parallelism;

    private ParquetScan(Path file, List<String> columns, Predicate predicate, int batchSize,
                        boolean metrics, long maxRowGroupBytes, int parallelism) {
        this.file = file;
        this.columns = columns;
        this.predicate = predicate;
        this.batchSize = batchSize;
        this.metrics = metrics;
        this.maxRowGroupBytes = maxRowGroupBytes;
        this.parallelism = parallelism;
    }

    public static ParquetScan scan(Path file) {
        if (file == null) throw new NullPointerException("file");
        return new ParquetScan(file, null, null, DEFAULT_BATCH_SIZE, false, DEFAULT_MAX_ROW_GROUP_BYTES, DEFAULT_PARALLELISM);
    }

    /** Top-level columns to read. Unlisted columns are never decoded. Must be non-empty. */
    public ParquetScan select(String... cols) {
        if (cols == null || cols.length == 0) throw new IllegalArgumentException("select() needs at least one column");
        return new ParquetScan(file, List.copyOf(Arrays.asList(cols)), predicate, batchSize, metrics, maxRowGroupBytes, parallelism);
    }

    /** Pruning predicate (superset semantics). Replaces any earlier one; combine with {@link Predicate#and}. */
    public ParquetScan pushdown(Predicate p) {
        return new ParquetScan(file, columns, p, batchSize, metrics, maxRowGroupBytes, parallelism);
    }

    public ParquetScan batchSize(int rows) {
        if (rows <= 0) throw new IllegalArgumentException("batchSize must be positive");
        return new ParquetScan(file, columns, predicate, rows, metrics, maxRowGroupBytes, parallelism);
    }

    public ParquetScan withMetrics(boolean enabled) {
        return new ParquetScan(file, columns, predicate, batchSize, enabled, maxRowGroupBytes, parallelism);
    }

    /** Rejects any surviving row group whose declared uncompressed size exceeds this. */
    public ParquetScan maxRowGroupBytes(long bytes) {
        if (bytes <= 0) throw new IllegalArgumentException("maxRowGroupBytes must be positive");
        return new ParquetScan(file, columns, predicate, batchSize, metrics, bytes, parallelism);
    }

    /**
     * Number of row groups decoded concurrently. {@code 1} (the default) is fully sequential and reuses
     * one set of Arrow buffers for every batch. A value above 1 uses that many decoder threads/file
     * handles and delivers batches in the SAME order a sequential scan would, at the cost of allocating a
     * fresh batch per row group instead of reusing buffers. Choosing a value larger than the file's row
     * group count wastes threads; the caller/planner is expected to pick a sensible number (e.g.
     * {@code min(desiredParallelism, availableProcessors())}) — this class applies no adaptive logic.
     */
    public ParquetScan parallelism(int rowGroupsAtOnce) {
        if (rowGroupsAtOnce <= 0) throw new IllegalArgumentException("parallelism must be positive");
        return new ParquetScan(file, columns, predicate, batchSize, metrics, maxRowGroupBytes, rowGroupsAtOnce);
    }

    /**
     * Opens the scan. The caller owns {@code allocator} (it must outlive the reader and any detached
     * roots); the returned reader owns its file handle(s) and must be closed.
     */
    public ParquetBatchReader open(BufferAllocator allocator) {
        return new ParquetBatchReader(file, columns, predicate, batchSize, metrics, maxRowGroupBytes, parallelism, allocator);
    }

    /** Convenience loop; each batch is valid only for the duration of the callback. */
    public void forEachBatch(BufferAllocator allocator, Consumer<VectorSchemaRoot> consumer) {
        try (ParquetBatchReader r = open(allocator)) {
            while (r.next()) consumer.accept(r.root());
        }
    }
}
