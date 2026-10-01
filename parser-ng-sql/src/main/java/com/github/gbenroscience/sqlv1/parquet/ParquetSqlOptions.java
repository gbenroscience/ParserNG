package com.github.gbenroscience.sqlv1.parquet;

import java.nio.file.Path;

/**
 * Immutable scan and safety options for {@link ParquetSql}.
 *
 * <p>{@code parallelism}, {@code batchSize} and {@code maxRowGroupBytes} are
 * {@code null} by default, meaning "do not call the setter": parser-ng-parquet's
 * own defaults (sequential, its default batch size and row-group cap) apply, so
 * the two modules can never drift apart.
 *
 * @param parallelism row groups decoded concurrently, or {@code null} for the parser-ng-parquet default
 * @param batchSize rows per decoded batch, or {@code null} for the parser-ng-parquet default
 * @param maxRowGroupBytes per-row-group size guard, or {@code null} for the parser-ng-parquet default
 * @param metrics collect {@link com.github.gbenroscience.parser.ng.parquet.v1.ScanMetrics} (default {@code true})
 * @param exactFilter ask parser-ng-parquet to filter rows itself when the whole WHERE converts (default {@code false})
 * @param maxMaterializedRows cap on rows held by materializing queries (ORDER BY, aggregates)
 * @param int64Policy what to do with an int64/uint64/decimal value that a double cannot hold exactly
 * @param allowedRoot if non-null, every resolved file must live under this directory
 */
public record ParquetSqlOptions(Integer parallelism, Integer batchSize, Long maxRowGroupBytes,
        boolean metrics, boolean exactFilter, long maxMaterializedRows,
        Int64Policy int64Policy, Path allowedRoot) {

    /** Default cap on materialized rows. */
    public static final long DEFAULT_MAX_MATERIALIZED_ROWS = 20_000_000L;

    /**
     * Policy for integer/decimal values whose magnitude exceeds 2^53, the
     * limit of exact integers in a {@code double}. Such values reach the
     * numeric expression engine only through widening to {@code double}.
     */
    public enum Int64Policy {
        /** Fail with a {@link ParquetSqlException} naming column and value (default). */
        ERROR,
        /** Round to the nearest double, silently. Opt-in; results may be inexact. */
        ROUND_TO_DOUBLE
    }

    public ParquetSqlOptions {
        if (parallelism != null && parallelism < 1) throw new IllegalArgumentException("parallelism must be >= 1");
        if (batchSize != null && batchSize < 1) throw new IllegalArgumentException("batchSize must be >= 1");
        if (maxRowGroupBytes != null && maxRowGroupBytes < 1) throw new IllegalArgumentException("maxRowGroupBytes must be >= 1");
        if (maxMaterializedRows < 1) throw new IllegalArgumentException("maxMaterializedRows must be >= 1");
        if (int64Policy == null) throw new IllegalArgumentException("int64Policy must not be null");
        if (allowedRoot != null) allowedRoot = allowedRoot.toAbsolutePath().normalize();
    }

    /** @return the defaults */
    public static ParquetSqlOptions defaults() {
        return new ParquetSqlOptions(null, null, null, true, false, DEFAULT_MAX_MATERIALIZED_ROWS,
                Int64Policy.ERROR, null);
    }

    public ParquetSqlOptions withParallelism(int n) {
        return new ParquetSqlOptions(n, batchSize, maxRowGroupBytes, metrics, exactFilter, maxMaterializedRows, int64Policy, allowedRoot);
    }

    public ParquetSqlOptions withBatchSize(int rows) {
        return new ParquetSqlOptions(parallelism, rows, maxRowGroupBytes, metrics, exactFilter, maxMaterializedRows, int64Policy, allowedRoot);
    }

    public ParquetSqlOptions withMaxRowGroupBytes(long bytes) {
        return new ParquetSqlOptions(parallelism, batchSize, bytes, metrics, exactFilter, maxMaterializedRows, int64Policy, allowedRoot);
    }

    public ParquetSqlOptions withMetrics(boolean on) {
        return new ParquetSqlOptions(parallelism, batchSize, maxRowGroupBytes, on, exactFilter, maxMaterializedRows, int64Policy, allowedRoot);
    }

    public ParquetSqlOptions withExactFilter(boolean on) {
        return new ParquetSqlOptions(parallelism, batchSize, maxRowGroupBytes, metrics, on, maxMaterializedRows, int64Policy, allowedRoot);
    }

    public ParquetSqlOptions withMaxMaterializedRows(long rows) {
        return new ParquetSqlOptions(parallelism, batchSize, maxRowGroupBytes, metrics, exactFilter, rows, int64Policy, allowedRoot);
    }

    public ParquetSqlOptions withInt64Policy(Int64Policy p) {
        return new ParquetSqlOptions(parallelism, batchSize, maxRowGroupBytes, metrics, exactFilter, maxMaterializedRows, p, allowedRoot);
    }

    /** Restricts every file the SQL text can name to {@code root} and below. */
    public ParquetSqlOptions withAllowedRoot(Path root) {
        return new ParquetSqlOptions(parallelism, batchSize, maxRowGroupBytes, metrics, exactFilter, maxMaterializedRows, int64Policy, root);
    }
}
