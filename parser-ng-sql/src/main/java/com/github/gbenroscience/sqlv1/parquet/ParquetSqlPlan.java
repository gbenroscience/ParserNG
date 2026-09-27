package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.*;
import com.github.gbenroscience.sqlv1.ArrowQuery;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.VectorSchemaRootAppender;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * A SQL query bound to one Parquet file: parsed and compiled once, scan spec derived once.
 * Owns the underlying {@link ArrowQuery}; close it.
 *
 * <h2>Two execution modes (chosen by the query shape, never silently)</h2>
 * <ul>
 *   <li>{@link #stream}: for {@code SELECT ... WHERE ... [LIMIT n]} with no GROUP BY / aggregates /
 *       ORDER BY. Each Parquet batch goes through ArrowQuery independently and the results are
 *       delivered one by one. Memory is bounded by the batch size. LIMIT stops the scan early.</li>
 *   <li>{@link #execute}: any query. Matching (pruned, projected) input batches are gathered into
 *       ONE root and ArrowQuery runs once over it, so ORDER BY / GROUP BY / HAVING / LIMIT / aggregates
 *       are globally correct. Memory is proportional to the surviving rows, so it is capped by
 *       {@link #maxMaterializedRows(long)}; exceeding it fails loudly instead of running out of memory.</li>
 * </ul>
 * Running a non-streamable query through {@link #stream} throws: per-batch ORDER BY / GROUP BY / LIMIT
 * would return wrong answers.
 *
 * <p>Not thread-safe. The FROM table name in the SQL is ignored; the file passed to
 * {@link ParquetSql#plan} is the data source.
 */
public final class ParquetSqlPlan implements AutoCloseable {

    public static final long DEFAULT_MAX_MATERIALIZED_ROWS = 20_000_000L;

    private final ArrowQuery query;
    private final ScanPlanner.ScanPlan plan;
    private final ParquetScan scan;
    private long maxMaterializedRows = DEFAULT_MAX_MATERIALIZED_ROWS;
    private ScanMetrics lastMetrics;

    ParquetSqlPlan(ArrowQuery query, ScanPlanner.ScanPlan plan, ParquetScan scan) {
        this.query = query;
        this.plan = plan;
        this.scan = scan;
    }

    /** The compiled query, e.g. to call {@code withBackend(...)} / {@code withNullPolicy(...)}. */
    public ArrowQuery query() { return query; }

    /** What will be asked of the Parquet layer. */
    public ScanPlanner.ScanPlan scanPlan() { return plan; }

    public ParquetSqlPlan maxMaterializedRows(long rows) {
        if (rows <= 0) throw new IllegalArgumentException("rows must be positive");
        this.maxMaterializedRows = rows;
        return this;
    }

    /** Metrics of the most recent run, or null if none has run. Row groups skipped etc. */
    public ScanMetrics lastMetrics() { return lastMetrics; }

    public String explain() {
        return "columns=" + (plan.columns() == null ? "*" : plan.columns())
                + ", pruning=" + plan.pruning()
                + ", mode=" + (plan.streamable() ? "stream" : "materialize (" + plan.materializeReason() + ")")
                + ", limit=" + plan.limit();
    }

    /**
     * Streaming execution; see class docs. Each result passed to {@code sink} is valid only during the
     * callback (it is closed afterwards); copy or transfer inside the callback if you need to keep it.
     *
     * @throws UnsupportedOperationException if the query is not streamable
     */
    public void stream(BufferAllocator allocator, Consumer<VectorSchemaRoot> sink) {
        if (!plan.streamable()) {
            throw new UnsupportedOperationException(
                    "Query cannot be streamed batch-by-batch: " + plan.materializeReason() + ". Use execute().");
        }
        long remaining = plan.limit() == null ? Long.MAX_VALUE : plan.limit();
        if (remaining == 0) return;
        try (ParquetBatchReader r = scan.open(allocator)) {
            while (remaining > 0 && r.next()) {
                VectorSchemaRoot out = query.execute(r.root());
                try {
                    int rows = out.getRowCount();
                    if (rows > remaining) {
                        out.setRowCount((int) remaining); // truncating a view of the result; nothing is copied
                        rows = (int) remaining;
                    }
                    remaining -= rows;
                    if (rows > 0) sink.accept(out);
                } finally {
                    out.close();
                }
            }
            lastMetrics = r.metrics();
        }
    }

    /**
     * Global execution; see class docs.
     *
     * @return a root the caller owns and must close
     * @throws IllegalStateException if more than {@code maxMaterializedRows} rows survive pruning
     */
    public VectorSchemaRoot execute(BufferAllocator allocator) {
        List<VectorSchemaRoot> parts = new ArrayList<>();
        Schema schema;
        try {
            long rows = 0;
            try (ParquetBatchReader r = scan.open(allocator)) {
                schema = r.schema();
                while (r.next()) {
                    rows += r.root().getRowCount();
                    if (rows > maxMaterializedRows) {
                        throw new IllegalStateException("Query needs more than " + maxMaterializedRows
                                + " rows materialized (" + explain() + "). Add a more selective WHERE, "
                                + "select fewer columns, or raise maxMaterializedRows().");
                    }
                    parts.add(r.detach()); // zero-copy ownership transfer
                }
                lastMetrics = r.metrics();
            }
            VectorSchemaRoot combined = VectorSchemaRoot.create(schema, allocator);
            try {
                if (!parts.isEmpty()) {
                    VectorSchemaRootAppender.append(combined, parts.toArray(new VectorSchemaRoot[0]));
                }
                return query.execute(combined);
            } finally {
                combined.close();
            }
        } finally {
            for (VectorSchemaRoot p : parts) p.close();
        }
    }

    @Override
    public void close() { query.close(); }
}
