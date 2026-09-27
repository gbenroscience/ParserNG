package com.github.gbenroscience.parser.ng.parquet.v1;

import com.github.gbenroscience.parser.ng.parquet.v1.internal.RowFilterEvaluator;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.ArrayList;
import java.util.List;

/**
 * The mechanism behind {@code ParquetScan.exactFilter()}: wraps a {@link BatchSource} (sequential or
 * parallel, unmodified) and re-checks every row that survives row-group/page pruning against a
 * {@link RowFilterEvaluator}, emitting only true matches.
 *
 * <h2>Cost, precisely</h2>
 * For each batch pulled from the delegate: every row is evaluated (an unavoidable per-row cost —
 * this is exactly what "go-slow, exact" buys over pruning-only), and if at least one row matches, a
 * <b>freshly allocated</b> batch is built via selective copy ({@code FieldVector.copyFromSafe}, one
 * call per surviving row per column — no bulk vectorized copy path exists for a non-contiguous
 * selection). A batch with zero surviving rows is discarded and the next delegate batch is pulled
 * immediately, so the caller of {@link #next()} never sees an empty batch. This is strictly more
 * allocation than {@code SequentialSource}'s steady-state-zero-allocation reuse (see
 * {@code ParquetBatchReader}'s class Javadoc on ownership) — deliberately: exact mode already told
 * the caller it costs more, and pretending otherwise by reusing buffers across a variable-length
 * selection would be a real engineering trap (a selection's row count changes batch to batch, so
 * "reuse" would still mean reallocating whenever the previous buffer was too small, while ALSO
 * carrying the bookkeeping cost of tracking a shrinking/growing steady buffer for no actual benefit).
 *
 * <p>Not thread-safe; used only from {@code ParquetBatchReader}, which never exposes it directly.
 */
final class FilteredSource implements BatchSource {

    private final BatchSource delegate;
    private final RowFilterEvaluator evaluator;
    private final BufferAllocator allocator;
    private final Schema schema;

    private VectorSchemaRoot current;
    private boolean detached;
    private boolean delegateExhausted;

    FilteredSource(BatchSource delegate, RowFilterEvaluator evaluator, BufferAllocator allocator) {
        this.delegate = delegate;
        this.evaluator = evaluator;
        this.allocator = allocator;
        this.schema = delegate.schema();
    }

    @Override
    public Schema schema() {
        return schema;
    }

    @Override
    public boolean next() {
        releaseCurrentIfOwned();
        while (!delegateExhausted) {
            if (!delegate.next()) {
                delegateExhausted = true;
                return false;
            }
            VectorSchemaRoot in = delegate.root();
            int rowCount = in.getRowCount();
            int[] matching = matchingRows(in, rowCount);
            if (matching.length == 0) {
                continue; // this whole delegate batch matched nothing: pull the next one, don't surface an empty batch
            }
            current = selectRows(in, matching);
            detached = false;
            return true;
        }
        return false;
    }

    private int[] matchingRows(VectorSchemaRoot in, int rowCount) {
        int[] buf = new int[rowCount];
        int n = 0;
        for (int i = 0; i < rowCount; i++) {
            if (evaluator.matches(in, i)) buf[n++] = i;
        }
        if (n == rowCount) return buf; // no allocation-avoidance possible here without a resizable-int-array type; acceptable in the go-slow path
        int[] trimmed = new int[n];
        System.arraycopy(buf, 0, trimmed, 0, n);
        return trimmed;
    }

    private VectorSchemaRoot selectRows(VectorSchemaRoot in, int[] rows) {
        List<FieldVector> outVectors = new ArrayList<>(schema.getFields().size());
        try {
            for (org.apache.arrow.vector.types.pojo.Field f : schema.getFields()) {
                FieldVector src = in.getVector(f.getName());
                FieldVector dst = f.createVector(allocator);
                dst.setInitialCapacity(rows.length);
                dst.allocateNew();
                for (int i = 0; i < rows.length; i++) {
                    dst.copyFromSafe(rows[i], i, src);
                }
                dst.setValueCount(rows.length);
                outVectors.add(dst);
            }
        } catch (RuntimeException | Error e) {
            for (FieldVector v : outVectors) closeQuietly(v);
            throw e;
        }
        return new VectorSchemaRoot(schema, outVectors, rows.length);
    }

    @Override
    public VectorSchemaRoot root() {
        return current;
    }

    @Override
    public VectorSchemaRoot detach() {
        VectorSchemaRoot r = current;
        detached = true;
        current = null;
        return r;
    }

    private void releaseCurrentIfOwned() {
        if (current != null && !detached) {
            current.close();
        }
        current = null;
        detached = false;
    }

    @Override
    public void close() {
        releaseCurrentIfOwned();
        delegate.close();
    }

    private static void closeQuietly(FieldVector v) {
        try {
            v.close();
        } catch (RuntimeException ignored) {
            // best-effort cleanup on an already-failing path
        }
    }
}
