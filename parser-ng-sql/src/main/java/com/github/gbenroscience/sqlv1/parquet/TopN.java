package com.github.gbenroscience.sqlv1.parquet;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Bounded {@code ORDER BY ... LIMIT n}: keeps at most about {@code 2n} candidate rows however many
 * rows survive the WHERE, instead of materializing all of them.
 *
 * <p>Each input batch is the engine's own per-batch answer (already sorted and cut to {@code n}).
 * Its first {@code outCount} vectors are the output columns, in native type; the next
 * {@code keyCount} vectors are the sort keys as {@code Float8}, nulls preserved. Ordering is the
 * engine's: keys compare as doubles, nulls last whatever the direction, ties keep arrival order
 * (file, batch, row), so the result equals sorting every row at once.
 *
 * <p>Rows are copied once, with the vector's own {@code copyFromSafe}. After the first {@code n}
 * rows are known the {@code n}-th best key tuple is a threshold: a later row that is not strictly
 * better is dropped before any copy. Not thread safe.
 */
final class TopN implements AutoCloseable {

    private final BufferAllocator alloc;
    private final int limit;
    private final int outCount;
    private final int keyCount;
    private final boolean[] desc;

    private Field[] fields;
    private FieldVector[] acc;
    private double[][] keys;
    private boolean[][] nulls;
    private int size;
    /** Key tuple of the current n-th best row, valid once {@code haveThreshold}. */
    private final double[] thrVal;
    private final boolean[] thrNull;
    private boolean haveThreshold;

    TopN(BufferAllocator alloc, int limit, int outCount, boolean[] desc) {
        if (limit < 1) throw new IllegalArgumentException("limit must be at least 1");
        this.alloc = alloc;
        this.limit = limit;
        this.outCount = outCount;
        this.keyCount = desc.length;
        this.desc = desc.clone();
        this.thrVal = new double[keyCount];
        this.thrNull = new boolean[keyCount];
    }

    /** Offers every row of {@code batch} (layout in the class comment). The batch is not retained. */
    void add(VectorSchemaRoot batch) {
        int n = batch.getRowCount();
        if (n == 0) return;
        if (acc == null) init(batch);
        Float8Vector[] kv = new Float8Vector[keyCount];
        for (int k = 0; k < keyCount; k++) kv[k] = (Float8Vector) batch.getFieldVectors().get(outCount + k);
        int cap = 2 * limit;
        for (int r = 0; r < n; r++) {
            // The batch is sorted by the same keys, so once a row cannot enter the top n none after it can.
            if (haveThreshold && !strictlyBetter(kv, r)) break; // ties lose: they arrive later
            ensureCapacity(size + 1);
            for (int c = 0; c < outCount; c++) acc[c].copyFromSafe(r, size, batch.getFieldVectors().get(c));
            for (int k = 0; k < keyCount; k++) {
                boolean isNull = kv[k].isNull(r);
                nulls[k][size] = isNull;
                keys[k][size] = isNull ? 0.0 : kv[k].get(r);
            }
            size++;
            if (size >= cap) compact();
        }
    }

    /**
     * Returns the sorted top rows as a root the caller closes; this object then holds nothing.
     * With no input at all, returns {@code null} (the caller builds its own empty result).
     */
    VectorSchemaRoot finish() {
        if (acc == null) return null;
        compact();
        List<FieldVector> vecs = new ArrayList<>(outCount);
        List<Field> fs = new ArrayList<>(outCount);
        for (int c = 0; c < outCount; c++) {
            acc[c].setValueCount(size);
            vecs.add(acc[c]);
            fs.add(fields[c]);
            acc[c] = null;
        }
        acc = null;
        return new VectorSchemaRoot(new Schema(fs), vecs, size);
    }

    @Override
    public void close() {
        if (acc == null) return;
        for (FieldVector v : acc) if (v != null) v.close();
        acc = null;
    }

    // ------------------------------------------------------------------ internals

    private void init(VectorSchemaRoot batch) {
        fields = new Field[outCount];
        acc = new FieldVector[outCount];
        int cap = Math.min(Math.max(16, 2 * limit), 4096);
        try {
            for (int c = 0; c < outCount; c++) {
                fields[c] = batch.getFieldVectors().get(c).getField();
                acc[c] = fields[c].createVector(alloc);
                acc[c].allocateNew();
            }
        } catch (RuntimeException e) {
            close();
            throw e;
        }
        keys = new double[keyCount][cap];
        nulls = new boolean[keyCount][cap];
    }

    private void ensureCapacity(int need) {
        if (keyCount == 0) return;
        int cap = keys[0].length;
        if (need <= cap) return;
        int next = Math.max(need, cap * 2);
        for (int k = 0; k < keyCount; k++) {
            keys[k] = Arrays.copyOf(keys[k], next);
            nulls[k] = Arrays.copyOf(nulls[k], next);
        }
    }

    /** True if row {@code r} of the new batch sorts strictly before the current n-th best row. */
    private boolean strictlyBetter(Float8Vector[] kv, int r) {
        for (int k = 0; k < keyCount; k++) {
            boolean n1 = kv[k].isNull(r), n2 = thrNull[k];
            int cmp;
            if (n1 || n2) {
                cmp = (n1 == n2) ? 0 : (n1 ? 1 : -1);
            } else {
                cmp = Double.compare(kv[k].get(r), thrVal[k]);
                if (desc[k]) cmp = -cmp;
            }
            if (cmp != 0) return cmp < 0;
        }
        return false; // equal keys: the earlier arrival wins
    }

    private int compareRows(int i1, int i2) {
        for (int k = 0; k < keyCount; k++) {
            boolean n1 = nulls[k][i1], n2 = nulls[k][i2];
            int cmp;
            if (n1 || n2) {
                cmp = (n1 == n2) ? 0 : (n1 ? 1 : -1);
            } else {
                cmp = Double.compare(keys[k][i1], keys[k][i2]);
                if (desc[k]) cmp = -cmp;
            }
            if (cmp != 0) return cmp;
        }
        return Integer.compare(i1, i2); // stable: arrival order
    }

    /** Sorts the candidates, keeps the best {@code limit}, and records the n-th best as the threshold. */
    private void compact() {
        int m = size;
        Integer[] order = new Integer[m];
        for (int i = 0; i < m; i++) order[i] = i;
        Arrays.sort(order, this::compareRows);
        int keep = Math.min(m, limit);

        FieldVector[] next = new FieldVector[outCount];
        int kc = keyCount == 0 ? 0 : Math.max(keys[0].length, keep);
        double[][] nk = new double[keyCount][kc];
        boolean[][] nn = new boolean[keyCount][kc];
        try {
            for (int c = 0; c < outCount; c++) {
                next[c] = fields[c].createVector(alloc);
                next[c].allocateNew();
                for (int i = 0; i < keep; i++) next[c].copyFromSafe(order[i], i, acc[c]);
            }
        } catch (RuntimeException e) {
            for (FieldVector v : next) if (v != null) v.close();
            throw e;
        }
        for (int k = 0; k < keyCount; k++) {
            for (int i = 0; i < keep; i++) {
                nk[k][i] = keys[k][order[i]];
                nn[k][i] = nulls[k][order[i]];
            }
        }
        for (FieldVector v : acc) v.close();
        acc = next;
        keys = nk;
        nulls = nn;
        size = keep;
        if (keep == limit && keyCount > 0) {
            for (int k = 0; k < keyCount; k++) {
                thrVal[k] = keys[k][keep - 1];
                thrNull[k] = nulls[k][keep - 1];
            }
            haveThreshold = true;
        }
    }
}
