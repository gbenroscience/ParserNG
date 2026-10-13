package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ast.OrderItem;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Ordering and gathering of the (small) result rows of an aggregation, shared by the Java aggregation paths. */
final class ResultOps {

    private ResultOps() { }

    /** A new root holding rows {@code idx} of {@code full}, in that order; the caller closes it. */
    static VectorSchemaRoot gather(BufferAllocator alloc, FieldVector[] full, int[] idx) {
        List<FieldVector> out = new ArrayList<>(full.length);
        try {
            for (FieldVector f : full) {
                FieldVector v = f.getField().createVector(alloc);
                out.add(v);
                v.allocateNew();
                for (int i = 0; i < idx.length; i++) v.copyFromSafe(idx[i], i, f);
                v.setValueCount(idx.length);
            }
        } catch (RuntimeException e) {
            for (FieldVector v : out) v.close();
            throw e;
        }
        VectorSchemaRoot r = new VectorSchemaRoot(out);
        r.setRowCount(idx.length);
        return r;
    }

    /**
     * {@code idx} reordered by {@code orderBy}, whose items name columns of {@code full} by {@code names}: nulls
     * last in both directions, ties keep the order of {@code idx}.
     */
    static int[] sort(FieldVector[] full, String[] names, List<OrderItem> orderBy, int[] idx) {
        int k = orderBy.size();
        int[] colOf = new int[k];
        boolean[] desc = new boolean[k];
        Object[][] vals = new Object[k][];
        for (int i = 0; i < k; i++) {
            OrderItem o = orderBy.get(i);
            String n = com.github.gbenroscience.sqlv1.QuotedIdentifiers.decodeAll(o.exprText().trim());
            colOf[i] = Arrays.asList(names).indexOf(n);
            if (colOf[i] < 0) throw new ParquetSqlException("ORDER BY " + n + " does not name an output column (one of " + Arrays.toString(names) + ")");
            desc[i] = o.descending();
            vals[i] = new Object[full[colOf[i]].getValueCount()];
            for (int g = 0; g < vals[i].length; g++) vals[i][g] = full[colOf[i]].getObject(g);
        }
        Integer[] order = new Integer[idx.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (x, y) -> {
            for (int i = 0; i < k; i++) {
                Object a = vals[i][idx[x]], b = vals[i][idx[y]];
                int c;
                if (a == null || b == null) {
                    c = a == null ? (b == null ? 0 : 1) : -1;
                } else {
                    c = compare(a, b, names[colOf[i]]);
                    if (desc[i]) c = -c;
                }
                if (c != 0) return c;
            }
            return Integer.compare(x, y);
        });
        int[] out = new int[idx.length];
        for (int i = 0; i < out.length; i++) out[i] = idx[order[i]];
        return out;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static int compare(Object a, Object b, String column) {
        if (a instanceof Number x && b instanceof Number y) {
            if (a instanceof Long && b instanceof Long) return Long.compare(x.longValue(), y.longValue());
            return Double.compare(x.doubleValue(), y.doubleValue());
        }
        if (a instanceof org.apache.arrow.vector.util.Text && b instanceof org.apache.arrow.vector.util.Text) {
            return Arrays.compareUnsigned(a.toString().getBytes(StandardCharsets.UTF_8), b.toString().getBytes(StandardCharsets.UTF_8));
        }
        if (a.getClass() == b.getClass() && a instanceof Comparable) return ((Comparable) a).compareTo(b);
        throw new ParquetSqlException("ORDER BY " + column + ": values of type " + a.getClass().getSimpleName() + " are not orderable");
    }
}
