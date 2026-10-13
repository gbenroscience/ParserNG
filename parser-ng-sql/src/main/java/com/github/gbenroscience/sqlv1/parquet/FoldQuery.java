package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ast.AggFunc;
import com.github.gbenroscience.sqlv1.ast.AggregateSpec;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DateMilliVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.UInt1Vector;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The queries the numeric engine cannot answer, folded in Java over the native batches of an
 * <em>inner</em> plain projection (same {@code FROM} and {@code WHERE}, so row-group pruning, native
 * predicates and streaming all apply to it):
 *
 * <ul>
 * <li><b>Global aggregates</b> (no {@code GROUP BY}) that are not purely numeric: {@code COUNT(col)} on any
 * column, {@code COUNT(DISTINCT col)}, and {@code MIN}/{@code MAX} over strings, dates, timestamps and
 * booleans. In the same query {@code COUNT(*)}, {@code SUM}, {@code AVG} and numeric {@code MIN}/{@code MAX}
 * are folded too, so a query is never split between two engines. Every argument must be a plain column.
 * {@code COUNT}, {@code SUM}, {@code AVG} and numeric {@code MIN}/{@code MAX} are {@code Float8} (the
 * numeric engine's aggregate type); string, date, timestamp and boolean {@code MIN}/{@code MAX} keep the
 * column's own type. A numeric aggregate over a 64-bit integer or decimal that exceeds 2^53 fails loudly.</li>
 * <li><b>{@code SELECT DISTINCT}</b> over columns or numeric expressions, optionally {@code ORDER BY} output
 * columns and {@code LIMIT}. Without {@code ORDER BY}, {@code LIMIT n} stops the scan after {@code n}
 * distinct rows. Rows compare by value ({@code -0.0 = 0.0}, nulls are equal to each other); distinct
 * rows are held in memory, capped by {@code maxMaterializedRows}.</li>
 * </ul>
 *
 * <p>Only ungrouped queries come here: anything with a {@code GROUP BY} is folded by {@link GroupFold}
 * (see {@link ParquetSqlPlan}). {@code HAVING} without {@code GROUP BY} is rejected with a named error.
 * Distinct keys are boxed (one object per value), so this path is correct and bounded but not
 * zero-allocation. Not thread safe.
 */
final class FoldQuery implements Folder {

    /** Thrown from the sink to end the scan early; never escapes {@link #execute}. */
    private static final class StopScan extends RuntimeException {
        StopScan() {
            super(null, null, false, false);
        }
    }

    private static final double TWO_POW_53 = 9007199254740992.0;

    private final SelectStatement stmt;
    private final TableSchema schema;
    private final Schema arrow;
    private final boolean aggregates;

    /**
     * @return true if {@code st} must be folded rather than sent to the numeric engine: {@code SELECT DISTINCT},
     *     {@code COUNT(DISTINCT x)}, an aggregate over a string, date, timestamp or boolean, or, with
     *     {@code GROUP BY}, such a group key
     */
    static boolean needed(SelectStatement st, TableSchema schema) {
        if (st.distinct()) return true;
        for (String g : st.groupBy()) {
            if (nonNumeric(aliasTarget(st, g), schema)) return true;
        }
        for (SelectItem it : st.items()) {
            if (!it.isAggregate()) continue;
            AggregateSpec a = it.aggregate();
            if (a.distinct()) return true;
            if (!a.star() && nonNumeric(a.argExprText(), schema)) return true;
        }
        return false;
    }

    private static String aliasTarget(SelectStatement st, String g) {
        String t = g.trim();
        for (SelectItem it : st.items()) {
            if (!it.isAggregate() && it.alias() != null && it.alias().equals(t)) return it.exprText();
        }
        return t;
    }

    /** True if {@code text} is a column or expression whose value is not a number. */
    static boolean nonNumeric(String text, TableSchema schema) {
        String t = text.trim();
        ColumnKind k = schema.kind(t);
        if (k != null) return !k.numeric();
        // LENGTH(...) and EXTRACT(...) are numeric whatever they contain
        String bare = TemporalFns.stripCalls(StringFns.stripCalls(t, StringFns.LENGTH_NAMES), java.util.Set.of("EXTRACT"));
        return StringFns.hasStringCall(bare) || TemporalFns.producesTemporal(bare);
    }

    FoldQuery(SelectStatement stmt, TableSchema schema, Schema arrow) {
        this.stmt = stmt;
        this.schema = schema;
        this.arrow = arrow;
        this.aggregates = stmt.items().stream().anyMatch(SelectItem::isAggregate);
        if (!stmt.groupBy().isEmpty()) {
            // Invariant: ParquetSqlPlan sends every grouped query to GroupFold. Fail loudly rather than fold it wrongly.
            throw new IllegalStateException("FoldQuery does not handle GROUP BY; grouped queries are folded by GroupFold");
        }
        if (stmt.having() != null) {
            throw new ParquetSqlException("HAVING without GROUP BY is not supported together with SELECT DISTINCT, COUNT(DISTINCT ...)"
                    + " or an aggregate over a string, date, timestamp or boolean column; add a GROUP BY or filter in WHERE");
        }
        if (aggregates) validateAggregates();
        else validateDistinctRows();
    }

    /** The plain projection the scan runs: the aggregate argument columns, or the select list. */
    @Override
    public SelectStatement inner() {
        List<SelectItem> items = new ArrayList<>();
        if (aggregates) {
            Set<String> cols = new LinkedHashSet<>();
            for (SelectItem it : stmt.items()) if (!it.aggregate().star()) cols.add(it.aggregate().argExprText().trim());
            for (String c : cols) items.add(SelectItem.plain(c, null));
        } else {
            items.addAll(stmt.items());
        }
        return new SelectStatement(!aggregates && stmt.selectAll(), items, stmt.table(), stmt.where(), List.of(), null,
                List.of(), null, stmt.tableKind(), stmt.tablePaths(), false);
    }

    // ------------------------------------------------------------------ validation

    private void validateAggregates() {
        for (SelectItem it : stmt.items()) {
            if (!it.isAggregate()) {
                throw new ParquetSqlException("Select item '" + it.exprText() + "' is not an aggregate; without GROUP BY every item must be");
            }
            AggregateSpec a = it.aggregate();
            if (a.star()) continue;
            String col = a.argExprText().trim();
            ColumnKind k = schema.kind(col);
            if (k == null) {
                throw new ParquetSqlException(a.defaultOutputName() + ": '" + col + "' must be a plain column here (this query has a DISTINCT or"
                        + " non-numeric aggregate, so aggregates over expressions are not supported); split the query");
            }
            if (a.func() == AggFunc.COUNT) {
                if (k == ColumnKind.OTHER) {
                    throw new ParquetSqlException(a.defaultOutputName() + ": column " + col + " has an unsupported type for DISTINCT (nested or UUID)");
                }
                continue;
            }
            boolean minMax = a.func() == AggFunc.MIN || a.func() == AggFunc.MAX;
            boolean ok = k.numeric() || (minMax && (k == ColumnKind.STRING || k == ColumnKind.DATE || k == ColumnKind.TIMESTAMP || k == ColumnKind.BOOL));
            if (!ok) {
                throw new ParquetSqlException(a.defaultOutputName() + ": " + a.func() + " is not supported on column " + col + " of type " + k
                        + " (supported: numeric; MIN/MAX also string, date, timestamp, boolean)");
            }
        }
        if (!stmt.orderBy().isEmpty() || stmt.limit() != null && stmt.limit() < 1) {
            // one-row result: ORDER BY is meaningless, LIMIT 0 would empty it; keep the semantics explicit
            if (!stmt.orderBy().isEmpty()) throw new ParquetSqlException("ORDER BY is not meaningful on a single-row aggregate result");
        }
    }

    private void validateDistinctRows() {
        Set<String> names = new LinkedHashSet<>();
        if (stmt.selectAll()) names.addAll(schema.names());
        else for (SelectItem it : stmt.items()) names.add(it.outputName());
        for (OrderItem o : stmt.orderBy()) {
            if (!names.contains(com.github.gbenroscience.sqlv1.QuotedIdentifiers.real(o.exprText().trim()))) {
                throw new ParquetSqlException("ORDER BY " + o.exprText() + " with SELECT DISTINCT must name a select-list column or alias (one of " + names + ")");
            }
        }
    }

    // ------------------------------------------------------------------ execution

    /**
     * Runs the fold over {@code inner}'s batches.
     *
     * @param maxRows cap on distinct rows held in memory
     * @return the result; the caller closes it
     */
    @Override
    public VectorSchemaRoot execute(BufferAllocator alloc, ParquetSqlPlan inner, long maxRows) {
        return aggregates ? executeAggregates(alloc, inner, maxRows) : executeDistinct(alloc, inner, maxRows);
    }

    private VectorSchemaRoot executeAggregates(BufferAllocator alloc, ParquetSqlPlan inner, long maxRows) {
        List<Acc> accs = new ArrayList<>();
        for (SelectItem it : stmt.items()) accs.add(Acc.of(it, schema, arrow, maxRows));
        inner.stream(alloc, root -> {
            int n = root.getRowCount();
            for (Acc a : accs) a.accept(a.column == null ? null : root.getVector(a.column), n);
        });
        List<FieldVector> out = new ArrayList<>();
        try {
            for (Acc a : accs) {
                FieldVector v = a.newOutput(alloc);
                out.add(v);
                v.allocateNew();
                a.emit(v);
                v.setValueCount(1);
            }
        } catch (RuntimeException e) {
            for (FieldVector v : out) v.close();
            throw e;
        }
        VectorSchemaRoot r = new VectorSchemaRoot(out);
        r.setRowCount(1);
        return r;
    }

    private VectorSchemaRoot executeDistinct(BufferAllocator alloc, ParquetSqlPlan inner, long maxRows) {
        final boolean ordered = !stmt.orderBy().isEmpty();
        // LIMIT + OFFSET rows are produced; the plan drops the offset afterwards.
        final long limit = stmt.limit() == null ? Long.MAX_VALUE : (long) stmt.limit() + stmt.offset();
        if (stmt.limit() != null && stmt.limit() == 0) return inner.emptyResult(alloc);
        final Set<List<Object>> seen = new HashSet<>();
        final List<Object[]> rows = new ArrayList<>();
        final FieldVector[][] acc = new FieldVector[1][];
        final int[] count = {0};
        try {
            try {
                inner.stream(alloc, root -> {
                    List<FieldVector> vs = root.getFieldVectors();
                    if (acc[0] == null) {
                        acc[0] = new FieldVector[vs.size()];
                        for (int c = 0; c < vs.size(); c++) {
                            acc[0][c] = vs.get(c).getField().createVector(alloc);
                            acc[0][c].allocateNew();
                        }
                    }
                    int n = root.getRowCount();
                    for (int r = 0; r < n; r++) {
                        Object[] key = new Object[vs.size()];
                        for (int c = 0; c < key.length; c++) key[c] = norm(vs.get(c).getObject(r));
                        if (!seen.add(Arrays.asList(key))) continue;
                        if (count[0] >= maxRows) {
                            throw new ParquetSqlException("SELECT DISTINCT produced more than maxMaterializedRows (" + maxRows
                                    + ") distinct rows; add a WHERE or LIMIT, or raise the cap");
                        }
                        for (int c = 0; c < key.length; c++) acc[0][c].copyFromSafe(r, count[0], vs.get(c));
                        rows.add(key);
                        count[0]++;
                        if (!ordered && count[0] >= limit) throw new StopScan();
                    }
                });
            } catch (StopScan done) {
                // limit reached without ORDER BY: the reader was closed by the plan's own finally blocks
            }
            if (acc[0] == null) return inner.emptyResult(alloc);
            int[] order = new int[count[0]];
            for (int i = 0; i < order.length; i++) order[i] = i;
            if (ordered) {
                order = sortedOrder(rows, inner);
            }
            int keep = (int) Math.min(order.length, limit);
            boolean identity = !ordered && keep == count[0];
            FieldVector[] outV = acc[0];
            if (!identity) {
                FieldVector[] next = new FieldVector[outV.length];
                try {
                    for (int c = 0; c < outV.length; c++) {
                        next[c] = outV[c].getField().createVector(alloc);
                        next[c].allocateNew();
                        for (int i = 0; i < keep; i++) next[c].copyFromSafe(order[i], i, outV[c]);
                    }
                } catch (RuntimeException e) {
                    for (FieldVector v : next) if (v != null) v.close();
                    throw e;
                }
                for (FieldVector v : outV) v.close();
                outV = next;
            }
            acc[0] = null;
            List<FieldVector> list = new ArrayList<>(Arrays.asList(outV));
            for (FieldVector v : list) v.setValueCount(keep);
            VectorSchemaRoot res = new VectorSchemaRoot(list);
            res.setRowCount(keep);
            return res;
        } finally {
            if (acc[0] != null) for (FieldVector v : acc[0]) if (v != null) v.close();
        }
    }

    /** Stable ordering of the distinct rows by the ORDER BY list: nulls last in both directions, as the engine does. */
    private int[] sortedOrder(List<Object[]> rows, ParquetSqlPlan inner) {
        List<String> names = inner.outputNames();
        int k = stmt.orderBy().size();
        int[] pos = new int[k];
        boolean[] desc = new boolean[k];
        for (int i = 0; i < k; i++) {
            OrderItem o = stmt.orderBy().get(i);
            pos[i] = names.indexOf(com.github.gbenroscience.sqlv1.QuotedIdentifiers.real(o.exprText().trim()));
            desc[i] = o.descending();
            if (pos[i] < 0) throw new ParquetSqlException("ORDER BY " + o.exprText() + " is not an output column of this DISTINCT query");
        }
        Integer[] idx = new Integer[rows.size()];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (x, y) -> {
            for (int i = 0; i < k; i++) {
                Object a = rows.get(x)[pos[i]], b = rows.get(y)[pos[i]];
                int c;
                if (a == null || b == null) {
                    c = a == null ? (b == null ? 0 : 1) : -1;
                } else {
                    c = compareValues(a, b, names.get(pos[i]));
                    if (desc[i]) c = -c;
                }
                if (c != 0) return c;
            }
            return Integer.compare(x, y);
        });
        int[] out = new int[idx.length];
        for (int i = 0; i < out.length; i++) out[i] = idx[i];
        return out;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static int compareValues(Object a, Object b, String column) {
        if (a instanceof ByteBuffer || !(a instanceof Comparable) || a.getClass() != b.getClass()) {
            throw new ParquetSqlException("ORDER BY " + column + " with SELECT DISTINCT: values of type " + a.getClass().getSimpleName()
                    + " are not orderable");
        }
        return ((Comparable) a).compareTo(b);
    }

    /** Value identity for DISTINCT: byte arrays by content, {@code -0.0} equal to {@code 0.0}. */
    static Object norm(Object o) {
        if (o instanceof byte[] b) return ByteBuffer.wrap(b);
        if (o instanceof Double d && d == 0.0) return 0.0;
        if (o instanceof Float f && f == 0.0f) return 0.0f;
        return o;
    }

    // ------------------------------------------------------------------ aggregate accumulators

    private abstract static class Acc {
        final String outName;
        /** The inner column this accumulator reads, or null for COUNT(*). */
        final String column;

        Acc(String outName, String column) {
            this.outName = outName;
            this.column = column;
        }

        abstract void accept(FieldVector v, int rows);

        abstract void emit(FieldVector out);

        FieldVector newOutput(BufferAllocator a) {
            return new Float8Vector(outName, a);
        }

        static Acc of(SelectItem it, TableSchema schema, Schema arrow, long maxRows) {
            AggregateSpec a = it.aggregate();
            String name = it.outputName();
            if (a.star()) return new CountStar(name);
            String col = a.argExprText().trim();
            ColumnKind k = schema.kind(col);
            if (a.func() == AggFunc.COUNT) return a.distinct() ? new CountDistinct(name, col, maxRows) : new CountCol(name, col);
            if (k.numeric()) return new NumAgg(name, col, a.func(), k);
            org.apache.arrow.vector.types.pojo.Field src = schema.pathField(col);
            if (src == null) src = arrow.findField(com.github.gbenroscience.sqlv1.QuotedIdentifiers.real(col));
            return new Extreme(name, col, a.func() == AggFunc.MAX, src);
        }
    }

    private static final class CountStar extends Acc {
        private long n;

        CountStar(String name) {
            super(name, null);
        }

        @Override void accept(FieldVector v, int rows) { n += rows; }

        @Override void emit(FieldVector out) { ((Float8Vector) out).set(0, n); }
    }

    private static final class CountCol extends Acc {
        private long n;

        CountCol(String name, String col) {
            super(name, col);
        }

        @Override void accept(FieldVector v, int rows) { n += rows - v.getNullCount(); }

        @Override void emit(FieldVector out) { ((Float8Vector) out).set(0, n); }
    }

    private static final class CountDistinct extends Acc {
        private final Set<Object> seen = new HashSet<>();
        private ByteKeyTable bytes;
        private LongKeyTable longs;
        private ByteScratch scratch;
        private final long maxRows;

        CountDistinct(String name, String col, long maxRows) {
            super(name, col);
            this.maxRows = maxRows;
        }

        @Override void accept(FieldVector v, int rows) {
            if (v instanceof VarCharVector s) { // strings are hashed from a reused buffer: no object per row
                if (bytes == null) {
                    bytes = new ByteKeyTable();
                    scratch = new ByteScratch();
                }
                for (int i = 0; i < rows; i++) {
                    if (s.isNull(i)) continue;
                    { int len = scratch.load(s, i); bytes.idOf(scratch.buf, len); }
                }
            } else if (VecKeys.hasLongKey(v)) {
                if (longs == null) longs = new LongKeyTable();
                for (int i = 0; i < rows; i++) {
                    if (!v.isNull(i)) longs.idOf(VecKeys.longOf(v, i));
                }
            } else {
                for (int i = 0; i < rows; i++) {
                    if (v.isNull(i)) continue;
                    seen.add(norm(v.getObject(i)));
                }
            }
            if (size() > maxRows) {
                throw new ParquetSqlException(outName + " has more than maxMaterializedRows (" + maxRows + ") distinct values; raise the cap");
            }
        }

        private int size() {
            return bytes != null ? bytes.size() : longs != null ? longs.size() : seen.size();
        }

        @Override void emit(FieldVector out) { ((Float8Vector) out).set(0, size()); }
    }

    private static final class NumAgg extends Acc {
        private final AggFunc func;
        private final ColumnKind kind;
        private long count;
        private double sum;
        private double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;

        NumAgg(String name, String col, AggFunc func, ColumnKind kind) {
            super(name, col);
            this.func = func;
            this.kind = kind;
        }

        @Override void accept(FieldVector v, int rows) {
            for (int i = 0; i < rows; i++) {
                if (v.isNull(i)) continue;
                double d = read(v, i, column, kind);
                count++;
                sum += d;
                if (d < min) min = d;
                if (d > max) max = d;
            }
        }

        @Override void emit(FieldVector out) {
            Float8Vector o = (Float8Vector) out;
            if (count == 0) return; // SQL: SUM/AVG/MIN/MAX of no rows is NULL
            o.set(0, switch (func) {
                case SUM -> sum;
                case AVG -> sum / count;
                case MIN -> min;
                default -> max;
            });
        }
    }

    /** MIN/MAX over strings (UTF-8 byte order), dates, timestamps and booleans, in the column's own type. */
    private static final class Extreme extends Acc {
        private final boolean wantMax;
        private final Field srcField;
        private boolean has;
        private long best;
        private byte[] bestBytes;

        Extreme(String name, String col, boolean wantMax, Field srcField) {
            super(name, col);
            this.wantMax = wantMax;
            this.srcField = srcField;
        }

        @Override FieldVector newOutput(BufferAllocator a) {
            return new Field(outName, srcField.getFieldType(), srcField.getChildren()).createVector(a);
        }

        @Override void accept(FieldVector v, int rows) {
            if (v instanceof VarCharVector s) {
                for (int i = 0; i < rows; i++) {
                    if (s.isNull(i)) continue;
                    // compared in place; the value is copied only when it becomes the new best
                    if (!has || (wantMax ? VecKeys.compareRow(s, i, bestBytes) > 0 : VecKeys.compareRow(s, i, bestBytes) < 0)) {
                        bestBytes = s.get(i);
                        has = true;
                    }
                }
                return;
            }
            for (int i = 0; i < rows; i++) {
                if (v.isNull(i)) continue;
                long x = readLong(v, i);
                if (!has || (wantMax ? x > best : x < best)) {
                    best = x;
                    has = true;
                }
            }
        }

        @Override void emit(FieldVector out) {
            if (!has) return;
            if (out instanceof VarCharVector s) s.setSafe(0, bestBytes);
            else if (out instanceof DateDayVector d) d.setSafe(0, (int) best);
            else if (out instanceof DateMilliVector d) d.setSafe(0, best);
            else if (out instanceof TimeStampVector t) t.setSafe(0, best);
            else if (out instanceof BitVector b) b.setSafe(0, (int) best);
            else throw new ParquetSqlException("MIN/MAX is not supported on column " + column + " of vector type " + out.getClass().getSimpleName());
        }
    }

    static long readLong(FieldVector v, int i) {
        if (v instanceof DateDayVector d) return d.get(i);
        if (v instanceof DateMilliVector d) return d.get(i);
        if (v instanceof TimeStampVector t) return t.get(i);
        if (v instanceof BitVector b) return b.get(i);
        throw new ParquetSqlException("MIN/MAX is not supported on vector type " + v.getClass().getSimpleName());
    }

    /** One numeric value as a double; a 64-bit integer or decimal beyond 2^53 fails loudly rather than rounding. */
    static double read(FieldVector v, int i, String column, ColumnKind kind) {
        double d;
        if (v instanceof Float8Vector x) d = x.get(i);
        else if (v instanceof Float4Vector x) d = x.get(i);
        else if (v instanceof IntVector x) d = x.get(i);
        else if (v instanceof BigIntVector x) d = x.get(i);
        else if (v instanceof SmallIntVector x) d = x.get(i);
        else if (v instanceof TinyIntVector x) d = x.get(i);
        else if (v instanceof UInt1Vector x) d = x.get(i) & 0xFF;
        else if (v instanceof UInt2Vector x) d = x.get(i) & 0xFFFF;
        else if (v instanceof UInt4Vector x) d = x.get(i) & 0xFFFFFFFFL;
        else if (v instanceof UInt8Vector x) {
            long l = x.get(i);
            d = l < 0 ? Double.POSITIVE_INFINITY : (double) l;
        } else if (v instanceof DecimalVector x) {
            BigDecimal bd = x.getObject(i);
            d = bd.doubleValue();
        } else {
            throw new ParquetSqlException("Column " + column + " (" + kind + ", vector " + v.getClass().getSimpleName() + ") cannot be aggregated");
        }
        if (kind.wide() && Math.abs(d) > TWO_POW_53) {
            throw new ParquetSqlException("Column " + column + " (" + kind + ") holds a value beyond 2^53; its aggregate would be inexact as a double."
                    + " Aggregate it on its own with Int64Policy.ROUND_TO_DOUBLE if rounding is acceptable");
        }
        return d;
    }
}
