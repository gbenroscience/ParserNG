package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.QuotedIdentifiers;
import com.github.gbenroscience.sqlv1.SqlParser;
import com.github.gbenroscience.sqlv1.SqlSyntaxException;
import com.github.gbenroscience.sqlv1.ast.AggFunc;
import com.github.gbenroscience.sqlv1.ast.AggregateSpec;
import com.github.gbenroscience.sqlv1.ast.AndExpr;
import com.github.gbenroscience.sqlv1.ast.BetweenExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.CompOp;
import com.github.gbenroscience.sqlv1.ast.ComparisonExpr;
import com.github.gbenroscience.sqlv1.ast.InExpr;
import com.github.gbenroscience.sqlv1.ast.IsNullExpr;
import com.github.gbenroscience.sqlv1.ast.NotExpr;
import com.github.gbenroscience.sqlv1.ast.OrExpr;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DateMilliVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * {@code GROUP BY} the numeric engine cannot do, folded in Java: a string, boolean, date or timestamp group
 * key or expression ({@code GROUP BY name}, {@code GROUP BY LOWER(name)}, {@code GROUP BY CAST(ts AS DATE)}),
 * {@code COUNT(DISTINCT x)}, {@code MIN}/{@code MAX} over strings, dates, timestamps and booleans, and
 * {@code SELECT DISTINCT} combined with {@code GROUP BY}.
 *
 * <p>The scan is an inner plain projection of the group keys and aggregate arguments, so every argument
 * may be any expression the rest of the bridge understands (a numeric expression, a string function, a nested
 * field, a cast). Groups are found by a hash of the key values, nulls forming their own group as in SQL;
 * a group's key values are copied from its first row with the vector's own {@code copyFromSafe}, so keys keep
 * their native type. Aggregates ({@code COUNT(*)}, {@code COUNT(x)}, {@code COUNT(DISTINCT x)}, {@code SUM},
 * {@code AVG}, {@code MIN}, {@code MAX}) have the same types and null rules as {@link FoldQuery}: counts, sums,
 * averages and numeric extremes are {@code Float8}, extremes of strings (UTF-8 byte order), dates, timestamps
 * and booleans keep the column's type, a 64-bit integer or decimal beyond 2^53 fails loudly.
 *
 * <p>{@code HAVING} may combine comparisons, {@code BETWEEN}, {@code IN}, {@code IS NULL}, {@code AND},
 * {@code OR} and {@code NOT} over numeric aggregates (by their alias or spelled out, even if not selected) and
 * numbers. {@code ORDER BY} names output columns, aliases or aggregates as written in the select list (nulls
 * last, ties in first-seen order). {@code SELECT DISTINCT} removes duplicate output rows after grouping.
 * Without {@code ORDER BY} the groups come out in first-seen order. Every select item must be a group key or an
 * aggregate. Group count is capped by {@code maxMaterializedRows}; keys are boxed, so this path is correct and
 * bounded but not zero-allocation. Not thread safe.
 */
final class GroupFold implements Folder {

    private record AggDef(AggFunc func, String arg, boolean distinct) { }

    private record Col(String name, int key, int agg) { }

    private final SelectStatement stmt;
    private final List<String> keyTexts = new ArrayList<>();
    private final List<String> argTexts = new ArrayList<>();
    private final List<AggDef> aggs = new ArrayList<>();
    private final List<Col> cols = new ArrayList<>();
    private final HNode having;
    /** Position in the inner projection of each key and argument text (an argument equal to a key shares its position). */
    private final Map<String, Integer> innerPos = new LinkedHashMap<>();

    GroupFold(SelectStatement stmt, TableSchema schema) {
        this.stmt = stmt;
        if (stmt.selectAll()) throw new ParquetSqlException("SELECT * cannot be used with GROUP BY");
        for (String g : stmt.groupBy()) {
            String t = norm(resolveAlias(g));
            if (!keyTexts.contains(t)) keyTexts.add(t);
        }
        for (SelectItem it : stmt.items()) {
            if (it.isAggregate()) {
                AggregateSpec a = it.aggregate();
                cols.add(new Col(it.outputName(), -1, aggIndex(new AggDef(a.func(), a.star() ? null : norm(a.argExprText()), a.distinct()))));
            } else {
                int k = keyTexts.indexOf(norm(it.exprText()));
                if (k < 0) {
                    throw new ParquetSqlException("Select item '" + it.outputName() + "' must appear in the GROUP BY clause or be used in an aggregate function");
                }
                cols.add(new Col(it.outputName(), k, -1));
            }
        }
        this.having = stmt.having() == null ? null : compileHaving(stmt.having());
        for (AggDef d : aggs) if (d.arg() != null && !argTexts.contains(d.arg())) argTexts.add(d.arg());
        for (String t : keyTexts) innerPos.putIfAbsent(t, innerPos.size());
        for (String t : argTexts) innerPos.putIfAbsent(t, innerPos.size());
        Set<String> names = new HashSet<>();
        for (Col c : cols) names.add(c.name());
        for (OrderItem o : stmt.orderBy()) {
            String n = QuotedIdentifiers.decodeAll(o.exprText().trim());
            if (!names.contains(n)) {
                throw new ParquetSqlException("ORDER BY " + n + " with GROUP BY must name a select-list column, alias or aggregate (one of " + names + ")");
            }
        }
    }

    /** Texts come from one parser, so they are already spelled canonically; only surrounding space is dropped. */
    private static String norm(String t) {
        return t.trim();
    }

    /** A {@code GROUP BY} item that is the alias of a select item stands for that item's expression. */
    private String resolveAlias(String g) {
        String t = g.trim();
        for (SelectItem it : stmt.items()) {
            if (!it.isAggregate() && it.alias() != null && it.alias().equals(t) && !norm(it.exprText()).equals(norm(t))) return it.exprText();
        }
        return t;
    }

    private int aggIndex(AggDef d) {
        int i = aggs.indexOf(d);
        if (i >= 0) return i;
        aggs.add(d);
        return aggs.size() - 1;
    }

    @Override
    public SelectStatement inner() {
        List<SelectItem> items = new ArrayList<>();
        for (String t : innerPos.keySet()) items.add(SelectItem.plain(t, null));
        return new SelectStatement(false, items, stmt.table(), stmt.where(), List.of(), null, List.of(), null,
                stmt.tableKind(), stmt.tablePaths(), false, 0);
    }

    // ------------------------------------------------------------------ execution

    @Override
    public VectorSchemaRoot execute(BufferAllocator alloc, ParquetSqlPlan inner, long maxRows) {
        List<Field> fields = inner.outputFields();
        int nk = keyTexts.size();
        GAcc[] acc = new GAcc[aggs.size()];
        int[] argPos = new int[aggs.size()];
        for (int i = 0; i < acc.length; i++) {
            AggDef d = aggs.get(i);
            argPos[i] = d.arg() == null ? -1 : innerPos.get(d.arg());
            acc[i] = GAcc.of(d, argPos[i] < 0 ? null : fields.get(argPos[i]), d.arg());
        }
        FieldVector[] keyVecs = new FieldVector[nk];
        List<FieldVector> toClose = new ArrayList<>();
        try {
            for (int k = 0; k < nk; k++) {
                keyVecs[k] = fields.get(k).createVector(alloc);
                keyVecs[k].allocateNew();
                toClose.add(keyVecs[k]);
            }
            Map<List<Object>, Integer> ids = new HashMap<>();
            int[] groups = {0};
            inner.stream(alloc, root -> {
                List<FieldVector> vs = root.getFieldVectors();
                int n = root.getRowCount();
                for (int r = 0; r < n; r++) {
                    Object[] key = new Object[nk];
                    for (int k = 0; k < nk; k++) key[k] = FoldQuery.norm(vs.get(k).getObject(r));
                    List<Object> kl = Arrays.asList(key);
                    Integer g = ids.get(kl);
                    if (g == null) {
                        if (groups[0] >= maxRows) {
                            throw new ParquetSqlException("GROUP BY produced more than maxMaterializedRows (" + maxRows + ") groups; add a WHERE or raise the cap");
                        }
                        g = groups[0]++;
                        ids.put(kl, g);
                        for (int k = 0; k < nk; k++) keyVecs[k].copyFromSafe(r, g, vs.get(k));
                    }
                    for (int i = 0; i < acc.length; i++) acc[i].accept(g, argPos[i] < 0 ? null : vs.get(argPos[i]), r);
                }
            });
            int g = groups[0];

            FieldVector[] full = new FieldVector[cols.size()];
            for (int ci = 0; ci < full.length; ci++) {
                Col c = cols.get(ci);
                if (c.key() >= 0) {
                    Field f = fields.get(c.key());
                    full[ci] = new Field(c.name(), f.getFieldType(), f.getChildren()).createVector(alloc);
                    toClose.add(full[ci]);
                    full[ci].allocateNew();
                    for (int i = 0; i < g; i++) full[ci].copyFromSafe(i, i, keyVecs[c.key()]);
                    full[ci].setValueCount(g);
                } else {
                    full[ci] = acc[c.agg()].emit(c.name(), g, alloc);
                    toClose.add(full[ci]);
                }
            }

            int[] idx = new int[g];
            int kept = 0;
            for (int i = 0; i < g; i++) {
                if (having != null && !Boolean.TRUE.equals(having.eval(acc, i))) continue;
                idx[kept++] = i;
            }
            idx = Arrays.copyOf(idx, kept);
            if (stmt.distinct()) idx = distinct(full, idx);
            if (!stmt.orderBy().isEmpty()) idx = sort(full, idx);
            long want = stmt.limit() == null ? Long.MAX_VALUE : (long) stmt.limit() + stmt.offset();
            if (idx.length > want) idx = Arrays.copyOf(idx, (int) want);
            return gather(alloc, full, idx);
        } finally {
            for (FieldVector v : toClose) v.close();
        }
    }

    private static VectorSchemaRoot gather(BufferAllocator alloc, FieldVector[] full, int[] idx) {
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

    private static int[] distinct(FieldVector[] full, int[] idx) {
        Set<List<Object>> seen = new HashSet<>();
        int[] keep = new int[idx.length];
        int n = 0;
        for (int g : idx) {
            Object[] row = new Object[full.length];
            for (int c = 0; c < row.length; c++) row[c] = FoldQuery.norm(full[c].getObject(g));
            if (seen.add(Arrays.asList(row))) keep[n++] = g;
        }
        return Arrays.copyOf(keep, n);
    }

    private int[] sort(FieldVector[] full, int[] idx) {
        int k = stmt.orderBy().size();
        int[] colOf = new int[k];
        boolean[] desc = new boolean[k];
        Object[][] vals = new Object[k][];
        for (int i = 0; i < k; i++) {
            OrderItem o = stmt.orderBy().get(i);
            String n = QuotedIdentifiers.decodeAll(o.exprText().trim());
            for (int c = 0; c < cols.size(); c++) if (cols.get(c).name().equals(n)) { colOf[i] = c; break; }
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
                    c = compare(a, b, cols.get(colOf[i]).name());
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
    private static int compare(Object a, Object b, String column) {
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

    // ------------------------------------------------------------------ HAVING

    private interface HVal {
        Double eval(GAcc[] acc, int g);
    }

    private interface HNode {
        Boolean eval(GAcc[] acc, int g);
    }

    private static final Pattern NUMBER = Pattern.compile("[+-]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?");

    private HNode compileHaving(BoolExpr e) {
        return switch (e) {
            case AndExpr(var l, var r) -> {
                HNode a = compileHaving(l), b = compileHaving(r);
                yield (acc, g) -> {
                    Boolean x = a.eval(acc, g), y = b.eval(acc, g);
                    if (Boolean.FALSE.equals(x) || Boolean.FALSE.equals(y)) return false;
                    return x == null || y == null ? null : Boolean.TRUE;
                };
            }
            case OrExpr(var l, var r) -> {
                HNode a = compileHaving(l), b = compileHaving(r);
                yield (acc, g) -> {
                    Boolean x = a.eval(acc, g), y = b.eval(acc, g);
                    if (Boolean.TRUE.equals(x) || Boolean.TRUE.equals(y)) return true;
                    return x == null || y == null ? null : Boolean.FALSE;
                };
            }
            case NotExpr(var inner) -> {
                HNode a = compileHaving(inner);
                yield (acc, g) -> {
                    Boolean x = a.eval(acc, g);
                    return x == null ? null : !x;
                };
            }
            case ComparisonExpr(var l, var op, var r) -> {
                HVal a = val(l), b = val(r);
                yield (acc, g) -> {
                    Double x = a.eval(acc, g), y = b.eval(acc, g);
                    return x == null || y == null ? null : cmp(op, x, y);
                };
            }
            case BetweenExpr(var t, var lo, var hi, var neg) -> {
                HVal a = val(t), l = val(lo), h = val(hi);
                yield (acc, g) -> {
                    Double x = a.eval(acc, g), lv = l.eval(acc, g), hv = h.eval(acc, g);
                    if (x == null || lv == null || hv == null) return null;
                    boolean in = x >= lv && x <= hv;
                    return neg ? !in : in;
                };
            }
            case InExpr(var t, var vals, var neg) -> {
                HVal a = val(t);
                List<HVal> vs = new ArrayList<>();
                for (String v : vals) vs.add(val(v));
                yield (acc, g) -> {
                    Double x = a.eval(acc, g);
                    if (x == null) return null;
                    boolean hit = false, unknown = false;
                    for (HVal v : vs) {
                        Double y = v.eval(acc, g);
                        if (y == null) unknown = true;
                        else if (y.doubleValue() == x.doubleValue()) hit = true;
                    }
                    if (hit) return !neg;
                    return unknown ? null : (Boolean) neg;
                };
            }
            case IsNullExpr(var t, var neg) -> {
                HVal a = val(t);
                yield (acc, g) -> (a.eval(acc, g) == null) != neg;
            }
            default -> throw new ParquetSqlException("HAVING supports comparisons, BETWEEN, IN, IS NULL, AND, OR and NOT over aggregates and numbers, not " + e);
        };
    }

    private static boolean cmp(CompOp op, double x, double y) {
        return switch (op) {
            case EQ -> x == y;
            case NEQ -> x != y;
            case LT -> x < y;
            case LE -> x <= y;
            case GT -> x > y;
            case GE -> x >= y;
        };
    }

    private HVal val(String text) {
        String t = norm(text).replaceAll("^([+-])\\s+", "$1");
        if (NUMBER.matcher(t).matches()) {
            Double c = Double.valueOf(t);
            return (acc, g) -> c;
        }
        String readable = QuotedIdentifiers.decodeAll(t);
        for (Col c : cols) {
            if (c.agg() >= 0 && c.name().equals(readable)) return agg(c.agg());
        }
        try {
            SelectStatement s = SqlParser.parse("SELECT " + t + " FROM t");
            if (s.items().size() == 1 && s.items().get(0).isAggregate()) {
                AggregateSpec a = s.items().get(0).aggregate();
                return agg(aggIndex(new AggDef(a.func(), a.star() ? null : norm(a.argExprText()), a.distinct())));
            }
        } catch (SqlSyntaxException notAnAggregate) {
            // fall through to the error below
        }
        throw new ParquetSqlException("HAVING operand '" + readable + "' must be a number, a select-list aggregate alias, or an aggregate call such as COUNT(*)");
    }

    private static HVal agg(int i) {
        return (acc, g) -> acc[i].number(g);
    }

    // ------------------------------------------------------------------ accumulators

    private abstract static class GAcc {
        abstract void accept(int g, FieldVector v, int row);

        abstract FieldVector emit(String name, int groups, BufferAllocator a);

        /** The group's value as a number for {@code HAVING}; null for no value. */
        Double number(int g) {
            throw new ParquetSqlException("HAVING over a string, date, timestamp or boolean MIN/MAX is not supported");
        }

        static GAcc of(AggDef d, Field arg, String argText) {
            if (d.arg() == null) return new CountStar();
            ColumnKind k = TableSchema.kindOf(arg.getType());
            if (d.func() == AggFunc.COUNT) return d.distinct() ? new CountDistinct() : new CountCol();
            if (k.numeric()) return new NumAgg(d.func(), k, argText);
            boolean minMax = d.func() == AggFunc.MIN || d.func() == AggFunc.MAX;
            if (minMax && (k == ColumnKind.STRING || k == ColumnKind.DATE || k == ColumnKind.TIMESTAMP || k == ColumnKind.BOOL)) {
                return new Extreme(d.func() == AggFunc.MAX, arg, k);
            }
            throw new ParquetSqlException(d.func() + "(" + QuotedIdentifiers.decodeAll(argText) + ") is not supported on a " + k
                    + " value (supported: numeric; MIN/MAX also string, date, timestamp, boolean)");
        }

        static FieldVector floats(String name, int groups, BufferAllocator a, java.util.function.IntFunction<Double> value) {
            Float8Vector v = new Float8Vector(name, a);
            v.allocateNew(Math.max(groups, 1));
            for (int g = 0; g < groups; g++) {
                Double x = value.apply(g);
                if (x == null) v.setNull(g);
                else v.set(g, x);
            }
            v.setValueCount(groups);
            return v;
        }
    }

    private abstract static class Counting extends GAcc {
        long[] n = new long[16];

        final void ensure(int g) {
            if (g >= n.length) n = Arrays.copyOf(n, Math.max(g + 1, n.length * 2));
        }

        @Override FieldVector emit(String name, int groups, BufferAllocator a) {
            return floats(name, groups, a, g -> (double) n[g]);
        }

        @Override Double number(int g) {
            return (double) n[g];
        }
    }

    private static final class CountStar extends Counting {
        @Override void accept(int g, FieldVector v, int row) {
            ensure(g);
            n[g]++;
        }
    }

    private static final class CountCol extends Counting {
        @Override void accept(int g, FieldVector v, int row) {
            ensure(g);
            if (!v.isNull(row)) n[g]++;
        }
    }

    private static final class CountDistinct extends GAcc {
        private final List<Set<Object>> sets = new ArrayList<>();

        @Override void accept(int g, FieldVector v, int row) {
            while (sets.size() <= g) sets.add(new HashSet<>());
            if (!v.isNull(row)) sets.get(g).add(FoldQuery.norm(v.getObject(row)));
        }

        private int size(int g) {
            return g < sets.size() ? sets.get(g).size() : 0;
        }

        @Override FieldVector emit(String name, int groups, BufferAllocator a) {
            return floats(name, groups, a, g -> (double) size(g));
        }

        @Override Double number(int g) {
            return (double) size(g);
        }
    }

    private static final class NumAgg extends GAcc {
        private final AggFunc func;
        private final ColumnKind kind;
        private final String column;
        private long[] count = new long[16];
        private double[] sum = new double[16], min = new double[16], max = new double[16];

        NumAgg(AggFunc func, ColumnKind kind, String column) {
            this.func = func;
            this.kind = kind;
            this.column = column;
            Arrays.fill(min, Double.POSITIVE_INFINITY);
            Arrays.fill(max, Double.NEGATIVE_INFINITY);
        }

        private void ensure(int g) {
            if (g < count.length) return;
            int cap = Math.max(g + 1, count.length * 2), old = count.length;
            count = Arrays.copyOf(count, cap);
            sum = Arrays.copyOf(sum, cap);
            min = Arrays.copyOf(min, cap);
            max = Arrays.copyOf(max, cap);
            for (int i = old; i < cap; i++) {
                min[i] = Double.POSITIVE_INFINITY;
                max[i] = Double.NEGATIVE_INFINITY;
            }
        }

        @Override void accept(int g, FieldVector v, int row) {
            ensure(g);
            if (v.isNull(row)) return;
            double d = FoldQuery.read(v, row, QuotedIdentifiers.decodeAll(column), kind);
            count[g]++;
            sum[g] += d;
            if (d < min[g]) min[g] = d;
            if (d > max[g]) max[g] = d;
        }

        @Override Double number(int g) {
            if (g >= count.length || count[g] == 0) return null;
            return switch (func) {
                case SUM -> sum[g];
                case AVG -> sum[g] / count[g];
                case MIN -> min[g];
                default -> max[g];
            };
        }

        @Override FieldVector emit(String name, int groups, BufferAllocator a) {
            return floats(name, groups, a, this::number);
        }
    }

    /** MIN/MAX over strings (UTF-8 byte order), dates, timestamps and booleans, in the column's own type. */
    private static final class Extreme extends GAcc {
        private final boolean wantMax;
        private final Field src;
        private final ColumnKind kind;
        private boolean[] has = new boolean[16];
        private long[] best = new long[16];
        private byte[][] bestBytes = new byte[16][];

        Extreme(boolean wantMax, Field src, ColumnKind kind) {
            this.wantMax = wantMax;
            this.src = src;
            this.kind = kind;
        }

        private void ensure(int g) {
            if (g < has.length) return;
            int cap = Math.max(g + 1, has.length * 2);
            has = Arrays.copyOf(has, cap);
            best = Arrays.copyOf(best, cap);
            bestBytes = Arrays.copyOf(bestBytes, cap);
        }

        @Override void accept(int g, FieldVector v, int row) {
            ensure(g);
            if (v.isNull(row)) return;
            if (v instanceof VarCharVector s) {
                byte[] b = s.get(row);
                if (!has[g] || (wantMax ? Arrays.compareUnsigned(b, bestBytes[g]) > 0 : Arrays.compareUnsigned(b, bestBytes[g]) < 0)) {
                    bestBytes[g] = b;
                    has[g] = true;
                }
                return;
            }
            long x = FoldQuery.readLong(v, row);
            if (!has[g] || (wantMax ? x > best[g] : x < best[g])) {
                best[g] = x;
                has[g] = true;
            }
        }

        @Override FieldVector emit(String name, int groups, BufferAllocator a) {
            FieldVector out = new Field(name, src.getFieldType(), src.getChildren()).createVector(a);
            out.allocateNew();
            for (int g = 0; g < groups; g++) {
                if (g >= has.length || !has[g]) continue; // stays null
                if (out instanceof VarCharVector s) s.setSafe(g, bestBytes[g]);
                else if (out instanceof DateDayVector d) d.setSafe(g, (int) best[g]);
                else if (out instanceof DateMilliVector d) d.setSafe(g, best[g]);
                else if (out instanceof TimeStampVector t) t.setSafe(g, best[g]);
                else if (out instanceof BitVector b) b.setSafe(g, (int) best[g]);
                else throw new ParquetSqlException("MIN/MAX is not supported on vector type " + out.getClass().getSimpleName());
            }
            out.setValueCount(groups);
            return out;
        }
    }
}
