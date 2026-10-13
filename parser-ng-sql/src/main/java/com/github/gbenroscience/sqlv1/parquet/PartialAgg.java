package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.QuotedIdentifiers;
import com.github.gbenroscience.sqlv1.SqlParser;
import com.github.gbenroscience.sqlv1.SqlSyntaxException;
import com.github.gbenroscience.sqlv1.ast.AggFunc;
import com.github.gbenroscience.sqlv1.ast.AggregateSpec;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Streaming aggregation for the queries the numeric engine can answer: {@code COUNT}, {@code SUM}, {@code AVG},
 * {@code MIN} and {@code MAX}, with or without {@code GROUP BY}, {@code HAVING}, {@code ORDER BY} and
 * {@code LIMIT}.
 *
 * <p>The engine used to see the whole table at once: every scanned batch was copied into one root, every column
 * widened to doubles, and the engine run a single time. Now the engine runs on <em>each batch</em> and produces
 * partial aggregates (a count; a sum, minimum or maximum with the count of values behind it), which this class
 * merges ({@link MergeCore}) and turns into the final result. Memory is one batch plus one row per group, and
 * nothing is copied but the partial rows. {@code AVG} is merged as a sum over a count; {@code HAVING},
 * {@code ORDER BY}, {@code LIMIT} and {@code OFFSET} are applied to the merged groups, since they cannot be applied
 * per batch.
 *
 * <p>{@link #tryCreate} returns {@code null} for anything it does not handle exactly (a {@code DISTINCT}, an
 * aggregate other than the five, a {@code HAVING} or {@code ORDER BY} operand that is not an output, ...) and the
 * plan then uses the materializing path as before. Differences from that path, all in line with SQL: an
 * {@code AVG}, {@code SUM}, {@code MIN} or {@code MAX} over no values is null, a {@code HAVING} over a null is
 * not true, and the groups come out in first-seen order unless {@code ORDER BY} says otherwise.
 */
final class PartialAgg {

    private static final class Decline extends RuntimeException {
        Decline(String why) {
            super(why, null, false, false);
        }
    }

    private record AggDef(AggFunc func, String arg) { }

    private record Slot(MergeCore.Kind kind, AggFunc func, String arg, String name) { }

    private record Col(String name, int key, int agg) { }

    private static final Pattern NUMBER = Pattern.compile("[+-]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?");

    private final SelectStatement stmt;
    private final List<String> keyTexts = new ArrayList<>();
    private final List<String> keyNames = new ArrayList<>();
    private final List<Col> cols = new ArrayList<>();
    private final List<AggDef> aggs = new ArrayList<>();
    private final List<Slot> slots = new ArrayList<>();
    private int[] valSlot, cntSlot;
    private MergeCore.Kind[] kinds;
    private HavingEval.Node having;
    private final boolean global;

    /** @return the plan, or {@code null} if this statement must take the materializing path */
    static PartialAgg tryCreate(SelectStatement st) {
        if (st.selectAll() || st.distinct()) return null;
        try {
            return new PartialAgg(st);
        } catch (Decline | ParquetSqlException notExact) {
            return null;
        }
    }

    private PartialAgg(SelectStatement st) {
        this.stmt = st;
        this.global = st.groupBy().isEmpty();
        for (String g : st.groupBy()) {
            String t = resolveAlias(g).trim();
            if (!keyTexts.contains(t)) {
                keyTexts.add(t);
                keyNames.add("pngpartk" + NativeLeaves.maskName(keyTexts.size() - 1).substring("pngmask".length()));
            }
        }
        for (SelectItem it : st.items()) {
            if (it.isAggregate()) {
                AggregateSpec a = it.aggregate();
                if (a.distinct()) throw new Decline("distinct");
                cols.add(new Col(it.outputName(), -1, aggIndex(a.func(), a.star() ? null : a.argExprText().trim())));
            } else {
                int k = keyTexts.indexOf(it.exprText().trim());
                if (k < 0) throw new Decline("select item is not a group key");
                cols.add(new Col(it.outputName(), k, -1));
            }
        }
        if (st.having() != null) having = HavingEval.compile(st.having(), this::havingOperand);
        List<String> names = new ArrayList<>();
        for (Col c : cols) names.add(c.name());
        for (OrderItem o : st.orderBy()) {
            if (!names.contains(QuotedIdentifiers.decodeAll(o.exprText().trim()))) throw new Decline("order by is not an output");
        }
        buildSlots();
    }

    /** A {@code GROUP BY} item that is the alias of a select item stands for that item's expression. */
    private String resolveAlias(String g) {
        String t = g.trim();
        for (SelectItem it : stmt.items()) {
            if (!it.isAggregate() && it.alias() != null && it.alias().equals(t) && !it.exprText().trim().equals(t)) return it.exprText();
        }
        return t;
    }

    private int aggIndex(AggFunc func, String arg) {
        switch (func) {
            case COUNT, SUM, AVG, MIN, MAX -> { }
            default -> throw new Decline("aggregate " + func);
        }
        AggDef d = new AggDef(func, arg);
        int i = aggs.indexOf(d);
        if (i >= 0) return i;
        if (arg == null && func != AggFunc.COUNT) throw new Decline("star argument");
        aggs.add(d);
        return aggs.size() - 1;
    }

    private HavingEval.Num havingOperand(String text) {
        String t = text.trim().replaceAll("^([+-])\\s+", "$1");
        if (NUMBER.matcher(t).matches()) {
            Double c = Double.valueOf(t);
            return g -> c;
        }
        String readable = QuotedIdentifiers.decodeAll(t);
        for (Col c : cols) if (c.agg() >= 0 && c.name().equals(readable)) return agg(c.agg());
        try {
            SelectStatement s = SqlParser.parse("SELECT " + t + " FROM t");
            if (s.items().size() == 1 && s.items().get(0).isAggregate() && !s.items().get(0).aggregate().distinct()) {
                AggregateSpec a = s.items().get(0).aggregate();
                return agg(aggIndex(a.func(), a.star() ? null : a.argExprText().trim()));
            }
        } catch (SqlSyntaxException notAnAggregate) {
            // fall through
        }
        throw new Decline("having operand " + readable);
    }

    private HavingEval.Num agg(int a) {
        return g -> core == null ? null : core.number(a, g);
    }

    /** Set while a merge is running so HAVING operands can read the finished groups. */
    private MergeCore core;

    private int slot(MergeCore.Kind kind, AggFunc func, String arg) {
        for (int i = 0; i < slots.size(); i++) {
            Slot s = slots.get(i);
            if (s.kind() == kind && s.func() == func && java.util.Objects.equals(s.arg(), arg)) return i;
        }
        slots.add(new Slot(kind, func, arg, "pngparta" + NativeLeaves.maskName(slots.size()).substring("pngmask".length())));
        return slots.size() - 1;
    }

    private void buildSlots() {
        valSlot = new int[aggs.size()];
        cntSlot = new int[aggs.size()];
        kinds = new MergeCore.Kind[aggs.size()];
        for (int a = 0; a < aggs.size(); a++) {
            AggDef d = aggs.get(a);
            switch (d.func()) {
                case COUNT -> {
                    kinds[a] = MergeCore.Kind.COUNT;
                    valSlot[a] = slot(MergeCore.Kind.COUNT, AggFunc.COUNT, d.arg());
                    cntSlot[a] = -1;
                }
                default -> {
                    kinds[a] = MergeCore.Kind.valueOf(d.func().name());
                    // SUM and AVG both need the sum; MIN and MAX their extreme; all need the count of values
                    AggFunc partial = d.func() == AggFunc.AVG ? AggFunc.SUM : d.func();
                    valSlot[a] = slot(MergeCore.Kind.valueOf(partial.name()), partial, d.arg());
                    cntSlot[a] = slot(MergeCore.Kind.COUNT, AggFunc.COUNT, d.arg());
                }
            }
        }
    }

    /** The per-batch statement the engine runs: group keys then partial aggregates, with the given {@code WHERE}. */
    SelectStatement engineStatement(BoolExpr engineWhere) {
        List<SelectItem> items = new ArrayList<>();
        for (int k = 0; k < keyTexts.size(); k++) items.add(SelectItem.plain(keyTexts.get(k), keyNames.get(k)));
        for (Slot s : slots) {
            items.add(SelectItem.aggregate(new AggregateSpec(s.func(), s.arg(), s.arg() == null), s.name()));
        }
        return new SelectStatement(false, items, stmt.table(), engineWhere, List.copyOf(keyTexts), null, List.of(), null,
                stmt.tableKind(), stmt.tablePaths(), false, 0);
    }

    Merger newMerger(long maxGroups) {
        return new Merger(maxGroups);
    }

    /** Folds the engine's per-batch partial results; {@link #finish} builds the answer. */
    final class Merger {
        private final long maxGroups;
        private final MergeCore mc = new MergeCore(kinds, valSlot, cntSlot);
        private final Map<List<Object>, Integer> ids = new HashMap<>();
        private FieldVector[] keyVecs;
        private Field[] keyFields;
        private final double[] vals = new double[slots.size()];
        private final boolean[] nulls = new boolean[slots.size()];
        private BufferAllocator alloc;

        Merger(long maxGroups) {
            this.maxGroups = maxGroups;
            if (global) mc.addGroup(); // a global aggregate always has its one row, even over no input
            core = mc;
        }

        /** Merges one batch's partial rows. {@code alloc} owns the key copies. */
        void accept(VectorSchemaRoot partial, BufferAllocator allocator) {
            this.alloc = allocator;
            int n = partial.getRowCount();
            int nk = keyTexts.size();
            FieldVector[] kv = new FieldVector[nk];
            for (int k = 0; k < nk; k++) {
                kv[k] = partial.getVector(keyNames.get(k));
                if (kv[k] == null) throw new ParquetSqlException("The engine's partial result is missing group key " + keyTexts.get(k));
            }
            FieldVector[] sv = new FieldVector[slots.size()];
            for (int s = 0; s < sv.length; s++) {
                sv[s] = partial.getVector(slots.get(s).name());
                if (sv[s] == null) throw new ParquetSqlException("The engine's partial result is missing " + slots.get(s).name());
            }
            if (nk > 0 && keyVecs == null) {
                keyVecs = new FieldVector[nk];
                keyFields = new Field[nk];
                for (int k = 0; k < nk; k++) {
                    Field f = kv[k].getField();
                    keyFields[k] = f;
                    keyVecs[k] = new Field("key" + k, f.getFieldType(), f.getChildren()).createVector(alloc);
                    keyVecs[k].allocateNew();
                }
            }
            for (int r = 0; r < n; r++) {
                int g;
                if (global) {
                    g = 0;
                } else {
                    Object[] key = new Object[nk];
                    for (int k = 0; k < nk; k++) key[k] = FoldQuery.norm(kv[k].getObject(r));
                    List<Object> kl = Arrays.asList(key);
                    Integer id = ids.get(kl);
                    if (id == null) {
                        if (mc.groups() >= maxGroups) {
                            throw new ParquetSqlException("GROUP BY produced more than maxMaterializedRows (" + maxGroups + ") groups; add a WHERE or raise the cap");
                        }
                        id = mc.addGroup();
                        ids.put(kl, id);
                        for (int k = 0; k < nk; k++) keyVecs[k].copyFromSafe(r, id, kv[k]);
                    }
                    g = id;
                }
                for (int s = 0; s < sv.length; s++) {
                    FieldVector v = sv[s];
                    boolean isNull = v.isNull(r);
                    nulls[s] = isNull;
                    vals[s] = isNull ? 0.0 : (v instanceof Float8Vector f ? f.get(r) : ((Number) v.getObject(r)).doubleValue());
                }
                mc.mergeRow(g, vals, nulls);
            }
        }

        /** The final result: HAVING, ORDER BY, LIMIT/OFFSET applied; the caller closes it. */
        VectorSchemaRoot finish(BufferAllocator allocator, int want) {
            int g = mc.groups();
            List<FieldVector> toClose = new ArrayList<>();
            try {
                FieldVector[] full = new FieldVector[cols.size()];
                String[] names = new String[cols.size()];
                for (int ci = 0; ci < full.length; ci++) {
                    Col c = cols.get(ci);
                    names[ci] = c.name();
                    if (c.key() >= 0) {
                        Field f = keyFields != null ? keyFields[c.key()] : doubleField();
                        full[ci] = new Field(c.name(), f.getFieldType(), f.getChildren()).createVector(allocator);
                        toClose.add(full[ci]);
                        full[ci].allocateNew();
                        if (keyVecs != null) for (int i = 0; i < g; i++) full[ci].copyFromSafe(i, i, keyVecs[c.key()]);
                        full[ci].setValueCount(g);
                    } else {
                        Float8Vector v = new Float8Vector(c.name(), allocator);
                        toClose.add(v);
                        v.allocateNew(Math.max(g, 1));
                        for (int i = 0; i < g; i++) {
                            Double x = mc.number(c.agg(), i);
                            if (x == null) v.setNull(i);
                            else v.set(i, x);
                        }
                        v.setValueCount(g);
                        full[ci] = v;
                    }
                }
                int[] idx = new int[g];
                int kept = 0;
                for (int i = 0; i < g; i++) {
                    if (having != null && !Boolean.TRUE.equals(having.eval(i))) continue;
                    idx[kept++] = i;
                }
                idx = Arrays.copyOf(idx, kept);
                if (!stmt.orderBy().isEmpty()) idx = ResultOps.sort(full, names, stmt.orderBy(), idx);
                if (idx.length > want) idx = Arrays.copyOf(idx, want);
                return ResultOps.gather(allocator, full, idx);
            } finally {
                for (FieldVector v : toClose) v.close();
                close();
            }
        }

        private Field doubleField() {
            return new Field("key", org.apache.arrow.vector.types.pojo.FieldType.nullable(
                    new org.apache.arrow.vector.types.pojo.ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE)), null);
        }

        void close() {
            if (keyVecs != null) {
                for (FieldVector v : keyVecs) if (v != null) v.close();
                keyVecs = null;
            }
            core = null;
        }
    }
}
