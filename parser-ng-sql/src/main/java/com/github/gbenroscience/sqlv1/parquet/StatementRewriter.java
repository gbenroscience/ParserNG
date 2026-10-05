package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ast.AggregateSpec;
import com.github.gbenroscience.sqlv1.ast.AndExpr;
import com.github.gbenroscience.sqlv1.ast.BetweenExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.ComparisonExpr;
import com.github.gbenroscience.sqlv1.ast.InExpr;
import com.github.gbenroscience.sqlv1.ast.IsNullExpr;
import com.github.gbenroscience.sqlv1.ast.LikeExpr;
import com.github.gbenroscience.sqlv1.ast.NotExpr;
import com.github.gbenroscience.sqlv1.ast.OrExpr;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
import com.github.gbenroscience.sqlv1.ast.RegexExpr;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import com.github.gbenroscience.sqlv1.ast.TruthExpr;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Applies one text transformation to every expression of a statement (select items, aggregate
 * arguments, {@code WHERE}, {@code HAVING}, {@code GROUP BY}, {@code ORDER BY}), keeping each output name:
 * an item whose text changes keeps its original output name as an alias. Returns the same instance when
 * nothing changed. {@code LIKE}/{@code REGEXP} patterns are not expressions and are never passed through.
 */
final class StatementRewriter {

    private StatementRewriter() { }

    static SelectStatement map(SelectStatement st, UnaryOperator<String> f) {
        return map(st, f, f);
    }

    /**
     * @param value   applied to select-item expressions, where a value is produced
     * @param operand applied everywhere else (predicate operands, {@code GROUP BY}, {@code ORDER BY}, aggregate arguments)
     */
    static SelectStatement map(SelectStatement st, UnaryOperator<String> value, UnaryOperator<String> operand) {
        boolean[] changed = {false};
        UnaryOperator<String> g = wrap(operand, changed);
        UnaryOperator<String> v = wrap(value, changed);

        List<SelectItem> items = new ArrayList<>(st.items().size());
        for (SelectItem it : st.items()) {
            if (it.isAggregate()) {
                AggregateSpec a = it.aggregate();
                if (a.star()) {
                    items.add(it);
                    continue;
                }
                String arg = g.apply(a.argExprText());
                items.add(arg.equals(a.argExprText()) ? it
                        : SelectItem.aggregate(new AggregateSpec(a.func(), arg, false, a.distinct()), it.outputName()));
            } else {
                String t = v.apply(it.exprText());
                items.add(t.equals(it.exprText()) ? it : SelectItem.plain(t, it.outputName()));
            }
        }
        BoolExpr where = st.where() == null ? null : mapBool(st.where(), g);
        BoolExpr having = st.having() == null ? null : mapBool(st.having(), g);
        List<String> groupBy = new ArrayList<>(st.groupBy().size());
        for (String x : st.groupBy()) groupBy.add(g.apply(x));
        List<OrderItem> orderBy = new ArrayList<>(st.orderBy().size());
        for (OrderItem o : st.orderBy()) orderBy.add(new OrderItem(g.apply(o.exprText()), o.descending()));

        if (!changed[0]) return st;
        return new SelectStatement(st.selectAll(), items, st.table(), where, groupBy, having, orderBy,
                st.limit(), st.tableKind(), st.tablePaths(), st.distinct(), st.offset());
    }

    private static UnaryOperator<String> wrap(UnaryOperator<String> f, boolean[] changed) {
        return t -> {
            if (t == null) return null;
            String r = f.apply(t);
            if (!r.equals(t)) changed[0] = true;
            return r;
        };
    }

    static BoolExpr mapBool(BoolExpr e, UnaryOperator<String> f) {
        return switch (e) {
            case AndExpr(var l, var r) -> new AndExpr(mapBool(l, f), mapBool(r, f));
            case OrExpr(var l, var r) -> new OrExpr(mapBool(l, f), mapBool(r, f));
            case NotExpr(var inner) -> new NotExpr(mapBool(inner, f));
            case ComparisonExpr(var l, var op, var r) -> new ComparisonExpr(f.apply(l), op, f.apply(r));
            case BetweenExpr(var t, var lo, var hi, var neg) -> new BetweenExpr(f.apply(t), f.apply(lo), f.apply(hi), neg);
            case InExpr(var t, var vals, var neg) -> {
                List<String> mapped = new ArrayList<>(vals.size());
                for (String v : vals) mapped.add(f.apply(v));
                yield new InExpr(f.apply(t), mapped, neg);
            }
            case IsNullExpr(var t, var neg) -> new IsNullExpr(f.apply(t), neg);
            case LikeExpr(var t, var p, var esc, var ci, var neg) -> new LikeExpr(f.apply(t), p, esc, ci, neg);
            case RegexExpr(var t, var p, var neg) -> new RegexExpr(f.apply(t), p, neg);
            case TruthExpr(var t, var value, var neg) -> new TruthExpr(f.apply(t), value, neg);
        };
    }
}
