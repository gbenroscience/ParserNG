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
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * {@code COALESCE(a, b, ...)} and {@code NULLIF(a, b)} over numeric columns and numeric literals.
 *
 * <p><b>Why not in the engine.</b> ParserNG evaluates doubles and has no notion of null; the engine
 * propagates a null by validity bit around it. Null-aware functions therefore cannot be written as
 * ParserNG text. Each call is instead evaluated here, per batch, over the <em>widened</em> Float8
 * columns (nulls preserved), into a derived Float8 column that the engine reads like any other
 * column. The statement is rewritten so the call text is replaced by that column's name, and the
 * original text stays the output name.
 *
 * <p><b>Accepted arguments.</b> A numeric column, a numeric literal, {@code NULL}, or a nested
 * {@code COALESCE}/{@code NULLIF}. Arithmetic or a string, date or boolean column as an argument fails
 * with a named {@link ParquetSqlException}: {@code COALESCE(a + b, 0)} is not supported, write it as
 * {@code COALESCE(a, 0) + COALESCE(b, 0)} if that is what is meant, or filter first.
 *
 * <p><b>Semantics.</b> {@code COALESCE} returns the first non-null argument, null if all are null.
 * {@code NULLIF(a, b)} is null when {@code a} is null or {@code a = b}, else {@code a}; a null
 * {@code b} never equals {@code a}. Comparison is by double value ({@code -0.0 = 0.0}). The result is
 * always {@code Float8}.
 *
 * <p>Not supported as the direct operand of {@code IS [NOT] NULL}, {@code LIKE}, {@code REGEXP} or a
 * boolean test (a named error). Under {@code OR} a predicate over a derived column follows the
 * engine's null propagation, like any numeric expression leaf.
 *
 * <p>Calls inside a call's own argument list are evaluated by one tree, so nesting costs no extra
 * column. Identical calls in one statement share one derived column.
 */
final class ScalarFns {

    private ScalarFns() { }

    /** The rewritten statement and the derived-column kernels it needs ({@code stmt} is the input if nothing was rewritten). */
    record Result(SelectStatement stmt, List<NativeLeaves.Mask> masks) { }

    /** @return true if {@code text} contains a {@code COALESCE(} or {@code NULLIF(} call outside string literals */
    static boolean containsCall(String text) {
        return text != null && new Scanner(text).findCall(0) != null;
    }

    static Result rewrite(SelectStatement st, TableSchema schema) {
        Ctx ctx = new Ctx(schema);
        UnaryOperator<String> f = ctx::rewriteText;

        List<SelectItem> items = new ArrayList<>(st.items().size());
        for (SelectItem it : st.items()) {
            if (it.isAggregate()) {
                AggregateSpec a = it.aggregate();
                if (a.star()) {
                    items.add(it);
                    continue;
                }
                String arg = f.apply(a.argExprText());
                if (arg.equals(a.argExprText())) {
                    items.add(it);
                } else {
                    items.add(SelectItem.aggregate(new AggregateSpec(a.func(), arg, false, a.distinct()), it.outputName()));
                }
            } else {
                String t = f.apply(it.exprText());
                items.add(t.equals(it.exprText()) ? it : SelectItem.plain(t, it.outputName()));
            }
        }
        BoolExpr where = st.where() == null ? null : mapBool(st.where(), f);
        BoolExpr having = st.having() == null ? null : mapBool(st.having(), f);
        List<String> groupBy = new ArrayList<>(st.groupBy().size());
        for (String g : st.groupBy()) groupBy.add(f.apply(g));
        List<OrderItem> orderBy = new ArrayList<>(st.orderBy().size());
        for (OrderItem o : st.orderBy()) orderBy.add(new OrderItem(f.apply(o.exprText()), o.descending()));

        if (!ctx.changed) return new Result(st, List.of());
        SelectStatement out = new SelectStatement(st.selectAll(), items, st.table(), where, groupBy, having, orderBy,
                st.limit(), st.tableKind(), st.tablePaths(), st.distinct(), st.offset());
        return new Result(out, List.copyOf(ctx.masks));
    }

    private static BoolExpr mapBool(BoolExpr e, UnaryOperator<String> f) {
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
            case IsNullExpr n -> reject(n, n.target(), "IS [NOT] NULL");
            case LikeExpr l -> reject(l, l.target(), "LIKE/ILIKE");
            case RegexExpr r -> reject(r, r.target(), "REGEXP/RLIKE");
            case TruthExpr t -> reject(t, t.target(), "a boolean test");
        };
    }

    private static BoolExpr reject(BoolExpr e, String target, String what) {
        if (containsCall(target)) {
            throw new ParquetSqlException("COALESCE/NULLIF cannot be the operand of " + what + " (" + target
                    + "); compare it with a numeric value instead");
        }
        return e;
    }

    // ------------------------------------------------------------------ text scanning

    /** One call found in a text: {@code [start, end)} spans {@code NAME(...)} including the closing paren. */
    private record Call(String func, int start, int end, List<String> args) { }

    private static final class Scanner {
        final String t;

        Scanner(String text) {
            this.t = text;
        }

        /** The first {@code COALESCE(}/{@code NULLIF(} call at or after {@code from}, outside string literals. */
        Call findCall(int from) {
            int n = t.length(), i = from;
            while (i < n) {
                char c = t.charAt(i);
                if (c == '\'' || c == '"') {
                    i = skipString(i, c);
                } else if (Character.isLetter(c) || c == '_') {
                    int s = i++;
                    while (i < n && (Character.isLetterOrDigit(t.charAt(i)) || t.charAt(i) == '_')) i++;
                    boolean prevIdentChar = s > 0 && (t.charAt(s - 1) == '.' );
                    String word = t.substring(s, i);
                    if (!prevIdentChar && (word.equalsIgnoreCase("COALESCE") || word.equalsIgnoreCase("NULLIF"))) {
                        int j = i;
                        while (j < n && Character.isWhitespace(t.charAt(j))) j++;
                        if (j < n && t.charAt(j) == '(') {
                            int close = matching(j);
                            if (close < 0) {
                                throw new ParquetSqlException(word.toUpperCase() + " call is missing its closing parenthesis in: " + t);
                            }
                            return new Call(word.toUpperCase(), s, close + 1, splitArgs(j + 1, close));
                        }
                    }
                } else if (Character.isDigit(c)) {
                    while (i < n && (Character.isLetterOrDigit(t.charAt(i)) || t.charAt(i) == '.')) i++;
                } else {
                    i++;
                }
            }
            return null;
        }

        private int skipString(int start, char q) {
            int i = start + 1, n = t.length();
            while (i < n) {
                if (t.charAt(i) == q) {
                    if (i + 1 < n && t.charAt(i + 1) == q) { i += 2; continue; }
                    return i + 1;
                }
                i++;
            }
            return n;
        }

        /** Index of the paren closing the one at {@code open}, or -1. */
        private int matching(int open) {
            int depth = 0, n = t.length();
            for (int i = open; i < n; i++) {
                char c = t.charAt(i);
                if (c == '\'' || c == '"') {
                    i = skipString(i, c) - 1;
                } else if (c == '(') {
                    depth++;
                } else if (c == ')' && --depth == 0) {
                    return i;
                }
            }
            return -1;
        }

        private List<String> splitArgs(int from, int to) {
            List<String> out = new ArrayList<>();
            int depth = 0, start = from;
            for (int i = from; i < to; i++) {
                char c = t.charAt(i);
                if (c == '\'' || c == '"') {
                    i = skipString(i, c) - 1;
                } else if (c == '(') {
                    depth++;
                } else if (c == ')') {
                    depth--;
                } else if (c == ',' && depth == 0) {
                    out.add(t.substring(start, i).trim());
                    start = i + 1;
                }
            }
            out.add(t.substring(start, to).trim());
            return out;
        }
    }

    // ------------------------------------------------------------------ rewriting context

    private static final class Ctx {
        final TableSchema schema;
        final List<NativeLeaves.Mask> masks = new ArrayList<>();
        final Map<String, String> byText = new LinkedHashMap<>();
        boolean changed;

        Ctx(TableSchema schema) {
            this.schema = schema;
        }

        String rewriteText(String text) {
            if (text == null) return null;
            Scanner sc = new Scanner(text);
            StringBuilder out = null;
            int from = 0, copied = 0;
            Call c;
            while ((c = sc.findCall(from)) != null) {
                if (out == null) out = new StringBuilder(text.length());
                out.append(text, copied, c.start());
                out.append(nameFor(text.substring(c.start(), c.end()), c));
                copied = from = c.end();
            }
            if (out == null) return text;
            changed = true;
            out.append(text, copied, text.length());
            return out.toString();
        }

        private String nameFor(String callText, Call c) {
            String key = callText.replaceAll("\\s+", " ");
            String known = byText.get(key);
            if (known != null) return known;
            Node root = node(c);
            Set<String> cols = new LinkedHashSet<>();
            root.columns(cols);
            // Same name shape as every other engine-visible synthetic column; the WHERE rewriter continues the numbering.
            String name = NativeLeaves.maskName(masks.size());
            masks.add(new ScalarMask(name, callText, root, new ArrayList<>(cols)));
            byText.put(key, name);
            return name;
        }

        private Node node(Call c) {
            if (c.func().equals("NULLIF")) {
                if (c.args().size() != 2) {
                    throw new ParquetSqlException("NULLIF takes exactly 2 arguments, got " + c.args().size());
                }
                return new NullIf(arg(c, c.args().get(0)), arg(c, c.args().get(1)));
            }
            if (c.args().isEmpty() || c.args().get(0).isEmpty()) {
                throw new ParquetSqlException("COALESCE needs at least one argument");
            }
            List<Node> args = new ArrayList<>();
            for (String a : c.args()) args.add(arg(c, a));
            return new Coalesce(args.toArray(new Node[0]));
        }

        private Node arg(Call parent, String a) {
            String t = a.trim();
            Call nested = new Scanner(t).findCall(0);
            if (nested != null && nested.start() == 0 && nested.end() == t.length()) return node(nested);
            if (t.equalsIgnoreCase("NULL")) return new Const(Double.NaN, true);
            ColumnKind k = schema.kind(t);
            if (k != null) {
                if (!k.numeric()) {
                    throw new ParquetSqlException(parent.func() + " argument '" + t + "' is " + k
                            + "; only numeric columns and numeric literals are supported");
                }
                return new Col(t);
            }
            Double lit = literal(t);
            if (lit != null) return new Const(lit, false);
            throw new ParquetSqlException(parent.func() + " argument '" + t + "' is not a numeric column, a numeric literal, NULL or a nested"
                    + " COALESCE/NULLIF (expressions as arguments are not supported)");
        }

        private static Double literal(String t) {
            String s = t.trim();
            boolean neg = false;
            if (s.startsWith("-") || s.startsWith("+")) {
                neg = s.charAt(0) == '-';
                s = s.substring(1).trim();
            }
            if (s.isEmpty() || !(Character.isDigit(s.charAt(0)) || s.charAt(0) == '.')) return null;
            try {
                double d = Double.parseDouble(s);
                return neg ? -d : d;
            } catch (NumberFormatException e) {
                return null;
            }
        }
    }

    // ------------------------------------------------------------------ evaluation tree

    /** One node of an argument tree. {@code eval} returns false for null and leaves the value in {@code val}. */
    private abstract static class Node {
        double val;

        abstract boolean eval(int i);

        void bind(VectorSchemaRoot widened) { }

        void columns(Set<String> out) { }
    }

    private static final class Const extends Node {
        private final boolean isNull;

        Const(double v, boolean isNull) {
            this.val = v;
            this.isNull = isNull;
        }

        @Override boolean eval(int i) { return !isNull; }
    }

    private static final class Col extends Node {
        private final String name;
        private Float8Vector v;

        Col(String name) {
            this.name = name;
        }

        @Override void bind(VectorSchemaRoot widened) {
            v = (Float8Vector) widened.getVector(name);
            if (v == null) throw new ParquetSqlException("Column '" + name + "' is missing from the widened columns");
        }

        @Override void columns(Set<String> out) { out.add(name); }

        @Override boolean eval(int i) {
            if (v.isNull(i)) return false;
            val = v.get(i);
            return true;
        }
    }

    private static final class Coalesce extends Node {
        private final Node[] args;

        Coalesce(Node[] args) {
            this.args = args;
        }

        @Override void bind(VectorSchemaRoot w) { for (Node a : args) a.bind(w); }

        @Override void columns(Set<String> out) { for (Node a : args) a.columns(out); }

        @Override boolean eval(int i) {
            for (Node a : args) {
                if (a.eval(i)) {
                    val = a.val;
                    return true;
                }
            }
            return false;
        }
    }

    private static final class NullIf extends Node {
        private final Node a, b;

        NullIf(Node a, Node b) {
            this.a = a;
            this.b = b;
        }

        @Override void bind(VectorSchemaRoot w) { a.bind(w); b.bind(w); }

        @Override void columns(Set<String> out) { a.columns(out); b.columns(out); }

        @Override boolean eval(int i) {
            if (!a.eval(i)) return false;
            double x = a.val;
            if (b.eval(i) && b.val == x) return false;
            val = x;
            return true;
        }
    }

    /** The derived column: one {@link Node} tree evaluated over the widened Float8 columns, nulls kept. */
    static final class ScalarMask extends NativeLeaves.Mask {
        private final Node root;
        private final List<String> cols;

        ScalarMask(String name, String text, Node root, List<String> cols) {
            super(name, text + " (derived Float8 column)");
            this.root = root;
            this.cols = List.copyOf(cols);
        }

        @Override List<String> widenedColumns() {
            return cols;
        }

        @Override void eval(VectorSchemaRoot widened, int n, Float8Vector out) {
            root.bind(widened);
            for (int i = 0; i < n; i++) {
                if (root.eval(i)) out.set(i, root.val);
                else out.setNull(i);
            }
        }
    }
}
