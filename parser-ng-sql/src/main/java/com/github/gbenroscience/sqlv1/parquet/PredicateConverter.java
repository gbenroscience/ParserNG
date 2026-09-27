package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.*;
import com.github.gbenroscience.sqlv1.ast.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Derives a Parquet <b>pruning</b> predicate from a (negation-normal-form) WHERE tree.
 *
 * <p>Only shapes of the form {@code <bare file column> <op> <literal>} (either order), BETWEEN and
 * IN over literals, and IS [NOT] NULL on a bare column are converted. Anything else (function calls,
 * arithmetic, aliases, column-vs-column) yields {@code null} for that node. Combination rules keep
 * the result a sound superset filter: an AND keeps whichever side converts; an OR converts only if
 * both sides do.
 *
 * <p>The result is used ONLY to skip row groups. The original WHERE is always still evaluated by
 * ArrowQuery, so a conservative or partial conversion can never change results.
 */
final class PredicateConverter {

    private PredicateConverter() { }

    private static final Pattern IDENT = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
    private static final Pattern INT = Pattern.compile("[+-]?\\d+");
    private static final Pattern DEC = Pattern.compile("[+-]?(\\d+\\.\\d*|\\.\\d+|\\d+)([eE][+-]?\\d+)?");

    static Predicate convert(BoolExpr e, Set<String> fileColumns) {
        if (e == null) return null;
        return switch (e) {
            case AndExpr(var l, var r) -> {
                Predicate a = convert(l, fileColumns), b = convert(r, fileColumns);
                yield a == null ? b : b == null ? a : Predicate.and(a, b);
            }
            case OrExpr(var l, var r) -> {
                Predicate a = convert(l, fileColumns), b = convert(r, fileColumns);
                yield (a == null || b == null) ? null : Predicate.or(a, b);
            }
            case NotExpr n -> null; // absent after toNnf; never guess
            case ComparisonExpr(var l, var op, var r) -> comparison(l, op, r, fileColumns);
            case BetweenExpr(var t, var lo, var hi, var neg) -> between(t, lo, hi, neg, fileColumns);
            case InExpr(var t, var vs, var neg) -> in(t, vs, neg, fileColumns);
            case IsNullExpr(var t, var neg) -> {
                String c = column(t, fileColumns);
                yield c == null ? null : (neg ? Predicate.isNotNull(c) : Predicate.isNull(c));
            }
        };
    }

    private static Predicate comparison(String l, CompOp op, String r, Set<String> cols) {
        String lc = column(l, cols), rc = column(r, cols);
        if (lc != null && rc == null) return cmp(lc, op, literal(r));
        if (rc != null && lc == null) return cmp(rc, flip(op), literal(l));
        return null; // column-vs-column or no bare column
    }

    private static Predicate between(String t, String lo, String hi, boolean neg, Set<String> cols) {
        String c = column(t, cols);
        if (c == null) return null;
        Object l = literal(lo), h = literal(hi);
        if (!neg) { // col >= lo AND col <= hi: either half alone is still sound
            Predicate a = cmp(c, CompOp.GE, l), b = cmp(c, CompOp.LE, h);
            return a == null ? b : b == null ? a : Predicate.and(a, b);
        }
        // col < lo OR col > hi: needs both halves
        Predicate a = cmp(c, CompOp.LT, l), b = cmp(c, CompOp.GT, h);
        return (a == null || b == null) ? null : Predicate.or(a, b);
    }

    private static Predicate in(String t, List<String> vs, boolean neg, Set<String> cols) {
        String c = column(t, cols);
        if (c == null) return null;
        List<Object> lits = new ArrayList<>(vs.size());
        for (String v : vs) {
            Object o = literal(v);
            if (o == null) {
                if (!neg) return null; // OR of equalities: every member must be understood
                continue;              // AND of inequalities: skipping a member is still sound
            }
            lits.add(o);
        }
        if (lits.isEmpty()) return null;
        if (!neg) return Predicate.in(c, lits.toArray());
        Predicate acc = null;
        for (Object o : lits) {
            Predicate ne = Predicate.ne(c, o);
            acc = acc == null ? ne : Predicate.and(acc, ne);
        }
        return acc;
    }

    private static Predicate cmp(String col, CompOp op, Object lit) {
        if (lit == null) return null;
        return switch (op) {
            case EQ -> Predicate.eq(col, lit);
            case NEQ -> Predicate.ne(col, lit);
            case LT -> Predicate.lt(col, lit);
            case LE -> Predicate.le(col, lit);
            case GT -> Predicate.gt(col, lit);
            case GE -> Predicate.ge(col, lit);
        };
    }

    private static CompOp flip(CompOp op) {
        return switch (op) {
            case LT -> CompOp.GT;
            case LE -> CompOp.GE;
            case GT -> CompOp.LT;
            case GE -> CompOp.LE;
            default -> op;
        };
    }

    /** @return the trimmed text if it is exactly one identifier naming a real file column, else null. */
    static String column(String text, Set<String> cols) {
        if (text == null) return null;
        String t = text.trim();
        return (IDENT.matcher(t).matches() && cols.contains(t)) ? t : null;
    }

    /**
     * @return Integer, Long, Double or String for a plain literal; null for anything else (including
     * integers overflowing long). Type choice is by spelling only; the Parquet layer refuses to push
     * a literal whose type does not fit the column, so a mismatch just means no pruning.
     */
    static Object literal(String text) {
        if (text == null) return null;
        String t = text.trim();
        if (t.isEmpty()) return null;
        char q = t.charAt(0);
        if (q == '\'' || q == '"') return stringLiteral(t, q);
        try {
            if (INT.matcher(t).matches()) {
                long v = Long.parseLong(t.startsWith("+") ? t.substring(1) : t);
                return (v >= Integer.MIN_VALUE && v <= Integer.MAX_VALUE) ? (Object) (int) v : (Object) v;
            }
            if (DEC.matcher(t).matches()) {
                double d = Double.parseDouble(t);
                return (Double.isNaN(d) || Double.isInfinite(d)) ? null : (Object) d;
            }
        } catch (NumberFormatException ex) {
            return null;
        }
        return null;
    }

    private static String stringLiteral(String t, char q) {
        StringBuilder sb = new StringBuilder();
        int n = t.length();
        for (int i = 1; i < n; i++) {
            char c = t.charAt(i);
            if (c == q) {
                if (i + 1 < n && t.charAt(i + 1) == q) { sb.append(q); i++; continue; }
                return i == n - 1 ? sb.toString() : null; // closing quote must be the last char
            }
            sb.append(c);
        }
        return null; // unterminated
    }
}
