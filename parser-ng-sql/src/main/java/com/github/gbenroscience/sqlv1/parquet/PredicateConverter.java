package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import com.github.gbenroscience.sqlv1.ast.*;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Converts a WHERE tree (already in negation-normal form) into a
 * parser-ng-parquet {@link Predicate} used <em>only for row-group pruning</em>.
 *
 * <h2>Soundness</h2>
 * The result is always a superset filter: it may keep rows the WHERE rejects,
 * never drop rows it accepts. The full WHERE is re-evaluated downstream.
 * Anything not understood becomes "no constraint": dropped from an AND,
 * poisoning an OR.
 *
 * <h2>Literal typing</h2>
 * parser-ng-parquet silently skips a leaf whose literal Java type does not
 * match the column ({@code Float} for float32, {@code Integer} for int32,
 * {@code Double}/{@code Float}/{@code Integer} for double). So each literal
 * is promoted to the column's physical type here, following the semantics of
 * the numeric engine, which compares in {@code double}:
 * <ul>
 * <li>int8/16/32: integral literal in int range becomes {@code Integer}; a
 * fractional or out-of-range literal is not pushed.</li>
 * <li>int64: integral literal becomes {@code Long}. Beyond 2^53 the engine
 * sees the literal as a double, so the pushed value is that double's value.</li>
 * <li>float64: any numeric literal becomes {@code Double}.</li>
 * <li>float32: an exactly representable literal is pushed as {@code Float};
 * otherwise a range bound is rounded <em>outward</em> ({@code >} becomes
 * {@code >=} the next float below, and so on) and {@code =}/{@code <>} are
 * not pushed. Rounding to nearest would be unsound.</li>
 * <li>strings: pushed as {@code String}. DATE columns: pushed as {@code LocalDate}. DECIMAL: pushed
 * as a {@code BigDecimal} at the column scale when column and literal have at most 15 digits (see
 * {@link #bindDecimal}). Timestamp, boolean, unsigned and nested columns are not pushed (a timestamp needs a change in
 * parser-ng-parquet's translator; see the README type matrix). LIKE, REGEXP and IS TRUE/FALSE
 * are never pushed.</li>
 * </ul>
 */
final class PredicateConverter {

    private PredicateConverter() { }

    private static final Pattern IDENT = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
    private static final Pattern INT = Pattern.compile("[+-]?\\d+");
    private static final Pattern DEC = Pattern.compile("[+-]?(\\d+\\.\\d*|\\.\\d+|\\d+)([eE][+-]?\\d+)?");

    /** A numeric literal: its double value, and its exact long value if integral. */
    private record Num(double dbl, boolean integral, long lng) { }

    /** An operator plus the typed literal to push. */
    record Bound(CompOp op, Object value) { }

    /** Superset pruning predicate (see class comment). */
    static Predicate convert(BoolExpr e, TableSchema schema) {
        return convert(e, schema, false);
    }

    /**
     * Exact predicate: non-null only if the <em>whole</em> tree converts with
     * semantics identical to the SQL WHERE (no dropped branches, no outward
     * rounding). Only then may the scan's {@code exactFilter} replace the
     * downstream WHERE.
     */
    static Predicate convertExact(BoolExpr e, TableSchema schema) {
        return convert(e, schema, true);
    }

    private static Predicate convert(BoolExpr e, TableSchema schema, boolean strict) {
        if (e == null) return null;
        return switch (e) {
            case AndExpr(var l, var r) -> {
                Predicate a = convert(l, schema, strict), b = convert(r, schema, strict);
                if (strict && (a == null || b == null)) yield null;
                yield a == null ? b : b == null ? a : Predicate.and(a, b);
            }
            case OrExpr(var l, var r) -> {
                Predicate a = convert(l, schema, strict), b = convert(r, schema, strict);
                yield (a == null || b == null) ? null : Predicate.or(a, b);
            }
            case NotExpr n -> null; // absent after toNnf; never guess
            case ComparisonExpr(var l, var op, var r) -> comparison(l, op, r, schema, strict);
            case BetweenExpr(var t, var lo, var hi, var neg) -> between(t, lo, hi, neg, schema, strict);
            case InExpr(var t, var vs, var neg) -> in(t, vs, neg, schema, strict);
            case IsNullExpr(var t, var neg) -> {
                String c = column(t, schema);
                yield c == null ? null : (neg ? Predicate.isNotNull(c) : Predicate.isNull(c));
            }
            case LikeExpr l -> null;   // not pruned: no sound row-group bound is derived from a pattern
            case RegexExpr r -> null;
            case TruthExpr t -> null;
        };
    }

    private static Predicate comparison(String l, CompOp op, String r, TableSchema s, boolean strict) {
        String lc = column(l, s), rc = column(r, s);
        if (lc != null && rc == null) return cmp(lc, op, r, s, strict);
        if (rc != null && lc == null) return cmp(rc, flip(op), l, s, strict);
        return null; // column-vs-column or no bare column
    }

    private static Predicate between(String t, String lo, String hi, boolean neg, TableSchema s, boolean strict) {
        String c = column(t, s);
        if (c == null) return null;
        if (!neg) { // col >= lo AND col <= hi: either half alone is still sound
            Predicate a = cmp(c, CompOp.GE, lo, s, strict), b = cmp(c, CompOp.LE, hi, s, strict);
            if (strict && (a == null || b == null)) return null;
            return a == null ? b : b == null ? a : Predicate.and(a, b);
        }
        Predicate a = cmp(c, CompOp.LT, lo, s, strict), b = cmp(c, CompOp.GT, hi, s, strict);
        return (a == null || b == null) ? null : Predicate.or(a, b);
    }

    private static Predicate in(String t, List<String> vs, boolean neg, TableSchema s, boolean strict) {
        String c = column(t, s);
        if (c == null) return null;
        List<Predicate> parts = new ArrayList<>(vs.size());
        for (String v : vs) {
            Predicate p = cmp(c, neg ? CompOp.NEQ : CompOp.EQ, v, s, strict);
            if (p == null) {
                if (!neg || strict) return null; // OR of equalities: every member must be understood
                continue;              // AND of inequalities: skipping a member is still sound
            }
            parts.add(p);
        }
        if (parts.isEmpty()) return null;
        Predicate acc = parts.get(0);
        for (int i = 1; i < parts.size(); i++) acc = neg ? Predicate.and(acc, parts.get(i)) : Predicate.or(acc, parts.get(i));
        return acc;
    }

    private static Predicate cmp(String col, CompOp op, String literalText, TableSchema s, boolean strict) {
        Bound b = s.kind(col) == ColumnKind.DECIMAL
                ? bindDecimal(s.decimal(col), op, literalText, strict)
                : bind(s.kind(col), op, literalText, strict);
        if (b == null) return null;
        Object v = b.value();
        return switch (b.op()) {
            case EQ -> Predicate.eq(col, v);
            case NEQ -> Predicate.ne(col, v);
            case LT -> Predicate.lt(col, v);
            case LE -> Predicate.le(col, v);
            case GT -> Predicate.gt(col, v);
            case GE -> Predicate.ge(col, v);
        };
    }

    /** Promotes {@code literalText} to the column kind's pruning type, or returns null for "do not push". */
    static Bound bindForTest(ColumnKind k, CompOp op, String text) {
        return bind(k, op, text, false);
    }

    private static Bound bind(ColumnKind k, CompOp op, String text, boolean strict) {
        if (k == null) return null;
        if (k == ColumnKind.STRING) {
            if (strict) return null; // string semantics differ (UTF-8 vs UTF-16 order); exact mode is numeric only
            String str = stringLiteral(text);
            return str == null ? null : new Bound(op, str);
        }
        if (k == ColumnKind.DATE) {
            if (strict) return null;
            java.time.LocalDate d = date(text);
            return d == null ? null : new Bound(op, d);
        }
        Num n = number(text);
        if (n == null) return null;
        switch (k) {
            case INT8, INT16, INT32 -> {
                if (!n.integral() || n.lng() < Integer.MIN_VALUE || n.lng() > Integer.MAX_VALUE) return null;
                return new Bound(op, (int) n.lng());
            }
            case INT64 -> {
                if (!n.integral()) return null;
                long v = n.lng();
                if (v > 9007199254740992L || v < -9007199254740992L) {
                    if (strict) return null;
                    v = (long) (double) v; // the engine compares (double) v
                }
                return new Bound(op, v);
            }
            case FLOAT64 -> {
                return new Bound(op, n.dbl());
            }
            case FLOAT32 -> {
                return float32(op, n.dbl(), strict);
            }
            default -> {
                return null;
            }
        }
    }

    /** Digits a double holds without merging two distinct decimals (so double order equals decimal order). */
    private static final int DECIMAL_SAFE_DIGITS = 15;

    /**
     * DECIMAL pruning bound: a {@code BigDecimal} at the column's scale, or null for "do not push".
     * The engine compares decimals as doubles, so a bound is sent only when column and literal both
     * have at most 15 significant digits, where double order equals decimal order. A literal with
     * more fraction digits than the column scale is rounded outward for ranges ({@code >} becomes
     * {@code >=} the floor, {@code <} becomes {@code <=} the ceiling) and not pushed for
     * {@code =}/{@code <>}. Never pushed in strict (exactFilter) mode. parser-ng-parquet prunes
     * ranges only on INT32/INT64-backed decimals and {@code =}/{@code <>} on every backing, and drops
     * the rest itself, which is always sound.
     */
    private static Bound bindDecimal(int[] ps, CompOp op, String text, boolean strict) {
        if (strict || ps == null || ps[0] > DECIMAL_SAFE_DIGITS) return null;
        java.math.BigDecimal d = bigDecimal(text);
        if (d == null || d.precision() > DECIMAL_SAFE_DIGITS) return null;
        int scale = ps[1];
        try {
            return new Bound(op, d.setScale(scale, java.math.RoundingMode.UNNECESSARY));
        } catch (ArithmeticException inexact) {
            switch (op) {
                case GT, GE -> { return new Bound(CompOp.GE, d.setScale(scale, java.math.RoundingMode.FLOOR)); }
                case LT, LE -> { return new Bound(CompOp.LE, d.setScale(scale, java.math.RoundingMode.CEILING)); }
                default -> { return null; }
            }
        }
    }

    private static java.math.BigDecimal bigDecimal(String text) {
        if (text == null) return null;
        String t = text.trim();
        if (t.length() > 1 && (t.charAt(0) == '-' || t.charAt(0) == '+')) t = t.charAt(0) + t.substring(1).trim();
        if (t.startsWith("+")) t = t.substring(1);
        if (!DEC.matcher(t).matches()) return null;
        try {
            return new java.math.BigDecimal(t);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static Bound float32(CompOp op, double d, boolean strict) {
        if (Double.isNaN(d) || Double.isInfinite(d)) return null;
        float f = (float) d;
        if (Float.isInfinite(f)) return null;
        if ((double) f == d) return new Bound(op, f); // exact: any operator is sound
        if (strict) return null;
        switch (op) {
            case GT, GE -> { // col > d  implies  col >= (largest float <= d)
                float lo = (double) f <= d ? f : Math.nextDown(f);
                return new Bound(CompOp.GE, lo);
            }
            case LT, LE -> { // col < d  implies  col <= (smallest float >= d)
                float hi = (double) f >= d ? f : Math.nextUp(f);
                return new Bound(CompOp.LE, hi);
            }
            default -> {
                return null; // no float equals d; "<>" prunes nothing
            }
        }
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

    static String column(String text, TableSchema s) {
        if (text == null) return null;
        String t = text.trim();
        return (IDENT.matcher(t).matches() && s.has(t)) ? t : null;
    }

    /**
     * Parses a numeric literal as {@link SqlParser} re-renders it: a sign may be
     * separated from the digits by whitespace ({@code "- 5"}).
     */
    private static Num number(String text) {
        if (text == null) return null;
        String t = text.trim();
        if (t.length() > 1 && (t.charAt(0) == '-' || t.charAt(0) == '+')) {
            t = t.charAt(0) + t.substring(1).trim();
        }
        if (t.isEmpty()) return null;
        try {
            if (INT.matcher(t).matches()) {
                long v = Long.parseLong(t.startsWith("+") ? t.substring(1) : t);
                return new Num((double) v, true, v);
            }
            if (DEC.matcher(t).matches()) {
                double d = Double.parseDouble(t);
                if (Double.isNaN(d) || Double.isInfinite(d)) return null;
                boolean integral = d == Math.rint(d) && Math.abs(d) < 9.0e18;
                return new Num(d, integral, integral ? (long) d : 0L);
            }
        } catch (NumberFormatException ex) {
            return null; // e.g. an integer beyond long range: not pushed
        }
        return null;
    }

    /** A {@code DATE '...'} literal or an ISO date string, or null (not pushed). */
    private static java.time.LocalDate date(String text) {
        SqlLiterals.Operand o = SqlLiterals.operand(text, c -> false);
        if (o.kind() != SqlLiterals.Kind.DATE && o.kind() != SqlLiterals.Kind.STRING) return null;
        try {
            return java.time.LocalDate.parse(o.text().trim());
        } catch (java.time.DateTimeException e) {
            return null;
        }
    }

    /** Parses a SQL single-quoted string literal ({@code ''} escapes a quote), or returns null. */
    static String stringLiteral(String text) {
        if (text == null) return null;
        String t = text.trim();
        if (t.length() < 2 || t.charAt(0) != '\'') return null;
        StringBuilder sb = new StringBuilder();
        int n = t.length();
        for (int i = 1; i < n; i++) {
            char c = t.charAt(i);
            if (c == '\'') {
                if (i + 1 < n && t.charAt(i + 1) == '\'') { sb.append('\''); i++; continue; }
                return i == n - 1 ? sb.toString() : null;
            }
            sb.append(c);
        }
        return null; // unterminated
    }
}
