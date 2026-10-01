package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ast.*;
import com.github.gbenroscience.sqlv1.parquet.SqlLiterals.Kind;
import com.github.gbenroscience.sqlv1.parquet.SqlLiterals.LongTest;
import com.github.gbenroscience.sqlv1.parquet.SqlLiterals.Operand;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Predicates over string, boolean, date and timestamp columns (and {@code IS NULL} on any column),
 * which the numeric expression engine cannot evaluate.
 *
 * <h2>How they run</h2>
 * {@link #rewrite} walks the WHERE tree (already in negation-normal form). Each such leaf is replaced
 * by a numeric leaf {@code (__mK == 1)} over a synthetic {@code Float8} <em>mask column</em>; the
 * mask is filled per batch by a {@link Mask} kernel that reads the Arrow vector directly and writes
 * {@code 1.0}/{@code 0.0}. Leaves over numeric columns are left untouched for the engine.
 *
 * <p>Masks are <b>two-valued</b>: a null input yields {@code 0.0}, never null. That is exactly what a
 * WHERE needs (only TRUE keeps a row) and it makes {@code OR} correct for these leaves even when a
 * sibling numeric leaf is null.
 *
 * <h2>Allocation</h2>
 * Kernels allocate nothing per row. Strings are copied into one reusable scratch array; {@code LIKE}
 * runs on those bytes without building a {@code String}. {@code REGEXP} must build one {@code String}
 * per non-null row (a {@code java.util.regex} requirement) and is the slow path.
 *
 * <h2>Ordering of strings</h2>
 * {@code <}, {@code >}, {@code BETWEEN} compare UTF-8 bytes unsigned, which is code point order. It
 * is not locale collation, and differs from {@code String.compareTo} only for supplementary
 * characters versus U+E000 to U+FFFF.
 */
final class NativeLeaves {

    private NativeLeaves() { }

    /**
     * Name of mask column {@code index}. Letters only: ParserNG identifiers cannot start with an
     * underscore, and the name appears in the expression text handed to the engine.
     */
    static String maskName(int index) {
        StringBuilder sb = new StringBuilder("pngmask");
        int n = index;
        do {
            sb.append((char) ('a' + n % 26));
            n = n / 26 - 1;
        } while (n >= 0);
        return sb.toString();
    }

    /** Result of {@link #rewrite}. */
    record Result(BoolExpr where, List<Mask> masks) { }

    /** A kernel that fills one mask column for a batch. */
    abstract static class Mask {
        final String name;
        final String description;

        Mask(String name, String description) {
            this.name = name;
            this.description = description;
        }

        /** Writes {@code 1.0}/{@code 0.0} for rows {@code [0, n)} of {@code src} into {@code out}. */
        abstract void eval(VectorSchemaRoot src, int n, Float8Vector out);

        /**
         * The numeric column this kernel reads, or {@code null} if it reads the native scan vectors.
         * A non-null name means {@link #eval} receives the root of <em>widened</em> Float8 columns
         * (nulls preserved), and the column must be registered as a numeric reference.
         */
        String widenedColumn() {
            return null;
        }

        /**
         * Every numeric column this kernel reads from the widened root (empty if it reads the native
         * scan vectors). Defaults to {@link #widenedColumn()}; derived-column kernels read several.
         */
        List<String> widenedColumns() {
            String c = widenedColumn();
            return c == null ? List.of() : List.of(c);
        }
    }

    // =================================================================== rewriter

    static Result rewrite(BoolExpr where, TableSchema schema) {
        return rewrite(where, schema, 0);
    }

    /**
     * @param firstIndex index of the first mask name, so masks named elsewhere (derived columns of
     *                   {@link ScalarFns}) and these never collide
     */
    static Result rewrite(BoolExpr where, TableSchema schema, int firstIndex) {
        Rewriter r = new Rewriter(schema, containsOr(where), firstIndex);
        BoolExpr out = r.walk(where);
        return new Result(out, r.masks);
    }

    private static boolean containsOr(BoolExpr e) {
        return switch (e) {
            case OrExpr o -> true;
            case AndExpr(var l, var r) -> containsOr(l) || containsOr(r);
            default -> false;
        };
    }

    private static final class Rewriter {
        final TableSchema schema;
        final List<Mask> masks = new ArrayList<>();
        /**
         * True when the WHERE has an OR. The engine propagates null through a numeric leaf
         * ({@code null OR true} drops the row), so under an OR every simple numeric leaf
         * (column versus numeric literal, BETWEEN, IN) becomes a two-valued mask, which is exactly
         * SQL's three-valued logic for a WHERE in negation-normal form. Without an OR the fused
         * engine path is equivalent and faster.
         */
        final boolean twoValuedNumeric;

        final int firstIndex;

        Rewriter(TableSchema schema, boolean twoValuedNumeric, int firstIndex) {
            this.schema = schema;
            this.twoValuedNumeric = twoValuedNumeric;
            this.firstIndex = firstIndex;
        }

        /** Numeric column of {@code o}, or null if {@code o} is not a numeric column. */
        String numericColumn(Operand o) {
            return o.kind() == Kind.COLUMN && schema.kind(o.text()) != null && schema.kind(o.text()).numeric() ? o.text() : null;
        }

        BoolExpr numCmp(String col, CompOp op, String literal) {
            return ref(new NumCmpMask(nextName(), col, op, Double.parseDouble(literal)));
        }

        BoolExpr walk(BoolExpr e) {
            return switch (e) {
                case AndExpr(var l, var r) -> new AndExpr(walk(l), walk(r));
                case OrExpr(var l, var r) -> new OrExpr(walk(l), walk(r));
                case NotExpr n -> throw new IllegalStateException("WHERE must be in negation-normal form");
                case ComparisonExpr c -> comparison(c);
                case BetweenExpr b -> between(b);
                case InExpr i -> in(i);
                case IsNullExpr n -> isNull(n);
                case LikeExpr l -> like(l);
                case RegexExpr r -> regex(r);
                case TruthExpr t -> truth(t);
            };
        }

        Operand operand(String text) {
            return SqlLiterals.operand(text, schema::has);
        }

        ColumnKind kindOf(Operand o) {
            return o.kind() == Kind.COLUMN ? schema.kind(o.text()) : null;
        }

        BoolExpr ref(Mask m) {
            masks.add(m);
            return new ComparisonExpr(m.name, CompOp.EQ, "1");
        }

        String nextName() {
            return maskName(firstIndex + masks.size());
        }

        BoolExpr constant(boolean v, String why) {
            return ref(new ConstMask(nextName(), v, why));
        }

        static boolean nativeKind(ColumnKind k) {
            return k == ColumnKind.STRING || k == ColumnKind.BOOL || k == ColumnKind.DATE || k == ColumnKind.TIMESTAMP;
        }

        static boolean nativeLiteral(Kind k) {
            return k == Kind.STRING || k == Kind.BOOL || k == Kind.DATE || k == Kind.TIMESTAMP;
        }

        /** Rejects a column no predicate can use: binary, uuid, nested. */
        void rejectUnusable(String col, ColumnKind k) {
            if (k == ColumnKind.BINARY || k == ColumnKind.OTHER) {
                throw new ParquetSqlException("Column '" + col + "' is " + k
                        + " and cannot be used in WHERE; it can still be SELECTed as-is (IS [NOT] NULL is supported).");
            }
        }

        // ---- comparison ------------------------------------------------------------------
        BoolExpr comparison(ComparisonExpr c) {
            Operand a = operand(c.left()), b = operand(c.right());
            ColumnKind ka = kindOf(a), kb = kindOf(b);
            if (ka != null) rejectUnusable(a.text(), ka);
            if (kb != null) rejectUnusable(b.text(), kb);
            boolean isNative = (ka != null && nativeKind(ka)) || (kb != null && nativeKind(kb))
                    || nativeLiteral(a.kind()) || nativeLiteral(b.kind());
            if (!isNative) {
                if (a.kind() == Kind.NULL || b.kind() == Kind.NULL) {
                    return constant(false, "comparison with NULL is never true");
                }
                if (twoValuedNumeric) {
                    String ca = numericColumn(a), cb = numericColumn(b);
                    if (ca != null && b.kind() == Kind.NUMBER) return numCmp(ca, c.op(), b.text());
                    if (cb != null && a.kind() == Kind.NUMBER) return numCmp(cb, flip(c.op()), a.text());
                }
                return c; // numeric: the engine
            }
            if (a.kind() == Kind.NULL || b.kind() == Kind.NULL) return constant(false, "comparison with NULL is never true");
            CompOp op = c.op();
            Operand col, lit;
            if (a.kind() == Kind.COLUMN && b.kind() != Kind.COLUMN) { col = a; lit = b; }
            else if (b.kind() == Kind.COLUMN && a.kind() != Kind.COLUMN) { col = b; lit = a; op = flip(op); }
            else if (a.kind() == Kind.COLUMN) {
                throw new ParquetSqlException("Comparing two columns ('" + a.text() + "' " + ka + " and '" + b.text() + "' " + kb
                        + ") is not supported yet for non-numeric types; compare a column with a literal.");
            } else {
                throw new ParquetSqlException("A comparison needs a column on one side: (" + c.left() + " " + c.op() + " " + c.right() + ")");
            }
            return compareColumnLiteral(col.text(), schema.kind(col.text()), op, lit);
        }

        BoolExpr compareColumnLiteral(String col, ColumnKind k, CompOp op, Operand lit) {
            switch (k) {
                case STRING -> {
                    if (lit.kind() != Kind.STRING) throw mismatch(col, k, lit);
                    return ref(new StrCmpMask(nextName(), col, op, lit.text().getBytes(StandardCharsets.UTF_8)));
                }
                case BOOL -> {
                    if (lit.kind() != Kind.BOOL) throw mismatch(col, k, lit);
                    if (op != CompOp.EQ && op != CompOp.NEQ) {
                        throw new ParquetSqlException("Boolean column '" + col + "' supports only = and <> (or IS [NOT] TRUE/FALSE), not " + op);
                    }
                    return ref(new LongMask(nextName(), col, new LongTest(SqlLiterals.mask(op), lit.bool() ? 1 : 0, -1),
                            col + " " + op.parserNgSymbol() + " " + lit.bool()));
                }
                case DATE -> {
                    long day = lit.kind() == Kind.DATE || lit.kind() == Kind.STRING ? SqlLiterals.epochDay(lit.text())
                            : fail(mismatch(col, k, lit));
                    return ref(new LongMask(nextName(), col, SqlLiterals.lattice(op, day, true), col + " " + op.parserNgSymbol() + " DATE '" + lit.text() + "'"));
                }
                case TIMESTAMP -> {
                    ArrowType.Timestamp t = (ArrowType.Timestamp) schema.type(col);
                    boolean utc = t.getTimezone() != null;
                    long perSec = perSecond(t.getUnit());
                    LongTest test;
                    if (lit.kind() == Kind.DATE) test = SqlLiterals.dateAsTimestampTest(col, utc, op, SqlLiterals.epochDay(lit.text()), perSec);
                    else if (lit.kind() == Kind.TIMESTAMP || lit.kind() == Kind.STRING) test = SqlLiterals.timestampTest(col, utc, op, SqlLiterals.timestamp(lit.text()), perSec);
                    else throw mismatch(col, k, lit);
                    return ref(new LongMask(nextName(), col, test, col + " " + op.parserNgSymbol() + " TIMESTAMP '" + lit.text() + "'"));
                }
                default -> throw mismatch(col, k, lit);
            }
        }

        static long fail(ParquetSqlException e) {
            throw e;
        }

        static ParquetSqlException mismatch(String col, ColumnKind k, Operand lit) {
            return new ParquetSqlException("Cannot compare " + k + " column '" + col + "' with a " + lit.kind() + " literal (" + lit.text() + ")");
        }

        static long perSecond(TimeUnit u) {
            return switch (u) {
                case SECOND -> 1L;
                case MILLISECOND -> 1_000L;
                case MICROSECOND -> 1_000_000L;
                case NANOSECOND -> 1_000_000_000L;
            };
        }

        static CompOp flip(CompOp op) {
            return switch (op) {
                case LT -> CompOp.GT;
                case LE -> CompOp.GE;
                case GT -> CompOp.LT;
                case GE -> CompOp.LE;
                default -> op;
            };
        }

        // ---- between ---------------------------------------------------------------------
        BoolExpr between(BetweenExpr b) {
            Operand t = operand(b.target()), lo = operand(b.low()), hi = operand(b.high());
            ColumnKind k = kindOf(t);
            if (k != null) rejectUnusable(t.text(), k);
            boolean isNative = (k != null && nativeKind(k)) || nativeLiteral(lo.kind()) || nativeLiteral(hi.kind());
            if (!isNative) {
                String nc = twoValuedNumeric ? numericColumn(t) : null;
                if (nc != null && lo.kind() == Kind.NUMBER && hi.kind() == Kind.NUMBER) {
                    return b.negated()
                            ? new OrExpr(numCmp(nc, CompOp.LT, lo.text()), numCmp(nc, CompOp.GT, hi.text()))
                            : new AndExpr(numCmp(nc, CompOp.GE, lo.text()), numCmp(nc, CompOp.LE, hi.text()));
                }
                return b;
            }
            if (t.kind() != Kind.COLUMN) throw new ParquetSqlException("BETWEEN over a non-numeric value needs a column target: " + b.target());
            if (lo.kind() == Kind.NULL || hi.kind() == Kind.NULL) return constant(false, "BETWEEN with NULL is never true");
            if (!b.negated()) {
                return new AndExpr(compareColumnLiteral(t.text(), k, CompOp.GE, lo), compareColumnLiteral(t.text(), k, CompOp.LE, hi));
            }
            return new OrExpr(compareColumnLiteral(t.text(), k, CompOp.LT, lo), compareColumnLiteral(t.text(), k, CompOp.GT, hi));
        }

        // ---- in --------------------------------------------------------------------------
        BoolExpr in(InExpr in) {
            Operand t = operand(in.target());
            ColumnKind k = kindOf(t);
            if (k != null) rejectUnusable(t.text(), k);
            boolean anyNativeLit = false;
            List<Operand> vals = new ArrayList<>();
            for (String v : in.values()) {
                Operand o = operand(v);
                vals.add(o);
                anyNativeLit |= nativeLiteral(o.kind());
            }
            if (!((k != null && nativeKind(k)) || anyNativeLit)) {
                String nc = twoValuedNumeric ? numericColumn(t) : null;
                if (nc != null && !vals.isEmpty()) {
                    boolean simple = true, hasNull = false;
                    for (Operand o : vals) {
                        if (o.kind() == Kind.NULL) hasNull = true;
                        else if (o.kind() != Kind.NUMBER) simple = false;
                    }
                    if (simple) {
                        if (in.negated() && hasNull) return constant(false, "NOT IN a list containing NULL is never true");
                        BoolExpr acc = null;
                        for (Operand o : vals) {
                            if (o.kind() == Kind.NULL) continue;
                            BoolExpr leaf = numCmp(nc, in.negated() ? CompOp.NEQ : CompOp.EQ, o.text());
                            acc = acc == null ? leaf : in.negated() ? new AndExpr(acc, leaf) : new OrExpr(acc, leaf);
                        }
                        return acc == null ? constant(false, "IN (NULL) is never true") : acc;
                    }
                }
                return in;
            }
            if (t.kind() != Kind.COLUMN) throw new ParquetSqlException("IN over a non-numeric value needs a column target: " + in.target());
            boolean hasNull = false;
            List<Operand> real = new ArrayList<>();
            for (Operand o : vals) { if (o.kind() == Kind.NULL) hasNull = true; else real.add(o); }
            if (in.negated() && hasNull) return constant(false, "NOT IN a list containing NULL is never true");
            String col = t.text();
            switch (k) {
                case STRING -> {
                    byte[][] lits = new byte[real.size()][];
                    for (int i = 0; i < lits.length; i++) {
                        if (real.get(i).kind() != Kind.STRING) throw mismatch(col, k, real.get(i));
                        lits[i] = real.get(i).text().getBytes(StandardCharsets.UTF_8);
                    }
                    return ref(new StrInMask(nextName(), col, lits, in.negated()));
                }
                case BOOL -> {
                    boolean t1 = false, f1 = false;
                    for (Operand o : real) {
                        if (o.kind() != Kind.BOOL) throw mismatch(col, k, o);
                        if (o.bool()) t1 = true; else f1 = true;
                    }
                    long[] vs = t1 && f1 ? new long[] {0, 1} : t1 ? new long[] {1} : f1 ? new long[] {0} : new long[0];
                    return ref(new LongInMask(nextName(), col, vs, in.negated(), col + " IN (booleans)"));
                }
                case DATE -> {
                    long[] vs = new long[real.size()];
                    for (int i = 0; i < vs.length; i++) {
                        Operand o = real.get(i);
                        if (o.kind() != Kind.DATE && o.kind() != Kind.STRING) throw mismatch(col, k, o);
                        vs[i] = SqlLiterals.epochDay(o.text());
                    }
                    Arrays.sort(vs);
                    return ref(new LongInMask(nextName(), col, vs, in.negated(), col + " IN (dates)"));
                }
                case TIMESTAMP -> {
                    ArrowType.Timestamp ty = (ArrowType.Timestamp) schema.type(col);
                    boolean utc = ty.getTimezone() != null;
                    long ps = perSecond(ty.getUnit());
                    long[] tmp = new long[real.size()];
                    int cnt = 0;
                    for (Operand o : real) {
                        LongTest lt;
                        if (o.kind() == Kind.DATE) lt = SqlLiterals.dateAsTimestampTest(col, utc, CompOp.EQ, SqlLiterals.epochDay(o.text()), ps);
                        else if (o.kind() == Kind.TIMESTAMP || o.kind() == Kind.STRING) lt = SqlLiterals.timestampTest(col, utc, CompOp.EQ, SqlLiterals.timestamp(o.text()), ps);
                        else throw mismatch(col, k, o);
                        if (lt.constant() < 0) tmp[cnt++] = lt.threshold(); // finer than the unit or out of range: can never equal
                    }
                    long[] vs = Arrays.copyOf(tmp, cnt);
                    Arrays.sort(vs);
                    return ref(new LongInMask(nextName(), col, vs, in.negated(), col + " IN (timestamps)"));
                }
                default -> throw mismatch(col, k, real.isEmpty() ? new Operand(Kind.NULL, "NULL", false) : real.get(0));
            }
        }

        // ---- null / truth / like / regex --------------------------------------------------
        BoolExpr isNull(IsNullExpr n) {
            Operand t = operand(n.target());
            if (t.kind() != Kind.COLUMN) return n; // an expression target: left to the engine's own path
            return ref(new NullMask(nextName(), t.text(), n.negated()));
        }

        BoolExpr truth(TruthExpr t) {
            Operand o = operand(t.target());
            if (o.kind() != Kind.COLUMN || schema.kind(o.text()) != ColumnKind.BOOL) {
                throw new ParquetSqlException("IS [NOT] " + (t.value() ? "TRUE" : "FALSE") + " needs a BOOLEAN column, got: " + t.target()
                        + (o.kind() == Kind.COLUMN ? " (" + schema.kind(o.text()) + ")" : ""));
            }
            return ref(new TruthMask(nextName(), o.text(), t.value(), t.negated()));
        }

        String stringColumn(String what, String target) {
            Operand o = operand(target);
            if (o.kind() != Kind.COLUMN || schema.kind(o.text()) != ColumnKind.STRING) {
                throw new ParquetSqlException(what + " needs a STRING column on the left, got: " + target
                        + (o.kind() == Kind.COLUMN ? " (" + schema.kind(o.text()) + ")" : ""));
            }
            return o.text();
        }

        String stringLiteral(String what, String text) {
            Operand o = operand(text);
            if (o.kind() != Kind.STRING) throw new ParquetSqlException(what + " needs a string literal pattern, got: " + text);
            return o.text();
        }

        BoolExpr like(LikeExpr l) {
            String col = stringColumn(l.caseInsensitive() ? "ILIKE" : "LIKE", l.target());
            String pat = stringLiteral(l.caseInsensitive() ? "ILIKE" : "LIKE", l.pattern());
            int esc = -1;
            if (l.escape() != null) {
                String e = stringLiteral("ESCAPE", l.escape());
                if (e.codePointCount(0, e.length()) != 1) throw new ParquetSqlException("ESCAPE must be a single character, got '" + e + "'");
                esc = e.codePointAt(0);
            }
            LikePattern lp = LikePattern.compile(pat, esc, l.caseInsensitive());
            return ref(new LikeMask(nextName(), col, lp, l.negated(),
                    col + (l.negated() ? " NOT " : " ") + (l.caseInsensitive() ? "ILIKE '" : "LIKE '") + pat + "'"));
        }

        BoolExpr regex(RegexExpr r) {
            String col = stringColumn("REGEXP", r.target());
            String pat = stringLiteral("REGEXP", r.pattern());
            Pattern p;
            try {
                p = Pattern.compile(pat);
            } catch (PatternSyntaxException e) {
                throw new ParquetSqlException("Invalid REGEXP pattern '" + pat + "': " + e.getDescription());
            }
            return ref(new RegexMask(nextName(), col, p, r.negated(), col + (r.negated() ? " NOT" : "") + " REGEXP '" + pat + "'"));
        }
    }

    // =================================================================== kernels

    private static VarCharVector varchar(VectorSchemaRoot src, String col) {
        return (VarCharVector) vec(src, col);
    }

    private static FieldVector vec(VectorSchemaRoot src, String col) {
        FieldVector v = src.getVector(col);
        if (v == null) throw new ParquetSqlException("Column '" + col + "' is missing from the scan");
        return v;
    }

    /** Reusable growth buffer for one string kernel. */
    private static byte[] ensure(byte[] scratch, int len) {
        return len <= scratch.length ? scratch : new byte[Math.max(len, scratch.length * 2)];
    }

    static final class ConstMask extends Mask {
        private final double value;

        ConstMask(String name, boolean v, String why) {
            super(name, "constant " + v + " (" + why + ")");
            this.value = v ? 1.0 : 0.0;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            for (int i = 0; i < n; i++) out.set(i, value);
        }
    }

    /** Numeric column versus numeric literal, two-valued: a null input yields 0.0. Reads the widened Float8 column. */
    static final class NumCmpMask extends Mask {
        private final String col;
        private final CompOp op;
        private final double lit;

        NumCmpMask(String name, String col, CompOp op, double lit) {
            super(name, col + " " + op.parserNgSymbol() + " " + lit + " (two-valued, null is false)");
            this.col = col;
            this.op = op;
            this.lit = lit;
        }

        @Override String widenedColumn() {
            return col;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            Float8Vector v = (Float8Vector) vec(src, col);
            boolean noNulls = v.getNullCount() == 0;
            switch (op) {
                case EQ -> { for (int i = 0; i < n; i++) out.set(i, (noNulls || !v.isNull(i)) && v.get(i) == lit ? 1.0 : 0.0); }
                case NEQ -> { for (int i = 0; i < n; i++) out.set(i, (noNulls || !v.isNull(i)) && v.get(i) != lit ? 1.0 : 0.0); }
                case LT -> { for (int i = 0; i < n; i++) out.set(i, (noNulls || !v.isNull(i)) && v.get(i) < lit ? 1.0 : 0.0); }
                case LE -> { for (int i = 0; i < n; i++) out.set(i, (noNulls || !v.isNull(i)) && v.get(i) <= lit ? 1.0 : 0.0); }
                case GT -> { for (int i = 0; i < n; i++) out.set(i, (noNulls || !v.isNull(i)) && v.get(i) > lit ? 1.0 : 0.0); }
                case GE -> { for (int i = 0; i < n; i++) out.set(i, (noNulls || !v.isNull(i)) && v.get(i) >= lit ? 1.0 : 0.0); }
            }
        }
    }

    static final class NullMask extends Mask {
        private final String col;
        private final boolean negated;

        NullMask(String name, String col, boolean negated) {
            super(name, col + (negated ? " IS NOT NULL" : " IS NULL"));
            this.col = col;
            this.negated = negated;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            FieldVector v = vec(src, col);
            if (v.getNullCount() == 0) {
                double all = negated ? 1.0 : 0.0;
                for (int i = 0; i < n; i++) out.set(i, all);
                return;
            }
            for (int i = 0; i < n; i++) out.set(i, (v.isNull(i) != negated) ? 1.0 : 0.0);
        }
    }

    /** Boolean, date and timestamp comparisons: an exact integer test on the stored value. */
    static final class LongMask extends Mask {
        private final String col;
        private final LongTest test;

        LongMask(String name, String col, LongTest test, String description) {
            super(name, description);
            this.col = col;
            this.test = test;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            if (test.constant() >= 0) {
                double c = test.constant();
                for (int i = 0; i < n; i++) out.set(i, vec(src, col).isNull(i) ? 0.0 : c);
                return;
            }
            FieldVector v = vec(src, col);
            boolean noNulls = v.getNullCount() == 0;
            if (v instanceof BitVector b) {
                for (int i = 0; i < n; i++) out.set(i, (noNulls || !b.isNull(i)) && test.test(b.get(i)) ? 1.0 : 0.0);
            } else if (v instanceof DateDayVector d) {
                for (int i = 0; i < n; i++) out.set(i, (noNulls || !d.isNull(i)) && test.test(d.get(i)) ? 1.0 : 0.0);
            } else if (v instanceof TimeStampVector t) {
                for (int i = 0; i < n; i++) out.set(i, (noNulls || !t.isNull(i)) && test.test(t.get(i)) ? 1.0 : 0.0);
            } else {
                throw new ParquetSqlException("Column '" + col + "' has unexpected vector " + v.getClass().getSimpleName());
            }
        }
    }

    static final class LongInMask extends Mask {
        private final String col;
        private final long[] sorted;
        private final boolean negated;

        LongInMask(String name, String col, long[] sorted, boolean negated, String description) {
            super(name, description + (negated ? " negated" : ""));
            this.col = col;
            this.sorted = sorted;
            this.negated = negated;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            FieldVector v = vec(src, col);
            boolean noNulls = v.getNullCount() == 0;
            for (int i = 0; i < n; i++) {
                if (!noNulls && v.isNull(i)) { out.set(i, 0.0); continue; }
                long x = v instanceof BitVector b ? b.get(i) : v instanceof DateDayVector d ? d.get(i) : ((TimeStampVector) v).get(i);
                out.set(i, ((Arrays.binarySearch(sorted, x) >= 0) != negated) ? 1.0 : 0.0);
            }
        }
    }

    static final class TruthMask extends Mask {
        private final String col;
        private final int value;
        private final boolean negated;

        TruthMask(String name, String col, boolean value, boolean negated) {
            super(name, col + " IS " + (negated ? "NOT " : "") + (value ? "TRUE" : "FALSE"));
            this.col = col;
            this.value = value ? 1 : 0;
            this.negated = negated;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            BitVector b = (BitVector) vec(src, col);
            for (int i = 0; i < n; i++) {
                boolean is = !b.isNull(i) && b.get(i) == value;   // two-valued: null is never TRUE nor FALSE
                out.set(i, (is != negated) ? 1.0 : 0.0);
            }
        }
    }

    static final class StrCmpMask extends Mask {
        private final String col;
        private final int opMask;
        private final byte[] lit;
        private byte[] scratch = new byte[64];

        StrCmpMask(String name, String col, CompOp op, byte[] lit) {
            super(name, col + " " + op.parserNgSymbol() + " '" + new String(lit, StandardCharsets.UTF_8) + "'");
            this.col = col;
            this.opMask = SqlLiterals.mask(op);
            this.lit = lit;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            VarCharVector v = varchar(src, col);
            ArrowBuf off = v.getOffsetBuffer(), data = v.getDataBuffer();
            boolean noNulls = v.getNullCount() == 0;
            boolean equality = opMask == 0b010 || opMask == 0b101;
            for (int i = 0; i < n; i++) {
                if (!noNulls && v.isNull(i)) { out.set(i, 0.0); continue; }
                int s = off.getInt((long) i * 4), len = off.getInt((long) (i + 1) * 4) - s;
                int c;
                if (equality && len != lit.length) {
                    c = 1; // cheap reject: different lengths are never equal
                } else {
                    scratch = ensure(scratch, len);
                    data.getBytes(s, scratch, 0, len);
                    c = compare(scratch, len, lit);
                }
                out.set(i, ((opMask >>> (Integer.signum(c) + 1)) & 1) != 0 ? 1.0 : 0.0);
            }
        }

        static int compare(byte[] a, int alen, byte[] b) {
            int m = Math.min(alen, b.length);
            for (int k = 0; k < m; k++) {
                int x = a[k] & 0xFF, y = b[k] & 0xFF;
                if (x != y) return x - y;
            }
            return alen - b.length;
        }
    }

    static final class StrInMask extends Mask {
        private final String col;
        private final byte[][] lits;
        private final boolean negated;
        private byte[] scratch = new byte[64];

        StrInMask(String name, String col, byte[][] lits, boolean negated) {
            super(name, col + (negated ? " NOT IN " : " IN ") + "(" + lits.length + " strings)");
            this.col = col;
            this.lits = lits;
            this.negated = negated;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            VarCharVector v = varchar(src, col);
            ArrowBuf off = v.getOffsetBuffer(), data = v.getDataBuffer();
            boolean noNulls = v.getNullCount() == 0;
            for (int i = 0; i < n; i++) {
                if (!noNulls && v.isNull(i)) { out.set(i, 0.0); continue; }
                int s = off.getInt((long) i * 4), len = off.getInt((long) (i + 1) * 4) - s;
                boolean found = false;
                boolean copied = false;
                for (byte[] l : lits) {
                    if (l.length != len) continue;
                    if (!copied) { scratch = ensure(scratch, len); data.getBytes(s, scratch, 0, len); copied = true; }
                    boolean eq = true;
                    for (int k = 0; k < len; k++) if (scratch[k] != l[k]) { eq = false; break; }
                    if (eq) { found = true; break; }
                }
                out.set(i, (found != negated) ? 1.0 : 0.0);
            }
        }
    }

    static final class LikeMask extends Mask {
        private final String col;
        private final LikePattern pattern;
        private final boolean negated;
        private byte[] scratch = new byte[64];

        LikeMask(String name, String col, LikePattern pattern, boolean negated, String description) {
            super(name, description);
            this.col = col;
            this.pattern = pattern;
            this.negated = negated;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            VarCharVector v = varchar(src, col);
            ArrowBuf off = v.getOffsetBuffer(), data = v.getDataBuffer();
            boolean noNulls = v.getNullCount() == 0;
            for (int i = 0; i < n; i++) {
                if (!noNulls && v.isNull(i)) { out.set(i, 0.0); continue; } // NULL LIKE x, and NULL NOT LIKE x, are unknown
                int s = off.getInt((long) i * 4), len = off.getInt((long) (i + 1) * 4) - s;
                scratch = ensure(scratch, len);
                data.getBytes(s, scratch, 0, len);
                out.set(i, (pattern.matches(scratch, 0, len) != negated) ? 1.0 : 0.0);
            }
        }
    }

    static final class RegexMask extends Mask {
        private final String col;
        private final Pattern pattern;
        private final boolean negated;
        private byte[] scratch = new byte[64];

        RegexMask(String name, String col, Pattern pattern, boolean negated, String description) {
            super(name, description + " [allocates one String per non-null row]");
            this.col = col;
            this.pattern = pattern;
            this.negated = negated;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            VarCharVector v = varchar(src, col);
            ArrowBuf off = v.getOffsetBuffer(), data = v.getDataBuffer();
            boolean noNulls = v.getNullCount() == 0;
            for (int i = 0; i < n; i++) {
                if (!noNulls && v.isNull(i)) { out.set(i, 0.0); continue; }
                int s = off.getInt((long) i * 4), len = off.getInt((long) (i + 1) * 4) - s;
                scratch = ensure(scratch, len);
                data.getBytes(s, scratch, 0, len);
                boolean hit = pattern.matcher(new String(scratch, 0, len, StandardCharsets.UTF_8)).find();
                out.set(i, (hit != negated) ? 1.0 : 0.0);
            }
        }
    }
}
