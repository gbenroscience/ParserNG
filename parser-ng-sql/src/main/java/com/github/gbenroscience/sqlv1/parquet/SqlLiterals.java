package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ast.CompOp;

import java.time.DateTimeException;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parsing of SQL literal operands (as re-rendered by {@code SqlParser}) and the exact integer
 * "lattice" arithmetic used to compare integer-backed columns (dates, timestamps, booleans)
 * against literals. Pure Java, no Arrow.
 *
 * <h2>Temporal semantics</h2>
 * <ul>
 * <li>{@code DATE '2024-01-31'} is an epoch day.</li>
 * <li>{@code TIMESTAMP '2024-01-31 12:00:00[.fraction]'} has an optional zone suffix
 * ({@code Z} or {@code +HH:MM}). A column whose Parquet type is adjusted to UTC compares by instant:
 * a literal without a zone is read as UTC, a literal with a zone is converted. A column that is
 * <em>not</em> adjusted to UTC holds wall-clock time with no zone: a literal without a zone is
 * compared as-is, and a literal with a zone is an error (the comparison would be ambiguous).</li>
 * <li>A string literal compared with a DATE or TIMESTAMP column is parsed the same way.</li>
 * <li>Literals finer than the column unit (a nanosecond literal against a millisecond column) never
 * lose the comparison: {@code =} cannot match, and range operators use floor/ceiling so the answer
 * equals the exact real-number comparison.</li>
 * </ul>
 */
final class SqlLiterals {

    private SqlLiterals() { }

    enum Kind { COLUMN, STRING, BOOL, NULL, NUMBER, DATE, TIMESTAMP, OTHER }

    /** A classified operand. {@code text} is the column name, the decoded string, or the raw text. */
    record Operand(Kind kind, String text, boolean bool) { }

    private static final Pattern IDENT = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
    private static final Pattern NUM = Pattern.compile("[+-]?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?");
    private static final Pattern TYPED = Pattern.compile("(?is)^(DATE|TIMESTAMP)\\s*('(?:[^']|'')*')$");

    /** Classifies {@code text}; {@code isColumn} says whether an identifier names a column. */
    static Operand operand(String text, Predicate<String> isColumn) {
        String t = text == null ? "" : text.trim();
        if (t.equalsIgnoreCase("NULL")) return new Operand(Kind.NULL, t, false);
        if (t.equalsIgnoreCase("TRUE")) return new Operand(Kind.BOOL, t, true);
        if (t.equalsIgnoreCase("FALSE")) return new Operand(Kind.BOOL, t, false);
        if (t.startsWith("'")) {
            String s = PredicateConverter.stringLiteral(t);
            return s != null ? new Operand(Kind.STRING, s, false) : new Operand(Kind.OTHER, t, false);
        }
        Matcher m = TYPED.matcher(t);
        if (m.matches()) {
            String s = PredicateConverter.stringLiteral(m.group(2));
            Kind k = m.group(1).equalsIgnoreCase("DATE") ? Kind.DATE : Kind.TIMESTAMP;
            return s != null ? new Operand(k, s, false) : new Operand(Kind.OTHER, t, false);
        }
        if (IDENT.matcher(t).matches() && isColumn.test(t)) return new Operand(Kind.COLUMN, t, false);
        String n = t;
        if (n.length() > 1 && (n.charAt(0) == '-' || n.charAt(0) == '+')) n = n.charAt(0) + n.substring(1).trim();
        if (NUM.matcher(n).matches()) return new Operand(Kind.NUMBER, n, false);
        return new Operand(Kind.OTHER, t, false);
    }

    // ------------------------------------------------------------------ integer lattice

    /**
     * An integer comparison {@code value OP threshold} expressed as a bit mask over
     * {@code Long.compare(value, threshold)}: bit0 accepts "less", bit1 "equal", bit2 "greater".
     * {@code constant} is {@code -1} for "evaluate", {@code 0} for always false, {@code 1} for always true.
     */
    record LongTest(int opMask, long threshold, int constant) {
        static final LongTest FALSE = new LongTest(0, 0, 0);
        static final LongTest TRUE = new LongTest(7, 0, 1);

        boolean test(long v) {
            return ((opMask >>> (Long.compare(v, threshold) + 1)) & 1) != 0;
        }
    }

    static int mask(CompOp op) {
        return switch (op) {
            case LT -> 0b001;
            case LE -> 0b011;
            case EQ -> 0b010;
            case GE -> 0b110;
            case GT -> 0b100;
            case NEQ -> 0b101;
        };
    }

    /**
     * Compares integer column values with the real number {@code L}, given {@code floor(L)} and
     * whether {@code L} is itself an integer ({@code exact}). Exact for every operator.
     */
    static LongTest lattice(CompOp op, long floor, boolean exact) {
        switch (op) {
            case EQ:
                return exact ? new LongTest(mask(CompOp.EQ), floor, -1) : LongTest.FALSE;
            case NEQ:
                return exact ? new LongTest(mask(CompOp.NEQ), floor, -1) : LongTest.TRUE;
            case GT:
                return new LongTest(mask(CompOp.GT), floor, -1);
            case LE:
                return new LongTest(mask(CompOp.LE), floor, -1);
            case GE:
                return exact ? new LongTest(mask(CompOp.GE), floor, -1) : new LongTest(mask(CompOp.GE), floor + 1, -1);
            case LT:
                return exact ? new LongTest(mask(CompOp.LT), floor, -1) : new LongTest(mask(CompOp.LT), floor + 1, -1);
            default:
                throw new IllegalStateException();
        }
    }

    /** The constant answer for a literal beyond the representable range: {@code sign > 0} means +infinity. */
    static LongTest beyondRange(CompOp op, int sign) {
        boolean columnBelow = sign > 0; // every column value is below the literal
        return switch (op) {
            case LT, LE -> columnBelow ? LongTest.TRUE : LongTest.FALSE;
            case GT, GE -> columnBelow ? LongTest.FALSE : LongTest.TRUE;
            case EQ -> LongTest.FALSE;
            case NEQ -> LongTest.TRUE;
        };
    }

    // ------------------------------------------------------------------ dates and timestamps

    /** Parses an ISO date, or throws a {@link ParquetSqlException} naming the bad text. */
    static long epochDay(String s) {
        try {
            return LocalDate.parse(s.trim()).toEpochDay();
        } catch (DateTimeException e) {
            throw new ParquetSqlException("Invalid DATE literal '" + s + "': expected yyyy-MM-dd");
        }
    }

    /** A parsed timestamp literal. */
    record Ts(long epochSecond, int nano, boolean hasZone) { }

    /**
     * Parses {@code yyyy-MM-dd[( |T)HH:mm[:ss[.fraction]]][Z|+HH:MM|-HH:MM]}; a bare date means midnight.
     */
    static Ts timestamp(String raw) {
        String s = raw.trim();
        if (s.length() > 10 && s.charAt(10) == ' ') s = s.substring(0, 10) + "T" + s.substring(11).trim();
        try {
            if (s.length() == 10) return of(LocalDate.parse(s).atStartOfDay(), false);
            try {
                OffsetDateTime o = OffsetDateTime.parse(s);
                Instant i = o.toInstant();
                return new Ts(i.getEpochSecond(), i.getNano(), true);
            } catch (DateTimeException noZone) {
                return of(LocalDateTime.parse(s), false);
            }
        } catch (DateTimeException e) {
            throw new ParquetSqlException("Invalid TIMESTAMP literal '" + raw
                    + "': expected yyyy-MM-dd[ HH:mm[:ss[.fraction]]][Z|+HH:MM]");
        }
    }

    private static Ts of(LocalDateTime t, boolean zone) {
        return new Ts(t.toEpochSecond(ZoneOffset.UTC), t.getNano(), zone);
    }

    /**
     * Compares a timestamp column (stored as an integer count of {@code perSecond} units since the
     * epoch) with a literal, per the class comment's zone rules.
     *
     * @param columnName for error messages
     * @param utcAdjusted true if the Parquet column has {@code isAdjustedToUTC}
     */
    static LongTest timestampTest(String columnName, boolean utcAdjusted, CompOp op, Ts lit, long perSecond) {
        if (lit.hasZone() && !utcAdjusted) {
            throw new ParquetSqlException("Column '" + columnName + "' is a TIMESTAMP without a time zone (not adjusted to UTC); "
                    + "comparing it with a zoned literal is ambiguous. Write the literal without a zone.");
        }
        try {
            long whole = Math.multiplyExact(lit.epochSecond(), perSecond);
            long fracScaled = (long) lit.nano() * perSecond;          // < 1e9 * 1e9, fits
            long fracFloor = Math.floorDiv(fracScaled, 1_000_000_000L);
            boolean exact = Math.floorMod(fracScaled, 1_000_000_000L) == 0;
            return lattice(op, Math.addExact(whole, fracFloor), exact);
        } catch (ArithmeticException overflow) {
            return beyondRange(op, lit.epochSecond() >= 0 ? 1 : -1);
        }
    }

    /** As {@link #timestampTest} for a literal that is a plain date: midnight (UTC or wall clock). */
    static LongTest dateAsTimestampTest(String columnName, boolean utcAdjusted, CompOp op, long epochDay, long perSecond) {
        return timestampTest(columnName, utcAdjusted, op, new Ts(Math.multiplyExact(epochDay, 86_400L), 0, false), perSecond);
    }
}
