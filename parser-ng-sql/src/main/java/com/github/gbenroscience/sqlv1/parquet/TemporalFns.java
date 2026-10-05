package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.QuotedIdentifiers;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DateMilliVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;

import com.github.gbenroscience.sqlv1.parquet.CivilTime.Field;
import com.github.gbenroscience.sqlv1.parquet.CivilTime.Interval;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * {@code EXTRACT(field FROM x)}, {@code CAST(x AS DATE | TIMESTAMP | DECIMAL(p,s))} and {@code INTERVAL}.
 *
 * <p>ParserNG evaluates doubles, so date and time arithmetic cannot be written as engine text. As with
 * {@link ScalarFns} and {@link StringFns}, each call is evaluated here and replaced in the statement by a
 * derived column:
 * <ul>
 * <li>{@code EXTRACT} and {@code CAST(... AS DECIMAL(p,s))} are numeric ({@code Float8}) and work anywhere a
 * number does.</li>
 * <li>A date or timestamp expression becomes a derived {@code DATE} or {@code TIMESTAMP} column, so the native
 * date and timestamp kernels, the select list and string functions read it like a file column. Where an
 * operand is built only from literals ({@code TIMESTAMP '2024-01-31 00:00:00' - INTERVAL '7 days'},
 * {@code CAST('2024-02-01' AS DATE)}) it is folded at planning time into one {@code DATE '...'} or
 * {@code TIMESTAMP '...'} literal instead, which is what makes it usable in a comparison against a column.</li>
 * </ul>
 *
 * <p><b>Values.</b> Every temporal value is held as microseconds since the epoch (a {@code DATE} as midnight).
 * A nanosecond column is truncated to microseconds. A UTC-adjusted timestamp column is read as its UTC wall
 * clock, so {@code EXTRACT} and {@code CAST(... AS DATE)} use UTC. Derived timestamps are
 * {@code TIMESTAMP} without a time zone (microsecond unit): compare them with a literal that has no zone suffix.
 *
 * <p><b>EXTRACT fields.</b> {@code YEAR QUARTER MONTH WEEK} (ISO week) {@code DAY DOW} (0 = Sunday)
 * {@code ISODOW} (1 = Monday) {@code DOY HOUR MINUTE SECOND} (with fraction) {@code MILLISECOND}
 * {@code MICROSECOND} (the seconds field, with fraction, times 1000 and 1000000, as PostgreSQL does) and
 * {@code EPOCH} (seconds, with fraction). Unknown fields fail with the list.
 *
 * <p><b>CAST.</b> From a string column or literal to {@code DATE} ({@code yyyy-MM-dd}) or {@code TIMESTAMP}
 * ({@code yyyy-MM-dd[ T]HH:mm[:ss[.f]]}, no zone, at most 6 fraction digits unless the rest are zeros): an invalid
 * value fails loudly naming the expression and the value. From a timestamp to {@code DATE} (truncate), from
 * a date to {@code TIMESTAMP} (midnight). {@code CAST(n AS DECIMAL(p,s))} rounds a numeric column or literal
 * half away from zero to scale {@code s} on its decimal text and fails if the result has more than {@code p - s}
 * integer digits; it is {@code Float8}, so values beyond about 15 digits lose precision. Casting a string to a
 * number and any cast to a type not listed is not supported (a named error).
 *
 * <p><b>INTERVAL.</b> {@code INTERVAL '3 days'}, {@code INTERVAL '1' MONTH}, {@code INTERVAL '1 year 2 months'},
 * with units year, month, week, day, hour, minute, second, millisecond, microsecond (singular or plural).
 * It can be added to or subtracted from a date or timestamp. Year and month parts are calendar steps (the day
 * is clamped to the end of the month) and are only allowed on literals, which are folded; a column accepts the
 * fixed-length units (a day is exactly 24 hours). {@code INTERVAL} anywhere else is an error.
 */
final class TemporalFns {

    private static final long DAY_MICROS = CivilTime.DAY_MICROS;
    private static final long SEC_MICROS = CivilTime.SEC_MICROS;

    private TemporalFns() { }

    record Result(SelectStatement stmt, List<NativeLeaves.Mask> masks, List<DerivedColumns.Column> columns, int used) { }

    static String stripCalls(String text, Set<String> names) {
        return StringFns.stripCalls(text, names);
    }

    /** True if {@code text} is a date or timestamp expression (a temporal CAST or an INTERVAL), ignoring EXTRACT calls inside it. */
    static boolean producesTemporal(String text) {
        return text != null && (hasTemporalCast(text) || hasInterval(text));
    }

    /** True if {@code text} has something this class rewrites. */
    static boolean containsCall(String text) {
        if (text == null) return false;
        return StringFns.findCall(text, 0, Set.of("EXTRACT")) != null
                || hasTemporalCast(text) || decimalCast(text) != null || hasInterval(text);
    }

    static Result rewrite(SelectStatement st, TableSchema schema, int firstIndex) {
        Ctx ctx = new Ctx(schema, firstIndex);
        SelectStatement out = StatementRewriter.map(st, t -> ctx.text(t, true), t -> ctx.text(t, false));
        return new Result(out, List.copyOf(ctx.masks), List.copyOf(ctx.columns), ctx.next - firstIndex);
    }

    // ------------------------------------------------------------------ text scanning

    private static final java.util.regex.Pattern INTERVAL_RE = java.util.regex.Pattern.compile(
            "(?is)\\bINTERVAL\\s*'((?:[^']|'')*)'(?:\\s+([A-Za-z]+))?");
    private static final java.util.regex.Pattern DECIMAL_CAST = java.util.regex.Pattern.compile(
            "(?is)^(.*)\\s+AS\\s+DECIMAL\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)\\s*$");
    private static final java.util.regex.Pattern TEMPORAL_CAST = java.util.regex.Pattern.compile(
            "(?is)^(.*)\\s+AS\\s+(DATE|TIMESTAMP)\\s*$");

    private static boolean hasInterval(String t) {
        return INTERVAL_RE.matcher(t).find();
    }

    private static boolean hasTemporalCast(String t) {
        int from = 0;
        StringFns.Call c;
        while ((c = StringFns.findCall(t, from, Set.of("CAST"))) != null) {
            if (TEMPORAL_CAST.matcher(t.substring(c.argStart(), c.argEnd())).matches()) return true;
            from = c.end();
        }
        return false;
    }

    private static StringFns.Call decimalCast(String t) {
        int from = 0;
        StringFns.Call c;
        while ((c = StringFns.findCall(t, from, Set.of("CAST"))) != null) {
            if (DECIMAL_CAST.matcher(t.substring(c.argStart(), c.argEnd())).matches()) return c;
            from = c.end();
        }
        return null;
    }

    // ------------------------------------------------------------------ rewriting context

    private static final class Ctx {
        final TableSchema schema;
        int next;
        final List<NativeLeaves.Mask> masks = new ArrayList<>();
        final List<DerivedColumns.Column> columns = new ArrayList<>();
        final Map<String, String> byKey = new LinkedHashMap<>();

        Ctx(TableSchema schema, int firstIndex) {
            this.schema = schema;
            this.next = firstIndex;
        }

        String text(String text, boolean valueContext) {
            if (text == null || !containsCall(text)) return text;
            String t = extracts(text);
            t = decimals(t);
            if (hasTemporalCast(t) || hasInterval(t)) t = temporal(t, valueContext);
            return t;
        }

        private String extracts(String text) {
            StringFns.Call c = StringFns.findCall(text, 0, Set.of("EXTRACT"));
            if (c == null) return text;
            StringBuilder out = new StringBuilder();
            int copied = 0;
            while (c != null) {
                out.append(text, copied, c.start());
                String inside = text.substring(c.argStart(), c.argEnd());
                java.util.regex.Matcher m = java.util.regex.Pattern.compile("(?is)^\\s*([A-Za-z]+)\\s+FROM\\s+(.*)$").matcher(inside);
                if (!m.matches()) throw new ParquetSqlException("Expected EXTRACT(field FROM expression), got " + text.substring(c.start(), c.end()));
                Field field = Field.of(m.group(1));
                String arg = m.group(2).trim();
                String key = "E:" + field + ":" + arg.replaceAll("\\s+", " ");
                String name = byKey.get(key);
                if (name == null) {
                    TNode root = new TParser(arg, schema).parseWhole();
                    name = NativeLeaves.maskName(next++);
                    masks.add(new ExtractMask(name, text.substring(c.start(), c.end()), field, root));
                    byKey.put(key, name);
                }
                out.append(name);
                copied = c.end();
                c = StringFns.findCall(text, copied, Set.of("EXTRACT"));
            }
            return out.append(text, copied, text.length()).toString();
        }

        private String decimals(String text) {
            StringFns.Call c = decimalCast(text);
            if (c == null) return text;
            StringBuilder out = new StringBuilder();
            int copied = 0;
            while (c != null) {
                out.append(text, copied, c.start());
                java.util.regex.Matcher m = DECIMAL_CAST.matcher(text.substring(c.argStart(), c.argEnd()));
                m.matches();
                String arg = m.group(1).trim();
                int p = Integer.parseInt(m.group(2)), sc = Integer.parseInt(m.group(3));
                if (p < 1 || p > 38 || sc > p) throw new ParquetSqlException("DECIMAL(" + p + "," + sc + ") needs 1 <= precision <= 38 and 0 <= scale <= precision");
                String key = "D:" + p + "," + sc + ":" + arg;
                String name = byKey.get(key);
                if (name == null) {
                    name = NativeLeaves.maskName(next++);
                    masks.add(decimalMask(name, text.substring(c.start(), c.end()), arg, p, sc));
                    byKey.put(key, name);
                }
                out.append(name);
                copied = c.end();
                c = decimalCast(text.substring(copied));
                if (c != null) { // re-scan on the remainder: shift offsets back to the whole text
                    c = new StringFns.Call(c.func(), c.start() + copied, c.end() + copied, c.argStart() + copied, c.argEnd() + copied);
                }
            }
            return out.append(text, copied, text.length()).toString();
        }

        private NativeLeaves.Mask decimalMask(String name, String callText, String arg, int p, int sc) {
            String col = QuotedIdentifiers.real(arg);
            ColumnKind k = schema.kind(arg);
            Double lit = null;
            if (k == null) {
                try {
                    lit = Double.parseDouble(arg.replace(" ", ""));
                } catch (NumberFormatException e) {
                    throw new ParquetSqlException(callText + ": the argument must be a numeric column or a numeric literal, not '" + arg + "'");
                }
            } else if (!k.numeric()) {
                throw new ParquetSqlException(callText + ": column '" + col + "' is " + k + "; CAST to DECIMAL needs a numeric column");
            }
            return new DecimalMask(name, callText, k == null ? null : arg, lit, p, sc);
        }

        /** A whole temporal expression: folded to a literal (operand context, literals only) or derived as a column. */
        private String temporal(String text, boolean valueContext) {
            String key = "T:" + valueContext + ":" + text.replaceAll("\\s+", " ");
            String known = byKey.get(key);
            if (known != null) return known;
            TNode root = new TParser(text, schema).parseWhole();
            String out;
            if (root instanceof TLit lit && !valueContext) {
                out = lit.literalText();
            } else {
                String name = NativeLeaves.maskName(next++);
                columns.add(new TemporalColumn(name, text, root));
                out = name;
            }
            byKey.put(key, out);
            return out;
        }
    }

    // ------------------------------------------------------------------ temporal expression tree

    /** One node. {@code eval} returns false for null and leaves the microseconds since the epoch in {@code val}. */
    abstract static class TNode {
        long val;

        /** True if the value is a calendar date (midnight), so a derived column of it is a {@code DATE}. */
        abstract boolean isDate();

        abstract boolean eval(int row);

        void bind(VectorSchemaRoot src) { }
    }

    /** A literal date or timestamp, foldable. */
    static final class TLit extends TNode {
        final boolean date;

        TLit(long micros, boolean date) {
            this.val = micros;
            this.date = date;
        }

        @Override boolean isDate() { return date; }

        @Override boolean eval(int row) { return true; }

        String literalText() {
            if (date) return "DATE '" + java.time.LocalDate.ofEpochDay(Math.floorDiv(val, DAY_MICROS)) + "'";
            long sec = Math.floorDiv(val, SEC_MICROS);
            int micros = (int) Math.floorMod(val, SEC_MICROS);
            java.time.LocalDateTime t = java.time.LocalDateTime.ofEpochSecond(sec, micros * 1000, java.time.ZoneOffset.UTC);
            String frac = micros == 0 ? "" : "." + String.format("%06d", micros).replaceAll("0+$", "");
            return "TIMESTAMP '" + t.toLocalDate() + " " + String.format("%02d:%02d:%02d", t.getHour(), t.getMinute(), t.getSecond()) + frac + "'";
        }
    }

    private static final class TCol extends TNode {
        private final String name;
        private final boolean date;
        private FieldVector v;

        TCol(String name, boolean date) {
            this.name = name;
            this.date = date;
        }

        @Override boolean isDate() { return date; }

        @Override void bind(VectorSchemaRoot src) {
            v = src.getVector(QuotedIdentifiers.real(name));
            if (v == null) v = src.getVector(name);
            if (v == null) throw new ParquetSqlException("Column '" + QuotedIdentifiers.real(name) + "' is missing from the scan");
        }

        @Override boolean eval(int row) {
            if (v.isNull(row)) return false;
            if (v instanceof DateDayVector d) val = d.get(row) * DAY_MICROS;
            else if (v instanceof DateMilliVector d) val = Math.multiplyExact(d.get(row), 1000L);
            else if (v instanceof TimeStampVector t) {
                ArrowType.Timestamp ty = (ArrowType.Timestamp) t.getField().getType();
                long x = t.get(row);
                val = switch (ty.getUnit()) {
                    case SECOND -> Math.multiplyExact(x, SEC_MICROS);
                    case MILLISECOND -> Math.multiplyExact(x, 1000L);
                    case MICROSECOND -> x;
                    case NANOSECOND -> Math.floorDiv(x, 1000L);
                };
            } else {
                throw new ParquetSqlException("Column '" + QuotedIdentifiers.real(name) + "' (" + v.getClass().getSimpleName() + ") is not a date or timestamp");
            }
            return true;
        }
    }

    private static final class TToDate extends TNode {
        private final TNode child;

        TToDate(TNode child) {
            this.child = child;
        }

        @Override boolean isDate() { return true; }

        @Override void bind(VectorSchemaRoot src) { child.bind(src); }

        @Override boolean eval(int row) {
            if (!child.eval(row)) return false;
            val = Math.floorDiv(child.val, DAY_MICROS) * DAY_MICROS;
            return true;
        }
    }

    /** Identity on the value; only changes how a date reads as a type. */
    private static final class TToTimestamp extends TNode {
        private final TNode child;

        TToTimestamp(TNode child) {
            this.child = child;
        }

        @Override boolean isDate() { return false; }

        @Override void bind(VectorSchemaRoot src) { child.bind(src); }

        @Override boolean eval(int row) {
            if (!child.eval(row)) return false;
            val = child.val;
            return true;
        }
    }

    private static final class TAdd extends TNode {
        private final TNode child;
        private final long delta;

        TAdd(TNode child, long delta) {
            this.child = child;
            this.delta = delta;
        }

        @Override boolean isDate() { return child.isDate() && delta % DAY_MICROS == 0; }

        @Override void bind(VectorSchemaRoot src) { child.bind(src); }

        @Override boolean eval(int row) {
            if (!child.eval(row)) return false;
            val = Math.addExact(child.val, delta);
            return true;
        }
    }

    /** CAST of a string expression to DATE or TIMESTAMP. */
    private static final class TFromString extends TNode {
        private final StringFns.SNode child;
        private final boolean toDate;
        private final String text;

        TFromString(StringFns.SNode child, boolean toDate, String text) {
            this.child = child;
            this.toDate = toDate;
            this.text = text;
        }

        @Override boolean isDate() { return toDate; }

        @Override void bind(VectorSchemaRoot src) { child.bind(src); }

        @Override boolean eval(int row) {
            int len = child.eval(row);
            if (len < 0) return false;
            val = CivilTime.parse(child.buf, len, toDate, text);
            return true;
        }
    }

    // ------------------------------------------------------------------ expression parser

    /** {@code temporal := primary { ('+'|'-') interval }} over a text; see the class comment. */
    private static final class TParser {
        final String t;
        final TableSchema schema;
        int i;

        TParser(String text, TableSchema schema) {
            this.t = text;
            this.schema = schema;
        }

        TNode parseWhole() {
            TNode n = expr();
            skipWs();
            if (i != t.length()) throw fail("it mixes date or time arithmetic with something that is not a date, timestamp or INTERVAL");
            return n;
        }

        private ParquetSqlException fail(String why) {
            return new ParquetSqlException("Cannot evaluate '" + t.trim() + "': " + why);
        }

        private void skipWs() {
            while (i < t.length() && Character.isWhitespace(t.charAt(i))) i++;
        }

        private TNode expr() {
            TNode n = primary();
            while (true) {
                skipWs();
                if (i >= t.length() || (t.charAt(i) != '+' && t.charAt(i) != '-')) return n;
                boolean minus = t.charAt(i) == '-';
                i++;
                skipWs();
                java.util.regex.Matcher m = INTERVAL_RE.matcher(t);
                if (!m.find(i) || m.start() != i) throw fail("only an INTERVAL can be added to or subtracted from a date or timestamp");
                i = m.end();
                Interval iv = CivilTime.interval(m.group(1).replace("''", "'"), m.group(2));
                if (minus) iv = iv.negate();
                n = add(n, iv);
            }
        }

        private TNode add(TNode n, Interval iv) {
            if (n instanceof TLit lit) {
                long v = CivilTime.addInterval(lit.val, iv);
                boolean stillDate = lit.date && iv.micros() % DAY_MICROS == 0;
                return new TLit(v, stillDate);
            }
            if (iv.months() != 0) {
                throw fail("a year or month INTERVAL can only be added to a literal; use days, hours, minutes or seconds with a column");
            }
            return new TAdd(n, iv.micros());
        }

        private TNode primary() {
            skipWs();
            if (i >= t.length()) throw fail("an operand is missing");
            char c = t.charAt(i);
            if (c == '(') {
                i++;
                TNode n = expr();
                skipWs();
                if (i >= t.length() || t.charAt(i) != ')') throw fail("expected ')'");
                i++;
                return n;
            }
            if (!(Character.isLetter(c) || c == '_')) throw fail("unexpected '" + c + "'");
            int s = i++;
            while (i < t.length() && (Character.isLetterOrDigit(t.charAt(i)) || t.charAt(i) == '_')) i++;
            String word = t.substring(s, i);
            String up = word.toUpperCase(Locale.ROOT);
            int save = i;
            skipWs();
            if ((up.equals("DATE") || up.equals("TIMESTAMP")) && i < t.length() && t.charAt(i) == '\'') {
                String lit = quoted();
                return literal(up.equals("DATE"), lit);
            }
            if (up.equals("CAST") && i < t.length() && t.charAt(i) == '(') {
                int close = StringFns.matching(t, i);
                if (close < 0) throw fail("missing ')' after CAST(");
                String inside = t.substring(i + 1, close);
                i = close + 1;
                return cast(inside);
            }
            i = save;
            return column(word);
        }

        private String quoted() {
            StringBuilder sb = new StringBuilder();
            i++;
            while (true) {
                if (i >= t.length()) throw fail("unterminated string literal");
                char d = t.charAt(i);
                if (d == '\'') {
                    if (i + 1 < t.length() && t.charAt(i + 1) == '\'') { sb.append('\''); i += 2; continue; }
                    i++;
                    return sb.toString();
                }
                sb.append(d);
                i++;
            }
        }

        private TNode literal(boolean date, String text) {
            if (date) return new TLit(SqlLiterals.epochDay(text) * DAY_MICROS, true);
            SqlLiterals.Ts ts = SqlLiterals.timestamp(text);
            if (ts.hasZone()) {
                throw fail("a TIMESTAMP literal with a time zone cannot be combined with CAST or INTERVAL; write it without the zone suffix");
            }
            return new TLit(Math.addExact(Math.multiplyExact(ts.epochSecond(), SEC_MICROS), ts.nano() / 1000), false);
        }

        private TNode column(String name) {
            ColumnKind k = schema.kind(name);
            if (k == null) throw fail("unknown column '" + QuotedIdentifiers.real(name) + "'");
            if (k != ColumnKind.DATE && k != ColumnKind.TIMESTAMP) {
                throw fail("column '" + QuotedIdentifiers.real(name) + "' is " + k + "; this needs a date or timestamp");
            }
            return new TCol(name, k == ColumnKind.DATE);
        }

        private TNode cast(String inside) {
            java.util.regex.Matcher m = TEMPORAL_CAST.matcher(inside);
            if (!m.matches()) throw fail("only CAST(... AS DATE) and CAST(... AS TIMESTAMP) can be combined with date arithmetic");
            boolean toDate = m.group(2).equalsIgnoreCase("DATE");
            String inner = m.group(1).trim();
            String shown = "CAST(" + inner + " AS " + m.group(2).toUpperCase(Locale.ROOT) + ")";
            // a string literal or string column / function: parse the string
            boolean stringy = inner.startsWith("'");
            if (!stringy) {
                String head = inner.split("[^A-Za-z0-9_.]", 2)[0];
                ColumnKind k = head.isEmpty() ? null : schema.kind(head);
                stringy = k == ColumnKind.STRING || StringFns.containsCall(inner);
            }
            if (stringy) {
                if (inner.startsWith("'") && !inner.substring(1).contains("'") ) throw fail("unterminated string literal");
                StringFns.SNode s = StringFns.parseStringExpression(inner, schema);
                if (s instanceof StringFns.LitNode lit) { // a literal: validate now and fold
                    byte[] bytes = lit.bytes();
                    return new TLit(CivilTime.parse(bytes, bytes.length, toDate, shown), toDate);
                }
                return new TFromString(s, toDate, shown);
            }
            TNode child = new TParser(inner, schema).parseWhole();
            return toDate ? (child.isDate() ? child : new TToDate(child)) : (child.isDate() ? new TToTimestamp(child) : child);
        }
    }

    // ------------------------------------------------------------------ derived columns and kernels

    /** A date or timestamp expression, as a column of its own type. */
    private static final class TemporalColumn extends DerivedColumns.Column {
        private final TNode root;

        TemporalColumn(String name, String text, TNode root) {
            super(name, text, root.isDate() ? ColumnKind.DATE : ColumnKind.TIMESTAMP,
                    root.isDate() ? (ArrowType) new ArrowType.Date(DateUnit.DAY) : new ArrowType.Timestamp(TimeUnit.MICROSECOND, null));
            this.root = root;
        }

        @Override FieldVector newVector(BufferAllocator alloc) {
            return root.isDate() ? new DateDayVector(name, alloc) : new TimeStampMicroVector(name, alloc);
        }

        @Override void fill(VectorSchemaRoot src, int n, FieldVector out) {
            root.bind(src);
            if (root.isDate()) {
                DateDayVector o = (DateDayVector) out;
                for (int i = 0; i < n; i++) {
                    if (root.eval(i)) o.setSafe(i, (int) Math.floorDiv(root.val, DAY_MICROS));
                    else o.setNull(i);
                }
            } else {
                TimeStampMicroVector o = (TimeStampMicroVector) out;
                for (int i = 0; i < n; i++) {
                    if (root.eval(i)) o.setSafe(i, root.val);
                    else o.setNull(i);
                }
            }
        }
    }

    /** {@code EXTRACT(field FROM x)}. */
    private static final class ExtractMask extends NativeLeaves.Mask {
        private final Field field;
        private final TNode root;
        private final long[] scratch = new long[3];

        ExtractMask(String name, String text, Field field, TNode root) {
            super(name, text + " (derived Float8 column)");
            this.field = field;
            this.root = root;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            root.bind(src);
            for (int i = 0; i < n; i++) {
                if (root.eval(i)) out.set(i, CivilTime.extract(field, root.val, scratch));
                else out.setNull(i);
            }
        }
    }

    /** {@code CAST(x AS DECIMAL(p,s))} over a numeric column (widened) or literal: rounds half away from zero at scale {@code s}. */
    private static final class DecimalMask extends NativeLeaves.Mask {
        private final String col;
        private final Double lit;
        private final int p, s;
        private final String text;

        DecimalMask(String name, String text, String col, Double lit, int p, int s) {
            super(name, text + " (derived Float8 column)");
            this.col = col;
            this.lit = lit;
            this.p = p;
            this.s = s;
            this.text = text;
        }

        @Override List<String> widenedColumns() {
            return col == null ? List.of() : List.of(col);
        }

        @Override void eval(VectorSchemaRoot widened, int n, Float8Vector out) {
            Float8Vector v = col == null ? null : (Float8Vector) widened.getVector(col);
            for (int i = 0; i < n; i++) {
                double x;
                if (v != null) {
                    if (v.isNull(i)) { out.setNull(i); continue; }
                    x = v.get(i);
                } else {
                    x = lit;
                }
                if (Double.isNaN(x) || Double.isInfinite(x)) {
                    throw new ParquetSqlException(text + ": " + x + " cannot be represented as DECIMAL(" + p + "," + s + ")");
                }
                BigDecimal r = BigDecimal.valueOf(x).setScale(s, RoundingMode.HALF_UP);
                if (r.precision() - r.scale() > p - s) {
                    throw new ParquetSqlException(text + ": value " + x + " needs more than " + (p - s) + " integer digits for DECIMAL(" + p + "," + s + ")");
                }
                out.set(i, r.doubleValue());
            }
        }
    }
}
