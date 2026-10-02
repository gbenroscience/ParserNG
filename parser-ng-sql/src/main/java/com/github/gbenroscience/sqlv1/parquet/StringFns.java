package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.QuotedIdentifiers;
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
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * SQL string functions: {@code LOWER}, {@code UPPER}, {@code TRIM}, {@code SUBSTRING}/{@code SUBSTR},
 * {@code CONCAT}, the {@code ||} operator, and {@code LENGTH}/{@code CHAR_LENGTH}/{@code OCTET_LENGTH}.
 *
 * <p><b>Why a separate path.</b> ParserNG evaluates doubles; a string has no representation in it.
 * Strings are therefore evaluated here, directly over the UTF-8 bytes of the Arrow vectors, and
 * only their <em>results</em> meet the engine:
 * <ul>
 * <li>{@code LENGTH(...)} is numeric. It becomes a derived {@code Float8} column (like
 * {@code COALESCE}, see {@link ScalarFns}), so it works everywhere a number does: arithmetic,
 * {@code WHERE}, {@code ORDER BY}, {@code GROUP BY}, aggregates.</li>
 * <li>A string-valued expression becomes a derived {@code VarChar} column, filled once per batch and
 * added to the scan batch as a virtual column. The string predicates ({@code =}, {@code <},
 * {@code IN}, {@code BETWEEN}, {@code LIKE}, {@code REGEXP}, {@code IS NULL}) and the select list
 * read it like any file column, so {@code WHERE LOWER(name) LIKE 'a%'} and
 * {@code SELECT UPPER(name)} need nothing else.</li>
 * </ul>
 * The statement is rewritten so each call is replaced by the derived column's name; the original text
 * stays the output name.
 *
 * <p><b>Semantics.</b> A null argument gives null, except {@code CONCAT}, which skips nulls (an
 * all-null {@code CONCAT} is the empty string) while {@code ||} gives null if any operand is null.
 * Lengths count characters (UTF-8 code points), except {@code OCTET_LENGTH}, which counts bytes.
 * {@code LOWER}/{@code UPPER} fold ASCII by byte and fall back to {@code String.toLowerCase(Locale.ROOT)}
 * only for a value that holds a non-ASCII byte, so full Unicode case mapping is applied where it matters
 * and costs nothing where it does not. {@code TRIM} removes spaces (U+0020) at both ends, as SQL does.
 * {@code SUBSTRING(s, start [, count])} is 1-based by character, and {@code start} and {@code count}
 * must be integer literals; positions before 1 count as in PostgreSQL ({@code SUBSTRING('abc', 0, 2)}
 * is {@code 'a'}), and a negative {@code count} is an error. Arguments are string columns, string
 * literals, numeric literals (taken as their text) and other string expressions; a non-string column is an
 * error that names it.
 *
 * <p><b>Limits.</b> A string-valued expression cannot be an {@code ORDER BY}/{@code GROUP BY} key, an
 * aggregate argument, or be mixed with arithmetic; each fails with a named error. Every row of a batch
 * is evaluated, including rows the {@code WHERE} later drops. The kernels reuse their buffers, so steady
 * state allocates nothing per row except for a value with non-ASCII bytes under {@code LOWER}/{@code UPPER}.
 */
final class StringFns {

    private StringFns() { }

    /**
     * @param stmt    the rewritten statement ({@code input} itself when there was nothing to rewrite)
     * @param masks   derived numeric columns ({@code LENGTH})
     * @param strings derived string columns
     * @param virtual the derived string columns' names and kinds, for {@link TableSchema#withVirtual}
     * @param used    how many mask names were consumed starting at {@code firstIndex}
     */
    record Result(SelectStatement stmt, List<NativeLeaves.Mask> masks, List<Derived> strings,
                  Map<String, ColumnKind> virtual, int used) { }

    /** One derived string column and the expression that fills it. */
    static final class Derived {
        final String name;
        final String text;
        final SNode root;

        Derived(String name, String text, SNode root) {
            this.name = name;
            this.text = text;
            this.root = root;
        }
    }

    static boolean containsCall(String text) {
        return text != null && (hasStringCall(text) || findLength(text, 0) != null);
    }

    static Result rewrite(SelectStatement st, TableSchema schema, int firstIndex) {
        Ctx ctx = new Ctx(schema, firstIndex);

        List<SelectItem> items = new ArrayList<>(st.items().size());
        for (SelectItem it : st.items()) {
            if (it.isAggregate()) {
                AggregateSpec a = it.aggregate();
                if (a.star()) {
                    items.add(it);
                    continue;
                }
                String arg = ctx.numericOnly(a.argExprText(), a.func() + " argument");
                items.add(arg.equals(a.argExprText()) ? it
                        : SelectItem.aggregate(new AggregateSpec(a.func(), arg, false, a.distinct()), it.outputName()));
            } else {
                String t = ctx.value(it.exprText());
                items.add(t.equals(it.exprText()) ? it : SelectItem.plain(t, it.outputName()));
            }
        }
        BoolExpr where = st.where() == null ? null : mapBool(st.where(), ctx::value);
        BoolExpr having = st.having() == null ? null : mapBool(st.having(), t -> ctx.numericOnly(t, "HAVING"));
        List<String> groupBy = new ArrayList<>(st.groupBy().size());
        for (String g : st.groupBy()) groupBy.add(ctx.numericOnly(g, "GROUP BY"));
        List<OrderItem> orderBy = new ArrayList<>(st.orderBy().size());
        for (OrderItem o : st.orderBy()) orderBy.add(new OrderItem(ctx.numericOnly(o.exprText(), "ORDER BY"), o.descending()));

        if (!ctx.changed) return new Result(st, List.of(), List.of(), Map.of(), 0);
        SelectStatement out = new SelectStatement(st.selectAll(), items, st.table(), where, groupBy, having, orderBy,
                st.limit(), st.tableKind(), st.tablePaths(), st.distinct(), st.offset());
        Map<String, ColumnKind> virtual = new LinkedHashMap<>();
        for (Derived d : ctx.strings) virtual.put(d.name, ColumnKind.STRING);
        return new Result(out, List.copyOf(ctx.masks), List.copyOf(ctx.strings), virtual, ctx.next - firstIndex);
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
            case IsNullExpr(var t, var neg) -> new IsNullExpr(f.apply(t), neg);
            case LikeExpr(var t, var p, var esc, var ci, var neg) -> new LikeExpr(f.apply(t), p, esc, ci, neg);
            case RegexExpr(var t, var p, var neg) -> new RegexExpr(f.apply(t), p, neg);
            case TruthExpr tr -> tr;
        };
    }

    // ------------------------------------------------------------------ text scanning

    private static final Set<String> STRING_FUNCS = Set.of("LOWER", "UPPER", "TRIM", "SUBSTRING", "SUBSTR", "CONCAT");
    private static final Set<String> LENGTH_FUNCS = Set.of("LENGTH", "CHAR_LENGTH", "CHARACTER_LENGTH", "OCTET_LENGTH");

    /** A call to a function of {@code names}: {@code [start, end)} spans {@code NAME(...)}; {@code argStart..argEnd} is the inside. */
    private record Call(String func, int start, int end, int argStart, int argEnd) { }

    private static int skipString(String t, int start, char q) {
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

    private static int matching(String t, int open) {
        int depth = 0, n = t.length();
        for (int i = open; i < n; i++) {
            char c = t.charAt(i);
            if (c == '\'' || c == '"') {
                i = skipString(t, i, c) - 1;
            } else if (c == '(') {
                depth++;
            } else if (c == ')' && --depth == 0) {
                return i;
            }
        }
        return -1;
    }

    /** The first call at or after {@code from} to a function in {@code names}, outside string literals; null if none. */
    private static Call findCall(String t, int from, Set<String> names) {
        int n = t.length(), i = from;
        while (i < n) {
            char c = t.charAt(i);
            if (c == '\'' || c == '"') {
                i = skipString(t, i, c);
            } else if (Character.isLetter(c) || c == '_') {
                int s = i++;
                while (i < n && (Character.isLetterOrDigit(t.charAt(i)) || t.charAt(i) == '_')) i++;
                String word = t.substring(s, i).toUpperCase(Locale.ROOT);
                if (names.contains(word) && !(s > 0 && t.charAt(s - 1) == '.')) {
                    int j = i;
                    while (j < n && Character.isWhitespace(t.charAt(j))) j++;
                    if (j < n && t.charAt(j) == '(') {
                        int close = matching(t, j);
                        if (close < 0) throw new ParquetSqlException(word + " call is missing its closing parenthesis in: " + t);
                        return new Call(word, s, close + 1, j + 1, close);
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

    private static Call findLength(String t, int from) {
        return findCall(t, from, LENGTH_FUNCS);
    }

    /** True if {@code t} has a string function call or a {@code ||} outside string literals. */
    private static boolean hasStringCall(String t) {
        if (findCall(t, 0, STRING_FUNCS) != null) return true;
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            if (c == '\'' || c == '"') i = skipString(t, i, c) - 1;
            else if (c == '|' && i + 1 < t.length() && t.charAt(i + 1) == '|') return true;
        }
        return false;
    }

    // ------------------------------------------------------------------ rewriting context

    private static final class Ctx {
        final TableSchema schema;
        final int firstIndex;
        int next;
        final List<NativeLeaves.Mask> masks = new ArrayList<>();
        final List<Derived> strings = new ArrayList<>();
        final Map<String, String> byText = new LinkedHashMap<>();
        /** Names of the derived numeric columns, so a string function that is handed one can say why it fails. */
        final Set<String> numericNames = new java.util.HashSet<>();
        boolean changed;

        Ctx(TableSchema schema, int firstIndex) {
            this.schema = schema;
            this.firstIndex = firstIndex;
            this.next = firstIndex;
        }

        /** A position where a string-valued expression is allowed (select item, comparison operand, LIKE target...). */
        String value(String text) {
            if (text == null) return null;
            String t = replaceLengths(text);
            if (!hasStringCall(t)) return t;
            String key = "S:" + t.replaceAll("\\s+", " ");
            String known = byText.get(key);
            if (known != null) return known;
            SNode root = new Parser(t, schema, numericNames).parseWhole();
            String name = NativeLeaves.maskName(next++);
            strings.add(new Derived(name, t, root));
            byText.put(key, name);
            changed = true;
            return name;
        }

        /** A position that needs a number (ORDER BY, GROUP BY, aggregate argument, HAVING): only LENGTH is allowed. */
        String numericOnly(String text, String where) {
            if (text == null) return null;
            String t = replaceLengths(text);
            if (hasStringCall(t)) {
                throw new ParquetSqlException("A string-valued expression cannot be used in " + where + " (" + text
                        + "); only LENGTH(...) of a string can. Select the expression and sort or group in your application,"
                        + " or filter with WHERE instead");
            }
            return t;
        }

        private String replaceLengths(String text) {
            Call c = findLength(text, 0);
            if (c == null) return text;
            StringBuilder out = new StringBuilder(text.length());
            int copied = 0;
            while (c != null) {
                out.append(text, copied, c.start());
                String callText = text.substring(c.start(), c.end());
                String key = "L:" + callText.replaceAll("\\s+", " ");
                String name = byText.get(key);
                if (name == null) {
                    SNode arg = new Parser(text.substring(c.argStart(), c.argEnd()), schema, numericNames).parseWhole();
                    name = NativeLeaves.maskName(next++);
                    masks.add(new LengthMask(name, callText, arg, c.func().equals("OCTET_LENGTH")));
                    numericNames.add(name);
                    byText.put(key, name);
                }
                out.append(name);
                copied = c.end();
                c = findLength(text, copied);
            }
            out.append(text, copied, text.length());
            changed = true;
            return out.toString();
        }
    }

    // ------------------------------------------------------------------ string expression parser

    /** Parses one string expression: {@code term ('||' term)*}. */
    private static final class Parser {
        final String t;
        final TableSchema schema;
        final Set<String> numericNames;
        int i;

        Parser(String text, TableSchema schema, Set<String> numericNames) {
            this.t = text;
            this.schema = schema;
            this.numericNames = numericNames;
        }

        SNode parseWhole() {
            SNode n = concat();
            skipWs();
            if (i != t.length()) {
                throw fail("it mixes string functions with something that is not a string expression"
                        + " (string results cannot be used in arithmetic, only compared, selected or passed to other string functions)");
            }
            return n;
        }

        private ParquetSqlException fail(String why) {
            return new ParquetSqlException("Cannot evaluate '" + t.trim() + "': " + why);
        }

        private void skipWs() {
            while (i < t.length() && Character.isWhitespace(t.charAt(i))) i++;
        }

        private boolean at(String s) {
            return t.startsWith(s, i);
        }

        private SNode concat() {
            List<SNode> parts = new ArrayList<>();
            parts.add(term());
            while (true) {
                skipWs();
                if (!at("||")) break;
                i += 2;
                parts.add(term());
            }
            return parts.size() == 1 ? parts.get(0) : new Concat(parts.toArray(new SNode[0]), false);
        }

        private SNode term() {
            skipWs();
            if (i >= t.length()) throw fail("an operand is missing");
            char c = t.charAt(i);
            if (c == '(') {
                i++;
                SNode n = concat();
                skipWs();
                expect(')');
                return n;
            }
            if (c == '\'') {
                StringBuilder sb = new StringBuilder();
                i++;
                while (true) {
                    if (i >= t.length()) throw fail("unterminated string literal");
                    char d = t.charAt(i);
                    if (d == '\'') {
                        if (i + 1 < t.length() && t.charAt(i + 1) == '\'') { sb.append('\''); i += 2; continue; }
                        i++;
                        break;
                    }
                    sb.append(d);
                    i++;
                }
                return new Lit(sb.toString().getBytes(StandardCharsets.UTF_8));
            }
            if (Character.isDigit(c) || ((c == '-' || c == '+') && i + 1 < t.length() && Character.isDigit(t.charAt(i + 1)))) {
                int s = i++;
                while (i < t.length() && (Character.isDigit(t.charAt(i)) || t.charAt(i) == '.')) i++;
                return new Lit(t.substring(s, i).getBytes(StandardCharsets.UTF_8));
            }
            if (Character.isLetter(c) || c == '_') {
                int s = i++;
                while (i < t.length() && (Character.isLetterOrDigit(t.charAt(i)) || t.charAt(i) == '_')) i++;
                String word = t.substring(s, i);
                int save = i;
                skipWs();
                if (i < t.length() && t.charAt(i) == '(') {
                    i++;
                    return function(word.toUpperCase(Locale.ROOT));
                }
                i = save;
                return column(word);
            }
            throw fail("unexpected '" + c + "'");
        }

        private void expect(char c) {
            skipWs();
            if (i >= t.length() || t.charAt(i) != c) throw fail("expected '" + c + "'");
            i++;
        }

        private SNode column(String name) {
            ColumnKind k = schema.kind(name);
            String shown = QuotedIdentifiers.real(name);
            if (numericNames.contains(name)) throw fail("LENGTH returns a number; it cannot be an argument of a string function");
            if (k == null) throw fail("unknown column '" + shown + "'");
            if (k != ColumnKind.STRING) {
                throw fail("column '" + shown + "' is " + k + "; string functions take string columns, string literals or other string expressions");
            }
            return new Col(QuotedIdentifiers.real(name));
        }

        private SNode function(String f) {
            switch (f) {
                case "LOWER", "UPPER" -> {
                    SNode a = concat();
                    expect(')');
                    return new Case(a, f.equals("UPPER"));
                }
                case "TRIM" -> {
                    SNode a = concat();
                    expect(')');
                    return new Trim(a);
                }
                case "SUBSTRING", "SUBSTR" -> {
                    SNode a = concat();
                    expect(',');
                    long start = intLiteral(f + " start");
                    long count = -1;
                    boolean hasCount = false;
                    skipWs();
                    if (i < t.length() && t.charAt(i) == ',') {
                        i++;
                        count = intLiteral(f + " length");
                        hasCount = true;
                        if (count < 0) throw fail(f + " length must not be negative");
                    }
                    expect(')');
                    return new Substr(a, start, count, hasCount);
                }
                case "CONCAT" -> {
                    List<SNode> parts = new ArrayList<>();
                    parts.add(concat());
                    skipWs();
                    while (i < t.length() && t.charAt(i) == ',') {
                        i++;
                        parts.add(concat());
                        skipWs();
                    }
                    expect(')');
                    return new Concat(parts.toArray(new SNode[0]), true);
                }
                case "LENGTH", "CHAR_LENGTH", "CHARACTER_LENGTH", "OCTET_LENGTH" ->
                        throw fail(f + " returns a number; it cannot be an argument of a string function");
                default -> throw fail("'" + f + "' is not a string function this engine supports (LOWER, UPPER, TRIM, SUBSTRING, CONCAT, ||)");
            }
        }

        private long intLiteral(String what) {
            skipWs();
            int s = i;
            if (i < t.length() && (t.charAt(i) == '-' || t.charAt(i) == '+')) {
                i++;
                skipWs();
            }
            int d = i;
            while (i < t.length() && Character.isDigit(t.charAt(i))) i++;
            if (d == i) {
                i = s;
                throw fail(what + " must be an integer literal");
            }
            try {
                long v = Long.parseLong(t.substring(d, i));
                return t.charAt(s) == '-' ? -v : v;
            } catch (NumberFormatException e) {
                throw fail(what + " is out of range");
            }
        }
    }

    // ------------------------------------------------------------------ evaluation tree

    /** One node. {@code eval} returns the byte length of the result in {@link #buf}, or -1 for null. Not thread safe. */
    abstract static class SNode {
        byte[] buf = new byte[32];
        /** Scratch for a parent that evaluates several children before combining them. */
        int len;

        abstract int eval(int row);

        void bind(VectorSchemaRoot src) { }

        final byte[] room(int n) {
            if (n > buf.length) buf = new byte[Math.max(n, buf.length * 2)];
            return buf;
        }
    }

    private static final class Lit extends SNode {
        private final int len;

        Lit(byte[] v) {
            this.buf = v.length == 0 ? new byte[1] : v;
            this.len = v.length;
        }

        @Override int eval(int row) { return len; }
    }

    private static final class Col extends SNode {
        private final String name;
        private VarCharVector v;

        Col(String name) {
            this.name = name;
        }

        @Override void bind(VectorSchemaRoot src) {
            FieldVector f = src.getVector(name);
            if (f == null) throw new ParquetSqlException("Column '" + name + "' is missing from the scan");
            v = (VarCharVector) f;
        }

        @Override int eval(int row) {
            if (v.isNull(row)) return -1;
            int s = v.getStartOffset(row), len = v.getEndOffset(row) - s;
            room(len);
            if (len > 0) v.getDataBuffer().getBytes(s, buf, 0, len);
            return len;
        }
    }

    /** Wraps {@code child}; binds it. */
    private abstract static class Unary extends SNode {
        final SNode child;

        Unary(SNode child) {
            this.child = child;
        }

        @Override void bind(VectorSchemaRoot src) { child.bind(src); }
    }

    private static final class Case extends Unary {
        private final boolean upper;

        Case(SNode child, boolean upper) {
            super(child);
            this.upper = upper;
        }

        @Override int eval(int row) {
            int len = child.eval(row);
            if (len < 0) return -1;
            byte[] in = child.buf;
            boolean ascii = true;
            for (int k = 0; k < len; k++) {
                if (in[k] < 0) { ascii = false; break; }
            }
            if (ascii) {
                byte[] out = room(len);
                if (upper) {
                    for (int k = 0; k < len; k++) { byte b = in[k]; out[k] = (b >= 'a' && b <= 'z') ? (byte) (b - 32) : b; }
                } else {
                    for (int k = 0; k < len; k++) { byte b = in[k]; out[k] = (b >= 'A' && b <= 'Z') ? (byte) (b + 32) : b; }
                }
                return len;
            }
            String s = new String(in, 0, len, StandardCharsets.UTF_8);
            byte[] mapped = (upper ? s.toUpperCase(Locale.ROOT) : s.toLowerCase(Locale.ROOT)).getBytes(StandardCharsets.UTF_8);
            System.arraycopy(mapped, 0, room(mapped.length), 0, mapped.length);
            return mapped.length;
        }
    }

    private static final class Trim extends Unary {
        Trim(SNode child) {
            super(child);
        }

        @Override int eval(int row) {
            int len = child.eval(row);
            if (len < 0) return -1;
            byte[] in = child.buf;
            int a = 0, b = len;
            while (a < b && in[a] == ' ') a++;
            while (b > a && in[b - 1] == ' ') b--;
            int n = b - a;
            System.arraycopy(in, a, room(n), 0, n);
            return n;
        }
    }

    private static final class Substr extends Unary {
        private final long start, count;
        private final boolean hasCount;

        Substr(SNode child, long start, long count, boolean hasCount) {
            super(child);
            this.start = start;
            this.count = count;
            this.hasCount = hasCount;
        }

        @Override int eval(int row) {
            int len = child.eval(row);
            if (len < 0) return -1;
            byte[] in = child.buf;
            long from = Math.max(start, 1);                       // 1-based first character kept
            long toExcl = hasCount ? start + count : Long.MAX_VALUE; // first character not kept
            if (toExcl <= from) return 0;
            // byte offsets of character positions [from, toExcl)
            long ch = 0;
            int bStart = -1, bEnd = len;
            for (int k = 0; k < len; k++) {
                if ((in[k] & 0xC0) == 0x80) continue; // continuation byte
                ch++;
                if (ch == from) bStart = k;
                if (ch == toExcl) { bEnd = k; break; }
            }
            if (bStart < 0) return 0; // starts past the end
            int n = bEnd - bStart;
            System.arraycopy(in, bStart, room(n), 0, n);
            return n;
        }
    }

    private static final class Concat extends SNode {
        private final SNode[] parts;
        private final boolean skipNulls;

        Concat(SNode[] parts, boolean skipNulls) {
            this.parts = parts;
            this.skipNulls = skipNulls;
        }

        @Override void bind(VectorSchemaRoot src) { for (SNode p : parts) p.bind(src); }

        @Override int eval(int row) {
            int total = 0;
            // children keep their results in their own buffers, so evaluate all first, then copy
            for (SNode p : parts) {
                int l = p.eval(row);
                if (l < 0) {
                    if (!skipNulls) return -1;
                    p.len = -1;
                } else {
                    p.len = l;
                    total += l;
                }
            }
            byte[] out = room(total);
            int at = 0;
            for (SNode p : parts) {
                if (p.len > 0) {
                    System.arraycopy(p.buf, 0, out, at, p.len);
                    at += p.len;
                }
            }
            return total;
        }
    }

    // ------------------------------------------------------------------ kernels

    /** {@code LENGTH}/{@code OCTET_LENGTH}: a numeric derived column, null for a null string. */
    static final class LengthMask extends NativeLeaves.Mask {
        private final SNode root;
        private final boolean octets;

        LengthMask(String name, String text, SNode root, boolean octets) {
            super(name, text + " (derived Float8 column)");
            this.root = root;
            this.octets = octets;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            root.bind(src);
            for (int i = 0; i < n; i++) {
                int len = root.eval(i);
                if (len < 0) {
                    out.setNull(i);
                } else if (octets) {
                    out.set(i, len);
                } else {
                    byte[] b = root.buf;
                    int chars = 0;
                    for (int k = 0; k < len; k++) if ((b[k] & 0xC0) != 0x80) chars++;
                    out.set(i, chars);
                }
            }
        }
    }

    /** The reusable derived {@code VarChar} vectors of one plan; released after every execution. */
    static final class Holder implements AutoCloseable {
        private final List<Derived> defs;
        private VarCharVector[] vecs;
        private BufferAllocator owner;

        Holder(List<Derived> defs) {
            this.defs = defs;
        }

        boolean isEmpty() {
            return defs.isEmpty();
        }

        /** {@code src} plus one derived column per definition, evaluated for rows {@code [0, n)}. The result is a holder: do not close it. */
        VectorSchemaRoot extend(VectorSchemaRoot src, BufferAllocator alloc) {
            if (defs.isEmpty()) return src;
            if (vecs == null || owner != alloc) {
                release();
                vecs = new VarCharVector[defs.size()];
                owner = alloc;
                for (int d = 0; d < vecs.length; d++) {
                    vecs[d] = new VarCharVector(defs.get(d).name, alloc);
                    vecs[d].allocateNew();
                }
            }
            int n = src.getRowCount();
            List<FieldVector> all = new ArrayList<>(src.getFieldVectors());
            List<Field> fields = new ArrayList<>(all.size() + vecs.length);
            for (FieldVector f : all) fields.add(f.getField());
            for (int d = 0; d < vecs.length; d++) {
                SNode root = defs.get(d).root;
                VarCharVector out = vecs[d];
                root.bind(src);
                out.reset();
                for (int i = 0; i < n; i++) {
                    int len = root.eval(i);
                    if (len < 0) out.setNull(i);
                    else out.setSafe(i, root.buf, 0, len);
                }
                out.setValueCount(n);
                all.add(out);
                fields.add(out.getField());
            }
            return new VectorSchemaRoot(new Schema(fields), all, n);
        }

        void release() {
            if (vecs != null) {
                for (VarCharVector v : vecs) if (v != null) v.close();
                vecs = null;
                owner = null;
            }
        }

        @Override
        public void close() {
            release();
        }
    }
}
