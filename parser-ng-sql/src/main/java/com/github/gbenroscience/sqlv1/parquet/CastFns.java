package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.QuotedIdentifiers;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Numeric {@code CAST(x AS INTEGER | SMALLINT | BIGINT | DOUBLE | FLOAT | REAL | NUMERIC | DECIMAL)}.
 *
 * <p>{@code ParquetSql} asks the parser to keep these casts as SQL text. A cast of a <em>number</em> is then
 * rewritten into exactly the arithmetic the parser would have produced ({@code (x)} for the floating
 * targets, truncation toward zero for the integer ones), which the numeric engine evaluates. A cast of a
 * <em>string</em> (a string column, a string literal, a string function or {@code ||}, a nested string leaf)
 * cannot be arithmetic: it becomes a derived {@code Float8} column that parses the value's bytes strictly
 * ({@link NumberText}). Integer targets accept {@code [+-]digits} only and check the target's range; the
 * floating targets accept {@code [+-]digits[.digits][e[+-]digits]}. Anything else, or an empty string, fails
 * loudly naming the expression and the value; a null stays null. The result is {@code Float8}, so an integer
 * beyond 2^53 is an error rather than a rounded value.
 *
 * <p>{@code DATE}, {@code TIMESTAMP}, {@code VARCHAR} and {@code DECIMAL(p,s)} casts are left for
 * {@link TemporalFns} and {@link StringFns}.
 */
final class CastFns {

    private CastFns() { }

    record Result(SelectStatement stmt, List<NativeLeaves.Mask> masks, int used) { }

    private static final Pattern INSIDE = Pattern.compile("(?is)^(.*)\\s+AS\\s+([A-Za-z]+)\\s*$");
    private static final Set<String> INTS = Set.of("INTEGER", "SMALLINT", "BIGINT");
    private static final Set<String> FLOATS = Set.of("DOUBLE", "FLOAT", "REAL", "NUMERIC", "DECIMAL");

    static boolean containsCall(String text) {
        if (text == null) return false;
        int from = 0;
        StringFns.Call c;
        while ((c = StringFns.findCall(text, from, Set.of("CAST"))) != null) {
            Matcher m = INSIDE.matcher(text.substring(c.argStart(), c.argEnd()));
            if (m.matches()) {
                String t = m.group(2).toUpperCase(Locale.ROOT);
                if (INTS.contains(t) || FLOATS.contains(t)) return true;
            }
            from = c.argStart(); // look inside as well: a numeric cast may sit in the argument of another
        }
        return false;
    }

    static Result rewrite(SelectStatement st, TableSchema schema, int firstIndex) {
        Ctx ctx = new Ctx(schema, firstIndex);
        SelectStatement out = StatementRewriter.map(st, ctx::text);
        return new Result(out, List.copyOf(ctx.masks), ctx.next - firstIndex);
    }

    private static final class Ctx {
        final TableSchema schema;
        int next;
        final List<NativeLeaves.Mask> masks = new ArrayList<>();
        final Map<String, String> byKey = new LinkedHashMap<>();

        Ctx(TableSchema schema, int firstIndex) {
            this.schema = schema;
            this.next = firstIndex;
        }

        /** Rewrites every numeric cast in {@code text}, innermost first. */
        String text(String text) {
            if (!containsCall(text)) return text;
            StringBuilder out = new StringBuilder(text.length());
            int copied = 0;
            StringFns.Call c;
            int from = 0;
            while ((c = StringFns.findCall(text, from, Set.of("CAST"))) != null) {
                Matcher m = INSIDE.matcher(text.substring(c.argStart(), c.argEnd()));
                if (!m.matches()) {
                    from = c.end();
                    continue;
                }
                String type = m.group(2).toUpperCase(Locale.ROOT);
                boolean numeric = INTS.contains(type) || FLOATS.contains(type);
                String inner = text(m.group(1)); // casts nested in the operand first
                out.append(text, copied, c.start());
                if (!numeric) {
                    out.append("CAST(").append(inner).append(" AS ").append(m.group(2)).append(')');
                } else {
                    out.append(replacement(text.substring(c.start(), c.end()), inner, type));
                }
                copied = from = c.end();
            }
            return out.append(text, copied, text.length()).toString();
        }

        private String replacement(String callText, String inner, String type) {
            String t = inner.trim();
            boolean stringy = t.startsWith("'") || StringFns.hasStringCall(t);
            if (!stringy) {
                ColumnKind k = schema.kind(t);
                stringy = k == ColumnKind.STRING;
            }
            if (!stringy) {
                return INTS.contains(type) ? "((" + inner + ") - ((" + inner + ") % 1))" : "(" + inner + ")";
            }
            String key = type + ":" + t.replaceAll("\\s+", " ");
            String known = byKey.get(key);
            if (known != null) return known;
            StringFns.SNode root = StringFns.parseStringExpression(t, schema);
            String name = NativeLeaves.maskName(next++);
            String shown = QuotedIdentifiers.decodeAll(callText);
            masks.add(new StringToNumber(name, shown, root, INTS.contains(type) ? NumberText.IntType.valueOf(type) : null));
            byKey.put(key, name);
            return name;
        }
    }

    /** {@code CAST(string AS number)}: parses the bytes of each value. */
    static final class StringToNumber extends NativeLeaves.Mask {
        private final StringFns.SNode root;
        private final NumberText.IntType intType; // null for the floating targets
        private final String text;

        StringToNumber(String name, String text, StringFns.SNode root, NumberText.IntType intType) {
            super(name, text + " (derived Float8 column)");
            this.root = root;
            this.intType = intType;
            this.text = text;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            root.bind(src);
            for (int i = 0; i < n; i++) {
                int len = root.eval(i);
                if (len < 0) out.setNull(i);
                else out.set(i, intType != null ? NumberText.parseInteger(root.buf, len, intType, text) : NumberText.parseDouble(root.buf, len, text));
            }
        }
    }
}
