package com.github.gbenroscience.sqlv1.ast;

import java.util.Map;

/**
 * Lets a {@code WHERE} clause name a {@code SELECT}-list alias by rewriting
 * every such reference, in place, into the (parenthesized) expression the
 * alias stands for — before the clause is rendered/compiled.
 *
 * <h2>Why this exists</h2>
 * {@code ArrowQuery} evaluates {@code WHERE} against the <i>input</i>
 * root's own columns, before projection/aliasing happens — a name in
 * {@code WHERE} that was never bound to a real input column simply fails to
 * bind at evaluate time. That is still true of the underlying evaluation
 * strategy; what changes here is that a name matching a {@code SELECT}-list
 * alias is no longer treated as "never bound" — it is expanded back into
 * the real, input-column-based expression it was an alias <i>for</i>, so
 * {@code WHERE} ends up evaluating something it always could have. This
 * turns
 * <pre>{@code
 * SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM data WHERE magnitude > 60
 * }</pre>
 * into the equivalent of manually repeating the expression:
 * <pre>{@code
 * SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM data WHERE (sqrt(x*x + y*y)) > 60
 * }</pre>
 *
 * <h2>Scoping</h2>
 * A real input column always wins over a same-named alias — callers are
 * expected to omit any alias whose {@link SelectItem#outputName()} collides
 * with an actual column of the {@code VectorSchemaRoot} in hand from the
 * {@code aliasToExpr} map before calling {@link #resolve}, since only the
 * caller (who has the concrete schema) can know which names those are. An
 * alias standing for a plain passthrough column (no computation, no
 * {@code AS}) never needs to appear in that map either — its own output
 * name already resolves to a real column and reaches {@code WHERE} directly.
 *
 * <h2>Chained aliases and cycles</h2>
 * Substitution runs to a fixed point, so an alias whose own expression text
 * references a second alias resolves transitively (bounded by
 * {@code aliasToExpr.size() + 1} passes — enough for any acyclic chain
 * through every entry in the map exactly once). A genuine cycle (an alias
 * that, however indirectly, refers back to itself) fails fast with an
 * {@link IllegalArgumentException} rather than looping forever.
 *
 * <h2>What is, and is not, touched</h2>
 * Substitution is identifier-aware, not a blind string replace: it walks
 * each leaf's raw expression text character by character, skips over
 * quoted string-literal regions verbatim (so an alias name that happens to
 * also appear inside a string literal is left alone), and only rewrites a
 * bare identifier — never one immediately followed by {@code (}, which
 * marks a function call rather than a variable reference, even if the
 * function happens to share its name with an alias.
 *
 * @author GBEMIRO
 */
public final class WhereAliasResolver {

    private WhereAliasResolver() {
    }

    /**
     * Returns an equivalent {@link BoolExpr} tree with every leaf-level
     * reference to a key of {@code aliasToExpr} rewritten to
     * {@code "(" + aliasToExpr.get(name) + ")"}. {@code where} itself, and
     * every intermediate node, is left structurally unchanged (same
     * {@code AND}/{@code OR}/{@code NOT} shape) — only leaf operand text is
     * ever rewritten.
     *
     * @param where the clause to resolve; {@code null} is returned as-is
     * @param aliasToExpr alias output name -&gt; the raw expression text it
     * stands for. An empty map is a fast no-op path (as opposed to
     * possibly-expensive tree reconstruction).
     * @throws IllegalArgumentException if {@code aliasToExpr} contains a
     * cycle reachable from a name used in {@code where}
     */
    public static BoolExpr resolve(BoolExpr where, Map<String, String> aliasToExpr) {
        if (where == null || aliasToExpr.isEmpty()) {
            return where;
        }
        return switch (where) {
            case AndExpr(var l, var r) -> new AndExpr(resolve(l, aliasToExpr), resolve(r, aliasToExpr));
            case OrExpr(var l, var r) -> new OrExpr(resolve(l, aliasToExpr), resolve(r, aliasToExpr));
            case NotExpr(var inner) -> new NotExpr(resolve(inner, aliasToExpr));
            case ComparisonExpr(var l, var op, var r) ->
                new ComparisonExpr(substitute(l, aliasToExpr), op, substitute(r, aliasToExpr));
            case BetweenExpr(var target, var low, var high, var negated) ->
                new BetweenExpr(substitute(target, aliasToExpr), substitute(low, aliasToExpr),
                        substitute(high, aliasToExpr), negated);
            case InExpr(var target, var values, var negated) ->
                new InExpr(substitute(target, aliasToExpr), values.stream().map(v -> substitute(v, aliasToExpr)).toList(),
                        negated);
            case IsNullExpr(var target, var negated) -> new IsNullExpr(substitute(target, aliasToExpr), negated);
        };
    }

    /**
     * Substitutes every alias reference in {@code exprText}, iterating to a
     * fixed point so a chain of aliases (one alias's expression naming
     * another) resolves fully.
     */
    static String substitute(String exprText, Map<String, String> aliasToExpr) {
        String current = exprText;
        int maxPasses = aliasToExpr.size() + 1;
        for (int pass = 0; pass < maxPasses; pass++) {
            String next = substituteOnce(current, aliasToExpr);
            if (next.equals(current)) {
                return current;
            }
            current = next;
        }
        throw new IllegalArgumentException(
                "Cyclic SELECT-list alias reference reachable from WHERE clause expression \""
                + exprText + "\"");
    }

    /**
     * One identifier-aware substitution pass: copies {@code text} verbatim
     * except that a bare identifier matching a key of {@code aliasToExpr}
     * (and not immediately followed by {@code (}) is replaced with its
     * parenthesized mapped expression. Quoted string-literal regions are
     * copied through untouched, whole, and never scanned for identifiers.
     */
    private static String substituteOnce(String text, Map<String, String> aliasToExpr) {
        int n = text.length();
        StringBuilder out = new StringBuilder(n);
        int i = 0;
        while (i < n) {
            char c = text.charAt(i);
            if (c == '\'' || c == '"') {
                int start = i;
                i = skipStringLiteral(text, i, c);
                out.append(text, start, i);
                continue;
            }
            if (Character.isLetter(c) || c == '_') {
                int start = i;
                i++;
                while (i < n && (Character.isLetterOrDigit(text.charAt(i)) || text.charAt(i) == '_')) {
                    i++;
                }
                String identifier = text.substring(start, i);
                int j = i;
                while (j < n && Character.isWhitespace(text.charAt(j))) {
                    j++;
                }
                boolean isFunctionCall = j < n && text.charAt(j) == '(';
                String replacement = isFunctionCall ? null : aliasToExpr.get(identifier);
                if (replacement != null) {
                    out.append('(').append(replacement).append(')');
                } else {
                    out.append(identifier);
                }
                continue;
            }
            out.append(c);
            i++;
        }
        return out.toString();
    }

    /**
     * @return the index one past the closing {@code quote}, treating a
     * doubled quote character as an escaped literal quote (matching
     * {@code SqlLexer}'s own string-literal escaping rule); the end of
     * {@code text} if the literal is unterminated, so a malformed source
     * string (which would already have failed lexing/parsing earlier, long
     * before a WHERE clause exists to resolve aliases against) can never
     * make this method loop forever
     */
    private static int skipStringLiteral(String text, int start, char quote) {
        int n = text.length();
        int i = start + 1;
        while (i < n) {
            if (text.charAt(i) == quote) {
                if (i + 1 < n && text.charAt(i + 1) == quote) {
                    i += 2;
                    continue;
                }
                return i + 1;
            }
            i++;
        }
        return n;
    }

    /**
     * Public entry point for a single raw expression string (as opposed to
     * a {@link BoolExpr} tree) — used by {@code ArrowQuery} to resolve
     * {@code SELECT}-list aliases referenced from an {@code ORDER BY} key,
     * exactly the same way {@link #resolve(BoolExpr, Map)} resolves them
     * for {@code WHERE}/{@code HAVING} leaf operands. Thin public wrapper
     * around the package-private {@link #substitute(String, Map)}.
     *
     * @param exprText the raw ParserNG expression text to resolve aliases
     * within
     * @param aliasToExpr alias output name -&gt; the raw expression text it
     * stands for; an empty map is a no-op
     * @throws IllegalArgumentException if {@code aliasToExpr} contains a
     * cycle reachable from a name used in {@code exprText}
     */
    public static String substituteText(String exprText, Map<String, String> aliasToExpr) {
        if (aliasToExpr.isEmpty()) {
            return exprText;
        }
        return substitute(exprText, aliasToExpr);
    }
}