package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code target [NOT] LIKE|ILIKE pattern [ESCAPE 'c']}. {@code %} matches any run of characters
 * (including none), {@code _} exactly one character. Operands are raw expression text, as in the
 * other leaves; the pattern is normally a quoted string literal.
 *
 * <p>Only {@code ParquetSql} evaluates this leaf (it is string-typed; the numeric engine has no
 * rendering for it). A null target never matches, for {@code LIKE} and {@code NOT LIKE} alike.
 *
 * @param target the value being matched, raw expression text
 * @param pattern the pattern operand, raw expression text (a quoted literal)
 * @param escape the {@code ESCAPE} operand, raw expression text (a one-character quoted literal), or {@code null}
 * @param caseInsensitive {@code true} for {@code ILIKE}
 * @param negated {@code true} for {@code NOT LIKE}/{@code NOT ILIKE}
 */
public record LikeExpr(String target, String pattern, String escape, boolean caseInsensitive, boolean negated)
        implements BoolExpr {

    public LikeExpr negate() {
        return new LikeExpr(target, pattern, escape, caseInsensitive, !negated);
    }
}
