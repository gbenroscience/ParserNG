package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code target IS [NOT] TRUE|FALSE}. Unlike a comparison this is two-valued: it never yields
 * unknown. {@code x IS TRUE} matches only a non-null true; {@code x IS NOT TRUE} matches false
 * <em>and</em> null. So {@code NOT (x IS TRUE)} is exactly {@code x IS NOT TRUE}, which is what
 * {@link #negate()} produces.
 *
 * <p>A bare boolean column ({@code WHERE flag}) is not this node: it parses as
 * {@code flag = TRUE}, a three-valued comparison, so {@code NOT flag} keeps SQL null semantics.
 * Evaluated only by {@code ParquetSql}.
 *
 * @param target the boolean operand, raw expression text
 * @param value {@code true} for {@code IS [NOT] TRUE}, {@code false} for {@code IS [NOT] FALSE}
 * @param negated {@code true} for {@code IS NOT ...}
 */
public record TruthExpr(String target, boolean value, boolean negated) implements BoolExpr {

    public TruthExpr negate() {
        return new TruthExpr(target, value, !negated);
    }
}
