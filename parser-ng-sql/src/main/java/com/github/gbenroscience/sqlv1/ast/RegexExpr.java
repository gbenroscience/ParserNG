package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code target [NOT] REGEXP|RLIKE pattern}: {@code java.util.regex} {@code find()} semantics (the
 * pattern may match anywhere in the value). Evaluated only by {@code ParquetSql}. A null target
 * never matches, for the negated form as well.
 *
 * @param target the value being matched, raw expression text
 * @param pattern the pattern operand, raw expression text (a quoted literal)
 * @param negated {@code true} for {@code NOT REGEXP}
 */
public record RegexExpr(String target, String pattern, boolean negated) implements BoolExpr {

    public RegexExpr negate() {
        return new RegexExpr(target, pattern, !negated);
    }
}
