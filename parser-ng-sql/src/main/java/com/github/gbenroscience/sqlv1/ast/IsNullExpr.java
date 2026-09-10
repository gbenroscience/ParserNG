package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code expression 'IS' ['NOT'] 'NULL'}.
 *
 * <p>
 * Unlike every other {@link BoolExpr} leaf, this one cannot be rendered into
 * ParserNG expression text at all: parser-ng-arrow's bulk evaluators operate
 * on raw data buffers and never expose a null-test operator through
 * {@code MathExpression} (see {@code NullPolicy}'s javadoc in
 * parser-ng-arrow — validity bitmaps are handled as a separate pass, not as
 * part of expression evaluation). {@code ArrowQuery} evaluates this leaf
 * itself, directly against Arrow validity bitmaps, rather than delegating to
 * a compiled ParserNG expression the way every other leaf type does — see
 * {@code ArrowQuery}'s "Null handling" section.
 *
 * @param target raw ParserNG expression text being null-tested
 * @param negated {@code true} for {@code IS NOT NULL}
 *
 * @author GBEMIRO
 */
public record IsNullExpr(String target, boolean negated) implements BoolExpr {

    /**
     * @return an equivalent node with {@code negated} flipped — used by
     * {@link BoolExprs#toNnf(BoolExpr)} to push a {@code NOT} down through
     * this leaf without needing a runtime logical-not operator
     */
    public IsNullExpr negate() {
        return new IsNullExpr(target, !negated);
    }
}
