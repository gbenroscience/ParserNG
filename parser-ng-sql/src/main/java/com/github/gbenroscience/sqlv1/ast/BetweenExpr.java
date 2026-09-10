package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code expression ['NOT'] 'BETWEEN' expression 'AND' expression}.
 *
 * <p>
 * Renders (see {@link BoolExprs#renderLeaf(BoolExpr)}) to
 * {@code (target >= low && target <= high)} when {@code negated} is
 * {@code false}, or {@code (target < low || target > high)} when
 * {@code true} — the standard expansion of {@code BETWEEN}/{@code NOT
 * BETWEEN} into ParserNG's native comparison and logical operators, since
 * ParserNG has no {@code BETWEEN} operator of its own.
 *
 * @param target raw ParserNG expression text being range-tested
 * @param low raw ParserNG expression text for the lower bound (inclusive)
 * @param high raw ParserNG expression text for the upper bound (inclusive)
 * @param negated {@code true} for {@code NOT BETWEEN}
 *
 * @author GBEMIRO
 */
public record BetweenExpr(String target, String low, String high, boolean negated) implements BoolExpr {

    /**
     * @return an equivalent node with {@code negated} flipped — used by
     * {@link BoolExprs#toNnf(BoolExpr)} to push a {@code NOT} down through
     * this leaf without needing a runtime logical-not operator
     */
    public BetweenExpr negate() {
        return new BetweenExpr(target, low, high, !negated);
    }
}
