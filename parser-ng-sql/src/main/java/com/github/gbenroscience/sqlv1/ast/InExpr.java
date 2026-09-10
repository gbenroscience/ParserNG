package com.github.gbenroscience.sqlv1.ast;

import java.util.List;

/**
 * {@code expression ['NOT'] 'IN' '(' expression_list ')'}.
 *
 * <p>
 * Renders (see {@link BoolExprs#renderLeaf(BoolExpr)}) to
 * {@code (target == v1 || target == v2 || ...)} when {@code negated} is
 * {@code false}, or {@code (target != v1 && target != v2 && ...)} when
 * {@code true} — ParserNG has no {@code IN} operator of its own, so
 * membership is expanded into a disjunction/conjunction of equality
 * comparisons.
 *
 * @param target raw ParserNG expression text being tested for membership
 * @param values raw ParserNG expression text for each candidate value, in
 * order; never empty (the grammar requires at least one entry in
 * {@code expression_list})
 * @param negated {@code true} for {@code NOT IN}
 *
 * @author GBEMIRO
 */
public record InExpr(String target, List<String> values, boolean negated) implements BoolExpr {

    public InExpr {
        values = List.copyOf(values);
        if (values.isEmpty()) {
            throw new IllegalArgumentException("IN requires at least one value");
        }
    }

    /**
     * @return an equivalent node with {@code negated} flipped — used by
     * {@link BoolExprs#toNnf(BoolExpr)} to push a {@code NOT} down through
     * this leaf without needing a runtime logical-not operator
     */
    public InExpr negate() {
        return new InExpr(target, values, !negated);
    }
}
