package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code expression comparison_operator expression} — a single comparison.
 *
 * @param left raw ParserNG expression text for the left-hand operand
 * @param op the comparison operator
 * @param right raw ParserNG expression text for the right-hand operand
 *
 * @author GBEMIRO
 */
public record ComparisonExpr(String left, CompOp op, String right) implements BoolExpr {
}
