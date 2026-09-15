package com.github.gbenroscience.sqlv1.ast;

/**
 * One {@code order_item ::= expression [ASC|DESC]} in an {@code ORDER BY}
 * clause.
 *
 * @param exprText raw ParserNG expression text, captured verbatim from the
 * source SQL exactly like {@link SelectItem#exprText()}
 * @param descending {@code true} for {@code DESC}; {@code false} for
 * {@code ASC} or an unqualified item (SQL's own default sort direction)
 *
 * @author GBEMIRO
 */
public record OrderItem(String exprText, boolean descending) {
}
