package com.github.gbenroscience.sqlv1.ast;

/**
 * {@code order_item ::= expression ['ASC' | 'DESC']} — one key of an
 * {@code ORDER BY} clause.
 *
 * @param exprText raw ParserNG expression text to sort by, captured
 * verbatim from the source SQL (may name a {@code SELECT}-list alias — see
 * {@code ArrowQuery}'s handling of {@code ORDER BY})
 * @param descending {@code true} for {@code DESC}; {@code false} (the
 * default when neither {@code ASC} nor {@code DESC} is written) sorts
 * ascending
 *
 * @author GBEMIRO
 */
public record OrderItem(String exprText, boolean descending) {
}