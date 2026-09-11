package com.github.gbenroscience.sqlv1.ast;

import java.util.List;

/**
 * {@code query ::= SELECT select_list FROM table_reference [WHERE
 * boolean_expression] [GROUP BY expression_list] [HAVING boolean_expression]
 * [ORDER BY order_item (',' order_item)*] [LIMIT integer_literal]} — the
 * result of parsing one SQL statement.
 *
 * <p>
 * {@code table} is carried through purely for readability/self-documentation
 * of the query text (e.g. {@code "FROM points"}); {@code ArrowQuery} does
 * not use it to look anything up. This module operates on a single
 * already-in-hand {@code VectorSchemaRoot} per {@code execute} call — there
 * is no table catalog, so nothing resolves {@code table} against a real
 * data source. See the parser-ng-sql module javadoc for why (no JOIN, no
 * catalogs, "not a database").
 *
 * <h2>Aggregate/{@code GROUP BY} validity</h2>
 * A query is an <i>aggregate query</i> ({@link #isAggregateQuery()}) if it
 * has a non-empty {@link #groupBy()} or at least one aggregate
 * {@link SelectItem} (see {@link SelectItem#isAggregate()}) — including a
 * query with aggregates but no explicit {@code GROUP BY} at all, which
 * treats the entire (filtered) result as one implicit group, exactly as
 * standard SQL does. For an aggregate query, every non-aggregate select
 * item's {@link SelectItem#exprText()} must equal (after trimming) one of
 * the {@link #groupBy()} expressions verbatim — the usual SQL rule that a
 * plain column in the select list of a grouped query must be functionally
 * determined by the grouping key. {@code SELECT *} can never be combined
 * with {@code GROUP BY}/{@code HAVING}/an aggregate item, since there is no
 * select list to reconcile against the grouping key. These are enforced
 * here, at construction, rather than downstream in {@code ArrowQuery}.
 *
 * @param selectAll {@code true} for {@code SELECT *}, in which case
 * {@link #items()} is empty
 * @param items the SELECT list, in source order; empty iff
 * {@link #selectAll()} is {@code true}
 * @param table the identifier named after {@code FROM}
 * @param where the parsed {@code WHERE} clause, already normalized to
 * negation-normal form (see {@link BoolExprs#toNnf(BoolExpr)}), or
 * {@code null} if the query has no {@code WHERE} clause
 * @param groupBy raw ParserNG expression text for each {@code GROUP BY}
 * key, in source order; empty if the query has no {@code GROUP BY}
 * @param having the parsed {@code HAVING} clause (same representation and
 * normalization as {@code where}), or {@code null} if the query has no
 * {@code HAVING} clause
 * @param orderBy the {@code ORDER BY} keys, in source order; empty if the
 * query has no {@code ORDER BY}
 * @param limit the {@code LIMIT} row cap, or {@code null} if the query has
 * no {@code LIMIT}
 *
 * @author GBEMIRO
 */
public record SelectStatement(
        boolean selectAll, List<SelectItem> items, String table, BoolExpr where,
        List<String> groupBy, BoolExpr having, List<OrderItem> orderBy, Integer limit) {

    public SelectStatement {
        items = List.copyOf(items);
        groupBy = groupBy == null ? List.of() : List.copyOf(groupBy);
        orderBy = orderBy == null ? List.of() : List.copyOf(orderBy);

        if (selectAll && !items.isEmpty()) {
            throw new IllegalArgumentException("selectAll queries must not also carry select items");
        }
        if (!selectAll && items.isEmpty()) {
            throw new IllegalArgumentException("non-selectAll queries must have at least one select item");
        }
        if (limit != null && limit < 0) {
            throw new IllegalArgumentException("LIMIT must not be negative");
        }

        boolean hasAggregateItem = items.stream().anyMatch(SelectItem::isAggregate);
        boolean aggregateQuery = !groupBy.isEmpty() || hasAggregateItem;

        if (selectAll && (!groupBy.isEmpty() || having != null)) {
            throw new IllegalArgumentException("SELECT * cannot be combined with GROUP BY/HAVING");
        }
        if (having != null && !aggregateQuery) {
            throw new IllegalArgumentException("HAVING requires GROUP BY or at least one aggregate select item");
        }
        if (aggregateQuery) {
            java.util.Set<String> groupByTexts = new java.util.HashSet<>();
            for (String gb : groupBy) {
                groupByTexts.add(gb.trim());
            }
            for (SelectItem item : items) {
                if (!item.isAggregate() && !groupByTexts.contains(item.exprText().trim())) {
                    throw new IllegalArgumentException(
                            "Column '" + item.exprText() + "' must appear in GROUP BY or be used inside an "
                            + "aggregate function (COUNT/SUM/AVG/MIN/MAX)");
                }
            }
        }
    }

    /**
     * @return {@code true} if this query groups/aggregates its result — a
     * non-empty {@link #groupBy()}, or at least one aggregate
     * {@link SelectItem} even with no explicit {@code GROUP BY} (the whole
     * result is then one implicit group, as in plain SQL)
     */
    public boolean isAggregateQuery() {
        return !groupBy.isEmpty() || items.stream().anyMatch(SelectItem::isAggregate);
    }
}