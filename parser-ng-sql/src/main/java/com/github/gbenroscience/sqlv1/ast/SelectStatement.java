package com.github.gbenroscience.sqlv1.ast;

import java.util.List;

/**
 * {@code query ::= SELECT select_list FROM table_reference [WHERE
 * boolean_expression] [GROUP BY expression_list] [HAVING
 * boolean_expression] [ORDER BY order_item (',' order_item)*] [LIMIT
 * integer_literal]} — the result of parsing one SQL statement.
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
 * <h2>{@code GROUP BY} / aggregates — a deliberately strict subset</h2>
 * A query {@link #isGrouped()} iff it either has a non-empty
 * {@link #groupBy()} or its {@link #items()} contain at least one
 * {@link SelectItem#isAggregate() aggregate} item (the standard SQL
 * behavior for aggregates with no explicit {@code GROUP BY}: the whole
 * table is one implicit group). For such a query, {@code ArrowQuery}
 * requires <em>every</em> non-aggregate {@code SELECT} item's
 * {@link SelectItem#exprText()} to match one of {@link #groupBy()}'s
 * expressions exactly (after trimming) — the strict/standard reading of
 * {@code GROUP BY} (PostgreSQL's, not MySQL's historical relaxed mode). A
 * non-aggregate, non-group-key item is a compile-time error, not a silently
 * arbitrary "first row of the group" pick.
 *
 * @param selectAll {@code true} for {@code SELECT *}, in which case
 * {@link #items()} is empty ({@code SELECT *} can never be combined with
 * {@code GROUP BY}/aggregates — there is no grammar production for it, see
 * {@code SqlParser})
 * @param items the SELECT list, in source order; empty iff
 * {@link #selectAll()} is {@code true}
 * @param table the identifier named after {@code FROM}
 * @param where the parsed {@code WHERE} clause, already normalized to
 * negation-normal form (see {@link BoolExprs#toNnf(BoolExpr)}), or
 * {@code null} if the query has no {@code WHERE} clause
 * @param groupBy raw ParserNG expression text for each {@code GROUP BY}
 * key, in source order; empty if the query has no {@code GROUP BY} clause
 * (which does not by itself mean the query isn't grouped — see
 * {@link #isGrouped()})
 * @param having the parsed {@code HAVING} clause, already normalized to
 * negation-normal form, or {@code null} if the query has no {@code HAVING}
 * clause; only meaningful when {@link #isGrouped()}
 * @param orderBy the {@code ORDER BY} keys, in source (i.e. sort-priority)
 * order; empty if the query has no {@code ORDER BY} clause
 * @param limit the {@code LIMIT} row cap, or {@code null} if the query has
 * no {@code LIMIT} clause
 *
 * @author GBEMIRO
 */
public record SelectStatement(
        boolean selectAll,
        List<SelectItem> items,
        String table,
        BoolExpr where,
        List<String> groupBy,
        BoolExpr having,
        List<OrderItem> orderBy,
        Integer limit) {

    public SelectStatement {
        items = List.copyOf(items);
        groupBy = List.copyOf(groupBy);
        orderBy = List.copyOf(orderBy);
        if (selectAll && !items.isEmpty()) {
            throw new IllegalArgumentException("selectAll queries must not also carry select items");
        }
        if (!selectAll && items.isEmpty()) {
            throw new IllegalArgumentException("non-selectAll queries must have at least one select item");
        }
        if (selectAll && !groupBy.isEmpty()) {
            throw new IllegalArgumentException("SELECT * cannot be combined with GROUP BY");
        }
        if (having != null && groupBy.isEmpty() && items.stream().noneMatch(SelectItem::isAggregate)) {
            throw new IllegalArgumentException("HAVING requires GROUP BY or at least one aggregate SELECT item");
        }
        if (limit != null && limit < 0) {
            throw new IllegalArgumentException("LIMIT must be non-negative, was " + limit);
        }
    }

    /**
     * @return {@code true} iff this query aggregates rows into groups —
     * either an explicit {@code GROUP BY}, or at least one aggregate
     * {@code SELECT} item with no {@code GROUP BY} (the whole table is then
     * one implicit group, standard SQL behavior)
     */
    public boolean isGrouped() {
        return !groupBy.isEmpty() || items.stream().anyMatch(SelectItem::isAggregate);
    }
}
