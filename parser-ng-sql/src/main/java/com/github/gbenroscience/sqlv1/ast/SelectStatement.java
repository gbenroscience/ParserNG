package com.github.gbenroscience.sqlv1.ast;

import java.util.List;

/**
 * {@code query ::= SELECT select_list FROM table_reference [WHERE
 * boolean_expression]} — the result of parsing one SQL statement.
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
 * @param selectAll {@code true} for {@code SELECT *}, in which case
 * {@link #items()} is empty
 * @param items the SELECT list, in source order; empty iff
 * {@link #selectAll()} is {@code true}
 * @param table the identifier named after {@code FROM}
 * @param where the parsed {@code WHERE} clause, already normalized to
 * negation-normal form (see {@link BoolExprs#toNnf(BoolExpr)}), or
 * {@code null} if the query has no {@code WHERE} clause
 *
 * @author GBEMIRO
 */
public record SelectStatement(boolean selectAll, List<SelectItem> items, String table, BoolExpr where) {

    public SelectStatement {
        items = List.copyOf(items);
        if (selectAll && !items.isEmpty()) {
            throw new IllegalArgumentException("selectAll queries must not also carry select items");
        }
        if (!selectAll && items.isEmpty()) {
            throw new IllegalArgumentException("non-selectAll queries must have at least one select item");
        }
    }
}
