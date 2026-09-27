package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.*;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Pure planning step: SelectStatement + the file's column names -> what the Parquet layer should be
 * asked for. No I/O, no Arrow, no Parquet types: unit-testable in isolation.
 *
 * <p>Note what is NOT here: there is no "residual predicate". Parquet pruning can only return a
 * superset, so the query's full WHERE always stays in ArrowQuery. {@link ScanPlan#pruning()} is a
 * derived hint for skipping row groups, nothing more.
 */
public final class ScanPlanner {

    private ScanPlanner() { }

    /**
     * @param columns physical columns to decode, in file order; {@code null} means all (SELECT *)
     * @param pruning pushdown-safe part of WHERE, or null if nothing could be derived
     * @param streamable true if batches may be run through ArrowQuery independently and concatenated
     * @param materializeReason why not streamable (null when streamable)
     * @param limit the query's LIMIT (null if none); enables early stop when streamable
     */
    public record ScanPlan(List<String> columns, Predicate pruning, boolean streamable,
                           String materializeReason, Integer limit) { }

    /**
     * @param fileColumns every top-level column of the file
     * @param readableColumns the subset a flat scan can decode (used only for the COUNT(*) fallback)
     */
    public static ScanPlan plan(SelectStatement st, Collection<String> fileColumns, Collection<String> readableColumns) {
        Set<String> fileSet = new HashSet<>(fileColumns);

        List<String> columns = null;
        if (!st.selectAll()) {
            Set<String> referenced = ColumnRefs.referencedBy(st);
            columns = new ArrayList<>();
            for (String c : fileColumns) if (referenced.contains(c)) columns.add(c);
            if (columns.isEmpty()) {
                // e.g. SELECT COUNT(*) FROM t: no column is needed, but a scan must decode something to
                // learn row counts. Take one readable column rather than failing.
                if (readableColumns.isEmpty()) {
                    throw new IllegalArgumentException("Query references no file column and the file has no flat readable column");
                }
                columns.add(readableColumns.iterator().next());
            }
        }

        Predicate pruning = PredicateConverter.convert(st.where(), fileSet);

        String reason = null;
        if (st.isGrouped()) reason = "GROUP BY / aggregates need all matching rows before producing output";
        else if (!st.orderBy().isEmpty()) reason = "ORDER BY needs all matching rows before producing output";
        return new ScanPlan(columns, pruning, reason == null, reason, st.limit());
    }
}
