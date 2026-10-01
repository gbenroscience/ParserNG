package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import com.github.gbenroscience.sqlv1.ast.AggFunc;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * Turns a parsed statement into the scan-shaped decisions: which columns to
 * decode, which pruning predicate to push, whether the query can be answered
 * from the footer alone, and whether it streams or must materialize. No
 * Arrow types and no file access, so it is unit-testable in isolation.
 *
 * <p>The pruning predicate is a <em>superset</em> filter (see
 * {@link PredicateConverter}); the full WHERE is always re-evaluated downstream.
 */
public final class ScanPlanner {

    private ScanPlanner() { }

    /**
     * @param columns top-level columns to decode, {@code null} for all, empty when {@code metadataOnly}
     * @param pruning superset pruning predicate, or {@code null}
     * @param streamable true if batches can be processed independently
     * @param materializeReason why not streamable, else {@code null}
     * @param limit the LIMIT, or {@code null}
     * @param metadataOnly true if the answer comes from the footer with no data page decoded
     */
    public record ScanPlan(List<String> columns, Predicate pruning, boolean streamable,
                           String materializeReason, Integer limit, boolean metadataOnly) {

        /** Compatibility constructor: the original five-component form (never metadata-only). */
        public ScanPlan(List<String> columns, Predicate pruning, boolean streamable, String materializeReason, Integer limit) {
            this(columns, pruning, streamable, materializeReason, limit, false);
        }
    }

    /**
     * Compatibility overload of the original signature: plans without knowing column types, so
     * literals are conservatively typed as double and only double columns can be pruned. Sound (a
     * mistyped leaf is skipped by parser-ng-parquet, never wrong) but less effective than
     * the schema-aware planner used by {@link ParquetSql}.
     */
    public static ScanPlan plan(SelectStatement st, Collection<String> fileColumns, Collection<String> readableColumns) {
        java.util.Map<String, ColumnKind> kinds = new java.util.LinkedHashMap<>();
        for (String c : fileColumns) kinds.put(c, ColumnKind.FLOAT64);
        return plan(st, fileColumns, readableColumns, TableSchema.ofKinds(kinds));
    }

    static ScanPlan plan(SelectStatement st, Collection<String> fileColumns, Collection<String> readableColumns,
                         TableSchema schema) {
        if (isCountStarOnly(st)) {
            return new ScanPlan(List.of(), null, true, null, st.limit(), true);
        }
        List<String> columns = columnsFor(st, fileColumns, readableColumns);
        Predicate pruning = PredicateConverter.convert(st.where(), schema);

        String reason = null;
        if (st.isGrouped()) reason = "GROUP BY / aggregates need all matching rows before producing output";
        else if (!st.orderBy().isEmpty()) reason = "ORDER BY needs all matching rows before producing output";
        return new ScanPlan(columns, pruning, reason == null, reason, st.limit(), false);
    }

    /**
     * The top-level columns the query needs, in file order; {@code null} for
     * {@code SELECT *}. Computed before the schema is known, so the scan that
     * discovers the schema decodes nothing it does not need.
     */
    static List<String> columnsFor(SelectStatement st, Collection<String> fileColumns, Collection<String> readableColumns) {
        if (st.selectAll()) return null;
        Set<String> referenced = ColumnRefs.referencedBy(st);
        List<String> columns = new ArrayList<>();
        for (String c : fileColumns) if (referenced.contains(c)) columns.add(c);
        if (columns.isEmpty()) {
            if (readableColumns.isEmpty()) {
                throw new ParquetSqlException("Query references no file column and the file has no flat readable column");
            }
            columns.add(readableColumns.iterator().next()); // e.g. constant projection: rows still need counting
        }
        return columns;
    }

    /**
     * True for {@code SELECT COUNT(*) [, COUNT(*) ...] FROM t} with no WHERE,
     * GROUP BY, HAVING or ORDER BY: the row count is in the footer.
     */
    static boolean isCountStarOnly(SelectStatement st) {
        if (st.selectAll() || st.where() != null || !st.groupBy().isEmpty() || st.having() != null
                || !st.orderBy().isEmpty()) return false;
        if (st.limit() != null && st.limit() < 1) return false;
        for (SelectItem it : st.items()) {
            if (!it.isAggregate() || it.aggregate().func() != AggFunc.COUNT || !it.aggregate().star()) return false;
        }
        return true;
    }
}
