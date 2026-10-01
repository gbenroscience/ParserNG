package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.*;
import com.github.gbenroscience.sqlv1.ArrowQuery;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.TransferPair;
import org.apache.arrow.vector.util.VectorSchemaRootAppender;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

/**
 * An executable plan for one SQL statement over one or more Parquet files.
 *
 * <h2>Execution model</h2>
 * <ul>
 * <li><b>Numeric work is delegated to {@link ArrowQuery}</b> (vectorized double
 * kernels). Only the numeric columns an expression, WHERE or ORDER BY touches
 * are widened to {@code Float8} ({@link NumericViews}); nothing else is copied
 * into the engine.</li>
 * <li><b>Every column can be SELECTed as-is</b> in its native Arrow type
 * (varchar, boolean, date, timestamp, decimal, int64 exact, nested...). Such
 * passthrough columns never enter the engine: the engine also emits a row-index
 * column, and the surviving rows are gathered from the native batch by index.
 * A query that is only a projection of plain columns skips the engine entirely
 * and hands the reader's batch through with no copy.</li>
 * <li>Expressions, WHERE, ORDER BY and GROUP BY may reference numeric columns
 * only. Using a string, boolean, date, timestamp, binary or nested column there
 * fails with a {@link ParquetSqlException} naming the column and type.</li>
 * <li><b>Pruning</b> is a superset filter derived from WHERE ({@link PredicateConverter});
 * the full WHERE is always re-evaluated, except when
 * {@link ParquetSqlOptions#exactFilter()} is on <em>and</em> the whole WHERE converts
 * exactly, in which case the scan filters and the engine sees no WHERE.</li>
 * <li><b>{@code SELECT COUNT(*) FROM t}</b> with no WHERE reads only the footer.</li>
 * </ul>
 *
 * <p>Aggregate results are {@code DOUBLE} columns (the engine's convention).
 * Not thread-safe; create one plan per thread.
 */
public final class ParquetSqlPlan implements AutoCloseable {

    /** Default cap on rows held by a materializing query. */
    public static final long DEFAULT_MAX_MATERIALIZED_ROWS = ParquetSqlOptions.DEFAULT_MAX_MATERIALIZED_ROWS;

    private static final String NOT_NUMERIC_HINT = " Only numeric columns can be used in expressions, WHERE, GROUP BY and ORDER BY "
            + "(string, boolean, date, timestamp predicates are not supported yet); any column can still be SELECTed as-is.";

    /** One output column: either a passthrough of {@code column}, or (column == null) an engine-computed value. */
    private record Out(String name, String column) { }

    /** A produced batch and whether the caller must close it. */
    private record Batch(VectorSchemaRoot root, boolean owned) { }

    private final String sql;
    private final SelectStatement stmt;
    private final List<Path> files;
    private final ParquetSqlOptions options;
    private final boolean explainOnly;

    private final ScanPlanner.ScanPlan plan;
    private final TableSchema schema;
    private final Schema arrowSchema;
    private final long totalRows;
    private final boolean exact;
    private final List<Out> outs = new ArrayList<>();
    private final List<String> numericColumns = new ArrayList<>();
    /** Hidden Float8 sort-key items added to the engine query for bounded top-N, else empty. */
    private final List<String> sortKeys = new ArrayList<>();
    private boolean[] sortDesc = new boolean[0];
    private final boolean grouped;
    private final boolean engineless;
    private final ArrowQuery engine;
    private final List<NativeLeaves.Mask> masks;
    private final BoolExpr engineWhere;
    private long maxMaterializedRows;
    private ScanMetrics lastMetrics;
    private long roundedValues;

    ParquetSqlPlan(String sql, SelectStatement stmt, List<Path> files, ParquetSqlOptions options, boolean explainOnly) {
        this.sql = sql;
        this.stmt = stmt;
        this.files = List.copyOf(files);
        this.options = options;
        this.explainOnly = explainOnly;
        this.maxMaterializedRows = options.maxMaterializedRows();

        long rows = 0;
        ParquetFileInfo first = null;
        for (Path f : files) {
            ParquetFileInfo info = ParquetFileInfo.read(f);
            if (first == null) first = info;
            else if (!first.topLevelColumns().equals(info.topLevelColumns())) {
                throw new ParquetSqlException("Schema mismatch: " + f + " has columns " + info.topLevelColumns()
                        + " but " + files.get(0) + " has " + first.topLevelColumns());
            }
            rows += info.rowCount();
        }
        this.totalRows = rows;

        if (ScanPlanner.isCountStarOnly(stmt)) {
            this.schema = TableSchema.ofKinds(java.util.Map.of());
            this.arrowSchema = null;
            this.plan = ScanPlanner.plan(stmt, first.topLevelColumns(), first.primitiveColumns(), schema);
            this.grouped = false;
            this.engineless = false;
            this.engine = null;
            this.exact = false;
            this.masks = List.of();
            this.engineWhere = null;
            return;
        }

        List<String> cols = ScanPlanner.columnsFor(stmt, first.topLevelColumns(), first.primitiveColumns());
        Schema s = discoverSchema(files.get(0), cols);
        for (int i = 1; i < files.size(); i++) {
            if (!s.equals(discoverSchema(files.get(i), cols))) {
                throw new ParquetSqlException("Schema mismatch between " + files.get(0) + " and " + files.get(i)
                        + " for the columns this query reads; all files must have identical column names and types");
            }
        }
        this.arrowSchema = s;
        this.schema = TableSchema.of(s);
        ScanPlanner.ScanPlan sp = ScanPlanner.plan(stmt, first.topLevelColumns(), first.primitiveColumns(), schema);

        Predicate ex = options.exactFilter() && stmt.where() != null ? PredicateConverter.convertExact(stmt.where(), schema) : null;
        this.exact = ex != null;
        this.plan = exact ? new ScanPlanner.ScanPlan(sp.columns(), ex, sp.streamable(), sp.materializeReason(), sp.limit(), false) : sp;

        NativeLeaves.Result nl = (stmt.where() == null || exact) ? null : NativeLeaves.rewrite(stmt.where(), schema);
        this.masks = nl == null ? List.of() : nl.masks();
        this.engineWhere = nl == null ? null : nl.where();

        this.grouped = stmt.isGrouped();
        ArrowQuery q = null;
        boolean bare = false;
        try {
            if (grouped) {
                SelectStatement eng = new SelectStatement(stmt.selectAll(), stmt.items(), stmt.table(), engineWhere,
                        stmt.groupBy(), stmt.having(), stmt.orderBy(), stmt.limit());
                Set<String> grefs = new LinkedHashSet<>(ColumnRefs.referencedBy(eng));
                grefs.addAll(maskColumns());
                addNumeric(grefs);
                q = ArrowQuery.compile(sql, eng);
                bare = false;
            } else {
                List<SelectItem> computed = new ArrayList<>();
                Set<String> refs = new LinkedHashSet<>();
                if (stmt.selectAll()) {
                    for (String c : schema.names()) outs.add(new Out(c, c));
                } else {
                    for (SelectItem it : stmt.items()) {
                        String t = it.exprText().trim();
                        if (schema.has(t)) {
                            outs.add(new Out(it.outputName(), t));
                        } else {
                            ColumnRefs.identifiers(it.exprText(), refs);
                            computed.add(SelectItem.plain(it.exprText(), it.outputName()));
                            outs.add(new Out(it.outputName(), null));
                        }
                    }
                }
                refs.addAll(ColumnRefs.referencedBy(engineWhere)); // native leaves are masks: their columns are not numeric refs
                refs.addAll(maskColumns()); // except numeric masks, which read the widened column
                for (OrderItem o : stmt.orderBy()) ColumnRefs.identifiers(o.exprText(), refs);
                addNumeric(refs);

                boolean sameNames = outs.stream().allMatch(o -> o.column() != null && o.column().equals(o.name()));
                bare = computed.isEmpty() && engineWhere == null && stmt.orderBy().isEmpty() && sameNames;
                if (!bare) {
                    List<SelectItem> items = new ArrayList<>(computed);
                    if (topNEligible()) {
                        // ORDER BY text resolves against computed aliases exactly as the engine resolves it,
                        // unless a widened column of the same name wins the name.
                        java.util.Map<String, String> bind = new java.util.LinkedHashMap<>();
                        for (SelectItem c : computed) {
                            if (!numericColumns.contains(c.outputName())) bind.put(c.outputName(), c.exprText());
                        }
                        List<OrderItem> ob = stmt.orderBy();
                        sortDesc = new boolean[ob.size()];
                        for (int k = 0; k < ob.size(); k++) {
                            String name = "pngsortk" + NativeLeaves.maskName(k).substring("pngmask".length());
                            String text;
                            try {
                                text = com.github.gbenroscience.sqlv1.ast.WhereAliasResolver.substituteText(ob.get(k).exprText(), bind);
                            } catch (IllegalArgumentException cyclic) {
                                throw new ParquetSqlException("ORDER BY " + ob.get(k).exprText() + ": " + cyclic.getMessage());
                            }
                            items.add(SelectItem.plain(text, name));
                            sortKeys.add(name);
                            sortDesc[k] = ob.get(k).descending();
                        }
                    }
                    items.add(SelectItem.plain(NumericViews.ROW_ID, NumericViews.ROW_ID));
                    SelectStatement es = new SelectStatement(false, items, stmt.table(), engineWhere,
                            List.of(), null, stmt.orderBy(), stmt.limit());
                    q = ArrowQuery.compile(sql, es);
                }
            }
        } catch (RuntimeException e) {
            if (q != null) q.close();
            throw e;
        }
        this.engineless = bare;
        this.engine = q;
    }

    /**
     * Bounded top-N applies to a non-grouped {@code ORDER BY ... LIMIT n} whose n fits the
     * materialization cap: see {@link TopN}.
     */
    private boolean topNEligible() {
        Integer lim = plan.limit();
        return lim != null && lim >= 1 && lim <= maxMaterializedRows && lim <= 1_000_000
                && !grouped && !stmt.orderBy().isEmpty();
    }

    /** Numeric columns read (widened) by two-valued numeric masks. */
    private Set<String> maskColumns() {
        Set<String> s = new LinkedHashSet<>();
        for (NativeLeaves.Mask m : masks) if (m.widenedColumn() != null) s.add(m.widenedColumn());
        return s;
    }

    /**
     * Registers the numeric columns the engine must see. {@code refs} holds every identifier used
     * in an expression, WHERE, GROUP BY or ORDER BY; a schema column among them must be numeric.
     * Columns that are only SELECTed as-is never appear here.
     */
    private void addNumeric(Set<String> refs) {
        for (String c : schema.names()) {
            if (!refs.contains(c)) continue;
            ColumnKind k = schema.kind(c);
            if (!k.numeric()) {
                throw new ParquetSqlException("Column '" + c + "' is " + k
                        + " and cannot be used in an expression, WHERE, GROUP BY or ORDER BY." + NOT_NUMERIC_HINT);
            }
            numericColumns.add(c);
        }
    }

    private static Schema discoverSchema(Path file, List<String> cols) {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = scanOf(file, cols).open(a)) {
            return r.schema();
        }
    }

    private static ParquetScan scanOf(Path file, List<String> cols) {
        ParquetScan s = ParquetScan.scan(file);
        return cols == null || cols.isEmpty() ? s : s.select(cols.toArray(new String[0]));
    }

    private ParquetScan scanFor(Path file) {
        ParquetScan s = scanOf(file, plan.columns()).withMetrics(options.metrics());
        if (options.parallelism() != null) s = s.parallelism(options.parallelism());
        if (options.batchSize() != null) s = s.batchSize(options.batchSize());
        if (options.maxRowGroupBytes() != null) s = s.maxRowGroupBytes(options.maxRowGroupBytes());
        if (plan.pruning() != null) s = s.pushdown(plan.pruning());
        if (exact) s = s.exactFilter();
        return s;
    }

    // ------------------------------------------------------------------ public API

    /**
     * Overrides the cap on rows a materializing query may hold (initially
     * {@link ParquetSqlOptions#maxMaterializedRows()}). Exceeding it fails loudly.
     *
     * @return this plan
     */
    public ParquetSqlPlan maxMaterializedRows(long rows) {
        if (rows < 1) throw new IllegalArgumentException("maxMaterializedRows must be >= 1");
        this.maxMaterializedRows = rows;
        return this;
    }

    /**
     * The compiled numeric engine query (filter, computed columns, sort, aggregate), or {@code null}
     * when the plan needs none: a footer-only {@code COUNT(*)} or a pure projection of plain columns.
     * Its statement is the bridge's rewritten one (computed columns plus a row-index column), not the
     * user's SQL.
     */
    public ArrowQuery query() { return engine; }

    /** @return the scan-shaped decisions (columns, pruning predicate, mode) */
    public ScanPlanner.ScanPlan scanPlan() { return plan; }

    /** @return metrics of the most recent scan (last file for multi-file), or {@code null} if none ran or the answer came from the footer */
    public ScanMetrics lastMetrics() { return lastMetrics; }

    /** @return true if this plan was created from {@code EXPLAIN SELECT ...} */
    public boolean isExplain() { return explainOnly; }

    /** @return values rounded so far under {@link ParquetSqlOptions.Int64Policy#ROUND_TO_DOUBLE} */
    public long roundedValues() { return roundedValues; }

    /** Describes columns, pruning, what is not pushed and why, and the execution mode. */
    public String explain() {
        StringBuilder b = new StringBuilder();
        b.append("files: ").append(files.size()).append(files.size() == 1 ? " (" + files.get(0) + ")" : "").append('\n');
        if (plan.metadataOnly()) {
            b.append("mode: metadata-only (COUNT(*) answered from the footer, no data page decoded)\n");
            b.append("rows in footer: ").append(totalRows).append('\n');
            return b.toString();
        }
        b.append("mode: ").append(plan.streamable() ? "stream"
                : !sortKeys.isEmpty() ? "bounded top-" + plan.limit() + " (keeps at most " + 2L * plan.limit() + " candidate rows)"
                : "materialize (" + plan.materializeReason() + ")").append('\n');
        b.append("columns decoded: ").append(plan.columns() == null ? "*" : plan.columns()).append('\n');
        if (stmt.where() == null) {
            b.append("pruning: none (no WHERE)\n");
        } else if (exact) {
            b.append("pruning: ").append(plan.pruning()).append(" [exactFilter: the scan filters rows; WHERE is not re-evaluated]\n");
        } else if (plan.pruning() == null) {
            b.append("pruning: none. Nothing in the WHERE could be pushed as a sound row-group predicate")
                    .append(" (an expression, column-vs-column, LIKE/REGEXP, or a type not pushed: decimal, timestamp, boolean, unsigned, nested)\n");
        } else {
            b.append("pruning: ").append(plan.pruning()).append(" (superset only; the full WHERE is re-evaluated in Arrow)\n");
        }
        if (options.exactFilter() && !exact && stmt.where() != null) {
            b.append("exactFilter: requested but not applied: the whole WHERE does not convert with exact semantics\n");
        }
        if (!masks.isEmpty()) {
            b.append("native predicates (evaluated by the bridge on the Arrow vectors, never pushed to Parquet):\n");
            for (NativeLeaves.Mask m : masks) b.append("  ").append(m.description).append('\n');
        }
        b.append("numeric columns widened to double: ").append(numericColumns).append('\n');
        List<String> pass = new ArrayList<>();
        for (Out o : outs) if (o.column() != null) pass.add(o.column() + " (" + schema.kind(o.column()) + ")");
        b.append("passthrough columns (native type): ").append(pass).append('\n');
        b.append("engine: ").append(engine == null ? "none (pure projection, zero-copy)" : grouped ? "aggregate over widened columns" : "filter/project/sort + row-index gather").append('\n');
        b.append("limit: ").append(plan.limit()).append('\n');
        return b.toString();
    }

    /**
     * Streams result batches, one per decoded batch, in bounded memory. The
     * root passed to {@code sink} is valid only during the call; do not close it.
     * Only for {@link ScanPlanner.ScanPlan#streamable() streamable} queries.
     */
    public void stream(BufferAllocator allocator, Consumer<VectorSchemaRoot> sink) {
        if (!plan.streamable()) {
            throw new UnsupportedOperationException(
                    "Query cannot be streamed batch-by-batch: " + plan.materializeReason() + ". Use execute().");
        }
        lastMetrics = null;
        if (plan.metadataOnly()) {
            if (plan.limit() != null && plan.limit() < 1) return;
            VectorSchemaRoot r = footerAnswer(allocator);
            try { sink.accept(r); } finally { r.close(); }
            return;
        }
        long remaining = plan.limit() == null ? Long.MAX_VALUE : plan.limit();
        if (remaining == 0) return;
        for (Path f : files) {
            if (remaining <= 0) break;
            try (ParquetBatchReader r = scanFor(f).open(allocator);
                 NumericViews views = views(allocator, f)) {
                while (remaining > 0 && r.next()) {
                    Batch b = process(r.root(), views, allocator);
                    try {
                        int rows = b.root().getRowCount();
                        if (rows > remaining) {
                            b.root().setRowCount((int) remaining);
                            rows = (int) remaining;
                        }
                        remaining -= rows;
                        if (rows > 0) sink.accept(b.root());
                    } finally {
                        if (b.owned()) b.root().close();
                    }
                }
                lastMetrics = r.metrics();
                roundedValues += views == null ? 0 : views.roundedValues();
            }
        }
    }

    /**
     * Executes the whole query and returns the result. The caller closes it.
     * Rows surviving row-group pruning are held in memory (capped by
     * {@link ParquetSqlOptions#maxMaterializedRows()}).
     */
    public VectorSchemaRoot execute(BufferAllocator allocator) {
        lastMetrics = null;
        if (plan.metadataOnly()) return footerAnswer(allocator);
        if (!sortKeys.isEmpty()) return executeTopN(allocator);

        List<VectorSchemaRoot> parts = new ArrayList<>();
        try {
            long rows = 0;
            for (Path f : files) {
                try (ParquetBatchReader r = scanFor(f).open(allocator)) {
                    while (r.next()) {
                        rows += r.root().getRowCount();
                        if (rows > maxMaterializedRows) {
                            throw new ParquetSqlException("Query needs more than " + maxMaterializedRows
                                    + " rows materialized. Add a more selective WHERE, select fewer columns, "
                                    + "use stream() if the query allows it, or raise maxMaterializedRows.");
                        }
                        parts.add(r.detach()); // zero-copy ownership transfer
                    }
                    lastMetrics = r.metrics();
                }
            }
            VectorSchemaRoot combined = VectorSchemaRoot.create(arrowSchema, allocator);
            try {
                // VectorAppender reads the target's offset buffer for variable-width vectors, which is a
                // zero-length buffer until allocated: appending into a fresh root fails with an
                // IndexOutOfBoundsException on the first VARCHAR/BINARY column.
                combined.allocateNew();
                if (!parts.isEmpty()) VectorSchemaRootAppender.append(combined, parts.toArray(new VectorSchemaRoot[0]));
                combined.setRowCount((int) rows);
                try (NumericViews views = views(allocator, files.get(0))) {
                    if (grouped) {
                        VectorSchemaRoot res = engine.execute(views == null ? emptyRoot(combined.getRowCount()) : views.bind(combined));
                        if (views != null) roundedValues += views.roundedValues();
                        return res;
                    }
                    if (engineless) return copyAll(combined, allocator);
                    Batch b = process(combined, views, allocator);
                    roundedValues += views.roundedValues();
                    return b.root();
                }
            } finally {
                combined.close();
            }
        } finally {
            for (VectorSchemaRoot p : parts) p.close();
        }
    }

    /** Bounded {@code ORDER BY ... LIMIT n}: each batch's engine answer is merged into a {@link TopN}. */
    private VectorSchemaRoot executeTopN(BufferAllocator allocator) {
        try (TopN top = new TopN(allocator, plan.limit(), outs.size(), sortDesc)) {
            for (Path f : files) {
                try (ParquetBatchReader r = scanFor(f).open(allocator);
                     NumericViews views = views(allocator, f)) {
                    while (r.next()) {
                        Batch b = process(r.root(), views, allocator);
                        try {
                            top.add(b.root());
                        } finally {
                            if (b.owned()) b.root().close();
                        }
                    }
                    lastMetrics = r.metrics();
                    roundedValues += views == null ? 0 : views.roundedValues();
                }
            }
            VectorSchemaRoot res = top.finish();
            return res != null ? res : emptyResult(allocator);
        }
    }

    /** A zero-row result with the output columns, for a top-N query that matched nothing. */
    private VectorSchemaRoot emptyResult(BufferAllocator alloc) {
        List<FieldVector> vecs = new ArrayList<>(outs.size());
        try {
            for (Out o : outs) {
                FieldVector v;
                if (o.column() == null) {
                    v = new Float8Vector(o.name(), alloc);
                } else {
                    Field sf = arrowSchema.findField(o.column());
                    v = new Field(o.name(), sf.getFieldType(), sf.getChildren()).createVector(alloc);
                }
                v.allocateNew();
                v.setValueCount(0);
                vecs.add(v);
            }
        } catch (RuntimeException e) {
            for (FieldVector v : vecs) v.close();
            throw e;
        }
        return new VectorSchemaRoot(vecs);
    }

    @Override
    public void close() {
        if (engine != null) engine.close();
    }

    // ------------------------------------------------------------------ internals

    private NumericViews views(BufferAllocator alloc, Path file) {
        if (arrowSchema == null) return null;
        if (numericColumns.isEmpty() && masks.isEmpty() && (grouped || engineless)) return null;
        return new NumericViews(alloc, numericColumns, schema, options.int64Policy(), !grouped, file.toString(), masks);
    }

    private Batch process(VectorSchemaRoot src, NumericViews views, BufferAllocator alloc) {
        int n = src.getRowCount();
        if (grouped) {
            return new Batch(engine.execute(views == null ? emptyRoot(n) : views.bind(src)), true);
        }
        if (engineless) {
            List<FieldVector> vecs = new ArrayList<>(outs.size());
            List<Field> fields = new ArrayList<>(outs.size());
            for (Out o : outs) {
                FieldVector v = src.getVector(o.column());
                vecs.add(v);
                fields.add(v.getField());
            }
            return new Batch(new VectorSchemaRoot(new Schema(fields), vecs, n), false);
        }
        VectorSchemaRoot res = engine.execute(views.bind(src));
        int rows = res.getRowCount();
        FieldVector rid = res.getVector(NumericViews.ROW_ID);
        List<FieldVector> vecs = new ArrayList<>(outs.size());
        List<Field> fields = new ArrayList<>(outs.size());
        List<FieldVector> gathered = new ArrayList<>();
        try {
            for (Out o : outs) {
                FieldVector v;
                if (o.column() == null) {
                    v = res.getVector(o.name());
                } else {
                    v = gather(src.getVector(o.column()), (Float8Vector) rid, rows, o.name(), alloc);
                    gathered.add(v);
                }
                vecs.add(v);
                fields.add(v.getField());
            }
        } catch (RuntimeException e) {
            for (FieldVector g : gathered) g.close();
            res.close();
            throw e;
        }
        for (String k : sortKeys) {
            FieldVector kv = res.getVector(k);
            vecs.add(kv);
            fields.add(kv.getField());
        }
        rid.close();
        return new Batch(new VectorSchemaRoot(new Schema(fields), vecs, rows), true);
    }

    /** Copies {@code rows} values of {@code src} at the indices in {@code ids} into a new vector named {@code name}. */
    private static FieldVector gather(FieldVector src, Float8Vector ids, int rows, String name, BufferAllocator alloc) {
        Field sf = src.getField();
        FieldVector dst = new Field(name, sf.getFieldType(), sf.getChildren()).createVector(alloc);
        try {
            dst.allocateNew();
            for (int i = 0; i < rows; i++) dst.copyFromSafe((int) ids.get(i), i, src);
            dst.setValueCount(rows);
            return dst;
        } catch (RuntimeException e) {
            dst.close();
            throw e;
        }
    }

    /** Whole-table copy for a pure projection executed in materialize mode (zero-copy buffer sharing). */
    private VectorSchemaRoot copyAll(VectorSchemaRoot combined, BufferAllocator alloc) {
        int n = combined.getRowCount();
        if (plan.limit() != null) n = (int) Math.min(n, plan.limit());
        List<FieldVector> vecs = new ArrayList<>(outs.size());
        List<Field> fields = new ArrayList<>(outs.size());
        for (Out o : outs) {
            TransferPair tp = combined.getVector(o.column()).getTransferPair(o.name(), alloc);
            tp.splitAndTransfer(0, n);
            FieldVector v = (FieldVector) tp.getTo();
            vecs.add(v);
            fields.add(v.getField());
        }
        return new VectorSchemaRoot(new Schema(fields), vecs, n);
    }

    private VectorSchemaRoot emptyRoot(int rows) {
        return new VectorSchemaRoot(new Schema(List.of()), new ArrayList<>(), rows);
    }

    /** {@code COUNT(*)} answered from the footer row count: one row, Float8 columns (the engine's aggregate type). */
    private VectorSchemaRoot footerAnswer(BufferAllocator a) {
        List<FieldVector> vecs = new ArrayList<>();
        for (SelectItem it : stmt.items()) {
            Float8Vector v = new Float8Vector(it.outputName(), a);
            v.allocateNew(1);
            v.set(0, totalRows);
            v.setValueCount(1);
            vecs.add(v);
        }
        return new VectorSchemaRoot(vecs);
    }
}
