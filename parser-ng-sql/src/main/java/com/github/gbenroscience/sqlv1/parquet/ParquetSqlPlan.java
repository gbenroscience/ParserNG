package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.*;
import com.github.gbenroscience.sqlv1.ArrowQuery;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BaseFixedWidthVector;
import org.apache.arrow.vector.BaseVariableWidthVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
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
    /** True when the top-N keys are plain numeric columns and nothing needs the engine: see {@link #topNDirect}. */
    private boolean topDirect;
    /** The numeric column behind each sort key when {@link #topDirect}. */
    private final List<String> topKeyCols = new ArrayList<>();
    /** Non-null when aggregates are computed per batch and merged: see {@link PartialAgg}. */
    private PartialAgg partial;
    private boolean[] sortDesc = new boolean[0];
    private final boolean grouped;
    private final boolean engineless;
    private final ArrowQuery engine;
    private final List<NativeLeaves.Mask> masks;
    private final BoolExpr engineWhere;
    private long maxMaterializedRows;
    private ScanMetrics lastMetrics;
    private long roundedValues;
    /** Non-null when this query is folded in Java (DISTINCT, non-numeric aggregates): see {@link FoldQuery}. */
    private Folder fold;
    /** The statement with COALESCE/NULLIF calls replaced by derived-column names; {@code stmt} when there are none. */
    private SelectStatement rstmt;
    /** Derived string columns (LOWER, CONCAT, ...) added to every scan batch; null when the query has none. */
    private DerivedColumns.Holder derived;
    /** The derived columns, in fill order (for explain and the schema). */
    private List<DerivedColumns.Column> derivedCols = List.of();
    /** Non-null when the whole answer comes from the Parquet footer: see {@link FooterAggregates}. */
    private FooterAggregates.Answer footer;
    private ParquetSqlPlan foldInner;

    /** Compiles the engine statement on the backend chosen in {@link ParquetSqlOptions#backend()}. */
    private ArrowQuery engineQuery(String sql, SelectStatement engineStmt) {
        return ArrowQuery.compile(sql, engineStmt).withBackend(options.backend());
    }

    ParquetSqlPlan(String sql, SelectStatement stmt, List<Path> files, ParquetSqlOptions options, boolean explainOnly) {
        this.sql = sql;
        this.stmt = stmt;
        this.files = List.copyOf(files);
        this.options = options;
        this.explainOnly = explainOnly;
        this.maxMaterializedRows = options.maxMaterializedRows();

        long rows = 0;
        ParquetFileInfo first = null;
        List<ParquetFileInfo> infos = new ArrayList<>();
        for (Path f : files) {
            ParquetFileInfo info = ParquetFileInfo.read(f);
            infos.add(info);
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
        TableSchema fileSchema = TableSchema.of(s);
        ScanPlanner.ScanPlan sp = ScanPlanner.plan(stmt, first.topLevelColumns(), first.primitiveColumns(), fileSchema);

        Predicate ex = options.exactFilter() && stmt.where() != null ? PredicateConverter.convertExact(stmt.where(), fileSchema) : null;
        this.exact = ex != null;
        this.plan = exact ? new ScanPlanner.ScanPlan(sp.columns(), ex, sp.streamable(), sp.materializeReason(), sp.limit(), false) : sp;

        FooterAggregates.Answer fa = FooterAggregates.tryAnswer(stmt, fileSchema, infos);
        if (fa != null) {
            this.footer = fa;
            this.schema = fileSchema;
            this.rstmt = stmt;
            this.masks = List.of();
            this.engineWhere = null;
            this.grouped = false;
            this.engineless = false;
            this.engine = null;
            return;
        }

        checkQuotedReferences(stmt, fileSchema);

        // Folded queries (DISTINCT, GROUP BY over non-numeric keys, COUNT(DISTINCT), string aggregates) are decided
        // before any rewriting: their inner scan is a plain projection that does its own rewriting.
        if (FoldQuery.needed(stmt, fileSchema)) {
            this.fold = stmt.groupBy().isEmpty() ? new FoldQuery(stmt, fileSchema, arrowSchema) : new GroupFold(stmt, fileSchema);
            this.foldInner = new ParquetSqlPlan(sql, fold.inner(), files, options, false);
            this.schema = fileSchema;
            this.rstmt = stmt;
            this.masks = List.of();
            this.engineWhere = null;
            this.grouped = false;
            this.engineless = false;
            this.engine = null;
            return;
        }

        // Rewriters run in dependency order; each sees the derived columns of the ones before it and continues
        // their mask numbering, so every name the engine sees is unique.
        List<DerivedColumns.Column> dcols = new ArrayList<>();
        List<NativeLeaves.Mask> allMasks = new ArrayList<>();
        int used = 0;
        NestedFns.Result nr = NestedFns.rewrite(stmt, fileSchema, s, used);
        SelectStatement cur = nr.stmt();
        dcols.addAll(nr.columns());
        allMasks.addAll(nr.masks());
        used += nr.used();
        TableSchema sch = fileSchema.withVirtual(dcols);
        CastFns.Result cr = CastFns.rewrite(cur, sch, used);
        cur = cr.stmt();
        allMasks.addAll(cr.masks());
        used += cr.used();
        TemporalFns.Result tr = TemporalFns.rewrite(cur, sch, used);
        cur = tr.stmt();
        dcols.addAll(tr.columns());
        allMasks.addAll(tr.masks());
        used += tr.used();
        sch = fileSchema.withVirtual(dcols);
        StringFns.Result sr = StringFns.rewrite(cur, sch, used);
        cur = sr.stmt();
        dcols.addAll(sr.columns());
        allMasks.addAll(sr.masks());
        used += sr.used();
        sch = fileSchema.withVirtual(dcols);
        ScalarFns.Result sf = ScalarFns.rewrite(cur, sch, used);
        cur = sf.stmt();
        allMasks.addAll(sf.masks());
        used += sf.masks().size();
        this.schema = sch;
        this.rstmt = cur;
        this.derivedCols = List.copyOf(dcols);
        this.derived = dcols.isEmpty() ? null : new DerivedColumns.Holder(dcols);
        NativeLeaves.Result nl = (rstmt.where() == null || exact) ? null : NativeLeaves.rewrite(rstmt.where(), schema, used);
        if (nl != null) allMasks.addAll(nl.masks());
        this.masks = List.copyOf(allMasks);
        this.engineWhere = nl == null ? null : nl.where();

        this.grouped = rstmt.isGrouped();
        ArrowQuery q = null;
        boolean bare = false;
        try {
            if (grouped) {
                PartialAgg pa = PartialAgg.tryCreate(rstmt);
                if (pa != null) {
                    // Aggregate each batch in the engine and merge the partials, instead of copying the whole scan into
                    // one root. Anything unexpected falls back to the materializing path below.
                    List<String> savedNumeric = new ArrayList<>(numericColumns);
                    try {
                        SelectStatement pe = pa.engineStatement(engineWhere);
                        Set<String> prefs = new LinkedHashSet<>(ColumnRefs.referencedBy(pe));
                        prefs.addAll(maskColumns());
                        addNumeric(prefs);
                        q = engineQuery(sql, pe);
                        partial = pa;
                    } catch (RuntimeException notPartial) {
                        if (q != null) q.close();
                        q = null;
                        numericColumns.clear();
                        numericColumns.addAll(savedNumeric);
                    }
                }
                if (partial == null) {
                    SelectStatement eng = new SelectStatement(rstmt.selectAll(), rstmt.items(), rstmt.table(), engineWhere,
                            rstmt.groupBy(), rstmt.having(), rstmt.orderBy(), effLimit());
                    Set<String> grefs = new LinkedHashSet<>(ColumnRefs.referencedBy(eng));
                    grefs.addAll(maskColumns());
                    addNumeric(grefs);
                    q = engineQuery(sql, eng);
                }
                bare = false;
            } else {
                List<SelectItem> computed = new ArrayList<>();
                Set<String> refs = new LinkedHashSet<>();
                if (rstmt.selectAll()) {
                    for (String c : schema.names()) outs.add(new Out(c, c));
                } else {
                    for (SelectItem it : rstmt.items()) {
                        String t = it.exprText().trim();
                        if (schema.has(t)) {
                            outs.add(new Out(it.outputName(), com.github.gbenroscience.sqlv1.QuotedIdentifiers.real(t)));
                        } else {
                            ColumnRefs.identifiers(it.exprText(), refs);
                            computed.add(SelectItem.plain(it.exprText(), it.outputName()));
                            outs.add(new Out(it.outputName(), null));
                        }
                    }
                }
                refs.addAll(ColumnRefs.referencedBy(engineWhere)); // native leaves are masks: their columns are not numeric refs
                refs.addAll(maskColumns()); // except numeric masks, which read the widened column
                for (OrderItem o : rstmt.orderBy()) ColumnRefs.identifiers(o.exprText(), refs);
                addNumeric(refs);

                boolean sameNames = outs.stream().allMatch(o -> o.column() != null && o.column().equals(o.name()));
                bare = computed.isEmpty() && engineWhere == null && rstmt.orderBy().isEmpty() && sameNames;
                if (!bare) {
                    List<SelectItem> items = new ArrayList<>(computed);
                    boolean topN = topNEligible();
                    if (topN) {
                        // ORDER BY text resolves against computed aliases exactly as the engine resolves it,
                        // unless a widened column of the same name wins the name.
                        java.util.Map<String, String> bind = new java.util.LinkedHashMap<>();
                        for (SelectItem c : computed) {
                            if (!numericColumns.contains(c.outputName())) bind.put(c.outputName(), c.exprText());
                        }
                        List<OrderItem> ob = rstmt.orderBy();
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
                        // With no WHERE, no computed output and plain numeric columns as keys, the engine is not needed at
                        // all: the keys are read from the widened columns and the best rows picked with a heap.
                        boolean direct = engineWhere == null && computed.isEmpty();
                        for (int k = 0; k < ob.size() && direct; k++) {
                            String t = ob.get(k).exprText().trim();
                            ColumnKind kd = schema.kind(t);
                            direct = kd != null && kd.numeric() && schema.has(t) && numericColumns.contains(t);
                            if (direct) topKeyCols.add(t);
                        }
                        topDirect = direct;
                        if (!direct) topKeyCols.clear();
                    }
                    if (!topDirect) {
                        items.add(SelectItem.plain(NumericViews.ROW_ID, NumericViews.ROW_ID));
                        // top-N: the engine evaluates the keys and filters, and does NOT sort or limit; the heap picks the rows
                        SelectStatement es = topN
                                ? new SelectStatement(false, items, rstmt.table(), engineWhere, List.of(), null, List.of(), null)
                                : new SelectStatement(false, items, rstmt.table(), engineWhere, List.of(), null, rstmt.orderBy(), effLimit());
                        q = engineQuery(sql, es);
                    }
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
     * Allocates {@code combined} for exactly the rows about to be appended (the count and the string bytes are known
     * from the parts), so appending copies each value once instead of growing the vectors by doubling and copying them
     * again each time.
     */
    private static void presize(VectorSchemaRoot combined, List<VectorSchemaRoot> parts, long rows) {
        if (rows > 0 && rows < Integer.MAX_VALUE / 2) {
            for (FieldVector v : combined.getFieldVectors()) {
                if (v instanceof BaseFixedWidthVector f) {
                    f.setInitialCapacity((int) rows);
                } else if (v instanceof BaseVariableWidthVector b) {
                    long bytes = 0;
                    for (VectorSchemaRoot p : parts) {
                        FieldVector pv = p.getVector(v.getName());
                        if (pv instanceof BaseVariableWidthVector pb) bytes += pb.sizeOfValueBuffer();
                    }
                    b.setInitialCapacity((int) rows, Math.max(1.0, (double) bytes / rows));
                }
            }
        }
        combined.allocateNew();
    }

    /**
     * True if {@link #stream} delivers this query batch by batch (a projection or filter), which never holds the
     * result in memory; false if it needs {@link #execute}.
     */
    public boolean isStreamable() {
        return fold == null && footer == null && plan.streamable();
    }

    /** {@code LIMIT + OFFSET}: rows to produce before the offset is dropped, or {@code null} for no limit. */
    private Integer effLimit() {
        Integer l = plan.limit();
        return l == null ? null : (int) Math.min(Integer.MAX_VALUE, (long) l + stmt.offset());
    }

    /**
     * Bounded top-N applies to a non-grouped {@code ORDER BY ... LIMIT n} whose n fits the
     * materialization cap: see {@link TopN}.
     */
    private boolean topNEligible() {
        Integer lim = effLimit();
        return lim != null && lim >= 1 && lim <= maxMaterializedRows && lim <= 1_000_000
                && !grouped && !rstmt.orderBy().isEmpty();
    }

    /** Numeric columns read (widened) by two-valued numeric masks. */
    private Set<String> maskColumns() {
        Set<String> s = new LinkedHashSet<>();
        for (NativeLeaves.Mask m : masks) s.addAll(m.widenedColumns());
        return s;
    }

    /**
     * A quoted identifier must name a column of the file. A quoted alias defined with {@code AS "x"} names an
     * output only, so referencing it in {@code WHERE}, {@code ORDER BY} and so on fails here, with its real name.
     */
    private static void checkQuotedReferences(SelectStatement st, TableSchema fileSchema) {
        for (String r : ColumnRefs.referencedBy(st)) {
            String real = com.github.gbenroscience.sqlv1.QuotedIdentifiers.decode(r);
            if (real != null && !fileSchema.hasFileColumn(r) && !NestedFns.isPath(real, fileSchema)) {
                throw new ParquetSqlException("Unknown column \"" + real + "\". A quoted alias defined with AS can only name an output;"
                        + " it cannot be referenced in WHERE, ORDER BY or other clauses, so repeat its expression instead");
            }
        }
    }

    /**
     * Registers the numeric columns the engine must see. {@code refs} holds every identifier used
     * in an expression, WHERE, GROUP BY or ORDER BY; a schema column among them must be numeric.
     * Columns that are only SELECTed as-is never appear here.
     */
    private void addNumeric(Set<String> refs) {
        for (String r : refs) {
            if (!schema.has(r)) continue; // a function name or an alias, not a column
            ColumnKind k = schema.kind(r);
            if (!k.numeric()) {
                throw new ParquetSqlException("Column '" + com.github.gbenroscience.sqlv1.QuotedIdentifiers.real(r) + "' is " + k
                        + " and cannot be used in an expression, WHERE, GROUP BY or ORDER BY." + NOT_NUMERIC_HINT);
            }
            if (!numericColumns.contains(r)) numericColumns.add(r);
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
        if (footer != null) {
            b.append("mode: footer statistics (COUNT/MIN/MAX answered from the Parquet footer, no data page is read)\n");
            return b.toString();
        }
        if (fold != null) {
            b.append("mode: fold in Java (").append(stmt.distinct() ? "SELECT DISTINCT"
                    : !stmt.groupBy().isEmpty() ? "GROUP BY the numeric engine cannot do" : "aggregates the numeric engine cannot do")
                    .append("); not streamable, the inner scan below is\n");
            b.append("inner scan:\n");
            for (String line : foldInner.explain().split("\n")) b.append("  ").append(line).append('\n');
            return b.toString();
        }
        b.append("mode: ").append(plan.streamable() ? "stream"
                : !sortKeys.isEmpty() ? "bounded top-" + effLimit() + " (heap selection per batch, keeps at most " + 2L * effLimit() + " candidate rows"
                        + (topDirect ? "; keys read directly from the widened columns, no engine" : "; the engine evaluates keys and filter, it does not sort") + ")"
                : partial != null ? "streaming partial aggregation (the engine aggregates each batch, the partials are merged)"
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
        if (engine != null) b.append("backend: ").append(options.backend()).append('\n');
        b.append("engine: ").append(engine == null ? "none (pure projection, zero-copy)" : grouped ? "aggregate over widened columns" : "filter/project/sort + row-index gather").append('\n');
        b.append("limit: ").append(plan.limit()).append(stmt.offset() > 0 ? ", offset: " + stmt.offset() : "").append('\n');
        for (NativeLeaves.Mask m : masks) {
            if (m instanceof ScalarFns.ScalarMask || m instanceof StringFns.LengthMask) {
                b.append("derived column ").append(m.name).append(": ").append(m.description).append('\n');
            }
        }
        for (DerivedColumns.Column c : derivedCols) {
            b.append("derived column ").append(c.name).append(" (").append(c.kind).append("): ").append(c.text)
                    .append(" [evaluated per batch; never pushed to row-group pruning]\n");
        }
        return b.toString();
    }

    /**
     * Streams result batches, one per decoded batch, in bounded memory. The
     * root passed to {@code sink} is valid only during the call; do not close it.
     * Only for {@link ScanPlanner.ScanPlan#streamable() streamable} queries.
     */
    public void stream(BufferAllocator allocator, Consumer<VectorSchemaRoot> sink) {
        try {
            streamRaw(allocator, sink);
        } finally {
            if (derived != null) derived.release();
        }
    }

    private void streamRaw(BufferAllocator allocator, Consumer<VectorSchemaRoot> sink) {
        if (footer != null) {
            if ((plan.limit() != null && plan.limit() < 1) || stmt.offset() > 0) return;
            VectorSchemaRoot r = footer.toRoot(allocator);
            try {
                sink.accept(r);
            } finally {
                r.close();
            }
            return;
        }
        if (fold != null) {
            throw new UnsupportedOperationException("SELECT DISTINCT and non-numeric aggregates are folded over the whole scan. Use execute().");
        }
        if (!plan.streamable()) {
            throw new UnsupportedOperationException(
                    "Query cannot be streamed batch-by-batch: " + plan.materializeReason() + ". Use execute().");
        }
        lastMetrics = null;
        if (plan.metadataOnly()) {
            if ((plan.limit() != null && plan.limit() < 1) || stmt.offset() > 0) return;
            VectorSchemaRoot r = footerAnswer(allocator);
            try { sink.accept(r); } finally { r.close(); }
            return;
        }
        long remaining = plan.limit() == null ? Long.MAX_VALUE : plan.limit();
        long skip = stmt.offset();
        if (remaining == 0) return;
        for (Path f : files) {
            if (remaining <= 0) break;
            try (ParquetBatchReader r = scanFor(f).open(allocator);
                 NumericViews views = views(allocator, f)) {
                while (remaining > 0 && r.next()) {
                    Batch b = process(r.root(), views, allocator);
                    try {
                        int rows = b.root().getRowCount();
                        int from = 0;
                        if (skip > 0) {
                            from = (int) Math.min(skip, rows);
                            skip -= from;
                        }
                        int avail = (int) Math.min(rows - from, remaining);
                        remaining -= avail;
                        if (avail > 0) {
                            if (from == 0) {
                                if (avail < rows) b.root().setRowCount(avail);
                                sink.accept(b.root());
                            } else {
                                try (VectorSchemaRoot window = b.root().slice(from, avail)) {
                                    sink.accept(window);
                                }
                            }
                        }
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
        try {
            return applyWindow(executeRaw(allocator));
        } finally {
            if (derived != null) derived.release();
        }
    }

    /**
     * Drops the first {@code OFFSET} rows and anything past {@code LIMIT}. Every path produces at
     * most {@code LIMIT + OFFSET} rows, so this is a zero-copy slice (buffers are shared, and the
     * original root is released).
     */
    private VectorSchemaRoot applyWindow(VectorSchemaRoot r) {
        int off = stmt.offset();
        Integer lim = stmt.limit();
        int rows = r.getRowCount();
        if (off == 0 && (lim == null || rows <= lim)) return r;
        int from = Math.min(off, rows);
        int len = (int) Math.min(rows - from, lim == null ? Long.MAX_VALUE : lim);
        try {
            return r.slice(from, len);
        } finally {
            r.close();
        }
    }

    private VectorSchemaRoot executeRaw(BufferAllocator allocator) {
        lastMetrics = null;
        if (footer != null) return footer.toRoot(allocator);
        if (fold != null) {
            try {
                return fold.execute(allocator, foldInner, maxMaterializedRows);
            } finally {
                lastMetrics = foldInner.lastMetrics();
                roundedValues += foldInner.roundedValues();
            }
        }
        if (plan.metadataOnly()) return footerAnswer(allocator);
        if (!sortKeys.isEmpty()) return executeTopN(allocator);
        if (partial != null) return executePartial(allocator);

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
                presize(combined, parts, rows);
                if (!parts.isEmpty()) VectorSchemaRootAppender.append(combined, parts.toArray(new VectorSchemaRoot[0]));
                combined.setRowCount((int) rows);
                try (NumericViews views = views(allocator, files.get(0))) {
                    if (grouped) {
                        // through process(), so derived string columns (LOWER, CONCAT, ...) are added to the batch first
                        Batch gb = process(combined, views, allocator);
                        if (views != null) roundedValues += views.roundedValues();
                        return gb.root();
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

    /**
     * Bounded {@code ORDER BY ... LIMIT n}. Per batch the best rows are picked with a bounded heap
     * ({@link TopSelect}), not by sorting the batch, and copied straight from their source vectors into the running
     * {@link TopN}. With plain numeric key columns and no {@code WHERE} the engine is not involved at all; otherwise
     * it evaluates the keys and the filter for the batch (without sorting) and the heap picks among the survivors.
     */
    private VectorSchemaRoot executeTopN(BufferAllocator allocator) {
        int limit = effLimit();
        TopSelect sel = new TopSelect(limit, sortDesc);
        try (TopN top = new TopN(allocator, limit, outs.size(), sortDesc)) {
            for (Path f : files) {
                try (ParquetBatchReader r = scanFor(f).open(allocator);
                     NumericViews views = views(allocator, f)) {
                    while (r.next()) {
                        VectorSchemaRoot src = derived == null ? r.root() : derived.extend(r.root(), allocator);
                        if (topDirect) topDirectBatch(src, views, sel, top);
                        else topEngineBatch(src, views, sel, top);
                    }
                    lastMetrics = r.metrics();
                    roundedValues += views == null ? 0 : views.roundedValues();
                }
            }
            VectorSchemaRoot res = top.finish();
            return res != null ? res : emptyResult(allocator);
        }
    }

    private static TopSelect.Key keyOf(Float8Vector v) {
        return new TopSelect.Key() {
            @Override public boolean isNull(int row) { return v.isNull(row); }

            @Override public double get(int row) { return v.get(row); }
        };
    }

    private void topDirectBatch(VectorSchemaRoot src, NumericViews views, TopSelect sel, TopN top) {
        int n = src.getRowCount();
        if (n == 0) return;
        VectorSchemaRoot w = views.bind(src);
        TopSelect.Key[] keys = new TopSelect.Key[topKeyCols.size()];
        Float8Vector[] kvec = new Float8Vector[keys.length];
        for (int k = 0; k < keys.length; k++) {
            kvec[k] = (Float8Vector) w.getVector(topKeyCols.get(k));
            keys[k] = keyOf(kvec[k]);
        }
        int[] picks = sel.select(n, keys, top.threshold());
        if (picks.length == 0) return;
        FieldVector[] vecs = new FieldVector[outs.size()];
        int[][] rows = new int[outs.size()][];
        for (int c = 0; c < vecs.length; c++) {
            vecs[c] = src.getVector(outs.get(c).column());
            rows[c] = picks;
        }
        offer(top, vecs, rows, picks, kvec, picks);
    }

    private void topEngineBatch(VectorSchemaRoot src, NumericViews views, TopSelect sel, TopN top) {
        int n = src.getRowCount();
        VectorSchemaRoot res = engine.execute(views == null ? emptyRoot(n) : views.bind(src));
        try {
            int count = res.getRowCount();
            if (count == 0) return;
            Float8Vector rid = (Float8Vector) res.getVector(NumericViews.ROW_ID);
            TopSelect.Key[] keys = new TopSelect.Key[sortKeys.size()];
            Float8Vector[] kvec = new Float8Vector[keys.length];
            for (int k = 0; k < keys.length; k++) {
                kvec[k] = (Float8Vector) res.getVector(sortKeys.get(k));
                keys[k] = keyOf(kvec[k]);
            }
            int[] picks = sel.select(count, keys, top.threshold());
            if (picks.length == 0) return;
            int[] srcRows = new int[picks.length];
            for (int i = 0; i < picks.length; i++) srcRows[i] = (int) rid.get(picks[i]);
            FieldVector[] vecs = new FieldVector[outs.size()];
            int[][] rows = new int[outs.size()][];
            for (int c = 0; c < vecs.length; c++) {
                Out o = outs.get(c);
                if (o.column() == null) { // computed by the engine: indexed by the engine's row
                    vecs[c] = res.getVector(o.name());
                    rows[c] = picks;
                } else {                   // a column of the scan: indexed by the row the engine reports
                    vecs[c] = src.getVector(o.column());
                    rows[c] = srcRows;
                }
            }
            offer(top, vecs, rows, picks, kvec, picks);
        } finally {
            res.close();
        }
    }

    /** Hands the picked rows to the running top-N. {@code keyRows[i]} is the row of {@code kvec} behind pick {@code i}. */
    private void offer(TopN top, FieldVector[] vecs, int[][] rows, int[] picks, Float8Vector[] kvec, int[] keyRows) {
        int nk = kvec.length;
        double[][] kv = new double[nk][picks.length];
        boolean[][] kn = new boolean[nk][picks.length];
        for (int k = 0; k < nk; k++) {
            for (int i = 0; i < picks.length; i++) {
                boolean isNull = kvec[k].isNull(keyRows[i]);
                kn[k][i] = isNull;
                kv[k][i] = isNull ? 0.0 : kvec[k].get(keyRows[i]);
            }
        }
        Field[] fields = null;
        if (!top.initialized()) {
            fields = new Field[vecs.length];
            for (int c = 0; c < vecs.length; c++) {
                Field f = vecs[c].getField();
                fields[c] = new Field(outs.get(c).name(), f.getFieldType(), f.getChildren());
            }
        }
        top.addPicked(fields, vecs, rows, picks.length, kv, kn);
    }

    /**
     * Aggregation with the engine run per batch: each batch yields partial aggregates, merged by
     * {@link PartialAgg}. Memory is one batch plus one row per group; nothing is copied into a combined root.
     */
    private VectorSchemaRoot executePartial(BufferAllocator allocator) {
        PartialAgg.Merger m = partial.newMerger(maxMaterializedRows);
        try {
            for (Path f : files) {
                try (ParquetBatchReader r = scanFor(f).open(allocator);
                     NumericViews views = views(allocator, f)) {
                    while (r.next()) {
                        Batch b = process(r.root(), views, allocator);
                        try {
                            m.accept(b.root(), allocator);
                        } finally {
                            if (b.owned()) b.root().close();
                        }
                    }
                    lastMetrics = r.metrics();
                    roundedValues += views == null ? 0 : views.roundedValues();
                }
            }
            Integer lim = effLimit();
            return m.finish(allocator, lim == null ? Integer.MAX_VALUE : lim);
        } catch (RuntimeException e) {
            m.close();
            throw e;
        }
    }

    /** A zero-row result with the output columns, for a top-N query that matched nothing. */
    VectorSchemaRoot emptyResult(BufferAllocator alloc) {
        List<FieldVector> vecs = new ArrayList<>(outs.size());
        try {
            for (Out o : outs) {
                FieldVector v;
                if (o.column() == null) {
                    v = new Float8Vector(o.name(), alloc);
                } else {
                    if (schema.isVirtual(o.column())) {
                        v = new Field(o.name(), org.apache.arrow.vector.types.pojo.FieldType.nullable(schema.type(o.column())), null).createVector(alloc);
                    } else {
                        Field sf = arrowSchema.findField(o.column());
                        v = new Field(o.name(), sf.getFieldType(), sf.getChildren()).createVector(alloc);
                    }
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

    /** The Arrow field of each output column, in order (a computed column is a nullable double). */
    List<Field> outputFields() {
        List<Field> r = new ArrayList<>(outs.size());
        for (Out o : outs) {
            if (o.column() == null) {
                r.add(new Field(o.name(), org.apache.arrow.vector.types.pojo.FieldType.nullable(
                        new org.apache.arrow.vector.types.pojo.ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE)), null));
            } else if (schema.isVirtual(o.column())) {
                r.add(new Field(o.name(), org.apache.arrow.vector.types.pojo.FieldType.nullable(schema.type(o.column())), null));
            } else {
                Field sf = arrowSchema.findField(o.column());
                r.add(new Field(o.name(), sf.getFieldType(), sf.getChildren()));
            }
        }
        return r;
    }

    /** Output column names, in order. */
    List<String> outputNames() {
        List<String> n = new ArrayList<>(outs.size());
        for (Out o : outs) n.add(o.name());
        return n;
    }

    @Override
    public void close() {
        if (foldInner != null) foldInner.close();
        if (derived != null) derived.close();
        if (engine != null) engine.close();
    }

    // ------------------------------------------------------------------ internals

    private NumericViews views(BufferAllocator alloc, Path file) {
        if (arrowSchema == null) return null;
        if (numericColumns.isEmpty() && masks.isEmpty() && (grouped || engineless)) return null;
        return new NumericViews(alloc, numericColumns, schema, options.int64Policy(), !grouped, file.toString(), masks);
    }

    private Batch process(VectorSchemaRoot scanned, NumericViews views, BufferAllocator alloc) {
        VectorSchemaRoot src = derived == null ? scanned : derived.extend(scanned, alloc);
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
        if (effLimit() != null) n = Math.min(n, effLimit());
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
