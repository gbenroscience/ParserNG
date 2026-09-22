package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.arrow.tools.box.ArrowBindingException;
import com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend;
import com.github.gbenroscience.arrow.tools.box.ArrowExpressionEvaluator;
import com.github.gbenroscience.arrow.tools.box.ArrowExpressionEvaluators;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import com.github.gbenroscience.sqlv1.ast.AggFunc;
import com.github.gbenroscience.sqlv1.ast.AggregateSpec;
import com.github.gbenroscience.sqlv1.ast.AndExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExprs;
import com.github.gbenroscience.sqlv1.ast.IsNullExpr;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
import com.github.gbenroscience.sqlv1.ast.OrExpr;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import com.github.gbenroscience.sqlv1.ast.WhereAliasResolver;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.TransferPair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A compiled parser-ng-sql query: parse once, compile expressions once
 * (lazily, on first {@link #execute(VectorSchemaRoot)}), execute many times.
 *
 * <pre>{@code
 * try (ArrowQuery query = ArrowQuery.compile("SELECT sqrt(x*x + y*y) AS distance FROM data WHERE x > 10")) {
 *     VectorSchemaRoot result = query.execute(root); // caller closes result
 * }
 * }</pre>
 *
 * <h2>Compilation model</h2>
 * {@link #compile(String)} only parses. ParserNG evaluators are compiled on
 * the first {@code execute} (float64 vs float32 kernels depend on the root's
 * column types) and cached per schema fingerprint. A different schema
 * triggers a one-time recompilation, safely w.r.t. in-flight calls (see
 * "Thread-safety").
 *
 * <h2>Execution pipeline (non-grouped queries)</h2>
 * {@code WHERE} yields a selection ({@code null} = every row, in order).
 * {@code ORDER BY} permutes/narrows that selection; {@code LIMIT} shortens it.
 * Projection then reads <b>directly from the input root</b> and produces only
 * the output columns: no intermediate "filtered" copy of the input is ever
 * built, so unreferenced input columns cost nothing and referenced ones are
 * read once. A computed column with no narrowing is evaluated straight into
 * its result vector. A computed column with a narrowing selection is
 * evaluated over the full batch into one per-call scratch vector (shared by
 * all projections, closed before returning) and only the selected rows are
 * gathered. Un-narrowed passthrough columns are a {@link TransferPair}
 * buffer transfer (no per-row work); narrowed ones are a typed gather.
 * Grouped queries evaluate keys/aggregates against the input root directly
 * when there is no {@code WHERE} narrowing, and against a materialized
 * filtered copy otherwise.
 *
 * <h2>WHERE / alias resolution</h2>
 * {@code WHERE} evaluates against the input root. A leaf naming a
 * {@code SELECT} alias that is not also a real column is expanded back into
 * its expression ({@link WhereAliasResolver}); a real column always wins.
 * Non-grouped {@code ORDER BY} works the same way (real column, or alias
 * expanded), so it may reference columns absent from the {@code SELECT} list.
 *
 * <h2>Predicate strategy</h2>
 * Without {@code IS [NOT] NULL}, the {@code WHERE} clause is fused into one
 * ParserNG boolean expression (fast path). With it, it is compiled
 * leaf-by-leaf into boolean masks combined in Java. A null predicate result
 * always excludes the row (SQL three-valued logic), for every
 * {@link NullPolicy}; {@code isNull(i)} is always tested before {@code get(i)}.
 * If every row passes, the selection collapses to the "all rows" sentinel so
 * the zero-copy paths apply.
 *
 * <h2>NullPolicy</h2>
 * As verified against parser-ng-arrow 3.0.6, only {@link NullPolicy#PROPAGATE}
 * produces correct output; {@link NullPolicy#IGNORE} currently marks every
 * output row null. The default is therefore PROPAGATE. Each call snapshots the
 * policy once at its start.
 *
 * <h2>Reusable output (hot path)</h2>
 * {@link #execute(VectorSchemaRoot)} allocates (and Arrow zeroes) fresh output
 * vectors on every call, which for large batches is a per-call cost the
 * reusable overload avoids. {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}
 * writes into a caller-owned buffer built by {@link #allocateReusableOutput};
 * it supports only queries without GROUP BY / HAVING / ORDER BY / LIMIT and is
 * <b>not</b> safe for concurrent use on one instance (one query per thread).
 * In steady state -- once its internal scratch buffers have grown to the
 * batch size -- this overload performs no heap allocations at all, including
 * for a {@code WHERE}-narrowed selection (the row-selection buffer is reused
 * and only grows, never reallocated per call). The one exception is a
 * {@code WHERE}/{@code HAVING} clause containing {@code IS [NOT] NULL}, whose
 * leaf-by-leaf mask fallback (see {@link PredicateNode}) still allocates; that
 * fallback is a correctness path, not a performance one. For benchmarking
 * against engines that preallocate outputs (e.g. Gandiva), use this overload.
 *
 * <h2>Resource ownership</h2>
 * The query owns its compiled evaluators and must be {@link #close()}d. Every
 * root returned by {@link #execute(VectorSchemaRoot)} is independent and owned
 * by the caller (passthrough columns may share refcounted buffers with the
 * input, exactly as Arrow's TransferPair does; closing either side is safe).
 *
 * <h2>Thread-safety</h2>
 * {@link #execute(VectorSchemaRoot)} is safe to call concurrently, including
 * with {@link #close()}, {@link #withBackend} and schema-driven recompilation.
 * The plan a call uses is reference-counted lock-free ({@link CompiledPlan}):
 * a superseded plan's evaluators are closed only after the last in-flight call
 * releases it. The cache is a single volatile {@code CompiledPlan} field keyed
 * purely by schema fingerprint (see {@link #fingerprintOf}), never by root
 * identity, so a fresh {@link VectorSchemaRoot} instance of an
 * already-compiled schema -- the common case in production, where every call
 * gets a new batch -- still takes the lock-free fast path instead of falling
 * through to synchronized recompilation. A call in flight during
 * {@code withBackend}/{@code withNullPolicy} completes with what it captured.
 *
 * <h2>Warming up</h2>
 * The first run of a compiled expression pays a one-time JIT/classload cost.
 * {@link #warmup(VectorSchemaRoot)} pays it on synthetic data of the same
 * schema, priming the real cached plan.
 *
 * @author GBEMIRO
 */
public final class ArrowQuery implements AutoCloseable {

    /** Default row count of the synthetic batch built by {@link #warmup(VectorSchemaRoot)}. */
    public static final int DEFAULT_WARMUP_ROWS = 20_000;

    /** Default repetition count for {@link #warmup(VectorSchemaRoot)}. */
    public static final int DEFAULT_WARMUP_REPETITIONS = 8;

    private static final int[] EMPTY = new int[0];

    private final String sql;
    private final SelectStatement stmt;

    private volatile ArrowExecutionBackend backend = ArrowExecutionBackend.CPU_SIMD;
    private volatile NullPolicy nullPolicy = NullPolicy.PROPAGATE;

    /**
     * The most recently compiled plan, published as one volatile write and
     * looked up purely by schema fingerprint (see {@link #fingerprintOf}) --
     * deliberately NOT keyed by the input root's identity. Keying by root
     * identity would force every call with a freshly-allocated (but
     * same-schema) batch through the synchronized slow path; keying by
     * fingerprint alone keeps that case on the lock-free, allocation-free
     * fast path, which is the common case in production. Null until first
     * compile / after close.
     */
    private volatile CompiledPlan cachedPlan;

    /** Only touched by {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}; see class docs. */
    private volatile ReusableScratch reusableScratch;

    /** Computed once from {@code stmt}; see {@link #requireReusableOutputSupportedShape()}. */
    private final String reusableUnsupportedReason;

    private ArrowQuery(String sql, SelectStatement stmt) {
        this.sql = sql;
        this.stmt = stmt;
        this.reusableUnsupportedReason = computeReusableUnsupportedReason(stmt);
    }

    private static String computeReusableUnsupportedReason(SelectStatement stmt) {
        if (stmt.isGrouped()) {
            return "GROUP BY";
        }
        if (stmt.having() != null) {
            return "HAVING";
        }
        if (!stmt.orderBy().isEmpty()) {
            return "ORDER BY";
        }
        if (stmt.limit() != null) {
            return "LIMIT";
        }
        return null;
    }

    /**
     * Parses {@code sql} into a reusable, not-yet-compiled query.
     *
     * @param sql the query text
     * @return the query
     * @throws SqlSyntaxException if {@code sql} does not conform to the grammar
     */
    public static ArrowQuery compile(String sql) {
        SelectStatement stmt = SqlParser.parse(sql);
        return new ArrowQuery(sql, stmt);
    }

    /** @return the parsed statement backing this query */
    public SelectStatement statement() {
        return stmt;
    }

    /**
     * Selects the parser-ng-arrow backend (default CPU_SIMD). Changing it
     * invalidates the compiled plan, safely w.r.t. in-flight calls.
     *
     * @param backend the backend
     * @return this
     */
    public ArrowQuery withBackend(ArrowExecutionBackend backend) {
        if (backend == null) {
            throw new NullPointerException("backend must not be null");
        }
        synchronized (this) {
            if (this.backend != backend) {
                this.backend = backend;
                invalidatePlan();
            }
        }
        return this;
    }

    /**
     * Selects the {@link NullPolicy} (default PROPAGATE; see class docs for why
     * IGNORE is not currently usable). No recompilation is needed.
     *
     * @param nullPolicy the policy
     * @return this
     */
    public ArrowQuery withNullPolicy(NullPolicy nullPolicy) {
        if (nullPolicy == null) {
            throw new NullPointerException("nullPolicy must not be null");
        }
        this.nullPolicy = nullPolicy;
        return this;
    }

    /**
     * Warms up using {@link #DEFAULT_WARMUP_ROWS} rows x {@link #DEFAULT_WARMUP_REPETITIONS}.
     *
     * @param schemaTemplate root whose schema (only) is used; never closed here
     * @return this
     */
    public ArrowQuery warmup(VectorSchemaRoot schemaTemplate) {
        return warmup(schemaTemplate, DEFAULT_WARMUP_ROWS, DEFAULT_WARMUP_REPETITIONS);
    }

    /**
     * Warms up with {@code rows} synthetic rows repeated {@code repetitions} times.
     *
     * @param schemaTemplate root whose schema (only) is used; never closed here
     * @param rows synthetic row count; positive
     * @param repetitions execution count; positive
     * @return this
     */
    public ArrowQuery warmup(VectorSchemaRoot schemaTemplate, int rows, int repetitions) {
        if (schemaTemplate == null) {
            throw new NullPointerException("schemaTemplate must not be null");
        }
        if (rows <= 0) {
            throw new IllegalArgumentException("rows must be positive, was " + rows);
        }
        if (repetitions <= 0) {
            throw new IllegalArgumentException("repetitions must be positive, was " + repetitions);
        }
        try (RootAllocator syntheticAllocator = new RootAllocator(Long.MAX_VALUE)) {
            VectorSchemaRoot synthetic = buildSyntheticRoot(schemaTemplate.getSchema(), rows, syntheticAllocator);
            try {
                for (int i = 0; i < repetitions; i++) {
                    closeAll(execute(synthetic));
                }
            } finally {
                closeAll(synthetic);
            }
        }
        return this;
    }

    private static VectorSchemaRoot buildSyntheticRoot(Schema schema, int rows, BufferAllocator allocator) {
        Random random = new Random(0x50415252_4E47L);
        List<Field> fields = new ArrayList<>(schema.getFields().size());
        List<FieldVector> vectors = new ArrayList<>(schema.getFields().size());
        try {
            for (Field templateField : schema.getFields()) {
                Field field = new Field(templateField.getName(), templateField.getFieldType(), templateField.getChildren());
                FieldVector fv = field.createVector(allocator);
                vectors.add(fv);
                if (fv instanceof Float8Vector v) {
                    v.allocateNew(rows);
                    for (int i = 0; i < rows; i++) {
                        v.set(i, (random.nextDouble() * 2000) - 1000);
                    }
                } else if (fv instanceof Float4Vector v) {
                    v.allocateNew(rows);
                    for (int i = 0; i < rows; i++) {
                        v.set(i, (random.nextFloat() * 2000f) - 1000f);
                    }
                } else {
                    fv.allocateNew();
                }
                fv.setValueCount(rows);
                fields.add(field);
            }
        } catch (RuntimeException | Error e) {
            for (FieldVector v : vectors) {
                closeQuietly(v);
            }
            throw e;
        }
        return new VectorSchemaRoot(new Schema(fields), vectors, rows);
    }

    // =====================================================================
    // public execution API
    // =====================================================================

    /**
     * Executes this query against {@code root}. Safe for concurrent use.
     *
     * @param root the input batch
     * @return a fresh, independently-owned result the caller must close
     * @throws ArrowSqlException if compilation against {@code root}'s schema fails
     * @throws ArrowBindingException if evaluation fails at runtime
     */
    public VectorSchemaRoot execute(VectorSchemaRoot root) {
        if (root == null) {
            throw new NullPointerException("root must not be null");
        }
        NullPolicy effectiveNullPolicy = this.nullPolicy;
        CompiledPlan p = acquirePlan(root);
        try {
            return runPlan(p, root, effectiveNullPolicy);
        } finally {
            p.release();
        }
    }

    /**
     * Like {@link #execute(VectorSchemaRoot)} but writes into a caller-owned
     * {@code reusableOutput} (see {@link #allocateReusableOutput}) instead of
     * allocating a result. Only for queries without GROUP BY / HAVING / ORDER BY
     * / LIMIT. <b>Not safe for concurrent use on one instance.</b>
     *
     * <p>Every column of {@code reusableOutput} must have capacity of at least
     * {@code root.getRowCount()}. Nothing is (re)allocated on your behalf.
     *
     * @param root the input batch
     * @param reusableOutput caller-owned output; returned after being populated
     * @return {@code reusableOutput}
     * @throws UnsupportedOperationException for an unsupported query shape
     * @throws IllegalArgumentException if {@code reusableOutput} is undersized
     */
    public VectorSchemaRoot execute(VectorSchemaRoot root, VectorSchemaRoot reusableOutput) {
        if (root == null) {
            throw new NullPointerException("root must not be null");
        }
        if (reusableOutput == null) {
            throw new NullPointerException("reusableOutput must not be null");
        }
        requireReusableOutputSupportedShape();

        NullPolicy effectiveNullPolicy = this.nullPolicy;
        CompiledPlan p = acquirePlan(root);
        try {
            int rowCount = root.getRowCount();
            BufferAllocator allocator = allocatorOf(root);
            ReusableScratch scratch = scratchFor(p);
            int[] selected = selectRowsReusable(p, root, effectiveNullPolicy, scratch, allocator);
            int outRowCount = selected == null ? rowCount : scratch.selectionCount;
            List<FieldVector> outVectors = reusableOutput.getFieldVectors();

            if (stmt.selectAll()) {
                List<FieldVector> sourceVectors = root.getFieldVectors();
                validateReusableOutputShape(sourceVectors.size(), outVectors, rowCount);
                for (int i = 0; i < sourceVectors.size(); i++) {
                    gatherInto(sourceVectors.get(i), selected, outVectors.get(i), outRowCount);
                }
                reusableOutput.setRowCount(outRowCount);
                return reusableOutput;
            }

            List<ProjectionPlan> projections = p.projections;
            validateReusableOutputShape(projections.size(), outVectors, rowCount);
            for (int i = 0; i < projections.size(); i++) {
                ProjectionPlan proj = projections.get(i);
                FieldVector out = outVectors.get(i);
                if (proj.passthrough) {
                    FieldVector src = resolvePassthrough(root, proj.sourceColumnIndex, proj.sourceColumnName);
                    gatherInto(src, selected, out, outRowCount);
                } else if (selected == null) {
                    out.setValueCount(rowCount);
                    proj.evaluator.evaluate(root, out, effectiveNullPolicy);
                } else {
                    FieldVector scratchVec = scratch.projectionScratch(rowCount, p.float64, allocator);
                    proj.evaluator.evaluate(root, scratchVec, effectiveNullPolicy);
                    gatherInto(scratchVec, selected, out, outRowCount);
                }
            }
            reusableOutput.setRowCount(outRowCount);
            return reusableOutput;
        } finally {
            p.release();
        }
    }

    /**
     * Builds a root shaped/typed to pass as {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}'s
     * {@code reusableOutput}, with every column allocated for at least {@code maxRows}.
     *
     * @param schemaTemplate root with the schema real calls will use (schema only; not closed)
     * @param maxRows capacity per column; positive
     * @return a zero-row root the caller owns and must close
     */
    public VectorSchemaRoot allocateReusableOutput(VectorSchemaRoot schemaTemplate, int maxRows) {
        if (schemaTemplate == null) {
            throw new NullPointerException("schemaTemplate must not be null");
        }
        if (maxRows <= 0) {
            throw new IllegalArgumentException("maxRows must be positive, was " + maxRows);
        }
        requireReusableOutputSupportedShape();

        CompiledPlan p = acquirePlan(schemaTemplate);
        try {
            BufferAllocator allocator = allocatorOf(schemaTemplate);
            List<Field> outFields = new ArrayList<>();
            List<FieldVector> outVectors = new ArrayList<>();
            try {
                if (stmt.selectAll()) {
                    for (FieldVector src : schemaTemplate.getFieldVectors()) {
                        FieldVector v = src.getField().createVector(allocator);
                        outVectors.add(v);
                        v.setInitialCapacity(maxRows);
                        v.allocateNew();
                        outFields.add(v.getField());
                    }
                } else {
                    for (ProjectionPlan proj : p.projections) {
                        FieldType type;
                        if (proj.passthrough) {
                            FieldVector src = schemaTemplate.getVector(proj.sourceColumnName);
                            if (src == null) {
                                throw new ArrowBindingException(
                                        "Column '" + proj.sourceColumnName + "' not found in schemaTemplate.");
                            }
                            type = src.getField().getFieldType();
                        } else {
                            type = floatFieldType(p.float64);
                        }
                        Field field = new Field(proj.outputName, type, null);
                        FieldVector v = field.createVector(allocator);
                        outVectors.add(v);
                        v.setInitialCapacity(maxRows);
                        v.allocateNew();
                        outFields.add(field);
                    }
                }
            } catch (RuntimeException | Error e) {
                for (FieldVector v : outVectors) {
                    closeQuietly(v);
                }
                throw e;
            }
            return new VectorSchemaRoot(new Schema(outFields), outVectors, 0);
        } finally {
            p.release();
        }
    }

    private void requireReusableOutputSupportedShape() {
        if (reusableUnsupportedReason != null) {
            throw new UnsupportedOperationException(
                    "execute(root, reusableOutput)/allocateReusableOutput do not support " + reusableUnsupportedReason
                            + " -- the output row count cannot be bounded by a fixed-capacity buffer. "
                            + "Use execute(root) instead.");
        }
    }

    private static void validateReusableOutputShape(int expectedColumns, List<FieldVector> outVectors, int rowCount) {
        if (outVectors.size() != expectedColumns) {
            throw new IllegalArgumentException(
                    "reusableOutput has " + outVectors.size() + " column(s) but this query produces "
                            + expectedColumns + ". Build reusableOutput with allocateReusableOutput(...) "
                            + "against the same query.");
        }
        for (int i = 0; i < outVectors.size(); i++) {
            FieldVector v = outVectors.get(i);
            if (v.getValueCapacity() < rowCount) {
                throw new IllegalArgumentException(
                        "reusableOutput column " + i + " (\"" + v.getField().getName() + "\") has capacity "
                                + v.getValueCapacity() + " but root has " + rowCount + " rows; size it to at "
                                + "least root's row count. See allocateReusableOutput.");
            }
        }
    }

    /**
     * Not synchronized: {@link #execute(VectorSchemaRoot, VectorSchemaRoot)} is
     * documented as single-thread-only per instance, so this needs no lock --
     * and taking one here would defeat the point of the zero-alloc hot path.
     */
    private ReusableScratch scratchFor(CompiledPlan p) {
        ReusableScratch s = reusableScratch;
        if (s == null || s.forPlan != p) {
            if (s != null) {
                s.close();
            }
            s = new ReusableScratch(p);
            reusableScratch = s;
        }
        return s;
    }

    /**
     * Zero-allocation WHERE evaluation for the reusable-output path, steady
     * state: once {@code scratch}'s buffers have grown to the batch size, no
     * further allocations occur here. Returns {@code null} for "every row, in
     * order" (read {@code root.getRowCount()} rows); otherwise returns
     * {@code scratch}'s reusable selection buffer, which may be oversized from
     * a larger previous call -- read only the first {@code scratch.selectionCount}
     * entries.
     *
     * <p>The {@code IS [NOT] NULL} mask fallback still allocates a
     * {@code boolean[]} per leaf (see {@link PredicateNode}); only the fused
     * (no-IS-NULL) predicate path is allocation-free.
     */
    private static int[] selectRowsReusable(
            CompiledPlan p, VectorSchemaRoot root, NullPolicy nullPolicy, ReusableScratch scratch,
            BufferAllocator allocator) {
        if (p.fusedPredicate == null && p.maskPredicateRoot == null) {
            return null;
        }
        if (p.maskPredicateRoot != null) {
            boolean[] mask = p.maskPredicateRoot.evalMask(root, nullPolicy, p.float64);
            return selectionFromMask(mask, scratch);
        }
        int rowCount = root.getRowCount();
        if (rowCount == 0) {
            scratch.selectionCount = 0;
            return EMPTY;
        }
        FieldVector out = scratch.predicateScratch(rowCount, p.float64, allocator);
        long[] bits = scratch.bits((rowCount + 63) >>> 6);
        p.fusedPredicate.evaluate(root, out, nullPolicy);
        int count = countAndMark(out, bits, rowCount, p.float64);
        if (count == rowCount) {
            return null;
        }
        if (count == 0) {
            scratch.selectionCount = 0;
            return EMPTY;
        }
        int[] buf = scratch.selectionBuffer(count);
        expandBitsInto(bits, buf, rowCount);
        scratch.selectionCount = count;
        return buf;
    }

    private static int[] selectionFromMask(boolean[] mask, ReusableScratch scratch) {
        int count = 0;
        for (boolean b : mask) {
            if (b) {
                count++;
            }
        }
        if (count == mask.length) {
            return null;
        }
        if (count == 0) {
            scratch.selectionCount = 0;
            return EMPTY;
        }
        int[] buf = scratch.selectionBuffer(count);
        int idx = 0;
        for (int i = 0; i < mask.length; i++) {
            if (mask[i]) {
                buf[idx++] = i;
            }
        }
        scratch.selectionCount = count;
        return buf;
    }

    // =====================================================================
    // lifecycle
    // =====================================================================

    /**
     * Releases every compiled evaluator, or marks them for release once any
     * in-flight {@link #execute(VectorSchemaRoot)} finishes. Idempotent and
     * safe to call concurrently with execution.
     */
    @Override
    public synchronized void close() {
        invalidatePlan();
    }

    private synchronized void invalidatePlan() {
        CompiledPlan old = cachedPlan;
        cachedPlan = null;
        if (old != null) {
            old.retire();
        }
        ReusableScratch s = reusableScratch;
        if (s != null) {
            s.close();
            reusableScratch = null;
        }
    }

    // =====================================================================
    // plan compilation (lazy, cached, schema-fingerprinted, ref-counted)
    // =====================================================================

    /** Returns the plan for {@code root}, already acquired; caller MUST release() it in a finally. */
    private CompiledPlan acquirePlan(VectorSchemaRoot root) {
        long fingerprint = fingerprintOf(root);
        // Lock-free, allocation-free fast path: same schema fingerprint as the
        // cached plan, regardless of whether `root` is the same instance. A new
        // batch every call (same schema) is the common production case and must
        // not fall through to the synchronized path below.
        CompiledPlan p = cachedPlan;
        if (p != null && p.fingerprint == fingerprint && p.acquire()) {
            return p;
        }
        try {
            return acquirePlanSlow(root, fingerprint);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new ArrowSqlException(
                    "Failed to compile query \"" + sql + "\" against the given schema: " + t.getMessage(), t);
        }
    }

    private synchronized CompiledPlan acquirePlanSlow(VectorSchemaRoot root, long fingerprint) throws Throwable {
        CompiledPlan existing = cachedPlan;
        if (existing != null && existing.fingerprint == fingerprint && existing.acquire()) {
            return existing;
        }
        CompiledPlan fresh = buildPlan(root, fingerprint);
        fresh.acquire(); // brand new: never retired, always succeeds
        cachedPlan = fresh; // published once, fully built
        if (existing != null) {
            existing.retire(); // evaluators closed once in-flight users release it
        }
        ReusableScratch s = reusableScratch;
        if (s != null) {
            s.close();
            reusableScratch = null;
        }
        return fresh;
    }

    private CompiledPlan buildPlan(VectorSchemaRoot root, long fingerprint) throws Throwable {
        boolean float64 = isFloat64(root);

        // One evaluator per distinct ParserNG text within this plan.
        Map<String, ArrowExpressionEvaluator> compiledCache = new LinkedHashMap<>();
        try {
            BoolExpr where = stmt.where();
            if (where != null) {
                Map<String, String> aliasBindings = collectWhereAliasBindings(root);
                if (!aliasBindings.isEmpty()) {
                    try {
                        where = WhereAliasResolver.resolve(where, aliasBindings);
                    } catch (IllegalArgumentException cyclicAlias) {
                        throw new ArrowSqlException(
                                "Failed to compile query \"" + sql + "\": " + cyclicAlias.getMessage(), cyclicAlias);
                    }
                }
            }

            ArrowExpressionEvaluator fusedPredicate = null;
            PredicateNode maskPredicateRoot = null;
            if (where != null) {
                if (BoolExprs.containsIsNull(where)) {
                    maskPredicateRoot = buildPredicateNode(where, backend, float64, root, compiledCache);
                } else {
                    fusedPredicate = compileCached(BoolExprs.renderFused(where), backend, float64, compiledCache);
                }
            }

            List<ProjectionPlan> projections = new ArrayList<>();
            GroupPlan groupPlan = null;
            if (stmt.isGrouped()) {
                groupPlan = buildGroupPlan(root, backend, float64, compiledCache);
            } else if (!stmt.selectAll()) {
                for (SelectItem item : stmt.items()) {
                    String expr = item.exprText();
                    String outputName = item.outputName();
                    int idx = indexOfColumn(root, expr);
                    if (idx >= 0) {
                        projections.add(new ProjectionPlan(outputName, true, expr, idx, null));
                    } else {
                        projections.add(new ProjectionPlan(outputName, false, null, -1,
                                compileCached(expr, backend, float64, compiledCache)));
                    }
                }
            }

            PredicateNode havingNode = null;
            ArrowExpressionEvaluator havingFused = null;
            if (stmt.having() != null) {
                BoolExpr having = stmt.having();
                if (BoolExprs.containsIsNull(having)) {
                    VectorSchemaRoot phantom = phantomResultRoot(float64);
                    try {
                        havingNode = buildPredicateNode(having, backend, float64, phantom, compiledCache);
                    } finally {
                        closePhantomRoot(phantom);
                    }
                } else {
                    havingFused = compileCached(BoolExprs.renderFused(having), backend, float64, compiledCache);
                }
            }

            List<OrderByPlan> orderByPlans = new ArrayList<>(stmt.orderBy().size());
            if (stmt.isGrouped()) {
                for (OrderItem item : stmt.orderBy()) {
                    orderByPlans.add(new OrderByPlan(false, null,
                            compileCached(item.exprText(), backend, float64, compiledCache)));
                }
            } else {
                Map<String, String> orderByAliasBindings = collectWhereAliasBindings(root);
                for (OrderItem item : stmt.orderBy()) {
                    String resolved;
                    try {
                        resolved = WhereAliasResolver.substituteText(item.exprText(), orderByAliasBindings);
                    } catch (IllegalArgumentException cyclicAlias) {
                        throw new ArrowSqlException(
                                "Failed to compile query \"" + sql + "\": " + cyclicAlias.getMessage(), cyclicAlias);
                    }
                    if (root.getVector(resolved) != null) {
                        orderByPlans.add(new OrderByPlan(true, resolved, null));
                    } else {
                        orderByPlans.add(new OrderByPlan(false, null,
                                compileCached(resolved, backend, float64, compiledCache)));
                    }
                }
            }

            return new CompiledPlan(fingerprint, float64, fusedPredicate, maskPredicateRoot, projections,
                    groupPlan, havingFused, havingNode, orderByPlans,
                    List.copyOf(compiledCache.values()));
        } catch (Throwable t) {
            // Do not leak evaluators compiled before a later failure.
            for (ArrowExpressionEvaluator e : compiledCache.values()) {
                try {
                    e.close();
                } catch (RuntimeException ignored) {
                    // best-effort
                }
            }
            throw t;
        }
    }

    private static ArrowExpressionEvaluator compileCached(
            String text, ArrowExecutionBackend backend, boolean float64,
            Map<String, ArrowExpressionEvaluator> cache) throws Throwable {
        ArrowExpressionEvaluator existing = cache.get(text);
        if (existing != null) {
            return existing;
        }
        ArrowExpressionEvaluator fresh = float64
                ? ArrowExpressionEvaluators.compile(text, backend)
                : ArrowExpressionEvaluators.compileF32(text, backend);
        cache.put(text, fresh);
        return fresh;
    }

    private Map<String, String> collectWhereAliasBindings(VectorSchemaRoot root) {
        if (stmt.selectAll()) {
            return Map.of();
        }
        Map<String, String> bindings = new LinkedHashMap<>();
        for (SelectItem item : stmt.items()) {
            String outputName = item.outputName();
            if (root.getVector(outputName) != null) {
                continue; // a real column always wins the name
            }
            bindings.put(outputName, item.exprText());
        }
        return bindings;
    }

    private static PredicateNode buildPredicateNode(
            BoolExpr expr, ArrowExecutionBackend backend, boolean float64, VectorSchemaRoot root,
            Map<String, ArrowExpressionEvaluator> compiledCache) throws Throwable {

        if (expr instanceof AndExpr a) {
            return new AndNode(
                    buildPredicateNode(a.left(), backend, float64, root, compiledCache),
                    buildPredicateNode(a.right(), backend, float64, root, compiledCache));
        }
        if (expr instanceof OrExpr o) {
            return new OrNode(
                    buildPredicateNode(o.left(), backend, float64, root, compiledCache),
                    buildPredicateNode(o.right(), backend, float64, root, compiledCache));
        }
        if (expr instanceof IsNullExpr n) {
            String target = n.target().trim();
            if (root.getVector(target) != null) {
                return new IsNullLeafNode(target, null, n.negated());
            }
            return new IsNullLeafNode(null, compileCached(target, backend, float64, compiledCache), n.negated());
        }
        return new CompareLeafNode(compileCached(BoolExprs.renderLeaf(expr), backend, float64, compiledCache));
    }

    private GroupPlan buildGroupPlan(
            VectorSchemaRoot root, ArrowExecutionBackend backend, boolean float64,
            Map<String, ArrowExpressionEvaluator> compiledCache) throws Throwable {

        List<KeyPlan> keys = new ArrayList<>(stmt.groupBy().size());
        for (String keyExpr : stmt.groupBy()) {
            if (root.getVector(keyExpr) != null) {
                keys.add(new KeyPlan(keyExpr, true, keyExpr, null));
            } else {
                keys.add(new KeyPlan(keyExpr, false, null, compileCached(keyExpr, backend, float64, compiledCache)));
            }
        }

        List<SelectItem> items = stmt.items();
        int[] itemKeyIndex = new int[items.size()];
        int[] itemAggIndex = new int[items.size()];
        List<AggPlan> aggregates = new ArrayList<>();

        for (int i = 0; i < items.size(); i++) {
            SelectItem item = items.get(i);
            if (item.isAggregate()) {
                AggregateSpec spec = item.aggregate();
                String passthroughColumn = null;
                ArrowExpressionEvaluator eval = null;
                if (!spec.star()) {
                    if (root.getVector(spec.argExprText()) != null) {
                        passthroughColumn = spec.argExprText();
                    } else {
                        eval = compileCached(spec.argExprText(), backend, float64, compiledCache);
                    }
                }
                aggregates.add(new AggPlan(spec.func(), spec.star(), passthroughColumn, eval));
                itemAggIndex[i] = aggregates.size() - 1;
                itemKeyIndex[i] = -1;
            } else {
                int keyIdx = -1;
                String trimmed = item.exprText().trim();
                for (int k = 0; k < stmt.groupBy().size(); k++) {
                    if (stmt.groupBy().get(k).trim().equals(trimmed)) {
                        keyIdx = k;
                        break;
                    }
                }
                if (keyIdx < 0) {
                    throw new ArrowSqlException(
                            "Failed to compile query \"" + sql + "\": SELECT item \"" + item.exprText()
                                    + "\" is neither an aggregate call nor one of the GROUP BY expressions ("
                                    + stmt.groupBy() + "). A grouped query's non-aggregate SELECT items must "
                                    + "match a GROUP BY key exactly -- see SelectStatement's javadoc.", null);
                }
                itemKeyIndex[i] = keyIdx;
                itemAggIndex[i] = -1;
            }
        }
        return new GroupPlan(keys, aggregates, itemKeyIndex, itemAggIndex);
    }

    private VectorSchemaRoot phantomResultRoot(boolean float64) {
        BufferAllocator allocator = new RootAllocator(1024);
        List<Field> fields = new ArrayList<>(stmt.items().size());
        List<FieldVector> vectors = new ArrayList<>(stmt.items().size());
        for (SelectItem item : stmt.items()) {
            Field field = new Field(item.outputName(), floatFieldType(float64), null);
            FieldVector v = field.createVector(allocator);
            v.setInitialCapacity(0);
            v.allocateNew();
            v.setValueCount(0);
            fields.add(field);
            vectors.add(v);
        }
        return new VectorSchemaRoot(new Schema(fields), vectors, 0);
    }

    private static void closePhantomRoot(VectorSchemaRoot phantom) {
        BufferAllocator allocator = phantom.getFieldVectors().isEmpty() ? null : allocatorOf(phantom);
        closeAll(phantom);
        if (allocator != null) {
            try {
                allocator.close();
            } catch (RuntimeException ignored) {
                // best-effort cleanup of the throwaway allocator
            }
        }
    }

    private static FieldType floatFieldType(boolean float64) {
        return FieldType.nullable(new ArrowType.FloatingPoint(
                float64 ? FloatingPointPrecision.DOUBLE : FloatingPointPrecision.SINGLE));
    }

    /** Allocation-free schema fingerprint (column-name hash + minor-type ordinal, List.hashCode-style). */
    private static long fingerprintOf(VectorSchemaRoot root) {
        List<FieldVector> vectors = root.getFieldVectors();
        long h = 1125899906842597L;
        for (FieldVector v : vectors) {
            h = 31 * h + v.getName().hashCode();
            h = 31 * h + v.getMinorType().ordinal();
        }
        return h;
    }

    /** @return true iff every column is a Float8Vector (parser-ng-arrow's convention for double kernels) */
    private static boolean isFloat64(VectorSchemaRoot root) {
        List<FieldVector> vectors = root.getFieldVectors();
        if (vectors.isEmpty()) {
            throw new ArrowBindingException("Cannot execute a query against a VectorSchemaRoot with no columns.");
        }
        for (FieldVector v : vectors) {
            if (!(v instanceof Float8Vector)) {
                return false;
            }
        }
        return true;
    }

    private static BufferAllocator allocatorOf(VectorSchemaRoot root) {
        List<FieldVector> vectors = root.getFieldVectors();
        if (vectors.isEmpty()) {
            throw new ArrowBindingException("Cannot execute a query against a VectorSchemaRoot with no columns.");
        }
        return vectors.get(0).getAllocator();
    }

    /** @return the index of the field named {@code name} within {@code root}, or -1 if absent. */
    private static int indexOfColumn(VectorSchemaRoot root, String name) {
        List<FieldVector> vectors = root.getFieldVectors();
        for (int i = 0; i < vectors.size(); i++) {
            if (vectors.get(i).getName().equals(name)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Resolves a passthrough projection's source column. The fast path is a
     * direct positional lookup at {@code index} -- O(1), no string comparison
     * against every column -- which is valid because {@link #fingerprintOf}
     * folds column name and type via a sequential (order-sensitive) hash, so
     * any root sharing a plan's fingerprint has its fields in the exact order
     * the plan was compiled against. Falls back to a by-name scan only if that
     * invariant is somehow violated (e.g. an astronomically unlikely
     * fingerprint collision across differently-ordered schemas), so a bad
     * assumption fails safe instead of silently reading the wrong column.
     */
    private static FieldVector resolvePassthrough(VectorSchemaRoot root, int index, String name) {
        List<FieldVector> vectors = root.getFieldVectors();
        if (index >= 0 && index < vectors.size()) {
            FieldVector v = vectors.get(index);
            if (v.getName().equals(name)) {
                return v;
            }
        }
        FieldVector v = root.getVector(name);
        if (v == null) {
            throw new ArrowBindingException("Column '" + name + "' not found while projecting.");
        }
        return v;
    }

    private static void closeQuietly(FieldVector v) {
        try {
            v.close();
        } catch (RuntimeException ignored) {
            // best-effort cleanup while unwinding a different failure
        }
    }

    private static void closeAll(VectorSchemaRoot r) {
        for (FieldVector v : r.getFieldVectors()) {
            closeQuietly(v);
        }
    }

    // =====================================================================
    // per-call execution
    // =====================================================================

    private VectorSchemaRoot runPlan(CompiledPlan p, VectorSchemaRoot root, NullPolicy nullPolicy) {
        return p.groupPlan == null
                ? runNonGrouped(p, root, nullPolicy)
                : runGrouped(p, root, nullPolicy);
    }

    /**
     * WHERE -> ORDER BY -> LIMIT all operate on a row selection over the
     * untouched input root; projection is the only step that touches column data.
     */
    private VectorSchemaRoot runNonGrouped(CompiledPlan p, VectorSchemaRoot root, NullPolicy nullPolicy) {
        int rowCount = root.getRowCount();
        int[] selected = selectRows(p, root, nullPolicy); // null = every row, in order
        if (!p.orderByPlans.isEmpty()) {
            selected = sortedSelection(p, root, selected, nullPolicy);
        }
        int n = selected == null ? rowCount : selected.length;
        Integer limit = stmt.limit();
        if (limit != null && limit < n) {
            n = limit; // only the first n entries of `selected` are ever read
        }
        return project(p, root, selected, n, nullPolicy);
    }

    /**
     * Builds the result from {@code root}: {@code n} output rows, row i taken
     * from {@code root} row {@code selected[i]} (or row i when {@code selected} is null).
     */
    private VectorSchemaRoot project(
            CompiledPlan p, VectorSchemaRoot root, int[] selected, int n, NullPolicy nullPolicy) {
        BufferAllocator allocator = allocatorOf(root);
        int rowCount = root.getRowCount();
        boolean gather = selected != null || n != rowCount;
        List<FieldVector> outVectors = new ArrayList<>();
        List<Field> outFields = new ArrayList<>();
        FieldVector scratch = null; // one full-batch scratch, shared by all projections of this call
        try {
            if (stmt.selectAll()) {
                for (FieldVector src : root.getFieldVectors()) {
                    FieldVector out = copyColumn(src, src.getName(), selected, n, gather, allocator);
                    outVectors.add(out);
                    outFields.add(out.getField());
                }
            } else {
                for (ProjectionPlan proj : p.projections) {
                    FieldVector out;
                    if (proj.passthrough) {
                        FieldVector src = resolvePassthrough(root, proj.sourceColumnIndex, proj.sourceColumnName);
                        out = copyColumn(src, proj.outputName, selected, n, gather, allocator);
                    } else if (!gather) {
                        out = newFloat(proj.outputName, rowCount, p.float64, allocator);
                        try {
                            proj.evaluator.evaluate(root, out, nullPolicy);
                        } catch (RuntimeException | Error e) {
                            closeQuietly(out);
                            throw e;
                        }
                    } else if (n == 0) {
                        out = newFloat(proj.outputName, 0, p.float64, allocator);
                    } else {
                        if (scratch == null) {
                            scratch = newFloat("__parser_ng_sql_scratch__", rowCount, p.float64, allocator);
                        }
                        proj.evaluator.evaluate(root, scratch, nullPolicy);
                        out = newFloat(proj.outputName, n, p.float64, allocator);
                        try {
                            gatherInto(scratch, selected, out, n);
                        } catch (RuntimeException | Error e) {
                            closeQuietly(out);
                            throw e;
                        }
                    }
                    outVectors.add(out);
                    outFields.add(out.getField());
                }
            }
        } catch (RuntimeException | Error e) {
            for (FieldVector v : outVectors) {
                closeQuietly(v);
            }
            throw e;
        } finally {
            if (scratch != null) {
                closeQuietly(scratch);
            }
        }
        return new VectorSchemaRoot(new Schema(outFields), outVectors, n);
    }

    private static FieldVector newFloat(String name, int rows, boolean float64, BufferAllocator allocator) {
        if (float64) {
            Float8Vector v = new Float8Vector(name, allocator);
            v.allocateNew(rows);
            v.setValueCount(rows);
            return v;
        }
        Float4Vector v = new Float4Vector(name, allocator);
        v.allocateNew(rows);
        v.setValueCount(rows);
        return v;
    }

    /**
     * Independent output column named {@code name}: a buffer transfer when the
     * whole column is kept as-is, otherwise a typed gather of {@code n} rows.
     */
    private static FieldVector copyColumn(
            FieldVector src, String name, int[] selected, int n, boolean gather, BufferAllocator allocator) {
        if (!gather) {
            TransferPair tp = src.getTransferPair(name, allocator);
            tp.splitAndTransfer(0, n);
            return (FieldVector) tp.getTo();
        }
        Field f = src.getField();
        FieldVector dst = new Field(name, f.getFieldType(), f.getChildren()).createVector(allocator);
        try {
            if (n > 0) {
                dst.setInitialCapacity(n);
            }
            dst.allocateNew();
            gatherInto(src, selected, dst, n);
        } catch (RuntimeException | Error e) {
            closeQuietly(dst);
            throw e;
        }
        return dst;
    }

    /**
     * Copies {@code n} rows into {@code dst} (capacity must already suffice):
     * row i from {@code src[selected[i]]}, or {@code src[i]} when
     * {@code selected == null}. Typed fast paths for float columns (including a
     * bulk buffer copy for the full-column case); generic {@code copyFromSafe}
     * otherwise. Nulls are always written explicitly, so {@code dst} need not
     * be zeroed (required for reusable output buffers). Only {@code selected[0..n)}
     * is ever read, so a caller-supplied {@code selected} may be oversized.
     */
    private static void gatherInto(FieldVector src, int[] selected, FieldVector dst, int n) {
        if (src instanceof Float8Vector s && dst instanceof Float8Vector d) {
            if (selected == null && n == s.getValueCount()) {
                d.getDataBuffer().setBytes(0, s.getDataBuffer(), 0, (long) n * Float8Vector.TYPE_WIDTH);
                d.getValidityBuffer().setBytes(0, s.getValidityBuffer(), 0, (n + 7L) >>> 3);
            } else {
                for (int i = 0; i < n; i++) {
                    int j = selected == null ? i : selected[i];
                    if (s.isNull(j)) {
                        d.setNull(i);
                    } else {
                        d.set(i, s.get(j));
                    }
                }
            }
        } else if (src instanceof Float4Vector s && dst instanceof Float4Vector d) {
            if (selected == null && n == s.getValueCount()) {
                d.getDataBuffer().setBytes(0, s.getDataBuffer(), 0, (long) n * Float4Vector.TYPE_WIDTH);
                d.getValidityBuffer().setBytes(0, s.getValidityBuffer(), 0, (n + 7L) >>> 3);
            } else {
                for (int i = 0; i < n; i++) {
                    int j = selected == null ? i : selected[i];
                    if (s.isNull(j)) {
                        d.setNull(i);
                    } else {
                        d.set(i, s.get(j));
                    }
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                dst.copyFromSafe(selected == null ? i : selected[i], i, src);
            }
        }
        dst.setValueCount(n);
    }

    private VectorSchemaRoot runGrouped(CompiledPlan p, VectorSchemaRoot root, NullPolicy nullPolicy) {
        int[] selected = selectRows(p, root, nullPolicy);
        // No WHERE narrowing: aggregate straight over the input, no copy at all.
        VectorSchemaRoot filtered = selected == null ? root : materializeRows(root, selected);
        VectorSchemaRoot result;
        try {
            result = buildGroupedResult(p, filtered, nullPolicy);
        } finally {
            if (filtered != root) {
                closeAll(filtered);
            }
        }
        try {
            if (p.havingFused != null || p.havingNode != null) {
                VectorSchemaRoot next = applyHaving(p, result, nullPolicy);
                closeAll(result);
                result = next;
            }
            if (!p.orderByPlans.isEmpty()) {
                VectorSchemaRoot next = materializeRows(result, sortedSelection(p, result, null, nullPolicy));
                closeAll(result);
                result = next;
            }
            Integer limit = stmt.limit();
            if (limit != null && limit < result.getRowCount()) {
                VectorSchemaRoot next = materializePrefix(result, limit);
                closeAll(result);
                result = next;
            }
        } catch (RuntimeException | Error e) {
            closeAll(result);
            throw e;
        }
        return result;
    }

    private VectorSchemaRoot buildGroupedResult(CompiledPlan p, VectorSchemaRoot filtered, NullPolicy nullPolicy) {
        GroupPlan gp = p.groupPlan;
        int rowCount = filtered.getRowCount();
        BufferAllocator allocator = allocatorOf(filtered);

        int numKeys = gp.keys.size();
        double[][] keyValues = new double[numKeys][];
        boolean[][] keyIsNull = new boolean[numKeys][];
        for (int k = 0; k < numKeys; k++) {
            keyValues[k] = new double[rowCount];
            keyIsNull[k] = new boolean[rowCount];
            KeyPlan kp = gp.keys.get(k);
            evaluateNumericColumn(kp.evaluator, kp.sourceColumnName, filtered, p.float64, nullPolicy,
                    keyValues[k], keyIsNull[k]);
        }

        int numAggs = gp.aggregates.size();
        double[][] aggValues = new double[numAggs][];
        boolean[][] aggIsNull = new boolean[numAggs][];
        for (int a = 0; a < numAggs; a++) {
            AggPlan ap = gp.aggregates.get(a);
            if (ap.star) {
                continue;
            }
            aggValues[a] = new double[rowCount];
            aggIsNull[a] = new boolean[rowCount];
            evaluateNumericColumn(ap.evaluator, ap.passthroughColumn, filtered, p.float64, nullPolicy,
                    aggValues[a], aggIsNull[a]);
        }

        Map<GroupKey, Integer> groupIndex = new LinkedHashMap<>();
        List<double[]> groupKeyValuesOut = new ArrayList<>();
        List<Aggregator[]> groupAggregators = new ArrayList<>();

        for (int row = 0; row < rowCount; row++) {
            double[] key = new double[numKeys];
            for (int k = 0; k < numKeys; k++) {
                key[k] = keyIsNull[k][row] ? Double.NaN : keyValues[k][row];
            }
            GroupKey gk = new GroupKey(key);
            Integer idx = groupIndex.get(gk);
            if (idx == null) {
                idx = groupAggregators.size();
                groupIndex.put(gk, idx);
                groupKeyValuesOut.add(key);
                Aggregator[] aggs = new Aggregator[numAggs];
                for (int a = 0; a < numAggs; a++) {
                    aggs[a] = newAggregator(gp.aggregates.get(a).func);
                }
                groupAggregators.add(aggs);
            }
            Aggregator[] aggs = groupAggregators.get(idx);
            for (int a = 0; a < numAggs; a++) {
                if (gp.aggregates.get(a).star) {
                    aggs[a].accumulate(0.0, false);
                } else {
                    aggs[a].accumulate(aggValues[a][row], aggIsNull[a][row]);
                }
            }
        }

        int numGroups = groupAggregators.size();
        List<SelectItem> items = stmt.items();
        List<Field> outFields = new ArrayList<>(items.size());
        List<FieldVector> outVectors = new ArrayList<>(items.size());
        try {
            for (int i = 0; i < items.size(); i++) {
                SelectItem item = items.get(i);
                FieldVector out = newFloat(item.outputName(), numGroups, p.float64, allocator);
                outVectors.add(out);
                if (gp.itemAggIndex[i] >= 0) {
                    int a = gp.itemAggIndex[i];
                    for (int g = 0; g < numGroups; g++) {
                        Aggregator agg = groupAggregators.get(g)[a];
                        if (agg.isNullResult()) {
                            out.setNull(g);
                        } else {
                            setNumeric(out, g, agg.result(), p.float64);
                        }
                    }
                } else {
                    int k = gp.itemKeyIndex[i];
                    for (int g = 0; g < numGroups; g++) {
                        double v = groupKeyValuesOut.get(g)[k];
                        if (Double.isNaN(v)) {
                            out.setNull(g);
                        } else {
                            setNumeric(out, g, v, p.float64);
                        }
                    }
                }
                outFields.add(out.getField());
            }
        } catch (RuntimeException | Error e) {
            for (FieldVector v : outVectors) {
                closeQuietly(v);
            }
            throw e;
        }
        return new VectorSchemaRoot(new Schema(outFields), outVectors, numGroups);
    }

    private static void setNumeric(FieldVector v, int index, double value, boolean float64) {
        if (float64) {
            ((Float8Vector) v).set(index, value);
        } else {
            ((Float4Vector) v).set(index, (float) value);
        }
    }

    /**
     * Bulk-evaluates one expression (or reads a passthrough column) over every
     * row of {@code root} into {@code outValues}/{@code outIsNull}.
     */
    private static void evaluateNumericColumn(
            ArrowExpressionEvaluator evaluator, String passthroughColumn, VectorSchemaRoot root,
            boolean float64, NullPolicy nullPolicy, double[] outValues, boolean[] outIsNull) {

        int rowCount = root.getRowCount();
        if (rowCount == 0) {
            return;
        }
        if (passthroughColumn != null) {
            FieldVector v = root.getVector(passthroughColumn);
            if (v == null) {
                throw new ArrowBindingException("Column '" + passthroughColumn + "' not found.");
            }
            readNumeric(v, float64, rowCount, outValues, outIsNull);
            return;
        }
        BufferAllocator allocator = allocatorOf(root);
        if (float64) {
            try (Float8Vector out = new Float8Vector("__parser_ng_sql_bulk__", allocator)) {
                out.allocateNew(rowCount);
                out.setValueCount(rowCount);
                evaluator.evaluate(root, out, nullPolicy);
                readNumeric(out, true, rowCount, outValues, outIsNull);
            }
        } else {
            try (Float4Vector out = new Float4Vector("__parser_ng_sql_bulk__", allocator)) {
                out.allocateNew(rowCount);
                out.setValueCount(rowCount);
                evaluator.evaluate(root, out, nullPolicy);
                readNumeric(out, false, rowCount, outValues, outIsNull);
            }
        }
    }

    private static void readNumeric(FieldVector v, boolean float64, int rowCount, double[] values, boolean[] isNull) {
        if (float64) {
            Float8Vector fv = (Float8Vector) v;
            for (int i = 0; i < rowCount; i++) {
                isNull[i] = fv.isNull(i);
                values[i] = isNull[i] ? 0.0 : fv.get(i);
            }
        } else {
            Float4Vector fv = (Float4Vector) v;
            for (int i = 0; i < rowCount; i++) {
                isNull[i] = fv.isNull(i);
                values[i] = isNull[i] ? 0.0 : fv.get(i);
            }
        }
    }

    private static VectorSchemaRoot applyHaving(CompiledPlan p, VectorSchemaRoot result, NullPolicy nullPolicy) {
        int[] selected;
        if (p.havingFused != null) {
            selected = selectFromEvaluator(p.havingFused, result, result.getRowCount(), p.float64, nullPolicy);
        } else {
            selected = fromMask(p.havingNode.evalMask(result, nullPolicy, p.float64));
        }
        return materializeRows(result, selected);
    }

    /**
     * Stable multi-key sort of {@code selected} (or of every row of {@code root}
     * when null) by {@link SelectStatement#orderBy()}. Keys are evaluated once,
     * in bulk, over {@code root}; returns row indices into {@code root} in
     * sorted order. Nulls sort last regardless of ASC/DESC (PostgreSQL default).
     */
    private int[] sortedSelection(CompiledPlan p, VectorSchemaRoot root, int[] selected, NullPolicy nullPolicy) {
        int rowCount = root.getRowCount();
        List<OrderItem> orderBy = stmt.orderBy();
        int numKeys = orderBy.size();
        double[][] values = new double[numKeys][];
        boolean[][] isNull = new boolean[numKeys][];
        boolean[] desc = new boolean[numKeys];
        for (int k = 0; k < numKeys; k++) {
            values[k] = new double[rowCount];
            isNull[k] = new boolean[rowCount];
            desc[k] = orderBy.get(k).descending();
            OrderByPlan op = p.orderByPlans.get(k);
            evaluateNumericColumn(op.evaluator, op.sourceColumnName, root, p.float64, nullPolicy,
                    values[k], isNull[k]);
        }

        int m = selected == null ? rowCount : selected.length;
        Integer[] order = new Integer[m];
        for (int i = 0; i < m; i++) {
            order[i] = selected == null ? i : selected[i];
        }
        Arrays.sort(order, (i1, i2) -> {
            for (int k = 0; k < numKeys; k++) {
                boolean n1 = isNull[k][i1];
                boolean n2 = isNull[k][i2];
                int cmp;
                if (n1 || n2) {
                    cmp = (n1 == n2) ? 0 : (n1 ? 1 : -1);
                } else {
                    cmp = Double.compare(values[k][i1], values[k][i2]);
                    if (desc[k]) {
                        cmp = -cmp;
                    }
                }
                if (cmp != 0) {
                    return cmp;
                }
            }
            return 0;
        });
        int[] indices = new int[m];
        for (int i = 0; i < m; i++) {
            indices[i] = order[i];
        }
        return indices;
    }

    // ---------------------------------------------------------------------
    // WHERE row selection
    // ---------------------------------------------------------------------

    /**
     * @return {@code null} meaning "every row, in original order" (no WHERE, or
     * every row passed), otherwise the surviving row indices in ascending order.
     */
    private static int[] selectRows(CompiledPlan p, VectorSchemaRoot root, NullPolicy nullPolicy) {
        if (p.fusedPredicate == null && p.maskPredicateRoot == null) {
            return null;
        }
        if (p.fusedPredicate != null) {
            return selectFromEvaluator(p.fusedPredicate, root, root.getRowCount(), p.float64, nullPolicy);
        }
        return fromMask(p.maskPredicateRoot.evalMask(root, nullPolicy, p.float64));
    }

    private static int[] selectFromEvaluator(
            ArrowExpressionEvaluator predicate, VectorSchemaRoot root, int rowCount,
            boolean float64, NullPolicy nullPolicy) {
        if (rowCount == 0) {
            return EMPTY;
        }
        long[] bits = new long[(rowCount + 63) >>> 6];
        BufferAllocator allocator = allocatorOf(root);
        try (FieldVector out = newFloat("__parser_ng_sql_predicate__", rowCount, float64, allocator)) {
            return evaluatePredicate(predicate, root, out, rowCount, float64, nullPolicy, bits);
        }
    }

    /**
     * Evaluates {@code predicate} into {@code out} and records passing rows in
     * {@code bits} (a zeroed {@code long[(rowCount+63)/64]}), then expands to
     * indices. isNull(i) is checked before get(i), unconditionally: a null
     * predicate result excludes the row (SQL three-valued logic).
     */
    private static int[] evaluatePredicate(
            ArrowExpressionEvaluator predicate, VectorSchemaRoot root, FieldVector out, int rowCount,
            boolean float64, NullPolicy nullPolicy, long[] bits) {
        predicate.evaluate(root, out, nullPolicy);
        int count = countAndMark(out, bits, rowCount, float64);
        return expandBits(bits, count, rowCount);
    }

    /** Evaluates a predicate output vector into a passing-row bitmask; returns the pass count. */
    private static int countAndMark(FieldVector out, long[] bits, int rowCount, boolean float64) {
        int count = 0;
        if (float64) {
            Float8Vector o = (Float8Vector) out;
            for (int i = 0; i < rowCount; i++) {
                if (!o.isNull(i) && o.get(i) != 0.0) {
                    bits[i >>> 6] |= 1L << i;
                    count++;
                }
            }
        } else {
            Float4Vector o = (Float4Vector) out;
            for (int i = 0; i < rowCount; i++) {
                if (!o.isNull(i) && o.get(i) != 0.0f) {
                    bits[i >>> 6] |= 1L << i;
                    count++;
                }
            }
        }
        return count;
    }

    private static int[] expandBits(long[] bits, int count, int rowCount) {
        if (count == rowCount) {
            return null; // everything passed: keep the zero-copy "all rows" paths
        }
        if (count == 0) {
            return EMPTY;
        }
        int[] out = new int[count];
        int idx = 0;
        for (int w = 0; w < bits.length; w++) {
            long b = bits[w];
            while (b != 0) {
                out[idx++] = (w << 6) + Long.numberOfTrailingZeros(b);
                b &= b - 1;
            }
        }
        return out;
    }

    /**
     * Like {@link #expandBits}, but writes into a caller-supplied, possibly
     * oversized {@code dest} (at least {@code count} entries) instead of
     * allocating -- the zero-alloc counterpart used by the reusable-output
     * path. Iterates only the words implied by {@code rowCount}, never
     * {@code bits.length}, so a headroom-grown {@code bits} array (see
     * {@link ReusableScratch#bits}) is safe to pass here.
     */
    private static void expandBitsInto(long[] bits, int[] dest, int rowCount) {
        int idx = 0;
        int words = (rowCount + 63) >>> 6;
        for (int w = 0; w < words; w++) {
            long b = bits[w];
            while (b != 0) {
                dest[idx++] = (w << 6) + Long.numberOfTrailingZeros(b);
                b &= b - 1;
            }
        }
    }

    private static int[] fromMask(boolean[] mask) {
        int count = 0;
        for (boolean b : mask) {
            if (b) {
                count++;
            }
        }
        if (count == mask.length) {
            return null;
        }
        if (count == 0) {
            return EMPTY;
        }
        int[] out = new int[count];
        int idx = 0;
        for (int i = 0; i < mask.length; i++) {
            if (mask[i]) {
                out[idx++] = i;
            }
        }
        return out;
    }

    // ---------------------------------------------------------------------
    // row materialization (used by grouped queries only)
    // ---------------------------------------------------------------------

    private static VectorSchemaRoot materializeRows(VectorSchemaRoot source, int[] selectedIndices) {
        if (selectedIndices == null) {
            return materializePrefix(source, source.getRowCount());
        }
        BufferAllocator allocator = allocatorOf(source);
        List<FieldVector> sourceVectors = source.getFieldVectors();
        List<FieldVector> outVectors = new ArrayList<>(sourceVectors.size());
        List<Field> outFields = new ArrayList<>(sourceVectors.size());
        int outRowCount = selectedIndices.length;
        try {
            for (FieldVector src : sourceVectors) {
                Field field = src.getField();
                FieldVector dst = field.createVector(allocator);
                outVectors.add(dst);
                if (outRowCount > 0) {
                    dst.setInitialCapacity(outRowCount);
                }
                dst.allocateNew();
                gatherInto(src, selectedIndices, dst, outRowCount);
                outFields.add(field);
            }
        } catch (RuntimeException | Error e) {
            for (FieldVector v : outVectors) {
                closeQuietly(v);
            }
            throw e;
        }
        return new VectorSchemaRoot(new Schema(outFields), outVectors, outRowCount);
    }

    private static VectorSchemaRoot materializePrefix(VectorSchemaRoot source, int length) {
        BufferAllocator allocator = allocatorOf(source);
        List<FieldVector> sourceVectors = source.getFieldVectors();
        List<FieldVector> outVectors = new ArrayList<>(sourceVectors.size());
        List<Field> outFields = new ArrayList<>(sourceVectors.size());
        try {
            for (FieldVector src : sourceVectors) {
                TransferPair tp = src.getTransferPair(allocator);
                tp.splitAndTransfer(0, length);
                FieldVector dst = (FieldVector) tp.getTo();
                outVectors.add(dst);
                outFields.add(dst.getField());
            }
        } catch (RuntimeException | Error e) {
            for (FieldVector v : outVectors) {
                closeQuietly(v);
            }
            throw e;
        }
        return new VectorSchemaRoot(new Schema(outFields), outVectors, length);
    }

    // =====================================================================
    // compiled plan model
    // =====================================================================

    /**
     * Immutable execution plan for one schema shape with a lock-free
     * reference-counted lifecycle: {@link #acquire()} before use,
     * {@link #release()} exactly once after (try/finally), {@link #retire()}
     * when superseded. Evaluators are closed exactly once, by whichever thread
     * observes "retired and no outstanding references". A retired plan can no
     * longer be acquired, so lookups fall through to the synchronized slow path.
     */
    private static final class CompiledPlan {

        private static final int RETIRED = 1 << 30;
        private static final int CLOSED = 1 << 31;
        private static final int REFS = RETIRED - 1;

        final long fingerprint;
        final boolean float64;
        final ArrowExpressionEvaluator fusedPredicate;
        final PredicateNode maskPredicateRoot;
        final List<ProjectionPlan> projections;
        final GroupPlan groupPlan;
        final ArrowExpressionEvaluator havingFused;
        final PredicateNode havingNode;
        final List<OrderByPlan> orderByPlans;
        /** Every distinct compiled evaluator (deduplicated by text); the sole set closed on retirement. */
        final List<ArrowExpressionEvaluator> compiledEvaluators;

        private final AtomicInteger state = new AtomicInteger();

        CompiledPlan(long fingerprint, boolean float64, ArrowExpressionEvaluator fusedPredicate,
                PredicateNode maskPredicateRoot, List<ProjectionPlan> projections, GroupPlan groupPlan,
                ArrowExpressionEvaluator havingFused, PredicateNode havingNode,
                List<OrderByPlan> orderByPlans,
                List<ArrowExpressionEvaluator> compiledEvaluators) {
            this.fingerprint = fingerprint;
            this.float64 = float64;
            this.fusedPredicate = fusedPredicate;
            this.maskPredicateRoot = maskPredicateRoot;
            this.projections = projections;
            this.groupPlan = groupPlan;
            this.havingFused = havingFused;
            this.havingNode = havingNode;
            this.orderByPlans = orderByPlans;
            this.compiledEvaluators = compiledEvaluators;
        }

        /** @return false if this plan is retired or closed; caller should re-resolve the plan. */
        boolean acquire() {
            for (;;) {
                int s = state.get();
                if ((s & (RETIRED | CLOSED)) != 0) {
                    return false;
                }
                if (state.compareAndSet(s, s + 1)) {
                    return true;
                }
            }
        }

        void release() {
            for (;;) {
                int s = state.get();
                int ns = s - 1;
                boolean close = (ns & RETIRED) != 0 && (ns & REFS) == 0 && (ns & CLOSED) == 0;
                if (close) {
                    ns |= CLOSED;
                }
                if (state.compareAndSet(s, ns)) {
                    if (close) {
                        closeEvaluators();
                    }
                    return;
                }
            }
        }

        void retire() {
            for (;;) {
                int s = state.get();
                if ((s & RETIRED) != 0) {
                    return;
                }
                int ns = s | RETIRED;
                boolean close = (ns & REFS) == 0 && (ns & CLOSED) == 0;
                if (close) {
                    ns |= CLOSED;
                }
                if (state.compareAndSet(s, ns)) {
                    if (close) {
                        closeEvaluators();
                    }
                    return;
                }
            }
        }

        private void closeEvaluators() {
            RuntimeException first = null;
            for (ArrowExpressionEvaluator e : compiledEvaluators) {
                try {
                    e.close();
                } catch (RuntimeException ex) {
                    if (first == null) {
                        first = ex;
                    }
                }
            }
            if (first != null) {
                throw first;
            }
        }
    }

    private static final class ProjectionPlan {

        final String outputName;
        final boolean passthrough;
        final String sourceColumnName;
        /**
         * Positional index of {@code sourceColumnName} within the root this plan
         * was compiled against, or -1 when not passthrough. See
         * {@link #resolvePassthrough} for why this is safe to reuse positionally
         * against any root sharing this plan's schema fingerprint.
         */
        final int sourceColumnIndex;
        final ArrowExpressionEvaluator evaluator;

        ProjectionPlan(String outputName, boolean passthrough, String sourceColumnName, int sourceColumnIndex,
                ArrowExpressionEvaluator evaluator) {
            this.outputName = outputName;
            this.passthrough = passthrough;
            this.sourceColumnName = sourceColumnName;
            this.sourceColumnIndex = sourceColumnIndex;
            this.evaluator = evaluator;
        }
    }

    private static final class GroupPlan {

        final List<KeyPlan> keys;
        final List<AggPlan> aggregates;
        final int[] itemKeyIndex;
        final int[] itemAggIndex;

        GroupPlan(List<KeyPlan> keys, List<AggPlan> aggregates, int[] itemKeyIndex, int[] itemAggIndex) {
            this.keys = keys;
            this.aggregates = aggregates;
            this.itemKeyIndex = itemKeyIndex;
            this.itemAggIndex = itemAggIndex;
        }
    }

    private static final class KeyPlan {

        final String exprText;
        final boolean passthrough;
        final String sourceColumnName;
        final ArrowExpressionEvaluator evaluator;

        KeyPlan(String exprText, boolean passthrough, String sourceColumnName, ArrowExpressionEvaluator evaluator) {
            this.exprText = exprText;
            this.passthrough = passthrough;
            this.sourceColumnName = sourceColumnName;
            this.evaluator = evaluator;
        }
    }

    private static final class AggPlan {

        final AggFunc func;
        final boolean star;
        final String passthroughColumn;
        final ArrowExpressionEvaluator evaluator;

        AggPlan(AggFunc func, boolean star, String passthroughColumn, ArrowExpressionEvaluator evaluator) {
            this.func = func;
            this.star = star;
            this.passthroughColumn = passthroughColumn;
            this.evaluator = evaluator;
        }
    }

    private static final class OrderByPlan {

        final boolean passthrough;
        final String sourceColumnName;
        final ArrowExpressionEvaluator evaluator;

        OrderByPlan(boolean passthrough, String sourceColumnName, ArrowExpressionEvaluator evaluator) {
            this.passthrough = passthrough;
            this.sourceColumnName = sourceColumnName;
            this.evaluator = evaluator;
        }
    }

    /**
     * Scratch state for {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}:
     * ONE full-batch scratch vector shared by every computed projection (they
     * run sequentially: evaluate then gather), one predicate output vector, one
     * predicate bitmask, and one row-selection index buffer. All grow with
     * headroom (never shrink, never reallocate on a same-or-smaller batch), so
     * steady-state execution against varying batch sizes performs no
     * allocations. Tied to a plan instance so recompilation can never leave
     * stale-typed scratch behind.
     */
    private static final class ReusableScratch {

        final CompiledPlan forPlan;
        private FieldVector projectionScratch;
        private FieldVector predicateScratch;
        private long[] bits;
        private int[] selectionIndices = EMPTY;
        /** Valid entry count in {@link #selectionIndices} after the most recent selection; the buffer may be larger. */
        int selectionCount;

        ReusableScratch(CompiledPlan forPlan) {
            this.forPlan = forPlan;
        }

        FieldVector projectionScratch(int rowCount, boolean float64, BufferAllocator allocator) {
            FieldVector v = projectionScratch;
            if (v == null || v.getValueCapacity() < rowCount) {
                int capacity = v == null ? rowCount : Math.max(rowCount, v.getValueCapacity() * 2);
                if (v != null) {
                    closeQuietly(v);
                }
                v = newFloat("__parser_ng_sql_reuse_scratch__", capacity, float64, allocator);
                projectionScratch = v;
            }
            v.setValueCount(rowCount);
            return v;
        }

        FieldVector predicateScratch(int rowCount, boolean float64, BufferAllocator allocator) {
            FieldVector v = predicateScratch;
            if (v == null || v.getValueCapacity() < rowCount) {
                int capacity = v == null ? rowCount : Math.max(rowCount, v.getValueCapacity() * 2);
                if (v != null) {
                    closeQuietly(v);
                }
                v = newFloat("__parser_ng_sql_reuse_predicate_scratch__", capacity, float64, allocator);
                predicateScratch = v;
            }
            v.setValueCount(rowCount);
            return v;
        }

        /** @return a bitmask of at least {@code words} longs, with those first {@code words} zeroed. */
        long[] bits(int words) {
            if (bits == null || bits.length < words) {
                bits = new long[Math.max(words, bits == null ? words : bits.length * 2)];
                // freshly allocated array is already zero-filled by the JVM
            } else {
                Arrays.fill(bits, 0, words, 0L);
            }
            return bits;
        }

        /** @return the reusable selection buffer, grown (with headroom) to hold at least {@code n} indices. */
        int[] selectionBuffer(int n) {
            if (selectionIndices.length < n) {
                selectionIndices = new int[Math.max(n, selectionIndices.length * 2)];
            }
            return selectionIndices;
        }

        void close() {
            if (projectionScratch != null) {
                closeQuietly(projectionScratch);
                projectionScratch = null;
            }
            if (predicateScratch != null) {
                closeQuietly(predicateScratch);
                predicateScratch = null;
            }
            bits = null;
            selectionIndices = EMPTY;
            selectionCount = 0;
        }
    }

    /** Hashable tuple of GROUP BY key values; null is represented as NaN (all nulls group together). */
    private static final class GroupKey {

        private final double[] values;

        GroupKey(double[] values) {
            this.values = values;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof GroupKey other) || values.length != other.values.length) {
                return false;
            }
            for (int i = 0; i < values.length; i++) {
                if (Double.doubleToLongBits(values[i]) != Double.doubleToLongBits(other.values[i])) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public int hashCode() {
            int h = 1;
            for (double v : values) {
                h = 31 * h + Double.hashCode(v);
            }
            return h;
        }
    }

    /** Per-group running accumulator; null inputs are skipped (standard SQL). */
    private interface Aggregator {

        void accumulate(double value, boolean isNull);

        boolean isNullResult();

        double result();
    }

    private static Aggregator newAggregator(AggFunc func) {
        return switch (func) {
            case SUM -> new SumAgg();
            case COUNT -> new CountAgg();
            case AVG -> new AvgAgg();
            case MIN -> new MinAgg();
            case MAX -> new MaxAgg();
        };
    }

    private static final class SumAgg implements Aggregator {

        private double sum;
        private boolean any;

        public void accumulate(double v, boolean isNull) {
            if (!isNull) {
                sum += v;
                any = true;
            }
        }

        public boolean isNullResult() {
            return !any;
        }

        public double result() {
            return sum;
        }
    }

    private static final class CountAgg implements Aggregator {

        private long count;

        public void accumulate(double v, boolean isNull) {
            if (!isNull) {
                count++;
            }
        }

        public boolean isNullResult() {
            return false;
        }

        public double result() {
            return (double) count;
        }
    }

    private static final class AvgAgg implements Aggregator {

        private double sum;
        private long count;

        public void accumulate(double v, boolean isNull) {
            if (!isNull) {
                sum += v;
                count++;
            }
        }

        public boolean isNullResult() {
            return count == 0;
        }

        public double result() {
            return sum / count;
        }
    }

    private static final class MinAgg implements Aggregator {

        private double min = Double.POSITIVE_INFINITY;
        private boolean any;

        public void accumulate(double v, boolean isNull) {
            if (!isNull && (!any || v < min)) {
                min = v;
                any = true;
            }
        }

        public boolean isNullResult() {
            return !any;
        }

        public double result() {
            return min;
        }
    }

    private static final class MaxAgg implements Aggregator {

        private double max = Double.NEGATIVE_INFINITY;
        private boolean any;

        public void accumulate(double v, boolean isNull) {
            if (!isNull && (!any || v > max)) {
                max = v;
                any = true;
            }
        }

        public boolean isNullResult() {
            return !any;
        }

        public double result() {
            return max;
        }
    }

    /** Node in the leaf-by-leaf mask fallback used for WHERE/HAVING clauses containing IS [NOT] NULL. */
    private interface PredicateNode {

        boolean[] evalMask(VectorSchemaRoot root, NullPolicy nullPolicy, boolean float64);
    }

    private static final class AndNode implements PredicateNode {

        private final PredicateNode left;
        private final PredicateNode right;

        AndNode(PredicateNode left, PredicateNode right) {
            this.left = left;
            this.right = right;
        }

        @Override
        public boolean[] evalMask(VectorSchemaRoot root, NullPolicy nullPolicy, boolean float64) {
            boolean[] l = left.evalMask(root, nullPolicy, float64);
            boolean[] r = right.evalMask(root, nullPolicy, float64);
            for (int i = 0; i < l.length; i++) { // reuse l; it is private to this call
                l[i] = l[i] && r[i];
            }
            return l;
        }
    }

    private static final class OrNode implements PredicateNode {

        private final PredicateNode left;
        private final PredicateNode right;

        OrNode(PredicateNode left, PredicateNode right) {
            this.left = left;
            this.right = right;
        }

        @Override
        public boolean[] evalMask(VectorSchemaRoot root, NullPolicy nullPolicy, boolean float64) {
            boolean[] l = left.evalMask(root, nullPolicy, float64);
            boolean[] r = right.evalMask(root, nullPolicy, float64);
            for (int i = 0; i < l.length; i++) {
                l[i] = l[i] || r[i];
            }
            return l;
        }
    }

    private static final class CompareLeafNode implements PredicateNode {

        private final ArrowExpressionEvaluator evaluator;

        CompareLeafNode(ArrowExpressionEvaluator evaluator) {
            this.evaluator = evaluator;
        }

        @Override
        public boolean[] evalMask(VectorSchemaRoot root, NullPolicy nullPolicy, boolean float64) {
            int rowCount = root.getRowCount();
            boolean[] mask = new boolean[rowCount];
            if (rowCount == 0) {
                return mask;
            }
            BufferAllocator allocator = allocatorOf(root);
            // A null leaf result is simply "false", regardless of NullPolicy.
            try (FieldVector out = newFloat("__parser_ng_sql_leaf__", rowCount, float64, allocator)) {
                evaluator.evaluate(root, out, nullPolicy);
                if (float64) {
                    Float8Vector o = (Float8Vector) out;
                    for (int i = 0; i < rowCount; i++) {
                        mask[i] = !o.isNull(i) && o.get(i) != 0.0;
                    }
                } else {
                    Float4Vector o = (Float4Vector) out;
                    for (int i = 0; i < rowCount; i++) {
                        mask[i] = !o.isNull(i) && o.get(i) != 0.0f;
                    }
                }
            }
            return mask;
        }
    }

    private static final class IsNullLeafNode implements PredicateNode {

        private final String bareColumn;
        private final ArrowExpressionEvaluator probeEvaluator;
        private final boolean negated;

        IsNullLeafNode(String bareColumn, ArrowExpressionEvaluator probeEvaluator, boolean negated) {
            this.bareColumn = bareColumn;
            this.probeEvaluator = probeEvaluator;
            this.negated = negated;
        }

        @Override
        public boolean[] evalMask(VectorSchemaRoot root, NullPolicy nullPolicy, boolean float64) {
            int rowCount = root.getRowCount();
            boolean[] mask = new boolean[rowCount];
            if (rowCount == 0) {
                return mask;
            }
            if (bareColumn != null) {
                FieldVector v = root.getVector(bareColumn);
                if (v == null) {
                    throw new ArrowBindingException("IS [NOT] NULL references unknown column '" + bareColumn + "'");
                }
                for (int i = 0; i < rowCount; i++) {
                    mask[i] = negated != v.isNull(i);
                }
                return mask;
            }
            BufferAllocator allocator = allocatorOf(root);
            // Probe is always PROPAGATE so the output validity bitmap reflects input nulls.
            try (FieldVector out = newFloat("__parser_ng_sql_isnull_probe__", rowCount, float64, allocator)) {
                probeEvaluator.evaluate(root, out, NullPolicy.PROPAGATE);
                for (int i = 0; i < rowCount; i++) {
                    mask[i] = negated != out.isNull(i);
                }
            }
            return mask;
        }
    }
}