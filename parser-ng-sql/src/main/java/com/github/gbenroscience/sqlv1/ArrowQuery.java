package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.arrow.tools.box.ArrowBindingException;
import com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend;
import com.github.gbenroscience.arrow.tools.box.ArrowExpressionEvaluator;
import com.github.gbenroscience.arrow.tools.box.ArrowExpressionEvaluators;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import com.github.gbenroscience.sqlv1.ast.AndExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExprs;
import com.github.gbenroscience.sqlv1.ast.IsNullExpr;
import com.github.gbenroscience.sqlv1.ast.OrExpr;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import com.github.gbenroscience.sqlv1.ast.WhereAliasResolver;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A compiled parser-ng-sql query: parse once, compile expressions once
 * (lazily, on first {@link #execute(VectorSchemaRoot)}), execute many times.
 *
 * <h2>Getting one</h2>
 * <pre>{@code
 * ArrowQuery query = ArrowQuery.compile(
 *         "SELECT sqrt(x*x + y*y) AS distance FROM data WHERE x > 10");
 * VectorSchemaRoot result = query.execute(root); // call as many times as you like
 * query.close(); // release compiled evaluators when done
 * }</pre>
 * For a single one-off execution, {@link ArrowSql#execute(VectorSchemaRoot, String)}
 * is a shorter equivalent that compiles, runs once, and closes for you.
 *
 * <h2>What "compiled" means here, precisely</h2>
 * {@link #compile(String)} only parses the SQL text into a
 * {@link SelectStatement} — cheap, and independent of any particular Arrow
 * schema. The actual {@code ArrowExpressionEvaluator}s (one per computed
 * projection, plus whatever the {@code WHERE} clause needs — see "Predicate
 * evaluation strategy" below) are compiled lazily on the <i>first</i> call to
 * {@link #execute(VectorSchemaRoot)}, because whether ParserNG compiles
 * against {@code float64} or {@code float32} kernels depends on the actual
 * column types of the root passed in (see {@link #isFloat64}). Once built,
 * that compiled plan is cached and reused on every subsequent
 * {@code execute} call whose root has the same column names/types (a
 * schema-fingerprint check); a root with a genuinely different schema
 * transparently triggers a one-time recompilation.
 *
 * <h2>{@code WHERE} referencing a {@code SELECT}-list alias</h2>
 * {@code WHERE} still evaluates against the <i>input</i> root's own
 * columns, before projection/aliasing happens — that part of the
 * evaluation strategy is unchanged. What is handled now is the common case
 * of naming a {@code SELECT}-list alias from {@code WHERE} instead of
 * repeating its expression:
 * <pre>{@code
 * SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM data WHERE magnitude > 60
 * }</pre>
 * Before compiling {@code WHERE}, every leaf reference to a name that (a)
 * matches a {@code SELECT}-list alias and (b) is <i>not</i> also a real
 * column of the root in hand (a real column always wins the name) is
 * textually expanded back into the expression that alias stands for — see
 * {@link WhereAliasResolver} for exactly how. This is resolved fresh
 * whenever the plan is (re)compiled, i.e. against the schema of whichever
 * root triggered compilation, consistent with everything else in "What
 * 'compiled' means here, precisely" above.
 *
 * <h2>Predicate evaluation strategy</h2>
 * A {@code WHERE} clause with no {@code IS [NOT] NULL} anywhere is rendered
 * (see {@link BoolExprs#renderFused(BoolExpr)}) into a single ParserNG
 * boolean expression string and compiled/evaluated as <b>one</b> fused
 * kernel — the fast path, and the common case.
 * <p>
 * {@code IS [NOT] NULL} has no ParserNG rendering at all (see
 * {@link IsNullExpr}'s javadoc: parser-ng-arrow's bulk evaluators never
 * expose a null-test operator through expression text). A {@code WHERE}
 * clause containing one is instead compiled leaf-by-leaf into a small tree
 * of independently-compiled sub-predicates, each producing a boolean mask
 * over all rows, combined with plain Java {@code &&}/{@code ||} on the
 * masks — {@code IS [NOT] NULL} leaves read Arrow validity bitmaps directly
 * (or, for a leaf whose target is a computed expression rather than a bare
 * column, via a probe evaluation forced to {@link NullPolicy#PROPAGATE} so
 * its output vector's own validity bitmap can be read). This is slower than
 * the fused path (each leaf is its own kernel dispatch instead of one fused
 * pass) but is the only way to give {@code IS [NOT] NULL} a real, correct
 * meaning at all; see this class's package for the tradeoff discussion.
 * <p>
 * Whatever the path, every predicate result is read back through
 * {@code isNull(i)} before {@code get(i)} — never the reverse — and a
 * {@code null} predicate result is treated as SQL three-valued logic
 * dictates: the row is excluded, exactly as an {@code UNKNOWN} {@code WHERE}
 * result would be. This holds unconditionally, for every {@link NullPolicy},
 * not only when nulls are actually expected — see "A note on
 * {@code NullPolicy}" below for why that unconditional guard is required
 * even on ordinary, entirely-non-null data.
 *
 * <h2>Expression deduplication</h2>
 * Within one {@link #buildPlan} call, every leaf that needs an
 * {@code ArrowExpressionEvaluator} — a predicate comparison/{@code BETWEEN}/
 * {@code IN} leaf, an {@code IS [NOT] NULL} probe, a computed projection —
 * is compiled through a single text-keyed cache local to that call, so two
 * leaves that render to identical ParserNG text share one compiled
 * evaluator instead of each getting their own. This is not a rare
 * coincidence: {@link WhereAliasResolver} routinely re-expands a
 * {@code SELECT}-list alias's expression text verbatim into {@code WHERE},
 * so the projection producing the alias and the predicate leaf consuming it
 * are frequently the exact same text. {@link CompiledPlan} holds the
 * deduplicated set directly (not derived from {@code fusedPredicate}/
 * {@code maskPredicateRoot}/{@code projections}, which may reference the
 * same instance more than once) and is the sole owner responsible for
 * closing each distinct evaluator exactly once.
 *
 * <h2>A note on {@code NullPolicy} (read before choosing {@link NullPolicy#IGNORE})</h2>
 * As verified against a real build/run of {@code parser-ng-arrow} 3.0.6
 * (see the module's {@code ArrowSqlDemo}), {@link NullPolicy#PROPAGATE} is
 * the only policy that currently produces correct output through this
 * evaluator: a computed value everywhere the inputs allow one, {@code null}
 * only where an input actually was. {@link NullPolicy#IGNORE} — despite its
 * name suggesting nulls are simply skipped over — currently comes back with
 * <b>every</b> output row marked invalid, even against a batch with no
 * nulls in it at all; this reproduces consistently and is not specific to
 * any one expression shape (arithmetic projections, {@code if(...)}, and
 * fused boolean predicates were all affected). Because of this,
 * {@link #compile(String)} defaults {@link #nullPolicy} to
 * {@link NullPolicy#PROPAGATE} rather than {@link NullPolicy#IGNORE}.
 * {@link #withNullPolicy(NullPolicy)} still accepts {@link NullPolicy#IGNORE}
 * for forward compatibility (a future {@code parser-ng-arrow} release may
 * fix it), but selecting it today will make every projection column and
 * every fused/leaf predicate come back {@code null} — this class will not
 * throw when that happens (see the unconditional {@code isNull} guard
 * above) but the result will not contain the data you asked for. Prefer the
 * default unless and until this is confirmed fixed upstream.
 *
 * <h2>{@code SELECT *}</h2>
 * Passes every column of the (filtered) input straight through, under its
 * original name — there is no aliasing syntax for {@code *} in the grammar
 * (see {@link SqlParser}).
 *
 * <h2>Column aliasing</h2>
 * An unaliased passthrough or computed column keeps ParserNG's own
 * convention of being named after its (trimmed) expression text; an
 * {@code AS} alias always wins. Every output column — passthrough or
 * computed — is materialized into a freshly allocated vector under its
 * final name; a passthrough column is not returned as a zero-copy alias of
 * its source vector, so that a source column referenced more than once
 * (e.g. {@code SELECT x, x AS y FROM t}) is never accidentally shared or
 * emptied by a buffer transfer. This trades a copy for straightforward
 * correctness; see the module README for the zero-copy optimization this
 * leaves on the table for a future version.
 *
 * <h2>Resource ownership</h2>
 * An {@code ArrowQuery} owns whatever {@code ArrowExpressionEvaluator}s it
 * has compiled and must be {@link #close()}d when no longer needed (a
 * try-with-resources block is the simplest way, as in the example above).
 * Every {@code VectorSchemaRoot} returned by {@link #execute} is a fresh,
 * independent batch that the <i>caller</i> owns and must close in turn —
 * this class never returns a view over, or a root that shares ownership
 * with, the root passed in.
 *
 * <h2>Thread-safety</h2>
 * Configuring a query ({@link #withBackend}/{@link #withNullPolicy}) and
 * calling {@link #execute} concurrently from multiple threads is not
 * supported. Calling {@link #execute} concurrently from multiple threads
 * once a query's configuration is no longer changing is only as safe as the
 * underlying {@code ArrowExpressionEvaluator}s' own backend — this class
 * adds no additional synchronization around {@code evaluate} calls beyond
 * what is needed to build/cache the compiled plan itself.
 *
 * @author GBEMIRO
 */
public final class ArrowQuery implements AutoCloseable {

    private final String sql;
    private final SelectStatement stmt;

    private volatile ArrowExecutionBackend backend = ArrowExecutionBackend.CPU_SIMD;
    private volatile NullPolicy nullPolicy = NullPolicy.PROPAGATE;

    private volatile CompiledPlan plan;
    private volatile VectorSchemaRoot lastRoot;

    private ArrowQuery(String sql, SelectStatement stmt) {
        this.sql = sql;
        this.stmt = stmt;
    }

    /**
     * Parses {@code sql} (per the grammar documented on {@link SqlParser})
     * into a reusable, not-yet-compiled query.
     *
     * @param sql
     * @return 
     * @throws SqlSyntaxException if {@code sql} does not conform to the
     * grammar
     */
    public static ArrowQuery compile(String sql) {
        SelectStatement stmt = SqlParser.parse(sql);
        return new ArrowQuery(sql, stmt);
    }

    /**
     * @return the parsed statement backing this query
     */
    public SelectStatement statement() {
        return stmt;
    }

    /**
     * Selects which parser-ng-arrow execution backend compiled expressions
     * target (default: {@link ArrowExecutionBackend#CPU_SIMD}). Changing
     * the backend after a plan has already been compiled invalidates and
     * recompiles it on the next {@link #execute}.
     * @param backend
     * @return 
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
     * Selects the {@link NullPolicy} used when evaluating both the
     * {@code WHERE} clause and every computed projection (default:
     * {@link NullPolicy#PROPAGATE} — see this class's "A note on
     * {@code NullPolicy}" for why {@link NullPolicy#IGNORE}, despite being
     * the more obviously-named default, is not one currently). Unlike
     * {@link #withBackend}, this never requires recompilation —
     * {@code NullPolicy} is a per-{@code evaluate} concern, not a
     * per-compiled-kernel one.
     * @param nullPolicy
     * @return 
     */
    public ArrowQuery withNullPolicy(NullPolicy nullPolicy) {
        if (nullPolicy == null) {
            throw new NullPointerException("nullPolicy must not be null");
        }
        this.nullPolicy = nullPolicy;
        return this;
    }

    /**
     * Executes this query against {@code root}, compiling (or, on a later
     * call against a root with an unchanged schema, reusing) the underlying
     * ParserNG expressions as needed.
     *
     * @param root
     * @return a fresh, independently-owned result batch: the projected
     * columns of {@code root}, filtered by the {@code WHERE} clause if any
     * @throws ArrowSqlException if compiling the query's expressions
     * against {@code root}'s schema fails
     * @throws ArrowBindingException if evaluation fails at runtime (a
     * required column missing from {@code root}, a row-count mismatch,
     * etc.) -- thrown directly by parser-ng-arrow's own evaluators
     */
    public VectorSchemaRoot execute(VectorSchemaRoot root) {
        if (root == null) {
            throw new NullPointerException("root must not be null");
        }
        CompiledPlan p;
        try {
            p = ensurePlan(root);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new ArrowSqlException(
                    "Failed to compile query \"" + sql + "\" against the given schema: " + t.getMessage(), t);
        }
        return runPlan(p, root);
    }

    /**
     * Releases every {@code ArrowExpressionEvaluator} this query has
     * compiled. Safe to call more than once; safe to call even if
     * {@link #execute} was never called.
     */
    @Override
    public synchronized void close() {
        invalidatePlan();
    }

    private synchronized void invalidatePlan() {
        lastRoot = null;
        if (plan != null) {
            plan.close();
            plan = null;
        }
    }

    // =====================================================================
    // plan compilation (lazy, cached, schema-fingerprinted)
    // =====================================================================

    private CompiledPlan ensurePlan(VectorSchemaRoot root) throws Throwable {
        // Lock-free fast path: the overwhelmingly common case is the exact
        // same VectorSchemaRoot instance being re-executed in a loop (a
        // cached materialized batch, a benchmark, a retry) - reference
        // equality against the last-seen root skips the fingerprint
        // computation AND the synchronized section entirely, with no
        // allocation and no monitor acquisition. Safe under races: a stale
        // read here just falls through to the synchronized slow path below,
        // it can never return a plan that doesn't match some earlier state.
        VectorSchemaRoot seenRoot = lastRoot;
        CompiledPlan cached = plan;
        if (seenRoot == root && cached != null) {
            return cached;
        }
        return ensurePlanSlow(root);
    }

    private synchronized CompiledPlan ensurePlanSlow(VectorSchemaRoot root) throws Throwable {
        long fingerprint = fingerprintOf(root);
        CompiledPlan existing = plan;
        if (existing != null && existing.fingerprint == fingerprint) {
            lastRoot = root;
            return existing;
        }
        CompiledPlan fresh = buildPlan(root, fingerprint);
        lastRoot = root;
        if (existing != null) {
            existing.close();
        }
        plan = fresh;
        return fresh;
    }

    private CompiledPlan buildPlan(VectorSchemaRoot root, long fingerprint) throws Throwable {
        boolean float64 = isFloat64(root);

        // Shared within this one buildPlan() call: any two leaves (a
        // predicate comparison, an IS NULL probe, a computed projection)
        // that render to the exact same ParserNG text compile to, and
        // share, a single ArrowExpressionEvaluator instead of one each.
        // This is common -- e.g. WhereAliasResolver routinely re-expands a
        // SELECT-list alias's expression text verbatim into WHERE, so the
        // projection and the predicate leaf it feeds are frequently
        // identical text. CompiledPlan.close() owns closing every distinct
        // evaluator exactly once via this map's values(); no other site
        // closes one directly. See "Expression deduplication" in this
        // class's javadoc.
        Map<String, ArrowExpressionEvaluator> compiledCache = new LinkedHashMap<>();

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
                String text = BoolExprs.renderFused(where);
                fusedPredicate = compileCached(text, backend, float64, compiledCache);
            }
        }

        List<ProjectionPlan> projections = new ArrayList<>();
        if (!stmt.selectAll()) {
            for (SelectItem item : stmt.items()) {
                String expr = item.exprText();
                String outputName = item.outputName();
                if (root.getVector(expr) != null) {
                    projections.add(new ProjectionPlan(outputName, true, expr, null));
                } else {
                    ArrowExpressionEvaluator eval = compileCached(expr, backend, float64, compiledCache);
                    projections.add(new ProjectionPlan(outputName, false, null, eval));
                }
            }
        }

        return new CompiledPlan(fingerprint, float64, fusedPredicate, maskPredicateRoot, projections,
                List.copyOf(compiledCache.values()));
    }

    /**
     * Looks up an already-compiled evaluator for {@code text} in
     * {@code cache} (the current {@link #buildPlan} call's dedup map),
     * compiling and caching a new one on a miss. See "Expression
     * deduplication" in this class's javadoc.
     */
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

    /**
     * Builds the alias-&gt;expression-text map {@link WhereAliasResolver}
     * needs: every {@code SELECT}-list output name that is <i>not</i> also
     * a real column of {@code root} (a real column always wins {@code
     * WHERE}-clause scoping over a same-named alias — see this class's
     * "{@code WHERE} referencing a {@code SELECT}-list alias"), mapped to
     * the raw expression text it stands for. Empty for {@code SELECT *}
     * (there are no aliases to speak of) or a {@code WHERE}-less query
     * (nothing to resolve against) — checked by the caller, but harmless to
     * call regardless.
     */
    private Map<String, String> collectWhereAliasBindings(VectorSchemaRoot root) {
        if (stmt.selectAll()) {
            return Map.of();
        }
        Map<String, String> bindings = new LinkedHashMap<>();
        for (SelectItem item : stmt.items()) {
            String outputName = item.outputName();
            if (root.getVector(outputName) != null) {
                continue;
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
            ArrowExpressionEvaluator probe = compileCached(target, backend, float64, compiledCache);
            return new IsNullLeafNode(null, probe, n.negated());
        }
        // ComparisonExpr / BetweenExpr / InExpr: every other leaf type
        // renders to a single self-contained ParserNG boolean fragment.
        String text = BoolExprs.renderLeaf(expr);
        ArrowExpressionEvaluator eval = compileCached(text, backend, float64, compiledCache);
        return new CompareLeafNode(eval);
    }

    /**
     * Allocation-free replacement for the original {@code List<String>}
     * fingerprint: combines each column's name hash with a cheap type
     * discriminant ({@link FieldVector#getMinorType()}'s ordinal - an enum
     * accessor, not a string) into a single running {@code long}, the same
     * way {@link java.util.List#hashCode()} combines its elements. No
     * {@code String} concatenation, no {@code ArrayList}, no per-call heap
     * allocation at all on the (overwhelmingly common) cache-hit path.
     *
     * <p>Collision risk is the standard hash-fingerprint trade-off: two
     * genuinely different schemas could theoretically collide onto the same
     * {@code long} and be treated as equal. For a 64-bit fingerprint over
     * realistically-sized schemas this is astronomically unlikely (on the
     * order of 1 in 2^64 for any specific pair, birthday-bound for a cache
     * holding many distinct schemas simultaneously) - a level of risk this
     * class already implicitly accepted elsewhere (e.g. {@code Field}'s own
     * {@code equals}/{@code hashCode} contract), and one every schema/plan
     * cache of this shape (Spark, Arrow's own dictionary encoding, etc.)
     * takes for the same reason: the alternative is paying full structural
     * comparison cost on every single call, which defeats the point of
     * caching in the first place.
     */
    private static long fingerprintOf(VectorSchemaRoot root) {
        List<FieldVector> vectors = root.getFieldVectors();
        long h = 1125899906842597L; // large odd seed, same technique as Arrays.hashCode
        for (FieldVector v : vectors) {
            h = 31 * h + v.getName().hashCode();
            h = 31 * h + v.getMinorType().ordinal();
        }
        return h;
    }

    /**
     * @return {@code true} iff every column of {@code root} is a
     * {@code Float8Vector} -- parser-ng-arrow's own convention for when to
     * compile/evaluate at {@code double} precision rather than
     * {@code float}.
     */
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

    private static void closeQuietly(FieldVector v) {
        try {
            v.close();
        } catch (RuntimeException ignored) {
            // best-effort cleanup while unwinding a different failure
        }
    }

    // =====================================================================
    // per-call execution: select rows -> materialize -> project/alias
    // =====================================================================

    private VectorSchemaRoot runPlan(CompiledPlan p, VectorSchemaRoot root) {
        int[] selected = selectRows(p, root);
        VectorSchemaRoot filtered = materializeRows(root, selected);
        if (stmt.selectAll()) {
            return filtered;
        }
        try {
            return buildProjection(p, filtered);
        } catch (RuntimeException | Error e) {
            for (FieldVector v : filtered.getFieldVectors()) {
                closeQuietly(v);
            }
            throw e;
        }
    }

    private int[] selectRows(CompiledPlan p, VectorSchemaRoot root) {
        int rowCount = root.getRowCount();
        if (p.fusedPredicate == null && p.maskPredicateRoot == null) {
            int[] all = new int[rowCount];
            for (int i = 0; i < rowCount; i++) {
                all[i] = i;
            }
            return all;
        }
        if (p.fusedPredicate != null) {
            return selectFromEvaluator(p.fusedPredicate, root, rowCount, p.float64, nullPolicy);
        }
        boolean[] mask = p.maskPredicateRoot.evalMask(root, nullPolicy, p.float64);
        int count = 0;
        for (boolean b : mask) {
            if (b) {
                count++;
            }
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

    private static int[] selectFromEvaluator(
            ArrowExpressionEvaluator predicate, VectorSchemaRoot root, int rowCount,
            boolean float64, NullPolicy nullPolicy) {

        if (rowCount == 0) {
            return new int[0];
        }
        BufferAllocator allocator = allocatorOf(root);
        int[] buffer = new int[Math.max(16, rowCount / 4)];
        int count = 0;

        // isNull(i) is checked unconditionally, for every NullPolicy, before
        // ever calling get(i) — never the reverse. A null predicate result
        // excludes the row (SQL three-valued logic for WHERE), and this
        // guard must not be gated on NullPolicy.PROPAGATE: see this class's
        // "A note on NullPolicy" for why NullPolicy.IGNORE can currently
        // come back fully null even against entirely-non-null input.
        if (float64) {
            try (Float8Vector out = new Float8Vector("__parser_ng_sql_predicate__", allocator)) {
                out.allocateNew(rowCount);
                out.setValueCount(rowCount);
                predicate.evaluate(root, out, nullPolicy);
                for (int i = 0; i < rowCount; i++) {
                    if (out.isNull(i)) {
                        continue;
                    }
                    if (out.get(i) != 0.0) {
                        if (count == buffer.length) {
                            buffer = Arrays.copyOf(buffer, buffer.length * 2);
                        }
                        buffer[count++] = i;
                    }
                }
            }
        } else {
            try (Float4Vector out = new Float4Vector("__parser_ng_sql_predicate__", allocator)) {
                out.allocateNew(rowCount);
                out.setValueCount(rowCount);
                predicate.evaluate(root, out, nullPolicy);
                for (int i = 0; i < rowCount; i++) {
                    if (out.isNull(i)) {
                        continue;
                    }
                    if (out.get(i) != 0.0f) {
                        if (count == buffer.length) {
                            buffer = Arrays.copyOf(buffer, buffer.length * 2);
                        }
                        buffer[count++] = i;
                    }
                }
            }
        }
        return Arrays.copyOf(buffer, count);
    }

    private static VectorSchemaRoot materializeRows(VectorSchemaRoot source, int[] selectedIndices) {
        BufferAllocator allocator = allocatorOf(source);
        List<FieldVector> sourceVectors = source.getFieldVectors();
        List<FieldVector> outVectors = new ArrayList<>(sourceVectors.size());
        List<Field> outFields = new ArrayList<>(sourceVectors.size());
        int outRowCount = selectedIndices.length;
        try {
            for (FieldVector src : sourceVectors) {
                Field field = src.getField();
                FieldVector dst = field.createVector(allocator);
                if (outRowCount > 0) {
                    dst.setInitialCapacity(outRowCount);
                }
                dst.allocateNew();
                for (int i = 0; i < outRowCount; i++) {
                    dst.copyFromSafe(selectedIndices[i], i, src);
                }
                dst.setValueCount(outRowCount);
                outVectors.add(dst);
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

    private VectorSchemaRoot buildProjection(CompiledPlan p, VectorSchemaRoot filtered) {
        BufferAllocator allocator = allocatorOf(filtered);
        int rowCount = filtered.getRowCount();
        List<Field> outFields = new ArrayList<>(p.projections.size());
        List<FieldVector> outVectors = new ArrayList<>(p.projections.size());
        // Tracks output vectors that are just a reused reference into
        // `filtered`'s own vectors (no rename, so no copy was needed) -
        // these must NOT be closed by the cleanup loop below, since the
        // returned VectorSchemaRoot still owns them.
        Set<FieldVector> reused = java.util.Collections.newSetFromMap(new IdentityHashMap<>());

        try {
            for (ProjectionPlan proj : p.projections) {
                FieldVector out;
                if (proj.passthrough) {
                    FieldVector src = filtered.getVector(proj.sourceColumnName);
                    if (src == null) {
                        throw new ArrowBindingException(
                                "Column '" + proj.sourceColumnName + "' not found while projecting.");
                    }
                    if (proj.outputName.equals(src.getField().getName())) {
                        // No rename: reuse the vector filtered() already
                        // built for us instead of allocating a fresh one
                        // and copying every row through copyFromSafe - the
                        // exact optimization implA's own projection loop
                        // already relies on for this (overwhelmingly
                        // common) case.
                        out = src;
                        reused.add(src);
                    } else {
                        Field outField = new Field(proj.outputName, src.getField().getFieldType(), src.getField().getChildren());
                        out = outField.createVector(allocator);
                        if (rowCount > 0) {
                            out.setInitialCapacity(rowCount);
                        }
                        out.allocateNew();
                        for (int i = 0; i < rowCount; i++) {
                            out.copyFromSafe(i, i, src);
                        }
                        out.setValueCount(rowCount);
                    }
                } else if (p.float64) {
                    Float8Vector out8 = new Float8Vector(proj.outputName, allocator);
                    out8.allocateNew(rowCount);
                    out8.setValueCount(rowCount);
                    proj.evaluator.evaluate(filtered, out8, nullPolicy);
                    out = out8;
                } else {
                    Float4Vector out4 = new Float4Vector(proj.outputName, allocator);
                    out4.allocateNew(rowCount);
                    out4.setValueCount(rowCount);
                    proj.evaluator.evaluate(filtered, out4, nullPolicy);
                    out = out4;
                }
                outFields.add(out.getField());
                outVectors.add(out);
            }
        } catch (RuntimeException | Error e) {
            for (FieldVector v : outVectors) {
                if (!reused.contains(v)) {
                    closeQuietly(v);
                }
            }
            for (FieldVector v : filtered.getFieldVectors()) {
                closeQuietly(v);
            }
            throw e;
        }

        for (FieldVector v : filtered.getFieldVectors()) {
            if (!reused.contains(v)) {
                closeQuietly(v);
            }
        }

        return new VectorSchemaRoot(new Schema(outFields), outVectors, rowCount);
    }

    // =====================================================================
    // compiled plan model
    // =====================================================================

    private static final class CompiledPlan {

        final long fingerprint;
        final boolean float64;
        final ArrowExpressionEvaluator fusedPredicate;
        final PredicateNode maskPredicateRoot;
        final List<ProjectionPlan> projections;
        // Every distinct compiled evaluator this plan owns, deduplicated by
        // rendered ParserNG text (see buildPlan's compiledCache) -- the
        // *sole* owner responsible for closing each one exactly once, even
        // though the very same instance may also be referenced by
        // fusedPredicate, a leaf inside maskPredicateRoot, and/or one or
        // more entries of projections. See "Expression deduplication" in
        // this class's javadoc.
        final List<ArrowExpressionEvaluator> compiledEvaluators;

        CompiledPlan(long fingerprint, boolean float64, ArrowExpressionEvaluator fusedPredicate,
                PredicateNode maskPredicateRoot, List<ProjectionPlan> projections,
                List<ArrowExpressionEvaluator> compiledEvaluators) {
            this.fingerprint = fingerprint;
            this.float64 = float64;
            this.fusedPredicate = fusedPredicate;
            this.maskPredicateRoot = maskPredicateRoot;
            this.projections = projections;
            this.compiledEvaluators = compiledEvaluators;
        }

        void close() {
            for (ArrowExpressionEvaluator e : compiledEvaluators) {
                e.close();
            }
        }
    }

    private static final class ProjectionPlan {

        final String outputName;
        final boolean passthrough;
        final String sourceColumnName;
        final ArrowExpressionEvaluator evaluator;

        ProjectionPlan(String outputName, boolean passthrough, String sourceColumnName, ArrowExpressionEvaluator evaluator) {
            this.outputName = outputName;
            this.passthrough = passthrough;
            this.sourceColumnName = sourceColumnName;
            this.evaluator = evaluator;
        }
    }

    /**
     * A compiled node in the leaf-by-leaf mask-evaluation fallback used for
     * {@code WHERE} clauses containing {@code IS [NOT] NULL} -- see this
     * class's "Predicate evaluation strategy".
     */
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
            boolean[] out = new boolean[l.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = l[i] && r[i];
            }
            return out;
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
            boolean[] out = new boolean[l.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = l[i] || r[i];
            }
            return out;
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
            // Same unconditional isNull-before-get guard as
            // selectFromEvaluator — a null leaf result is simply "false" for
            // this leaf, regardless of NullPolicy; see "A note on
            // NullPolicy" on this class.
            if (float64) {
                try (Float8Vector out = new Float8Vector("__parser_ng_sql_leaf__", allocator)) {
                    out.allocateNew(rowCount);
                    out.setValueCount(rowCount);
                    evaluator.evaluate(root, out, nullPolicy);
                    for (int i = 0; i < rowCount; i++) {
                        mask[i] = !out.isNull(i) && out.get(i) != 0.0;
                    }
                }
            } else {
                try (Float4Vector out = new Float4Vector("__parser_ng_sql_leaf__", allocator)) {
                    out.allocateNew(rowCount);
                    out.setValueCount(rowCount);
                    evaluator.evaluate(root, out, nullPolicy);
                    for (int i = 0; i < rowCount; i++) {
                        mask[i] = !out.isNull(i) && out.get(i) != 0.0f;
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
                    boolean isNull = v.isNull(i);
                    mask[i] = negated != isNull;
                }
                return mask;
            }
            BufferAllocator allocator = allocatorOf(root);
            if (float64) {
                try (Float8Vector out = new Float8Vector("__parser_ng_sql_isnull_probe__", allocator)) {
                    out.allocateNew(rowCount);
                    out.setValueCount(rowCount);
                    probeEvaluator.evaluate(root, out, NullPolicy.PROPAGATE);
                    for (int i = 0; i < rowCount; i++) {
                        boolean isNull = out.isNull(i);
                        mask[i] = negated != isNull;
                    }
                }
            } else {
                try (Float4Vector out = new Float4Vector("__parser_ng_sql_isnull_probe__", allocator)) {
                    out.allocateNew(rowCount);
                    out.setValueCount(rowCount);
                    probeEvaluator.evaluate(root, out, NullPolicy.PROPAGATE);
                    for (int i = 0; i < rowCount; i++) {
                        boolean isNull = out.isNull(i);
                        mask[i] = negated != isNull;
                    }
                }
            }
            return mask;
        }
    }
}