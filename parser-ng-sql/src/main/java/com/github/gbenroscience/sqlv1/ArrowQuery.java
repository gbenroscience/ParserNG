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
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
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
 * {@code AS} alias always wins. A passthrough column whose output name
 * matches its source column's own name is returned as a zero-copy
 * reference to that source vector rather than a freshly allocated,
 * row-by-row copy — {@link #buildProjection} tracks which output vectors
 * are such reused references (in an identity set, since two different
 * {@code FieldVector} instances can be {@code equals}-equal) so its cleanup
 * path never closes one still owned by the result it's returning. A
 * passthrough column that *is* renamed (e.g. {@code SELECT x AS y FROM t})
 * still gets a real, freshly allocated copy, since the same source column
 * referenced twice under different names (e.g. {@code SELECT x, x AS y
 * FROM t}) must never share one mutable vector between two independent
 * output columns.
 *
 * <h2>{@code GROUP BY} / {@code HAVING} / {@code ORDER BY} / {@code LIMIT}</h2>
 * Aggregation ({@code SUM}/{@code COUNT}/{@code AVG}/{@code MIN}/{@code MAX})
 * is evaluated entirely in Java, not by ParserNG (see {@link AggFunc}'s
 * javadoc) — {@link #buildGroupedResult} bulk-evaluates every {@code GROUP
 * BY} key and every aggregate's argument once each across all filtered
 * rows (see {@link #evaluateNumericColumn}), then buckets rows into groups
 * in a single pass. A non-aggregate {@code SELECT} item in a grouped query
 * must match a {@code GROUP BY} key expression exactly — see
 * {@code SelectStatement}'s javadoc for this (standard-SQL) restriction,
 * enforced once in {@link #buildGroupPlan} rather than per row.
 * <p>
 * {@code HAVING} only ever applies to a grouped query and always operates
 * on the final grouped result, reusing the exact same fused-evaluator /
 * leaf-by-leaf {@code IS NULL} mask machinery {@code WHERE} uses (see
 * {@link #buildPredicateNode}) — compiled once in {@link #buildPlan} against
 * a throwaway zero-row "phantom" root carrying only the grouped result's
 * eventual column names, since that column set (the {@code SELECT} list's
 * own output names) is fixed by the parsed statement alone, independent of
 * any particular input root.
 * <p>
 * {@code ORDER BY} sorts at a different pipeline stage depending on query
 * shape (see {@link #runPlan}): a <b>grouped</b> query's {@code ORDER BY}
 * runs after projection, against that same final grouped result, exactly
 * like {@code HAVING}. A <b>non-grouped</b> query's {@code ORDER BY} instead
 * runs <i>before</i> projection, against the {@code WHERE}-filtered root —
 * so it may reference a real input column that was never in the
 * {@code SELECT} list at all (e.g. {@code SELECT x FROM t ORDER BY y}), or
 * a {@code SELECT}-list alias, resolved back to the input-column-based
 * expression it stands for via the same {@link WhereAliasResolver}
 * machinery {@code WHERE} itself uses (see
 * {@link WhereAliasResolver#substituteText}) — matching ORDER BY's usual
 * SQL semantics (it conceptually sorts the {@code FROM}/{@code WHERE} row
 * set, only afterward narrowed by the {@code SELECT} list), not a
 * restriction to columns that happen to survive projection.
 * <p>
 * {@code LIMIT} always applies last, after any {@code ORDER BY}, on
 * whatever the final result is at that point (grouped-and-{@code HAVING}-filtered,
 * or plain projected).
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
        // predicate comparison, an IS NULL probe, a computed projection,
        // a GROUP BY key, an aggregate's argument, a HAVING/ORDER BY
        // expression) that render to the exact same ParserNG text compile
        // to, and share, a single ArrowExpressionEvaluator instead of one
        // each. This is common -- e.g. WhereAliasResolver routinely
        // re-expands a SELECT-list alias's expression text verbatim into
        // WHERE, so the projection and the predicate leaf it feeds are
        // frequently identical text. CompiledPlan.close() owns closing
        // every distinct evaluator exactly once via this map's values(); no
        // other site closes one directly. See "Expression deduplication"
        // in this class's javadoc.
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
        GroupPlan groupPlan = null;
        if (stmt.isGrouped()) {
            groupPlan = buildGroupPlan(root, backend, float64, compiledCache);
        } else if (!stmt.selectAll()) {
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

        // HAVING always operates on the final grouped/projected result (see
        // buildGroupPlan's javadoc for why that column set is always
        // statically known). ORDER BY differs by query shape: a grouped
        // query's ORDER BY operates on that same final grouped result
        // (compiled unconditionally, same as HAVING, for the same reason);
        // a non-grouped query's ORDER BY instead operates on the
        // WHERE-filtered root *before* projection narrows its columns --
        // exactly like WHERE itself, an ORDER BY key may reference a real
        // input column that never appears in the SELECT list at all, or a
        // SELECT-list alias (resolved via the same WhereAliasResolver
        // machinery WHERE uses), and gets the same bare-column passthrough
        // optimization projections/GROUP BY keys already get. See
        // ArrowQuery's "ORDER BY and LIMIT" javadoc.
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
     * Builds the {@link GroupPlan} for a {@link SelectStatement#isGrouped()}
     * query: one {@link KeyPlan} per {@link SelectStatement#groupBy()} entry
     * (with the same bare-column passthrough optimization projections
     * already get), one {@link AggPlan} per aggregate {@link SelectItem},
     * and a binding from each {@code SELECT} item's position back to
     * whichever of those two lists produces its value. Validates, once
     * here rather than per-execute, that every non-aggregate item matches a
     * {@code GROUP BY} key exactly -- see {@code SelectStatement}'s
     * javadoc, "{@code GROUP BY} / aggregates — a deliberately strict
     * subset".
     */
    private GroupPlan buildGroupPlan(
            VectorSchemaRoot root, ArrowExecutionBackend backend, boolean float64,
            Map<String, ArrowExpressionEvaluator> compiledCache) throws Throwable {

        List<KeyPlan> keys = new ArrayList<>(stmt.groupBy().size());
        for (String keyExpr : stmt.groupBy()) {
            if (root.getVector(keyExpr) != null) {
                keys.add(new KeyPlan(keyExpr, true, keyExpr, null));
            } else {
                ArrowExpressionEvaluator eval = compileCached(keyExpr, backend, float64, compiledCache);
                keys.add(new KeyPlan(keyExpr, false, null, eval));
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

    /**
     * Builds a zero-row {@code VectorSchemaRoot} whose schema is exactly
     * the final grouped result's column set (one column per
     * {@link SelectStatement#items()}, named by {@link SelectItem#outputName()})
     * -- used only to let {@link #buildPredicateNode} run its bare-column
     * {@code IS [NOT] NULL} detection (a pure schema check; it never reads
     * row data) against a {@code HAVING} clause during {@link #buildPlan},
     * before any real grouped result exists. Always closed immediately
     * after that one use via {@link #closePhantomRoot}; never retained,
     * never returned to a caller.
     */
    private VectorSchemaRoot phantomResultRoot(boolean float64) {
        BufferAllocator allocator = new org.apache.arrow.memory.RootAllocator(1024);
        List<Field> fields = new ArrayList<>(stmt.items().size());
        List<FieldVector> vectors = new ArrayList<>(stmt.items().size());
        for (SelectItem item : stmt.items()) {
            Field field = new Field(item.outputName(),
                    FieldType.nullable(float64
                            ? new org.apache.arrow.vector.types.pojo.ArrowType.FloatingPoint(
                                    org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE)
                            : new org.apache.arrow.vector.types.pojo.ArrowType.FloatingPoint(
                                    org.apache.arrow.vector.types.FloatingPointPrecision.SINGLE)),
                    null);
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
        for (FieldVector v : phantom.getFieldVectors()) {
            closeQuietly(v);
        }
        if (allocator != null) {
            try {
                allocator.close();
            } catch (RuntimeException ignored) {
                // best-effort cleanup of the throwaway allocator created
                // solely for this phantom root -- see phantomResultRoot
            }
        }
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

        // Non-grouped queries sort *before* projection: an ORDER BY key may
        // name a real input column that never appears in the SELECT list,
        // or a SELECT-list alias resolved back to its input-column-based
        // expression (see buildPlan's ORDER BY comment) -- both only
        // resolve correctly against the pre-projection root.
        if (p.groupPlan == null && !p.orderByPlans.isEmpty()) {
            VectorSchemaRoot sortedFiltered = applyOrderBy(p, filtered);
            for (FieldVector v : filtered.getFieldVectors()) {
                closeQuietly(v);
            }
            filtered = sortedFiltered;
        }

        VectorSchemaRoot result;
        if (p.groupPlan != null) {
            try {
                result = buildGroupedResult(p, filtered);
            } catch (RuntimeException | Error e) {
                for (FieldVector v : filtered.getFieldVectors()) {
                    closeQuietly(v);
                }
                throw e;
            }
            for (FieldVector v : filtered.getFieldVectors()) {
                closeQuietly(v);
            }
        } else if (stmt.selectAll()) {
            result = filtered;
        } else {
            try {
                result = buildProjection(p, filtered);
            } catch (RuntimeException | Error e) {
                for (FieldVector v : filtered.getFieldVectors()) {
                    closeQuietly(v);
                }
                throw e;
            }
        }

        // HAVING only ever applies to a grouped query (see
        // SelectStatement's validation) and always operates on the final
        // grouped result. A grouped query's ORDER BY, unlike a
        // non-grouped query's (handled above, before projection), runs
        // *after* projection too, against that same final grouped result
        // -- see buildPlan's ORDER BY comment for why grouped and
        // non-grouped queries sort at different pipeline stages.
        if (p.havingFused != null || p.havingNode != null) {
            VectorSchemaRoot havingFiltered = applyHaving(p, result);
            for (FieldVector v : result.getFieldVectors()) {
                closeQuietly(v);
            }
            result = havingFiltered;
        }
        if (p.groupPlan != null && !p.orderByPlans.isEmpty()) {
            VectorSchemaRoot sorted = applyOrderBy(p, result);
            for (FieldVector v : result.getFieldVectors()) {
                closeQuietly(v);
            }
            result = sorted;
        }
        if (stmt.limit() != null && stmt.limit() < result.getRowCount()) {
            VectorSchemaRoot limited = materializeRows(result,
                    java.util.stream.IntStream.range(0, stmt.limit()).toArray());
            for (FieldVector v : result.getFieldVectors()) {
                closeQuietly(v);
            }
            result = limited;
        }
        return result;
    }

    /**
     * Turns a {@code WHERE}-filtered row set into one row per distinct
     * {@code GROUP BY} key: evaluates every key expression and every
     * aggregate's argument once, in bulk, over all of {@code filtered}'s
     * rows (never per-row calls into ParserNG -- see
     * {@link #evaluateNumericColumn}), buckets rows into groups by key
     * (first-seen order, via {@link GroupKey}), accumulates each group's
     * {@link Aggregator}s, then materializes exactly one output column per
     * {@code SELECT} item using {@link GroupPlan#itemKeyIndex}/
     * {@link GroupPlan#itemAggIndex} to pick, per item, either a group's key
     * value or an aggregate's result.
     */
    private VectorSchemaRoot buildGroupedResult(CompiledPlan p, VectorSchemaRoot filtered) {
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
                continue; // COUNT(*) needs no per-row value at all
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
                FieldVector out = p.float64
                        ? new Float8Vector(item.outputName(), allocator)
                        : new Float4Vector(item.outputName(), allocator);
                if (p.float64) {
                    ((Float8Vector) out).allocateNew(numGroups);
                } else {
                    ((Float4Vector) out).allocateNew(numGroups);
                }
                out.setValueCount(numGroups);
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
                outVectors.add(out);
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
     * Bulk-evaluates one expression (a compiled {@code evaluator}, or a
     * direct column read when {@code passthroughColumn} is non-{@code null})
     * across every row of {@code root} into {@code outValues}/{@code outIsNull} —
     * the shared building block behind reading a {@code GROUP BY} key or an
     * aggregate's argument for every row at once, rather than one ParserNG
     * dispatch per row.
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
            if (float64) {
                Float8Vector fv = (Float8Vector) v;
                for (int i = 0; i < rowCount; i++) {
                    outIsNull[i] = fv.isNull(i);
                    outValues[i] = outIsNull[i] ? 0.0 : fv.get(i);
                }
            } else {
                Float4Vector fv = (Float4Vector) v;
                for (int i = 0; i < rowCount; i++) {
                    outIsNull[i] = fv.isNull(i);
                    outValues[i] = outIsNull[i] ? 0.0 : fv.get(i);
                }
            }
            return;
        }
        BufferAllocator allocator = allocatorOf(root);
        if (float64) {
            try (Float8Vector out = new Float8Vector("__parser_ng_sql_bulk__", allocator)) {
                out.allocateNew(rowCount);
                out.setValueCount(rowCount);
                evaluator.evaluate(root, out, nullPolicy);
                for (int i = 0; i < rowCount; i++) {
                    outIsNull[i] = out.isNull(i);
                    outValues[i] = outIsNull[i] ? 0.0 : out.get(i);
                }
            }
        } else {
            try (Float4Vector out = new Float4Vector("__parser_ng_sql_bulk__", allocator)) {
                out.allocateNew(rowCount);
                out.setValueCount(rowCount);
                evaluator.evaluate(root, out, nullPolicy);
                for (int i = 0; i < rowCount; i++) {
                    outIsNull[i] = out.isNull(i);
                    outValues[i] = outIsNull[i] ? 0.0 : out.get(i);
                }
            }
        }
    }

    /**
     * Filters {@code result} (the final projected/grouped output) by
     * {@code HAVING}, reusing exactly the same fused-evaluator /
     * leaf-by-leaf-mask machinery {@code WHERE} uses (see
     * {@link #selectFromEvaluator}/{@link PredicateNode}) — {@code result}
     * is a real {@code VectorSchemaRoot} by this point, so
     * {@link PredicateNode#evalMask} (and the bare-column {@code IS NULL}
     * detection {@link #buildPredicateNode} baked into {@code havingNode}
     * against a phantom root back in {@link #buildPlan}) both resolve
     * correctly against it.
     */
    private VectorSchemaRoot applyHaving(CompiledPlan p, VectorSchemaRoot result) {
        int[] selected;
        int rowCount = result.getRowCount();
        if (p.havingFused != null) {
            selected = selectFromEvaluator(p.havingFused, result, rowCount, p.float64, nullPolicy);
        } else {
            boolean[] mask = p.havingNode.evalMask(result, nullPolicy, p.float64);
            int count = 0;
            for (boolean b : mask) {
                if (b) {
                    count++;
                }
            }
            selected = new int[count];
            int idx = 0;
            for (int i = 0; i < mask.length; i++) {
                if (mask[i]) {
                    selected[idx++] = i;
                }
            }
        }
        return materializeRows(result, selected);
    }

    /**
     * Sorts {@code root} by {@link SelectStatement#orderBy()}, evaluating
     * every key once in bulk (see {@link #evaluateNumericColumn}) rather
     * than per-comparison, then gathering rows via a stable multi-key
     * {@link java.util.Comparator} over row indices. {@code root} is
     * whichever root {@link #runPlan} passes in: the final grouped/HAVING-filtered
     * result for a grouped query, or the pre-projection {@code WHERE}-filtered
     * root for a non-grouped one — see {@code buildPlan}'s "ORDER BY" comment
     * for why those differ, and {@code CompiledPlan#orderByPlans}, which is
     * built to match whichever one will actually be passed here. SQL
     * null-ordering convention: nulls sort last regardless of
     * {@code ASC}/{@code DESC} (a null is "unknown", not "smallest" or
     * "largest" — this matches PostgreSQL's default, one of the two common
     * real-engine conventions; MySQL/SQLite instead treat null as the
     * smallest value under {@code ASC}).
     */
    private VectorSchemaRoot applyOrderBy(CompiledPlan p, VectorSchemaRoot root) {
        int rowCount = root.getRowCount();
        List<OrderItem> orderBy = stmt.orderBy();
        int numKeys = orderBy.size();
        double[][] values = new double[numKeys][];
        boolean[][] isNull = new boolean[numKeys][];
        for (int k = 0; k < numKeys; k++) {
            values[k] = new double[rowCount];
            isNull[k] = new boolean[rowCount];
            OrderByPlan op = p.orderByPlans.get(k);
            evaluateNumericColumn(op.evaluator, op.sourceColumnName, root, p.float64, nullPolicy,
                    values[k], isNull[k]);
        }

        Integer[] order = new Integer[rowCount];
        for (int i = 0; i < rowCount; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (i1, i2) -> {
            for (int k = 0; k < numKeys; k++) {
                boolean n1 = isNull[k][i1];
                boolean n2 = isNull[k][i2];
                int cmp;
                if (n1 || n2) {
                    cmp = (n1 == n2) ? 0 : (n1 ? 1 : -1); // nulls last, regardless of ASC/DESC
                } else {
                    cmp = Double.compare(values[k][i1], values[k][i2]);
                    if (orderBy.get(k).descending()) {
                        cmp = -cmp;
                    }
                }
                if (cmp != 0) {
                    return cmp;
                }
            }
            return 0; // stable: ties keep their original relative order
        });

        int[] indices = new int[rowCount];
        for (int i = 0; i < rowCount; i++) {
            indices[i] = order[i];
        }
        return materializeRows(root, indices);
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
        // Non-null iff this is a GROUP BY / aggregate query (SelectStatement#isGrouped());
        // mutually exclusive with `projections`, which stays empty in that
        // case. See "GROUP BY / HAVING / ORDER BY / LIMIT" in this class's
        // javadoc.
        final GroupPlan groupPlan;
        // At most one of havingFused/havingNode is non-null, mirroring
        // fusedPredicate/maskPredicateRoot's own "fused fast path vs.
        // leaf-by-leaf IS-NULL fallback" split for WHERE -- both are always
        // null when the query has no HAVING clause.
        final ArrowExpressionEvaluator havingFused;
        final PredicateNode havingNode;
        // Parallel to stmt.orderBy(), in the same (sort-priority) order.
        // For a grouped query, every entry is a plain (non-passthrough)
        // compiled evaluator operating on the final grouped result. For a
        // non-grouped query, an entry may instead be a passthrough marker
        // (see OrderByPlan) -- ORDER BY there runs against the
        // pre-projection filtered root, where a bare-column reference is
        // both common and worth optimizing. See buildPlan's ORDER BY
        // comment.
        final List<OrderByPlan> orderByPlans;
        // Every distinct compiled evaluator this plan owns, deduplicated by
        // rendered ParserNG text (see buildPlan's compiledCache) -- the
        // *sole* owner responsible for closing each one exactly once, even
        // though the very same instance may also be referenced by
        // fusedPredicate, a leaf inside maskPredicateRoot/havingNode, an
        // entry of projections/groupPlan, and/or orderByEvaluators. See
        // "Expression deduplication" in this class's javadoc.
        final List<ArrowExpressionEvaluator> compiledEvaluators;

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
     * Everything needed to turn a filtered row set into a grouped result:
     * how to compute each {@code GROUP BY} key, how to compute each
     * aggregate's per-row input, and how each {@code SELECT} item's output
     * column is produced from those two lists. See
     * {@link #buildGroupPlan} and {@link #buildGroupedResult}.
     */
    private static final class GroupPlan {

        final List<KeyPlan> keys;
        final List<AggPlan> aggregates;
        // Parallel to stmt.items(): for item i, exactly one of
        // itemKeyIndex[i] (an index into `keys`) or itemAggIndex[i] (an
        // index into `aggregates`) is >= 0 and the other is -1.
        final int[] itemKeyIndex;
        final int[] itemAggIndex;

        GroupPlan(List<KeyPlan> keys, List<AggPlan> aggregates, int[] itemKeyIndex, int[] itemAggIndex) {
            this.keys = keys;
            this.aggregates = aggregates;
            this.itemKeyIndex = itemKeyIndex;
            this.itemAggIndex = itemAggIndex;
        }
    }

    /** One {@code GROUP BY} key expression, compiled (or passthrough-optimized) once. */
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

    /** One aggregate call's compiled argument (or {@code COUNT(*)}'s absence of one). */
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

    /**
     * One {@code ORDER BY} key, compiled (or passthrough-optimized) once.
     * See {@code CompiledPlan#orderByPlans}' javadoc for when the
     * passthrough path is actually used.
     */
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
     * A hashable tuple of {@code GROUP BY} key values for one row, used as
     * the key of the {@code LinkedHashMap} that buckets rows into groups in
     * {@link #buildGroupedResult} (insertion order preserved, so a query
     * with no {@code ORDER BY} still gets a deterministic, first-seen
     * output order). A {@code null} key value (an input row where a
     * {@code GROUP BY} expression evaluated to {@code null}) is represented
     * as {@code Double.NaN} — every SQL null groups with every other null
     * on that key, matching standard {@code GROUP BY} null-handling, and
     * {@code Double.doubleToLongBits}-based equality/hashing (rather than
     * {@code ==}/{@code Double.hashCode}'s own contract, which already
     * agrees for {@code NaN}) makes that comparison exact and stable.
     */
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

    /**
     * A per-group running accumulator for one aggregate call. {@code null}
     * input values (an aggregate argument that evaluated to {@code null}
     * for some row) are skipped entirely -- standard SQL aggregate
     * null-handling -- except {@link CountAgg}, which increments
     * unconditionally for {@code COUNT(*)} and skips nulls for
     * {@code COUNT(expr)} exactly like every other aggregate.
     */
    private interface Aggregator {

        void accumulate(double value, boolean isNull);

        /** @return {@code true} iff this aggregate's result is SQL null (no non-null input seen; never true for COUNT) */
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