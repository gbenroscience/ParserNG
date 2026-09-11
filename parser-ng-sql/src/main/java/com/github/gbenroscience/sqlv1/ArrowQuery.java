package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.arrow.tools.box.ArrowBindingException;
import com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend;
import com.github.gbenroscience.arrow.tools.box.ArrowExpressionEvaluator;
import com.github.gbenroscience.arrow.tools.box.ArrowExpressionEvaluators;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import com.github.gbenroscience.sqlv1.ast.AggregateKind;
import com.github.gbenroscience.sqlv1.ast.AndExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExprs;
import com.github.gbenroscience.sqlv1.ast.IsNullExpr;
import com.github.gbenroscience.sqlv1.ast.OrExpr;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
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
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;

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
 * <h2>Aggregation strategy ({@code GROUP BY}/{@code HAVING}/aggregate {@code SELECT} items)</h2>
 * A query is handled by the aggregation path whenever {@link SelectStatement#isAggregateQuery()}
 * is {@code true} (a non-empty {@code GROUP BY}, or at least one aggregate
 * {@code SELECT} item even without one — see that method's javadoc for the
 * implicit-single-group case). The pipeline is:
 * <ol>
 * <li>{@code WHERE} is applied first, exactly as in the non-aggregate case,
 * against the original per-row {@code VectorSchemaRoot}.</li>
 * <li>Each {@code GROUP BY} key expression is evaluated once over every
 * (post-{@code WHERE}) row (compiled once per plan, like everything else —
 * see "Expression deduplication"), and rows are partitioned by the
 * resulting composite key, preserving first-seen group order.</li>
 * <li>A small synthetic, in-memory {@code VectorSchemaRoot} — one row per
 * group — is materialized: one column per {@code GROUP BY} expression
 * (named after its own trimmed text) plus one column per {@code SELECT}
 * item (named after {@link SelectItem#outputName()}). A group-key
 * {@code SELECT} item's column is copied from the matching {@code GROUP BY}
 * column's value for that group (validated at parse time — see
 * {@link SelectStatement}'s javadoc); an aggregate item's column is
 * computed by materializing just that group's rows into their own small
 * sub-{@code VectorSchemaRoot}, evaluating the aggregate's argument
 * expression against it, and reducing client-side ({@code SUM}/{@code AVG}/
 * {@code MIN}/{@code MAX} skip {@code NULL} argument values the same way
 * SQL does; {@code COUNT(*)} is simply the group's row count; an aggregate
 * over zero non-null values is {@code NULL}, except {@code COUNT} which is
 * {@code 0}). This is a plain nested-loop grouped aggregation, not a
 * vectorized one — adequate for parser-ng-sql's goal of making {@code SQL}
 * a convenient way to describe a computation, not a claim of
 * database-grade grouped-aggregation performance.</li>
 * <li>{@code HAVING}, if present, is compiled and evaluated against that
 * synthetic per-group root using <i>exactly</i> the same fused/leaf-mask
 * machinery as {@code WHERE} (see "Predicate evaluation strategy") — it is
 * simply a {@code WHERE}-shaped filter over a different root. Its operand
 * text must resolve to a real column of that root: a bare {@code GROUP BY}
 * expression's own text, or a {@code SELECT} item's {@link SelectItem#outputName()}.
 * <b>Alias an aggregate {@code SELECT} item you intend to reference from
 * {@code HAVING}</b> (e.g. {@code SUM(x) AS total ... HAVING total > 10})
 * — an unaliased reference only resolves if written identically to that
 * item's own reconstructed call text, which is fragile to rely on.</li>
 * <li>{@code ORDER BY} and {@code LIMIT} are then applied to that
 * (possibly {@code HAVING}-filtered) per-group root exactly as described
 * below, and finally the ordinary projection/aliasing step runs — every
 * {@code SELECT} item is, by construction, already a same-named passthrough
 * column of that root at this point.</li>
 * </ol>
 *
 * <h2>{@code ORDER BY} and {@code LIMIT}</h2>
 * Both apply <i>after</i> {@code WHERE} (and, for an aggregate query,
 * {@code GROUP BY}/{@code HAVING}) but <i>before</i> the final projection —
 * an {@code ORDER BY} key may therefore name a real input column or a
 * {@code SELECT}-list alias (resolved the same way {@code WHERE} resolves
 * an alias — see "{@code WHERE} referencing a {@code SELECT}-list alias" —
 * for a non-aggregate query only; an aggregate query's {@code ORDER BY}
 * follows the same alias-or-verbatim-call-text resolution rule just
 * described for {@code HAVING}) even when that alias is not itself the
 * sort key of the final output ordering. Sorting is a stable multi-key sort
 * (later keys break ties among equal earlier keys); {@code NULL} values
 * sort last regardless of {@code ASC}/{@code DESC}. {@code LIMIT} simply
 * caps the row count after sorting (or, with no {@code ORDER BY}, after
 * whatever order the rows already have) — both are folded into a single
 * row-selecting/reordering pass, so at most one extra copy of the
 * pre-projection data is made regardless of whether one, both, or neither
 * clause is present.
 *
 * <h2>{@code SELECT *}</h2>
 * Passes every column of the (filtered) input straight through, under its
 * original name — there is no aliasing syntax for {@code *} in the grammar
 * (see {@link SqlParser}). {@code SELECT *} can never be combined with
 * {@code GROUP BY}/{@code HAVING}/an aggregate item (rejected at parse
 * time by {@link SelectStatement}), so this path is never an aggregate
 * query; {@code ORDER BY}/{@code LIMIT} still apply to it normally.
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
                Predicate<String> hasColumn = name -> root.getVector(name) != null;
                maskPredicateRoot = buildPredicateNode(where, backend, float64, hasColumn, compiledCache);
            } else {
                String text = BoolExprs.renderFused(where);
                fusedPredicate = compileCached(text, backend, float64, compiledCache);
            }
        }

        boolean aggregateQuery = stmt.isAggregateQuery();

        // --- GROUP BY / aggregate SELECT items / HAVING (see "Aggregation
        // strategy" in this class's javadoc) -- every evaluator compiled
        // here is root-agnostic text (ArrowExpressionEvaluators.compile
        // takes only text + backend, never a root), so it is safe to
        // compile all of this once here even though the synthetic
        // per-group root it eventually runs against does not exist until
        // execute() actually groups some rows.
        List<ArrowExpressionEvaluator> groupByEvaluators = List.of();
        List<AggregateItemPlan> aggregateItems = List.of();
        ArrowExpressionEvaluator havingFusedPredicate = null;
        PredicateNode havingMaskPredicateRoot = null;

        if (aggregateQuery) {
            List<ArrowExpressionEvaluator> gbEvals = new ArrayList<>();
            for (String gbExpr : stmt.groupBy()) {
                gbEvals.add(compileCached(gbExpr, backend, float64, compiledCache));
            }
            groupByEvaluators = gbEvals;

            Map<String, Integer> groupByIndexByText = new LinkedHashMap<>();
            for (int i = 0; i < stmt.groupBy().size(); i++) {
                groupByIndexByText.put(stmt.groupBy().get(i).trim(), i);
            }

            List<AggregateItemPlan> items = new ArrayList<>();
            for (SelectItem item : stmt.items()) {
                if (item.isAggregate()) {
                    ArrowExpressionEvaluator argEval = item.aggregateStar()
                            ? null : compileCached(item.aggregateArgText(), backend, float64, compiledCache);
                    items.add(AggregateItemPlan.aggregate(
                            item.outputName(), item.aggregateKind(), item.aggregateStar(), argEval));
                } else {
                    Integer idx = groupByIndexByText.get(item.exprText().trim());
                    if (idx == null) {
                        // SelectStatement's own constructor already rejects
                        // this; defensive guard in case that invariant is
                        // ever loosened without updating this class.
                        throw new ArrowSqlException(
                                "Column '" + item.exprText() + "' must appear in GROUP BY or be used "
                                        + "inside an aggregate function.", null);
                    }
                    items.add(AggregateItemPlan.groupKey(item.outputName(), idx));
                }
            }
            aggregateItems = items;

            BoolExpr having = stmt.having();
            if (having != null) {
                Set<String> aggregatedColumnNames = new HashSet<>();
                for (String gb : stmt.groupBy()) {
                    aggregatedColumnNames.add(gb.trim());
                }
                for (SelectItem item : stmt.items()) {
                    aggregatedColumnNames.add(item.outputName());
                }
                if (BoolExprs.containsIsNull(having)) {
                    havingMaskPredicateRoot = buildPredicateNode(
                            having, backend, float64, aggregatedColumnNames::contains, compiledCache);
                } else {
                    String text = BoolExprs.renderFused(having);
                    havingFusedPredicate = compileCached(text, backend, float64, compiledCache);
                }
            }
        }

        // --- ORDER BY (see "ORDER BY and LIMIT" in this class's javadoc).
        // A non-aggregate query may name a SELECT-list alias, resolved the
        // same way WHERE resolves one; an aggregate query's keys are
        // expected to already be verbatim column names of the synthetic
        // per-group root (a GROUP BY expression's own text, or a SELECT
        // item's outputName()) and are used as-is.
        List<ArrowExpressionEvaluator> orderByEvaluators = new ArrayList<>();
        Map<String, String> orderAliasBindings = aggregateQuery ? Map.of() : collectWhereAliasBindings(root);
        for (OrderItem orderItem : stmt.orderBy()) {
            String text = aggregateQuery
                    ? orderItem.exprText()
                    : WhereAliasResolver.substituteText(orderItem.exprText(), orderAliasBindings);
            orderByEvaluators.add(compileCached(text, backend, float64, compiledCache));
        }

        List<ProjectionPlan> projections = new ArrayList<>();
        if (!stmt.selectAll()) {
            if (aggregateQuery) {
                // Every SELECT item is, by construction, already a
                // same-named column of the synthetic per-group root built
                // at execute() time -- see buildAggregatedRoot(). No
                // evaluator to compile here; this is always a passthrough.
                for (SelectItem item : stmt.items()) {
                    projections.add(new ProjectionPlan(item.outputName(), true, item.outputName(), null));
                }
            } else {
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
        }

        return new CompiledPlan(fingerprint, float64, fusedPredicate, maskPredicateRoot, projections,
                List.copyOf(compiledCache.values()), aggregateQuery, groupByEvaluators, aggregateItems,
                havingFusedPredicate, havingMaskPredicateRoot, orderByEvaluators);
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
     * call regardless. An aggregate {@link SelectItem} (see
     * {@link SelectItem#isAggregate()}) is never included: its
     * {@link SelectItem#exprText()} is the canonical aggregate call text
     * (e.g. {@code "SUM(x)"}), which is meaningless spliced into a
     * per-row {@code WHERE}/{@code ORDER BY} expression evaluated against
     * the original, pre-aggregation root — referencing an aggregate result
     * from {@code WHERE} is invalid SQL in the first place (that is what
     * {@code HAVING} is for), and this method is also reused (see
     * {@code buildPlan}'s {@code ORDER BY} handling) only for the
     * non-aggregate-query case, where no {@link SelectItem} is ever an
     * aggregate anyway.
     */
    private Map<String, String> collectWhereAliasBindings(VectorSchemaRoot root) {
        if (stmt.selectAll()) {
            return Map.of();
        }
        Map<String, String> bindings = new LinkedHashMap<>();
        for (SelectItem item : stmt.items()) {
            if (item.isAggregate()) {
                continue;
            }
            String outputName = item.outputName();
            if (root.getVector(outputName) != null) {
                continue;
            }
            bindings.put(outputName, item.exprText());
        }
        return bindings;
    }

    /**
     * Builds the leaf-by-leaf mask-evaluation fallback for a {@code WHERE}
     * or {@code HAVING} clause containing {@code IS [NOT] NULL} (see
     * "Predicate evaluation strategy").
     *
     * @param hasColumn tells an {@link IsNullExpr} leaf whether its target
     * is a bare column of the root it will eventually be evaluated
     * against, so it can read that column's validity bitmap directly
     * instead of compiling a probe evaluator. For {@code WHERE} this is
     * {@code name -> root.getVector(name) != null} against the real,
     * already-in-hand root; for {@code HAVING} the synthetic per-group
     * root does not exist yet at compile time, so this is instead a
     * membership test against the statically-known set of column names
     * that root will have (every {@code GROUP BY} expression's text, plus
     * every {@code SELECT} item's {@link SelectItem#outputName()}) — see
     * "Aggregation strategy" in this class's javadoc.
     */
    private static PredicateNode buildPredicateNode(
            BoolExpr expr, ArrowExecutionBackend backend, boolean float64, Predicate<String> hasColumn,
            Map<String, ArrowExpressionEvaluator> compiledCache) throws Throwable {

        if (expr instanceof AndExpr a) {
            return new AndNode(
                    buildPredicateNode(a.left(), backend, float64, hasColumn, compiledCache),
                    buildPredicateNode(a.right(), backend, float64, hasColumn, compiledCache));
        }
        if (expr instanceof OrExpr o) {
            return new OrNode(
                    buildPredicateNode(o.left(), backend, float64, hasColumn, compiledCache),
                    buildPredicateNode(o.right(), backend, float64, hasColumn, compiledCache));
        }
        if (expr instanceof IsNullExpr n) {
            String target = n.target().trim();
            if (hasColumn.test(target)) {
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
    // per-call execution: select rows -> [group/aggregate/HAVING] ->
    // [ORDER BY/LIMIT] -> project/alias
    // =====================================================================

    /**
     * Runs the full per-execute pipeline against {@code root}, tracking
     * ownership of exactly one {@code VectorSchemaRoot} ({@code current})
     * at a time: each stage below either returns {@code current} unchanged
     * or produces a brand-new root and closes the old one, so a failure at
     * any point only ever needs to close whatever {@code current} refers
     * to right then. See this class's javadoc, "Aggregation strategy" and
     * "{@code ORDER BY} and {@code LIMIT}".
     */
    private VectorSchemaRoot runPlan(CompiledPlan p, VectorSchemaRoot root) {
        int[] selected = selectRows(p, root);
        VectorSchemaRoot current = materializeRows(root, selected);
        try {
            if (p.aggregateQuery) {
                VectorSchemaRoot aggregated = buildAggregatedRoot(p, current);
                closeAll(current);
                current = aggregated;

                int[] havingSelected = selectHavingRows(p, current);
                VectorSchemaRoot havingFiltered = materializeRows(current, havingSelected);
                closeAll(current);
                current = havingFiltered;
            }

            if (!p.orderByEvaluators.isEmpty() || stmt.limit() != null) {
                int[] permutation = computeOrderPermutation(p, current);
                VectorSchemaRoot ordered = materializeRows(current, permutation);
                closeAll(current);
                current = ordered;
            }

            if (stmt.selectAll()) {
                return current;
            }
            // buildProjection takes ownership of `current` from here: it
            // closes every one of its vectors that isn't reused verbatim
            // in the returned result (see buildProjection's own javadoc).
            return buildProjection(p, current);
        } catch (RuntimeException | Error e) {
            closeAll(current);
            throw e;
        }
    }

    private static void closeAll(VectorSchemaRoot r) {
        for (FieldVector v : r.getFieldVectors()) {
            closeQuietly(v);
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
        return maskToIndices(mask);
    }

    /**
     * {@code HAVING}'s counterpart to {@link #selectRows(CompiledPlan, VectorSchemaRoot)},
     * evaluated against the synthetic per-group root built by
     * {@link #buildAggregatedRoot} instead of the original input root. See
     * this class's javadoc, "Aggregation strategy".
     */
    private int[] selectHavingRows(CompiledPlan p, VectorSchemaRoot aggregated) {
        int rowCount = aggregated.getRowCount();
        if (p.havingFusedPredicate == null && p.havingMaskPredicateRoot == null) {
            int[] all = new int[rowCount];
            for (int i = 0; i < rowCount; i++) {
                all[i] = i;
            }
            return all;
        }
        if (p.havingFusedPredicate != null) {
            return selectFromEvaluator(p.havingFusedPredicate, aggregated, rowCount, p.float64, nullPolicy);
        }
        boolean[] mask = p.havingMaskPredicateRoot.evalMask(aggregated, nullPolicy, p.float64);
        return maskToIndices(mask);
    }

    private static int[] maskToIndices(boolean[] mask) {
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
    // GROUP BY / aggregation -- see this class's javadoc, "Aggregation
    // strategy"
    // =====================================================================

    /**
     * Groups {@code filtered} (the post-{@code WHERE} rows) by
     * {@code p.groupByEvaluators} and materializes a synthetic, one-row-
     * per-group {@code VectorSchemaRoot} carrying every column that
     * {@code HAVING}, {@code ORDER BY}, or the final projection might need:
     * one per {@code GROUP BY} expression (named after its own trimmed
     * text) and one per {@code SELECT} item (named after
     * {@link SelectItem#outputName()}) -- see "Aggregation strategy" for
     * why both sets exist and how a name collision between them is
     * resolved (harmlessly: the values are guaranteed identical whenever
     * the names coincide).
     */
    private VectorSchemaRoot buildAggregatedRoot(CompiledPlan p, VectorSchemaRoot filtered) {
        int rowCount = filtered.getRowCount();
        int groupCount = p.groupByEvaluators.size();
        List<String> groupByTexts = stmt.groupBy();

        double[][] keyValues = new double[groupCount][];
        boolean[][] keyNulls = new boolean[groupCount][];
        for (int g = 0; g < groupCount; g++) {
            EvalResult r = evaluateToDoubles(p.groupByEvaluators.get(g), filtered, p.float64, nullPolicy);
            keyValues[g] = r.values();
            keyNulls[g] = r.nulls();
        }

        Map<GroupKey, List<Integer>> groups = new LinkedHashMap<>();
        if (groupCount == 0) {
            // No explicit GROUP BY but at least one aggregate item (see
            // SelectStatement#isAggregateQuery()): the entire (possibly
            // empty) filtered result is one implicit group, exactly as
            // plain SQL treats an aggregate query with no GROUP BY.
            List<Integer> all = new ArrayList<>(rowCount);
            for (int i = 0; i < rowCount; i++) {
                all.add(i);
            }
            groups.put(GroupKey.EMPTY, all);
        } else {
            for (int i = 0; i < rowCount; i++) {
                double[] key = new double[groupCount];
                boolean[] nulls = new boolean[groupCount];
                for (int g = 0; g < groupCount; g++) {
                    key[g] = keyValues[g][i];
                    nulls[g] = keyNulls[g][i];
                }
                groups.computeIfAbsent(new GroupKey(key, nulls), unused -> new ArrayList<>()).add(i);
            }
        }

        List<GroupKey> keys = new ArrayList<>(groups.keySet());
        List<List<Integer>> rowsByGroup = new ArrayList<>(keys.size());
        for (GroupKey k : keys) {
            rowsByGroup.add(groups.get(k));
        }
        int outRows = keys.size();

        // Column name -> how to fill it, deduplicated by name so a SELECT
        // item whose outputName() coincides with a GROUP BY expression's
        // own text (or with another item) does not produce two Arrow
        // fields of the same name.
        Map<String, AggColumnSpec> columns = new LinkedHashMap<>();
        for (int g = 0; g < groupCount; g++) {
            columns.putIfAbsent(groupByTexts.get(g).trim(), new AggColumnSpec(g));
        }
        for (AggregateItemPlan item : p.aggregateItems) {
            columns.putIfAbsent(item.outputName,
                    item.isGroupKey ? new AggColumnSpec(item.groupByIndex) : new AggColumnSpec(item));
        }

        BufferAllocator allocator = allocatorOf(filtered);
        List<Field> outFields = new ArrayList<>(columns.size());
        List<FieldVector> outVectors = new ArrayList<>(columns.size());
        try {
            for (Map.Entry<String, AggColumnSpec> entry : columns.entrySet()) {
                FieldVector v = newNumericVector(entry.getKey(), allocator, p.float64);
                if (outRows > 0) {
                    v.setInitialCapacity(outRows);
                }
                v.allocateNew();
                AggColumnSpec spec = entry.getValue();
                for (int r = 0; r < outRows; r++) {
                    if (spec.isGroupColumn()) {
                        GroupKey k = keys.get(r);
                        writeValue(v, r, k.values[spec.groupIndex], k.nulls[spec.groupIndex], p.float64);
                    } else {
                        computeAggregate(spec.item, filtered, rowsByGroup.get(r), p.float64, v, r);
                    }
                }
                v.setValueCount(outRows);
                outFields.add(v.getField());
                outVectors.add(v);
            }
        } catch (RuntimeException | Error e) {
            for (FieldVector v : outVectors) {
                closeQuietly(v);
            }
            throw e;
        }
        return new VectorSchemaRoot(new Schema(outFields), outVectors, outRows);
    }

    /**
     * Computes one aggregate {@code SELECT} item's value for a single
     * group and writes it into {@code outVector} at {@code outIdx}. Every
     * kind except {@code COUNT(*)} materializes the group's own rows into
     * a small sub-{@code VectorSchemaRoot} (reusing {@link #materializeRows}),
     * evaluates the aggregate's argument expression against it, and
     * reduces client-side, skipping {@code NULL} argument values -- see
     * {@link AggregateKind}'s javadoc for the exact per-kind semantics.
     */
    private void computeAggregate(
            AggregateItemPlan item, VectorSchemaRoot filtered, List<Integer> rowsInGroup,
            boolean float64, FieldVector outVector, int outIdx) {

        if (item.kind == AggregateKind.COUNT && item.star) {
            writeValue(outVector, outIdx, rowsInGroup.size(), false, float64);
            return;
        }

        int[] indices = new int[rowsInGroup.size()];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = rowsInGroup.get(i);
        }
        VectorSchemaRoot sub = materializeRows(filtered, indices);
        try {
            EvalResult r = evaluateToDoubles(item.argEvaluator, sub, float64, nullPolicy);
            switch (item.kind) {
                case COUNT -> {
                    int count = 0;
                    for (boolean isNull : r.nulls()) {
                        if (!isNull) {
                            count++;
                        }
                    }
                    writeValue(outVector, outIdx, count, false, float64);
                }
                case SUM -> {
                    double sum = 0.0;
                    boolean any = false;
                    for (int i = 0; i < r.values().length; i++) {
                        if (!r.nulls()[i]) {
                            sum += r.values()[i];
                            any = true;
                        }
                    }
                    writeValue(outVector, outIdx, sum, !any, float64);
                }
                case AVG -> {
                    double sum = 0.0;
                    int count = 0;
                    for (int i = 0; i < r.values().length; i++) {
                        if (!r.nulls()[i]) {
                            sum += r.values()[i];
                            count++;
                        }
                    }
                    writeValue(outVector, outIdx, count == 0 ? 0.0 : sum / count, count == 0, float64);
                }
                case MIN -> {
                    double min = Double.POSITIVE_INFINITY;
                    boolean any = false;
                    for (int i = 0; i < r.values().length; i++) {
                        if (!r.nulls()[i] && (!any || r.values()[i] < min)) {
                            min = r.values()[i];
                            any = true;
                        }
                    }
                    writeValue(outVector, outIdx, min, !any, float64);
                }
                case MAX -> {
                    double max = Double.NEGATIVE_INFINITY;
                    boolean any = false;
                    for (int i = 0; i < r.values().length; i++) {
                        if (!r.nulls()[i] && (!any || r.values()[i] > max)) {
                            max = r.values()[i];
                            any = true;
                        }
                    }
                    writeValue(outVector, outIdx, max, !any, float64);
                }
            }
        } finally {
            closeAll(sub);
        }
    }

    private static FieldVector newNumericVector(String name, BufferAllocator allocator, boolean float64) {
        return float64 ? new Float8Vector(name, allocator) : new Float4Vector(name, allocator);
    }

    private static void writeValue(FieldVector v, int idx, double value, boolean isNull, boolean float64) {
        if (float64) {
            Float8Vector fv = (Float8Vector) v;
            if (isNull) {
                fv.setNull(idx);
            } else {
                fv.set(idx, value);
            }
        } else {
            Float4Vector fv = (Float4Vector) v;
            if (isNull) {
                fv.setNull(idx);
            } else {
                fv.set(idx, (float) value);
            }
        }
    }

    /**
     * Evaluates {@code eval} against every row of {@code root}, reading
     * each result back through {@code isNull} before {@code get} (same
     * unconditional guard as everywhere else in this class -- see "A note
     * on {@code NullPolicy}"). Shared by {@code GROUP BY} key evaluation,
     * aggregate-argument evaluation, and {@code ORDER BY} key evaluation.
     */
    private static EvalResult evaluateToDoubles(
            ArrowExpressionEvaluator eval, VectorSchemaRoot root, boolean float64, NullPolicy nullPolicy) {
        int rowCount = root.getRowCount();
        double[] values = new double[rowCount];
        boolean[] nulls = new boolean[rowCount];
        if (rowCount == 0) {
            return new EvalResult(values, nulls);
        }
        BufferAllocator allocator = allocatorOf(root);
        if (float64) {
            try (Float8Vector out = new Float8Vector("__parser_ng_sql_eval__", allocator)) {
                out.allocateNew(rowCount);
                out.setValueCount(rowCount);
                eval.evaluate(root, out, nullPolicy);
                for (int i = 0; i < rowCount; i++) {
                    nulls[i] = out.isNull(i);
                    values[i] = nulls[i] ? 0.0 : out.get(i);
                }
            }
        } else {
            try (Float4Vector out = new Float4Vector("__parser_ng_sql_eval__", allocator)) {
                out.allocateNew(rowCount);
                out.setValueCount(rowCount);
                eval.evaluate(root, out, nullPolicy);
                for (int i = 0; i < rowCount; i++) {
                    nulls[i] = out.isNull(i);
                    values[i] = nulls[i] ? 0.0 : out.get(i);
                }
            }
        }
        return new EvalResult(values, nulls);
    }

    private record EvalResult(double[] values, boolean[] nulls) {
    }

    // =====================================================================
    // ORDER BY / LIMIT -- see this class's javadoc, "ORDER BY and LIMIT"
    // =====================================================================

    /**
     * Computes the row permutation (into {@code base}'s current row
     * order) that {@code ORDER BY}/{@code LIMIT} require: a stable
     * multi-key sort (later {@link OrderItem}s break ties among equal
     * earlier keys; {@code NULL} sorts last regardless of direction),
     * truncated to {@link SelectStatement#limit()} if present. Called only
     * when at least one of {@code ORDER BY}/{@code LIMIT} is actually
     * present (see {@link #runPlan}), but safe (a no-op identity
     * permutation) otherwise too.
     */
    private int[] computeOrderPermutation(CompiledPlan p, VectorSchemaRoot base) {
        int rowCount = base.getRowCount();
        int keyCount = p.orderByEvaluators.size();

        List<Integer> order = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) {
            order.add(i);
        }

        if (keyCount > 0) {
            double[][] keys = new double[keyCount][];
            boolean[][] nulls = new boolean[keyCount][];
            for (int k = 0; k < keyCount; k++) {
                EvalResult r = evaluateToDoubles(p.orderByEvaluators.get(k), base, p.float64, nullPolicy);
                keys[k] = r.values();
                nulls[k] = r.nulls();
            }
            List<OrderItem> orderItems = stmt.orderBy();
            order.sort((a, b) -> {
                for (int k = 0; k < keyCount; k++) {
                    boolean aNull = nulls[k][a];
                    boolean bNull = nulls[k][b];
                    if (aNull && bNull) {
                        continue;
                    }
                    if (aNull) {
                        return 1; // NULLs last, regardless of ASC/DESC
                    }
                    if (bNull) {
                        return -1;
                    }
                    int cmp = Double.compare(keys[k][a], keys[k][b]);
                    if (cmp != 0) {
                        return orderItems.get(k).descending() ? -cmp : cmp;
                    }
                }
                return Integer.compare(a, b); // stable tie-break
            });
        }

        int[] result = new int[order.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = order.get(i);
        }
        Integer limit = stmt.limit();
        if (limit != null && limit < result.length) {
            result = Arrays.copyOf(result, limit);
        }
        return result;
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

        // --- GROUP BY / aggregate / HAVING / ORDER BY -- see this class's
        // javadoc, "Aggregation strategy" and "ORDER BY and LIMIT". Every
        // evaluator referenced below is also present in compiledEvaluators
        // (via buildPlan's shared dedup cache), which remains the sole
        // owner responsible for closing them -- these fields only borrow
        // references for use at runtime.
        final boolean aggregateQuery;
        final List<ArrowExpressionEvaluator> groupByEvaluators;
        final List<AggregateItemPlan> aggregateItems;
        final ArrowExpressionEvaluator havingFusedPredicate;
        final PredicateNode havingMaskPredicateRoot;
        final List<ArrowExpressionEvaluator> orderByEvaluators;

        CompiledPlan(long fingerprint, boolean float64, ArrowExpressionEvaluator fusedPredicate,
                PredicateNode maskPredicateRoot, List<ProjectionPlan> projections,
                List<ArrowExpressionEvaluator> compiledEvaluators, boolean aggregateQuery,
                List<ArrowExpressionEvaluator> groupByEvaluators, List<AggregateItemPlan> aggregateItems,
                ArrowExpressionEvaluator havingFusedPredicate, PredicateNode havingMaskPredicateRoot,
                List<ArrowExpressionEvaluator> orderByEvaluators) {
            this.fingerprint = fingerprint;
            this.float64 = float64;
            this.fusedPredicate = fusedPredicate;
            this.maskPredicateRoot = maskPredicateRoot;
            this.projections = projections;
            this.compiledEvaluators = compiledEvaluators;
            this.aggregateQuery = aggregateQuery;
            this.groupByEvaluators = groupByEvaluators;
            this.aggregateItems = aggregateItems;
            this.havingFusedPredicate = havingFusedPredicate;
            this.havingMaskPredicateRoot = havingMaskPredicateRoot;
            this.orderByEvaluators = orderByEvaluators;
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
     * One {@code SELECT} item's plan within an aggregate query: either a
     * copy of a {@code GROUP BY} key's value for the group ({@code isGroupKey})
     * or a compiled aggregate ({@code kind}/{@code star}/{@code argEvaluator}).
     * See {@link #buildAggregatedRoot} and {@link #computeAggregate}.
     */
    private static final class AggregateItemPlan {

        final String outputName;
        final boolean isGroupKey;
        final int groupByIndex;
        final AggregateKind kind;
        final boolean star;
        final ArrowExpressionEvaluator argEvaluator;

        private AggregateItemPlan(String outputName, boolean isGroupKey, int groupByIndex,
                AggregateKind kind, boolean star, ArrowExpressionEvaluator argEvaluator) {
            this.outputName = outputName;
            this.isGroupKey = isGroupKey;
            this.groupByIndex = groupByIndex;
            this.kind = kind;
            this.star = star;
            this.argEvaluator = argEvaluator;
        }

        static AggregateItemPlan groupKey(String outputName, int groupByIndex) {
            return new AggregateItemPlan(outputName, true, groupByIndex, null, false, null);
        }

        static AggregateItemPlan aggregate(
                String outputName, AggregateKind kind, boolean star, ArrowExpressionEvaluator argEvaluator) {
            return new AggregateItemPlan(outputName, false, -1, kind, star, argEvaluator);
        }
    }

    /**
     * One column of the synthetic per-group root built by
     * {@link #buildAggregatedRoot}: either a copy of group-key value
     * {@code groupIndex}, or the reduction described by {@code item}.
     */
    private static final class AggColumnSpec {

        final int groupIndex;
        final AggregateItemPlan item;

        AggColumnSpec(int groupIndex) {
            this.groupIndex = groupIndex;
            this.item = null;
        }

        AggColumnSpec(AggregateItemPlan item) {
            this.groupIndex = -1;
            this.item = item;
        }

        boolean isGroupColumn() {
            return item == null;
        }
    }

    /**
     * A composite {@code GROUP BY} key: one {@code double} value (plus a
     * {@code NULL} flag) per {@code GROUP BY} expression, compared by
     * value equality so rows with equal key tuples land in the same group.
     * {@link #EMPTY} (a zero-length key) is used for the implicit single
     * group of an aggregate query with no explicit {@code GROUP BY}.
     */
    private static final class GroupKey {

        static final GroupKey EMPTY = new GroupKey(new double[0], new boolean[0]);

        final double[] values;
        final boolean[] nulls;

        GroupKey(double[] values, boolean[] nulls) {
            this.values = values;
            this.nulls = nulls;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof GroupKey other) || other.values.length != values.length) {
                return false;
            }
            for (int i = 0; i < values.length; i++) {
                if (nulls[i] != other.nulls[i]) {
                    return false;
                }
                if (!nulls[i] && Double.compare(values[i], other.values[i]) != 0) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public int hashCode() {
            int h = 1;
            for (int i = 0; i < values.length; i++) {
                h = 31 * h + Boolean.hashCode(nulls[i]);
                h = 31 * h + (nulls[i] ? 0 : Double.hashCode(values[i]));
            }
            return h;
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