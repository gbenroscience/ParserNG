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
import com.github.gbenroscience.sqlv1.ast.NotExpr;
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
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.TransferPair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
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
 * transparently triggers a one-time recompilation — see "Thread-safety"
 * for how that recompilation is made safe against a concurrently in-flight
 * call still using the plan being replaced.
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
 * <h2>No-{@code WHERE} and {@code LIMIT} fast paths</h2>
 * When a query has no {@code WHERE} clause at all, every row of the input
 * survives unfiltered; when a query has a {@code LIMIT}, the surviving rows
 * are always a contiguous prefix of whatever came before it. Both cases are
 * detected explicitly ({@link #selectRows} returns {@code null} as a "keep
 * everything, in order" sentinel for the first; {@link #runPlan} calls
 * {@link #materializePrefix} directly for the second) and handled with a
 * single {@link TransferPair}-based buffer transfer per column — an
 * O(1)-relative-to-row-count buffer copy — instead of the general
 * {@link #materializeRows} path's per-row {@code copyFromSafe} loop, which
 * is reserved for genuinely non-contiguous/reordered row sets (an actual
 * {@code WHERE} filter, or a sort). {@link #buildProjection}'s renamed
 * passthrough case (e.g. {@code SELECT x AS y}) uses the same bulk-transfer
 * technique for the same reason: it is always copying an already-materialized,
 * contiguous {@code [0, rowCount)} range, just under a new name.
 * <p>
 * <b>Known remaining cost, deliberately not addressed here:</b> both fast
 * paths above still copy <i>every</i> column of the source root, even one a
 * query never references (e.g. {@code SELECT x*y FROM data} where
 * {@code data} also has unrelated columns {@code z, a, b}). Pruning to only
 * the columns a query actually needs is possible in principle, but is only
 * <i>safe</i> if the underlying {@code ArrowExpressionEvaluator} resolves
 * its referenced columns by name from whichever root it is given, rather
 * than by position against the exact schema it was compiled with — this
 * class has no visibility into that implementation detail, so it does not
 * attempt the pruning rather than risk silently reading the wrong column
 * into the wrong output.
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
 * same instance more than once) — see "Thread-safety" for how and when
 * those evaluators are actually closed.
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
 * <p>
 * <b>Per-call consistency:</b> each {@link #execute(VectorSchemaRoot)} /
 * {@link #execute(VectorSchemaRoot, VectorSchemaRoot)} call reads
 * {@link #nullPolicy} exactly once, at the very start, and threads that one
 * snapshot through every sub-evaluation the call performs ({@code WHERE},
 * every projection, {@code ORDER BY}, {@code HAVING}). This guarantees a
 * single call is internally consistent even if another thread calls
 * {@link #withNullPolicy(NullPolicy)} while it is in flight — see
 * "Thread-safety".
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
 * reference to that source vector rather than a freshly allocated copy —
 * {@link #buildProjection} tracks which output vectors are such reused
 * references (in an identity set, since two different {@code FieldVector}
 * instances can be {@code equals}-equal) so its cleanup path never closes
 * one still owned by the result it's returning. A passthrough column that
 * *is* renamed (e.g. {@code SELECT x AS y FROM t}) still gets a real,
 * independent copy, since the same source column referenced twice under
 * different names (e.g. {@code SELECT x, x AS y FROM t}) must never share
 * one mutable vector between two independent output columns — that copy is
 * a single bulk {@link TransferPair} buffer transfer, not a per-row loop
 * (see "No-{@code WHERE} and {@code LIMIT} fast paths"). Both optimizations
 * only apply to {@link #execute(VectorSchemaRoot)}'s fresh-result-per-call
 * path — the reusable-output overload below always copies into the
 * caller's buffer row-by-row, since that buffer's identity must remain
 * stable across calls (see "Reusable output buffers").
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
 * <h2>Reusable output buffers (avoiding a fresh result per call)</h2>
 * {@link #execute(VectorSchemaRoot)} always returns a brand-new,
 * independently-owned {@link VectorSchemaRoot} — simple and safe, but for a
 * tight, repeated-call hot path (a benchmark, a streaming loop) it means a
 * fresh allocation on every single call. {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}
 * is an opt-in alternative for that hot path: it writes into a
 * caller-supplied, caller-owned output root instead, reusing the exact same
 * output buffers call after call, and reuses an internal scratch buffer
 * (see {@link ReusableScratch}) for the two things that overload cannot
 * write directly into the caller's buffer: a {@code WHERE}-narrowed
 * computed column (still needs a full-batch scratch evaluation before
 * gathering) and the {@code WHERE} predicate's own evaluation output. It is
 * deliberately restricted to queries with no {@code GROUP BY},
 * {@code ORDER BY}, {@code HAVING} or {@code LIMIT} — every one of those
 * can make the output row count depend on the data in a way a
 * fixed-capacity buffer cannot safely be pre-sized for — and throws
 * {@link UnsupportedOperationException} up front for any query shape it
 * does not support, rather than silently doing something surprising.
 * {@link #allocateReusableOutput} builds a correctly-shaped/typed buffer
 * for a given query, so callers do not have to hand-construct one. See
 * both methods' own javadoc for the full contract, and "Thread-safety"
 * below for the concurrency restriction this overload carries that
 * {@link #execute(VectorSchemaRoot)} does not.
 *
 * <h2>Resource ownership</h2>
 * An {@code ArrowQuery} owns whatever {@code ArrowExpressionEvaluator}s it
 * has compiled and must be {@link #close()}d when no longer needed (a
 * try-with-resources block is the simplest way, as in the example above).
 * Every {@code VectorSchemaRoot} returned by {@link #execute(VectorSchemaRoot)}
 * is a fresh, independent batch that the <i>caller</i> owns and must close
 * in turn — this class never returns a view over, or a root that shares
 * ownership with, the root passed in. {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}
 * is the one deliberate exception: it returns the very {@code reusableOutput}
 * the caller passed in, which the caller already owns and keeps owning.
 *
 * <h2>Thread-safety</h2>
 * <b>{@link #execute(VectorSchemaRoot)} is safe to call concurrently from
 * multiple threads</b> — including concurrently with {@link #close()},
 * {@link #withBackend}, or another thread triggering a schema-driven plan
 * recompilation. This is deliberate, not incidental: the compiled plan a
 * call uses is reference-counted ({@link CompiledPlan#acquire()}/
 * {@link CompiledPlan#release()}/{@link CompiledPlan#retire()}) — a call
 * acquires the plan it looked up before using it and releases it when
 * done, and a plan superseded by recompilation, a backend change, or
 * {@code close()} only has its evaluators actually closed once every
 * in-flight call holding a reference to it has released. A thread mid
 * {@link #execute(VectorSchemaRoot)} therefore always finishes safely
 * against the plan it started with, never against one already closed out
 * from under it. Similarly, {@link #withNullPolicy(NullPolicy)} is safe to
 * call concurrently: each {@code execute} call snapshots {@link #nullPolicy}
 * once at the start (see "A note on {@code NullPolicy}"'s "Per-call
 * consistency"), so a policy change never splits across one call's own
 * sub-evaluations.
 * <p>
 * What this does <i>not</i> give you: a single {@code execute} call is not
 * linearized against a concurrent {@code withBackend}/{@code withNullPolicy}
 * call — a call already in flight when the configuration changes completes
 * using whatever it already captured, not the new configuration. For
 * predictable, easy-to-reason-about behavior, configure backend/nullPolicy
 * once before traffic starts rather than changing them under concurrent
 * load, even though doing so is now memory-safe.
 * <p>
 * {@link #execute(VectorSchemaRoot, VectorSchemaRoot)} and
 * {@link #allocateReusableOutput} are the exception to all of the above:
 * they are <b>not</b> safe to call concurrently on the same
 * {@code ArrowQuery} instance, even under a stable schema/configuration,
 * because they read and mutate a per-instance {@link ReusableScratch} —
 * that mutable state is exactly what lets them avoid the allocations
 * {@link #execute(VectorSchemaRoot)} pays for. Use one {@code ArrowQuery}
 * per thread for this overload (queries are cheap to {@link #compile}), or
 * fully serialize access to a shared instance yourself.
 *
 * <h2>Warming up</h2>
 * The very first time a given compiled expression actually runs in a JVM
 * process, it is meaningfully slower than every run after it — the
 * underlying ParserNG/SIMD machinery is paying a one-time classloading and
 * JIT-warmup cost, not a cost proportional to the data. Left alone, that
 * cost lands on whichever {@link #execute} call happens to be first —
 * frequently a real request a real caller is waiting on. {@link #warmup}
 * pays that cost deliberately, ahead of time, on synthetic data:
 * <pre>{@code
 * try (ArrowQuery query = ArrowQuery.compile(sql)) {
 *     query.warmup(sampleRoot);   // synthetic data, same schema as sampleRoot
 *     // ... later, in the real hot path:
 *     query.execute(realRoot);    // no cold-start penalty left to pay
 * }
 * }</pre>
 * {@code sampleRoot} is read only for its schema (column names and types,
 * via {@link #isFloat64}) — its row data, if it has any, is never touched,
 * and it is never closed by this call. Because the synthetic batch
 * {@link #warmup} builds shares that exact schema, the plan it compiles is
 * the very same plan a real {@link #execute} call against that schema would
 * need — {@link #warmup} does not compile a separate, throwaway plan that
 * then gets discarded; it primes the real one, cached exactly the way any
 * other {@link #execute} call caches it (see "What \"compiled\" means here,
 * precisely" above). Only the row values are synthetic and thrown away.
 *
 * @author GBEMIRO
 */
public final class ArrowQuery implements AutoCloseable {

    /**
     * Default row count for the synthetic batch {@link #warmup()} builds.
     * Chosen empirically: large enough that a single {@link #execute} call
     * against it gives the JIT enough loop iterations to fully tier up,
     * small enough that a handful of repetitions stays fast. A too-small
     * warmup batch (a handful of rows) does not reliably reach this —
     * see {@link #warmup(VectorSchemaRoot, int, int)} if tuning is needed.
     */
    public static final int DEFAULT_WARMUP_ROWS = 20_000;

    /**
     * Default repetition count for {@link #warmup()}. Repeating against
     * the same synthetic batch several times, not once, is what lets the
     * JIT actually converge — see {@link #warmup(VectorSchemaRoot, int, int)}.
     */
    public static final int DEFAULT_WARMUP_REPETITIONS = 8;

    private final String sql;
    private final SelectStatement stmt;

    private volatile ArrowExecutionBackend backend = ArrowExecutionBackend.CPU_SIMD;
    private volatile NullPolicy nullPolicy = NullPolicy.PROPAGATE;

    private volatile CompiledPlan plan;
    private volatile VectorSchemaRoot lastRoot;

    /**
     * Scratch buffers for {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}'s
     * WHERE-evaluation and WHERE-narrowed-computed-column paths — see that
     * method's javadoc and this class's "Thread-safety". Never touched by
     * {@link #execute(VectorSchemaRoot)}.
     */
    private volatile ReusableScratch reusableScratch;

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
     * recompiles it on the next {@link #execute} — safely with respect to
     * any other thread currently mid-{@link #execute(VectorSchemaRoot)}
     * against the old plan; see this class's "Thread-safety".
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
     * per-compiled-kernel one — and each in-flight {@link #execute} call is
     * unaffected by a concurrent call to this method; see "Per-call
     * consistency" and "Thread-safety".
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
     * Warms up this query using {@link #DEFAULT_WARMUP_ROWS} synthetic rows
     * repeated {@link #DEFAULT_WARMUP_REPETITIONS} times, over the same
     * schema as {@code schemaTemplate} — see this class's "Warming up".
     *
     * @param schemaTemplate a root with the same column names and types
     * real calls to {@link #execute} will use; read only for its schema,
     * never for its row data, and never closed by this call
     * @return {@code this}, for chaining
     * @throws ArrowSqlException if compiling the query's expressions
     * against {@code schemaTemplate}'s schema fails
     */
    public ArrowQuery warmup(VectorSchemaRoot schemaTemplate) {
        return warmup(schemaTemplate, DEFAULT_WARMUP_ROWS, DEFAULT_WARMUP_REPETITIONS);
    }

    /**
     * Warms up this query using {@code rows} synthetic rows repeated
     * {@code repetitions} times, over the same schema as
     * {@code schemaTemplate} — see this class's "Warming up". Most callers
     * want {@link #warmup(VectorSchemaRoot)}; this overload exists for
     * cases where {@link #DEFAULT_WARMUP_ROWS}/{@link #DEFAULT_WARMUP_REPETITIONS}
     * either overshoot a tight startup budget or undershoot for an
     * unusually heavy expression.
     *
     * @param schemaTemplate a root with the same column names and types
     * real calls to {@link #execute} will use; read only for its schema,
     * never for its row data, and never closed by this call
     * @param rows how many synthetic rows to generate; must be positive
     * @param repetitions how many times to execute against that synthetic
     * batch; must be positive — a single repetition is rarely enough to
     * reach a steady JIT state (see this class's "Warming up")
     * @return {@code this}, for chaining
     * @throws IllegalArgumentException if {@code rows} or {@code repetitions}
     * is not positive
     * @throws ArrowSqlException if compiling the query's expressions
     * against {@code schemaTemplate}'s schema fails
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
        boolean float64 = isFloat64(schemaTemplate);
        try (RootAllocator syntheticAllocator = new RootAllocator(Long.MAX_VALUE)) {
            VectorSchemaRoot synthetic = buildSyntheticRoot(schemaTemplate.getSchema(), rows, float64, syntheticAllocator);
            try {
                for (int i = 0; i < repetitions; i++) {
                    execute(synthetic).close();
                }
            } finally {
                for (FieldVector v : synthetic.getFieldVectors()) {
                    closeQuietly(v);
                }
            }
        }
        return this;
    }

    /**
     * Builds a throwaway root matching {@code schema}'s column names and
     * types exactly, filled with deterministic pseudo-random values (a
     * fixed seed, so a given {@code (schema, rows)} pair always produces
     * the same synthetic data run to run) — used only by {@link #warmup}.
     * Values are drawn from a wide, arbitrary range with no attempt to
     * respect any real-world meaning a column name might suggest, since
     * warmup only needs the underlying ParserNG kernels to actually run,
     * not to produce meaningful output.
     */
    private static VectorSchemaRoot buildSyntheticRoot(
            Schema schema, int rows, boolean float64, BufferAllocator allocator) {
        Random random = new Random(0x50415252_4E47L); // "PARRNG" -- fixed, arbitrary seed
        List<Field> fields = new ArrayList<>(schema.getFields().size());
        List<FieldVector> vectors = new ArrayList<>(schema.getFields().size());
        for (org.apache.arrow.vector.types.pojo.Field templateField : schema.getFields()) {
            Field field = new Field(templateField.getName(), templateField.getFieldType(), null);
            if (float64) {
                Float8Vector v = (Float8Vector) field.createVector(allocator);
                v.allocateNew(rows);
                for (int i = 0; i < rows; i++) {
                    v.set(i, (random.nextDouble() * 2000) - 1000);
                }
                v.setValueCount(rows);
                fields.add(field);
                vectors.add(v);
            } else {
                Float4Vector v = (Float4Vector) field.createVector(allocator);
                v.allocateNew(rows);
                for (int i = 0; i < rows; i++) {
                    v.set(i, (random.nextFloat() * 2000f) - 1000f);
                }
                v.setValueCount(rows);
                fields.add(field);
                vectors.add(v);
            }
        }
        return new VectorSchemaRoot(new Schema(fields), vectors, rows);
    }

    /**
     * Executes this query against {@code root}, compiling (or, on a later
     * call against a root with an unchanged schema, reusing) the underlying
     * ParserNG expressions as needed. Safe to call concurrently from
     * multiple threads — see this class's "Thread-safety".
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
        // Snapshotted once, up front -- see "A note on NullPolicy"'s
        // "Per-call consistency" and this class's "Thread-safety".
        NullPolicy effectiveNullPolicy = this.nullPolicy;
        CompiledPlan p;
        try {
            p = ensurePlan(root);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new ArrowSqlException(
                    "Failed to compile query \"" + sql + "\" against the given schema: " + t.getMessage(), t);
        }
        try {
            return runPlan(p, root, effectiveNullPolicy);
        } finally {
            p.release();
        }
    }

    /**
     * Like {@link #execute(VectorSchemaRoot)}, but writes into a
     * caller-supplied, caller-owned {@code reusableOutput} root instead of
     * allocating a fresh result on every call — see this class's "Reusable
     * output buffers" for the motivation. {@link #allocateReusableOutput}
     * builds a correctly-shaped {@code reusableOutput} for this query, so
     * most callers should not need to hand-construct one.
     *
     * <p><b>Not safe for concurrent use on the same {@code ArrowQuery}
     * instance</b> — see this class's "Thread-safety". Use one
     * {@code ArrowQuery} per thread for this overload.
     *
     * <h4>Supported query shapes</h4>
     * Only a query with no {@code GROUP BY}, {@code HAVING}, {@code ORDER BY}
     * or {@code LIMIT} is supported — every one of those can make the
     * result's row count depend on the data in a way a fixed-capacity
     * buffer cannot be safely pre-sized for. Calling this on an
     * unsupported query shape throws {@link UnsupportedOperationException}
     * immediately, without touching {@code root} or {@code reusableOutput}
     * at all; use {@link #execute(VectorSchemaRoot)} for those queries
     * instead.
     *
     * <h4>Sizing {@code reusableOutput}</h4>
     * {@code reusableOutput} must have exactly one column per
     * {@code SELECT} item (or, for {@code SELECT *}, one column per column
     * of {@code root}), and every one of its vectors must have a
     * {@link FieldVector#getValueCapacity()} of at least
     * {@code root.getRowCount()} — a {@code WHERE} clause can only shrink
     * the row count relative to {@code root}, never grow it, so sizing to
     * {@code root}'s own row count is always sufficient headroom. This is
     * checked up front and reported as an {@link IllegalArgumentException}
     * naming the offending column, rather than left to fail obscurely
     * mid-write. This method never calls {@code allocateNew()} or otherwise
     * reallocates {@code reusableOutput}'s vectors — sizing them adequately
     * up front is entirely the caller's responsibility, which is the whole
     * point of reuse: this call performs no allocation of its own on the
     * repeat-call path (other than internal scratch space, kept and grown
     * as needed rather than freed between calls — see below).
     *
     * <h4>What this does <i>not</i> avoid</h4>
     * A passthrough column (a bare {@code SELECT x FROM ...} item) is still
     * copied row-by-row into {@code reusableOutput}, not zero-copy
     * referenced the way {@link #execute(VectorSchemaRoot)} can — the
     * output buffer's identity must stay stable across calls, so there is
     * no vector to alias into it without invalidating that guarantee.
     *
     * @param root the input batch, exactly as for {@link #execute(VectorSchemaRoot)}
     * @param reusableOutput the caller-owned buffer to write into; returned
     * unchanged as this call's result (with {@link VectorSchemaRoot#setRowCount})
     * updated to the actual output row count)
     * @return {@code reusableOutput}, after being populated
     * @throws UnsupportedOperationException if this query has a
     * {@code GROUP BY}, {@code HAVING}, {@code ORDER BY} or {@code LIMIT}
     * @throws IllegalArgumentException if {@code reusableOutput}'s column
     * count or any column's capacity is insufficient
     * @throws ArrowSqlException if compiling the query's expressions
     * against {@code root}'s schema fails
     * @throws ArrowBindingException if evaluation fails at runtime
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
        CompiledPlan p;
        try {
            p = ensurePlan(root);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new ArrowSqlException(
                    "Failed to compile query \"" + sql + "\" against the given schema: " + t.getMessage(), t);
        }
        try {
            int rowCount = root.getRowCount();
            ReusableScratch scratch = scratchFor(p, p.projections.size());
            int[] selected = selectRowsReusable(p, root, effectiveNullPolicy, scratch);
            int outRowCount = selected == null ? rowCount : selected.length;

            if (stmt.selectAll()) {
                List<FieldVector> sourceVectors = root.getFieldVectors();
                List<FieldVector> outVectors = reusableOutput.getFieldVectors();
                validateReusableOutputShape(sourceVectors.size(), outVectors, rowCount);
                for (int i = 0; i < sourceVectors.size(); i++) {
                    gatherColumnInto(sourceVectors.get(i), selected, outVectors.get(i), outRowCount);
                }
                reusableOutput.setRowCount(outRowCount);
                return reusableOutput;
            }

            List<ProjectionPlan> projections = p.projections;
            List<FieldVector> outVectors = reusableOutput.getFieldVectors();
            validateReusableOutputShape(projections.size(), outVectors, rowCount);

            for (int i = 0; i < projections.size(); i++) {
                ProjectionPlan proj = projections.get(i);
                FieldVector out = outVectors.get(i);

                if (proj.passthrough) {
                    FieldVector src = root.getVector(proj.sourceColumnName);
                    if (src == null) {
                        throw new ArrowBindingException(
                                "Column '" + proj.sourceColumnName + "' not found while projecting.");
                    }
                    gatherColumnInto(src, selected, out, outRowCount);

                } else if (selected == null) {
                    // No WHERE narrowed anything: evaluate straight into the
                    // caller's buffer, no scratch/gather step needed at all.
                    out.setValueCount(rowCount);
                    proj.evaluator.evaluate(root, out, effectiveNullPolicy);

                } else {
                    // WHERE narrowed rows: the compiled evaluator only knows
                    // how to evaluate over an entire batch, so evaluate over
                    // all of `root` into a reusable scratch vector, then
                    // gather just the accepted rows into `out`.
                    FieldVector scratchVec = scratch.forProjection(i, rowCount, p.float64, allocatorOf(root));
                    proj.evaluator.evaluate(root, scratchVec, effectiveNullPolicy);
                    gatherColumnInto(scratchVec, selected, out, outRowCount);
                }
            }

            reusableOutput.setRowCount(outRowCount);
            return reusableOutput;
        } finally {
            p.release();
        }
    }

    /**
     * Builds a {@code VectorSchemaRoot} correctly shaped and typed to pass
     * as {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}'s
     * {@code reusableOutput} argument for this query: one column per
     * {@code SELECT} item (matching name and, for a computed column, the
     * {@code float64}/{@code float32} kernel precision {@code schemaTemplate}
     * implies; for a passthrough column, the source column's own type), or
     * one column per column of {@code schemaTemplate} for {@code SELECT *}.
     * Every column is allocated with capacity for at least {@code maxRows}
     * rows — pass the largest {@code root.getRowCount()} you expect to
     * {@link #execute(VectorSchemaRoot, VectorSchemaRoot)} across this
     * buffer's lifetime, since this method (like that one) never
     * reallocates on your behalf later.
     *
     * @param schemaTemplate a root with the same column names/types real
     * calls to {@link #execute(VectorSchemaRoot, VectorSchemaRoot)} will
     * use; read only for its schema, never for its row data, and never
     * closed by this call
     * @param maxRows capacity to allocate for each output column; must be
     * positive
     * @return a zero-row, fully allocated {@code VectorSchemaRoot} the
     * caller owns and must eventually close
     * @throws UnsupportedOperationException if this query has a
     * {@code GROUP BY}, {@code HAVING}, {@code ORDER BY} or {@code LIMIT}
     * @throws IllegalArgumentException if {@code maxRows} is not positive
     * @throws ArrowSqlException if compiling the query's expressions
     * against {@code schemaTemplate}'s schema fails
     */
    public VectorSchemaRoot allocateReusableOutput(VectorSchemaRoot schemaTemplate, int maxRows) {
        if (schemaTemplate == null) {
            throw new NullPointerException("schemaTemplate must not be null");
        }
        if (maxRows <= 0) {
            throw new IllegalArgumentException("maxRows must be positive, was " + maxRows);
        }
        requireReusableOutputSupportedShape();

        CompiledPlan p;
        try {
            p = ensurePlan(schemaTemplate);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new ArrowSqlException(
                    "Failed to compile query \"" + sql + "\" against the given schema: " + t.getMessage(), t);
        }
        try {
            BufferAllocator allocator = allocatorOf(schemaTemplate);
            List<Field> outFields = new ArrayList<>();
            List<FieldVector> outVectors = new ArrayList<>();
            try {
                if (stmt.selectAll()) {
                    for (FieldVector src : schemaTemplate.getFieldVectors()) {
                        FieldVector v = src.getField().createVector(allocator);
                        v.setInitialCapacity(maxRows);
                        v.allocateNew();
                        outFields.add(v.getField());
                        outVectors.add(v);
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
                            type = FieldType.nullable(p.float64
                                    ? new org.apache.arrow.vector.types.pojo.ArrowType.FloatingPoint(
                                            org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE)
                                    : new org.apache.arrow.vector.types.pojo.ArrowType.FloatingPoint(
                                            org.apache.arrow.vector.types.FloatingPointPrecision.SINGLE));
                        }
                        Field field = new Field(proj.outputName, type, null);
                        FieldVector v = field.createVector(allocator);
                        v.setInitialCapacity(maxRows);
                        v.allocateNew();
                        outFields.add(field);
                        outVectors.add(v);
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
        if (stmt.isGrouped()) {
            throw new UnsupportedOperationException(
                    "execute(root, reusableOutput)/allocateReusableOutput do not support GROUP BY queries "
                            + "-- their row count depends on the data and cannot be bounded by a "
                            + "fixed-capacity output buffer. Use execute(root) instead.");
        }
        if (stmt.having() != null) {
            throw new UnsupportedOperationException(
                    "execute(root, reusableOutput)/allocateReusableOutput do not support HAVING. "
                            + "Use execute(root) instead.");
        }
        if (!stmt.orderBy().isEmpty()) {
            throw new UnsupportedOperationException(
                    "execute(root, reusableOutput)/allocateReusableOutput do not support ORDER BY. "
                            + "Use execute(root) instead.");
        }
        if (stmt.limit() != null) {
            throw new UnsupportedOperationException(
                    "execute(root, reusableOutput)/allocateReusableOutput do not support LIMIT. "
                            + "Use execute(root) instead.");
        }
    }

    /**
     * Validates that {@code reusableOutput}'s column count and per-column
     * capacity ({@link FieldVector#getValueCapacity()}) are sufficient for
     * {@code rowCount} input rows — see
     * {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}'s "Sizing
     * reusableOutput". Does not check per-column Arrow type: a mismatch
     * there (e.g. passing a {@code Float4Vector} where this query produces
     * {@code Float8Vector} values) surfaces as a {@link ClassCastException}
     * from the underlying {@code copyFromSafe}/{@code evaluate} call
     * instead, since the correct expected type can differ per column
     * (passthrough columns keep their source's own type) and re-deriving
     * it here would duplicate logic {@link #allocateReusableOutput} and
     * {@link #buildProjection} already own.
     */
    private static void validateReusableOutputShape(
            int expectedColumns, List<FieldVector> outVectors, int rowCount) {
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
                                + v.getValueCapacity() + " but root has " + rowCount + " rows; a WHERE "
                                + "clause can only shrink the row count, never grow it, so reusableOutput "
                                + "must be sized to at least root's row count. See allocateReusableOutput.");
            }
        }
    }

    /**
     * Copies either every row of {@code src} in order ({@code selected == null})
     * or exactly the rows named by {@code selected}, in that order, into
     * {@code dst} — the shared gather step behind
     * {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}'s passthrough
     * and WHERE-narrowed-computed-column paths. Always a per-row
     * {@code copyFromSafe} loop, even for {@code selected == null}: unlike
     * {@link #materializeRows}'s bulk {@link TransferPair} fast path,
     * {@code dst} here is a caller-owned buffer whose identity must remain
     * stable across calls, so its backing storage cannot simply be handed
     * over from {@code src} the way a fresh, this-call-only result root's
     * could be.
     */
    private static void gatherColumnInto(FieldVector src, int[] selected, FieldVector dst, int outRowCount) {
        if (selected == null) {
            for (int i = 0; i < outRowCount; i++) {
                dst.copyFromSafe(i, i, src);
            }
        } else {
            for (int i = 0; i < outRowCount; i++) {
                dst.copyFromSafe(selected[i], i, src);
            }
        }
        dst.setValueCount(outRowCount);
    }

    private synchronized ReusableScratch scratchFor(CompiledPlan p, int projectionCount) {
        ReusableScratch s = reusableScratch;
        if (s == null || s.forPlan != p) {
            if (s != null) {
                s.close();
            }
            s = new ReusableScratch(p, projectionCount);
            reusableScratch = s;
        }
        return s;
    }

    /**
     * {@code WHERE}-clause row selection for
     * {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}: identical
     * result semantics to {@link #selectRows}, but evaluates the fused
     * predicate into {@code scratch}'s reusable predicate buffer instead of
     * allocating a fresh one — safe only because this overload is
     * documented as never called concurrently on the same instance (see
     * this class's "Thread-safety"). The leaf-by-leaf {@code IS [NOT] NULL}
     * mask path is left as-is (still allocates several {@code boolean[]}
     * arrays per call) — a lower-priority cost than the Arrow off-heap
     * buffer allocations the fused path avoids, and safely reusing those
     * arrays across an arbitrary {@code AndNode}/{@code OrNode} tree shape
     * is more involved than the win currently justifies.
     */
    private static int[] selectRowsReusable(
            CompiledPlan p, VectorSchemaRoot root, NullPolicy nullPolicy, ReusableScratch scratch) {
        if (p.fusedPredicate == null && p.maskPredicateRoot == null) {
            return null;
        }
        int rowCount = root.getRowCount();
        if (p.fusedPredicate != null) {
            return selectFromEvaluatorReusable(p.fusedPredicate, root, rowCount, p.float64, nullPolicy, scratch);
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

    private static int[] selectFromEvaluatorReusable(
            ArrowExpressionEvaluator predicate, VectorSchemaRoot root, int rowCount,
            boolean float64, NullPolicy nullPolicy, ReusableScratch scratch) {

        if (rowCount == 0) {
            return new int[0];
        }
        FieldVector out = scratch.predicateScratch(rowCount, float64, allocatorOf(root));
        int[] buffer = new int[Math.max(16, rowCount / 4)];
        int count = 0;

        predicate.evaluate(root, out, nullPolicy);
        if (float64) {
            Float8Vector out8 = (Float8Vector) out;
            for (int i = 0; i < rowCount; i++) {
                if (out8.isNull(i)) {
                    continue;
                }
                if (out8.get(i) != 0.0) {
                    if (count == buffer.length) {
                        buffer = Arrays.copyOf(buffer, buffer.length * 2);
                    }
                    buffer[count++] = i;
                }
            }
        } else {
            Float4Vector out4 = (Float4Vector) out;
            for (int i = 0; i < rowCount; i++) {
                if (out4.isNull(i)) {
                    continue;
                }
                if (out4.get(i) != 0.0f) {
                    if (count == buffer.length) {
                        buffer = Arrays.copyOf(buffer, buffer.length * 2);
                    }
                    buffer[count++] = i;
                }
            }
        }
        return Arrays.copyOf(buffer, count);
    }

    /**
     * Releases every {@code ArrowExpressionEvaluator} this query has
     * compiled — or, if another thread is currently mid-{@link #execute(VectorSchemaRoot)}
     * against the current plan, marks it for release as soon as that call
     * finishes (see this class's "Thread-safety" and
     * {@link CompiledPlan#retire()}). Safe to call more than once, safe to
     * call even if {@link #execute} was never called, and safe to call
     * concurrently with an in-flight {@link #execute(VectorSchemaRoot)}
     * call on another thread.
     */
    @Override
    public synchronized void close() {
        invalidatePlan();
    }

    private synchronized void invalidatePlan() {
        lastRoot = null;
        if (plan != null) {
            plan.retire();
            plan = null;
        }
        if (reusableScratch != null) {
            reusableScratch.close();
            reusableScratch = null;
        }
    }

    // =====================================================================
    // plan compilation (lazy, cached, schema-fingerprinted, ref-counted)
    // =====================================================================

    /**
     * Looks up (compiling if necessary) the {@link CompiledPlan} for
     * {@code root}'s schema and returns it already
     * {@link CompiledPlan#acquire() acquired} — the caller MUST
     * {@link CompiledPlan#release()} it exactly once, in a {@code finally}
     * block, when done. See this class's "Thread-safety".
     */
    private CompiledPlan ensurePlan(VectorSchemaRoot root) throws Throwable {
        // Lock-free fast path: the overwhelmingly common case is the exact
        // same VectorSchemaRoot instance being re-executed in a loop (a
        // cached materialized batch, a benchmark, a retry) - reference
        // equality against the last-seen root skips the fingerprint
        // computation AND the synchronized section entirely. Safe under
        // races: a stale read here, or an acquire() that loses a race
        // against a concurrent retire(), just falls through to the
        // synchronized slow path below, which always resolves correctly.
        VectorSchemaRoot seenRoot = lastRoot;
        CompiledPlan cached = plan;
        if (seenRoot == root && cached != null && cached.acquire()) {
            return cached;
        }
        return ensurePlanSlow(root);
    }

    private synchronized CompiledPlan ensurePlanSlow(VectorSchemaRoot root) throws Throwable {
        long fingerprint = fingerprintOf(root);
        CompiledPlan existing = plan;
        if (existing != null && existing.fingerprint == fingerprint) {
            lastRoot = root;
            if (existing.acquire()) {
                return existing;
            }
            // existing was retired-and-closed by a race between the
            // volatile read above and this acquire() (e.g. a concurrent
            // close()) -- fall through and rebuild exactly as if there
            // were no cached plan at all.
        }
        CompiledPlan fresh = buildPlan(root, fingerprint);
        lastRoot = root;
        if (plan != null) {
            // Mark the outgoing plan superseded. If another thread is
            // currently mid-execute() holding an acquired reference to it,
            // its evaluators are NOT closed here -- only once that
            // reference is released. See CompiledPlan's javadoc.
            plan.retire();
        }
        plan = fresh;
        if (reusableScratch != null) {
            // Tied to the outgoing plan's identity (see ReusableScratch's
            // javadoc); scratchFor() would rebuild it lazily on the next
            // execute(root, reusableOutput) call regardless, but closing
            // eagerly here avoids it lingering in the meantime. Safe to
            // close immediately, unlike the plan itself: this overload is
            // documented as never called concurrently with anything else
            // on this instance, so there is no other thread that could be
            // mid-use of it right now.
            reusableScratch.close();
            reusableScratch = null;
        }
        fresh.acquire(); // brand new plan: never retired, this always succeeds
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
        // frequently identical text. See "Expression deduplication" in
        // this class's javadoc.
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

    private VectorSchemaRoot runPlan(CompiledPlan p, VectorSchemaRoot root, NullPolicy nullPolicy) {
        int[] selected = selectRows(p, root, nullPolicy);
        VectorSchemaRoot filtered = materializeRows(root, selected);

        // Non-grouped queries sort *before* projection: an ORDER BY key may
        // name a real input column that never appears in the SELECT list,
        // or a SELECT-list alias resolved back to its input-column-based
        // expression (see buildPlan's ORDER BY comment) -- both only
        // resolve correctly against the pre-projection root.
        if (p.groupPlan == null && !p.orderByPlans.isEmpty()) {
            VectorSchemaRoot sortedFiltered = applyOrderBy(p, filtered, nullPolicy);
            for (FieldVector v : filtered.getFieldVectors()) {
                closeQuietly(v);
            }
            filtered = sortedFiltered;
        }

        VectorSchemaRoot result;
        if (p.groupPlan != null) {
            try {
                result = buildGroupedResult(p, filtered, nullPolicy);
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
                result = buildProjection(p, filtered, nullPolicy);
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
            VectorSchemaRoot havingFiltered = applyHaving(p, result, nullPolicy);
            for (FieldVector v : result.getFieldVectors()) {
                closeQuietly(v);
            }
            result = havingFiltered;
        }
        if (p.groupPlan != null && !p.orderByPlans.isEmpty()) {
            VectorSchemaRoot sorted = applyOrderBy(p, result, nullPolicy);
            for (FieldVector v : result.getFieldVectors()) {
                closeQuietly(v);
            }
            result = sorted;
        }
        if (stmt.limit() != null && stmt.limit() < result.getRowCount()) {
            // Contiguous-prefix fast path: LIMIT always keeps rows
            // [0, limit) of whatever ORDER BY (or the query's natural
            // order) produced, so a bulk TransferPair copy applies here
            // exactly as it does for the no-WHERE case in selectRows/
            // materializeRows -- see this class's "No-WHERE and LIMIT fast
            // paths".
            VectorSchemaRoot limited = materializePrefix(result, stmt.limit());
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
    private static VectorSchemaRoot applyHaving(CompiledPlan p, VectorSchemaRoot result, NullPolicy nullPolicy) {
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
    private VectorSchemaRoot applyOrderBy(CompiledPlan p, VectorSchemaRoot root, NullPolicy nullPolicy) {
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

    /**
     * @return {@code null} as a "keep every row of {@code root}, in its
     * original order" sentinel when this query has no {@code WHERE} clause
     * at all (letting {@link #materializeRows} take its bulk
     * {@link TransferPair} fast path instead of allocating and populating
     * an identity index array just to hand it back), or the row indices
     * that survive {@code WHERE} otherwise. See this class's "No-WHERE and
     * LIMIT fast paths". Always allocates a fresh predicate-evaluation
     * output buffer per call, unlike {@link #selectRowsReusable} — this is
     * the version used by {@link #execute(VectorSchemaRoot)}, which must
     * stay safe for concurrent multi-threaded use and therefore cannot
     * share any mutable per-instance scratch state.
     */
    private static int[] selectRows(CompiledPlan p, VectorSchemaRoot root, NullPolicy nullPolicy) {
        if (p.fusedPredicate == null && p.maskPredicateRoot == null) {
            return null;
        }
        int rowCount = root.getRowCount();
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

    /**
     * General row-materialization path: copies exactly the rows named by
     * {@code selectedIndices}, in that order, into a fresh, independent
     * {@code VectorSchemaRoot} — one {@code copyFromSafe} call per
     * (column, output row) pair, because a genuinely filtered and/or
     * reordered index set has no cheaper representation.
     * {@code selectedIndices == null} instead means "every row of
     * {@code source}, in its original order" — handled by delegating to
     * {@link #materializePrefix}'s bulk {@link TransferPair} copy rather
     * than falling through to the per-row loop below, since that identity
     * case is exactly what a plain buffer transfer already does correctly
     * and far more cheaply. See this class's "No-WHERE and LIMIT fast
     * paths".
     */
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

    /**
     * Copies row range {@code [0, length)} of every column of
     * {@code source} into a fresh, independent {@code VectorSchemaRoot},
     * using one {@link TransferPair#splitAndTransfer} buffer-level copy per
     * column rather than a per-row {@code copyFromSafe} loop. Used for two
     * distinct, unrelated-looking cases that both happen to be "keep a
     * contiguous prefix, in order": a query with no {@code WHERE} clause at
     * all ({@code length == source.getRowCount()}, called from
     * {@link #materializeRows} for its {@code selectedIndices == null}
     * sentinel), and {@code LIMIT n} ({@code length == n}, called directly
     * from {@link #runPlan}). See this class's "No-WHERE and LIMIT fast
     * paths".
     *
     * <p>{@code length} must not exceed {@code source.getRowCount()} —
     * both call sites already guarantee this ({@code source.getRowCount()}
     * itself, or a {@code LIMIT} already checked to be smaller).
     */
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

    private static VectorSchemaRoot buildProjection(CompiledPlan p, VectorSchemaRoot filtered, NullPolicy nullPolicy) {
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
                        // built for us instead of allocating a fresh one -
                        // the exact optimization already relied on for
                        // this (overwhelmingly common) case.
                        out = src;
                        reused.add(src);
                    } else {
                        // Renamed passthrough (e.g. SELECT x AS y): still
                        // an independent copy (the same source column
                        // referenced twice under different names must
                        // never share one mutable vector -- see this
                        // class's "Column aliasing"), but a straight,
                        // already-contiguous [0, rowCount) duplication is
                        // exactly what a single TransferPair-based buffer
                        // copy handles -- no per-row copyFromSafe loop
                        // needed at all.
                        TransferPair tp = src.getTransferPair(proj.outputName, allocator);
                        tp.splitAndTransfer(0, rowCount);
                        out = (FieldVector) tp.getTo();
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

    /**
     * A compiled, immutable execution plan for one schema shape, with a
     * reference-counted lifecycle: {@link #acquire()} before use,
     * {@link #release()} exactly once when done (always paired via
     * try/finally at every call site), {@link #retire()} when superseded.
     * A plan's {@code compiledEvaluators} are only ever actually closed
     * once it has been both retired AND has no outstanding acquired
     * references — see this class's own javadoc and {@code ArrowQuery}'s
     * "Thread-safety" for why this is what makes concurrent
     * {@code close()}/{@code withBackend()}/schema-driven recompilation
     * safe against another thread's in-flight {@code execute()} call still
     * using the plan being replaced.
     */
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
        final List<OrderByPlan> orderByPlans;
        // Every distinct compiled evaluator this plan owns, deduplicated by
        // rendered ParserNG text (see buildPlan's compiledCache) -- the
        // *sole* set closed by closeIfReady() below, even though the very
        // same instance may also be referenced by fusedPredicate, a leaf
        // inside maskPredicateRoot/havingNode, an entry of
        // projections/groupPlan, and/or orderByPlans. See "Expression
        // deduplication" in ArrowQuery's javadoc.
        final List<ArrowExpressionEvaluator> compiledEvaluators;

        private int refCount = 0;
        private boolean retired = false;
        private boolean closed = false;

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

        /**
         * Acquires a usable reference to this plan for the duration of one
         * {@code execute(...)}-family call. Returns {@code false} (never
         * throws) if this plan has already been fully retired-and-closed —
         * which can only happen if the caller held a stale reference
         * across a race with a concurrent recompilation/close; the
         * caller's correct response is to look the plan up again (see
         * {@code ArrowQuery#ensurePlan}), not to treat this as an error.
         */
        synchronized boolean acquire() {
            if (closed) {
                return false;
            }
            refCount++;
            return true;
        }

        /** Releases one reference acquired via {@link #acquire()}. */
        synchronized void release() {
            refCount--;
            closeIfReady();
        }

        /**
         * Marks this plan superseded. Its evaluators are closed immediately
         * if nothing currently holds an acquired reference, or deferred
         * until the last such reference is {@link #release()}d.
         */
        synchronized void retire() {
            retired = true;
            closeIfReady();
        }

        private void closeIfReady() {
            if (retired && refCount == 0 && !closed) {
                closed = true;
                for (ArrowExpressionEvaluator e : compiledEvaluators) {
                    e.close();
                }
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
     * Per-{@code ArrowQuery} scratch state for
     * {@link #execute(VectorSchemaRoot, VectorSchemaRoot)}: one scratch
     * vector per computed {@link ProjectionPlan} (lazily allocated, grown —
     * never shrunk — on demand) plus one dedicated scratch vector for the
     * {@code WHERE} predicate's own evaluation output. Tied to a specific
     * {@link CompiledPlan} instance ({@code forPlan}) so a schema-driven
     * recompilation can never leave a stale scratch vector sized or typed
     * for a plan that no longer exists — {@link ArrowQuery#scratchFor}
     * checks {@code forPlan} identity and rebuilds on a mismatch. This
     * mutable state is exactly why
     * {@link #execute(VectorSchemaRoot, VectorSchemaRoot)} is documented as
     * not safe for concurrent use on the same {@code ArrowQuery} instance —
     * see that class's "Thread-safety".
     */
    private static final class ReusableScratch {

        final CompiledPlan forPlan;
        final FieldVector[] perProjection;
        private FieldVector predicateScratch;

        ReusableScratch(CompiledPlan forPlan, int projectionCount) {
            this.forPlan = forPlan;
            this.perProjection = new FieldVector[projectionCount];
        }

        FieldVector forProjection(int index, int rowCount, boolean float64, BufferAllocator allocator) {
            FieldVector v = perProjection[index];
            if (v == null || v.getValueCapacity() < rowCount) {
                if (v != null) {
                    closeQuietly(v);
                }
                v = float64
                        ? new Float8Vector("__parser_ng_sql_reuse_scratch__", allocator)
                        : new Float4Vector("__parser_ng_sql_reuse_scratch__", allocator);
                if (float64) {
                    ((Float8Vector) v).allocateNew(rowCount);
                } else {
                    ((Float4Vector) v).allocateNew(rowCount);
                }
                perProjection[index] = v;
            }
            v.setValueCount(rowCount);
            return v;
        }

        FieldVector predicateScratch(int rowCount, boolean float64, BufferAllocator allocator) {
            FieldVector v = predicateScratch;
            if (v == null || v.getValueCapacity() < rowCount) {
                if (v != null) {
                    closeQuietly(v);
                }
                v = float64
                        ? new Float8Vector("__parser_ng_sql_reuse_predicate_scratch__", allocator)
                        : new Float4Vector("__parser_ng_sql_reuse_predicate_scratch__", allocator);
                if (float64) {
                    ((Float8Vector) v).allocateNew(rowCount);
                } else {
                    ((Float4Vector) v).allocateNew(rowCount);
                }
                predicateScratch = v;
            }
            v.setValueCount(rowCount);
            return v;
        }

        void close() {
            for (FieldVector v : perProjection) {
                if (v != null) {
                    closeQuietly(v);
                }
            }
            if (predicateScratch != null) {
                closeQuietly(predicateScratch);
                predicateScratch = null;
            }
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