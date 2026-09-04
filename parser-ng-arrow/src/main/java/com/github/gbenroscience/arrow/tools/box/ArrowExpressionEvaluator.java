package com.github.gbenroscience.arrow.tools.box;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.Map;
import org.apache.arrow.vector.Float4Vector;

/**
 * Backend-agnostic contract implemented by both {@link ArrowBulkEvaluator}
 * (CPU, SIMD-vectorized) and {@link ArrowGpuBulkEvaluator} (GPU, CUDA or
 * OpenCL).
 *
 * <p>Code written against this interface does not need to know or care which
 * backend actually compiled and evaluates the expression. The same call sites
 * work whether the instance underneath is running on the CPU worker pool or
 * dispatching computation to a GPU device.</p>
 *
 * <h2>Getting an instance</h2>
 *
 * <p>Build instances through {@link ArrowExpressionEvaluators} rather than
 * choosing between {@link ArrowBulkEvaluator#compile} and
 * {@link ArrowGpuBulkEvaluator#compile} directly at each call site. This
 * makes the backend a one-line configuration change via an
 * {@link ArrowExecutionBackend} value instead of requiring a call-site
 * rewrite.</p>
 *
 * <h2>Evaluation and filtering</h2>
 *
 * <p>The primary operation of this interface is bulk expression evaluation:
 * one output value is produced for every input row.</p>
 *
 * <p>The interface also supports {@link #filter(VectorSchemaRoot, NullPolicy)}
 * for compiled boolean expressions. Filtering is deliberately part of this
 * contract because it is a logical Arrow operation that can be implemented
 * by both the CPU/SIMD and GPU backends.</p>
 *
 * <p>For filtering, the compiled ParserNG expression is evaluated as a
 * predicate over the Arrow columns. Internally, an implementation may
 * evaluate the predicate into a SIMD/GPU mask or another selection
 * representation and then materialize the selected rows into an Arrow
 * {@link VectorSchemaRoot}. The selection representation is an implementation
 * detail and is not exposed by this interface.</p>
 *
 * <p>Two further operations build on top of evaluation and filtering, both
 * implemented once as {@code default} methods on this interface (neither
 * backend needs its own override): {@link #project(VectorSchemaRoot, String,
 * NullPolicy)} evaluates this expression over every row and appends the
 * result as a new named column — the projection counterpart to
 * {@code filter}'s row-selection — and {@link #filterProject(VectorSchemaRoot,
 * ArrowExpressionEvaluator, String, NullPolicy)} fuses the two: it filters
 * with this evaluator's predicate first, then evaluates a second
 * ("projection") expression only over the rows that survived, so the
 * projection never spends SIMD/GPU work on rows that are about to be
 * discarded.</p>
 *
 * <p>This is intentionally a filtering/projection-oriented expression
 * interface rather than a SQL/query engine. Operations such as joins,
 * grouping, sorting, aggregation, and SQL parsing are outside the scope of
 * this contract.</p>
 *
 * <h2>Binding, null handling, thread safety</h2>
 *
 * <p>Identical across both implementations. See
 * {@link ArrowBulkEvaluator}'s and {@link ArrowGpuBulkEvaluator}'s class
 * javadocs for the full contract, including name-based variable binding via
 * {@code MathExpression.getSlotItems()}, supported Arrow vector types,
 * {@link NullPolicy} semantics, and each backend's own concurrency rules.</p>
 *
 * <p>Always {@link #close()} when the evaluator is no longer required.
 * Use try-with-resources where practical.</p>
 */
public interface ArrowExpressionEvaluator extends AutoCloseable {

    /**
     * Evaluates the compiled expression, writing one result per row into
     * {@code output}.
     *
     * <p>The expression is evaluated using the backend represented by this
     * evaluator. The caller is responsible for providing a destination vector
     * that is correctly sized for the input row count.</p>
     *
     * @param columns Arrow columns, keyed by the variable name they bind to
     * @param output pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled
     */
    void evaluate(
            Map<String, Float8Vector> columns,
            Float8Vector output,
            NullPolicy nullPolicy);

    /**
     * Convenience overload for
     * {@link #evaluate(Map, Float8Vector, NullPolicy)} using
     * {@link NullPolicy#IGNORE}.
     *
     * @param columns Arrow columns, keyed by variable name
     * @param output pre-sized destination vector
     */
    default void evaluate(
            Map<String, Float8Vector> columns,
            Float8Vector output) {

        evaluate(columns, output, NullPolicy.IGNORE);
    }

    /**
     * Evaluates the compiled expression, writing one result per row into
     * {@code output}.
     *
     * <p>This overload operates on Arrow {@link Float4Vector} columns and
     * produces a float result for every input row.</p>
     *
     * @param columns Arrow columns, keyed by the variable name they bind to
     * @param output pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled
     */
    void evaluate(
            Map<String, Float4Vector> columns,
            Float4Vector output,
            NullPolicy nullPolicy);

    /**
     * Convenience overload for
     * {@link #evaluate(Map, Float4Vector, NullPolicy)} using
     * {@link NullPolicy#IGNORE}.
     *
     * @param columns Arrow columns, keyed by variable name
     * @param output pre-sized destination vector
     */
    default void evaluate(
            Map<String, Float4Vector> columns,
            Float4Vector output) {

        evaluate(columns, output, NullPolicy.IGNORE);
    }

    /**
     * Evaluates the compiled expression against the columns contained in
     * {@code root}, resolving each required variable by name.
     *
     * <p>This is the record-batch convenience form of the map-based
     * {@link #evaluate(Map, Float8Vector, NullPolicy)} operation.</p>
     *
     * @param root Arrow record batch containing the required input columns
     * @param output pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled
     */
    void evaluate(
            VectorSchemaRoot root,
            Float8Vector output,
            NullPolicy nullPolicy);

    /**
     * Convenience overload for
     * {@link #evaluate(VectorSchemaRoot, Float8Vector, NullPolicy)} using
     * {@link NullPolicy#IGNORE}.
     *
     * @param root Arrow record batch containing the required input columns
     * @param output pre-sized destination vector
     */
    default void evaluate(
            VectorSchemaRoot root,
            Float8Vector output) {

        evaluate(root, output, NullPolicy.IGNORE);
    }

    /**
     * Evaluates the compiled expression against the columns contained in
     * {@code root}, resolving each required variable by name.
     *
     * <p>This overload operates on float Arrow columns and produces a float
     * result for every input row.</p>
     *
     * @param root Arrow record batch containing the required input columns
     * @param output pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled
     */
    void evaluate(
            VectorSchemaRoot root,
            Float4Vector output,
            NullPolicy nullPolicy);

    /**
     * Convenience overload for
     * {@link #evaluate(VectorSchemaRoot, Float4Vector, NullPolicy)} using
     * {@link NullPolicy#IGNORE}.
     *
     * @param root Arrow record batch containing the required input columns
     * @param output pre-sized destination vector
     */
    default void evaluate(
            VectorSchemaRoot root,
            Float4Vector output) {

        evaluate(root, output, NullPolicy.IGNORE);
    }

    /**
     * Filters an Arrow record batch using this evaluator's compiled
     * expression as a boolean predicate.
     *
     * <p>The expression represented by this evaluator is evaluated once for
     * each row in {@code root}. Rows for which the predicate evaluates to
     * {@code true} are retained; rows for which it evaluates to
     * {@code false} are discarded.</p>
     *
     * <p>This operation is deliberately defined on the compiled evaluator
     * rather than accepting a {@code MathExpression} argument. The evaluator
     * already represents a compiled ParserNG expression, so the expression
     * does not need to be extracted, converted back into source text, or
     * reparsed when filtering is requested.</p>
     *
     * <p>Implementations are expected to exploit their native execution
     * strategy when evaluating the predicate. The CPU implementation may
     * evaluate the predicate using SIMD vector operations, while the GPU
     * implementation may evaluate it on the selected GPU backend.</p>
     *
     * <p>A typical implementation will conceptually perform:</p>
     *
     * <pre>
     * Arrow columns
     *       |
     *       v
     * compiled ParserNG predicate
     *       |
     *       v
     * SIMD/GPU predicate evaluation
     *       |
     *       v
     * selection mask
     *       |
     *       v
     * Arrow result batch
     * </pre>
     *
     * <p>The selection mask or equivalent intermediate representation is an
     * implementation detail and is not exposed by this interface. This leaves
     * room for the CPU and GPU implementations to use different selection
     * mechanisms without changing the public API.</p>
     *
     * <p>The returned {@link VectorSchemaRoot} contains the selected rows and
     * therefore may contain fewer rows than the input. If no rows satisfy
     * the predicate, an empty result batch is returned.</p>
     *
     * <p>The schema and column ordering of the result should correspond to the
     * input {@code root}. Implementations must preserve row correspondence
     * across all columns when materializing the selected rows.</p>
     *
     * <p>Predicate null handling is governed by {@link NullPolicy}. The exact
     * treatment of null predicate values must be consistent with the
     * implementation's documented {@code NullPolicy} contract.</p>
     *
     * @param root Arrow record batch containing the columns referenced by the
     *             compiled predicate
     * @param nullPolicy how Arrow validity bitmaps and null predicate values
     *                   are handled
     * @return a new Arrow record batch containing only rows for which the
     *         compiled predicate evaluates to true
     * @throws IllegalArgumentException if {@code root} does not contain a
     *                                  required variable or otherwise violates
     *                                  the evaluator's input contract
     */
    VectorSchemaRoot filter(
            VectorSchemaRoot root,
            NullPolicy nullPolicy);

    /**
     * Convenience overload for
     * {@link #filter(VectorSchemaRoot, NullPolicy)} using
     * {@link NullPolicy#IGNORE}.
     *
     * <p>This is the preferred form for callers that do not require explicit
     * null-policy selection.</p>
     *
     * @param root Arrow record batch containing the columns referenced by the
     *             compiled predicate
     * @return a new Arrow record batch containing only rows for which the
     *         compiled predicate evaluates to true
     */
    default VectorSchemaRoot filter(VectorSchemaRoot root) {
        return filter(root, NullPolicy.IGNORE);
    }

    /**
     * Evaluates this compiled expression over every row of {@code root} and
     * returns a new {@link VectorSchemaRoot} with the same rows and the same
     * columns as {@code root}, plus one new trailing column named
     * {@code outputFieldName} holding this expression's per-row result.
     *
     * <p>This is the projection counterpart to {@link #filter}: where
     * {@code filter} keeps all columns but drops rows, {@code project} keeps
     * all rows but adds a column — the Arrow analogue of a SQL
     * {@code SELECT *, <expr> AS outputFieldName}.
     *
     * <h2>Implementation</h2>
     * This is a {@code default} method implemented once, here, in terms of
     * the backend-agnostic {@link #evaluate(VectorSchemaRoot, Float8Vector,
     * NullPolicy)} / {@link #evaluate(VectorSchemaRoot, Float4Vector,
     * NullPolicy)} primitives every implementation already provides — neither
     * {@link ArrowBulkEvaluator} nor {@link ArrowGpuBulkEvaluator} needs its
     * own override to support this. A backend is of course free to override
     * it later with something more specialized (e.g. a GPU kernel that writes
     * the projected column without a separate host-visible round trip), but
     * nothing about correctness depends on that.
     *
     * <p>Row precision (float64 vs. float32) is inferred from {@code root}
     * itself, exactly as {@link #filter} already does: if every existing
     * column in {@code root} is a {@link Float8Vector}, the projected column
     * is computed and appended as a {@link Float8Vector}; otherwise it is
     * computed and appended as a {@link Float4Vector}. As with {@code filter},
     * this means the caller is responsible for having compiled this evaluator
     * for the precision that matches {@code root} — a float64 batch handed to
     * an evaluator compiled via {@code compileF32(...)} will throw
     * {@link IllegalStateException} out of {@code evaluate(...)}, exactly as
     * it would from a direct {@code evaluate} call.
     *
     * <p>The existing columns of {@code root} are reused directly in the
     * returned batch — not copied — since projection does not change the row
     * count or row order; only the new column is freshly allocated and
     * computed. This makes {@code project} considerably cheaper than
     * {@code filter} for large batches.
     *
     * @param root Arrow record batch containing the columns referenced by
     * this compiled expression
     * @param outputFieldName name for the new column holding this
     * expression's result; must not already exist in {@code root}
     * @param nullPolicy how Arrow validity bitmaps are handled — see
     * {@link NullPolicy}
     * @return a new Arrow record batch with {@code root}'s columns plus the
     * projected column, over all of {@code root}'s rows
     * @throws NullPointerException if {@code root}, {@code outputFieldName},
     * or {@code nullPolicy} is null
     * @throws ArrowBindingException if {@code root} has no columns (there is
     * then no allocator to build the projected column from), if
     * {@code root} already has a column named {@code outputFieldName}, or if
     * a required variable's column is missing or of the wrong vector type
     */
    default VectorSchemaRoot project(VectorSchemaRoot root, String outputFieldName, NullPolicy nullPolicy) {
        if (root == null) {
            throw new NullPointerException("root must not be null");
        }
        if (outputFieldName == null) {
            throw new NullPointerException("outputFieldName must not be null");
        }
        if (nullPolicy == null) {
            throw new NullPointerException("nullPolicy must not be null");
        }
        if (root.getVector(outputFieldName) != null) {
            throw new ArrowBindingException(
                    "root already has a column named '" + outputFieldName
                    + "'; choose a different outputFieldName for project(...).");
        }

        int rowCount = root.getRowCount();
        BufferAllocator allocator = ArrowFilterSupport.resolveAllocator(root);

        if (ArrowGpuBulkEvaluator.isFloat64(root)) {
            Float8Vector output = ArrowBulkEvaluator.allocateOutput(allocator, outputFieldName, rowCount);
            evaluate(root, output, nullPolicy);
            return ArrowFilterSupport.appendColumn(root, output);
        } else {
            Float4Vector output = ArrowBulkEvaluator.allocateOutputF32(allocator, outputFieldName, rowCount);
            evaluate(root, output, nullPolicy);
            return ArrowFilterSupport.appendColumn(root, output);
        }
    }

    /**
     * Convenience overload for {@link #project(VectorSchemaRoot, String,
     * NullPolicy)} using {@link NullPolicy#IGNORE}.
     *
     * @param root Arrow record batch containing the columns referenced by
     * this compiled expression
     * @param outputFieldName name for the new column holding this
     * expression's result; must not already exist in {@code root}
     * @return a new Arrow record batch with {@code root}'s columns plus the
     * projected column, over all of {@code root}'s rows
     */
    default VectorSchemaRoot project(VectorSchemaRoot root, String outputFieldName) {
        return project(root, outputFieldName, NullPolicy.IGNORE);
    }

    /**
     * Fused filter + projection: evaluates <b>this</b> evaluator as a boolean
     * predicate over {@code root} (exactly as {@link #filter} would), then
     * evaluates {@code projection} — a second, independently compiled
     * expression — only over the rows that survived the predicate, appending
     * its result as a new trailing column named {@code outputFieldName}.
     *
     * <pre>
     * Arrow batch (N rows)
     *        |
     *        v
     * this expression, SIMD/GPU predicate over all N rows
     *        |
     *        v
     * selection mask -&gt; row indices (M &lt;= N survive)
     *        |
     *        v
     * gather the M surviving rows (existing columns; one copy pass)
     *        |
     *        v
     * projection expression, SIMD/GPU, evaluated over only the M rows
     *        |
     *        v
     * Arrow batch (M rows, original columns + projected column)
     * </pre>
     *
     * <p>The "fusion" here is deliberate and is the entire point of this
     * method over calling {@link #filter} and then {@link #project}
     * separately: the projection expression never runs over a row that the
     * predicate is going to discard. For an expensive projection and a
     * selective predicate this can be a large win — {@code filter(root)}
     * followed by {@code projectionEvaluator.project(filtered, name)} would
     * pay for row selection once and projection once too, but only after this
     * method's approach of shrinking the batch <i>before</i> projecting; doing
     * the two steps in the other order, or projecting over all N rows before
     * filtering, wastes work on rows that are thrown away.
     *
     * <p>{@code projection} may be compiled against any backend — it does not
     * need to match {@code this} evaluator's backend. A CPU {@code SIMD}
     * predicate can drive a {@code GPU}-evaluated projection, or vice versa;
     * each half of the pipeline dispatches through its own {@code evaluate}
     * independently. Both {@code this} and {@code projection} must, however,
     * each individually be compiled for the precision (float64/float32) that
     * matches {@code root} — see {@link #project}'s precision note, which
     * applies identically here to both halves of the pipeline.
     *
     * <p>Truthiness, and {@link NullPolicy#PROPAGATE} predicate-null
     * semantics for the filtering stage, are identical to {@link #filter}.
     * {@code nullPolicy} is also passed through to the projection stage,
     * governing whether the projected column's validity bitmap propagates
     * nulls from {@code projection}'s own required columns over the
     * surviving rows.
     *
     * <p>If no rows survive the predicate, an empty (zero-row) batch is
     * returned with {@code root}'s original schema plus the projected field —
     * {@code projection} is still invoked, with a zero row count, so that a
     * projection expression with side effects in {@code evaluate} sees a
     * consistent call, but no actual computation occurs.
     *
     * @param root Arrow record batch containing the columns referenced by
     * both {@code this} predicate and {@code projection}
     * @param projection compiled expression evaluated over the rows that
     * pass this evaluator's predicate; must not be {@code null} and must not
     * be closed
     * @param outputFieldName name for the new column holding
     * {@code projection}'s result; must not already exist in {@code root}
     * @param nullPolicy how Arrow validity bitmaps and null predicate values
     * are handled, for both the filtering and projection stages
     * @return a new Arrow record batch containing only the rows for which
     * {@code this} predicate evaluated to true, with {@code root}'s columns
     * plus the projected column
     * @throws NullPointerException if {@code root}, {@code projection},
     * {@code outputFieldName}, or {@code nullPolicy} is null
     * @throws ArrowBindingException if {@code root} has no columns, if
     * {@code root} already has a column named {@code outputFieldName}, or if
     * a required variable's column (for either {@code this} or
     * {@code projection}) is missing or of the wrong vector type
     */
    default VectorSchemaRoot filterProject(
            VectorSchemaRoot root,
            ArrowExpressionEvaluator projection,
            String outputFieldName,
            NullPolicy nullPolicy) {

        if (root == null) {
            throw new NullPointerException("root must not be null");
        }
        if (projection == null) {
            throw new NullPointerException("projection must not be null");
        }
        if (outputFieldName == null) {
            throw new NullPointerException("outputFieldName must not be null");
        }
        if (nullPolicy == null) {
            throw new NullPointerException("nullPolicy must not be null");
        }
        if (root.getVector(outputFieldName) != null) {
            throw new ArrowBindingException(
                    "root already has a column named '" + outputFieldName
                    + "'; choose a different outputFieldName for filterProject(...).");
        }

        int rowCount = root.getRowCount();
        BufferAllocator allocator = ArrowFilterSupport.resolveAllocator(root);
        boolean float64 = ArrowGpuBulkEvaluator.isFloat64(root);

        // --- Stage 1: this evaluator as a SIMD/GPU predicate over all N rows ---
        int[] selected;
        if (rowCount == 0) {
            selected = new int[0];
        } else if (float64) {
            try (Float8Vector predicate = ArrowBulkEvaluator.allocateOutput(
                    allocator, "__parser_ng_filter_project_predicate__", rowCount)) {
                evaluate(root, predicate, nullPolicy);
                selected = ArrowFilterSupport.selectIndices(predicate, nullPolicy);
            }
        } else {
            try (Float4Vector predicate = ArrowBulkEvaluator.allocateOutputF32(
                    allocator, "__parser_ng_filter_project_predicate__", rowCount)) {
                evaluate(root, predicate, nullPolicy);
                selected = ArrowFilterSupport.selectIndices(predicate, nullPolicy);
            }
        }

        // --- Stage 2: gather only the M surviving rows (existing columns) ---
        VectorSchemaRoot filtered = ArrowFilterSupport.materializeSelectedRows(root, selected, allocator);

        // --- Stage 3: SIMD/GPU projection, evaluated over only the M rows ---
        int filteredRowCount = filtered.getRowCount();
        if (float64) {
            Float8Vector output = ArrowBulkEvaluator.allocateOutput(allocator, outputFieldName, filteredRowCount);
            projection.evaluate(filtered, output, nullPolicy);
            return ArrowFilterSupport.appendColumn(filtered, output);
        } else {
            Float4Vector output = ArrowBulkEvaluator.allocateOutputF32(allocator, outputFieldName, filteredRowCount);
            projection.evaluate(filtered, output, nullPolicy);
            return ArrowFilterSupport.appendColumn(filtered, output);
        }
    }

    /**
     * Convenience overload for {@link #filterProject(VectorSchemaRoot,
     * ArrowExpressionEvaluator, String, NullPolicy)} using
     * {@link NullPolicy#IGNORE}.
     *
     * @param root Arrow record batch containing the columns referenced by
     * both {@code this} predicate and {@code projection}
     * @param projection compiled expression evaluated over the rows that
     * pass this evaluator's predicate
     * @param outputFieldName name for the new column holding
     * {@code projection}'s result; must not already exist in {@code root}
     * @return a new Arrow record batch containing only the rows for which
     * {@code this} predicate evaluated to true, with {@code root}'s columns
     * plus the projected column
     */
    default VectorSchemaRoot filterProject(
            VectorSchemaRoot root, ArrowExpressionEvaluator projection, String outputFieldName) {
        return filterProject(root, projection, outputFieldName, NullPolicy.IGNORE);
    }

    /**
     * The variable names this expression requires, in no particular order.
     *
     * @return required variable names
     */
    String[] requiredVariableNames();

    /**
     * Returns {@code true} if this expression references no variables at all,
     * for example a bare numeric literal or a fully constant-folded
     * expression.
     *
     * <p>Both backends handle constant expressions the same way: the
     * GPU/SIMD engine is never touched, and the output is filled directly
     * through the ordinary scalar path.</p>
     *
     * @return {@code true} when the expression is independent of input
     *         columns
     */
    boolean isConstantExpression();

    /**
     * Returns the original expression text represented by this evaluator.
     *
     * @return expression source text
     */
    String getExpressionText();

    /**
     * Identifies the execution backend actually used by this evaluator.
     *
     * <p>For an evaluator compiled with
     * {@link ArrowExecutionBackend#GPU_AUTO}, this reflects whichever
     * concrete backend was actually selected, such as CUDA or OpenCL, rather
     * than returning {@code GPU_AUTO}.</p>
     *
     * @return the actual execution backend
     */
    ArrowExecutionBackend backend();

    /**
     * Releases this evaluator's resources.
     *
     * <p>For the CPU implementation this may release worker-pool resources.
     * For the GPU implementation this may release device buffers, command
     * queues, compiled kernels, or other backend-specific resources.</p>
     *
     * <p>Do not call this method while another thread may still be inside
     * {@link #evaluate(Map, Float8Vector, NullPolicy)},
     * {@link #evaluate(Map, Float4Vector, NullPolicy)},
     * {@link #evaluate(VectorSchemaRoot, Float8Vector, NullPolicy)},
     * {@link #evaluate(VectorSchemaRoot, Float4Vector, NullPolicy)},
     * {@link #filter(VectorSchemaRoot, NullPolicy)},
     * {@link #project(VectorSchemaRoot, String, NullPolicy)}, or
     * {@link #filterProject(VectorSchemaRoot, ArrowExpressionEvaluator, String, NullPolicy)}.</p>
     */
    @Override
    void close();
}