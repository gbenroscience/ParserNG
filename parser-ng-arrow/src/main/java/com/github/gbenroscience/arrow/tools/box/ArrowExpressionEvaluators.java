package com.github.gbenroscience.arrow.tools.box;

import com.github.gbenroscience.gpu.GpuBackend;
import com.github.gbenroscience.parser.MathExpression;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import org.apache.arrow.memory.RootAllocator;

/**
 * Single entry point for compiling an {@link ArrowExpressionEvaluator} against
 * whichever backend you choose. This is the intended way to switch between the
 * CPU SIMD engine ({@link ArrowBulkEvaluator}) and a GPU backend
 * ({@link ArrowGpuBulkEvaluator}, CUDA or OpenCL) — pick an
 * {@link ArrowExecutionBackend} value rather than constructing either concrete
 * class directly, so changing backends later is a one-line change here instead
 * of a rewrite at every call site.
 *
 * <h2>Fixed backend vs "prefer GPU, fall back to CPU"</h2>
 * {@link #compile(MathExpression, ArrowExecutionBackend)} compiles against
 * exactly the backend you name and throws if that backend can't be bootstrapped
 * — including {@link ArrowExecutionBackend#GPU_AUTO}, which still throws if NO
 * GPU backend (CUDA or OpenCL) is available; it does not fall back to the CPU.
 * {@link #compilePreferGpu} is the method that actually falls back: it tries
 * {@code GPU_AUTO} first and only compiles the CPU SIMD evaluator if every GPU
 * backend fails to bootstrap. Use the fixed form when the caller has already
 * decided (e.g. from configuration, or after checking
 * {@link ArrowGpuBulkEvaluator#isBackendAvailable}); use
 * {@code compilePreferGpu} for a simple "use the GPU if there is one" default.
 *
 * <h2>Device selection</h2>
 * Selecting a specific GPU device (as opposed to just a backend) is
 * backend-specific and happens on {@link ArrowGpuBulkEvaluator} itself, before
 * compiling, since it isn't part of the portable
 * {@link ArrowExpressionEvaluator} surface — see
 * {@link ArrowGpuBulkEvaluator#listOpenClDevices()},
 * {@link ArrowGpuBulkEvaluator#selectOpenClDevice(String)}, and
 * {@link ArrowGpuBulkEvaluator#selectCudaDeviceIndex(int)}. Call the relevant
 * selection method, then compile through this class as usual — the selection is
 * picked up by whichever backend actually runs.
 *
 * <h2>Ad-hoc filter + multi-column projection: {@code filterProject}</h2>
 * {@link #filterProject(VectorSchemaRoot, String, ArrowExecutionBackend,
 * NullPolicy, String...)} is a SQL-{@code WHERE}/{@code SELECT}-shaped
 * convenience over the {@link ArrowExpressionEvaluator#filterProject} instance
 * method: instead of a pre-compiled predicate evaluator and a single
 * pre-compiled projection evaluator producing one appended column, this takes
 * the predicate and every output column as raw ParserNG expression text,
 * compiles all of them internally, and returns a batch containing <i>only</i>
 * the requested output columns — not the original schema plus one extra column.
 * For example:
 * <pre>
 * VectorSchemaRoot result = ArrowExpressionEvaluators.filterProject(
 *         root,
 *         "x &gt; 10 &amp;&amp; y &lt; 20",
 *         "x", "y", "x * y", "sqrt(x*x + y*y)");
 * </pre> filters {@code root} down to the rows where the predicate holds, and
 * returns a four-column batch: {@code x} and {@code y} passed straight through
 * unchanged (recognized as bare column references and never routed through the
 * expression engine at all — zero-copy), plus two freshly computed columns
 * named after their own expression text, {@code "x * y"} and
 * {@code "sqrt(x*x + y*y)"}, evaluated only over the rows that survived the
 * predicate.
 *
 * <p>
 * This method is the right tool for one-shot, ad-hoc queries where the
 * predicate/projection text is only known at the call site (e.g. built from
 * user input) and isn't worth pre-compiling and holding onto. If the same
 * predicate and the same projections will run against many batches, compile
 * once via {@link #compile} and reuse {@link ArrowExpressionEvaluator}
 * instances through {@link ArrowExpressionEvaluator#filterProject} instead —
 * that avoids recompiling every expression on every call, which this method
 * always does.</p>
 */
public final class ArrowExpressionEvaluators {

    private ArrowExpressionEvaluators() {
    }

    /**
     * Compiles {@code expr} against {@code backend}.
     *
     * @param expr
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compile(String expr, ArrowExecutionBackend backend) throws Throwable {
        return compile(new MathExpression(expr), backend);
    }

    /**
     * Compiles an already-constructed {@link MathExpression} against
     * {@code backend}.
     *
     * @param expression
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compile(MathExpression expression, ArrowExecutionBackend backend) throws Throwable {
        if (backend == null) {
            throw new NullPointerException("backend must not be null");
        }
        switch (backend) {
            case CPU_SIMD -> {
                return ArrowBulkEvaluator.compile(expression, 0);
            }
            case GPU_AUTO -> {
                return ArrowGpuBulkEvaluator.compile(expression);
            }
            case GPU_CUDA -> {
                return ArrowGpuBulkEvaluator.compile(expression, GpuBackend.CUDA);
            }
            case GPU_OPENCL -> {
                return ArrowGpuBulkEvaluator.compile(expression, GpuBackend.OPENCL);
            }
            case GPU_METAL -> {
                return ArrowGpuBulkEvaluator.compile(expression, GpuBackend.METAL);
            }
            default -> // Unreachable unless ArrowExecutionBackend grows a new constant
                // without a matching case here.
                throw new IllegalArgumentException("Unhandled ArrowExecutionBackend: " + backend);
        }
    }

    /**
     * Compiles {@code expr} against {@link ArrowExecutionBackend#GPU_AUTO},
     * falling back to {@link ArrowExecutionBackend#CPU_SIMD} if no GPU backend
     * bootstraps on this machine at all. The most common choice for "use the
     * GPU when there is one, otherwise just work".
     *
     * <p>
     * If both the GPU attempt and the CPU fallback fail, the CPU exception is
     * thrown with the original GPU failure attached via
     * {@link Throwable#addSuppressed}, so the real GPU bootstrap error isn't
     * lost — a fallback CPU compile failure almost always means the expression
     * itself is invalid, not that the CPU engine is unavailable, so that's the
     * more informative exception to surface as primary.
     *
     * @param expr
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compilePreferGpu(String expr) throws Throwable {
        return compilePreferGpu(new MathExpression(expr));
    }

    /**
     * Compiles an already-constructed {@link MathExpression} against
     * {@link ArrowExecutionBackend#GPU_AUTO}, falling back to
     * {@link ArrowExecutionBackend#CPU_SIMD} if no GPU backend bootstraps at
     * all. See {@link #compilePreferGpu(String)} for the fallback/exception
     * behavior.
     *
     * @param expression
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compilePreferGpu(MathExpression expression) throws Throwable {
        try {
            return ArrowGpuBulkEvaluator.compile(expression);
        } catch (Throwable gpuFailure) {
            try {
                return ArrowBulkEvaluator.compile(expression, 0);
            } catch (Throwable cpuFailure) {
                cpuFailure.addSuppressed(gpuFailure);
                throw cpuFailure;
            }
        }
    }

    /**
     * Compiles an already-constructed {@link MathExpression} against
     * {@code backend}.
     *
     * @param expression
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compileF32(MathExpression expression) throws Throwable {
        return compileF32(expression, ArrowExecutionBackend.CPU_SIMD);
    }

    public static ArrowExpressionEvaluator compileF32(String expression) throws Throwable {
        return compileF32(new MathExpression(expression));
    }

    /**
     * Compiles {@code expr} as a float32 expression against {@code backend}.
     *
     * <p>This is the float32 counterpart to
     * {@link #compile(String, ArrowExecutionBackend)} — for
     * {@link ArrowExecutionBackend#CPU_SIMD} it compiles a genuinely different
     * kernel from {@link #compile} (SIMD lane width is precision-specific, see
     * {@link ArrowBulkEvaluator}'s class javadoc), while for any GPU backend
     * value it compiles via {@link ArrowGpuBulkEvaluator#compileF32}, whose
     * resulting kernel supports float32 dispatch directly (see that class's
     * "Type support" section for why a GPU-compiled kernel serves both
     * precisions).
     *
     * @param expr
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compileF32(String expr, ArrowExecutionBackend backend) throws Throwable {
        return compileF32(new MathExpression(expr), backend);
    }

    /**
     * Compiles an already-constructed {@link MathExpression} as a float32
     * expression against {@code backend}. See
     * {@link #compileF32(String, ArrowExecutionBackend)}.
     *
     * @param expression
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compileF32(MathExpression expression, ArrowExecutionBackend backend) throws Throwable {
        if (backend == null) {
            throw new NullPointerException("backend must not be null");
        }
        switch (backend) {
            case CPU_SIMD -> {
                return ArrowBulkEvaluator.compileF32(expression, 0);
            }
            case GPU_AUTO -> {
                return ArrowGpuBulkEvaluator.compileF32(expression);
            }
            case GPU_CUDA -> {
                return ArrowGpuBulkEvaluator.compileF32(expression, GpuBackend.CUDA);
            }
            case GPU_OPENCL -> {
                return ArrowGpuBulkEvaluator.compileF32(expression, GpuBackend.OPENCL);
            }
            case GPU_METAL -> {
                return ArrowGpuBulkEvaluator.compileF32(expression, GpuBackend.METAL);
            }
            default -> // Unreachable unless ArrowExecutionBackend grows a new constant
                // without a matching case here.
                throw new IllegalArgumentException("Unhandled ArrowExecutionBackend: " + backend);
        }
    }

    // =========================================================================
    // Ad-hoc filter + multi-column projection (SQL WHERE/SELECT shaped)
    // =========================================================================
    /**
     * {@link #filterProject(VectorSchemaRoot, String, ArrowExecutionBackend,
     * NullPolicy, String...)} with {@link ArrowExecutionBackend#CPU_SIMD} and
     * {@link NullPolicy#IGNORE}.
     *
     * @param root Arrow record batch to filter and project
     * @param predicateExpr ParserNG boolean expression selecting rows
     * @param projections one or more ParserNG expressions; a projection that is
     * exactly an existing column's name is passed through unchanged, any other
     * expression is compiled and evaluated as a new column named after its own
     * (trimmed) expression text
     * @return a new batch containing exactly the requested projection columns,
     * over exactly the rows selected by {@code predicateExpr}
     * @throws java.lang.Throwable if {@code predicateExpr} or any projection
     * expression fails to compile, or if evaluation fails
     */
    public static VectorSchemaRoot filterProject(
            VectorSchemaRoot root,
            String predicateExpr,
            String... projections) throws Throwable {
        return filterProject(root, predicateExpr, ArrowExecutionBackend.CPU_SIMD, NullPolicy.IGNORE, projections);
    }

    /**
     * {@link #filterProject(VectorSchemaRoot, String, ArrowExecutionBackend,
     * NullPolicy, String...)} with {@link NullPolicy#IGNORE}.
     *
     * @param root Arrow record batch to filter and project
     * @param predicateExpr ParserNG boolean expression selecting rows
     * @param backend execution backend used to compile the predicate and every
     * non-passthrough projection expression
     * @param projections one or more ParserNG expressions — see
     * {@link #filterProject(VectorSchemaRoot, String, String...)}
     * @return a new batch containing exactly the requested projection columns,
     * over exactly the rows selected by {@code predicateExpr}
     * @throws java.lang.Throwable if {@code predicateExpr} or any projection
     * expression fails to compile, or if evaluation fails
     */
    public static VectorSchemaRoot filterProject(
            VectorSchemaRoot root,
            String predicateExpr,
            ArrowExecutionBackend backend,
            String... projections) throws Throwable {
        return filterProject(root, predicateExpr, backend, NullPolicy.IGNORE, projections);
    }

    /**
     * {@link #filterProject(VectorSchemaRoot, String, ArrowExecutionBackend,
     * NullPolicy, String...)} with {@link ArrowExecutionBackend#CPU_SIMD}.
     *
     * @param root Arrow record batch to filter and project
     * @param predicateExpr ParserNG boolean expression selecting rows
     * @param nullPolicy how Arrow validity bitmaps are handled, for both the
     * filtering stage and every computed projection column
     * @param projections one or more ParserNG expressions — see
     * {@link #filterProject(VectorSchemaRoot, String, String...)}
     * @return a new batch containing exactly the requested projection columns,
     * over exactly the rows selected by {@code predicateExpr}
     * @throws java.lang.Throwable if {@code predicateExpr} or any projection
     * expression fails to compile, or if evaluation fails
     */
    public static VectorSchemaRoot filterProject(
            VectorSchemaRoot root,
            String predicateExpr,
            NullPolicy nullPolicy,
            String... projections) throws Throwable {
        return filterProject(root, predicateExpr, ArrowExecutionBackend.CPU_SIMD, nullPolicy, projections);
    }

    /**
     * Filters {@code root} by {@code predicateExpr}, then evaluates every
     * expression in {@code projections} — only over the rows that survived the
     * filter — returning a new batch containing exactly those output columns,
     * in the order given, and none of {@code root}'s other columns.
     *
     * <pre>
     * Arrow batch root (N rows, arbitrary columns)
     *        |
     *        v
     * compile predicateExpr -&gt; SIMD/GPU predicate over all N rows
     *        |
     *        v
     * selection mask -&gt; row indices (M &lt;= N survive)
     *        |
     *        v
     * gather the M surviving rows of every column in root
     * (needed as expression inputs; not all are necessarily returned)
     *        |
     *        v
     * for each projection expression, over only the M rows:
     *   - bare column name (e.g. "x")   -&gt; pass the existing column through
     *   - any other expression          -&gt; compile it, evaluate it, name
     *                                       the output column after its own
     *                                       (trimmed) expression text
     *        |
     *        v
     * Arrow batch (M rows, exactly len(projections) columns)
     * </pre>
     *
     * <h2>Passthrough vs. computed projections</h2>
     * A projection string is treated as a <b>passthrough</b> — the existing
     * column reused directly, with no expression compiled and no SIMD/GPU work
     * performed for it — if and only if, after trimming, it exactly matches the
     * name of a column already present in {@code root}. Every other projection
     * string is compiled as a ParserNG expression and evaluated; the resulting
     * column is named after the exact (trimmed) expression text, mirroring how
     * SQL engines label an unaliased computed {@code SELECT} column after its
     * own expression text.
     *
     * <p>
     * Because passthrough detection is a literal name match, a projection
     * string that happens to equal an existing column's name is always treated
     * as that column, even if it could also be parsed as an expression (this
     * can only happen for single-variable column names, since ParserNG
     * expression text containing any operator cannot equal a bare identifier).
     * This mirrors ordinary SQL {@code SELECT x} behavior.
     *
     * <h2>Memory</h2>
     * Every column of {@code root} that survives filtering is materialized once
     * (to serve as input to whichever projections need it), but only the
     * columns actually named in {@code projections} — passthrough or computed —
     * are retained in the returned batch; every other filtered column is closed
     * before this method returns, so no off-heap Arrow memory is held onto for
     * columns the caller didn't ask for. If this method throws partway through
     * evaluating {@code projections}, every intermediate buffer it had
     * allocated up to that point (the filtered batch and any already-computed
     * projection columns) is released before the exception propagates.
     *
     * <h2>Precision</h2>
     * As with {@link ArrowExpressionEvaluator#filter} and
     * {@link ArrowExpressionEvaluator#project}, float64 vs. float32 is inferred
     * from {@code root}'s own schema (all-{@code Float8Vector} columns evaluate
     * as float64; anything else is treated as float32). Every projection
     * expression — and the predicate — is compiled for whichever precision
     * that turns out to be, <i>against the same {@code backend} the caller
     * requested</i>: a float32 {@code root} with a GPU backend compiles and
     * dispatches on the GPU, not silently on the CPU (via
     * {@link #compileF32(String, ArrowExecutionBackend)}).
     *
     * <h2>Backend, lifecycle</h2>
     * The predicate and every non-passthrough projection are compiled fresh
     * against {@code backend} on every call and closed again before this method
     * returns — nothing from this call is left open afterward except the
     * returned {@code VectorSchemaRoot} itself, which the caller owns and must
     * eventually close. For repeated use of the same predicate/projections
     * across many batches, compile once via {@link #compile} and call
     * {@link ArrowExpressionEvaluator#filterProject} on the reusable instances
     * instead — see that method's javadoc for the corresponding fused
     * predicate/projection pipeline built from already-compiled evaluators.
     *
     * @param root Arrow record batch to filter and project
     * @param predicateExpr ParserNG boolean expression selecting rows;
     * interpreted with the same C-style truthiness and {@link NullPolicy}
     * semantics as {@link ArrowExpressionEvaluator#filter}
     * @param backend execution backend used to compile the predicate and every
     * non-passthrough projection expression
     * @param nullPolicy how Arrow validity bitmaps are handled, for both the
     * filtering stage and every computed projection column
     * @param projections one or more ParserNG expressions naming the output
     * columns, in order; must contain at least one entry
     * @return a new batch containing exactly the requested projection columns,
     * over exactly the rows selected by {@code predicateExpr}
     * @throws NullPointerException if {@code root}, {@code predicateExpr},
     * {@code backend}, or {@code nullPolicy} is null, or if any individual
     * projection string is null
     * @throws IllegalArgumentException if {@code projections} is null, empty,
     * or contains a blank string
     * @throws ArrowBindingException if {@code root} has no columns, or if a
     * required variable's column (for the predicate or any projection) is
     * missing or of the wrong vector type
     * @throws java.lang.Throwable if {@code predicateExpr} or any projection
     * expression fails to compile, or if evaluation fails
     */
    public static VectorSchemaRoot filterProject(
            VectorSchemaRoot root,
            String predicateExpr,
            ArrowExecutionBackend backend,
            NullPolicy nullPolicy,
            String... projections) throws Throwable {

        if (root == null) {
            throw new NullPointerException("root must not be null");
        }
        if (predicateExpr == null) {
            throw new NullPointerException("predicateExpr must not be null");
        }
        if (backend == null) {
            throw new NullPointerException("backend must not be null");
        }
        if (nullPolicy == null) {
            throw new NullPointerException("nullPolicy must not be null");
        }
        if (projections == null || projections.length == 0) {
            throw new IllegalArgumentException(
                    "filterProject requires at least one projection expression");
        }

        BufferAllocator allocator = ArrowFilterSupport.resolveAllocator(root);
        boolean float64 = ArrowGpuBulkEvaluator.isFloat64(root);
        int rowCount = root.getRowCount();

        ArrowExpressionEvaluator predicate = null;
        try {
            // --- Stage 1: SIMD/GPU predicate over all N rows -> row indices ---
            int[] selected;
            if (rowCount == 0) {
                selected = new int[0];
            } else if (float64) {
                predicate = compile(predicateExpr, backend);
                try (Float8Vector predicateOut = ArrowBulkEvaluator.allocateOutput(
                        allocator, "__parser_ng_filter_project_predicate__", rowCount)) {
                    predicate.evaluate(root, predicateOut, nullPolicy);
                    selected = ArrowFilterSupport.selectIndices(predicateOut, nullPolicy);
                }
            } else {
                predicate = compileF32(predicateExpr, backend);
                try (Float4Vector predicateOut = ArrowBulkEvaluator.allocateOutputF32(
                        allocator, "__parser_ng_filter_project_predicate__", rowCount)) {
                    predicate.evaluate(root, predicateOut, nullPolicy);
                    selected = ArrowFilterSupport.selectIndices(predicateOut, nullPolicy);
                }
            }

            // --- Stage 2: gather every column of root for the M surviving rows ---
            VectorSchemaRoot filtered = ArrowFilterSupport.materializeSelectedRows(root, selected, allocator);
            int filteredRowCount = filtered.getRowCount();

            List<Field> outFields = new ArrayList<>(projections.length);
            List<FieldVector> outVectors = new ArrayList<>(projections.length);
            List<FieldVector> computedOutputs = new ArrayList<>(projections.length);
            List<ArrowExpressionEvaluator> compiledProjections = new ArrayList<>(projections.length);
            Set<FieldVector> reused = Collections.newSetFromMap(new IdentityHashMap<>());

            try {
                // --- Stage 3: each projection, over only the M surviving rows ---
                for (String rawProjection : projections) {
                    if (rawProjection == null) {
                        throw new NullPointerException("Projection expression must not be null");
                    }
                    String projExpr = rawProjection.trim();
                    if (projExpr.isEmpty()) {
                        throw new IllegalArgumentException("Projection expression must not be blank");
                    }

                    FieldVector existing = filtered.getVector(projExpr);
                    if (existing != null) {
                        // Bare column reference: passthrough, zero-copy, no
                        // expression compiled at all.
                        outFields.add(existing.getField());
                        outVectors.add(existing);
                        reused.add(existing);
                        continue;
                    }

                    FieldVector output;
                    if (float64) {
                        ArrowExpressionEvaluator projectionEvaluator = compile(projExpr, backend);
                        compiledProjections.add(projectionEvaluator);
                        Float8Vector out8 = ArrowBulkEvaluator.allocateOutput(allocator, projExpr, filteredRowCount);
                        // Track for rollback BEFORE evaluate() runs -- if evaluate()
                        // throws, this vector must still be in computedOutputs for the
                        // catch block below to close it, or it leaks off-heap memory.
                        computedOutputs.add(out8);
                        projectionEvaluator.evaluate(filtered, out8, nullPolicy);
                        output = out8;
                    } else {
                        ArrowExpressionEvaluator projectionEvaluator = compileF32(projExpr, backend);
                        compiledProjections.add(projectionEvaluator);
                        Float4Vector out4 = ArrowBulkEvaluator.allocateOutputF32(allocator, projExpr, filteredRowCount);
                        computedOutputs.add(out4);
                        projectionEvaluator.evaluate(filtered, out4, nullPolicy);
                        output = out4;
                    }
                    outFields.add(output.getField());
                    outVectors.add(output);
                    reused.add(output);
                }
            } catch (Throwable t) {
                // Roll back everything this call had allocated so far: every
                // column of the filtered batch (passthrough or not), and any
                // computed projection columns already built before the
                // failure. Disjoint sets by construction, so nothing here is
                // closed twice.
                for (FieldVector v : filtered.getFieldVectors()) {
                    closeQuietly(v);
                }
                for (FieldVector v : computedOutputs) {
                    closeQuietly(v);
                }
                for (ArrowExpressionEvaluator ev : compiledProjections) {
                    ev.close();
                }
                throw t;
            }

            for (ArrowExpressionEvaluator ev : compiledProjections) {
                ev.close();
            }

            // Release the filtered columns nobody actually asked for in
            // `projections` -- they were only materialized as potential
            // expression inputs, and would otherwise leak off-heap memory.
            for (FieldVector v : filtered.getFieldVectors()) {
                if (!reused.contains(v)) {
                    closeQuietly(v);
                }
            }

            return new VectorSchemaRoot(new Schema(outFields), outVectors, filteredRowCount);
        } finally {
            if (predicate != null) {
                predicate.close();
            }
        }
    }

    private static void closeQuietly(FieldVector vector) {
        try {
            vector.close();
        } catch (Throwable ignored) {
            // Best-effort cleanup during error unwind -- the exception that
            // triggered the rollback is what the caller needs to see, not a
            // secondary failure from closing an already-released buffer.
        }
    }

    private static VectorSchemaRoot createFloatRoot(
            float[] x,
            float[] y) {

        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);

        Float4Vector xVector = new Float4Vector("x", allocator);
        Float4Vector yVector = new Float4Vector("y", allocator);

        xVector.allocateNew(x.length);
        yVector.allocateNew(y.length);

        for (int i = 0; i < x.length; i++) {
            xVector.setSafe(i, x[i]);
            yVector.setSafe(i, y[i]);
        }

        xVector.setValueCount(x.length);
        yVector.setValueCount(y.length);

        return VectorSchemaRoot.of(xVector, yVector);
    }

    public static void printVectorSchemaRoot(VectorSchemaRoot root) {
        int rowCount = root.getRowCount();
        List<FieldVector> vectors = root.getFieldVectors();

        // Determine column widths from headers and values.
        int columnCount = vectors.size();
        String[] headers = new String[columnCount];
        int[] widths = new int[columnCount];

        for (int c = 0; c < columnCount; c++) {
            headers[c] = vectors.get(c).getName();
            widths[c] = headers[c] == null ? 4 : headers[c].length();
        }

        String[][] values = new String[rowCount][columnCount];

        for (int r = 0; r < rowCount; r++) {
            for (int c = 0; c < columnCount; c++) {
                Object value = vectors.get(c).getObject(r);
                String text = value == null ? "null" : String.valueOf(value);
                values[r][c] = text;
                widths[c] = Math.max(widths[c], text.length());
            }
        }

        // Separator
        StringBuilder separator = new StringBuilder("+");
        for (int c = 0; c < columnCount; c++) {
            separator.append("-".repeat(widths[c] + 2)).append("+");
        }

        System.out.println();
        System.out.println("VectorSchemaRoot: "
                + rowCount + " rows x " + columnCount + " columns");
        System.out.println(separator);

        // Header
        StringBuilder header = new StringBuilder("|");
        for (int c = 0; c < columnCount; c++) {
            header.append(" ")
                    .append(padRight(headers[c], widths[c]))
                    .append(" |");
        }
        System.out.println(header);
        System.out.println(separator);

        // Rows
        for (int r = 0; r < rowCount; r++) {
            StringBuilder row = new StringBuilder("|");

            for (int c = 0; c < columnCount; c++) {
                row.append(" ")
                        .append(padRight(values[r][c], widths[c]))
                        .append(" |");
            }

            System.out.println(row);
        }

        System.out.println(separator);
    }

    private static String padRight(String value, int width) {
        return value + " ".repeat(Math.max(0, width - value.length()));
    }

    public static void main(String[] args) {
        float[] xArr = new float[1000];
        float[] yArr = new float[1000]; 

        for (int i = 0; i < xArr.length; i++) {
            xArr[i] = i;
            yArr[i] = 2*i+1;
        }
        VectorSchemaRoot root = createFloatRoot(xArr, yArr);
        printVectorSchemaRoot(root);
        try {
            VectorSchemaRoot vsr = filterProject(root, "x<500", "x", "y", "if(sin(x) > 0, tan(x), 0.2)", "sin(x)", "x^3");
            printVectorSchemaRoot(vsr);
        } catch (Throwable ex) {
            System.getLogger(ArrowExpressionEvaluators.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }
    }

}