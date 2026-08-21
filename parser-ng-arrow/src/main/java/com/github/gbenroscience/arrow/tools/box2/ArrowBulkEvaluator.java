package com.github.gbenroscience.arrow.tools.box2;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simdext.turbo.tools.SIMDEngineEvaluator;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Evaluates a compiled ParserNG expression directly over Apache Arrow
 * columnar batches, using {@link SIMDEngineEvaluator}'s zero-copy
 * {@code MemorySegment[]}-backed bulk evaluation path — no coalescing copy,
 * no on-heap staging buffer, no per-row boxing.
 *
 * <h2>Binding model</h2>
 * Variables in the expression are bound to Arrow columns <b>by name</b>.
 * The authoritative name-to-slot mapping comes from
 * {@link MathExpression#getSlotItems()}, which reflects exactly the
 * variables the compiled expression actually references and the frame
 * index each one occupies — this class does not guess at slot ordering.
 *
 * <h2>Type support</h2>
 * Only {@link Float8Vector} (Arrow's float64 column type) columns are
 * supported for zero-copy binding, since {@code SIMDEngineEvaluator}
 * operates on {@code double} throughout. Columns of other numeric types
 * must be cast to float64 by the caller before binding — this class
 * deliberately does not perform an implicit narrowing/widening copy, since
 * doing so silently would reintroduce the exact copy this module exists to
 * eliminate.
 *
 * <h2>Constant expressions</h2>
 * An expression that references no variables (e.g. {@code "42.0"}) compiles
 * to a zero-slot evaluator. {@code SIMDEngineEvaluator.applyBulk(MemorySegment[], ...)}
 * treats a zero-length variable array as a no-op by design (see its internal
 * guard clause) — left unhandled, that would silently leave the output
 * buffer untouched. This class detects that case up front and fills the
 * output directly instead.
 *
 * <h2>Thread safety</h2>
 * A single {@code ArrowBulkEvaluator} instance may be shared and called
 * concurrently from multiple threads and will always produce correct
 * results, but the two evaluation modes differ in how much actual
 * concurrency you get: calls with {@code parallel=false} run fully
 * concurrently (the underlying evaluator uses a {@code ThreadLocal} context
 * per caller thread), while calls with {@code parallel=true} (the default)
 * are internally serialized against each other —
 * {@code SIMDVectorCompositeExpression}'s worker-pool dispatch uses a single
 * shared coordination structure that is only safe for one external caller at
 * a time, so concurrent parallel calls queue rather than overlap. If you
 * need true concurrent parallel evaluation from multiple threads, give each
 * thread its own {@code ArrowBulkEvaluator} (a separate {@link #compile}
 * call) rather than sharing one. The one hard exception either way is
 * {@link #close()}: do not call it while another thread may still be inside
 * {@link #evaluate}.
 *
 * <h2>Lifecycle</h2>
 * Call {@link #close()} when done — this shuts down the evaluator's
 * CPU-pinned worker thread pool (if one was created). This class implements
 * {@link AutoCloseable} for try-with-resources use.
 */
public final class ArrowBulkEvaluator implements AutoCloseable {

    private final MathExpression expression;
    private final SIMDEngineEvaluator.SIMDVectorCompositeExpression compiled;
    private final MathExpression.Slot[] requiredSlots;
    private final String[] requiredVariableNames;
    private final int slotCount;
    private final boolean constantExpression;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    // SIMDVectorCompositeExpression's worker-pool dispatch (used by the
    // parallel=true path) is only safe for one external caller at a time —
    // see the class javadoc. This lock is what actually enforces that
    // "queue rather than overlap" contract; without it, concurrent parallel
    // calls race on the engine's shared coordination structure instead of
    // being serialized.
    private final Object parallelLock = new Object();

    private ArrowBulkEvaluator(MathExpression expression,
                                SIMDEngineEvaluator.SIMDVectorCompositeExpression compiled) {
        this.expression = expression;
        this.compiled = compiled;
        this.requiredSlots = expression.getSlotItems();
        this.slotCount = expression.getRegistry().size();
        this.constantExpression = requiredSlots.length == 0;

        String[] names = new String[requiredSlots.length];
        for (int i = 0; i < requiredSlots.length; i++) {
            names[i] = requiredSlots[i].getName();
        }
        this.requiredVariableNames = names;
    }

    // =========================================================================
    // Compilation entry points
    // =========================================================================

    /**
     * Compiles {@code expr} with the default worker configuration.
     * @param expr
     * @return 
     * @throws java.lang.Throwable
     */
    public static ArrowBulkEvaluator compile(String expr) throws Throwable {
        return compile(new MathExpression(expr), 0);
    }

    /**
     * Compiles {@code expr} with an explicit CPU-pinned worker count for
     * {@link #evaluate}'s parallel path. Pass {@code 0} for the library
     * default.
     * @param expr
     * @param numWorkers
     * @return 
     * @throws java.lang.Throwable
     */
    public static ArrowBulkEvaluator compile(String expr, int numWorkers) throws Throwable {
        return compile(new MathExpression(expr), numWorkers);
    }

    /**
     * Compiles an already-constructed {@link MathExpression}. Useful when
     * the caller needs to inspect or configure the expression (e.g. via
     * {@link MathExpression#setWillFoldConstants(boolean)}) before compiling.
     * @param expression
     * @param numWorkers
     * @return 
     * @throws java.lang.Throwable 
     */
    public static ArrowBulkEvaluator compile(MathExpression expression, int numWorkers) throws Throwable {
        SIMDEngineEvaluator.SIMDVectorCompositeExpression raw = numWorkers > 0
            ? (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(expression, numWorkers).compile()
            : SIMDEngineEvaluator.getEvaluator(expression);
        return new ArrowBulkEvaluator(expression, raw);
    }

    // =========================================================================
    // Introspection
    // =========================================================================

    /**
     * The variable names this expression requires, in no particular order.
     * Every one of these must have a corresponding Arrow column bound at
     * evaluation time, or {@link #evaluate} throws {@link ArrowBindingException}.
     * @return 
     */
    public String[] requiredVariableNames() {
        return requiredVariableNames.clone();
    }

    /**
     * True if this expression references no variables at all (e.g. a bare
     * numeric literal or a fully constant-folded expression). Such
     * expressions still evaluate correctly via {@link #evaluate} — the
     * output is filled with the single constant value — but take a
     * different, non-SIMD internal path documented on {@link #evaluate}.
     * @return 
     */
    public boolean isConstantExpression() {
        return constantExpression;
    }

    public String getExpressionText() {
        return expression.getExpression();
    }

    // =========================================================================
    // Evaluation — Map<String, Float8Vector> binding
    // =========================================================================

    public void evaluate(Map<String, Float8Vector> columns, Float8Vector output) {
        evaluate(columns, output, NullPolicy.IGNORE, true);
    }

    public void evaluate(Map<String, Float8Vector> columns, Float8Vector output, NullPolicy nullPolicy) {
        evaluate(columns, output, nullPolicy, true);
    }

    /**
     * Evaluates the compiled expression, writing one result per row into
     * {@code output}.
     *
     * <p><b>Precondition:</b> {@code output} must already be sized —
     * {@code output.allocateNew(rowCount)} and {@code output.setValueCount(rowCount)}
     * must have been called before this method, where {@code rowCount} is
     * the number of rows to evaluate. Use {@link #allocateOutput} if you
     * don't already have an output vector prepared.
     *
     * @param columns    Arrow columns, keyed by the variable name they bind
     *                   to. Must contain an entry for every name in
     *                   {@link #requiredVariableNames()}; extra entries are
     *                   ignored. Every bound column's {@code getValueCount()}
     *                   must be at least {@code output.getValueCount()}.
     * @param output     pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled — see {@link NullPolicy}
     * @param parallel   if true, dispatches to the evaluator's CPU-pinned
     *                   worker pool for large batches (recommended for
     *                   standalone calls); pass false if this call is
     *                   already running inside the caller's own worker
     *                   thread and nested parallelism should be avoided
     * @throws ArrowBindingException if a required column is missing, a bound
     *                                column is shorter than the output, or
     *                                the output has not been sized
     */
    public void evaluate(Map<String, Float8Vector> columns, Float8Vector output, NullPolicy nullPolicy, boolean parallel) {
        ensureOpen();

        int rowCount = output.getValueCount();
        if (rowCount == 0) {
            // A rowCount of 0 is ambiguous on its own: it's either a legitimately
            // empty batch (output correctly sized to zero rows, e.g. via
            // allocateOutput(allocator, name, 0)) or an output vector that was
            // never sized at all (constructed but allocateNew()/setValueCount()
            // never called). Disambiguate by checking whether any bound column
            // actually has rows to evaluate — if it does, the output clearly
            // wasn't sized to match, which is exactly the caller mistake this
            // exception exists to catch.
            for (MathExpression.Slot slot : requiredSlots) {
                Float8Vector col = columns.get(slot.getName());
                if (col != null && col.getValueCount() > 0) {
                    throw new ArrowBindingException(
                        "Output vector has not been sized (valueCount=0) but bound column '"
                            + slot.getName() + "' has " + col.getValueCount() + " rows. Call "
                            + "output.allocateNew(rowCount) and output.setValueCount(rowCount) "
                            + "before evaluate(), or use allocateOutput(...).");
                }
            }
            return; // legitimately empty batch, nothing further to do
        }

        if (constantExpression) {
            fillConstant(output, rowCount);
            return;
        }

        MemorySegment[] variables = new MemorySegment[slotCount];
        for (MathExpression.Slot slot : requiredSlots) {
            Float8Vector col = columns.get(slot.getName());
            if (col == null) {
                throw new ArrowBindingException(
                    "Missing Arrow column for variable '" + slot.getName() + "'. Required variables: "
                        + Arrays.toString(requiredVariableNames));
            }
            if (col.getValueCount() < rowCount) {
                throw new ArrowBindingException(
                    "Column '" + slot.getName() + "' has " + col.getValueCount()
                        + " rows, but output expects " + rowCount + " rows.");
            }
            variables[slot.getSlot()] = ArrowMemoryBridge.wrapDoubles(col.getDataBuffer(), rowCount);
        }

        MemorySegment outSeg = ArrowMemoryBridge.wrapDoubles(output.getDataBuffer(), rowCount);

        if (parallel) {
            synchronized (parallelLock) {
                compiled.applyBulkParallel(variables, outSeg);
            }
        } else {
            compiled.applyBulk(variables, outSeg);
        }

        if (nullPolicy == NullPolicy.PROPAGATE) {
            propagateNulls(columns, output, rowCount);
        }
    }

    // =========================================================================
    // Evaluation — VectorSchemaRoot convenience binding
    // =========================================================================

    public void evaluate(VectorSchemaRoot root, Float8Vector output) {
        evaluate(root, output, NullPolicy.IGNORE, true);
    }

    public void evaluate(VectorSchemaRoot root, Float8Vector output, NullPolicy nullPolicy) {
        evaluate(root, output, nullPolicy, true);
    }

    /**
     * Convenience overload that resolves each required variable's column by
     * name from {@code root} instead of a caller-built {@code Map}.
     *
     * @param root
     * @param output
     * @param nullPolicy
     * @param parallel
     * @throws ArrowBindingException if a required field is missing from
     *                                {@code root}, or is present but is not
     *                                a {@link Float8Vector}
     */
    public void evaluate(VectorSchemaRoot root, Float8Vector output, NullPolicy nullPolicy, boolean parallel) {
        ensureOpen();

        Map<String, Float8Vector> columns = new HashMap<>(Math.max(4, requiredSlots.length * 2));
        for (MathExpression.Slot slot : requiredSlots) {
            FieldVector fv = root.getVector(slot.getName());
            if (fv == null) {
                continue; // reported uniformly by evaluate(Map, ...) below
            }
            if (!(fv instanceof Float8Vector)) {
                throw new ArrowBindingException(
                    "Column '" + slot.getName() + "' must be a Float8Vector (float64) for zero-copy "
                        + "evaluation; found " + fv.getClass().getSimpleName()
                        + ". Cast this column to float64 before binding.");
            }
            columns.put(slot.getName(), (Float8Vector) fv);
        }

        evaluate(columns, output, nullPolicy, parallel);
    }

    // =========================================================================
    // Output allocation convenience
    // =========================================================================

    /**
     * Allocates and sizes a {@link Float8Vector} suitable for use as
     * {@code output} in {@link #evaluate}.
     * @param allocator
     * @param name
     * @param rowCount
     * @return 
     */
    public static Float8Vector allocateOutput(BufferAllocator allocator, String name, int rowCount) {
        Float8Vector v = new Float8Vector(name, allocator);
        v.allocateNew(rowCount);
        v.setValueCount(rowCount);

        // Arrow zero-fills the validity bitmap on allocation, so a fresh vector
        // is all-null until something explicitly marks a row valid. evaluate()'s
        // SIMD path writes result data via a raw MemorySegment aliasing the data
        // buffer directly (see ArrowMemoryBridge) — by design, this never goes
        // through the ordinary per-element setter that would otherwise flip each
        // row's validity bit. So an output vector built by this method needs to
        // start out valid: NullPolicy.IGNORE callers (the default) can then read
        // results back immediately, and NullPolicy.PROPAGATE recomputes the
        // bitmap from scratch afterward regardless (see propagateNulls), so this
        // default is overwritten there, not relied upon.
        int validityBytes = (rowCount + 7) / 8;
        ArrowBuf validity = v.getValidityBuffer();
        for (int i = 0; i < validityBytes; i++) {
            validity.setByte(i, (byte) 0xFF);
        }
        return v;
    }

    // =========================================================================
    // Internals
    // =========================================================================

    private void fillConstant(Float8Vector output, int rowCount) {
        // varCount == 0: SIMDEngineEvaluator.applyBulk(MemorySegment[], ...) is a
        // documented no-op for a zero-length variable array, so we compute the
        // single constant value once via the ordinary scalar solver and fill the
        // output directly. This path is O(rowCount) either way (writing rowCount
        // output values is unavoidable), so the lack of a SIMD fast path here
        // costs nothing relative to the alternative.
        double value = expression.solveGeneric().scalar;
        for (int i = 0; i < rowCount; i++) {
            output.set(i, value);
        }
    }

    private void propagateNulls(Map<String, Float8Vector> columns, Float8Vector output, int rowCount) {
        int validityBytes = (rowCount + 7) / 8;
        ArrowBuf outValidity = output.getValidityBuffer();

        // Start from "all valid" and AND in each bound column's validity bitmap,
        // so the output is null wherever any required input was null.
        for (int i = 0; i < validityBytes; i++) {
            outValidity.setByte(i, (byte) 0xFF);
        }
        for (MathExpression.Slot slot : requiredSlots) {
            Float8Vector col = columns.get(slot.getName());
            ArrowBuf colValidity = col.getValidityBuffer();
            for (int i = 0; i < validityBytes; i++) {
                byte combined = (byte) (outValidity.getByte(i) & colValidity.getByte(i));
                outValidity.setByte(i, combined);
            }
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("ArrowBulkEvaluator has been closed: " + expression.getExpression());
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            compiled.close();
        }
    }
}