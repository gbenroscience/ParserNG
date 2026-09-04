package com.github.gbenroscience.arrow.tools.box;

import com.github.gbenroscience.parser.MathExpression; 
import com.github.gbenroscience.simd.turbo.tools.utils.HardwareDetector;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandSegmentF32;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandSegmentF64; 

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Evaluates a compiled ParserNG expression directly over Apache Arrow columnar
 * batches, using {@link SIMDCommandSegmentF64}'s zero-copy
 * {@code MemorySegment[]}-backed bulk evaluation path — no coalescing copy, no
 * on-heap staging buffer, no per-row boxing.
 * 
 * Also uses {@link SIMDCommandSegmentF32}'s zero copy for Float4Vector data
 *
 * <h2>Binding model</h2>
 * Variables in the expression are bound to Arrow columns <b>by name</b>. The
 * authoritative name-to-slot mapping comes from
 * {@link MathExpression#getSlotItems()}, which reflects exactly the variables
 * the compiled expression actually references and the frame index each one
 * occupies — this class does not guess at slot ordering.
 *
 * <h2>Type support — and each instance's SINGLE precision</h2>
 * Both {@link Float8Vector} (float64) and {@link Float4Vector} (float32)
 * columns are supported for zero-copy binding, and this class implements the
 * full four-method {@link ArrowExpressionEvaluator} surface (both precisions
 * x both binding styles) so it satisfies that interface. But <b>a single
 * compiled instance only ever holds ONE real engine</b> — {@link #compile}
 * builds a {@code compiled} (float64) instance with {@code compiledF32 ==
 * null}, {@link #compileF32} builds the reverse. Calling an
 * {@code evaluate(...)} overload for the precision this instance was NOT
 * compiled for now throws {@link IllegalStateException} up front (rather than
 * a bare {@code NullPointerException} from the unguarded engine field) — the
 * interface can't express "this instance only supports one precision" at the
 * type level, so this guard is the runtime enforcement of that constraint.
 * Constant expressions ({@link #isConstantExpression()}) are the one
 * exception: {@link #fillConstant} never touches either engine field, so
 * those succeed regardless of which {@code evaluate} overload you call —
 * worth knowing since it means the two precisions behave identically for
 * constants but diverge (guard vs. real dispatch) for anything with
 * variables.
 * <p>
 * Columns of other numeric types must be cast to their respective
 * floating-point precision by the caller before binding — this class
 * deliberately does not perform an implicit narrowing/widening copy, since
 * doing so silently would reintroduce the exact copy this module exists to
 * eliminate.
 * <h2>Constant expressions</h2>
 * An expression that references no variables (e.g. {@code "42.0"}) compiles to
 * a zero-slot evaluator.
 * {@code SIMDCommandSegmentF64.applyBulk(MemorySegment[], ...)} treats a
 * zero-length variable array as a no-op by design (see its internal guard
 * clause) — left unhandled, that would silently leave the output buffer
 * untouched. This class detects that case up front and fills the output
 * directly instead.
 *
 * <h2>Thread safety</h2>
 * A single {@code ArrowBulkEvaluator} instance may be shared and called
 * concurrently from multiple threads and will always produce correct results,
 * but the two evaluation modes differ in how much actual concurrency you get:
 * calls with {@code parallel=false} run fully concurrently (the underlying
 * evaluator uses a {@code ThreadLocal} context per caller thread), while calls
 * with {@code parallel=true} (the default) are internally serialized against
 * each other — {@code SIMDVectorCompositeExpression}'s worker-pool dispatch
 * uses a single shared coordination structure that is only safe for one
 * external caller at a time, so concurrent parallel calls queue rather than
 * overlap. If you need true concurrent parallel evaluation from multiple
 * threads, give each thread its own {@code ArrowBulkEvaluator} (a separate
 * {@link #compile} call) rather than sharing one. The one hard exception either
 * way is {@link #close()}: do not call it while another thread may still be
 * inside {@link #evaluate}.
 *
 * <h2>Lifecycle</h2>
 * Call {@link #close()} when done — this shuts down the evaluator's CPU-pinned
 * worker thread pool (if one was created). This class implements
 * {@link AutoCloseable} for try-with-resources use. {@link #close()} is
 * null-safe against whichever of {@code compiled}/{@code compiledF32} this
 * instance doesn't have, and idempotent.
 *
 * <h2>Switching backends</h2>
 * This class also implements {@link ArrowExpressionEvaluator}, the surface
 * shared with the GPU-backed {@link ArrowGpuBulkEvaluator}. Prefer compiling
 * through {@link ArrowExpressionEvaluators#compile} with
 * {@link ArrowExecutionBackend#CPU_SIMD} (or
 * {@link ArrowExpressionEvaluators#compilePreferGpu}) over calling
 * {@link #compile(String)} directly when the call site should stay
 * backend-agnostic.
 */
public final class ArrowBulkEvaluator implements ArrowExpressionEvaluator {

    private final MathExpression expression;
    private final SIMDCommandSegmentF64.SIMDVectorCompositeExpression compiled;
    private final SIMDCommandSegmentF32.SIMDVectorCompositeExpression compiledF32;
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
    // being serialized. Shared across both precisions since an instance only
    // ever has one real engine active.
    private final Object parallelLock = new Object();

    private ArrowBulkEvaluator(MathExpression expression,
            SIMDCommandSegmentF64.SIMDVectorCompositeExpression compiled) {
        this.expression = expression;
        this.compiled = compiled;
        this.compiledF32 = null;
        this.requiredSlots = expression.getSlotItems();
        this.slotCount = expression.getRegistry().size();
        this.constantExpression = requiredSlots.length == 0;

        String[] names = new String[requiredSlots.length];
        for (int i = 0; i < requiredSlots.length; i++) {
            names[i] = requiredSlots[i].getName();
        }
        this.requiredVariableNames = names;
    }

    private ArrowBulkEvaluator(MathExpression expression,
            SIMDCommandSegmentF32.SIMDVectorCompositeExpression compiled) {
        this.expression = expression;
        this.compiled = null;
        this.compiledF32 = compiled;
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
     * Compiles {@code expr} with the default worker configuration, for
     * float64 ({@link Float8Vector}) evaluation.
     *
     * @param expr
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowBulkEvaluator compile(String expr) throws Throwable {
        return compile(new MathExpression(expr), HardwareDetector.detectPhysicalCores());
    }

    /**
     * Compiles {@code expr} with an explicit CPU-pinned worker count for
     * {@link #evaluate}'s parallel path, for float64 evaluation. Pass
     * {@code 0} for the library default.
     *
     * @param expr
     * @param numWorkers
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowBulkEvaluator compile(String expr, int numWorkers) throws Throwable {
        return compile(new MathExpression(expr), numWorkers);
    }

    /**
     * Compiles {@code expr} with the default worker configuration, for
     * float32 ({@link Float4Vector}) evaluation. See the class javadoc's
     * "Known limitation" note before relying on results from this path.
     *
     * @param expr
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowBulkEvaluator compileF32(String expr) throws Throwable {
        return compileF32(new MathExpression(expr), HardwareDetector.detectPhysicalCores());
    }

    /**
     * Compiles {@code expr} with an explicit CPU-pinned worker count for
     * {@link #evaluate}'s parallel path, for float32 evaluation. Pass
     * {@code 0} for the library default. See the class javadoc's "Known
     * limitation" note before relying on results from this path.
     *
     * @param expr
     * @param numWorkers
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowBulkEvaluator compileF32(String expr, int numWorkers) throws Throwable {
        return compileF32(new MathExpression(expr), numWorkers);
    }

    /**
     * Compiles an already-constructed {@link MathExpression} for float64
     * evaluation. Useful when the caller needs to inspect or configure the
     * expression (e.g. via {@link MathExpression#setWillFoldConstants(boolean)})
     * before compiling.
     *
     * @param expression
     * @param numWorkers
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowBulkEvaluator compile(MathExpression expression, int numWorkers) throws Throwable {
        SIMDCommandSegmentF64.SIMDVectorCompositeExpression raw = SIMDCommandSegmentF64.getEvaluator(expression, numWorkers);
        return new ArrowBulkEvaluator(expression, raw);
    }

    /**
     * Compiles an already-constructed {@link MathExpression} for float32
     * evaluation. See the class javadoc's "Known limitation" note before
     * relying on results from this path.
     *
     * @param expression
     * @param numWorkers
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowBulkEvaluator compileF32(MathExpression expression, int numWorkers) throws Throwable { 
        SIMDCommandSegmentF32.SIMDVectorCompositeExpression raw = SIMDCommandSegmentF32.getEvaluator(expression, numWorkers);
        return new ArrowBulkEvaluator(expression, raw);
    }

    // =========================================================================
    // Introspection
    // =========================================================================
    /**
     * The variable names this expression requires, in no particular order.
     * Every one of these must have a corresponding Arrow column bound at
     * evaluation time, or {@link #evaluate} throws
     * {@link ArrowBindingException}.
     *
     * @return
     */
    @Override
    public String[] requiredVariableNames() {
        return requiredVariableNames.clone();
    }

    /**
     * True if this expression references no variables at all (e.g. a bare
     * numeric literal or a fully constant-folded expression). Such expressions
     * still evaluate correctly via {@link #evaluate} regardless of which
     * precision's overload is called — see the class javadoc's "Type support"
     * section — but take a different, non-SIMD internal path documented on
     * {@link #evaluate}.
     *
     * @return
     */
    @Override
    public boolean isConstantExpression() {
        return constantExpression;
    }

    @Override
    public String getExpressionText() {
        return expression.getExpression();
    }

    /**
     * Always {@link ArrowExecutionBackend#CPU_SIMD} for this class.
     *
     * @return
     */
    @Override
    public ArrowExecutionBackend backend() {
        return ArrowExecutionBackend.CPU_SIMD;
    }
 

    // =========================================================================
    // Evaluation — Map<String, Float8Vector> binding
    // =========================================================================
    @Override
    public void evaluate(Map<String, Float8Vector> columns, Float8Vector output) {
        evaluate(columns, output, NullPolicy.IGNORE, true);
    }

    @Override
    public void evaluate(Map<String, Float8Vector> columns, Float8Vector output, NullPolicy nullPolicy) {
        evaluate(columns, output, nullPolicy, true);
    }

    /**
     * Evaluates the compiled expression, writing one result per row into
     * {@code output}.
     *
     * <p>
     * <b>Precondition:</b> {@code output} must already be sized —
     * {@code output.allocateNew(rowCount)} and
     * {@code output.setValueCount(rowCount)} must have been called before this
     * method, where {@code rowCount} is the number of rows to evaluate. Use
     * {@link #allocateOutput} if you don't already have an output vector
     * prepared.
     *
     * @param columns Arrow columns, keyed by the variable name they bind to.
     * Must contain an entry for every name in {@link #requiredVariableNames()};
     * extra entries are ignored. Every bound column's {@code getValueCount()}
     * must be at least {@code output.getValueCount()}.
     * @param output pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled — see
     * {@link NullPolicy}
     * @param parallel if true, dispatches to the evaluator's CPU-pinned worker
     * pool for large batches (recommended for standalone calls); pass false if
     * this call is already running inside the caller's own worker thread and
     * nested parallelism should be avoided
     * @throws ArrowBindingException if a required column is missing, a bound
     * column is shorter than the output, or the output has not been sized
     * @throws IllegalStateException if this instance was compiled via
     * {@link #compileF32} (no float64 engine to dispatch to) and this is not a
     * constant expression
     */
    public void evaluate(Map<String, Float8Vector> columns, Float8Vector output, NullPolicy nullPolicy, boolean parallel) {
        ensureOpen();

        if (constantExpression) {
            fillConstant(output, output.getValueCount());
            return;
        }

        if (compiled == null) {
            throw new IllegalStateException(
                    "This ArrowBulkEvaluator was compiled for float32 (via compileF32(...)) and has no "
                    + "float64 engine to evaluate Float8Vector columns. Use the Float4Vector evaluate(...) "
                    + "overloads instead, or compile this expression with compile(...) for float64. "
                    + "Expression: " + expression.getExpression());
        }

        int rowCount = output.getValueCount();
        if (rowCount == 0) {
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
    @Override
    public void evaluate(VectorSchemaRoot root, Float8Vector output) {
        evaluate(root, output, NullPolicy.IGNORE, true);
    }

    @Override
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
     * {@code root}, or is present but is not a {@link Float8Vector}
     * @throws IllegalStateException if this instance was compiled via
     * {@link #compileF32} and this is not a constant expression — see
     * {@link #evaluate(Map, Float8Vector, NullPolicy, boolean)}
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
    // Evaluation — Map<String, Float4Vector> binding
    // =========================================================================
    @Override
    public void evaluate(Map<String, Float4Vector> columns, Float4Vector output) {
        evaluate(columns, output, NullPolicy.IGNORE, true);
    }

    @Override
    public void evaluate(Map<String, Float4Vector> columns, Float4Vector output, NullPolicy nullPolicy) {
        evaluate(columns, output, nullPolicy, true);
    }

    /**
     * Evaluates the compiled expression, writing one result per row into
     * {@code output}.
     *
     * <p>
     * <b>Precondition:</b> {@code output} must already be sized —
     * {@code output.allocateNew(rowCount)} and
     * {@code output.setValueCount(rowCount)} must have been called before this
     * method, where {@code rowCount} is the number of rows to evaluate. Use
     * {@link #allocateOutputF32} if you don't already have an output vector
     * prepared.
     *
     * <p>
     * <b>See the class javadoc's "Known limitation" note</b> — the underlying
     * float32 engine's {@code MemorySegment} dispatch currently has an
     * unresolved byte-stride bug affecting this method's results for any
     * non-constant expression.
     *
     * @param columns Arrow columns, keyed by the variable name they bind to.
     * Must contain an entry for every name in {@link #requiredVariableNames()};
     * extra entries are ignored. Every bound column's {@code getValueCount()}
     * must be at least {@code output.getValueCount()}.
     * @param output pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled — see
     * {@link NullPolicy}
     * @param parallel if true, dispatches to the evaluator's CPU-pinned worker
     * pool for large batches (recommended for standalone calls); pass false if
     * this call is already running inside the caller's own worker thread and
     * nested parallelism should be avoided
     * @throws ArrowBindingException if a required column is missing, a bound
     * column is shorter than the output, or the output has not been sized
     * @throws IllegalStateException if this instance was compiled via
     * {@link #compile} (no float32 engine to dispatch to) and this is not a
     * constant expression
     */
    public void evaluate(Map<String, Float4Vector> columns, Float4Vector output, NullPolicy nullPolicy, boolean parallel) {
        ensureOpen();

        if (constantExpression) {
            fillConstant(output, output.getValueCount());
            return;
        }

        if (compiledF32 == null) {
            throw new IllegalStateException(
                    "This ArrowBulkEvaluator was compiled for float64 (via compile(...)) and has no "
                    + "float32 engine to evaluate Float4Vector columns. Use the Float8Vector evaluate(...) "
                    + "overloads instead, or compile this expression with compileF32(...) for float32. "
                    + "Expression: " + expression.getExpression());
        }

        int rowCount = output.getValueCount();
        if (rowCount == 0) {
            for (MathExpression.Slot slot : requiredSlots) {
                Float4Vector col = columns.get(slot.getName());
                if (col != null && col.getValueCount() > 0) {
                    throw new ArrowBindingException(
                            "Output vector has not been sized (valueCount=0) but bound column '"
                            + slot.getName() + "' has " + col.getValueCount() + " rows. Call "
                            + "output.allocateNew(rowCount) and output.setValueCount(rowCount) "
                            + "before evaluate(), or use allocateOutputF32(...).");
                }
            }
            return;
        }

        MemorySegment[] variables = new MemorySegment[slotCount];
        for (MathExpression.Slot slot : requiredSlots) {
            Float4Vector col = columns.get(slot.getName());
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
            // Use wrapFloats for Float4Vector bindings to maintain proper 32-bit offset/stride mappings
            variables[slot.getSlot()] = ArrowMemoryBridge.wrapFloats(col.getDataBuffer(), rowCount);
        }

        // Output requires wrapFloats representation to accurately dispatch to F32 SIMD bulk operators
        MemorySegment outSeg = ArrowMemoryBridge.wrapFloats(output.getDataBuffer(), rowCount);

        if (parallel) {
            synchronized (parallelLock) {
                compiledF32.applyBulkParallel(variables, outSeg);
            }
        } else {
            compiledF32.applyBulk(variables, outSeg);
        }

        if (nullPolicy == NullPolicy.PROPAGATE) {
            propagateNulls(columns, output, rowCount);
        }
    }

    // =========================================================================
    // Evaluation — VectorSchemaRoot convenience binding for Float4Vector
    // =========================================================================
    @Override
    public void evaluate(VectorSchemaRoot root, Float4Vector output) {
        evaluate(root, output, NullPolicy.IGNORE, true);
    }

    @Override
    public void evaluate(VectorSchemaRoot root, Float4Vector output, NullPolicy nullPolicy) {
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
     * {@code root}, or is present but is not a {@link Float4Vector}
     * @throws IllegalStateException if this instance was compiled via
     * {@link #compile} and this is not a constant expression — see
     * {@link #evaluate(Map, Float4Vector, NullPolicy, boolean)}
     */
    public void evaluate(VectorSchemaRoot root, Float4Vector output, NullPolicy nullPolicy, boolean parallel) {
        ensureOpen();

        Map<String, Float4Vector> columns = new HashMap<>(Math.max(4, requiredSlots.length * 2));
        for (MathExpression.Slot slot : requiredSlots) {
            FieldVector fv = root.getVector(slot.getName());
            if (fv == null) {
                continue;
            }
            if (!(fv instanceof Float4Vector)) {
                throw new ArrowBindingException(
                        "Column '" + slot.getName() + "' must be a Float4Vector (float32) for zero-copy "
                        + "evaluation; found " + fv.getClass().getSimpleName()
                        + ". Cast this column to float32 before binding.");
            }
            columns.put(slot.getName(), (Float4Vector) fv);
        }

        evaluate(columns, output, nullPolicy, parallel);
    }

    // =========================================================================
    // Output allocation convenience
    // =========================================================================
    /**
     * Allocates and sizes a {@link Float8Vector} suitable for use as
     * {@code output} in {@link #evaluate}.
     *
     * @param allocator
     * @param name
     * @param rowCount
     * @return
     */
    public static Float8Vector allocateOutput(BufferAllocator allocator, String name, int rowCount) {
        Float8Vector v = new Float8Vector(name, allocator);
        v.allocateNew(rowCount);
        v.setValueCount(rowCount);

        int validityBytes = (rowCount + 7) / 8;
        ArrowBuf validity = v.getValidityBuffer();
        for (int i = 0; i < validityBytes; i++) {
            validity.setByte(i, (byte) 0xFF);
        }
        return v;
    }

    /**
     * Allocates and sizes a {@link Float4Vector} suitable for use as
     * {@code output} in {@link #evaluate}.
     *
     * @param allocator
     * @param name
     * @param rowCount
     * @return
     */
    public static Float4Vector allocateOutputF32(BufferAllocator allocator, String name, int rowCount) {
        Float4Vector v = new Float4Vector(name, allocator);
        v.allocateNew(rowCount);
        v.setValueCount(rowCount);

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
        double value = expression.solveGeneric().scalar;
        for (int i = 0; i < rowCount; i++) {
            output.set(i, value);
        }
    }

    private void propagateNulls(Map<String, Float8Vector> columns, Float8Vector output, int rowCount) {
        int validityBytes = (rowCount + 7) / 8;
        ArrowBuf outValidity = output.getValidityBuffer();

        for (int i = 0; i < validityBytes; i++) {
            outValidity.setByte(i, (byte) 0xFF);
        }
        for (MathExpression.Slot slot : requiredSlots) {
            Float8Vector col = columns.get(slot.getName());
            if (col == null || col.getNullCount() == 0) {
                continue;
            }
            ArrowBuf colValidity = col.getValidityBuffer();
            if (colValidity == null) {
                continue;
            }
            for (int i = 0; i < validityBytes; i++) {
                byte combined = (byte) (outValidity.getByte(i) & colValidity.getByte(i));
                outValidity.setByte(i, combined);
            }
        }
    }

    private void fillConstant(Float4Vector output, int rowCount) {
        float value = (float) expression.solveGeneric().scalar;
        for (int i = 0; i < rowCount; i++) {
            output.set(i, value);
        }
    }

    private void propagateNulls(Map<String, Float4Vector> columns, Float4Vector output, int rowCount) {
        int validityBytes = (rowCount + 7) / 8;
        ArrowBuf outValidity = output.getValidityBuffer();

        for (int i = 0; i < validityBytes; i++) {
            outValidity.setByte(i, (byte) 0xFF);
        }
        for (MathExpression.Slot slot : requiredSlots) {
            Float4Vector col = columns.get(slot.getName());
            if (col == null || col.getNullCount() == 0) {
                continue;
            }
            ArrowBuf colValidity = col.getValidityBuffer();
            if (colValidity == null) {
                continue;
            }
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
            if (compiled != null) {
                compiled.close();
            }
            if (compiledF32 != null) {
                compiledF32.close();
            }
        }
    }

    /**
     * Filters {@code root} by evaluating this instance's compiled expression
     * as a boolean predicate over its rows.
     *
     * <p>This class has no dedicated boolean vector type — see
     * {@link ArrowExpressionEvaluator}'s float64/float32-only surface — so
     * the predicate is computed via the ordinary {@code evaluate(...)} path,
     * using whichever precision this instance was compiled for (see
     * {@link #isFloat64()}), into a throwaway output vector that never
     * leaves this method. The result is interpreted with C-style
     * truthiness: {@code 0.0}/{@code 0.0f} is {@code false}; anything else
     * (including {@code NaN} and infinities) is {@code true}.
     *
     * <p>Under {@link NullPolicy#PROPAGATE}, a row whose predicate result is
     * null (because a bound input column was null there) is excluded from
     * the result — standard SQL {@code WHERE} semantics, where an unknown
     * predicate is not true. Under {@link NullPolicy#IGNORE}, validity
     * bitmaps are never consulted and the row is kept or dropped purely on
     * whatever value the arithmetic produced.
     *
     * <p>The result batch preserves {@code root}'s schema and column order.
     * Selected rows are copied — never aliased — via each column's
     * {@code copyFromSafe}, using a {@link BufferAllocator} taken from
     * {@code root}'s own first column.
     *
     * @param root Arrow record batch containing the columns referenced by
     * the compiled predicate
     * @param nullPolicy how Arrow validity bitmaps and null predicate values
     * are handled — see the null-handling note above
     * @return a new Arrow record batch containing only rows for which the
     * compiled predicate evaluates to a truthy value
     * @throws NullPointerException if {@code root} or {@code nullPolicy} is
     * null
     * @throws ArrowBindingException if {@code root} has no columns (there is
     * then no allocator to build the result batch from), or if a required
     * variable's column is missing or of the wrong vector type
     */
    @Override
    public VectorSchemaRoot filter(VectorSchemaRoot root, NullPolicy nullPolicy) {
        ensureOpen();
        if (root == null) {
            throw new NullPointerException("root must not be null");
        }
        if (nullPolicy == null) {
            throw new NullPointerException("nullPolicy must not be null");
        }

        int rowCount = root.getRowCount();
        BufferAllocator allocator = ArrowFilterSupport.resolveAllocator(root);

        int[] selected;
        if (rowCount == 0) {
            selected = new int[0];
        } else if (ArrowGpuBulkEvaluator.isFloat64(root)) {
            try (Float8Vector predicate = allocateOutput(allocator, "__parser_ng_filter_predicate__", rowCount)) {
                evaluate(root, predicate, nullPolicy, true);
                selected = ArrowFilterSupport.selectIndices(predicate, nullPolicy);
            }
        } else {
            try (Float4Vector predicate = allocateOutputF32(allocator, "__parser_ng_filter_predicate__", rowCount)) {
                evaluate(root, predicate, nullPolicy, true);
                selected = ArrowFilterSupport.selectIndices(predicate, nullPolicy);
            }
        }

        return ArrowFilterSupport.materializeSelectedRows(root, selected, allocator);
    }

    @Override
    public VectorSchemaRoot filter(VectorSchemaRoot root) {
        return ArrowExpressionEvaluator.super.filter(root);
    }
}