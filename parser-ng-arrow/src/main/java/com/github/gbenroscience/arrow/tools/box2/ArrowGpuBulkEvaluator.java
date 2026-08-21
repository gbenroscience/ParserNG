package com.github.gbenroscience.arrow.tools.box2;

import com.github.gbenroscience.gpu.GpuBackend;
import com.github.gbenroscience.gpu.evaluator.GpuCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.GpuExpressionBridge;
import com.github.gbenroscience.parser.MathExpression;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Evaluates a compiled ParserNG expression directly over Apache Arrow
 * columnar batches, using {@link GpuExpressionBridge}'s OpenCL/CUDA-backed
 * bulk evaluation path instead of {@code SIMDEngineEvaluator}'s CPU SIMD
 * path.
 *
 * <h2>Why this is a separate class, not a mode flag on {@link ArrowBulkEvaluator}</h2>
 * {@code SIMDEngineEvaluator.SIMDVectorCompositeExpression.applyBulk} takes
 * a {@code MemorySegment[]} — one independent segment per bound variable —
 * which is exactly what lets {@link ArrowBulkEvaluator} bind each Arrow
 * column's data buffer with zero copying. {@link GpuCompositeExpression
 * #applyBulk(MemorySegment, MemorySegment)} has a different contract: it
 * takes a <b>single</b> input segment holding every variable concatenated
 * column-major — slot {@code s}'s {@code rowCount} values occupy
 * {@code [s*rowCount, (s+1)*rowCount)} — matching
 * {@code VectorTurboEvaluator}'s own {@code flatVariables} layout (see
 * {@code GpuCompositeExpressionTest.buildSafeSamples}). Arrow's per-column
 * buffers are independent allocations, not laid out contiguously with each
 * other, so this class cannot offer the same zero-copy binding
 * {@link ArrowBulkEvaluator} does: every call stages each bound column's
 * data into its slice of one flat off-heap buffer before dispatch. This is
 * an additional host-side copy on top of whatever host-&gt;device copy the
 * GPU backend itself performs per call (see {@link GpuCompositeExpression}'s
 * javadoc on the unified-memory caveat) — for small/medium batches this is
 * usually still worth it for the GPU's throughput, but it means this class
 * is not a drop-in "same cost, more parallelism" swap for
 * {@link ArrowBulkEvaluator}; measure before committing to it for a given
 * batch size.
 *
 * <h2>Binding model</h2>
 * Identical to {@link ArrowBulkEvaluator}: variables are bound to Arrow
 * columns by name, with the authoritative name-to-slot mapping coming from
 * {@link MathExpression#getSlotItems()}.
 *
 * <h2>Type support</h2>
 * Only {@link Float8Vector} (Arrow's float64 column type) columns are
 * supported, same as {@link ArrowBulkEvaluator} — this class evaluates in
 * full double precision on the GPU (via
 * {@link GpuCompositeExpression#applyBulk(MemorySegment, MemorySegment)}),
 * not the native float32 kernel path
 * ({@link GpuCompositeExpression#applyBulkF32}). There is currently no
 * float32 counterpart here since Arrow columns bound by this class are
 * float64 to begin with.
 *
 * <h2>Constant expressions</h2>
 * Same handling as {@link ArrowBulkEvaluator}: a zero-slot expression skips
 * the GPU entirely and fills the output via the ordinary scalar solver.
 *
 * <h2>Backend selection</h2>
 * {@link #compile(MathExpression)} auto-selects a backend (CUDA preferred,
 * OpenCL fallback — see {@link GpuExpressionBridge}'s javadoc for the
 * preference order and the {@code -Dgpu.backend.preference} override).
 * {@link #compile(MathExpression, GpuBackend)} pins a specific backend.
 * Use {@link #isBackendAvailable(GpuBackend)} to probe before committing —
 * e.g. to decide at startup whether to build an {@link ArrowGpuBulkEvaluator}
 * at all or fall back to {@link ArrowBulkEvaluator} on machines with no GPU.
 *
 * <h2>Thread safety</h2>
 * A single instance may be shared and called concurrently from multiple
 * threads and will always produce correct results, but unlike
 * {@link ArrowBulkEvaluator} (which has a fully-concurrent {@code
 * parallel=false} path), every {@code evaluate} call here is internally
 * serialized against every other call on the same instance — both GPU
 * backends dispatch through shared per-instance device state (a command
 * queue/stream and kernel-arg buffers) that is not safe for concurrent use
 * from multiple threads (see {@code GpuCompositeExpressionTest}'s BUG#3
 * regression note). If you need concurrent GPU evaluation from multiple
 * threads, give each thread its own instance (a separate {@link #compile}
 * call) rather than sharing one. As with {@link ArrowBulkEvaluator}, do not
 * call {@link #close()} while another thread may still be inside
 * {@link #evaluate}.
 *
 * <h2>Lifecycle</h2>
 * Call {@link #close()} when done — this releases the compiled
 * expression's device-side resources ({@link GpuCompositeExpression} owns a
 * device buffer and, depending on backend, a staging {@code Arena}).
 * Implements {@link AutoCloseable} for try-with-resources use.
 */
public final class ArrowGpuBulkEvaluator implements AutoCloseable {

    private final MathExpression expression;
    private final GpuCompositeExpression compiled;
    private final MathExpression.Slot[] requiredSlots;
    private final String[] requiredVariableNames;
    private final int slotCount;
    private final boolean constantExpression;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    // Every GPU dispatch on this instance goes through shared per-instance
    // device state (kernel args, command queue/stream) that only one caller
    // may touch at a time -- see the class javadoc's Thread safety section.
    // Unlike ArrowBulkEvaluator's parallelLock (which only guards the
    // parallel=true path), this guards every evaluate() call, since there is
    // no non-serialized GPU dispatch path to fall back to.
    private final Object dispatchLock = new Object();

    private ArrowGpuBulkEvaluator(MathExpression expression, GpuCompositeExpression compiled) {
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
     * Compiles {@code expr}, auto-selecting a GPU backend (CUDA preferred,
     * OpenCL fallback). Throws whatever {@link GpuExpressionBridge} throws
     * if no usable backend is found on this machine — see its javadoc; that
     * exception carries the real per-backend bootstrap failures as
     * suppressed exceptions, not just a generic "no GPU" message.
     * @param expr
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowGpuBulkEvaluator compile(String expr) throws Throwable {
        return compile(new MathExpression(expr));
    }

    /**
     * Compiles an already-constructed {@link MathExpression}, auto-selecting
     * a GPU backend.
     * @param expression
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowGpuBulkEvaluator compile(MathExpression expression) throws Throwable {
        return new ArrowGpuBulkEvaluator(expression, GpuExpressionBridge.compile(expression));
    }

    /**
     * Compiles {@code expr} pinned to a specific GPU backend.
     * @param expr
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowGpuBulkEvaluator compile(String expr, GpuBackend backend) throws Throwable {
        return compile(new MathExpression(expr), backend);
    }

    /**
     * Compiles an already-constructed {@link MathExpression} pinned to a
     * specific GPU backend.
     * @param expression
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowGpuBulkEvaluator compile(MathExpression expression, GpuBackend backend) throws Throwable {
        return new ArrowGpuBulkEvaluator(expression, GpuExpressionBridge.compile(expression, backend));
    }

    /**
     * Whether {@code backend}'s bootstrap has succeeded (or would succeed,
     * probing it now) on this JVM. Safe to call speculatively before
     * deciding whether to build an {@link ArrowGpuBulkEvaluator} at all, or
     * to fall back to {@link ArrowBulkEvaluator} instead. Thin passthrough
     * to {@link GpuExpressionBridge#isAvailable(GpuBackend)}.
     * @param backend
     * @return
     */
    public static boolean isBackendAvailable(GpuBackend backend) {
        return GpuExpressionBridge.isAvailable(backend);
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
     * True if this expression references no variables at all. Such
     * expressions still evaluate correctly via {@link #evaluate} — the
     * output is filled with the single constant value on the CPU, and the
     * GPU is never touched.
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
        evaluate(columns, output, NullPolicy.IGNORE);
    }

    /**
     * Evaluates the compiled expression on the GPU, writing one result per
     * row into {@code output}.
     *
     * <p><b>Precondition:</b> {@code output} must already be sized —
     * {@code output.allocateNew(rowCount)} and {@code output.setValueCount(rowCount)}
     * must have been called before this method. Use
     * {@link ArrowBulkEvaluator#allocateOutput} if you don't already have an
     * output vector prepared.
     *
     * @param columns    Arrow columns, keyed by the variable name they bind
     *                   to. Must contain an entry for every name in
     *                   {@link #requiredVariableNames()}; extra entries are
     *                   ignored. Every bound column's {@code getValueCount()}
     *                   must be at least {@code output.getValueCount()}.
     * @param output     pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled — see {@link NullPolicy}
     * @throws ArrowBindingException if a required column is missing, a bound
     *                                column is shorter than the output, the
     *                                output has not been sized, or the GPU
     *                                dispatch itself throws
     */
    public void evaluate(Map<String, Float8Vector> columns, Float8Vector output, NullPolicy nullPolicy) {
        ensureOpen();

        int rowCount = output.getValueCount();
        if (rowCount == 0) {
            for (MathExpression.Slot slot : requiredSlots) {
                Float8Vector col = columns.get(slot.getName());
                if (col != null && col.getValueCount() > 0) {
                    throw new ArrowBindingException(
                        "Output vector has not been sized (valueCount=0) but bound column '"
                            + slot.getName() + "' has " + col.getValueCount() + " rows. Call "
                            + "output.allocateNew(rowCount) and output.setValueCount(rowCount) "
                            + "before evaluate(), or use ArrowBulkEvaluator.allocateOutput(...).");
                }
            }
            return; // legitimately empty batch, nothing further to do
        }

        if (constantExpression) {
            fillConstant(output, rowCount);
            return;
        }

        // Stage every bound column into one flat, column-major buffer --
        // see the class javadoc for why this copy is unavoidable here (the
        // GPU applyBulk(MemorySegment, MemorySegment) contract, unlike the
        // CPU SIMD one, takes a single concatenated input segment rather
        // than one segment per variable).
        try (Arena arena = Arena.ofConfined()) {
            long bytesPerColumn = (long) rowCount * Double.BYTES;
            MemorySegment flatIn = arena.allocate(ValueLayout.JAVA_DOUBLE, (long) slotCount * rowCount);

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
                MemorySegment colSeg = ArrowMemoryBridge.wrapDoubles(col.getDataBuffer(), rowCount);
                MemorySegment.copy(colSeg, 0, flatIn, (long) slot.getSlot() * bytesPerColumn, bytesPerColumn);
            }

            MemorySegment outSeg = ArrowMemoryBridge.wrapDoubles(output.getDataBuffer(), rowCount);

            synchronized (dispatchLock) {
                try {
                    compiled.applyBulk(flatIn, outSeg);
                } catch (ArrowBindingException e) {
                    throw e;
                } catch (Throwable t) {
                    throw new ArrowBindingException(
                        "GPU evaluation failed for expression '" + expression.getExpression() + "'", t);
                }
            }
        }

        if (nullPolicy == NullPolicy.PROPAGATE) {
            propagateNulls(columns, output, rowCount);
        }
    }

    // =========================================================================
    // Evaluation — VectorSchemaRoot convenience binding
    // =========================================================================

    public void evaluate(VectorSchemaRoot root, Float8Vector output) {
        evaluate(root, output, NullPolicy.IGNORE);
    }

    /**
     * Convenience overload that resolves each required variable's column by
     * name from {@code root} instead of a caller-built {@code Map}.
     *
     * @param root
     * @param output
     * @param nullPolicy
     * @throws ArrowBindingException if a required field is missing from
     *                                {@code root}, or is present but is not
     *                                a {@link Float8Vector}
     */
    public void evaluate(VectorSchemaRoot root, Float8Vector output, NullPolicy nullPolicy) {
        ensureOpen();

        Map<String, Float8Vector> columns = new HashMap<>(Math.max(4, requiredSlots.length * 2));
        for (MathExpression.Slot slot : requiredSlots) {
            FieldVector fv = root.getVector(slot.getName());
            if (fv == null) {
                continue; // reported uniformly by evaluate(Map, ...) above
            }
            if (!(fv instanceof Float8Vector)) {
                throw new ArrowBindingException(
                    "Column '" + slot.getName() + "' must be a Float8Vector (float64) for GPU "
                        + "evaluation; found " + fv.getClass().getSimpleName()
                        + ". Cast this column to float64 before binding.");
            }
            columns.put(slot.getName(), (Float8Vector) fv);
        }

        evaluate(columns, output, nullPolicy);
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
            ArrowBuf colValidity = col.getValidityBuffer();
            for (int i = 0; i < validityBytes; i++) {
                byte combined = (byte) (outValidity.getByte(i) & colValidity.getByte(i));
                outValidity.setByte(i, combined);
            }
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("ArrowGpuBulkEvaluator has been closed: " + expression.getExpression());
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            compiled.close();
        }
    }
}