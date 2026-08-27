package com.github.gbenroscience.arrow.tools.box;

import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.Map;
import org.apache.arrow.vector.Float4Vector;

/**
 * Backend-agnostic contract implemented by both {@link ArrowBulkEvaluator}
 * (CPU, SIMD-vectorized) and {@link ArrowGpuBulkEvaluator} (GPU, CUDA or
 * OpenCL). Code written against this interface doesn't need to know or care
 * which backend actually compiled and evaluates the expression — the same call
 * sites work whether the instance underneath is running on the CPU worker pool
 * or dispatching to a GPU device.
 *
 * <h2>Getting an instance</h2>
 * Build instances through {@link ArrowExpressionEvaluators} rather than
 * choosing between {@link ArrowBulkEvaluator#compile} and
 * {@link ArrowGpuBulkEvaluator#compile} directly at each call site — that's
 * what makes the backend a one-line change (an {@link ArrowExecutionBackend}
 * value) instead of a call-site rewrite.
 *
 * <h2>What's deliberately left out</h2>
 * This interface omits {@link ArrowBulkEvaluator}'s {@code parallel} flag and
 * {@link ArrowGpuBulkEvaluator}'s backend/device introspection and selection
 * methods ({@code actualBackend()}, {@code deviceDescription()},
 * {@code listOpenClDevices()}, etc.) — those are backend-specific tuning knobs
 * with no equivalent on the other side, not part of the portable surface.
 * Downcast to the concrete type (guided by {@link #backend()}) when you need
 * one of those.
 *
 * <h2>Binding, null handling, thread safety</h2>
 * Identical across both implementations — see {@link ArrowBulkEvaluator}'s and
 * {@link ArrowGpuBulkEvaluator}'s class javadocs for the full contract
 * (name-based variable binding via {@code MathExpression.getSlotItems()},
 * {@link Float8Vector}-only columns, {@link NullPolicy} semantics, and each
 * backend's own concurrency rules). Always {@link #close()} when done — use
 * try-with-resources.
 */
public interface ArrowExpressionEvaluator extends AutoCloseable {

    /**
     * Evaluates the compiled expression, writing one result per row into
     * {@code output}. See the implementing class's {@code evaluate} javadoc for
     * the full precondition/exception contract.
     *
     * @param columns Arrow columns, keyed by the variable name they bind to.
     * @param output pre-sized destination vector.
     * @param nullPolicy how Arrow validity bitmaps are handled.
     */
    void evaluate(Map<String, Float8Vector> columns, Float8Vector output, NullPolicy nullPolicy);

    /**
     * Convenience for {@link #evaluate(Map, Float8Vector, NullPolicy)} with
     * {@link NullPolicy#IGNORE}.
     *
     * @param columns
     * @param output
     */
    default void evaluate(Map<String, Float8Vector> columns, Float8Vector output) {
        evaluate(columns, output, NullPolicy.IGNORE);
    }

    /**
     * Evaluates the compiled expression, writing one result per row into
     * {@code output}. See the implementing class's {@code evaluate} javadoc for
     * the full precondition/exception contract.
     *
     * @param columns Arrow columns, keyed by the variable name they bind to.
     * @param output pre-sized destination vector.
     * @param nullPolicy how Arrow validity bitmaps are handled.
     */
    void evaluate(Map<String, Float4Vector> columns, Float4Vector output, NullPolicy nullPolicy);

    /**
     * Convenience for {@link #evaluate(Map, Float4Vector, NullPolicy)} with
     * {@link NullPolicy#IGNORE}.
     *
     * @param columns
     * @param output
     */
    default void evaluate(Map<String, Float4Vector> columns, Float4Vector output) {
        evaluate(columns, output, NullPolicy.IGNORE);
    }

    /**
     * Convenience overload that resolves each required variable's column by
     * name from {@code root} instead of a caller-built {@code Map}.
     *
     * @param root
     * @param output
     * @param nullPolicy
     */
    void evaluate(VectorSchemaRoot root, Float8Vector output, NullPolicy nullPolicy);

    /**
     * Convenience for
     * {@link #evaluate(VectorSchemaRoot, Float8Vector, NullPolicy)} with
     * {@link NullPolicy#IGNORE}.
     *
     * @param root
     * @param output
     */
    default void evaluate(VectorSchemaRoot root, Float8Vector output) {
        evaluate(root, output, NullPolicy.IGNORE);
    }

    /**
     * Convenience overload that resolves each required variable's column by
     * name from {@code root} instead of a caller-built {@code Map}.
     *
     * @param root
     * @param output
     * @param nullPolicy
     */
    void evaluate(VectorSchemaRoot root, Float4Vector output, NullPolicy nullPolicy);

    /**
     * Convenience for
     * {@link #evaluate(VectorSchemaRoot, Float4Vector, NullPolicy)} with
     * {@link NullPolicy#IGNORE}.
     *
     * @param root
     * @param output
     */
    default void evaluate(VectorSchemaRoot root, Float4Vector output) {
        evaluate(root, output, NullPolicy.IGNORE);
    }

    /**
     * The variable names this expression requires, in no particular order.
     *
     * @return
     */
    String[] requiredVariableNames();

    /**
     * True if this expression references no variables at all (e.g. a bare
     * numeric literal or a fully constant-folded expression). Both backends
     * handle this the same way: the GPU/SIMD engine is never touched, and the
     * output is filled directly via the ordinary scalar solver.
     *
     * @return
     */
    boolean isConstantExpression();

    String getExpressionText();

    /**
     * Which engine this instance actually runs on. For an instance compiled
     * with {@link ArrowExecutionBackend#GPU_AUTO}, this reflects whichever
     * concrete backend was actually selected (CUDA or OpenCL) — not the AUTO
     * request itself.
     *
     * @return
     */
    ArrowExecutionBackend backend();

    /**
     * Releases this evaluator's resources (CPU worker pool or GPU device
     * buffers, depending on backend). Do not call while another thread may
     * still be inside {@link #evaluate}.
     */
    @Override
    void close();
}
