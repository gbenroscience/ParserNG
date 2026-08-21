package com.github.gbenroscience.arrow.tools.box1;

import com.github.gbenroscience.simdext.turbo.tools.SIMDEngineEvaluator;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Zero-copy bulk evaluator bridging Apache Arrow columnar batches to
 * ParserNG's {@link SIMDEngineEvaluator}.
 * <p>
 * Each bound input column's Arrow data buffer is exposed directly to
 * SIMDEngineEvaluator as a {@link MemorySegment} (see {@link
 * ArrowSegments}), and the output {@link Float8Vector}'s data buffer is
 * bound the same way — so a batch is evaluated without copying Arrow's
 * column data into an intermediate {@code double[]} at any point.
 * {@code SIMDEngineEvaluator}'s {@code applyBulk(MemorySegment[],
 * MemorySegment)} entry point (added specifically for this kind of
 * per-column, independently-allocated-buffer source) reads operands
 * straight out of each MemorySegment for pure arithmetic chains
 * ({@code +}, {@code -}, {@code *}, {@code /}); anything invoking a
 * transcendental function, {@code POW}, a comparison, {@code IF}/{@code
 * AND}/{@code OR}, or {@code VMA} still copies that one operand into an
 * on-heap scratch buffer at the point it's needed — see the engine's own
 * class docs for the detail on that boundary.
 *
 * <h2>Variable ordering</h2>
 * {@code applyBulk(MemorySegment[], MemorySegment)} expects one segment
 * per variable, in the slot order ParserNG assigned internally when it
 * compiled the expression (i.e. {@code variables[i]} must be the data for
 * whatever variable occupies slot {@code i}). <b>Confirmed by ParserNG's
 * author:</b> {@code MathExpression} assigns slots on a first-appearance,
 * left-to-right basis as the expression is scanned — {@code "x + y * z"}
 * binds slot 0 = x, slot 1 = y, slot 2 = z. This class still does
 * <b>not</b> auto-derive that order from the expression string itself —
 * reimplementing ParserNG's tokenizer here (to distinguish a variable
 * like {@code y} from a function name like {@code sin}) would be its own
 * source of bugs — so {@link Builder#variables(String...)} takes it as an
 * explicit parameter that you must supply in the same left-to-right
 * order. A mismatched *order* does not throw — it silently binds the
 * wrong column to the wrong variable — so double-check the list against
 * your expression string. {@link Builder#build()} runs a one-row smoke
 * test against the on-heap {@code applyBulk(double[][], double[])} path
 * to catch a wrong variable *count* early; it cannot detect two
 * variables swapped in order, since both are structurally valid.
 *
 * <h2>What this module does not (yet) do</h2>
 * <ul>
 *   <li>Only {@code Float8Vector} columns are bound zero-copy. Other
 *       supported numeric Arrow types ({@code IntVector}, {@code
 *       BigIntVector}, {@code Float4Vector}) are coerced automatically via
 *       {@link VectorCoercion} — a real, non-zero-copy allocation and copy,
 *       scoped to the single {@link #evaluateInto} call that needed it and
 *       closed again once that call returns. A column type {@link
 *       VectorCoercion} doesn't handle (e.g. {@code DecimalVector}, {@code
 *       VarCharVector}) still throws {@link UnsupportedVectorTypeException};
 *       cast it upstream in your own pipeline first.</li>
 *   <li>{@link NullPolicy#PROPAGATE_NULL}'s validity-bitmap AND is a
 *       scalar byte loop, not SIMD — see {@link ArrowSegments} for why
 *       that's a reasonable v1 tradeoff.</li>
 *   <li>No streaming/batched-iterator convenience is provided yet; call
 *       {@link #evaluateInto} once per {@link VectorSchemaRoot} batch in
 *       your own read loop (e.g. over an {@code ArrowStreamReader} or
 *       {@code ArrowFileReader}).</li>
 * </ul>
 *
 * <h2>Lifecycle</h2>
 * Holds a compiled {@code SIMDVectorCompositeExpression}, which in turn
 * owns a worker thread pool when built with {@code parallel(true)} (or a
 * non-default worker count). Always {@code close()} — or use
 * try-with-resources — when done with an evaluator instance.
 */
public final class ArrowBulkEvaluator implements AutoCloseable {

    private final SIMDEngineEvaluator.SIMDVectorCompositeExpression compiled;
    private final List<String> variableOrder;
    private final NullPolicy nullPolicy;
    private final boolean parallel;
    private volatile boolean closed = false;

    private ArrowBulkEvaluator(SIMDEngineEvaluator.SIMDVectorCompositeExpression compiled,
                                List<String> variableOrder,
                                NullPolicy nullPolicy,
                                boolean parallel) {
        this.compiled = compiled;
        this.variableOrder = variableOrder;
        this.nullPolicy = nullPolicy;
        this.parallel = parallel;
    }

    public static Builder builder(String expression) {
        return new Builder(expression);
    }

    /**
     * Evaluates this expression over {@code root}, allocating and
     * returning a new {@link Float8Vector} of the result from {@code
     * allocator}. The returned vector is owned by the caller and must be
     * closed by the caller.
     * @param root
     * @param allocator
     * @return 
     */
    public Float8Vector evaluate(VectorSchemaRoot root, BufferAllocator allocator) {
        int rowCount = root.getRowCount();
        Float8Vector output = new Float8Vector("result", allocator);
        try {
            output.allocateNew(Math.max(rowCount, 1));
            output.setValueCount(rowCount);
            evaluateInto(root, output);
            return output;
        } catch (RuntimeException | Error e) {
            // evaluateInto can throw after output has already been
            // allocated (missing column, wrong vector type, null-policy
            // violation, ...). Without this, the caller never gets a
            // reference to `output` to close, and its off-heap buffers
            // leak for the lifetime of the allocator.
            output.close();
            throw e;
        }
    }

    /**
     * Evaluates this expression over {@code root}, writing results into
     * the caller-supplied {@code output} vector. {@code output} must
     * already have at least {@code root.getRowCount()} capacity allocated
     * (e.g. via {@code allocateNew(rowCount)}) — this method does not
     * allocate or grow that buffer, which is what keeps the output side
     * of the zero-copy path genuinely zero-copy: its existing data
     * buffer is bound directly as the evaluator's output segment. It
     * does, however, set {@code output}'s value count to {@code
     * root.getRowCount()} itself, overwriting whatever value count the
     * buffer had before the call — callers only need to ensure capacity,
     * not pre-set the count.
     * @param root
     * @param output
     */
    public void evaluateInto(VectorSchemaRoot root, Float8Vector output) {
        ensureOpen();
        long rowCount = root.getRowCount();
        output.setValueCount((int) rowCount);
        if (rowCount == 0) {
            return;
        }

        BoundColumns bound = bindColumns(root, output.getAllocator());
        try {
            Float8Vector[] boundVectors = bound.vectors;
            MemorySegment[] inputSegments = new MemorySegment[boundVectors.length];
            for (int i = 0; i < boundVectors.length; i++) {
                inputSegments[i] = ArrowSegments.ofData(boundVectors[i], rowCount);
            }
            MemorySegment outputSegment = ArrowSegments.ofData(output, rowCount);

            if (nullPolicy == NullPolicy.REJECT_ON_NULL) {
                rejectIfAnyNull(boundVectors, rowCount);
            }

            if (parallel) {
                compiled.applyBulkParallel(inputSegments, outputSegment);
            } else {
                compiled.applyBulk(inputSegments, outputSegment);
            }

            switch (nullPolicy) {
                case PROPAGATE_NULL -> propagateNullsInto(boundVectors, output, rowCount);
                case REJECT_ON_NULL -> {
                    // rejectIfAnyNull already proved every row is non-null, and
                    // the kernel above wrote a real value for each one — but it
                    // wrote straight into the raw data buffer, never touching
                    // Arrow's validity bitmap. A freshly allocateNew'd vector's
                    // validity bitmap starts zeroed (all-null), so without this
                    // every row would read back as null despite holding a
                    // correct, freshly computed value.
                    MemorySegment outputValidity = ArrowSegments.ofValidity(output, rowCount);
                    ArrowSegments.markAllValid(outputValidity, rowCount);
                }
            }
        } finally {
            // Any column VectorCoercion had to allocate (a non-Float8Vector
            // column) is scoped to this call — it must be closed here, once
            // we're done reading from it, or its buffers leak on every
            // evaluation. Vectors that were already Float8Vector (the
            // zero-copy case) are owned by the caller's VectorSchemaRoot and
            // must NOT be closed here.
            bound.closeCoerced();
        }
    }

    /**
     * Resolves {@link #variableOrder} against {@code root}'s columns.
     * Columns that are already {@link Float8Vector} are bound directly
     * (zero-copy); any other numeric type {@link VectorCoercion} supports
     * is converted into a freshly allocated {@code Float8Vector} — a real,
     * non-zero-copy copy, owned by the returned {@link BoundColumns} and
     * scoped to a single {@link #evaluateInto} call. Callers must invoke
     * {@link BoundColumns#closeCoerced()} once they're done reading from
     * the bound vectors.
     */
    private BoundColumns bindColumns(VectorSchemaRoot root, BufferAllocator allocator) {
        Float8Vector[] bound = new Float8Vector[variableOrder.size()];
        List<Float8Vector> coerced = new ArrayList<>();
        for (int i = 0; i < variableOrder.size(); i++) {
            String name = variableOrder.get(i);
            FieldVector fv = root.getVector(name);
            if (fv == null) {
                closeAll(coerced);
                throw new ArrowBindingException("No column named '" + name
                        + "' found in VectorSchemaRoot; declared schema fields: " + fieldNames(root));
            }
            if (fv instanceof Float8Vector f8) {
                bound[i] = f8;
            } else {
                Float8Vector converted;
                try {
                    // Throws UnsupportedVectorTypeException itself for any
                    // type it doesn't know how to convert (DecimalVector,
                    // VarCharVector, ...).
                    converted = VectorCoercion.toFloat8(fv, allocator);
                } catch (RuntimeException e) {
                    closeAll(coerced);
                    throw e;
                }
                bound[i] = converted;
                coerced.add(converted);
            }
        }
        return new BoundColumns(bound, coerced);
    }

    private static void closeAll(List<Float8Vector> vectors) {
        for (Float8Vector v : vectors) {
            v.close();
        }
    }

    /**
     * The columns bound for one {@link #evaluateInto} call. {@link
     * #vectors} is what the caller reads from; {@link #closeCoerced()}
     * must be called exactly once, after the caller is done reading, to
     * release any real (non-zero-copy) conversions {@link VectorCoercion}
     * had to allocate along the way. Vectors that were already {@code
     * Float8Vector} are not touched by {@link #closeCoerced()} — they
     * belong to the caller's {@code VectorSchemaRoot}.
     */
    private static final class BoundColumns {
        final Float8Vector[] vectors;
        private final List<Float8Vector> coerced;

        BoundColumns(Float8Vector[] vectors, List<Float8Vector> coerced) {
            this.vectors = vectors;
            this.coerced = coerced;
        }

        void closeCoerced() {
            closeAll(coerced);
        }
    }

    private void rejectIfAnyNull(Float8Vector[] boundVectors, long rowCount) {
        for (Float8Vector v : boundVectors) {
            MemorySegment validity = ArrowSegments.ofValidity(v, rowCount);
            for (long row = 0; row < rowCount; row++) {
                if (!ArrowSegments.isValid(validity, row)) {
                    throw new ArrowNullValueException(v.getField().getName(), row);
                }
            }
        }
    }

    private void propagateNullsInto(Float8Vector[] boundVectors, Float8Vector output, long rowCount) {
        if (boundVectors.length == 0) {
            return;
        }
        MemorySegment outValidity = ArrowSegments.ofValidity(output, rowCount);
        MemorySegment first = ArrowSegments.ofValidity(boundVectors[0], rowCount);
        ArrowSegments.copyValidityInto(outValidity, first);
        for (int i = 1; i < boundVectors.length; i++) {
            MemorySegment next = ArrowSegments.ofValidity(boundVectors[i], rowCount);
            ArrowSegments.andValidityInto(outValidity, outValidity, next);
        }
    }

    private static List<String> fieldNames(VectorSchemaRoot root) {
        List<String> names = new ArrayList<>();
        root.getSchema().getFields().forEach(f -> names.add(f.getName()));
        return names;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ArrowBulkEvaluator is closed");
        }
    }

    @Override
    public void close() {
        try (compiled) {
            if (closed) {
                return;
            }
            closed = true;
        }
    }

    public static final class Builder {
        private final String expression;
        private List<String> variables;
        private NullPolicy nullPolicy = NullPolicy.REJECT_ON_NULL;
        private boolean parallel = false;

        private Builder(String expression) {
            this.expression = Objects.requireNonNull(expression, "expression");
        }

        /**
         * Declares the variable-to-slot order this expression's columns
         * must be bound in. See the class-level javadoc's "Variable
         * ordering" section — this is <b>not</b> auto-discovered.
         * @param names
         * @return 
         */
        public Builder variables(String... names) {
            this.variables = List.of(names);
            return this;
        }

        public Builder nullPolicy(NullPolicy policy) {
            this.nullPolicy = Objects.requireNonNull(policy, "policy");
            return this;
        }

        /** Whether to dispatch through {@code applyBulkParallel} (SIMDEngineEvaluator's own worker pool).
         * @param parallel
         * @return  */
        public Builder parallel(boolean parallel) {
            this.parallel = parallel;
            return this;
        }

        public ArrowBulkEvaluator build() {
            if (variables == null) {
                throw new ArrowBindingException(
                        "ArrowBulkEvaluator.Builder.variables(...) must be called before build() — the bind "
                                + "order cannot be inferred automatically. Call variables() with no arguments "
                                + "for a constant expression that reads no columns. See the class javadoc's "
                                + "\"Variable ordering\" section.");
            }

            SIMDEngineEvaluator.SIMDVectorCompositeExpression compiled;
            try {
                compiled = SIMDEngineEvaluator.getEvaluator(expression);
            } catch (Throwable t) {
                throw new ArrowBindingException("Failed to compile expression: " + expression, t);
            }

            smokeTest(compiled, variables.size());

            return new ArrowBulkEvaluator(compiled, variables, nullPolicy, parallel);
        }

        /**
         * Runs a single-row evaluation with dummy on-heap data as a
         * sanity check that the declared variable count is consistent
         * with what the compiled expression expects. A wrong COUNT of
         * variables generally surfaces here as an
         * ArrayIndexOutOfBoundsException from the engine, which is
         * wrapped with a clearer message; two variables bound in swapped
         * ORDER will not, since both are structurally valid.
         */
        private static void smokeTest(SIMDEngineEvaluator.SIMDVectorCompositeExpression compiled,
                                       int declaredVarCount) {
            try {
                double[][] dummy = new double[declaredVarCount][1];
                double[] out = new double[1];
                compiled.applyBulk(dummy, out);
            } catch (ArrayIndexOutOfBoundsException e) {
                throw new ArrowBindingException(
                        "Declared " + declaredVarCount + " variable(s) via .variables(...), but the compiled "
                                + "expression expects a different count. Verify the variable list against the "
                                + "expression string.", e);
            }
        }
    }
}