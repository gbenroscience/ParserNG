/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.arrow.tools.box;

/**
 *
 * @author GBEMIRO
 */ 

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Package-private helper shared by {@link ArrowBulkEvaluator#filter} and
 * {@link ArrowGpuBulkEvaluator#filter}.
 *
 * <p>Both evaluators compute their boolean predicate result the same way —
 * dispatch through the ordinary {@code evaluate(...)} path into a throwaway
 * {@link Float8Vector} or {@link Float4Vector} — and then need the exact
 * same two things done with it: turn the predicate into a list of selected
 * row indices, and copy those rows out of the source batch into a new
 * {@link VectorSchemaRoot}. Neither operation is backend-specific, so it
 * lives here once instead of twice.
 *
 * <h2>Truthiness</h2>
 * Neither {@link ArrowBulkEvaluator} nor {@link ArrowGpuBulkEvaluator} has a
 * dedicated boolean vector type — {@link ArrowExpressionEvaluator}'s surface
 * is float64/float32 only. Predicate results are therefore interpreted with
 * C-style truthiness: {@code 0.0} (or {@code 0.0f}) is {@code false}; any
 * other value — including negative numbers, {@code NaN}, and infinities —
 * is {@code true}. This matches how {@code MathExpression} comparison and
 * logical operators are expected to report their results (nonzero for true,
 * {@code 0.0} for false) through this purely-numeric evaluation surface.
 *
 * <h2>Null handling</h2>
 * Under {@link NullPolicy#PROPAGATE}, a row whose predicate result is null
 * (because a bound input column was null there) is treated as excluded —
 * the standard SQL {@code WHERE} rule that an unknown predicate is not true.
 * Under {@link NullPolicy#IGNORE}, validity bitmaps are never consulted;
 * the row is kept or dropped purely on the raw data value, matching
 * {@code evaluate(...)}'s own documented behavior for that policy.
 */
final class ArrowFilterSupport {

    private ArrowFilterSupport() {
    }

    /**
     * Resolves the {@link BufferAllocator} to use for a filter result batch,
     * taken from {@code root}'s own first column. There is no other source
     * of an allocator available to a {@code filter(...)} call — the
     * evaluator itself does not own or store one — so a batch with no
     * columns at all cannot be filtered.
     *
     * @throws ArrowBindingException if {@code root} has no columns
     */
    static BufferAllocator resolveAllocator(VectorSchemaRoot root) {
        List<FieldVector> vectors = root.getFieldVectors();
        if (vectors.isEmpty()) {
            throw new ArrowBindingException(
                    "Cannot filter a VectorSchemaRoot with no columns; filter() needs at least one "
                    + "existing column to obtain a BufferAllocator for the result batch.");
        }
        return vectors.get(0).getAllocator();
    }

    /**
     * Row indices (in ascending order) at which {@code predicate} is
     * truthy, per {@code nullPolicy}. See the class javadoc's "Truthiness"
     * and "Null handling" sections.
     */
    static int[] selectIndices(Float8Vector predicate, NullPolicy nullPolicy) {
        int rowCount = predicate.getValueCount();
        boolean checkNulls = nullPolicy == NullPolicy.PROPAGATE;

        int[] buffer = new int[Math.max(16, rowCount / 4)];
        int count = 0;
        for (int i = 0; i < rowCount; i++) {
            if (checkNulls && predicate.isNull(i)) {
                continue;
            }
            if (predicate.get(i) != 0.0) {
                if (count == buffer.length) {
                    buffer = Arrays.copyOf(buffer, buffer.length * 2);
                }
                buffer[count++] = i;
            }
        }
        return Arrays.copyOf(buffer, count);
    }

    /**
     * Float32 counterpart of {@link #selectIndices(Float8Vector, NullPolicy)},
     * for {@link ArrowBulkEvaluator} instances compiled via
     * {@link ArrowBulkEvaluator#compileF32}.
     */
    static int[] selectIndices(Float4Vector predicate, NullPolicy nullPolicy) {
        int rowCount = predicate.getValueCount();
        boolean checkNulls = nullPolicy == NullPolicy.PROPAGATE;

        int[] buffer = new int[Math.max(16, rowCount / 4)];
        int count = 0;
        for (int i = 0; i < rowCount; i++) {
            if (checkNulls && predicate.isNull(i)) {
                continue;
            }
            if (predicate.get(i) != 0.0f) {
                if (count == buffer.length) {
                    buffer = Arrays.copyOf(buffer, buffer.length * 2);
                }
                buffer[count++] = i;
            }
        }
        return Arrays.copyOf(buffer, count);
    }

    /**
     * Builds a new {@link VectorSchemaRoot} with the same schema and column
     * order as {@code source}, containing only the rows at
     * {@code selectedIndices} (copied in the given order).
     *
     * <p>Rows are copied — never aliased — via each column's
     * {@code copyFromSafe(int, int, ValueVector)}, which works uniformly
     * across every Arrow vector type without this class needing to know
     * what those types are.
     */
    static VectorSchemaRoot materializeSelectedRows(
            VectorSchemaRoot source, int[] selectedIndices, BufferAllocator allocator) {

        List<FieldVector> sourceVectors = source.getFieldVectors();
        List<FieldVector> outVectors = new ArrayList<>(sourceVectors.size());
        int outRowCount = selectedIndices.length;

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
        }

        return new VectorSchemaRoot(source.getSchema(), outVectors, outRowCount);
    }

    /**
     * Builds a new {@link VectorSchemaRoot} containing every existing column
     * of {@code source} plus {@code computed} appended as a new trailing
     * column, sharing the same row count.
     *
     * <p>Unlike {@link #materializeSelectedRows}, this performs no copying of
     * {@code source}'s existing columns — they are reused directly (aliased)
     * in the returned batch's vector list. That's safe here because, unlike
     * filtering, projection never changes which rows exist or what order
     * they're in; only a new column is added, so there is nothing about the
     * existing columns that needs to change. This is what makes
     * {@link ArrowExpressionEvaluator#project} considerably cheaper than
     * {@link ArrowExpressionEvaluator#filter} for large batches — only the
     * one new column is freshly allocated and computed.
     *
     * <p>Used by {@link ArrowExpressionEvaluator#project} directly, and by
     * {@link ArrowExpressionEvaluator#filterProject} to attach its projected
     * column onto the already row-filtered batch produced by
     * {@link #materializeSelectedRows}.
     *
     * @param source batch whose columns and row count the result inherits
     * @param computed the new column to append; its row count must match
     * {@code source}'s row count
     * @return a new {@code VectorSchemaRoot} with {@code source}'s schema
     * plus {@code computed}'s field, and {@code source}'s columns plus
     * {@code computed}
     * @throws ArrowBindingException if {@code computed}'s row count does not
     * match {@code source}'s row count
     */
    static VectorSchemaRoot appendColumn(VectorSchemaRoot source, FieldVector computed) {
        int rowCount = source.getRowCount();
        if (computed.getValueCount() != rowCount) {
            throw new ArrowBindingException(
                    "Computed column '" + computed.getField().getName() + "' has "
                    + computed.getValueCount() + " rows but the batch it is being attached to has "
                    + rowCount + " rows; project()/filterProject() require the projected column to "
                    + "cover exactly every row of that batch.");
        }

        List<FieldVector> vectors = new ArrayList<>(source.getFieldVectors());
        vectors.add(computed);

        List<Field> fields = new ArrayList<>(source.getSchema().getFields());
        fields.add(computed.getField());

        return new VectorSchemaRoot(new Schema(fields), vectors, rowCount);
    }
}