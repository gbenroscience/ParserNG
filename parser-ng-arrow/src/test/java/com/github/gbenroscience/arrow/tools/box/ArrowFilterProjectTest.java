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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Coverage for {@link ArrowExpressionEvaluator#filter}, {@code #project}, and
 * {@code #filterProject}, exercised against {@link ArrowBulkEvaluator}
 * (CPU/SIMD). These are {@code default} methods implemented once on the
 * interface, so this suite is the correctness contract for both backends —
 * {@link ArrowGpuBulkEvaluator} inherits the same behavior without its own
 * override, aside from the cross-backend case in
 * {@link FilterProjectTests#filterProject_predicateAndProjectionCanUseDifferentBackends()}.
 *
 * <p>All tests use float64 ({@link Float8Vector}) columns; the known float32
 * limitation documented in the main README (non-constant {@code compileF32}
 * results) is out of scope here.
 */
final class ArrowFilterProjectTest {

    private BufferAllocator allocator;

    @BeforeEach
    void setUp() {
        allocator = new RootAllocator();
    }

    @AfterEach
    void tearDown() {
        allocator.close();
    }

    // ---- shared helpers -------------------------------------------------

    private Float8Vector column(String name, double... values) {
        Float8Vector vector = new Float8Vector(name, allocator);
        vector.allocateNew(values.length);
        for (int i = 0; i < values.length; i++) {
            vector.set(i, values[i]);
        }
        vector.setValueCount(values.length);
        return vector;
    }

    private Float8Vector columnWithNullAt(String name, int nullIndex, double... values) {
        Float8Vector vector = column(name, values);
        vector.setNull(nullIndex);
        return vector;
    }

    private VectorSchemaRoot rootOf(Float8Vector... columns) {
        return VectorSchemaRoot.of(columns);
    }

    private double[] toArray(Float8Vector vector) {
        double[] out = new double[vector.getValueCount()];
        for (int i = 0; i < out.length; i++) {
            out[i] = vector.get(i);
        }
        return out;
    }

    /**
     * Delegating wrapper that records the row count it was actually asked to
     * evaluate over — used to prove {@code filterProject}'s projection stage
     * only ever sees the surviving rows, not the original batch.
     */
    private static final class RowCountRecordingEvaluator implements ArrowExpressionEvaluator {
        private final ArrowExpressionEvaluator delegate;
        private int lastEvaluatedRowCount = -1;

        RowCountRecordingEvaluator(ArrowExpressionEvaluator delegate) {
            this.delegate = delegate;
        }

        int lastEvaluatedRowCount() {
            return lastEvaluatedRowCount;
        }

        @Override
        public void evaluate(Map<String, Float8Vector> columns, Float8Vector output, NullPolicy nullPolicy) {
            lastEvaluatedRowCount = output.getValueCount();
            delegate.evaluate(columns, output, nullPolicy);
        }

        @Override
        public void evaluate(Map<String, Float4Vector> columns, Float4Vector output, NullPolicy nullPolicy) {
            delegate.evaluate(columns, output, nullPolicy);
        }

        @Override
        public void evaluate(VectorSchemaRoot root, Float8Vector output, NullPolicy nullPolicy) {
            lastEvaluatedRowCount = output.getValueCount();
            delegate.evaluate(root, output, nullPolicy);
        }

        @Override
        public void evaluate(VectorSchemaRoot root, Float4Vector output, NullPolicy nullPolicy) {
            delegate.evaluate(root, output, nullPolicy);
        }

        @Override
        public VectorSchemaRoot filter(VectorSchemaRoot root, NullPolicy nullPolicy) {
            return delegate.filter(root, nullPolicy);
        }

        @Override
        public String[] requiredVariableNames() {
            return delegate.requiredVariableNames();
        }

        @Override
        public boolean isConstantExpression() {
            return delegate.isConstantExpression();
        }

        @Override
        public String getExpressionText() {
            return delegate.getExpressionText();
        }

        @Override
        public ArrowExecutionBackend backend() {
            return delegate.backend();
        }

        @Override
        public void close() {
            delegate.close();
        }
    }

    // =====================================================================
    // filter
    // =====================================================================

    @Nested
    class FilterTests {

        @Test
        void filter_selectsOnlyMatchingRows() throws Throwable {
            try (ArrowBulkEvaluator isHot = ArrowBulkEvaluator.compile("temperature > 90.0");
                 VectorSchemaRoot root = rootOf(column("temperature", 88.0, 95.0, 40.0, 91.5))) {

                VectorSchemaRoot result = isHot.filter(root);

                assertEquals(2, result.getRowCount());
                assertArrayEquals(new double[]{95.0, 91.5},
                        toArray((Float8Vector) result.getVector("temperature")));
                result.close();
            }
        }

        @Test
        void filter_returnsEmptyBatchWhenNoRowsMatch() throws Throwable {
            try (ArrowBulkEvaluator none = ArrowBulkEvaluator.compile("x > 1000.0");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0, 3.0))) {

                VectorSchemaRoot result = none.filter(root);

                assertEquals(0, result.getRowCount());
                assertEquals(root.getSchema(), result.getSchema());
                result.close();
            }
        }

        @Test
        void filter_returnsAllRowsWhenAllMatch() throws Throwable {
            try (ArrowBulkEvaluator all = ArrowBulkEvaluator.compile("x >= 0.0");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0, 3.0))) {

                VectorSchemaRoot result = all.filter(root);

                assertEquals(root.getRowCount(), result.getRowCount());
                assertArrayEquals(toArray((Float8Vector) root.getVector("x")),
                        toArray((Float8Vector) result.getVector("x")));
                result.close();
            }
        }

        @Test
        void filter_preservesSchemaAndColumnOrderAcrossMultipleColumns() throws Throwable {
            try (ArrowBulkEvaluator predicate = ArrowBulkEvaluator.compile("price > 50.0");
                 VectorSchemaRoot root = rootOf(
                         column("price", 40.0, 60.0, 70.0),
                         column("volume", 100.0, 200.0, 300.0))) {

                VectorSchemaRoot result = predicate.filter(root);

                assertEquals(root.getSchema(), result.getSchema());
                assertArrayEquals(new double[]{60.0, 70.0}, toArray((Float8Vector) result.getVector("price")));
                assertArrayEquals(new double[]{200.0, 300.0}, toArray((Float8Vector) result.getVector("volume")));
                result.close();
            }
        }

        @Test
        void filter_withPropagatePolicy_excludesRowsWithNullPredicateInput() throws Throwable {
            try (ArrowBulkEvaluator predicate = ArrowBulkEvaluator.compile("x > 0.0");
                 VectorSchemaRoot root = rootOf(columnWithNullAt("x", 1, 5.0, 0.0, 7.0))) {

                // Row 1's "x" is null -> predicate result is null -> PROPAGATE excludes it,
                // regardless of whatever raw bit pattern happens to sit in the data buffer.
                VectorSchemaRoot result = predicate.filter(root, NullPolicy.PROPAGATE);

                assertEquals(2, result.getRowCount());
                assertArrayEquals(new double[]{5.0, 7.0}, toArray((Float8Vector) result.getVector("x")));
                result.close();
            }
        }

        @Test
        void filter_defaultOverload_matchesExplicitIgnorePolicy() throws Throwable {
            try (ArrowBulkEvaluator predicate = ArrowBulkEvaluator.compile("x > 1.0");
                 VectorSchemaRoot rootA = rootOf(column("x", 0.5, 2.0, 3.0));
                 VectorSchemaRoot rootB = rootOf(column("x", 0.5, 2.0, 3.0))) {

                VectorSchemaRoot viaDefault = predicate.filter(rootA);
                VectorSchemaRoot viaExplicit = predicate.filter(rootB, NullPolicy.IGNORE);

                assertArrayEquals(
                        toArray((Float8Vector) viaDefault.getVector("x")),
                        toArray((Float8Vector) viaExplicit.getVector("x")));
                viaDefault.close();
                viaExplicit.close();
            }
        }

        @Test
        void filter_throwsArrowBindingException_whenRequiredColumnMissing() throws Throwable {
            try (ArrowBulkEvaluator predicate = ArrowBulkEvaluator.compile("y > 0.0");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0))) {

                // "y" is required but root only has "x".
                assertThrows(ArrowBindingException.class, () -> predicate.filter(root));
            }
        }
    }

    // =====================================================================
    // project
    // =====================================================================

    @Nested
    class ProjectTests {

        @Test
        void project_appendsComputedColumnWithCorrectValues() throws Throwable {
            try (ArrowBulkEvaluator score = ArrowBulkEvaluator.compile("0.5*rsi + 0.5*macd");
                 VectorSchemaRoot root = rootOf(
                         column("rsi", 20.0, 60.0),
                         column("macd", 10.0, 4.0))) {

                VectorSchemaRoot result = score.project(root, "score");

                assertArrayEquals(new double[]{15.0, 32.0}, toArray((Float8Vector) result.getVector("score")), 1e-9);
                result.close();
            }
        }

        @Test
        void project_preservesRowCountAndExistingColumnValues() throws Throwable {
            try (ArrowBulkEvaluator doubled = ArrowBulkEvaluator.compile("x * 2");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0, 3.0))) {

                VectorSchemaRoot result = doubled.project(root, "y");

                assertEquals(root.getRowCount(), result.getRowCount());
                assertArrayEquals(toArray((Float8Vector) root.getVector("x")),
                        toArray((Float8Vector) result.getVector("x")));
                assertArrayEquals(new double[]{2.0, 4.0, 6.0}, toArray((Float8Vector) result.getVector("y")));
                result.close();
            }
        }

        @Test
        void project_throwsWhenOutputFieldNameAlreadyExists() throws Throwable {
            try (ArrowBulkEvaluator expr = ArrowBulkEvaluator.compile("x + 1");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0))) {

                assertThrows(ArrowBindingException.class, () -> expr.project(root, "x"));
            }
        }

        @Test
        void project_throwsNullPointerException_whenRootIsNull() throws Throwable {
            try (ArrowBulkEvaluator expr = ArrowBulkEvaluator.compile("x + 1")) {
                assertThrows(NullPointerException.class, () -> expr.project(null, "y"));
            }
        }

        @Test
        void project_existingColumnsAreReusedNotCopied() throws Throwable {
            try (ArrowBulkEvaluator expr = ArrowBulkEvaluator.compile("x + 1");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0))) {

                VectorSchemaRoot result = expr.project(root, "y");

                // project() aliases root's existing columns rather than copying them.
                assertSame(root.getVector("x"), result.getVector("x"));
                result.close();
            }
        }

        @Test
        void project_worksForConstantExpression() throws Throwable {
            try (ArrowBulkEvaluator constant = ArrowBulkEvaluator.compile("2 * 21");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0, 3.0))) {

                VectorSchemaRoot result = constant.project(root, "answer");

                assertArrayEquals(new double[]{42.0, 42.0, 42.0}, toArray((Float8Vector) result.getVector("answer")));
                result.close();
            }
        }
    }

    // =====================================================================
    // filterProject
    // =====================================================================

    @Nested
    class FilterProjectTests {

        @Test
        void filterProject_basicFusedFilterAndProject() throws Throwable {
            try (ArrowBulkEvaluator liquidAndVolatile = ArrowBulkEvaluator.compile("volume > 150.0");
                 ArrowBulkEvaluator riskScore = ArrowBulkEvaluator.compile("rsi * 2");
                 VectorSchemaRoot root = rootOf(
                         column("volume", 100.0, 200.0, 300.0),
                         column("rsi", 10.0, 20.0, 30.0))) {

                VectorSchemaRoot result = liquidAndVolatile.filterProject(root, riskScore, "risk_score");

                assertEquals(2, result.getRowCount());
                assertArrayEquals(new double[]{200.0, 300.0}, toArray((Float8Vector) result.getVector("volume")));
                assertArrayEquals(new double[]{40.0, 60.0}, toArray((Float8Vector) result.getVector("risk_score")));
                result.close();
            }
        }

        @Test
        void filterProject_projectionOnlyRunsOverSurvivingRows() throws Throwable {
            try (ArrowBulkEvaluator predicate = ArrowBulkEvaluator.compile("volume > 150.0");
                 ArrowBulkEvaluator projection = ArrowBulkEvaluator.compile("rsi * 2");
                 VectorSchemaRoot root = rootOf(
                         column("volume", 100.0, 200.0, 300.0, 50.0, 400.0), // 5 rows in, 3 survive
                         column("rsi", 1.0, 2.0, 3.0, 4.0, 5.0))) {

                RowCountRecordingEvaluator recordingProjection = new RowCountRecordingEvaluator(projection);

                VectorSchemaRoot result = predicate.filterProject(root, recordingProjection, "risk_score");

                assertEquals(3, result.getRowCount());
                // The key fusion guarantee: projection saw only the 3 surviving rows, not all 5.
                assertEquals(3, recordingProjection.lastEvaluatedRowCount());
                result.close();
            }
        }

        @Test
        void filterProject_emptyResultWhenNoRowsSurvivePredicate() throws Throwable {
            try (ArrowBulkEvaluator none = ArrowBulkEvaluator.compile("x > 1000.0");
                 ArrowBulkEvaluator projection = ArrowBulkEvaluator.compile("x * 2");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0, 3.0))) {

                VectorSchemaRoot result = none.filterProject(root, projection, "doubled");

                assertEquals(0, result.getRowCount());
                // The projected column still exists in the schema, just with zero rows.
                assertEquals(0, result.getVector("doubled").getValueCount());
                result.close();
            }
        }

        @Test
        void filterProject_throwsWhenOutputFieldNameAlreadyExists() throws Throwable {
            try (ArrowBulkEvaluator predicate = ArrowBulkEvaluator.compile("x > 0.0");
                 ArrowBulkEvaluator projection = ArrowBulkEvaluator.compile("x * 2");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0))) {

                assertThrows(ArrowBindingException.class,
                        () -> predicate.filterProject(root, projection, "x"));
            }
        }

        @Test
        void filterProject_throwsNullPointerException_whenProjectionIsNull() throws Throwable {
            try (ArrowBulkEvaluator predicate = ArrowBulkEvaluator.compile("x > 0.0");
                 VectorSchemaRoot root = rootOf(column("x", 1.0, 2.0))) {

                assertThrows(NullPointerException.class,
                        () -> predicate.filterProject(root, null, "y"));
            }
        }

        @Test
        void filterProject_withPropagatePolicy_appliesToPredicateAndProjectionStages() throws Throwable {
            try (ArrowBulkEvaluator predicate = ArrowBulkEvaluator.compile("x > 0.0");
                 ArrowBulkEvaluator projection = ArrowBulkEvaluator.compile("y * 2");
                 VectorSchemaRoot root = rootOf(
                         columnWithNullAt("x", 0, 5.0, 6.0, 7.0),      // row 0's predicate input is null
                         columnWithNullAt("y", 2, 10.0, 20.0, 30.0))) { // row 2's projection input is null

                // Row 0 is dropped by PROPAGATE at the filter stage (null predicate -> excluded).
                // Of the two rows that survive (1, 2), row 2's projected value is null because
                // its own "y" input is null -- validated via the result's validity bitmap.
                VectorSchemaRoot result = predicate.filterProject(root, projection, "y2", NullPolicy.PROPAGATE);

                assertEquals(2, result.getRowCount());
                Float8Vector y2 = (Float8Vector) result.getVector("y2");
                assertEquals(40.0, y2.get(0)); // from original row 1: y=20 -> 40
                assertEquals(true, y2.isNull(1)); // from original row 2: y was null -> propagated
                result.close();
            }
        }

        @Test
        @EnabledIfSystemProperty(named = "gpu.tests", matches = "true")
        void filterProject_predicateAndProjectionCanUseDifferentBackends() throws Throwable {
            assumeTrue(ArrowGpuBulkEvaluator.isAnyGpuAvailable(),
                    "No GPU backend (CUDA/OpenCL) available on this machine");

            try (ArrowExpressionEvaluator cpuPredicate =
                         ArrowExpressionEvaluators.compile("volume > 150.0", ArrowExecutionBackend.CPU_SIMD);
                 ArrowExpressionEvaluator gpuProjection =
                         ArrowExpressionEvaluators.compile("rsi * 2", ArrowExecutionBackend.GPU_AUTO);
                 VectorSchemaRoot root = rootOf(
                         column("volume", 100.0, 200.0, 300.0),
                         column("rsi", 10.0, 20.0, 30.0))) {

                VectorSchemaRoot result = cpuPredicate.filterProject(root, gpuProjection, "risk_score");

                assertEquals(2, result.getRowCount());
                assertArrayEquals(new double[]{40.0, 60.0}, toArray((Float8Vector) result.getVector("risk_score")));
                result.close();
            }
        }
    }
}