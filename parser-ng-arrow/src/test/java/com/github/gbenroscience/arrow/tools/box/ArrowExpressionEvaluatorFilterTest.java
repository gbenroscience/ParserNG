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

import com.github.gbenroscience.parser.MathExpression;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests the Arrow filter() contract exposed by ArrowExpressionEvaluator.
 *
 * <p>The tests intentionally exercise the public evaluator API rather than
 * ArrowFilterSupport directly. This ensures that both CPU and GPU evaluators
 * are tested through their complete predicate -> selection -> materialization
 * pipeline.</p>
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ArrowExpressionEvaluatorFilterTest {

    private BufferAllocator allocator;

    @BeforeEach
    void setUp() {
        allocator = new RootAllocator(Long.MAX_VALUE);
    }

    @AfterEach
    void tearDown() {
        if (allocator != null) {
            allocator.close();
        }
    }

    // -------------------------------------------------------------------------
    // F64
    // -------------------------------------------------------------------------

    @Test
    void testFilterFloat64Basic() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{1, 2, 3, 4, 5},
                new double[]{10, 20, 30, 40, 50})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x > 2", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(3, result.getRowCount());

                    Float8Vector x = (Float8Vector) result.getVector("x");
                    Float8Vector y = (Float8Vector) result.getVector("y");

                    assertArrayEquals(
                            new double[]{3, 4, 5},
                            values(x),
                            0.0);

                    assertArrayEquals(
                            new double[]{30, 40, 50},
                            values(y),
                            0.0);
                }
            }
        }
    }

    @Test
    void testFilterFloat64CompoundPredicate() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{1, 2, 3, 4, 5, 6},
                new double[]{10, 60, 20, 70, 30, 80})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x > 2 && y > 50", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(2, result.getRowCount());

                    assertArrayEquals(
                            new double[]{4, 6},
                            values((Float8Vector) result.getVector("x")),
                            0.0);

                    assertArrayEquals(
                            new double[]{70, 80},
                            values((Float8Vector) result.getVector("y")),
                            0.0);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // F32
    // -------------------------------------------------------------------------

    @Test
    void testFilterFloat32Basic() throws Throwable {
        try (VectorSchemaRoot root = createFloatRoot(
                new float[]{1, 2, 3, 4, 5},
                new float[]{10, 20, 30, 40, 50})) {

            try (ArrowExpressionEvaluator evaluator =
                         ArrowExpressionEvaluators.compileF32(new MathExpression("x > 2"))) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(3, result.getRowCount());

                    assertArrayEquals(
                            new float[]{3, 4, 5},
                            values((Float4Vector) result.getVector("x")),
                            0.0f);

                    assertArrayEquals(
                            new float[]{30, 40, 50},
                            values((Float4Vector) result.getVector("y")),
                            0.0f);
                }
            }
        }
    }

    @Test
    void testFilterFloat32CompoundPredicate() throws Throwable {
        try (VectorSchemaRoot root = createFloatRoot(
                new float[]{1, 2, 3, 4, 5, 6},
                new float[]{10, 60, 20, 70, 30, 80})) {

            try (ArrowExpressionEvaluator evaluator =
                         ArrowExpressionEvaluators.compileF32("x > 2 && y > 50")) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(2, result.getRowCount());

                    assertArrayEquals(
                            new float[]{4, 6},
                            values((Float4Vector) result.getVector("x")),
                            0.0f);

                    assertArrayEquals(
                            new float[]{70, 80},
                            values((Float4Vector) result.getVector("y")),
                            0.0f);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Selection edge cases
    // -------------------------------------------------------------------------

    @Test
    void testFilterSelectsNoRows() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{1, 2, 3, 4},
                new double[]{10, 20, 30, 40})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x > 100", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(0, result.getRowCount());

                    assertEquals(2, result.getFieldVectors().size());
                    assertNotNull(result.getVector("x"));
                    assertNotNull(result.getVector("y"));

                    assertEquals(0, result.getVector("x").getValueCount());
                    assertEquals(0, result.getVector("y").getValueCount());
                }
            }
        }
    }

    @Test
    void testFilterSelectsAllRows() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{1, 2, 3, 4},
                new double[]{10, 20, 30, 40})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x > 0", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(4, result.getRowCount());

                    assertArrayEquals(
                            new double[]{1, 2, 3, 4},
                            values((Float8Vector) result.getVector("x")),
                            0.0);

                    assertArrayEquals(
                            new double[]{10, 20, 30, 40},
                            values((Float8Vector) result.getVector("y")),
                            0.0);
                }
            }
        }
    }

    @Test
    void testFilterSingleMatchingRow() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{10, 20, 30, 40},
                new double[]{100, 200, 300, 400})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x == 30", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(1, result.getRowCount());

                    assertEquals(
                            30.0,
                            ((Float8Vector) result.getVector("x")).get(0));

                    assertEquals(
                            300.0,
                            ((Float8Vector) result.getVector("y")).get(0));
                }
            }
        }
    }

    @Test
    void testFilterPreservesInputOrder() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{5, 1, 4, 2, 3},
                new double[]{50, 10, 40, 20, 30})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x >= 2", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertArrayEquals(
                            new double[]{5, 4, 2, 3},
                            values((Float8Vector) result.getVector("x")),
                            0.0);

                    assertArrayEquals(
                            new double[]{50, 40, 20, 30},
                            values((Float8Vector) result.getVector("y")),
                            0.0);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Zero / negative / truthiness
    // -------------------------------------------------------------------------

    @Test
    void testFilterZeroIsFalse() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{0, 1, -1, 0, 2},
                new double[]{10, 20, 30, 40, 50})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(3, result.getRowCount());

                    assertArrayEquals(
                            new double[]{1, -1, 2},
                            values((Float8Vector) result.getVector("x")),
                            0.0);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Null semantics
    // -------------------------------------------------------------------------

    @Test
    void testFilterNullPredicateIsExcludedWithPropagate() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRootWithNull(
                new double[]{1, 2, 3, 4},
                new boolean[]{false, true, false, false})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x > 0", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result =
                             evaluator.filter(root, NullPolicy.PROPAGATE)) {

                    /*
                     * Row 1 is NULL. Under predicate propagation its predicate
                     * result is NULL/UNKNOWN, therefore WHERE-style filtering
                     * must not select it.
                     */
                    assertEquals(3, result.getRowCount());

                    assertArrayEquals(
                            new double[]{1, 3, 4},
                            values((Float8Vector) result.getVector("x")),
                            0.0);
                }
            }
        }
    }

    @Test
    void testFilterNullPredicateNeverProducesNullOutputRows() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRootWithNull(
                new double[]{1, 2, 3},
                new boolean[]{true, false, false})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x > 0", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result =
                             evaluator.filter(root, NullPolicy.PROPAGATE)) {

                    assertEquals(2, result.getRowCount());

                    Float8Vector x =
                            (Float8Vector) result.getVector("x");

                    for (int i = 0; i < result.getRowCount(); i++) {
                        assertFalse(x.isNull(i));
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Schema preservation
    // -------------------------------------------------------------------------

    @Test
    void testFilterPreservesSchema() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{1, 2, 3},
                new double[]{10, 20, 30})) {

            Schema inputSchema = root.getSchema();

            try (ArrowExpressionEvaluator evaluator =
                         compile("x > 1", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(inputSchema, result.getSchema());
                    assertEquals(
                            root.getFieldVectors().size(),
                            result.getFieldVectors().size());

                    assertNotNull(result.getVector("x"));
                    assertNotNull(result.getVector("y"));
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Empty input
    // -------------------------------------------------------------------------

    @Test
    void testFilterEmptyBatch() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{},
                new double[]{})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x > 0", ArrowExecutionBackend.CPU_SIMD)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(0, result.getRowCount());
                    assertEquals(2, result.getFieldVectors().size());
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Backend coverage
    // -------------------------------------------------------------------------

    @Test
    void testFilterGpuAutoIfAvailable() throws Throwable {
        try (VectorSchemaRoot root = createDoubleRoot(
                new double[]{1, 2, 3, 4, 5},
                new double[]{10, 20, 30, 40, 50})) {

            try (ArrowExpressionEvaluator evaluator =
                         compile("x > 2", ArrowExecutionBackend.GPU_AUTO)) {

                try (VectorSchemaRoot result = evaluator.filter(root)) {

                    assertEquals(3, result.getRowCount());

                    assertArrayEquals(
                            new double[]{3, 4, 5},
                            values((Float8Vector) result.getVector("x")),
                            0.0);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private ArrowExpressionEvaluator compile(
            String expression,
            ArrowExecutionBackend backend) throws Throwable {

        return ArrowExpressionEvaluators.compile(
                new MathExpression(expression),
                backend);
    }

    private VectorSchemaRoot createDoubleRoot(
            double[] x,
            double[] y) {

        Float8Vector xVector = new Float8Vector("x", allocator);
        Float8Vector yVector = new Float8Vector("y", allocator);

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

    private VectorSchemaRoot createDoubleRootWithNull(
            double[] x,
            boolean[] nulls) {

        Float8Vector xVector = new Float8Vector("x", allocator);
        xVector.allocateNew(x.length);

        for (int i = 0; i < x.length; i++) {
            xVector.setSafe(i, x[i]);

            if (nulls[i]) {
                xVector.setNull(i);
            }
        }

        xVector.setValueCount(x.length);

        return VectorSchemaRoot.of(xVector);
    }

    private VectorSchemaRoot createFloatRoot(
            float[] x,
            float[] y) {

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

    private static double[] values(Float8Vector vector) {
        double[] result = new double[vector.getValueCount()];

        for (int i = 0; i < result.length; i++) {
            result[i] = vector.get(i);
        }

        return result;
    }

    private static float[] values(Float4Vector vector) {
        float[] result = new float[vector.getValueCount()];

        for (int i = 0; i < result.length; i++) {
            result[i] = vector.get(i);
        }

        return result;
    }
}