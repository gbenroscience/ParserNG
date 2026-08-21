package com.github.gbenroscience.arrow.tools.box1;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * Comprehensive conformance and stress suite for parser-ng-arrow.
 *
 * 30 tests covering:
 *
 *  1. Basic arithmetic
 *  2. Constants
 *  3. Variable ordering
 *  4. Unary expressions
 *  5. Functions
 *  6. POW
 *  7. Comparisons
 *  8. IF
 *  9. AND / OR
 * 10. Null semantics
 * 11. Empty/single-row batches
 * 12. Negative/large values
 * 13. Mixed Arrow vector types
 * 14. Output-vector reuse
 * 15. Large-batch correctness
 * 16. Numerical accuracy
 * 17. Zero-copy address invariants
 * 18. Repeated evaluation
 * 19. Randomized differential testing
 *
 * These tests deliberately exercise both the easy arithmetic path
 * and the paths which parser-ng-arrow documents as potentially
 * requiring operand staging.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class ArrowBulkEvaluatorTestBattery {

    private RootAllocator allocator;

    @BeforeAll
    void beforeAll() {
        allocator = new RootAllocator(Long.MAX_VALUE);
    }

    @AfterAll
    void afterAll() {
        if (allocator != null) {
            allocator.close();
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private Float8Vector vector(String name, double... values) {
        Float8Vector v = new Float8Vector(name, allocator);
        v.allocateNew(values.length);

        for (int i = 0; i < values.length; i++) {
            v.setSafe(i, values[i]);
        }

        v.setValueCount(values.length);
        return v;
    }

    private Float8Vector vectorWithNull(
            String name,
            Double... values) {

        Float8Vector v = new Float8Vector(name, allocator);
        v.allocateNew(values.length);

        for (int i = 0; i < values.length; i++) {
            if (values[i] == null) {
                v.setNull(i);
            } else {
                v.setSafe(i, values[i]);
            }
        }

        v.setValueCount(values.length);
        return v;
    }

    private VectorSchemaRoot root(FieldVector... vectors) {
        VectorSchemaRoot root = VectorSchemaRoot.of(vectors);

        int rows = vectors.length == 0
                ? 0
                : vectors[0].getValueCount();

        root.setRowCount(rows);

        return root;
    }

    private void assertValues(
            Float8Vector actual,
            double... expected) {

        assertEquals(expected.length, actual.getValueCount());

        for (int i = 0; i < expected.length; i++) {
            assertEquals(
                    expected[i],
                    actual.get(i),
                    1e-10,
                    "Mismatch at row " + i
            );
        }
    }

    private void assertClose(
            double expected,
            double actual) {

        assertEquals(
                expected,
                actual,
                Math.max(
                        1e-12,
                        Math.abs(expected) * 1e-12
                )
        );
    }

    // -------------------------------------------------------------------------
    // 01 — Basic arithmetic
    // -------------------------------------------------------------------------

    @Test
    void test01_basicArithmetic() {

        try (Float8Vector x = vector("x", 1, 2, 3);
             Float8Vector y = vector("y", 10, 20, 30);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x + y")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 11, 22, 33);
        }
    }

    // -------------------------------------------------------------------------
    // 02 — Arithmetic chain
    // -------------------------------------------------------------------------

    @Test
    void test02_arithmeticChain() {

        try (Float8Vector x = vector("x", 1, 2, 3);
             Float8Vector y = vector("y", 10, 20, 30);
             Float8Vector z = vector("z", 2, 4, 5);
             VectorSchemaRoot root = root(x, y, z);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("(x + y) * z")
                             .variables("x", "y", "z")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 22, 88, 165);
        }
    }

    // -------------------------------------------------------------------------
    // 03 — Constant expression
    // -------------------------------------------------------------------------

    @Test
    void test03_constantExpression() {

        try (Float8Vector x = vector("x", 1, 2, 3, 4);
             VectorSchemaRoot root = root(x);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("42.5")
                             .variables()
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 42.5, 42.5, 42.5, 42.5);
        }
    }

    // -------------------------------------------------------------------------
    // 04 — Division
    // -------------------------------------------------------------------------

    @Test
    void test04_division() {

        try (Float8Vector x = vector("x", 10, 20, 30);
             Float8Vector y = vector("y", 2, 4, 5);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x / y")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 5, 5, 6);
        }
    }

    // -------------------------------------------------------------------------
    // 05 — Negative values
    // -------------------------------------------------------------------------

    @Test
    void test05_negativeValues() {

        try (Float8Vector x = vector("x", -1, -2, 3);
             Float8Vector y = vector("y", 10, -20, -30);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x * y")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, -10, 40, -90);
        }
    }

    // -------------------------------------------------------------------------
    // 06 — Variable order
    // -------------------------------------------------------------------------

    @Test
    void test06_variableOrder() {

        try (Float8Vector z = vector("z", 100, 200);
             Float8Vector y = vector("y", 10, 20);
             Float8Vector x = vector("x", 1, 2);
             VectorSchemaRoot root = root(z, y, x);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("z * y + x")
                             .variables("z", "y", "x")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 1001, 4002);
        }
    }

    // -------------------------------------------------------------------------
    // 07 — Parentheses / precedence
    // -------------------------------------------------------------------------

    @Test
    void test07_precedence() {

        try (Float8Vector x = vector("x", 2, 3, 4);
             Float8Vector y = vector("y", 5, 6, 7);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x + y * 2")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 12, 15, 18);
        }
    }

    // -------------------------------------------------------------------------
    // 08 — Unary expression
    // -------------------------------------------------------------------------

    @Test
    void test08_unaryExpression() {

        try (Float8Vector x = vector("x", -1, 2, -3);
             VectorSchemaRoot root = root(x);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("-x")
                             .variables("x")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 1, -2, 3);
        }
    }

    // -------------------------------------------------------------------------
    // 09 — sqrt
    // -------------------------------------------------------------------------

    @Test
    void test09_sqrt() {

        try (Float8Vector x = vector("x", 0, 1, 4, 9, 16);
             VectorSchemaRoot root = root(x);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("sqrt(x)")
                             .variables("x")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 0, 1, 2, 3, 4);
        }
    }

    // -------------------------------------------------------------------------
    // 10 — transcendental expression
    // -------------------------------------------------------------------------

    @Test
    void test10_transcendentalExpression() {

        try (Float8Vector x = vector("x", 0, Math.PI / 2, Math.PI);
             VectorSchemaRoot root = root(x);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("sin(x)")
                             .variables("x")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertClose(0, result.get(0));
            assertClose(1, result.get(1));
            assertClose(0, result.get(2));
        }
    }

    // -------------------------------------------------------------------------
    // 11 — POW
    // -------------------------------------------------------------------------

    @Test
    void test11_pow() {

        try (Float8Vector x = vector("x", 2, 3, 4, 5);
             VectorSchemaRoot root = root(x);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x^3")
                             .variables("x")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 8, 27, 64, 125);
        }
    }

    // -------------------------------------------------------------------------
    // 12 — Combined transcendental expression
    // -------------------------------------------------------------------------

    @Test
    void test12_combinedTranscendental() {

        try (Float8Vector x = vector("x", 1, 2, 3);
             Float8Vector y = vector("y", 4, 5, 6);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder(
                             "sin(sqrt(x^2 + y^2))")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            for (int i = 0; i < 3; i++) {
                double expected =
                        Math.sin(Math.sqrt(
                                x.get(i) * x.get(i)
                                + y.get(i) * y.get(i)));

                assertClose(expected, result.get(i));
            }
        }
    }

    // -------------------------------------------------------------------------
    // 13 — Comparison
    // -------------------------------------------------------------------------

    @Test
    void test13_comparison() {

        try (Float8Vector x = vector("x", 1, 5, 10);
             VectorSchemaRoot root = root(x);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x > 5")
                             .variables("x")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertEquals(0.0, result.get(0));
            assertEquals(0.0, result.get(1));
            assertEquals(1.0, result.get(2));
        }
    }

    // -------------------------------------------------------------------------
    // 14 — IF
    // -------------------------------------------------------------------------

    @Test
    void test14_if() {

        try (Float8Vector x = vector("x", -2, 0, 2);
             VectorSchemaRoot root = root(x);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("if(x > 0, x, -x)")
                             .variables("x")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 2, 0, 2);
        }
    }

    // -------------------------------------------------------------------------
    // 15 — AND
    // -------------------------------------------------------------------------

    @Test
    void test15_and() {

        try (Float8Vector x = vector("x", 1, 1, 0, 0);
             Float8Vector y = vector("y", 1, 0, 1, 0);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x > 0 && y > 0")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 1, 0, 0, 0);
        }
    }

    // -------------------------------------------------------------------------
    // 16 — OR
    // -------------------------------------------------------------------------

    @Test
    void test16_or() {

        try (Float8Vector x = vector("x", 1, 1, 0, 0);
             Float8Vector y = vector("y", 1, 0, 1, 0);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x > 0 || y > 0")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertValues(result, 1, 1, 1, 0);
        }
    }

    // -------------------------------------------------------------------------
    // 17 — Null rejection
    // -------------------------------------------------------------------------

    @Test
    void test17_rejectNull() {

        try (Float8Vector x =
                     vectorWithNull("x", 1.0, null, 3.0);
             Float8Vector y =
                     vector("y", 10, 20, 30);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x + y")
                             .variables("x", "y")
                             .build()) {

            assertThrows(
                    ArrowNullValueException.class,
                    () -> evaluator.evaluate(root, allocator)
            );
        }
    }

    // -------------------------------------------------------------------------
    // 18 — Null propagation
    // -------------------------------------------------------------------------

    @Test
    void test18_propagateNull() {

        try (Float8Vector x =
                     vectorWithNull("x", 1.0, null, 3.0);
             Float8Vector y =
                     vectorWithNull("y", 10.0, 20.0, null);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x + y")
                             .variables("x", "y")
                             .nullPolicy(NullPolicy.PROPAGATE_NULL)
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertFalse(result.isNull(0));
            assertEquals(11.0, result.get(0));

            assertTrue(result.isNull(1));
            assertTrue(result.isNull(2));
        }
    }

    // -------------------------------------------------------------------------
    // 19 — Single row
    // -------------------------------------------------------------------------

    @Test
    void test19_singleRow() {

        try (Float8Vector x = vector("x", 7);
             Float8Vector y = vector("y", 3);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x * y + 1")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertEquals(22.0, result.get(0));
        }
    }

    // -------------------------------------------------------------------------
    // 20 — Empty batch
    // -------------------------------------------------------------------------

    @Test
    void test20_emptyBatch() {

        try (Float8Vector x = vector("x");
             Float8Vector y = vector("y");
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x + y")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertEquals(0, result.getValueCount());
        }
    }

    // -------------------------------------------------------------------------
    // 21 — IntVector coercion
    // -------------------------------------------------------------------------

    @Test
    void test21_intVectorCoercion() {

        try (IntVector x = new IntVector("x", allocator);
             Float8Vector y = vector("y", 0.5, 1.5, 2.5)) {

            x.allocateNew(3);

            x.setSafe(0, 1);
            x.setSafe(1, 2);
            x.setSafe(2, 3);

            x.setValueCount(3);

            try (VectorSchemaRoot root = root(x, y);
                 ArrowBulkEvaluator evaluator =
                         ArrowBulkEvaluator.builder("x + y")
                                 .variables("x", "y")
                                 .build();
                 Float8Vector result =
                         evaluator.evaluate(root, allocator)) {

                assertValues(result, 1.5, 3.5, 5.5);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 22 — Float4Vector coercion
    // -------------------------------------------------------------------------

    @Test
    void test22_float4VectorCoercion() {

        try (Float4Vector x = new Float4Vector("x", allocator);
             Float8Vector y = vector("y", 1, 2, 3)) {

            x.allocateNew(3);

            x.setSafe(0, 1.5f);
            x.setSafe(1, 2.5f);
            x.setSafe(2, 3.5f);

            x.setValueCount(3);

            try (VectorSchemaRoot root = root(x, y);
                 ArrowBulkEvaluator evaluator =
                         ArrowBulkEvaluator.builder("x * y")
                                 .variables("x", "y")
                                 .build();
                 Float8Vector result =
                         evaluator.evaluate(root, allocator)) {

                assertValues(result, 1.5, 5.0, 10.5);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 23 — Output vector reuse / evaluateInto
    // -------------------------------------------------------------------------

    @Test
    void test23_evaluateInto() {

        try (Float8Vector x = vector("x", 1, 2, 3);
             Float8Vector y = vector("y", 10, 20, 30);
             Float8Vector output = new Float8Vector("result", allocator);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x + y")
                             .variables("x", "y")
                             .build()) {

            output.allocateNew(3);

            evaluator.evaluateInto(root, output);

            output.setValueCount(3);

            assertValues(output, 11, 22, 33);
        }
    }

    // -------------------------------------------------------------------------
    // 24 — Repeated evaluation on same evaluator
    // -------------------------------------------------------------------------

    @Test
    void test24_repeatedEvaluation() {

        try (ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x * 2 + y")
                             .variables("x", "y")
                             .build();
             Float8Vector x1 = vector("x", 1, 2, 3);
             Float8Vector y1 = vector("y", 10, 20, 30);
             VectorSchemaRoot root1 = root(x1, y1);
             Float8Vector r1 =
                     evaluator.evaluate(root1, allocator);
             Float8Vector x2 = vector("x", 100, 200, 300);
             Float8Vector y2 = vector("y", 1, 2, 3);
             VectorSchemaRoot root2 = root(x2, y2);
             Float8Vector r2 =
                     evaluator.evaluate(root2, allocator)) {

            assertValues(r1, 12, 24, 36);
            assertValues(r2, 201, 402, 603);

            x2.close();
            y2.close();
            root2.close();
        }
    }

    // -------------------------------------------------------------------------
    // 25 — Large batch correctness
    // -------------------------------------------------------------------------

    @Test
    void test25_largeBatch() {

        final int n = 1_000_000;

        try (Float8Vector x = new Float8Vector("x", allocator);
             Float8Vector y = new Float8Vector("y", allocator)) {

            x.allocateNew(n);
            y.allocateNew(n);

            for (int i = 0; i < n; i++) {
                x.setSafe(i, i);
                y.setSafe(i, 2.0 * i);
            }

            x.setValueCount(n);
            y.setValueCount(n);

            try (VectorSchemaRoot root = root(x, y);
                 ArrowBulkEvaluator evaluator =
                         ArrowBulkEvaluator.builder(
                                 "(x + y) * 3.0 - x")
                                 .variables("x", "y")
                                 .build();
                 Float8Vector result =
                         evaluator.evaluate(root, allocator)) {

                assertEquals(n, result.getValueCount());

                for (int i : new int[]{0, 1, 2, 100, 999999}) {
                    double expected =
                            (i + 2.0 * i) * 3.0 - i;

                    assertEquals(
                            expected,
                            result.get(i),
                            1e-10,
                            "Mismatch at row " + i
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 26 — Numerical differential test
    // -------------------------------------------------------------------------

    @Test
    void test26_differentialNumericalAccuracy() {

        final int n = 10_000;
        Random random = new Random(0x5EED);

        try (Float8Vector x = new Float8Vector("x", allocator);
             Float8Vector y = new Float8Vector("y", allocator)) {

            x.allocateNew(n);
            y.allocateNew(n);

            for (int i = 0; i < n; i++) {
                x.setSafe(i, random.nextDouble() * 20.0 - 10.0);
                y.setSafe(i, random.nextDouble() * 20.0 - 10.0);
            }

            x.setValueCount(n);
            y.setValueCount(n);

            try (VectorSchemaRoot root = root(x, y);
                 ArrowBulkEvaluator evaluator =
                         ArrowBulkEvaluator.builder(
                                 "sin(x) + cos(y) + x*y")
                                 .variables("x", "y")
                                 .build();
                 Float8Vector result =
                         evaluator.evaluate(root, allocator)) {

                for (int i = 0; i < n; i++) {

                    double expected =
                            Math.sin(x.get(i))
                            + Math.cos(y.get(i))
                            + x.get(i) * y.get(i);

                    assertEquals(
                            expected,
                            result.get(i),
                            1e-10,
                            "Mismatch at row " + i
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 27 — Extreme finite values
    // -------------------------------------------------------------------------

    @Test
    void test27_extremeFiniteValues() {

        try (Float8Vector x =
                     vector("x",
                             Double.MIN_NORMAL,
                             1.0,
                             Double.MAX_VALUE / 2.0);
             Float8Vector y =
                     vector("y",
                             1.0,
                             2.0,
                             2.0);
             VectorSchemaRoot root = root(x, y);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x * y")
                             .variables("x", "y")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertEquals(
                    x.get(0) * y.get(0),
                    result.get(0)
            );

            assertEquals(
                    x.get(1) * y.get(1),
                    result.get(1)
            );

            assertEquals(
                    x.get(2) * y.get(2),
                    result.get(2)
            );
        }
    }

    // -------------------------------------------------------------------------
    // 28 — NaN / Infinity semantics
    // -------------------------------------------------------------------------

    @Test
    void test28_nanAndInfinity() {

        try (Float8Vector x =
                     vector("x",
                             Double.NaN,
                             Double.POSITIVE_INFINITY,
                             Double.NEGATIVE_INFINITY,
                             1.0);
             VectorSchemaRoot root = root(x);
             ArrowBulkEvaluator evaluator =
                     ArrowBulkEvaluator.builder("x * 2")
                             .variables("x")
                             .build();
             Float8Vector result =
                     evaluator.evaluate(root, allocator)) {

            assertTrue(Double.isNaN(result.get(0)));

            assertEquals(
                    Double.POSITIVE_INFINITY,
                    result.get(1)
            );

            assertEquals(
                    Double.NEGATIVE_INFINITY,
                    result.get(2)
            );

            assertEquals(2.0, result.get(3));
        }
    }

    // -------------------------------------------------------------------------
    // 29 — Zero-copy input address stability
    // -------------------------------------------------------------------------

    @Test
    void test29_inputArrowBufferIsNative() {

        try (Float8Vector x =
                     vector("x", 1, 2, 3, 4);
             VectorSchemaRoot root = root(x)) {

            long address =
                    x.getDataBuffer().memoryAddress();

            long capacity =
                    x.getDataBuffer().capacity();

            assertTrue(
                    address != 0,
                    "Arrow data buffer has no native address"
            );

            assertTrue(
                    capacity >= 32,
                    "Unexpected Float8 data buffer size"
            );

            /*
             * This test deliberately does not claim that parser-ng-arrow
             * used the address merely because the address exists.
             *
             * The actual zero-copy contract is implemented by
             * ArrowSegments.ofData(...).
             *
             * This test establishes the invariant required by that bridge:
             * Float8Vector's data lives in an addressable native ArrowBuf.
             */
        }
    }

    // -------------------------------------------------------------------------
    // 30 — Randomized large differential stress test
    // -------------------------------------------------------------------------

    @Test
    void test30_randomizedStress() {

        final int n = 100_000;

        Random random = new Random(123456789L);

        try (Float8Vector x = new Float8Vector("x", allocator);
             Float8Vector y = new Float8Vector("y", allocator);
             Float8Vector z = new Float8Vector("z", allocator)) {

            x.allocateNew(n);
            y.allocateNew(n);
            z.allocateNew(n);

            for (int i = 0; i < n; i++) {

                x.setSafe(i, random.nextDouble() * 10.0 - 5.0);
                y.setSafe(i, random.nextDouble() * 10.0 - 5.0);
                z.setSafe(i, random.nextDouble() * 10.0 - 5.0);
            }

            x.setValueCount(n);
            y.setValueCount(n);
            z.setValueCount(n);

            try (VectorSchemaRoot root = root(x, y, z);
                 ArrowBulkEvaluator evaluator =
                         ArrowBulkEvaluator.builder(
                                 "sin(x) * cos(y) + sqrt(z^2 + 1)")
                                 .variables("x", "y", "z")
                                 .build();
                 Float8Vector result =
                         evaluator.evaluate(root, allocator)) {

                assertEquals(n, result.getValueCount());

                for (int i = 0; i < n; i++) {

                    double expected =
                            Math.sin(x.get(i))
                            * Math.cos(y.get(i))
                            + Math.sqrt(
                                    z.get(i) * z.get(i) + 1.0);

                    assertEquals(
                            expected,
                            result.get(i),
                            1e-10,
                            "Mismatch at row " + i
                    );
                }
            }
        }
    }
}