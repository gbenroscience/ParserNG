package com.github.gbenroscience.arrow.tools.box1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import java.util.ArrayList;
import java.util.List;

@DisplayName("ArrowBulkEvaluator Test Suite")
public class ArrowBulkEvaluatorTest {

    private static final double EPSILON = 1e-6;
    private BufferAllocator allocator;

    @BeforeEach
    void setUp() {
        allocator = new RootAllocator(Long.MAX_VALUE);
    }

    @AfterEach
    void tearDown() {
        allocator.close();
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================
    private VectorSchemaRoot createRoot(String[] names, double[][] data) {
        int rowCount = data.length > 0 ? data[0].length : 0;
        List<Field> fields = new ArrayList<>();
        for (String name : names) {
            fields.add(Field.nullable(name,
                    new ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE)));
        }

        VectorSchemaRoot root = VectorSchemaRoot.create(new Schema(fields), allocator);
        root.allocateNew();

        for (int i = 0; i < names.length; i++) {
            Float8Vector vec = (Float8Vector) root.getVector(names[i]);
            for (int r = 0; r < rowCount; r++) {
                vec.setSafe(r, data[i][r]);
            }
        }
        root.setRowCount(rowCount);
        return root;
    }

    // =========================================================================
    // 1. BASIC ARITHMETIC OPERATIONS (10 TESTS)
    // =========================================================================
    @Nested
    @DisplayName("Basic Arithmetic Tests")
    class BasicArithmeticTests {

        @Test
        @DisplayName("Test 1: Vector Addition (x + y)")
        void testAddition() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + y").variables("x", "y").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(3, out.getValueCount());
                assertEquals(5.0, out.get(0), EPSILON);
                assertEquals(7.0, out.get(1), EPSILON);
                assertEquals(9.0, out.get(2), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 2: Vector Subtraction (x - y)")
        void testSubtraction() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{{10.0, 20.0}, {3.0, 5.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x - y").variables("x", "y").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(7.0, out.get(0), EPSILON);
                assertEquals(15.0, out.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 3: Vector Multiplication (x * y)")
        void testMultiplication() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{{2.5, 4.0}, {2.0, 3.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * y").variables("x", "y").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(5.0, out.get(0), EPSILON);
                assertEquals(12.0, out.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 4: Vector Division (x / y)")
        void testDivision() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{{10.0, 9.0}, {2.0, 3.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x / y").variables("x", "y").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(5.0, out.get(0), EPSILON);
                assertEquals(3.0, out.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 5: Scalar Addition (x + 10.5)")
        void testScalarAddition() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.5, 3.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 10.5").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(12.0, out.get(0), EPSILON);
                assertEquals(13.5, out.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 6: Operator Precedence ((x + y) * z)")
        void testPrecedence() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y", "z"}, new double[][]{{2.0}, {3.0}, {4.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("(x + y) * z").variables("x", "y", "z").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(20.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 7: Unary Minus (-x)")
        void testUnaryMinus() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{5.0, -12.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("-x").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(-5.0, out.get(0), EPSILON);
                assertEquals(12.0, out.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 8: Three Variables Addition (a + b + c)")
        void testThreeVariables() {
            try (VectorSchemaRoot root = createRoot(new String[]{"a", "b", "c"}, new double[][]{{1.0}, {2.0}, {3.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("a + b + c").variables("a", "b", "c").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(6.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 9: Four Variables Composite Expression (a * b + c * d)")
        void testFourVariables() {
            try (VectorSchemaRoot root = createRoot(new String[]{"a", "b", "c", "d"}, new double[][]{{2.0}, {3.0}, {4.0}, {5.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("a * b + c * d").variables("a", "b", "c", "d").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(26.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 10: Negative Constant Multiplication (x * -2.5)")
        void testNegativeConstant() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{4.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * -2.5").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(-10.0, out.get(0), EPSILON);
            }
        }
    }

    // =========================================================================
    // 2. MATHEMATICAL & TRANSCENDENTAL FUNCTIONS (8 TESTS)
    // =========================================================================
    @Nested
    @DisplayName("Math & Transcendental Function Tests")
    class MathFunctionTests {

        @Test
        @DisplayName("Test 11: Trigonometric Sine (sin(x))")
        void testSine() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{0.0, Math.PI / 2}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("sin(x)").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(0.0, out.get(0), EPSILON);
                assertEquals(1.0, out.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 12: Trigonometric Cosine (cos(x))")
        void testCosine() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{0.0, Math.PI}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("cos(x)").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(1.0, out.get(0), EPSILON);
                assertEquals(-1.0, out.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 13: Square Root (sqrt(x))")
        void testSqrt() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{16.0, 81.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("sqrt(x)").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(4.0, out.get(0), EPSILON);
                assertEquals(9.0, out.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 14: Absolute Value (abs(x))")
        void testAbs() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{-15.5, 10.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("abs(x)").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(15.5, out.get(0), EPSILON);
                assertEquals(10.0, out.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 15: Trigonometric Identity (sin(x)^2 + cos(x)^2)")
        void testTrigIdentity() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{0.5, 1.2, 2.5}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("sin(x)*sin(x) + cos(x)*cos(x)").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                for (int i = 0; i < 3; i++) {
                    assertEquals(1.0, out.get(i), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("Test 16: Benchmark Expression (3*sin(x)*cos(y)+sqrt(abs(x*y)))")
        void testBenchmarkExpression() {
            double x = 0.5;
            double y = 1.0;
            double expected = 3 * Math.sin(x) * Math.cos(y) + Math.sqrt(Math.abs(x * y));
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{{x}, {y}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("3*sin(x)*cos(y)+sqrt(abs(x*y))").variables("x", "y").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(expected, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 17: Nested Transcendental Functions (sqrt(abs(sin(x))))")
        void testNestedFunctions() {
            double x = -Math.PI / 6; // sin(-PI/6) = -0.5 -> abs = 0.5 -> sqrt = 0.7071067
            double expected = Math.sqrt(Math.abs(Math.sin(x)));
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{x}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("sqrt(abs(sin(x)))").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(expected, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 18: Combination of Arithmetic and Sqrt")
        void testPolynomialSqrt() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{{3.0}, {4.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("sqrt(x*x + y*y)").variables("x", "y").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(5.0, out.get(0), EPSILON);
            }
        }
    }

    // =========================================================================
    // 3. BATCH SIZES & SIMD BOUNDARY TESTING (6 TESTS)
    // =========================================================================
    @Nested
    @DisplayName("Batch Size & Boundary Tests")
    class BatchSizeTests {

        @Test
        @DisplayName("Test 19: Empty Vector Batch (rowCount = 0)")
        void testEmptyBatch() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{{}, {}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + y").variables("x", "y").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(0, out.getValueCount());
            }
        }

        @Test
        @DisplayName("Test 20: Single Row Batch (rowCount = 1)")
        void testSingleRowBatch() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{42.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 2").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(1, out.getValueCount());
                assertEquals(84.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 21: Exact SIMD Vector Species Alignment (rowCount = 8)")
        void testSIMDLaneAlignment() {
            double[] inputX = new double[8];
            double[] inputY = new double[8];
            for (int i = 0; i < 8; i++) {
                inputX[i] = i;
                inputY[i] = i * 2;
            }
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{inputX, inputY}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + y").variables("x", "y").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(8, out.getValueCount());
                for (int i = 0; i < 8; i++) {
                    assertEquals(i * 3.0, out.get(i), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("Test 22: Unaligned SIMD Batch Size (rowCount = 17)")
        void testUnalignedBatch() {
            double[] inputX = new double[17];
            for (int i = 0; i < 17; i++) {
                inputX[i] = i;
            }
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{inputX}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 1").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(17, out.getValueCount());
                for (int i = 0; i < 17; i++) {
                    assertEquals(i + 1.0, out.get(i), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("Test 23: Medium Dataset Batch (rowCount = 1,024)")
        void testMediumBatch() {
            int size = 1024;
            double[] inputX = new double[size];
            for (int i = 0; i < size; i++) {
                inputX[i] = i * 0.5;
            }
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{inputX}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 2").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(size, out.getValueCount());
                for (int i = 0; i < size; i++) {
                    assertEquals((double) i, out.get(i), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("Test 24: Large High-Throughput Batch (rowCount = 50,000)")
        void testLargeBatch() {
            int size = 50_000;
            double[] inputX = new double[size];
            double[] inputY = new double[size];
            for (int i = 0; i < size; i++) {
                inputX[i] = i;
                inputY[i] = 10.0;
            }
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{inputX, inputY}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x - y").variables("x", "y").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(size, out.getValueCount());
                assertEquals(-10.0, out.get(0), EPSILON);
                assertEquals(49989.0, out.get(size - 1), EPSILON);
            }
        }
    }

    // =========================================================================
    // 4. PARALLEL VS SEQUENTIAL EVALUATION (6 TESTS)
    // =========================================================================
    @Nested
    @DisplayName("Parallel Execution Tests")
    class ParallelExecutionTests {

        @Test
        @DisplayName("Test 25: Sequential Mode Evaluation (parallel = false)")
        void testSequentialMode() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.0, 2.0, 3.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 3").variables("x").parallel(false).build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(3.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 26: Parallel Mode Evaluation (parallel = true)")
        void testParallelMode() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.0, 2.0, 3.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 3").variables("x").parallel(true).build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(3.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 27: Parallel Execution Parity With Sequential Mode")
        void testParallelSequentialParity() {
            int size = 20_000;
            double[] data = new double[size];
            for (int i = 0; i < size; i++) {
                data[i] = i + 1.0;
            }
            try (VectorSchemaRoot root1 = createRoot(new String[]{"x"}, new double[][]{data}); VectorSchemaRoot root2 = createRoot(new String[]{"x"}, new double[][]{data}); ArrowBulkEvaluator seqEval = ArrowBulkEvaluator.builder("sin(x) + cos(x)").variables("x").parallel(false).build(); ArrowBulkEvaluator parEval = ArrowBulkEvaluator.builder("sin(x) + cos(x)").variables("x").parallel(true).build(); Float8Vector seqOut = seqEval.evaluate(root1, allocator); Float8Vector parOut = parEval.evaluate(root2, allocator)) {
                for (int i = 0; i < size; i += 100) {
                    assertEquals(seqOut.get(i), parOut.get(i), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("Test 28: Parallel Mode on Small Batch")
        void testParallelSmallBatch() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{5.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 5").variables("x").parallel(true).build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(10.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 29: Consecutive Parallel Evaluations")
        void testConsecutiveParallelEvaluations() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{2.0, 4.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 2").variables("x").parallel(true).build()) {
                Float8Vector out1 = evaluator.evaluate(root, allocator);
                assertEquals(4.0, out1.get(0), EPSILON);
                out1.close();
                Float8Vector out2 = evaluator.evaluate(root, allocator);
                assertEquals(4.0, out2.get(0), EPSILON);
                out2.close();
            }
        }

        @Test
        @DisplayName("Test 30: Large Batch Parallel Workload")
        void testParallelLargeWorkload() {
            int size = 100_000;
            double[] data = new double[size];
            for (int i = 0; i < size; i++) {
                data[i] = i * 0.01;
            }
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{data}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("abs(x) + 1.0").variables("x").parallel(true).build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(size, out.getValueCount());
                assertEquals(1000.99, out.get(99_999), EPSILON);
            }
        }
    }

    // =========================================================================
    // 5. EVALUATE VS EVALUATEINTO ZERO-COPY REUSE (5 TESTS)
    // =========================================================================
    @Nested
    @DisplayName("Evaluation Method & Vector Reuse Tests")
    class EvaluationMethodTests {

        @Test
        @DisplayName("Test 31: evaluate() Allocates New Output Vector")
        void testEvaluateAllocatesNew() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{10.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x / 2").variables("x").build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertNotNull(out);
                assertEquals(1, out.getValueCount());
                assertEquals(5.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 32: evaluateInto() Writes Into Existing Vector")
        void testEvaluateIntoExisting() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{10.0, 20.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 2").variables("x").build(); Float8Vector preAllocated = new Float8Vector("out", allocator)) {
                preAllocated.allocateNew(2);
                preAllocated.setValueCount(2);
                evaluator.evaluateInto(root, preAllocated);
                assertEquals(20.0, preAllocated.get(0), EPSILON);
                assertEquals(40.0, preAllocated.get(1), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 33: evaluateInto() Buffer Reuse Across Multiple Batches")
        void testEvaluateIntoBufferReuse() {
            try (VectorSchemaRoot batch1 = createRoot(new String[]{"x"}, new double[][]{{2.0}}); VectorSchemaRoot batch2 = createRoot(new String[]{"x"}, new double[][]{{5.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 1").variables("x").build(); Float8Vector outputBuffer = new Float8Vector("out", allocator)) {
                outputBuffer.allocateNew(1);
                outputBuffer.setValueCount(1);
                evaluator.evaluateInto(batch1, outputBuffer);
                assertEquals(3.0, outputBuffer.get(0), EPSILON);
                evaluator.evaluateInto(batch2, outputBuffer);
                assertEquals(6.0, outputBuffer.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 34: evaluateInto() Correctly Overwrites Stale Buffer Data")
        void testEvaluateIntoOverwritesStaleData() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{3.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 2").variables("x").build(); Float8Vector outputBuffer = new Float8Vector("out", allocator)) {
                outputBuffer.allocateNew(1);
                outputBuffer.set(0, 999.99); // Stale value
                outputBuffer.setValueCount(1);
                evaluator.evaluateInto(root, outputBuffer);
                assertEquals(6.0, outputBuffer.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 35: evaluateInto() Sets Correct Output Value Count")
        void testEvaluateIntoSetsValueCount() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.0, 2.0, 3.0, 4.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x").variables("x").build(); Float8Vector outputBuffer = new Float8Vector("out", allocator)) {
                outputBuffer.allocateNew(10); // Over-allocated
                evaluator.evaluateInto(root, outputBuffer);
                assertEquals(4, outputBuffer.getValueCount());
            }
        }
    }

    // =========================================================================
    // 6. NULL POLICY & VALIDITY MASK HANDLING (8 TESTS)
    // =========================================================================
    @Nested
    @DisplayName("Null Policy & Validity Mask Tests")
    class NullPolicyTests {

        @Test
        @DisplayName("Test 36: REJECT_ON_NULL Policy Passes For Completely Valid Vectors")
        void testRejectOnNullPassesWhenValid() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.0, 2.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 1")
                    .variables("x")
                    .nullPolicy(NullPolicy.REJECT_ON_NULL)
                    .build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertEquals(2.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 37: REJECT_ON_NULL Policy Throws Exception On First Index Null")
        void testRejectOnNullThrowsOnFirstRowNull() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.0, 2.0}})) {
                Float8Vector vec = (Float8Vector) root.getVector("x");
                vec.setNull(0); // Set index 0 to null
                try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 1")
                        .variables("x")
                        .nullPolicy(NullPolicy.REJECT_ON_NULL)
                        .build()) {
                    assertThrows(ArrowNullValueException.class, () -> evaluator.evaluate(root, allocator));
                }
            }
        }

        @Test
        @DisplayName("Test 38: REJECT_ON_NULL Policy Throws Exception On Middle Index Null")
        void testRejectOnNullThrowsOnMiddleRowNull() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.0, 2.0, 3.0}})) {
                Float8Vector vec = (Float8Vector) root.getVector("x");
                vec.setNull(1); // Set index 1 to null
                try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 1")
                        .variables("x")
                        .nullPolicy(NullPolicy.REJECT_ON_NULL)
                        .build()) {
                    assertThrows(ArrowNullValueException.class, () -> evaluator.evaluate(root, allocator));
                }
            }
        }

        @Test
        @DisplayName("Test 39: REJECT_ON_NULL Throws When Null Exists In Second Vector")
        void testRejectOnNullThrowsInSecondVector() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{{1.0, 2.0}, {3.0, 4.0}})) {
                Float8Vector yVec = (Float8Vector) root.getVector("y");
                yVec.setNull(1);
                try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + y")
                        .variables("x", "y")
                        .nullPolicy(NullPolicy.REJECT_ON_NULL)
                        .build()) {
                    assertThrows(ArrowNullValueException.class, () -> evaluator.evaluate(root, allocator));
                }
            }
        }

        @Test
        @DisplayName("Test 40: PROPAGATE_NULL Policy Leaves Valid Rows Intact")
        void testPropagateNullLeavesValidRowsIntact() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{10.0, 20.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 2")
                    .variables("x")
                    .nullPolicy(NullPolicy.PROPAGATE_NULL)
                    .build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                assertFalse(out.isNull(0));
                assertFalse(out.isNull(1));
                assertEquals(20.0, out.get(0), EPSILON);
            }
        }

        @Test
        @DisplayName("Test 41: PROPAGATE_NULL Propagates Single Vector Nulls To Output")
        void testPropagateNullSingleVector() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{10.0, 20.0, 30.0}})) {
                Float8Vector vec = (Float8Vector) root.getVector("x");
                vec.setNull(1); // Row 1 is null
                try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 5")
                        .variables("x")
                        .nullPolicy(NullPolicy.PROPAGATE_NULL)
                        .build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                    assertFalse(out.isNull(0));
                    assertTrue(out.isNull(1)); // Must be null in output
                    assertFalse(out.isNull(2));
                    assertEquals(15.0, out.get(0), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("Test 42: PROPAGATE_NULL Bitwise-ANDs Nulls Across Multiple Vectors")
        void testPropagateNullMultipleVectors() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x", "y"}, new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})) {
                Float8Vector xVec = (Float8Vector) root.getVector("x");
                Float8Vector yVec = (Float8Vector) root.getVector("y");
                xVec.setNull(0); // Null at row 0 in x
                yVec.setNull(1); // Null at row 1 in y
                try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + y")
                        .variables("x", "y")
                        .nullPolicy(NullPolicy.PROPAGATE_NULL)
                        .build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                    assertTrue(out.isNull(0));  // x was null
                    assertTrue(out.isNull(1));  // y was null
                    assertFalse(out.isNull(2)); // Both valid -> 3 + 6 = 9
                    assertEquals(9.0, out.get(2), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("Test 43: PROPAGATE_NULL Correctly Handles All-Null Vector")
        void testPropagateNullAllNullVector() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.0, 2.0, 3.0}})) {
                Float8Vector vec = (Float8Vector) root.getVector("x");
                vec.setNull(0);
                vec.setNull(1);
                vec.setNull(2);
                try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 10")
                        .variables("x")
                        .nullPolicy(NullPolicy.PROPAGATE_NULL)
                        .build(); Float8Vector out = evaluator.evaluate(root, allocator)) {
                    assertTrue(out.isNull(0));
                    assertTrue(out.isNull(1));
                    assertTrue(out.isNull(2));
                }
            }
        }
    }

    // =========================================================================
    // 7. ERRORS, BINDING VALIDATION & LIFECYCLE (7 TESTS)
    // =========================================================================
    @Nested
    @DisplayName("Validation, Errors & Lifecycle Tests")
    class ErrorHandlingAndLifecycleTests {

        @Test
        @DisplayName("Test 44: Builder Fails When variables() Is Not Specified")
        void testBuilderFailsWithoutVariables() {
            Exception exception = assertThrows(ArrowBindingException.class, () -> ArrowBulkEvaluator.builder("x + 1").build());
            assertTrue(exception.getMessage().contains("variables(...) must be called"));
        }

        @Test
        @DisplayName("Test 45: Builder Fails On Invalid Expression Syntax")
        void testBuilderFailsOnInvalidSyntax() {
            assertThrows(ArrowBindingException.class, () -> ArrowBulkEvaluator.builder("x +++ 123**").variables("x").build());
        }

        @Test
        @DisplayName("Test 46: Smoke Test Detects Variable Count Mismatch")
        void testSmokeTestDetectsVariableCountMismatch() {
            // Expression expects 2 variables (x, y), but builder provides 1
            Exception exception = assertThrows(ArrowBindingException.class, () -> ArrowBulkEvaluator.builder("x + y").variables("x").build());
            assertTrue(exception.getMessage().contains("expects a different count"));
        }

        @Test
        @DisplayName("Test 47: Missing Column In VectorSchemaRoot Throws ArrowBindingException")
        void testMissingColumnInRootThrowsException() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.0}}); ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + y").variables("x", "y").build()) {
                Exception exception = assertThrows(ArrowBindingException.class, () -> evaluator.evaluate(root, allocator));
                assertTrue(exception.getMessage().contains("No column named 'y' found"));
            }
        }

        @Test
        @DisplayName("Test 48: Non-Float8Vector Column Throws UnsupportedVectorTypeException")
        void testUnsupportedVectorTypeThrowsException() {
            // IntVector, BigIntVector, and Float4Vector are all now
            // auto-coerced by ArrowBulkEvaluator via VectorCoercion (see
            // ArrowBulkEvaluatorTestBattery's coercion tests), so they no
            // longer exercise this path. VarCharVector has no numeric
            // coercion defined for it and remains genuinely unsupported.
            Field varCharField = Field.nullable("x", new ArrowType.Utf8());
            try (VectorSchemaRoot root = VectorSchemaRoot.create(new Schema(List.of(varCharField)), allocator)) {
                root.allocateNew();
                VarCharVector varCharVec = (VarCharVector) root.getVector("x");
                varCharVec.setSafe(0, "10".getBytes(java.nio.charset.StandardCharsets.UTF_8));
                root.setRowCount(1);
                try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 1").variables("x").build()) {
                    assertThrows(UnsupportedVectorTypeException.class, () -> evaluator.evaluate(root, allocator));
                }
            }
        }

        @Test
        @DisplayName("Test 49: Evaluating Closed Evaluator Throws IllegalStateException")
        void testEvaluateOnClosedEvaluatorThrowsException() {
            try (VectorSchemaRoot root = createRoot(new String[]{"x"}, new double[][]{{1.0}})) {
                ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x + 1").variables("x").build();
                evaluator.close();
                Exception exception = assertThrows(IllegalStateException.class, () -> evaluator.evaluate(root, allocator));
                assertTrue(exception.getMessage().contains("is closed"));
            }
        }

        @Test
        @DisplayName("Test 50: Try-With-Resources Safely Closes Engine Resources")
        void testAutoCloseableSupport() {
            ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("x * 2").variables("x").build();
            evaluator.close();
            assertThrows(IllegalStateException.class, () -> evaluator.evaluateInto(null, null));
        }
    }
}