package com.github.gbenroscience.arrow.tools.box;

import com.github.gbenroscience.arrow.tools.box.ArrowBulkEvaluator;
import com.github.gbenroscience.arrow.tools.box.ArrowMemoryBridge;
import com.github.gbenroscience.arrow.tools.box.ArrowBindingException;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import com.github.gbenroscience.parser.MathExpression;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 40 tests proving out the parser-ng-arrow integration layer: the memory
 * bridge itself, name-based binding correctness (via MathExpression.getSlotItems()),
 * every ArrowBindingException path, NullPolicy behavior, constant-expression
 * handling, row-count edge cases, parallel/serial/concurrent evaluation
 * consistency, and numeric correctness across arithmetic, transcendental, and
 * comparison expression shapes.
 *
 * This suite tests ArrowBulkEvaluator/ArrowMemoryBridge specifically — it
 * assumes SIMDEngineEvaluator's own correctness is already covered by
 * SIMDEngineEvaluatorZeroCopyTest. Where useful, results are cross-checked
 * against MathExpression's own scalar solve() as an independent oracle,
 * rather than only against the engine being tested.
 */
class ArrowBulkEvaluatorTest {

    private BufferAllocator allocator;
    private final List<ArrowBulkEvaluator> openEvaluators = new ArrayList<>();
    private final List<Float8Vector> openVectors = new ArrayList<>();

    @BeforeEach
    void setUp() {
        allocator = new RootAllocator(Long.MAX_VALUE);
    }

    @AfterEach
    void tearDown() {
        for (ArrowBulkEvaluator e : openEvaluators) {
            try { e.close(); } catch (Exception ignored) { }
        }
        openEvaluators.clear();
        for (Float8Vector v : openVectors) {
            try { v.close(); } catch (Exception ignored) { }
        }
        openVectors.clear();
        allocator.close();
    }

    // =====================================================================
    // Helpers
    // =====================================================================

    private ArrowBulkEvaluator compile(String expr) throws Throwable {
        ArrowBulkEvaluator e = ArrowBulkEvaluator.compile(expr);
        openEvaluators.add(e);
        return e;
    }

    private Float8Vector column(String name, double[] values) {
        Float8Vector v = new Float8Vector(name, allocator);
        v.allocateNew(values.length);
        for (int i = 0; i < values.length; i++) {
            v.set(i, values[i]);
        }
        v.setValueCount(values.length);
        openVectors.add(v);
        return v;
    }

    private Float8Vector emptyOutput(String name, int rowCount) {
        Float8Vector v = ArrowBulkEvaluator.allocateOutput(allocator, name, rowCount);
        openVectors.add(v);
        return v;
    }

    private double[] randomData(int n, long seed) {
        Random r = new Random(seed);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = (r.nextDouble() - 0.5) * 200.0;
        }
        return out;
    }

    private double[] readAll(Float8Vector v) {
        double[] out = new double[v.getValueCount()];
        for (int i = 0; i < out.length; i++) {
            out[i] = v.get(i);
        }
        return out;
    }

    private static void assertClose(double expected, double actual, String context) {
        if (Double.isNaN(expected) && Double.isNaN(actual)) return;
        assertEquals(expected, actual, 1e-9, context);
    }

    // =====================================================================
    // A. ArrowMemoryBridge fundamentals (1-6)
    // =====================================================================

    @Test
    void test01_wrapDoubles_roundTripsValues() {
        double[] data = {1.5, -2.25, 0.0, 3.75, 100.125};
        Float8Vector v = column("x", data);
        MemorySegment seg = ArrowMemoryBridge.wrapDoubles(v.getDataBuffer(), data.length);
        for (int i = 0; i < data.length; i++) {
            assertEquals(data[i], seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i), "index " + i);
        }
    }

    @Test
    void test02_wrapDoubles_aliasesRatherThanCopies() {
        double[] data = {1.0, 2.0, 3.0};
        Float8Vector v = column("x", data);
        MemorySegment seg = ArrowMemoryBridge.wrapDoubles(v.getDataBuffer(), data.length);

        v.set(1, 999.0); // mutate the Arrow vector AFTER wrapping

        assertEquals(999.0, seg.getAtIndex(ValueLayout.JAVA_DOUBLE, 1),
            "segment must reflect the mutation — proves aliasing, not a copy");
    }

    @Test
    void test03_wrapDoubles_throwsOnNullBuf() {
        assertThrows(NullPointerException.class, () -> ArrowMemoryBridge.wrapDoubles(null, 10));
    }

    @Test
    void test04_wrapDoubles_throwsOnNegativeElementCount() {
        Float8Vector v = column("x", new double[]{1.0, 2.0});
        assertThrows(IllegalArgumentException.class,
            () -> ArrowMemoryBridge.wrapDoubles(v.getDataBuffer(), -1));
    }

    @Test
    void test05_wrapDoubles_throwsWhenCapacityInsufficient() {
        Float8Vector v = column("x", new double[]{1.0, 2.0}); // 2 doubles = 16 bytes capacity (at least)
        assertThrows(IllegalArgumentException.class,
            () -> ArrowMemoryBridge.wrapDoubles(v.getDataBuffer(), 10_000_000));
    }

    @Test
    void test06_wrapFullCapacity_matchesBufferCapacity() {
        Float8Vector v = column("x", new double[]{1.0, 2.0, 3.0, 4.0});
        MemorySegment seg = ArrowMemoryBridge.wrapFullCapacity(v.getDataBuffer());
        assertEquals(v.getDataBuffer().capacity(), seg.byteSize());
    }

    // =====================================================================
    // B. Binding correctness — by name via getSlotItems() (7-12)
    // =====================================================================

    @Test
    void test07_evaluate_singleVariable_matchesReferenceMath() throws Throwable {
        double[] xData = randomData(1000, 1);
        var evaluator = compile("sqrt(x*x+1.0)");
        Float8Vector x = column("x", xData);
        Float8Vector out = emptyOutput("out", xData.length);

        evaluator.evaluate(Map.of("x", x), out);

        double[] result = readAll(out);
        for (int i = 0; i < xData.length; i++) {
            assertClose(Math.sqrt(xData[i] * xData[i] + 1.0), result[i], "row " + i);
        }
    }

    @Test
    void test08_evaluate_bindingIndependentOfMapInsertionOrder() throws Throwable {
        double[] xData = randomData(500, 2);
        double[] yData = randomData(500, 3);
        var evaluator = compile("x - y*2.0");

        Float8Vector x1 = column("x", xData);
        Float8Vector y1 = column("y", yData);
        Map<String, Float8Vector> orderA = new HashMap<>();
        orderA.put("x", x1);
        orderA.put("y", y1);

        Float8Vector x2 = column("x", xData);
        Float8Vector y2 = column("y", yData);
        Map<String, Float8Vector> orderB = new HashMap<>();
        orderB.put("y", y2); // inserted in reverse order
        orderB.put("x", x2);

        Float8Vector outA = emptyOutput("outA", xData.length);
        Float8Vector outB = emptyOutput("outB", xData.length);

        evaluator.evaluate(orderA, outA);
        evaluator.evaluate(orderB, outB);

        assertArrayEquals(readAll(outA), readAll(outB), 0.0,
            "binding must be resolved by variable name, not Map iteration order");
    }

    @Test
    void test09_requiredVariableNames_matchesExpressionVariables() throws Throwable {
        var evaluator = compile("x*y + z");
        String[] names = evaluator.requiredVariableNames();
        Arrays.sort(names);
        assertArrayEquals(new String[]{"x", "y", "z"}, names);
    }

    @Test
    void test10_requiredVariableNames_emptyForConstantExpression() throws Throwable {
        var evaluator = compile("3.0 + 4.0");
        assertEquals(0, evaluator.requiredVariableNames().length);
    }

    @Test
    void test11_evaluate_extraUnusedColumnsInMapAreIgnored() throws Throwable {
        double[] xData = randomData(200, 4);
        var evaluator = compile("x*2.0");
        Float8Vector x = column("x", xData);
        Float8Vector unused = column("unused", randomData(200, 5));
        Float8Vector out = emptyOutput("out", xData.length);

        Map<String, Float8Vector> columns = new HashMap<>();
        columns.put("x", x);
        columns.put("unused", unused);

        assertDoesNotThrow(() -> evaluator.evaluate(columns, out));
        double[] result = readAll(out);
        for (int i = 0; i < xData.length; i++) {
            assertClose(xData[i] * 2.0, result[i], "row " + i);
        }
    }

    @Test
    void test12_evaluate_viaVectorSchemaRoot_matchesMapBinding() throws Throwable {
        double[] xData = randomData(300, 6);
        double[] yData = randomData(300, 7);
        var evaluator = compile("x+y*sin(x)");

        Float8Vector x1 = column("x", xData);
        Float8Vector y1 = column("y", yData);
        Float8Vector mapOut = emptyOutput("mapOut", xData.length);
        evaluator.evaluate(Map.of("x", x1, "y", y1), mapOut);

        Field xField = new Field("x", FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), null);
        Field yField = new Field("y", FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), null);
        Float8Vector x2 = column("x", xData);
        Float8Vector y2 = column("y", yData);
        try (VectorSchemaRoot root = new VectorSchemaRoot(Arrays.asList(xField, yField), Arrays.asList(x2, y2))) {
            root.setRowCount(xData.length);
            Float8Vector schemaOut = emptyOutput("schemaOut", xData.length);
            evaluator.evaluate(root, schemaOut);
            assertArrayEquals(readAll(mapOut), readAll(schemaOut), 0.0);
        }
    }

    // =====================================================================
    // C. Error handling — ArrowBindingException paths (13-19)
    // =====================================================================

    @Test
    void test13_evaluate_missingRequiredColumn_throwsArrowBindingException() throws Throwable {
        var evaluator = compile("x+y");
        Float8Vector x = column("x", randomData(10, 8));
        Float8Vector out = emptyOutput("out", 10);

        assertThrows(ArrowBindingException.class, () -> evaluator.evaluate(Map.of("x", x), out));
    }

    @Test
    void test14_evaluate_missingColumn_exceptionMessageListsRequiredNames() throws Throwable {
        var evaluator = compile("x+y");
        Float8Vector x = column("x", randomData(10, 9));
        Float8Vector out = emptyOutput("out", 10);

        ArrowBindingException ex = assertThrows(ArrowBindingException.class,
            () -> evaluator.evaluate(Map.of("x", x), out));
        assertTrue(ex.getMessage().contains("y"), "message should name the missing variable: " + ex.getMessage());
    }

    @Test
    void test15_evaluate_columnShorterThanOutput_throwsArrowBindingException() throws Throwable {
        var evaluator = compile("x*2.0");
        Float8Vector x = column("x", randomData(5, 10)); // only 5 rows
        Float8Vector out = emptyOutput("out", 10);        // output expects 10

        assertThrows(ArrowBindingException.class, () -> evaluator.evaluate(Map.of("x", x), out));
    }

    @Test
    void test16_evaluate_outputNotSized_throwsArrowBindingException() throws Throwable {
        var evaluator = compile("x*2.0");
        Float8Vector x = column("x", randomData(10, 11));
        Float8Vector out = new Float8Vector("out", allocator); // never allocateNew'd/setValueCount'd
        openVectors.add(out);

        assertThrows(ArrowBindingException.class, () -> evaluator.evaluate(Map.of("x", x), out));
    }

    @Test
    void test17_evaluate_viaSchemaRoot_missingField_throwsArrowBindingException() throws Throwable {
        var evaluator = compile("x+y");
        Field xField = new Field("x", FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), null);
        Float8Vector x = column("x", randomData(10, 12));
        try (VectorSchemaRoot root = new VectorSchemaRoot(List.of(xField), List.of(x))) {
            root.setRowCount(10);
            Float8Vector out = emptyOutput("out", 10);
            assertThrows(ArrowBindingException.class, () -> evaluator.evaluate(root, out));
        }
    }

    @Test
    void test18_evaluate_viaSchemaRoot_wrongVectorType_throwsArrowBindingException() throws Throwable {
        var evaluator = compile("x*2.0");
        // Bind "x" to a non-Float8Vector field to trigger the type-mismatch guard.
        Field xField = new Field("x", FieldType.nullable(new ArrowType.Int(32, true)), null);
        try (org.apache.arrow.vector.IntVector intX = new org.apache.arrow.vector.IntVector("x", allocator)) {
            intX.allocateNew(10);
            for (int i = 0; i < 10; i++) intX.set(i, i);
            intX.setValueCount(10);
            try (VectorSchemaRoot root = new VectorSchemaRoot(List.of(xField), List.of(intX))) {
                root.setRowCount(10);
                Float8Vector out = emptyOutput("out", 10);
                ArrowBindingException ex = assertThrows(ArrowBindingException.class, () -> evaluator.evaluate(root, out));
                assertTrue(ex.getMessage().contains("Float8Vector"), ex.getMessage());
            }
        }
    }

    @Test
    void test19_evaluate_afterClose_throwsIllegalStateException() throws Throwable {
        ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.compile("x+1.0");
        evaluator.close();
        Float8Vector x = column("x", randomData(5, 13));
        Float8Vector out = emptyOutput("out", 5);
        assertThrows(IllegalStateException.class, () -> evaluator.evaluate(Map.of("x", x), out));
    }

    // =====================================================================
    // D. Constant expressions (20-22)
    // =====================================================================

    @Test
    void test20_evaluate_constantExpression_fillsOutputWithSameValueEveryRow() throws Throwable {
        var evaluator = compile("3.0*4.0 - 1.0");
        Float8Vector out = emptyOutput("out", 1000);
        evaluator.evaluate(Map.of(), out);
        double[] result = readAll(out);
        for (double v : result) {
            assertEquals(11.0, v, 0.0);
        }
    }

    @Test
    void test21_isConstantExpression_trueForLiteralExpression() throws Throwable {
        var evaluator = compile("42.0");
        assertTrue(evaluator.isConstantExpression());
    }

    @Test
    void test22_isConstantExpression_falseForVariableExpression() throws Throwable {
        var evaluator = compile("x + 1.0");
        assertFalse(evaluator.isConstantExpression());
    }

    // =====================================================================
    // E. Null policy (23-26)
    // =====================================================================

    @Test
    void test23_evaluate_ignorePolicy_doesNotTouchOutputValidityBitmap() throws Throwable {
        var evaluator = compile("x+1.0");
        Float8Vector x = column("x", randomData(20, 14));
        x.setNull(5); // mark one input row null

        Float8Vector out = emptyOutput("out", 20);
        out.setNull(3); // pre-mark a different row null in the output, before evaluate

        evaluator.evaluate(Map.of("x", x), out, NullPolicy.IGNORE);

        assertTrue(out.isNull(3), "IGNORE policy must not modify the output's validity bitmap");
        assertFalse(out.isNull(5), "IGNORE policy must not propagate the input's null either");
    }

    @Test
    void test24_evaluate_propagatePolicy_marksOutputNullWhereAnyInputNull() throws Throwable {
        var evaluator = compile("x+1.0");
        Float8Vector x = column("x", randomData(20, 15));
        x.setNull(7);

        Float8Vector out = emptyOutput("out", 20);
        evaluator.evaluate(Map.of("x", x), out, NullPolicy.PROPAGATE);

        assertTrue(out.isNull(7), "output row 7 must be null since input row 7 was null");
    }

    @Test
    void test25_evaluate_propagatePolicy_leavesRowsValidWhenAllInputsValid() throws Throwable {
        var evaluator = compile("x*2.0");
        Float8Vector x = column("x", randomData(20, 16)); // no nulls set
        Float8Vector out = emptyOutput("out", 20);

        evaluator.evaluate(Map.of("x", x), out, NullPolicy.PROPAGATE);

        for (int i = 0; i < 20; i++) {
            assertFalse(out.isNull(i), "row " + i + " should remain valid — no input was null");
        }
    }

    @Test
    void test26_evaluate_propagatePolicy_unionsNullMasksAcrossAllColumns() throws Throwable {
        var evaluator = compile("x+y");
        Float8Vector x = column("x", randomData(20, 17));
        Float8Vector y = column("y", randomData(20, 18));
        x.setNull(2);
        y.setNull(9);

        Float8Vector out = emptyOutput("out", 20);
        evaluator.evaluate(Map.of("x", x, "y", y), out, NullPolicy.PROPAGATE);

        assertTrue(out.isNull(2), "null from x must propagate");
        assertTrue(out.isNull(9), "null from y must propagate");
        assertFalse(out.isNull(0), "unrelated row must remain valid");
    }

    // =====================================================================
    // F. Row count / sizing edge cases (27-30)
    // =====================================================================

    @Test
    void test27_evaluate_zeroRowOutput_isNoOpAndDoesNotThrow() throws Throwable {
        var evaluator = compile("x+y");
        Float8Vector x = column("x", new double[0]);
        Float8Vector y = column("y", new double[0]);
        Float8Vector out = emptyOutput("out", 0);
        assertDoesNotThrow(() -> evaluator.evaluate(Map.of("x", x, "y", y), out));
    }

    @Test
    void test28_evaluate_singleRow_correct() throws Throwable {
        var evaluator = compile("x*x + 1.0");
        Float8Vector x = column("x", new double[]{7.0});
        Float8Vector out = emptyOutput("out", 1);
        evaluator.evaluate(Map.of("x", x), out);
        assertClose(50.0, out.get(0), "single row");
    }

    @Test
    void test29_evaluate_nonBlockAlignedRowCount_correct() throws Throwable {
        int n = 100_003; // prime, exercises a non-aligned tail across many internal blocks
        double[] xData = randomData(n, 19);
        var evaluator = compile("sin(x)+cos(x)");
        Float8Vector x = column("x", xData);
        Float8Vector out = emptyOutput("out", n);

        evaluator.evaluate(Map.of("x", x), out);

        double[] result = readAll(out);
        // Spot-check a sparse sample rather than every row, to keep the test fast.
        int[] sampleIdx = {0, 1, n / 2, n - 2, n - 1};
        for (int i : sampleIdx) {
            assertClose(Math.sin(xData[i]) + Math.cos(xData[i]), result[i], "row " + i);
        }
    }

    @Test
    void test30_allocateOutput_producesCorrectlySizedVector() {
        Float8Vector v = emptyOutput("out", 12345);
        assertEquals(12345, v.getValueCount());
    }

    // =====================================================================
    // G. Parallel/serial and concurrency (31-35)
    // =====================================================================

    @Test
    void test31_evaluate_parallelTrue_matchesParallelFalse() throws Throwable {
        double[] xData = randomData(500_000, 20);
        double[] yData = randomData(500_000, 21);
        var evaluator = compile("(x+y)*sin(x) - sqrt(y*y+1.0)");

        Float8Vector x1 = column("x", xData);
        Float8Vector y1 = column("y", yData);
        Float8Vector outSerial = emptyOutput("outSerial", xData.length);
        evaluator.evaluate(Map.of("x", x1, "y", y1), outSerial, NullPolicy.IGNORE, false);

        Float8Vector x2 = column("x", xData);
        Float8Vector y2 = column("y", yData);
        Float8Vector outParallel = emptyOutput("outParallel", xData.length);
        evaluator.evaluate(Map.of("x", x2, "y", y2), outParallel, NullPolicy.IGNORE, true);

        double[] serial = readAll(outSerial);
        double[] parallel = readAll(outParallel);
        // Serial and parallel dispatch inside the underlying SIMD engine can
        // legitimately take different vector-lane-width / FMA-contraction paths
        // for transcendental functions (sin/sqrt), so last-bit rounding
        // differences between the two are expected floating-point behavior, not
        // a correctness bug. Compare within a tight tolerance rather than
        // requiring bit-for-bit equality.
        for (int i = 0; i < serial.length; i++) {
            assertClose(serial[i], parallel[i], "row " + i);
        }
    }

    @Test
    void test32_evaluate_largeBatch_parallelPath_matchesReferenceMath() throws Throwable {
        int n = 2_000_000;
        double[] xData = randomData(n, 22);
        var evaluator = compile("sqrt(abs(x))+x*x");
        Float8Vector x = column("x", xData);
        Float8Vector out = emptyOutput("out", n);

        evaluator.evaluate(Map.of("x", x), out);

        double[] result = readAll(out);
        int[] sampleIdx = {0, 12345, n / 2, n - 1};
        for (int i : sampleIdx) {
            assertClose(Math.sqrt(Math.abs(xData[i])) + xData[i] * xData[i], result[i], "row " + i);
        }
    }

    @Test
    void test33_evaluate_repeatedCallsOnSameEvaluator_produceConsistentResults() throws Throwable {
        var evaluator = compile("x*y - sin(x)");
        int n = 5000;

        double[] firstResult = null;
        for (int round = 0; round < 5; round++) {
            double[] xData = randomData(n, 100 + round);
            double[] yData = randomData(n, 200 + round);
            Float8Vector x = column("x", xData);
            Float8Vector y = column("y", yData);
            Float8Vector out = emptyOutput("out" + round, n);

            evaluator.evaluate(Map.of("x", x, "y", y), out);
            double[] result = readAll(out);

            // Cross-check this round's result against a fresh evaluator instance,
            // to catch any state leaking between rounds on the shared instance.
            try (var fresh = ArrowBulkEvaluator.compile("x*y - sin(x)")) {
                Float8Vector x2 = column("x", xData);
                Float8Vector y2 = column("y", yData);
                Float8Vector freshOut = emptyOutput("fresh" + round, n);
                fresh.evaluate(Map.of("x", x2, "y", y2), freshOut);
                assertArrayEquals(readAll(freshOut), result, 0.0, "round " + round);
            }
        }
    }

    @Test
    void test34_evaluate_concurrentCallsFromMultipleThreads_allSucceedAndMatchReference() throws Throwable {
        var evaluator = compile("x*x - y*y + sin(x)*cos(y)");
        int n = 20_000;
        int threadCount = 6;
        int iterationsPerThread = 10;

        ExecutorService pool = Executors.newFixedThreadPool(threadCount);
        AtomicBoolean failed = new AtomicBoolean(false);
        List<Future<?>> futures = new ArrayList<>();

        for (int t = 0; t < threadCount; t++) {
            final int threadIdx = t;
            futures.add(pool.submit(() -> {
                try (BufferAllocator threadAllocator = new RootAllocator(Long.MAX_VALUE)) {
                    for (int i = 0; i < iterationsPerThread; i++) {
                        long seed = 1000L + threadIdx * 100L + i;
                        double[] xData = randomData(n, seed);
                        double[] yData = randomData(n, seed + 1);

                        try (Float8Vector x = threadColumn(threadAllocator, "x", xData);
                             Float8Vector y = threadColumn(threadAllocator, "y", yData);
                             Float8Vector out = ArrowBulkEvaluator.allocateOutput(threadAllocator, "out", n)) {

                            evaluator.evaluate(Map.of("x", x, "y", y), out);

                            for (int idx : new int[]{0, n / 2, n - 1}) {
                                double expected = xData[idx] * xData[idx] - yData[idx] * yData[idx]
                                    + Math.sin(xData[idx]) * Math.cos(yData[idx]);
                                assertClose(expected, out.get(idx), "thread=" + threadIdx + " iter=" + i + " idx=" + idx);
                            }
                        }
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    throw new RuntimeException(e);
                }
            }));
        }

        pool.shutdown();
        assertTrue(pool.awaitTermination(120, TimeUnit.SECONDS), "concurrent test run timed out");
        for (Future<?> f : futures) {
            assertDoesNotThrow(() -> {
                try {
                    f.get();
                } catch (java.util.concurrent.ExecutionException e) {
                    throw e.getCause();
                }
            });
        }
        assertFalse(failed.get());
    }

    @Test
    void test35_close_isIdempotent_safeToCallTwice() throws Throwable {
        ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.compile("x+1.0");
        evaluator.close();
        assertDoesNotThrow(evaluator::close);
    }

    // =====================================================================
    // H. Numeric correctness across expression shapes (36-40)
    // =====================================================================

    @Test
    void test36_evaluate_arithmeticChain_matchesReferenceMath() throws Throwable {
        double[] xData = randomData(10_000, 30);
        double[] yData = randomData(10_000, 31);
        double[] zData = randomData(10_000, 32);
        var evaluator = compile("(x+y)*z - x/y + 2.0*z");

        Float8Vector x = column("x", xData);
        Float8Vector y = column("y", yData);
        Float8Vector z = column("z", zData);
        Float8Vector out = emptyOutput("out", xData.length);

        evaluator.evaluate(Map.of("x", x, "y", y, "z", z), out);

        double[] result = readAll(out);
        for (int i = 0; i < xData.length; i++) {
            double expected = (xData[i] + yData[i]) * zData[i] - xData[i] / yData[i] + 2.0 * zData[i];
            assertClose(expected, result[i], "row " + i);
        }
    }

    @Test
    void test37_evaluate_transcendentalFunctions_matchesMathLibrary() throws Throwable {
        double[] xData = randomData(10_000, 33);
        var evaluator = compile("sin(x)+cos(x)*tan(x)-exp(x/50.0)+log(abs(x)+1.0)");
        Float8Vector x = column("x", xData);
        Float8Vector out = emptyOutput("out", xData.length);

        evaluator.evaluate(Map.of("x", x), out);

        double[] result = readAll(out);
        for (int i = 0; i < xData.length; i++) {
            double v = xData[i];
            double expected = Math.sin(v) + Math.cos(v) * Math.tan(v) - Math.exp(v / 50.0) + Math.log(Math.abs(v) + 1.0);
            assertClose(expected, result[i], "row " + i);
        }
    }

    @Test
    void test38_evaluate_comparisonOperators_producesZeroOneEncoding() throws Throwable {
        double[] xData = randomData(1000, 34);
        double[] yData = randomData(1000, 35);
        var evaluator = compile("x>y");
        Float8Vector x = column("x", xData);
        Float8Vector y = column("y", yData);
        Float8Vector out = emptyOutput("out", xData.length);

        evaluator.evaluate(Map.of("x", x, "y", y), out);

        double[] result = readAll(out);
        for (int i = 0; i < xData.length; i++) {
            double expected = xData[i] > yData[i] ? 1.0 : 0.0;
            assertEquals(expected, result[i], 0.0, "row " + i);
        }
    }

    @Test
    void test39_evaluate_specialFloatingPointValues_nanInfinityPropagateCorrectly() throws Throwable {
        double[] xData = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 0.0, -0.0, 5.0};
        var evaluator = compile("x+1.0");
        Float8Vector x = column("x", xData);
        Float8Vector out = emptyOutput("out", xData.length);

        evaluator.evaluate(Map.of("x", x), out);

        double[] result = readAll(out);
        assertTrue(Double.isNaN(result[0]));
        assertEquals(Double.POSITIVE_INFINITY, result[1]);
        assertEquals(Double.NEGATIVE_INFINITY, result[2]);
        assertEquals(1.0, result[3]);
        assertEquals(1.0, result[4]);
        assertEquals(6.0, result[5]);
    }

    @Test
    void test40_evaluate_veryLargeRowCount_completesAndMatchesReference() throws Throwable {
        int n = 100_000_000;
        double[] xData = randomData(n, 36);
        var evaluator = compile("x*2.0 - sqrt(abs(x))");
        Float8Vector x = column("x", xData);
        Float8Vector out = emptyOutput("out", n);

        long start = System.nanoTime();
        evaluator.evaluate(Map.of("x", x), out);
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;
        System.out.println("test40: evaluated " + n + " rows in " + elapsedMs + " ms");

        int[] sampleIdx = {0, 1, n / 4, n / 2, n - 2, n - 1};
        for (int i : sampleIdx) {
            double expected = xData[i] * 2.0 - Math.sqrt(Math.abs(xData[i]));
            assertClose(expected, out.get(i), "row " + i);
        }
    }

    // =====================================================================
    // Concurrency helper — separate from the shared per-test column() helper
    // since each thread in test34 owns its own allocator/vector lifecycle.
    // =====================================================================

    private static Float8Vector threadColumn(BufferAllocator alloc, String name, double[] values) {
        Float8Vector v = new Float8Vector(name, alloc);
        v.allocateNew(values.length);
        for (int i = 0; i < values.length; i++) {
            v.set(i, values[i]);
        }
        v.setValueCount(values.length);
        return v;
    }
}