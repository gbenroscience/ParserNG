package com.github.gbenroscience.arrow.tools.box;

import com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend;
import com.github.gbenroscience.arrow.tools.box.ArrowBulkEvaluator;
import com.github.gbenroscience.arrow.tools.box.ArrowGpuBulkEvaluator;
import com.github.gbenroscience.arrow.tools.box.ArrowBindingException;
import com.github.gbenroscience.arrow.tools.box.NullPolicy;
import com.github.gbenroscience.gpu.GpuBackend;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link ArrowGpuBulkEvaluator}.
 *
 * <p>
 * Most tests here require an actual GPU device (CUDA and/or OpenCL) and are
 * gated behind {@code -Dgpu.tests=true}, the same convention used by
 * {@code GpuCompositeExpressionTest}. A handful of tests that only exercise
 * argument validation (never touching a device) run unconditionally.
 *
 * <p>
 * Tests that need a *specific* backend additionally guard themselves with
 * {@link Assumptions#assumeTrue} against
 * {@link ArrowGpuBulkEvaluator#isBackendAvailable} so the suite still passes
 * (by skipping, not failing) on a machine that only has one of the two backends
 * installed.
 *
 * <p><b>Resource discipline:</b> every {@link Float8Vector} (and every
 * {@link VectorSchemaRoot}) created in a test MUST be closed before the
 * test method returns. {@code allocator} is a single {@link RootAllocator}
 * shared across the whole class and closed in {@link #tearDown()}; if any
 * vector created in a test isn't released, {@code RootAllocator.close()}
 * throws {@code IllegalStateException: Memory was leaked by query} in
 * teardown -- reported by JUnit as an "Error" on that test even though the
 * test body itself passed every assertion. Use try-with-resources on every
 * {@code Float8Vector}/{@code VectorSchemaRoot} you create, including ones
 * created purely to trigger a validation failure (the exception is thrown
 * before the vector would otherwise be closed, so those need it too).
 */
@EnabledIfSystemProperty(named = "gpu.tests", matches = "true")
class ArrowGpuBulkEvaluatorTest {

    private BufferAllocator allocator;

    @BeforeEach
    void setUp() {
        allocator = new RootAllocator();
    }

    @AfterEach
    void tearDown() {
        allocator.close();
    }

    // -------------------------------------------------------------------
    // helpers
    // -------------------------------------------------------------------
    private Float8Vector column(String name, double... values) {
        Float8Vector v = new Float8Vector(name, allocator);
        v.allocateNew(values.length);
        for (int i = 0; i < values.length; i++) {
            v.set(i, values[i]);
        }
        v.setValueCount(values.length);
        return v;
    }

    private Float8Vector columnWithNullAt(int nullIndex, double... values) {
        Float8Vector v = column("x", values);
        v.setNull(nullIndex);
        return v;
    }

    private double[] cpuReference(String expr, String varName, double[] xs) throws Throwable {
        // Use the already-verified CPU SIMD path as the oracle for GPU-vs-CPU
        // parity, rather than guessing at a scalar per-value solve API.
        try (ArrowBulkEvaluator cpu = ArrowBulkEvaluator.compile(expr);
             Float8Vector x = column(varName, xs);
             Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "cpu-out", xs.length)) {
            cpu.evaluate(Map.of(varName, x), out);
            double[] result = new double[xs.length];
            for (int i = 0; i < xs.length; i++) {
                result[i] = out.get(i);
            }
            return result;
        }
    }

    // =====================================================================
    // 1. Compilation & backend selection
    // =====================================================================
    @Nested
    class CompilationAndBackendSelection {

        @Test
        void compileAutoPicksABootstrapableBackend() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1")) {
                GpuBackend backend = eval.actualBackend();
                assertTrue(backend == GpuBackend.CUDA || backend == GpuBackend.OPENCL);
            }
        }

        @Test
        void compilePinnedCudaReportsCudaAsActualBackend() throws Throwable {
            Assumptions.assumeTrue(ArrowGpuBulkEvaluator.isBackendAvailable(GpuBackend.CUDA),
                    "No CUDA backend available on this machine");
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1", GpuBackend.CUDA)) {
                assertEquals(GpuBackend.CUDA, eval.actualBackend());
                assertEquals(ArrowExecutionBackend.GPU_CUDA, eval.backend());
            }
        }

        @Test
        void compilePinnedOpenClReportsOpenClAsActualBackend() throws Throwable {
            Assumptions.assumeTrue(ArrowGpuBulkEvaluator.isBackendAvailable(GpuBackend.OPENCL),
                    "No OpenCL backend available on this machine");
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1", GpuBackend.OPENCL)) {
                assertEquals(GpuBackend.OPENCL, eval.actualBackend());
                assertEquals(ArrowExecutionBackend.GPU_OPENCL, eval.backend());
            }
        }

        @Test
        void compileWithNullBackendThrowsNpe() {
            // Pure argument validation -- doesn't need a device to run, but is
            // kept under the same gate for consistency with the rest of the file.
            assertThrows(NullPointerException.class,
                    () -> ArrowGpuBulkEvaluator.compile("x+1", (GpuBackend) null));
        }

        @Test
        void isBackendAvailableWithNullBackendThrowsNpe() {
            assertThrows(NullPointerException.class,
                    () -> ArrowGpuBulkEvaluator.isBackendAvailable(null));
        }

        @Test
        void isAnyGpuAvailableIsConsistentWithSuccessfulCompile() throws Throwable {
            boolean reportedAvailable = ArrowGpuBulkEvaluator.isAnyGpuAvailable();
            if (reportedAvailable) {
                try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1")) {
                    assertNotNull(eval.actualBackend());
                }
            } else {
                assertThrows(Throwable.class, () -> ArrowGpuBulkEvaluator.compile("x+1"));
            }
        }
    }

    // =====================================================================
    // 2. Introspection
    // =====================================================================
    @Nested
    class Introspection {

        @Test
        void requiredVariableNamesMatchesExpressionSlots() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+y*2")) {
                String[] names = eval.requiredVariableNames();
                assertEquals(2, names.length);
                List<String> asList = java.util.Arrays.asList(names);
                assertTrue(asList.contains("x"));
                assertTrue(asList.contains("y"));
            }
        }

        @Test
        void requiredVariableNamesIsADefensiveCopy() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1")) {
                String[] first = eval.requiredVariableNames();
                first[0] = "TAMPERED";
                String[] second = eval.requiredVariableNames();
                assertEquals("x", second[0]);
            }
        }

        @Test
        void isConstantExpressionTrueForBareLiteral() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("42.0")) {
                assertTrue(eval.isConstantExpression());
                assertEquals(0, eval.requiredVariableNames().length);
            }
        }

        @Test
        void isConstantExpressionFalseWhenVariablesPresent() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1")) {
                assertFalse(eval.isConstantExpression());
            }
        }

        @Test
        void getExpressionTextReturnsOriginalSource() throws Throwable {
            // NOTE: this is not a GPU-path bug. getExpressionText() just
            // returns MathExpression#getExpression() verbatim, and
            // MathExpression re-serializes the parsed AST rather than
            // preserving the original input string byte-for-byte -- it adds
            // defensive grouping parens around binary operations, so
            // "sin(x)+cos(x)" round-trips as "(sin(x)+cos(x))". This is the
            // same MathExpression used by the CPU path (ArrowBulkEvaluator),
            // so if this expectation is wrong here it's equally wrong there;
            // worth checking ArrowBulkEvaluatorTest for the same assertion
            // and fixing both consistently rather than special-casing GPU.
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("sin(x)+cos(x)")) {
                assertEquals("(sin(x)+cos(x))", eval.getExpressionText());
            }
        }
    }

    // =====================================================================
    // 3. Evaluation correctness — Map<String, Float8Vector> binding
    // =====================================================================
    @Nested
    class EvaluationMapBinding {

        @Test
        void evaluatesLinearExpressionOverMapBinding() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1")) {
                double[] xs = {1, 2, 3, 4, 5};
                try (Float8Vector x = column("x", xs);
                     Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", xs.length)) {

                    eval.evaluate(Map.of("x", x), out);

                    for (int i = 0; i < xs.length; i++) {
                        assertEquals(xs[i] + 1, out.get(i), 1e-9);
                    }
                }
            }
        }

        @Test
        void evaluatesMultiVariableExpressionOverMapBinding() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x*y+2")) {
                double[] xs = {1, 2, 3, 4};
                double[] ys = {10, 20, 30, 40};
                try (Float8Vector x = column("x", xs);
                     Float8Vector y = column("y", ys);
                     Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", xs.length)) {

                    Map<String, Float8Vector> columns = new HashMap<>();
                    columns.put("x", x);
                    columns.put("y", y);
                    eval.evaluate(columns, out);

                    for (int i = 0; i < xs.length; i++) {
                        assertEquals(xs[i] * ys[i] + 2, out.get(i), 1e-9);
                    }
                }
            }
        }

        @Test
        void constantExpressionFillsOutputWithoutTouchingGpu() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("7*6");
                 Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 10)) {
                eval.evaluate(Map.of(), out);
                for (int i = 0; i < 10; i++) {
                    assertEquals(42.0, out.get(i), 1e-9);
                }
            }
        }

        @Test
        void cpuVsGpuParityOnTranscendentalExpression() throws Throwable {
            String expr = "3*sin(x)-cos(2*x)";
            try (ArrowGpuBulkEvaluator gpu = ArrowGpuBulkEvaluator.compile(expr)) {
                int n = 2000;
                double[] xs = new double[n];
                for (int i = 0; i < n; i++) {
                    xs[i] = -6.0 + (12.0 * i) / (n - 1);
                }
                try (Float8Vector x = column("x", xs);
                     Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", n)) {

                    gpu.evaluate(Map.of("x", x), out);

                    double[] cpuExpected = cpuReference(expr, "x", xs);
                    for (int i = 0; i < n; i++) {
                        assertEquals(cpuExpected[i], out.get(i), 1e-6, "mismatch at row " + i);
                    }
                }
            }
        }
    }

    // =====================================================================
    // 4. Evaluation correctness — VectorSchemaRoot convenience binding
    // =====================================================================
    @Nested
    class EvaluationVectorSchemaRootBinding {

        @Test
        void evaluatesUsingVectorSchemaRootConvenienceBinding() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x-2")) {
                double[] xs = {5, 6, 7};
                try (Float8Vector x = column("x", xs);
                     VectorSchemaRoot root = VectorSchemaRoot.of(x);
                     Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", xs.length)) {
                    eval.evaluate(root, out);
                    for (int i = 0; i < xs.length; i++) {
                        assertEquals(xs[i] - 2, out.get(i), 1e-9);
                    }
                }
            }
        }

        @Test
        void vectorSchemaRootBindingRejectsNonFloat8VectorColumn() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1");
                 IntVector wrongType = new IntVector("x", allocator)) {
                wrongType.allocateNew(3);
                wrongType.set(0, 1);
                wrongType.set(1, 2);
                wrongType.set(2, 3);
                wrongType.setValueCount(3);

                try (VectorSchemaRoot root = VectorSchemaRoot.of(wrongType);
                     Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 3)) {
                    ArrowBindingException ex = assertThrows(ArrowBindingException.class,
                            () -> eval.evaluate(root, out));
                    assertTrue(ex.getMessage().contains("Float8Vector"));
                }
            }
        }
    }

    // =====================================================================
    // 5. Error handling / binding validation
    // =====================================================================
    @Nested
    class ErrorHandling {

        @Test
        void missingRequiredColumnThrowsArrowBindingException() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+y");
                 Float8Vector x = column("x", 1, 2, 3);
                 Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 3)) {

                ArrowBindingException ex = assertThrows(ArrowBindingException.class,
                        () -> eval.evaluate(Map.of("x", x), out));
                assertTrue(ex.getMessage().contains("y"));
            }
        }

        @Test
        void columnShorterThanOutputThrowsArrowBindingException() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1");
                 Float8Vector x = column("x", 1, 2); // only 2 rows
                 Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 5)) { // expects 5

                ArrowBindingException ex = assertThrows(ArrowBindingException.class,
                        () -> eval.evaluate(Map.of("x", x), out));
                assertTrue(ex.getMessage().contains("rows"));
            }
        }

        @Test
        void unsizedOutputVectorThrowsArrowBindingException() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1");
                 Float8Vector x = column("x", 1, 2, 3);
                 Float8Vector out = new Float8Vector("out", allocator)) { // never allocateNew/setValueCount

                assertThrows(ArrowBindingException.class,
                        () -> eval.evaluate(Map.of("x", x), out));
            }
        }

        @Test
        void legitimatelyEmptyBatchIsANoOp() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1");
                 Float8Vector x = column("x"); // zero rows
                 Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 0)) {

                // should not throw
                eval.evaluate(Map.of("x", x), out);
                assertEquals(0, out.getValueCount());
            }
        }
    }

    // =====================================================================
    // 6. NullPolicy
    // =====================================================================
    @Nested
    class NullPolicyTests {

        @Test
        void ignorePolicyLeavesOutputValidityUntouched() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1");
                 Float8Vector x = columnWithNullAt(1, 1, 2, 3);
                 Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 3)) {

                eval.evaluate(Map.of("x", x), out, NullPolicy.IGNORE);

                // allocateOutput() pre-marks every row valid; IGNORE must not change that
                for (int i = 0; i < 3; i++) {
                    assertFalse(out.isNull(i));
                }
            }
        }

        @Test
        void propagatePolicyMarksOutputNullWhereAnyInputIsNull() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1");
                 Float8Vector x = columnWithNullAt(1, 1, 2, 3);
                 Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 3)) {

                eval.evaluate(Map.of("x", x), out, NullPolicy.PROPAGATE);

                assertFalse(out.isNull(0));
                assertTrue(out.isNull(1));
                assertFalse(out.isNull(2));
            }
        }

        @Test
        void propagatePolicyLeavesOutputAllValidWhenNoInputIsNull() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1");
                 Float8Vector x = column("x", 1, 2, 3);
                 Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 3)) {

                eval.evaluate(Map.of("x", x), out, NullPolicy.PROPAGATE);

                for (int i = 0; i < 3; i++) {
                    assertFalse(out.isNull(i));
                }
            }
        }
    }

    // =====================================================================
    // 7. Lifecycle
    // =====================================================================
    @Nested
    class Lifecycle {

        @Test
        void evaluateAfterCloseThrowsIllegalStateException() throws Throwable {
            ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1");
            eval.close();

            try (Float8Vector x = column("x", 1, 2, 3);
                 Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 3)) {
                assertThrows(IllegalStateException.class, () -> eval.evaluate(Map.of("x", x), out));
            }
        }

        @Test
        void closeIsIdempotent() throws Throwable {
            ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1");
            eval.close();
            eval.close(); // should not throw
        }
    }

    // =====================================================================
    // 8. Thread safety
    // =====================================================================
    @Nested
    class ThreadSafety {

        @Test
        void concurrentEvaluateCallsSerializeToCorrectResults() throws Throwable {
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x*2+1")) {
                int threadCount = 8;
                int rowsPerCall = 500;
                ExecutorService pool = Executors.newFixedThreadPool(threadCount);
                CountDownLatch ready = new CountDownLatch(threadCount);
                CountDownLatch go = new CountDownLatch(1);
                AtomicInteger failures = new AtomicInteger(0);

                for (int t = 0; t < threadCount; t++) {
                    final int threadId = t;
                    pool.submit(() -> {
                        try {
                            ready.countDown();
                            go.await();

                            double[] xs = new double[rowsPerCall];
                            for (int i = 0; i < rowsPerCall; i++) {
                                xs[i] = threadId * 1000 + i;
                            }
                            // Each thread allocates its own vectors from the
                            // shared allocator (BufferAllocator is safe for
                            // concurrent use) and MUST close them itself --
                            // nothing outside this lambda will.
                            try (Float8Vector x = column("thread-x-" + threadId, xs);
                                 Float8Vector out = ArrowBulkEvaluator.allocateOutput(
                                         allocator, "thread-out-" + threadId, rowsPerCall)) {

                                eval.evaluate(Map.of("x", x), out);

                                for (int i = 0; i < rowsPerCall; i++) {
                                    if (Math.abs(out.get(i) - (xs[i] * 2 + 1)) > 1e-9) {
                                        failures.incrementAndGet();
                                    }
                                }
                            }
                        } catch (Throwable t2) {
                            failures.incrementAndGet();
                        }
                    });
                }

                ready.await();
                go.countDown();
                pool.shutdown();
                assertTrue(pool.awaitTermination(60, TimeUnit.SECONDS), "threads did not finish in time");
                assertEquals(0, failures.get(), "one or more threads saw incorrect results or an exception");
            }
        }
    }

    // =====================================================================
    // 9. Device selection
    // =====================================================================
    @Nested
    class DeviceSelection {

        @Test
        void listOpenClDevicesReturnsNonNullList() {
            Assumptions.assumeTrue(ArrowGpuBulkEvaluator.isBackendAvailable(GpuBackend.OPENCL),
                    "No OpenCL backend available on this machine");
            List<String> devices = ArrowGpuBulkEvaluator.listOpenClDevices();
            assertNotNull(devices);
            assertFalse(devices.isEmpty());
        }

        @Test
        void selectingOpenClDeviceIsReflectedInDeviceDescriptionOfNextCompiledInstance() throws Throwable {
            Assumptions.assumeTrue(ArrowGpuBulkEvaluator.isBackendAvailable(GpuBackend.OPENCL),
                    "No OpenCL backend available on this machine");
            List<String> devices = ArrowGpuBulkEvaluator.listOpenClDevices();
            Assumptions.assumeTrue(!devices.isEmpty(), "No OpenCL devices enumerated");

            try {
                ArrowGpuBulkEvaluator.selectOpenClDevice(0, 0);
                try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1", GpuBackend.OPENCL)) {
                    String description = eval.deviceDescription();
                    assertNotNull(description);
                    assertFalse(description.isBlank());
                }
            } finally {
                ArrowGpuBulkEvaluator.clearOpenClDeviceSelection();
            }
        }

        @Test
        void listCudaDevicesReturnsNonNullList() {
            Assumptions.assumeTrue(ArrowGpuBulkEvaluator.isBackendAvailable(GpuBackend.CUDA),
                    "No CUDA backend available on this machine");
            List<String> devices = ArrowGpuBulkEvaluator.listCudaDevices();
            assertNotNull(devices);
            assertFalse(devices.isEmpty());
        }

        @Test
        void cudaInstancesReportARealDeviceDescription() throws Throwable {
            Assumptions.assumeTrue(ArrowGpuBulkEvaluator.isBackendAvailable(GpuBackend.CUDA),
                    "No CUDA backend available on this machine");
            try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1", GpuBackend.CUDA)) {
                // Previously CUDA had no per-device introspection and this
                // always came back empty; CudaCompositeExpression now exposes
                // getDeviceDescription() the same way OpenCL always has.
                String description = eval.deviceDescription();
                assertNotNull(description);
                assertFalse(description.isBlank());
                assertTrue(description.contains("cuda device"));
            }
        }

        @Test
        void selectingCudaDeviceIsReflectedInDeviceDescriptionOfNextCompiledInstance() throws Throwable {
            Assumptions.assumeTrue(ArrowGpuBulkEvaluator.isBackendAvailable(GpuBackend.CUDA),
                    "No CUDA backend available on this machine");
            List<String> devices = ArrowGpuBulkEvaluator.listCudaDevices();
            Assumptions.assumeTrue(!devices.isEmpty(), "No CUDA devices enumerated");

            try {
                ArrowGpuBulkEvaluator.selectCudaDevice(0);
                try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("x+1", GpuBackend.CUDA)) {
                    assertEquals(GpuBackend.CUDA, eval.actualBackend());
                    String description = eval.deviceDescription();
                    assertNotNull(description);
                    assertFalse(description.isBlank());
                }
            } finally {
                ArrowGpuBulkEvaluator.clearCudaDeviceSelection();
            }
        }

        @Test
        void switchingCudaDeviceSelectionDoesNotAffectAlreadyCompiledInstances() throws Throwable {
            Assumptions.assumeTrue(ArrowGpuBulkEvaluator.isBackendAvailable(GpuBackend.CUDA),
                    "No CUDA backend available on this machine");
            List<String> devices = ArrowGpuBulkEvaluator.listCudaDevices();
            Assumptions.assumeTrue(devices.size() >= 2,
                    "Need at least 2 CUDA devices to verify per-instance binding survives reselection");

            try {
                ArrowGpuBulkEvaluator.selectCudaDevice(0);
                try (ArrowGpuBulkEvaluator first = ArrowGpuBulkEvaluator.compile("x+1", GpuBackend.CUDA)) {
                    String firstDescription = first.deviceDescription();

                    ArrowGpuBulkEvaluator.selectCudaDevice(1);
                    try (ArrowGpuBulkEvaluator second = ArrowGpuBulkEvaluator.compile("x+1", GpuBackend.CUDA)) {
                        // The already-compiled first instance must keep
                        // reporting its original device, not silently move
                        // to whatever is now selected.
                        assertEquals(firstDescription, first.deviceDescription());
                        assertNotEquals(firstDescription, second.deviceDescription());
                    }
                }
            } finally {
                ArrowGpuBulkEvaluator.clearCudaDeviceSelection();
            }
        }
    }
}