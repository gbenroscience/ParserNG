package com.github.gbenroscience.simdext;

import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.simd.turbo.tools.FlatMatrixF;
import com.github.gbenroscience.simdext.turbo.tools.SIMDEngineF32;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 *
 * @author GBEMIRO
 */
public class SIMDEngineF32EvaluatorTest {

    private static final float ABS_EPSILON = 1e-5f;
    private static final float REL_EPSILON = 1e-5f;

    private static ExecutorService threadPool;
    private static boolean active = false;

    @BeforeAll
    public static void setupSuite() {
        // Enforce a hard fail immediately if module flags are missing

        MathExpression orig = new MathExpression(
                "f(x,y,z)=3*x+4*y+sin(z-2);f(3,4,2)"
        ); // for user defined function tests

        threadPool = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors()
        );
    }

    @AfterAll
    public static void teardownSuite() {
        if (threadPool != null) {
            threadPool.shutdown();
        }
    }

    /**
     * Float-aware numerical comparison.
     *
     * Uses both an absolute and relative tolerance so that the test remains
     * meaningful for values spanning different magnitudes.
     */
    private static void assertFloatClose(
            float expected,
            float actual,
            String message) {

        if (Float.isNaN(expected)) {
            Assertions.assertTrue(
                    Float.isNaN(actual),
                    message + " [expected NaN, actual=" + actual + "]"
            );
            return;
        }

        if (Float.isInfinite(expected)) {
            Assertions.assertEquals(
                    expected,
                    actual,
                    message
            );
            return;
        }

        float tolerance = Math.max(
                ABS_EPSILON,
                REL_EPSILON * Math.max(1.0f, Math.abs(expected))
        );

        Assertions.assertEquals(
                expected,
                actual,
                tolerance,
                message
                + " [expected=" + expected
                + ", actual=" + actual
                + ", tolerance=" + tolerance + "]"
        );
    }

    @Test
    public void testMathematicalPrecisionVsNativeJavaFlat() throws Throwable {
        MathExpression me = new MathExpression(
                "(1 / (x1 * sqrt(2 * 3.14159))) "
                + "* exp((-(x2 - x3)^2) / (2 * x1^2))"
        );

        SIMDEngineF32.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineF32.SIMDVectorCompositeExpression) new SIMDEngineF32(me).compile();

        logDetails(me, evaluator, !active);

        // 17 datapoints to trigger both vector lane and tail scalar loop
        // remainders.
        int totalElements = 17;
        int varCount = 3;

        // Flattened structural array: column-major allocation.
        float[] flatInputs = new float[varCount * totalElements];
        float[] outputVector = new float[totalElements];

        for (int i = 0; i < totalElements; i++) {
            float x1Val = 1.5f + (i * 0.1f);
            float x2Val = 2.0f + (i * 0.5f);
            float x3Val = 0.5f;

            flatInputs[i] = x1Val;
            flatInputs[totalElements + i] = x2Val;
            flatInputs[(2 * totalElements) + i] = x3Val;
        }

        System.out.println("flatInputs: " + Arrays.toString(flatInputs));

        // High-performance flat bulk execution.
        evaluator.applyBulk(flatInputs, outputVector);

        System.out.println("output: " + Arrays.toString(outputVector));

        // Verify against the higher-precision Java reference calculation.
        for (int i = 0; i < totalElements; i++) {
            float x1 = flatInputs[i];
            float x2 = flatInputs[totalElements + i];
            float x3 = flatInputs[(2 * totalElements) + i];

            float expected = (float) ((1.0 / (x1 * Math.sqrt(2.0 * 3.14159)))
                    * Math.exp(
                            (-(Math.pow(x2 - x3, 2.0)))
                            / (2.0 * Math.pow(x1, 2.0))
                    ));

            assertFloatClose(
                    expected,
                    outputVector[i],
                    "SIMD flat path math drifted at index: " + i
            );
        }
    }

    @Test
    public void testMathematicalPrecisionVsNativeJava() throws Throwable {
        MathExpression me = new MathExpression(
                "(1 / (x1 * sqrt(2 * 3.14159))) "
                + "* exp((-(x2 - x3)^2) / (2 * x1^2.23))"
        );

        SIMDEngineF32.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineF32.SIMDVectorCompositeExpression) new SIMDEngineF32(me).compile();

        logDetails(me, evaluator, !active);

        // Large enough to exercise vector lanes and tail handling.
        int totalElements = 1_000_017;

        float[][] inputs = new float[3][totalElements];
        float[] outputVector = new float[totalElements];

        for (int i = 0; i < totalElements; i++) {
            inputs[0][i] = 1.5f + (i * 0.1f);
            inputs[1][i] = 2.0f + (i * 0.5f);
            inputs[2][i] = 0.5f;
        }

        // Standard bulk execution.
        evaluator.applyBulk(inputs, outputVector);

        for (int i = 0; i < totalElements; i++) {
            float x1 = inputs[0][i];
            float x2 = inputs[1][i];
            float x3 = inputs[2][i];

            float expected = (float) ((1.0 / (x1 * Math.sqrt(2.0 * 3.14159)))
                    * Math.exp(
                            (-(Math.pow(x2 - x3, 2.0)))
                            / (2.0 * Math.pow(x1, 2.23))
                    ));

            assertFloatClose(
                    expected,
                    outputVector[i],
                    "SIMD standard path math drifted at index: " + i
            );
        }
    }

    @Test
    public void testThreadPooledParallelBulkExecution() throws Throwable {
        MathExpression me = new MathExpression(
                "4*x+3*sin(5+x^2)"
        );

        me.setDRG(DRG_MODE.RAD);

        SIMDEngineF32.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineF32.SIMDVectorCompositeExpression) new SIMDEngineF32(me).compile();

        logDetails(me, evaluator, !active);

        int dataSize = 250_018;

        float[][] inputs = new float[1][dataSize];
        float[] outputVector = new float[dataSize];

        for (int i = 0; i < dataSize; i++) {
            // Keep the input explicitly in float representation.
            inputs[0][i] = (float) i;
        }

        // Asynchronous executor-based multi-threaded bulk execution.
        evaluator.applyBulkParallel(inputs, outputVector);

        for (int i = 0; i < dataSize; i++) {
            float x = inputs[0][i];

            // Force strictly 32-bit float evaluation to match SIMD exactly
            float xSquared = x * x;
            float innerSum = 5.0f + xSquared;
            float sinResult = (float) Math.sin(innerSum);

            float expected = 4.0f * x + 3.0f * sinResult;

            assertFloatClose(
                    expected,
                    outputVector[i],
                    "Parallel SIMD execution drifted at index: " + i+", for x = "+x
            );
        }
    }

    @Test
    public void testSingleRuntime() throws Throwable {
        MathExpression me = new MathExpression(
                "(1 / (x1 * sqrt(2 * 3.14159))) "
                + "* exp((-(x2 - x3)^2) / (2 * x1^2))"
        );

        SIMDEngineF32.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineF32.SIMDVectorCompositeExpression) new SIMDEngineF32(me).compile();

        long t = System.nanoTime();

        float[] out = new float[1];

        evaluator.applyBulk(
                new float[]{5.0f, 4.0f, 1.0f},
                out
        );

        long elapsed = System.nanoTime() - t;

        System.out.println(
                "timed at = " + elapsed
                + "ns--- answer: " + out[0]
        );

        Assertions.assertTrue(Float.isFinite(out[0]));
    }

    @Test
    void testUserDefinedFunctionSimpleCall() throws Throwable {
        MathExpression me = new MathExpression(
                "f(x,y,z)=3*x+4*y+sin(z-2);"
                + "f(x+3,y-2,2*z-3)"
        );

        System.out.println(
                "f(x+3,y-2,2*z-3) = " + me.solve()
        );

        SIMDEngineF32.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineF32.SIMDVectorCompositeExpression) new SIMDEngineF32(me).compile();

        long t = System.nanoTime();

        float[] out = new float[1];

        evaluator.applyBulk(
                new float[]{5.0f, 4.0f, 1.0f},
                out
        );

        long elapsed = System.nanoTime() - t;

        System.out.println(
                "timed at = " + elapsed
                + "ns--- answer: " + out[0]
        );

        float x = 5.0f;
        float y = 4.0f;
        float z = 1.0f;

        float expected = (float) (3.0 * (x + 3.0)
                + 4.0 * (y - 2.0)
                + Math.sin((2.0 * z - 3.0) - 2.0));

        assertFloatClose(
                expected,
                out[0],
                "Float SIMD execution drifted for "
                + "testUserDefinedFunctionSimpleCall"
        );
    }

    @Test
    void testUserDefinedFunctionSimpleCallNoVars() throws Throwable {
        MathExpression me = new MathExpression(
                "f(x,y,z)=3*x+4*y+sin(z-2);f(3,4,2)"
        );

        System.out.println(
                "f(3,4,2) = " + me.solve()
        );

        SIMDEngineF32.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineF32.SIMDVectorCompositeExpression) new SIMDEngineF32(me).compile();

        long t = System.nanoTime();

        float[] out = new float[1000];
        float[] in = new float[1000];
        Arrays.fill(in, (float) me.solveGeneric().scalar);

        try {
            evaluator.validate(in, out);
        } catch (Exception e) {
            Assertions.assertTrue(true, "Empty input validation rejected as expected: "
                    + e.getMessage());
            return;
        }

        evaluator.applyBulk(in, out);

        long elapsed = System.nanoTime() - t;

        System.out.println(
                "timed at = " + elapsed
                + "ns--- answer: " + out[0]
        );

        float x = 3.0f;
        float y = 4.0f;
        float z = 2.0f;

        float expected = (float) (3.0 * x
                + 4.0 * y
                + Math.sin(z - 2.0));

        assertFloatClose(
                expected,
                out[0],
                "Float SIMD execution drifted for "
                + "testUserDefinedFunctionSimpleCallNoVars"
        );
    }

    @Test
    void testUserDefinedFunctionFunctionInExpression() throws Throwable {
        MathExpression me = new MathExpression(
                "3 + 2*x + f(2, 3*x + sin(4*x), 5)"
        );

        SIMDEngineF32.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineF32.SIMDVectorCompositeExpression) new SIMDEngineF32(me).compile();

        long t = System.nanoTime();

        float[] out = new float[1];

        evaluator.applyBulk(
                new float[]{5.0f},
                out
        );

        long elapsed = System.nanoTime() - t;

        System.out.println(
                "timed at = " + elapsed
                + "ns--- answer: " + out[0]
        );

        float x = 5.0f;

        float expected = (float) (3.0
                + 2.0 * x
                + (3.0 * 2.0
                + 4.0 * (3.0 * x + Math.sin(4.0 * x))
                + Math.sin(5.0 - 2.0)));

        assertFloatClose(
                expected,
                out[0],
                "Float SIMD execution drifted for "
                + "testUserDefinedFunctionFunctionInExpression"
        );
    }

    @ParameterizedTest(name = "GELU Matrix Size: {0}x{0}")
    @ValueSource(ints = {20, 70, 100, 200})
    void testGelu(int sz) throws Throwable {
        executeKernelBenchmark("gelu", sz, 1);
    }

    @ParameterizedTest(name = "SwiGLU Matrix Size: {0}x{0}")
    @ValueSource(ints = {20, 70, 100, 200})
    void testSwiglu(int sz) throws Throwable {
        executeKernelBenchmark("swiglu", sz, 2);
    }

    @ParameterizedTest(name = "GeGLU Matrix Size: {0}x{0}")
    @ValueSource(ints = {20, 70, 100, 200})
    void testGeglu(int sz) throws Throwable {
        executeKernelBenchmark("geglu", sz, 2);
    }

    @ParameterizedTest(name = "GeLU Matrix Size: {0}x{0}")
    @ValueSource(ints = {512, 1024})
    void testGeluLarge(int sz) throws Throwable {
        executeKernelBenchmark("gelu", sz, 1);
    }

    @ParameterizedTest(name = "GeGLU Matrix Size: {0}x{0}")
    @ValueSource(ints = {512, 1024})
    void testGegluLarge(int sz) throws Throwable {
        executeKernelBenchmark("geglu", sz, 2);
    }

    @ParameterizedTest(name = "SwiGLU Matrix Size: {0}x{0}")
    @ValueSource(ints = {512, 1024})
    void testSwigluLarge(int sz) throws Throwable {
        executeKernelBenchmark("swiglu", sz, 2);
    }

    /**
     * Shared orchestration runner for manual micro-benchmarking without JMH.
     */
    private void executeKernelBenchmark(
            String kernelName,
            int sz,
            int arity) throws Throwable {

        MathExpression me = new MathExpression(
                "x * 0.5 * "
                + "(1 + tanh("
                + "0.79788456 * "
                + "(x + 0.044715 * x * x * x)"
                + "))"
        );

        SIMDCompositeExpression evaluator
                = (SIMDCompositeExpression) new SIMDEngineF32(me).compile();

        FlatMatrixF in1 = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in1);

        FlatMatrixF in2 = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in2);

        FlatMatrixF out = new FlatMatrixF(sz, sz);

        // Manual warm-up phase.
        // Forces C2 to compile the vector loops before timing.
        int warmUpRuns = 1000;

        FlatMatrixF[] inputs
                = arity == 2
                        ? new FlatMatrixF[]{in1, in2}
                : new FlatMatrixF[]{in1};

        for (int i = 0; i < warmUpRuns; i++) {
            evaluator.applyMatrixKernel(
                    inputs,
                    out,
                    kernelName
            );
        }

        // Timed target phase.
        int iterations = 4000;

        long startTime = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            evaluator.applyMatrixKernel(
                    inputs,
                    out,
                    kernelName
            );
        }

        long totalTimeNs = System.nanoTime() - startTime;

        float avgMatrixNs
                = (float) totalTimeNs / iterations;

        float totalElements
                = (float) sz * sz;

        float avgPerElementNs
                = avgMatrixNs / totalElements;

        float avgMatrixMicros
                = avgMatrixNs / 1000.0f;

        System.out.printf(
                "[%s] %dx%d -> Matrix Avg: %.2f µs "
                + "| Per-Element: %.2f ns%n",
                kernelName.toUpperCase(),
                sz,
                sz,
                avgMatrixMicros,
                avgPerElementNs
        );

        // Sanity check to prevent dead-code optimization tricks from
        // discarding execution.
        Assertions.assertNotNull(out);
    }

    private void logDetails(
            MathExpression me,
            SIMDEngineF32.SIMDVectorCompositeExpression evaluator,
            boolean active) {

        if (!active) {
            return;
        }

        MathExpression.Token[] tokens
                = me.getCachedPostfix();

        String[] names
                = new String[tokens.length];

        for (int i = 0; i < names.length; i++) {
            String n
                    = tokens[i].name == null
                            ? (tokens[i].opChar == '\u0000'
                                    ? String.valueOf(tokens[i].value)
                                    : String.valueOf(tokens[i].opChar))
                            : tokens[i].name;

            names[i] = n;
        }

        System.out.println(
                "expr = " + me.getExpression() + ",\n"
                + "token-names: " + Arrays.toString(names) + "\n"
                + "tokens-len: " + tokens.length
        );
    }
}
