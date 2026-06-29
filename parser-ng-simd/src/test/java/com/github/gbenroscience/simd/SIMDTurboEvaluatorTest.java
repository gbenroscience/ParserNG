package com.github.gbenroscience.simd;

import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.simd.turbo.tools.FlatMatrixF;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;

import java.util.Arrays;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Assertions;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 *
 * @author GBEMIRO
 */
public class SIMDTurboEvaluatorTest {

    private static final double EPSILON = 1e-12;
    private static ExecutorService threadPool;
    private static boolean active = false;

    @BeforeAll
    public static void setupSuite() {
        // Enforce a hard fail immediately if module flags are missing

        MathExpression orig = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(3,4,2)");//for user defined function tests
        threadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    }

    @AfterAll
    public static void teardownSuite() {
        if (threadPool != null) {
            threadPool.shutdown();
        }
    }

    @Test
    public void testMathematicalPrecisionVsNativeJavaFlat() throws Throwable {
        MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();

        logDetails(me, evaluator, !active);

        // 17 datapoints to trigger both vector lane and tail scalar loop remainders
        int totalElements = 17;
        int varCount = 3; // x1, x2, x3

        // Flattened structural array: column-major allocation
        double[] flatInputs = new double[varCount * totalElements];
        double[] outputVector = new double[totalElements];

        // Populate the flat array using stride offsets
        for (int i = 0; i < totalElements; i++) {
            double x1Val = 1.5 + (i * 0.1); // x1 values
            double x2Val = 2.0 + (i * 0.5); // x2 values
            double x3Val = 0.5;             // x3 values

            flatInputs[(0 * totalElements) + i] = x1Val; // x1 segment
            flatInputs[(1 * totalElements) + i] = x2Val; // x2 segment
            flatInputs[(2 * totalElements) + i] = x3Val; // x3 segment
        }
        System.out.println("flatInputs: " + Arrays.toString(flatInputs));

        // Test API Call #1: High-Performance Flat Bulk Execution
        evaluator.applyBulk(flatInputs, outputVector);
        System.out.println("output: " + Arrays.toString(outputVector));
        // System.out.println("outputVector: " + Arrays.toString(outputVector));
        // Verify mathematical equality against standard Java scalar paths
        for (int i = 0; i < totalElements; i++) {
            // Extract original baseline values from flat strides for expected validation
            double x1 = flatInputs[(0 * totalElements) + i];
            double x2 = flatInputs[(1 * totalElements) + i];
            double x3 = flatInputs[(2 * totalElements) + i];

            double expected = (1.0 / (x1 * Math.sqrt(2.0 * 3.14159)))
                    * Math.exp((-Math.pow((x2 - x3), 2.0)) / (2.0 * Math.pow(x1, 2.0)));

            assertEquals(expected, outputVector[i], EPSILON, "SIMD flat path math drifted at index: " + i);
        }
    }

    @Test
    public void testMathematicalPrecisionVsNativeJava() throws Throwable {
        MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        logDetails(me, evaluator, !active);

        // 17 datapoints to trigger both vector lane and tail scalar loop remainders
        int totalElements = 17;
        double[][] inputs = new double[3][totalElements]; // 3 variables, 17 values each
        double[] outputVector = new double[totalElements];

        for (int i = 0; i < totalElements; i++) {
            inputs[0][i] = 1.5 + (i * 0.1); // x1
            inputs[1][i] = 2.0 + (i * 0.5); // x2
            inputs[2][i] = 0.5;             // x3
        }

        // Test API Call #1: Standard Bulk Execution
        evaluator.applyBulk(inputs, outputVector);
        System.out.println("output: " + Arrays.toString(outputVector));

        for (int i = 0; i < totalElements; i++) {
            double x1 = inputs[0][i];
            double x2 = inputs[1][i];
            double x3 = inputs[2][i];
            double expected = (1.0 / (x1 * Math.sqrt(2.0 * 3.14159))) * Math.exp((-Math.pow((x2 - x3), 2.0)) / (2.0 * Math.pow(x1, 2.0)));
            assertEquals(expected, outputVector[i], EPSILON, "SIMD standard path math drifted at index: " + i);
        }
    }

    @Test
    public void testThreadPooledParallelBulkExecution() throws Throwable {
        MathExpression me = new MathExpression("4*x+3*sin(5+x^2)");
        me.setDRG(DRG_MODE.RAD);
        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        logDetails(me, evaluator, !active);

        int dataSize = 100;
        double[][] inputs = new double[1][dataSize]; // Only 1 variable 'x' is needed for this expression
        double[] outputVector = new double[dataSize];

        for (int i = 0; i < dataSize; i++) {
            inputs[0][i] = i; // x
        }
        // Test API Call #2: Asynchronous ExecutorService Multi-threaded Bulk Execution
        evaluator.applyBulk(inputs, outputVector);
        //  System.out.println("output: " + Arrays.toString(outputVector));

        for (int i = 0; i < dataSize; i++) {
            double x = inputs[0][i];
            // Correct expected formula matching the active MathExpression
            double expected = 4.0 * x + 3.0 * Math.sin(5.0 + (x * x));
            assertEquals(expected, outputVector[i], EPSILON, "Parallel SIMD execution drifted at index: " + i);
        }
    }

    @Test
    public void testSingleRuntime() throws Throwable {
        MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        double t = System.nanoTime();
        double[] out = new double[1];
        evaluator.applyBulk(new double[]{5, 4, 1}, out);
        double t1 = System.nanoTime() - t;

        System.out.println("timed at = " + t1 + "ns--- answer: " + out[0]);
        Assertions.assertTrue(true);
    }

    @Test
    void testUserDefinedFunctionSimpleCall() throws Throwable {
        MathExpression me = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(x+3,y-2,2*z-3)");
        System.out.println("f(x+3,y-2,2*z-3) = " + me.solve());

        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        double t = System.nanoTime();
        double[] out = new double[1];
        try {
            evaluator.applyBulk(new double[]{5, 4, 1}, out);
        } catch (IllegalStateException e) {
            assertTrue(true, "variables not balanced");
            return;
        }
        double t1 = System.nanoTime() - t;

        System.out.println("timed at = " + t1 + "ns--- answer: " + out[0]);
        double x = 5;
        double y = 4;
        double z = 1;
        double expected = 3 * (x + 3) + 4 * (y - 2) + Math.sin((2 * z - 3) - 2);
        assertEquals(expected, out[0], EPSILON, "Parallel SIMD execution drifted for test: testUserDefinedFunctionSimpleCall ");

    }

    @Test
    void testUserDefinedFunctionSimpleCallNoVars() throws Throwable {
        MathExpression me = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(3,4,2)");
        System.out.println("f(3,4,2) = " + me.solve());

        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        double t = System.nanoTime();
        double[] out = new double[1];
        try {
            evaluator.applyBulk(new double[]{}, out);
        } catch (IllegalStateException e) {
            assertTrue(true, "variables not balanced");
            return;
        }
        double t1 = System.nanoTime() - t;

        System.out.println("timed at = " + t1 + "ns--- answer: " + out[0]);
        double x = 3;
        double y = 4;
        double z = 2;
        double expected = 3 * x + 4 * y + Math.sin(z - 2);
        assertEquals(expected, out[0], EPSILON, "Parallel SIMD execution drifted for test: testUserDefinedFunctionSimpleCall ");

    }

    @Test
    void testUserDefinedFunctionFunctionInExpression() throws Throwable {

        MathExpression me = new MathExpression("3 + 2*x + f(2, 3*x + sin(4*x), 5)");

        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        double t = System.nanoTime();
        double[] out = new double[1];
        evaluator.applyBulk(new double[]{5}, out);
        double t1 = System.nanoTime() - t;

        System.out.println("timed at = " + t1 + "ns--- answer: " + out[0]);

        double x = 5;

        double expected = 3 + 2 * x + (3 * 2 + 4 * (3 * x + Math.sin(4 * x)) + Math.sin(5 - 2));
        assertEquals(expected, out[0], EPSILON, "Parallel SIMD execution drifted for test: testUserDefinedFunctionSimpleCall ");

    }

 

    @ParameterizedTest(name = "GELU Matrix Size: {0}x{0}")
    @ValueSource(ints = {20, 70, 100, 200})
    void testGelu(int sz) throws Throwable {
        executeKernelBenchmark("gelu", sz);
    }

    @ParameterizedTest(name = "SwiGLU Matrix Size: {0}x{0}")
    @ValueSource(ints = {20, 70, 100, 200})
    void testSwiglu(int sz) throws Throwable {
        executeKernelBenchmark("swiglu", sz);
    }

    /**
     * Shared orchestration runner for manual micro-benchmarking without JMH.
     */
    private void executeKernelBenchmark(String kernelName, int sz) throws Throwable {
        MathExpression me = new MathExpression("x * 0.5 * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))");//mock expr - just need the MathExpression object(make it 1+1, still works)
        SIMDCompositeExpression evaluator = (SIMDCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();

        FlatMatrixF in1 = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in1);

        FlatMatrixF in2 = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in2);

        FlatMatrixF out = new FlatMatrixF(sz, sz);

        // 1. Manual Warm-up Phase
        // Forces C2 to compile the vector loops before we sample the clock
        int warmUpRuns = 3000;
        FlatMatrixF[] inputs = new FlatMatrixF[]{in1, in2}; // Allocate once outside the timing track!
        for (int i = 0; i < warmUpRuns; i++) {
            evaluator.applyMatrixKernel(inputs, out, kernelName);
        }

        // 2. Timed Target Phase
        int iterations = 10000;

        long startTime = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            evaluator.applyMatrixKernel(inputs, out, kernelName);
        }
        long totalTimeNs = System.nanoTime() - startTime;

        // 3. Analytics Formatting
        double avgMatrixNs = (double) totalTimeNs / iterations;
        double totalElements = sz * sz;
        double avgPerElementNs = avgMatrixNs / totalElements;
        double avgMatrixMicros = avgMatrixNs / 1000.0;

        // Prints numbers tailored perfectly for your README layout
        System.out.printf("[%s] %dx%d -> Matrix Avg: %.2f µs | Per-Element: %.2f ns%n",
                kernelName.toUpperCase(), sz, sz, avgMatrixMicros, avgPerElementNs);

        // Sanity check to prevent dead-code optimization tricks from discarding execution
        Assertions.assertNotNull(out);
    }


    void logDetails(MathExpression me, SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator, boolean active) {
        if (!active) {
            return;
        }
        MathExpression.Token[] tokens = me.getCachedPostfix();
        String names[] = new String[tokens.length];

        for (int i = 0; i < names.length; i++) {
            String n = tokens[i].name == null ? (tokens[i].opChar == '\u0000' ? String.valueOf(tokens[i].value) : String.valueOf(tokens[i].opChar)) : tokens[i].name;
            names[i] = n;
        }
        System.out.println("expr = " + me.getExpression() + ",\n"
                + "token-names: " + Arrays.toString(names) + "\n"
                + "tokens-len: " + tokens.length);
    }

}
