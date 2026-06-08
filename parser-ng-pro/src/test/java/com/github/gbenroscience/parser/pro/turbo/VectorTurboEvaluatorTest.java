package com.github.gbenroscience.parser.pro.turbo;

import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.parser.MathExpression; 
import com.github.gbenroscience.parser.pro.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.parser.pro.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression;

import java.util.Arrays;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import static org.junit.jupiter.api.Assertions.*;

/**
 *
 * @author GBEMIRO
 */
public class VectorTurboEvaluatorTest {

    private static final double EPSILON = 1e-12;
    private static ExecutorService threadPool;

    @BeforeAll
    public static void setupSuite() {
        // Enforce a hard fail immediately if module flags are missing
        try {
            Class.forName("jdk.incubator.vector.DoubleVector");
        } catch (ClassNotFoundException e) {
            fail("CRITICAL: Incubator Vector module not enabled. Pass JVM arg: --add-modules jdk.incubator.vector");
        }
        threadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    }

    @AfterAll
    public static void teardownSuite() {
        if (threadPool != null) {
            threadPool.shutdown();
        }
    }

    @Test
    public void testMathematicalPrecisionVsNativeJava() throws Throwable {
        MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();

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
        BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();

        int dataSize = 100;
        double[][] inputs = new double[1][dataSize]; // Only 1 variable 'x' is needed for this expression
        double[] outputVector = new double[dataSize];

        for (int i = 0; i < dataSize; i++) {
            inputs[0][i] = i; // x
        }
 System.out.println("inputs: "+Arrays.toString(inputs[0]));
        // Test API Call #2: Asynchronous ExecutorService Multi-threaded Bulk Execution
        evaluator.applyBulk(inputs, outputVector);
        System.out.println("output: "+Arrays.toString(outputVector));

        for (int i = 0; i < dataSize; i++) {
            double x = inputs[0][i];
            // Correct expected formula matching the active MathExpression
            double expected = 4.0 * x + 3.0 * Math.sin(5.0 + (x * x));
            assertEquals(expected, outputVector[i], EPSILON, "Parallel SIMD execution drifted at index: " + i);
        }
    }

    @Test
    public void testTargetMemoryOffsetBoundsSafety() throws Throwable {
        MathExpression me = new MathExpression("x1 + 10");
        BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();

        // 4 elements to evaluate
        int dataSize = 4;
        double[][] inputs = new double[1][dataSize]; // 1 variable, 4 values
        inputs[0] = new double[]{5.0, 6.0, 7.0, 8.0};

        // Single flat 1D buffer array of length 10
        double[] secureBuffer = new double[10];

        // Test API Call #3: Memory-Reuse Offset-Based Bulk Execution
        int targetOffset = 3;
        evaluator.applyBulk(inputs, secureBuffer, targetOffset);

        // Verify guard bounds before the offset window are clean (0.0)
        for (int i = 0; i < targetOffset; i++) {
            assertEquals(0.0, secureBuffer[i], "Memory written before target offset.");
        }

        // Verify accurate calculation window calculations (e.g. 5.0 + 10.0 = 15.0)
        for (int i = 0; i < dataSize; i++) {
            double expected = inputs[0][i] + 10.0;
            assertEquals(expected, secureBuffer[targetOffset + i], EPSILON, "Calculation error inside offset zone.");
        }

        // Verify guard bounds after the payload window remain clean (0.0)
        for (int i = targetOffset + dataSize; i < secureBuffer.length; i++) {
            assertEquals(0.0, secureBuffer[i], "Memory overshot past target payload window.");
        }
    }
}
