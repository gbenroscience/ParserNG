package com.github.gbenroscience.parser;

import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.parser.ng.bench.JaninoVectorTurboEvaluator;

import java.util.Arrays;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;
import static org.junit.jupiter.api.Assertions.*;

/**
 *
 * @author GBEMIRO
 */
public class JaninoTurboEvaluatorTest {

    private static final double EPSILON = 1e-12;
    private static boolean active = false;

    @BeforeAll
    public static void setupSuite() {
        // Enforce a hard fail immediately if module flags are missing

        MathExpression orig = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(3,4,2)");//for user defined function tests
    }

    @AfterAll
    public static void teardownSuite() {

    }

    /**
     * @Test public void testMathematicalPrecisionVsNativeJavaFlat() throws
     * Throwable { MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 *
     * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
     * JaninoVectorTurboEvaluator.JaninoBulkExpression evaluator =
     * (JaninoVectorTurboEvaluator.JaninoBulkExpression) new
     * JaninoVectorTurboEvaluator(me).compile();
     *
     * logDetails(me, evaluator, !active);
     *
     * // 17 datapoints to trigger both vector lane and tail scalar loop
     * remainders int totalElements = 17; int varCount = 3; // x1, x2, x3
     *
     * // Flattened structural array: column-major allocation double[]
     * flatInputs = new double[varCount * totalElements]; double[] outputVector
     * = new double[totalElements];
     *
     * // Populate the flat array using stride offsets for (int i = 0; i <
     * totalElements; i++) { double x1Val = 1.5 + (i * 0.1); // x1 values double
     * x2Val = 2.0 + (i * 0.5); // x2 values double x3Val = 0.5; // x3 values
     *
     * flatInputs[(0 * totalElements) + i] = x1Val; // x1 segment flatInputs[(1
     * * totalElements) + i] = x2Val; // x2 segment flatInputs[(2 *
     * totalElements) + i] = x3Val; // x3 segment }
     * System.out.println("flatInputs: "+Arrays.toString(flatInputs));
     *
     * // Test API Call #1: High-Performance Flat Bulk Execution
     * evaluator.applyBulk(flatInputs, outputVector);
     * System.out.println("output: "+Arrays.toString(outputVector)); //
     * System.out.println("outputVector: " + Arrays.toString(outputVector)); //
     * Verify mathematical equality against standard Java scalar paths for (int
     * i = 0; i < totalElements; i++) { // Extract original baseline values from
     * flat strides for expected validation double x1 = flatInputs[(0 *
     * totalElements) + i]; double x2 = flatInputs[(1 * totalElements) + i];
     * double x3 = flatInputs[(2 * totalElements) + i];
     *
     * double expected = (1.0 / (x1 * Math.sqrt(2.0 * 3.14159)))
     * Math.exp((-Math.pow((x2 - x3), 2.0)) / (2.0 * Math.pow(x1, 2.0)));
     *
     * assertEquals(expected, outputVector[i], EPSILON, "SIMD flat path math
     * drifted at index: " + i); } }
     */
    @Test
    public void testMathematicalPrecisionVsNativeJava() throws Throwable {
        MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        JaninoVectorTurboEvaluator.JaninoBulkExpression evaluator = (JaninoVectorTurboEvaluator.JaninoBulkExpression) new JaninoVectorTurboEvaluator(me).compile();
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
        MathExpression me = new MathExpression("4*x+3*sin(5+y^2)+2*cos(z)");
        me.setDRG(DRG_MODE.RAD);
        JaninoVectorTurboEvaluator.JaninoBulkExpression evaluator = (JaninoVectorTurboEvaluator.JaninoBulkExpression) new JaninoVectorTurboEvaluator(me).compile();
        logDetails(me, evaluator, !active);

        int dataSize = 100;
        double[][] inputs = new double[me.registry.size()][dataSize]; // Only 1 variable 'x' is needed for this expression
        double[] outputVector = new double[dataSize];

        for (int i = 0; i < dataSize; i++) {
            inputs[0][i] = i; // x
            inputs[1][i] = i + 1; // x
            inputs[2][i] = i + 2; // x
        }
        // Test API Call #2: Asynchronous ExecutorService Multi-threaded Bulk Execution
        evaluator.applyBulk(inputs, outputVector);
        //  System.out.println("output: " + Arrays.toString(outputVector));

        for (int i = 0; i < dataSize; i++) {
            double x = inputs[0][i];
            double y = inputs[1][i];
            double z = inputs[2][i];
            // Correct expected formula matching the active MathExpression
            double expected = 4.0 * x + 3.0 * Math.sin(5.0 + (y * y)) + 2 * Math.cos(z);
            assertEquals(expected, outputVector[i], EPSILON, "Parallel SIMD execution drifted at index: " + i);
        }
    }

    @Test
    public void testSingleRuntimeGaussian() throws Throwable {
        MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        JaninoVectorTurboEvaluator.JaninoBulkExpression evaluator = (JaninoVectorTurboEvaluator.JaninoBulkExpression) new JaninoVectorTurboEvaluator(me).compile();
        double t = System.nanoTime();
        double[] out = new double[1];
        double[][] inputs = new double[3][1];
        inputs[0][0] = 5;  // x1 (Must be >0 for div/pow)
        inputs[1][0] = 4.0;        // x2
        inputs[2][0] = 1.0;        // x3
        evaluator.applyBulk(inputs, out);
        double t1 = System.nanoTime() - t;

        System.out.println("timed at = " + t1 + "ns--- answer: " + out[0]);
        double x1 = inputs[0][0];
        double x2 = inputs[1][0];
        double x3 = inputs[2][0];
        // Correct expected formula matching the active MathExpression
        double expected = (1.0 / (x1 * Math.sqrt(2.0 * 3.14159))) * Math.exp((-Math.pow((x2 - x3), 2.0)) / (2.0 * Math.pow(x1, 2.0)));
        assertEquals(expected, out[0], EPSILON, "Parallel SIMD execution drifted at index: 0");
    }

    @Test
    public void testSingleRuntimeLinear() throws Throwable {
        MathExpression me = new MathExpression("12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2");
        JaninoVectorTurboEvaluator.JaninoBulkExpression evaluator = (JaninoVectorTurboEvaluator.JaninoBulkExpression) new JaninoVectorTurboEvaluator(me).compile();
        double t = System.nanoTime();
        double[] out = new double[1];
        double[][] inputs = new double[3][1];
        inputs[0][0] = 3.141;  // x1 (Must be >0 for div/pow)
        inputs[1][0] = 2.011;        // x2
        inputs[2][0] = 1.732;        // x3
        evaluator.applyBulk(inputs, out);
        double t1 = System.nanoTime() - t;

        System.out.println("timed at = " + t1 + "ns--- answer: " + out[0]);
        double x1 = inputs[0][0];
        double x2 = inputs[1][0];
        double x3 = inputs[2][0];
        // Correct expected formula matching the active MathExpression
        double expected = 12*x1 +3*x2 - 4*x3 +5*x1 - x2 - 4*x3 + 2*x1 + x2;
        assertEquals(expected, out[0], EPSILON, "Parallel SIMD execution drifted at index: 0");
    }

    void logDetails(MathExpression me, JaninoVectorTurboEvaluator.JaninoBulkExpression evaluator, boolean active) {
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
