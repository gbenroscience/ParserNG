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
package com.github.gbenroscience.simd;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator.VectorMath;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class VectorMathTests1 {

    private static final double EPSILON = 1e-10;
    private static final boolean active = false;

    private void logDetails(Object me, Object evaluator, boolean log) {
        if (log) {
            System.out.println("Expression: " + me.toString());
        }
    }

    // ========================================================================
    // PART 1: EVALUATOR INTEGRATION TESTS (MathExpression)
    // ========================================================================

    @Test
    public void testEvaluatorTrigonometryAndReciprocal() throws Throwable {
        // Parser input strictly takes base names. DRG mode handles the mapping internally.
        MathExpression me = new MathExpression("sin(x1) * sec(x2) + cot(x3)");
        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = 
                (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();

        logDetails(me, evaluator, !active);

        int totalElements = 270; // > VECTOR_THRESHOLD (256) to trigger SIMD Vector Lanes
        int varCount = 3;
        double[] flatInputs = new double[varCount * totalElements];
        double[] outputVector = new double[totalElements];

        for (int i = 0; i < totalElements; i++) {
            flatInputs[(0 * totalElements) + i] = 1.0 + (i * 0.01); // x1
            flatInputs[(1 * totalElements) + i] = 0.5 + (i * 0.01); // x2
            flatInputs[(2 * totalElements) + i] = 1.5 + (i * 0.01); // x3
        }

        evaluator.applyBulk(flatInputs, outputVector);

        for (int i = 0; i < totalElements; i++) {
            double x1 = flatInputs[(0 * totalElements) + i];
            double x2 = flatInputs[(1 * totalElements) + i];
            double x3 = flatInputs[(2 * totalElements) + i];

            double expected = Math.sin(x1) * (1.0 / Math.cos(x2)) + (1.0 / Math.tan(x3));
            assertEquals(expected, outputVector[i], EPSILON, "Trig Evaluator drift at index: " + i);
        }
    }

    @Test
    public void testEvaluatorInverseReciprocal() throws Throwable {
        MathExpression me = new MathExpression("acsc(x1) + asec(x2) - acot(x3)");
        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = 
                (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();

        logDetails(me, evaluator, !active);

        int totalElements = 270;
        int varCount = 3;
        double[] flatInputs = new double[varCount * totalElements];
        double[] outputVector = new double[totalElements];

        for (int i = 0; i < totalElements; i++) {
            // Values must be >= 1.0 or <= -1.0 for valid domain of acsc and asec
            flatInputs[(0 * totalElements) + i] = 2.0 + (i * 0.1); // x1
            flatInputs[(1 * totalElements) + i] = -3.0 - (i * 0.1); // x2
            flatInputs[(2 * totalElements) + i] = 0.5 + (i * 0.1); // x3
        }

        evaluator.applyBulk(flatInputs, outputVector);

        for (int i = 0; i < totalElements; i++) {
            double x1 = flatInputs[(0 * totalElements) + i];
            double x2 = flatInputs[(1 * totalElements) + i];
            double x3 = flatInputs[(2 * totalElements) + i];

            double expected = Math.asin(1.0 / x1) + Math.acos(1.0 / x2) - Math.atan(1.0 / x3);
            assertEquals(expected, outputVector[i], EPSILON, "Inverse Reciprocal drift at index: " + i);
        }
    }

    // ========================================================================
    // PART 2: COMPREHENSIVE VECTOR MATH DIRECT TESTS
    // ========================================================================
    
    @FunctionalInterface
    interface VectorAction { void apply(int base, int n, double[] s); }
    @FunctionalInterface
    interface ScalarAction { double apply(double x); }

    private void runVectorAccuracyTest(VectorAction vAct, ScalarAction sAct, double min, double max) {
        int n = 265; // 256 vector lane + 9 scalar tail
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = min + (max - min) * (i / (double)(n - 1));
        }

        double[] expected = new double[n];
        for (int i = 0; i < n; i++) {
            expected[i] = sAct.apply(s[i]);
        }

        // Execute VectorMath operation
        vAct.apply(0, n, s);

        for (int i = 0; i < n; i++) {
            // Use slightly higher epsilon for polynomial approximations
            assertEquals(expected[i], s[i], 1e-8, "Math drifted at index " + i);
        }
    }

    @Test
    public void testDirectTrigonometry() {
        runVectorAccuracyTest(VectorMath::sin, Math::sin, -Math.PI, Math.PI);
        runVectorAccuracyTest(VectorMath::cos, Math::cos, -Math.PI, Math.PI);
        runVectorAccuracyTest(VectorMath::tan, Math::tan, -1.0, 1.0);
    }

    @Test
    public void testDirectReciprocalTrigonometry() {
        runVectorAccuracyTest(VectorMath::sec, x -> 1.0 / Math.cos(x), -1.0, 1.0);
        runVectorAccuracyTest(VectorMath::csc, x -> 1.0 / Math.sin(x), 0.5, 2.5);
        runVectorAccuracyTest(VectorMath::cot, x -> 1.0 / Math.tan(x), 0.5, 2.5);
    }

    @Test
    public void testDirectInverseTrigonometry() {
        runVectorAccuracyTest(VectorMath::asin, Math::asin, -0.99, 0.99);
        runVectorAccuracyTest(VectorMath::acos, Math::acos, -0.99, 0.99);
        runVectorAccuracyTest(VectorMath::atan, Math::atan, -10.0, 10.0);
    }

    @Test
    public void testDirectInverseReciprocalTrig() {
        runVectorAccuracyTest(VectorMath::acsc, x -> Math.asin(1.0 / x), 1.5, 10.0);
        runVectorAccuracyTest(VectorMath::asec, x -> Math.acos(1.0 / x), -10.0, -1.5);
        runVectorAccuracyTest(VectorMath::acot, x -> Math.atan(1.0 / x), -10.0, 10.0);
    }

    @Test
    public void testDirectHyperbolic() {
        runVectorAccuracyTest(VectorMath::sinh, Math::sinh, -5.0, 5.0);
        runVectorAccuracyTest(VectorMath::cosh, Math::cosh, -5.0, 5.0);
        runVectorAccuracyTest(VectorMath::tanh, Math::tanh, -5.0, 5.0);
    }

    @Test
    public void testDirectInverseHyperbolic() {
        // Standard Java equivalent for validation
        runVectorAccuracyTest(VectorMath::asinh, x -> Math.log(x + Math.sqrt(x*x + 1)), -5.0, 5.0);
        runVectorAccuracyTest(VectorMath::acosh, x -> Math.log(x + Math.sqrt(x*x - 1)), 1.1, 10.0);
        runVectorAccuracyTest(VectorMath::atanh, x -> 0.5 * Math.log((1+x)/(1-x)), -0.99, 0.99);
        
        runVectorAccuracyTest(VectorMath::asech, x -> Math.log((1.0/x) + Math.sqrt((1.0/(x*x)) - 1)), 0.1, 0.99);
        runVectorAccuracyTest(VectorMath::acsch, x -> Math.log((1.0/x) + Math.sqrt((1.0/(x*x)) + 1)), 1.1, 10.0);
        runVectorAccuracyTest(VectorMath::acoth, x -> 0.5 * Math.log((1+(1.0/x))/(1-(1.0/x))), 1.1, 10.0);
    }

    @Test
    public void testDirectExponentialAndLogarithmic() {
        runVectorAccuracyTest(VectorMath::exp, Math::exp, -10.0, 10.0);
        runVectorAccuracyTest(VectorMath::ln, Math::log, 0.1, 100.0);
        runVectorAccuracyTest(VectorMath::log10, Math::log10, 0.1, 100.0);
    }

    @Test
    public void testPowerOperations() {
        int n = 265;
        double[] base = new double[n];
        double[] dest = new double[n];
        for (int i = 0; i < n; i++) base[i] = 1.5 + (i * 0.1);

        // Test Uniform exponent = 3.0
        VectorMath.evaluateUniformExponent(base, 0, 3.0, dest, 0, n);
        for (int i = 0; i < n; i++) {
            assertEquals(Math.pow(base[i], 3.0), dest[i], EPSILON, "Uniform power drift");
        }

        // Test Variable Exponent
        double[] exps = new double[n];
        for (int i = 0; i < n; i++) exps[i] = 0.5 + (i * 0.01);
        VectorMath.evaluateVariableExponent(base, 0, exps, 0, dest, 0, n);
        
        for (int i = 0; i < n; i++) {
            assertEquals(Math.pow(base[i], exps[i]), dest[i], EPSILON, "Variable power drift");
        }
    }

    @Test
    public void testIf3Conditional() {
        int n = 265;
        double[] s = new double[4 * n]; // Needs space for cond, trueVal, falseVal, res
        
        int condBase = n;
        int trueBase = 2 * n;
        int falseBase = 3 * n;
        int resBase = 0;

        for (int i = 0; i < n; i++) {
            s[condBase + i] = (i % 2 == 0) ? 1.0 : 0.0; // condition
            s[trueBase + i] = 100.0; // True val
            s[falseBase + i] = -100.0; // False val
        }

        VectorMath.if3(0, n, s, n);

        for (int i = 0; i < n; i++) {
            double expected = (i % 2 == 0) ? 100.0 : -100.0;
            assertEquals(expected, s[resBase + i], 0.0, "if3 branch divergence");
        }
    }
}
