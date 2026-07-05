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
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator.VectorMath;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 *
 * @author GBEMIRO
 */
public class VectorMathTests2 {

    private static final double EPSILON = 1e-9;
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
        MathExpression me = new MathExpression("sin(x1) * sec(x2) + cot(x3)");
        SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = 
                (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();

        int totalElements = 270;
        int varCount = 3;
        double[] flatInputs = new double[varCount * totalElements];
        double[] outputVector = new double[totalElements];

        for (int i = 0; i < totalElements; i++) {
            flatInputs[(0 * totalElements) + i] = 1.0 + (i * 0.01);
            flatInputs[(1 * totalElements) + i] = 0.5 + (i * 0.01);
            flatInputs[(2 * totalElements) + i] = 1.5 + (i * 0.01);
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

    // ========================================================================
    // PART 2: COMPREHENSIVE VECTOR MATH DIRECT TESTS
    // ========================================================================
    
    @FunctionalInterface
    interface VectorAction { void apply(int base, int n, double[] s); }
    @FunctionalInterface
    interface ScalarAction { double apply(double x); }

    private void runVectorAccuracyTest(VectorAction vAct, ScalarAction sAct, double min, double max, double eps) {
        int n = 265; 
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = min + (max - min) * (i / (double)(n - 1));
        }

        double[] expected = new double[n];
        for (int i = 0; i < n; i++) {
            expected[i] = sAct.apply(s[i]);
        }

        vAct.apply(0, n, s);

        for (int i = 0; i < n; i++) {
            assertEquals(expected[i], s[i], eps, "Math drifted at index " + i + " Input: " + (min + (max - min) * (i / (double)(n - 1))));
        }
    }

    @Test
    public void testDirectTrigonometry() {
        runVectorAccuracyTest(VectorMath::sin, Math::sin, -Math.PI, Math.PI, 1e-9);
        runVectorAccuracyTest(VectorMath::cos, Math::cos, -Math.PI, Math.PI, 1e-9);
        runVectorAccuracyTest(VectorMath::tan, Math::tan, -1.0, 1.0, 1e-9);
    }

    @Test
    public void testDirectInverseTrigonometry() {
        runVectorAccuracyTest(VectorMath::asin, Math::asin, -0.9, 0.9, 1e-8);
        runVectorAccuracyTest(VectorMath::acos, Math::acos, -0.9, 0.9, 1e-8);
        runVectorAccuracyTest(VectorMath::atan, Math::atan, -5.0, 5.0, 1e-8);
    }

    @Test
    public void testDirectHyperbolic() {
        // sinh and tanh are odd functions. Check sign handling carefully.
        runVectorAccuracyTest(VectorMath::sinh, Math::sinh, -2.0, 2.0, 1e-8);
        runVectorAccuracyTest(VectorMath::cosh, Math::cosh, -2.0, 2.0, 1e-8);
        runVectorAccuracyTest(VectorMath::tanh, Math::tanh, -2.0, 2.0, 1e-8);
    }

    @Test
    public void testDirectInverseHyperbolic() {
        // Adjusted ranges to stay well away from singularities (e.g., acosh(1) -> undefined)
        runVectorAccuracyTest(VectorMath::asinh, x -> Math.log(x + Math.sqrt(x*x + 1)), -2.0, 2.0, 1e-8);
        runVectorAccuracyTest(VectorMath::acosh, x -> Math.log(x + Math.sqrt(x*x - 1)), 1.5, 5.0, 1e-8);
        runVectorAccuracyTest(VectorMath::atanh, x -> 0.5 * Math.log((1+x)/(1-x)), -0.5, 0.5, 1e-8);
    }

    @Test
    public void testDirectExponentialAndLogarithmic() {
        runVectorAccuracyTest(VectorMath::exp, Math::exp, -5.0, 5.0, 1e-9);
        runVectorAccuracyTest(VectorMath::ln, Math::log, 1.0, 10.0, 1e-9);
    }
}