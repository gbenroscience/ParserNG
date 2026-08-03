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
package com.github.gbenroscience.math.differentialcalculus.autodiff;

import com.github.gbenroscience.parser.MathExpression;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Assertion-based regression test for SystemAutoDiffEvaluator. 
 * Cross-checks the AD first derivative against a central finite difference, and
 * verifies closed forms and multi-variable frame-based resolution.
 */
class SystemAutoDiffEvaluatorTest {

    private static final double FD_H = 1e-6;
    private static final double TOLERANCE = 1e-5;

    @Test
    void testAgainstFiniteDifferences() {
        checkAgainstFiniteDifference("x^3+3*x^2-5*x-8", 2.0);
        checkAgainstFiniteDifference("sin(x)", 1.3);
        checkAgainstFiniteDifference("cos(x)", 0.7);
        checkAgainstFiniteDifference("sin(x)-cos(x)", 2.0);
        checkAgainstFiniteDifference("tan(x)", 0.4);
        checkAgainstFiniteDifference("exp(x)", 1.5);
        checkAgainstFiniteDifference("ln(x)", 3.2);
        checkAgainstFiniteDifference("sqrt(x)", 4.0);
        checkAgainstFiniteDifference("x^2*cos(x)-2*x*sin(x)-2*cos(x)", 4.0);
        checkAgainstFiniteDifference("sinh(x)", 0.9);
        checkAgainstFiniteDifference("cosh(x)", 0.9);
        checkAgainstFiniteDifference("tanh(x)", 0.9);
        checkAgainstFiniteDifference("asin(x)", 0.3);
        checkAgainstFiniteDifference("acos(x)", 0.3);
        checkAgainstFiniteDifference("atan(x)", 2.0);
        checkAgainstFiniteDifference("x^x", 2.0);
        checkAgainstFiniteDifference("atan2(2*x,3)", 2.0);
        checkAgainstFiniteDifference("log(x,2)", 8.0);
        checkAgainstFiniteDifference("abs(x)", 3.0);   // away from the kink at 0
        checkAgainstFiniteDifference("abs(x)", -3.0);
    }

    @Test
    void testSinHigherOrder() {
        // Closed-form cross-check: d^k/dx^k sin(x) = sin(x + k*pi/2)
        double x0 = 1.7;
        int maxOrder = 6;
        MathExpression me = new MathExpression("sin(x)");
        SystemAutoDiffEvaluator ad = new SystemAutoDiffEvaluator(me, maxOrder);
        int xIndex = ad.frameIndexOf("x");
        double[] frame = {x0};
        double[] derivatives = ad.evaluateDerivatives(frame, xIndex, maxOrder);

        for (int k = 0; k <= maxOrder; k++) {
            double expected = Math.sin(x0 + k * Math.PI / 2.0);
            assertEquals(expected, derivatives[k], TOLERANCE, 
                "d^" + k + "/dx^" + k + " sin(x) @ x=" + x0);
        }
    }

    @Test
    void testFloorCeilIsFlat() {
        // The floor/ceil regression: derivative must be exactly zero away
        // from the integer kink, at every order — NOT proportional to the
        // argument's own derivative.
        checkFloorCeilIsFlatHelper("floor(x)", 2.5);
        checkFloorCeilIsFlatHelper("ceil(x)", 2.5);
        checkFloorCeilIsFlatHelper("floor(x)", -1.3);
        checkFloorCeilIsFlatHelper("ceil(x)", 7.1);
    }

    @Test
    void testFrameBasedTwoVariable() {
        // f(x, y) = x^2 * y ; df/dx = 2xy, df/dy = x^2 — checked at (x,y) = (3, 5)
        MathExpression me = new MathExpression("x^2*y");
        SystemAutoDiffEvaluator ad = new SystemAutoDiffEvaluator(me, 2);
        int xIndex = ad.frameIndexOf("x");
        int yIndex = ad.frameIndexOf("y");

        int frameSize = Math.max(xIndex, yIndex) + 1;
        double[] frame = new double[frameSize];
        frame[xIndex] = 3.0;
        frame[yIndex] = 5.0;

        double dfdx = ad.taylorCoefficients(frame, xIndex, 1)[1];
        double dfdy = ad.taylorCoefficients(frame, yIndex, 1)[1];

        assertEquals(2 * 3.0 * 5.0, dfdx, TOLERANCE, "d(x^2*y)/dx @ (3,5)");
        assertEquals(3.0 * 3.0, dfdy, TOLERANCE, "d(x^2*y)/dy @ (3,5)");
    }

    @Test
    void testAbsentVariableIsZero() {
        // f(x) = x^2 ; differentiating w.r.t. a name that isn't in the
        // expression at all must be exactly zero, not an error.
        MathExpression me = new MathExpression("x^2");
        SystemAutoDiffEvaluator ad = new SystemAutoDiffEvaluator(me, 2);
        int xIndex = ad.frameIndexOf("x");
        int absentIndex = ad.frameIndexOf("q"); // not present -> NO_WRT_VARIABLE
        
        assertEquals(SystemAutoDiffEvaluator.NO_WRT_VARIABLE, absentIndex, 
            "Expected 'q' to be absent from expression 'x^2'");

        double[] frame = new double[xIndex + 1];
        frame[xIndex] = 4.0;
        double[] result = ad.taylorCoefficients(frame, absentIndex, 2);

        assertEquals(16.0, result[0], TOLERANCE, "x^2 value @ x=4 (wrt absent var)");
        assertEquals(0.0, result[1], TOLERANCE, "d(x^2)/d(absent) @ x=4");
    }

    // --- Helper Methods ---

    private void checkAgainstFiniteDifference(String expr, double x0) {
        MathExpression me = new MathExpression(expr);
        SystemAutoDiffEvaluator ad = new SystemAutoDiffEvaluator(me, 3);
        int xIndex = ad.frameIndexOf("x");
        
        assertNotEquals(SystemAutoDiffEvaluator.NO_WRT_VARIABLE, xIndex, 
            "Expression '" + expr + "' does not reference 'x' as expected");

        double[] frame = new double[xIndex + 1];
        frame[xIndex] = x0;
        double adDerivative = ad.taylorCoefficients(frame, xIndex, 1)[1];

        double[] framePlus = frame.clone();
        framePlus[xIndex] = x0 + FD_H;
        double[] frameMinus = frame.clone();
        frameMinus[xIndex] = x0 - FD_H;
        
        double fPlus = ad.taylorCoefficients(framePlus, SystemAutoDiffEvaluator.NO_WRT_VARIABLE, 0)[0];
        double fMinus = ad.taylorCoefficients(frameMinus, SystemAutoDiffEvaluator.NO_WRT_VARIABLE, 0)[0];
        double fdDerivative = (fPlus - fMinus) / (2.0 * FD_H);

        assertEquals(fdDerivative, adDerivative, 1e-4, 
            expr + " @ x=" + x0 + " failed finite difference check");
    }

    private void checkFloorCeilIsFlatHelper(String expr, double x0) {
        MathExpression me = new MathExpression(expr);
        SystemAutoDiffEvaluator ad = new SystemAutoDiffEvaluator(me, 4);
        int xIndex = ad.frameIndexOf("x");
        double[] frame = {x0};
        double[] derivatives = ad.evaluateDerivatives(frame, xIndex, 4);

        for (int k = 1; k <= 4; k++) {
            assertEquals(0.0, derivatives[k], 1e-12, 
                expr + " order " + k + " @ x=" + x0 + " (must be exactly flat)");
        }
    }
}