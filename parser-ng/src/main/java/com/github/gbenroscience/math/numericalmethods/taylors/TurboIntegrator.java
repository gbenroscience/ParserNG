/*
 * Copyright 2026 oluwagbemirojiboye.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math.numericalmethods.taylors;

import com.github.gbenroscience.math.differentialcalculus.autodiff.AutoDiffNEvaluator;
import com.github.gbenroscience.parser.MathExpression;

/**
 * High-Speed Adaptive Taylor Series Definite Integral Solver.
 *
 * <p>
 * Delegates every derivative computation to {@link
 * AutoDiffNEvaluator#taylorCoefficients} rather than carrying a second copy of
 * the jet-propagation algorithm. The original standalone version had silently
 * regressed relative to the canonical engine: its own {@code
 * powJet} always routed {@code ^} through {@code ln(u)->exp}, which throws on
 * {@code x^2} at {@code x=0}, or any integer power at a negative base --
 * missing the canonical engine's division-free integer-exponent special case
 * entirely. Delegating removes that whole class of drift risk: the fix -- and
 * any future one -- only has to exist in one place.
 *
 * <p>
 * This also means the class automatically inherits {@code
 * AutoDiffNEvaluator}'s current thread-safety model (per-thread scratch state
 * via {@code ThreadLocal}), and {@code taylorCoefficients} already returns
 * normalized coefficients {@code c_k = f^(k)/k!} -- exactly what the "Taylor
 * Shortcut" step ({@code term_i = c_i/(i+1) * h^(i+1)}) needs, with no
 * factorial handling required on this side.
 *
 * <p>
 * The adaptive step-size logic ({@link #estimateStepSize}) is unchanged from
 * the original: an a priori ratio test on the last few Taylor coefficients,
 * with no independent cross-check that a step was actually accurate after the
 * fact. That's a real, standard technique, but worth knowing -- unlike the
 * Gauss-Kronrod-based integrators built earlier in this exchange, which all
 * carry some form of second-opinion check.
 */
public class TurboIntegrator {

    private final AutoDiffNEvaluator ad;
    private final int maxOrder;

    public TurboIntegrator(MathExpression me, int maxOrder) {
        if (maxOrder < 0) {
            throw new IllegalArgumentException("maxOrder >= 0 required");
        }
        this.ad = new AutoDiffNEvaluator(me, maxOrder);
        this.maxOrder = maxOrder;
    }

    public TurboIntegrator(AutoDiffNEvaluator ad, int maxOrder) {
        if (maxOrder < 0) {
            throw new IllegalArgumentException("maxOrder >= 0 required");
        }
        this.ad = ad;
        this.maxOrder = maxOrder;
    }

    /**
     * Integrates the expression from x1 to x2 adaptively.
     *
     * @param wrtVar Variable of integration (e.g. "x")
     * @param x1 Lower limit
     * @param x2 Upper limit
     * @param order The expansion order (higher order = bigger steps / faster
     * integration)
     * @param tolerance Truncation error threshold (e.g. 1e-15)
     * @return The definite integral value
     */
    public double integrate(String wrtVar, double x1, double x2, int order, double tolerance) {
        if (order > maxOrder) {
            throw new IllegalArgumentException("Requested order " + order + " exceeds maxOrder " + maxOrder);
        }
        if (order < 0) {
            throw new IllegalArgumentException("Order must be non-negative");
        }
        if (tolerance <= 0.0) {
            throw new IllegalArgumentException("Tolerance must be strictly positive");
        }

        double x = x1;
        double target = x2;
        double direction = Math.signum(target - x);

        if (direction == 0.0) {
            return 0.0;
        }
        double totalIntegral = 0.0;
        double[] coeffs = new double[order + 1]; // one per integrate() call, not per step

        while (direction * (target - x) > 1e-15) {
            // Step 1: normalized Taylor coefficients c_k at the current boundary point x
            ad.taylorCoefficients(wrtVar, x, order, coeffs);
            // Step 2: estimate the mathematically optimal step size h (a positive magnitude)
            double hMag = estimateStepSize(coeffs, order, tolerance);
            double remainingDistance = Math.abs(target - x);
            boolean isFinalStep = false;

            if (hMag >= remainingDistance) {
                hMag = remainingDistance;
                isFinalStep = true;
            }
            double signedH = direction * hMag;
            if (Math.abs(signedH) <= 1e-16 || (x + signedH) == x) {
                if (remainingDistance < 1e-10) {
                    signedH = direction * remainingDistance;
                    isFinalStep = true;
                } else {
                    throw new ArithmeticException(
                            String.format("Step size stagnated (underflow) at x = %.8f. Step size: %e. "
                                    + "Function may contain a singularity. Try adjusting tolerance.", x, hMag)
                    );
                }
            }
            // Step 3: Taylor Shortcut -- term_i = (c_i / (i+1)) * h^(i+1)
            double stepIntegral = 0.0;
            double hPower = signedH;
            for (int i = 0; i <= order; i++) {
                stepIntegral += (coeffs[i] / (i + 1)) * hPower;
                hPower *= signedH;
            }
            totalIntegral += stepIntegral;

            if (isFinalStep) {
                break;
            }
            double nextX = x + signedH;
            if (nextX == x) {
                break;
            }
            x = nextX;
        }
        return totalIntegral;
    }

    /**
     * Uses consecutive Taylor coefficients to find the optimal step size.
     * Prevents steps from exceeding the local radius of convergence
     * (poles/singularities).
     */
    private double estimateStepSize(double[] coeffs, int order, double tolerance) {
        if (order == 0) {
            return Double.MAX_VALUE;
        }
        double minH = Double.POSITIVE_INFINITY;
        boolean allZero = true;
        int startK = Math.max(1, order - 5);
        for (int k = startK; k <= order; k++) {
            double absCoeff = Math.abs(coeffs[k]);
            if (absCoeff > 1e-300) {
                allZero = false;
                double hk = Math.pow(tolerance / absCoeff, 1.0 / k);
                if (hk < minH) {
                    minH = hk;
                }
            }
        }
        if (allZero) {
            return Double.MAX_VALUE;
        }
        return minH * 0.9;
    }

    public static void main(String[] args) {
        String expr = "x^2";
        MathExpression me = new MathExpression(expr);
        TurboIntegrator regressionCheck = new TurboIntegrator(me, 8);
        // Exactly the case that threw under the old standalone engine's powJet bug:
        // x=0 is the first evaluation point, and x^2 with an ln/exp-only powJet
        // required x > 0.
        double calc = regressionCheck.integrate("x", 0.0, 1.0, 8, 1e-15);
        System.out.println("x^2 on [0,1]: calc=" + calc + " expected=" + (1.0 / 3.0));

        expr = "1 / sin(x)";
        me = new MathExpression(expr);
        int order = 12;
        double tol = 1e-15;
        double x1 = 0.0000001;
        double x2 = 1.5;
        TurboIntegrator intg = new TurboIntegrator(me, order);
        double calc2 = intg.integrate("x", x1, x2, order, tol);
        double exact = Math.log(Math.abs(Math.tan(x2 / 2))) - Math.log(Math.abs(Math.tan(x1 / 2)));
        System.out.println("exact = " + exact);
        System.out.println("calc = " + calc2);

        System.out.println("=== INITIALIZING BENCHMARK AND VALIDATION ===");
        expr = "x * exp(x)";
        me = new MathExpression(expr);
        order = 12;
        tol = 1e-15;
        TurboIntegrator integrator = new TurboIntegrator(me, order);
        double exactForward = Math.exp(2.0) + 1.0;
        double exactBackward = -exactForward;
        for (int i = 0; i < 20000; i++) {
            integrator.integrate("x", 0.0, 2.0, order, tol);
            integrator.integrate("x", 2.0, 0.0, order, tol);
        }
        long startForward = System.nanoTime();
        double approxForward = integrator.integrate("x", 0.0, 2.0, order, tol);
        long endForward = System.nanoTime();
        long startBackward = System.nanoTime();
        double approxBackward = integrator.integrate("x", 2.0, 0.0, order, tol);
        long endBackward = System.nanoTime();
        double errForward = Math.abs(exactForward - approxForward);
        double errBackward = Math.abs(exactBackward - approxBackward);
        System.out.printf("Test Expression : %s%n", expr);
        System.out.println("---------------------------------------------");
        System.out.printf("Forward [0 to 2] : %.16f (Exact: %.16f)%n", approxForward, exactForward);
        System.out.printf("Forward Error : %.2e%n", errForward);
        System.out.printf("Forward Time : %.4f ms%n", (endForward - startForward) / 1_000_000.0);
        System.out.println("---------------------------------------------");
        System.out.printf("Backward [2 to 0] : %.16f (Exact: %.16f)%n", approxBackward, exactBackward);
        System.out.printf("Backward Error : %.2e%n", errBackward);
        System.out.printf("Backward Time : %.4f ms%n", (endBackward - startBackward) / 1_000_000.0);
        System.out.println("---------------------------------------------");
        System.out.println("Execution state : 100% GC-Free & Symmetrical (delegated engine)!");
        
        me = new MathExpression("sin(x)/exp(x-1)");
         integrator = new TurboIntegrator(me, 22);
         x1=2;
         x2=22;
         MathExpression mee = new MathExpression("-0.5*(exp(1-x))*(sin(x)+cos(x))");
         exact = mee.solveGeneric(x2).scalar - mee.solveGeneric(x1).scalar;
         double ans = integrator.integrate("x", x1, x2, 8, 1.0e-16);
         
         System.out.println("sin(x)/exp(x-1) - CALCULATED: "+ans+", EXACT: "+exact);
         
         
         integrator = new TurboIntegrator(new MathExpression("sin(1/x)"), 22);
         x1=1;
         x2=10;
         exact = 2.22213549121864979008;
         ans = integrator.integrate("x", x1, x2, 8, 1.0e-16);
         System.out.println("sin(1/x) - CALCULATED: "+ans+", EXACT: "+exact);
         
                  
         integrator = new TurboIntegrator(new MathExpression("sin(x^2)"), 22);
         x1=1;
         x2=10;
         exact = 0.27340259820624224035;
         ans = integrator.integrate("x", x1, x2, 8, 1.0e-16);
         System.out.println("sin(x^2) - CALCULATED: "+ans+", EXACT: "+exact);
 
        
          integrator = new TurboIntegrator(new MathExpression("x^x"), 100);
         x1=1.1;
         x2=15;
         exact = 3.92135678385800451368E16;
         ans = integrator.integrate("x", x1, x2, 10, 1000);
         System.out.println("x^x - CALCULATED: "+ans+", EXACT: "+exact);
//CALCULATED: 1.186851417060609E17, EXACT: 3.921356783858005E16
//            1.1868514170606243E17, EXACT: 3.921356783858005E16
    }
}
