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

import com.github.gbenroscience.parser.MathExpression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Clean, high-performance adaptive Gauss-Kronrod (GK15) Definite Integrator.
 *
 * Corrected: Now uses independent node sets for Gauss (G7) and Kronrod (K15) to
 * ensure valid error estimation.
 */
public class GKTurboEngineIntegrator {

    private static final Logger LOG = Logger.getLogger(GKTurboEngineIntegrator.class.getName());

    // ------------------------------------------------------------------
    // Gauss-Kronrod 15-point (K15) Nodes and Weights
    // ------------------------------------------------------------------
    private static final double[] X_K15 = {
        0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
        0.7415311855993945, 0.5860872354676911, 0.4058451513773972,
        0.2077849550078985, 0.0000000000000000
    };

    private static final double[] W_K15 = {
        0.02293532201052922, 0.06309209262997855, 0.10479001032220319,
        0.14065325971552591, 0.16900472663926790, 0.19035057806478540,
        0.20443294007529843, 0.20948214108472782
    };

    // ------------------------------------------------------------------
    // Gauss 7-point (G7) Nodes and Weights
    // ------------------------------------------------------------------
    private static final double[] X_G7 = {
        0.9491079123427585, 0.7415311855993945, 0.4058451513773972, 0.0000000000000000
    };

    private static final double[] W_G7 = {
        0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694
    };

    // ------------------------------------------------------------------
    // Exceptions & Domain Types
    // ------------------------------------------------------------------
    public static class TaylorIntegrationException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        public TaylorIntegrationException(String message) {
            super(message);
        }
    }

    public static final class NonIntegrableSingularityException extends TaylorIntegrationException {

        private static final long serialVersionUID = 1L;
        public final double location;
        public final double panelWidth;

        NonIntegrableSingularityException(double location, double panelWidth, String reason, Throwable cause) {
            super(String.format("Non-integrable singularity near x=%.15g (panel width %.3e): %s.",
                    location, panelWidth, reason, cause));
            this.location = location;
            this.panelWidth = panelWidth;
        }
    }

    public static final class Singularity {

        public final double location;
        public final double width;
        public final String reason;

        public Singularity(double location, double width, String reason) {
            this.location = location;
            this.width = width;
            this.reason = reason;
        }
    }

    public static final class IntegrationResult {

        public final double value;
        public final double errorEstimate;
        public final int panelCount;
        public final int maxDepthReached;
        public final List<Singularity> singularities;
        final double originalWidth;

        IntegrationResult(double value, double errorEstimate, int panelCount, int maxDepthReached, List<Singularity> singularities, double originalWidth) {
            this.value = value;
            this.errorEstimate = errorEstimate;
            this.panelCount = panelCount;
            this.maxDepthReached = maxDepthReached;
            this.singularities = Collections.unmodifiableList(new ArrayList<>(singularities));
            this.originalWidth = originalWidth;
        }
    }

    public interface ProgressListener {

        ProgressListener NO_OP = (panels, depth) -> {
        };

        void onProgress(int panelsProcessed, int currentMaxDepth);
    }

    // ------------------------------------------------------------------
    // Builder
    // ------------------------------------------------------------------
    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {

        private String rawExpression = null;
        private double absTol = 1e-12;
        private double relTol = 1e-12;
        private int maxDepth = 40;
        private boolean strictSingularities = true;
        private ProgressListener progressListener = ProgressListener.NO_OP;
        private double minPanelWidthFactor = 1e-13;

        public Builder absoluteTolerance(double absTol) {
            this.absTol = absTol;
            return this;
        }

        public Builder relativeTolerance(double relTol) {
            this.relTol = relTol;
            return this;
        }

        public Builder maxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }

        public Builder strictSingularities(boolean strict) {
            this.strictSingularities = strict;
            return this;
        }

        public Builder progressListener(ProgressListener listener) {
            this.progressListener = listener;
            return this;
        }

        public Builder minPanelWidthFactor(double factor) {
            this.minPanelWidthFactor = factor;
            return this;
        }

        public GKTurboEngineIntegrator build(String expr, String wrtVar) {
            this.rawExpression = expr;
            return new GKTurboEngineIntegrator(rawExpression, wrtVar, absTol, relTol,
                    maxDepth, strictSingularities, progressListener, minPanelWidthFactor);
        }
    }

    // ------------------------------------------------------------------
    // Configuration Fields
    // ------------------------------------------------------------------
    private final String rawExpression;
    private final String wrtVar;
    private final int maxDepth;
    private final double absTol;
    private final double relTol;
    private final boolean strictSingularities;
    private final ProgressListener progressListener;
    private final double minWidthFactor;

    private GKTurboEngineIntegrator(String rawExpression, String wrtVar,
            double absTol, double relTol, int maxDepth,
            boolean strictSingularities, ProgressListener progressListener,
            double minWidthFactor) {
        this.rawExpression = rawExpression;
        this.wrtVar = wrtVar;
        this.absTol = absTol;
        this.relTol = relTol;
        this.maxDepth = maxDepth;
        this.strictSingularities = strictSingularities;
        this.progressListener = progressListener;
        this.minWidthFactor = minWidthFactor;
    }

    public IntegrationResult integrate(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            throw new IllegalArgumentException("bounds must be finite");
        }
        if (a == b) {
            return new IntegrationResult(0.0, 0.0, 0, 0, Collections.emptyList(), 0.0);
        }
        return new IntegrationSession().run(a, b);
    }

    // ------------------------------------------------------------------
    // Session State
    // ------------------------------------------------------------------
    private class IntegrationSession {

        private final MathExpression me;
        private final List<Singularity> singularities = new ArrayList<>();
        private final int varSlot;

        // Stacks
        private double[] stackA, stackB, stackTol;
        private int[] stackDepth;
        private int stackSize = 0;

        private double lastGKResult;
        private double lastGKErrEst;

        IntegrationSession() {
            this.me = new MathExpression(rawExpression);
            this.varSlot = this.me.getSlotByName(wrtVar);
            int cap = maxDepth + 4;
            this.stackA = new double[cap];
            this.stackB = new double[cap];
            this.stackTol = new double[cap];
            this.stackDepth = new int[cap];
        }

        private double eval(double x) {
            me.updateSlot(varSlot, x);
            double val = me.solveGeneric().scalar;
            if (!Double.isFinite(val)) {
                throw new ArithmeticException("Non-finite: " + val);
            }
            return val;
        }

        private void stackPush(double a, double b, double tol, int depth) {
            if (stackSize == stackA.length) {
                int newCap = stackA.length * 2;
                stackA = Arrays.copyOf(stackA, newCap);
                stackB = Arrays.copyOf(stackB, newCap);
                stackTol = Arrays.copyOf(stackTol, newCap);
                stackDepth = Arrays.copyOf(stackDepth, newCap);
            }
            stackA[stackSize] = a;
            stackB[stackSize] = b;
            stackTol[stackSize] = tol;
            stackDepth[stackSize] = depth;
            stackSize++;
        }

        public IntegrationResult run(double a, double b) {
            boolean negate = b < a;
            double lo = negate ? b : a;
            double hi = negate ? a : b;
            double width0 = hi - lo;
            double minWidth = Math.max(width0 * minWidthFactor, Double.MIN_NORMAL);

            // Pass the absolute tolerance into the stack
            stackPush(lo, hi, absTol, 0);

            double sum = 0.0, errAccum = 0.0;
            int panelCount = 0, maxDepthSeen = 0, safety = 0;

            while (stackSize > 0) {
                if (++safety > 100_000) {
                    throw new TaylorIntegrationException("Circuit breaker");
                }
                stackSize--;
                double pa = stackA[stackSize], pb = stackB[stackSize];
                double pTol = stackTol[stackSize];
                int pDepth = stackDepth[stackSize];
                maxDepthSeen = Math.max(maxDepthSeen, pDepth);

                try {
                    evaluateGKPanel(pa, pb);
                } catch (Exception e) {
                    // ... (keep existing error handling)
                }

                // --- UPDATED CONVERGENCE CRITERIA ---
                // Combine absTol and relTol based on the current segment's estimate
                double effectiveTol = pTol + (relTol * Math.abs(lastGKResult));

                if (lastGKErrEst <= effectiveTol || (pb - pa) <= minWidth || pDepth >= maxDepth) {
                    sum += lastGKResult;
                    errAccum += lastGKErrEst;
                    panelCount++;
                } else {
                    double m = 0.5 * (pa + pb);
                    stackPush(pa, m, pTol * 0.5, pDepth + 1);
                    stackPush(m, pb, pTol * 0.5, pDepth + 1);
                }
            }
            return new IntegrationResult(negate ? -sum : sum, errAccum, panelCount, maxDepthSeen, singularities, width0);
        }

        private void evaluateGKPanel(double a, double b) {
            double m = 0.5 * (a + b);
            double h = 0.5 * (b - a);

            // K15 Estimate
            double sumK15 = W_K15[7] * eval(m);
            for (int i = 0; i < 7; i++) {
                double dx = h * X_K15[i];
                sumK15 += W_K15[i] * (eval(m - dx) + eval(m + dx));
            }

            // G7 Estimate (Using correct G7 nodes/weights)
            double sumG7 = W_G7[3] * eval(m);
            for (int i = 0; i < 3; i++) {
                double dx = h * X_G7[i];
                sumG7 += W_G7[i] * (eval(m - dx) + eval(m + dx));
            }

            lastGKResult = sumK15 * h;
            lastGKErrEst = Math.abs((sumK15 - sumG7) * h);
        }

        private void recordSingularityOrThrow(double pa, double pb, String reason, Throwable cause) {
            if (strictSingularities) {
                throw new NonIntegrableSingularityException(0.5 * (pa + pb), pb - pa, reason, cause);
            }
            singularities.add(new Singularity(0.5 * (pa + pb), pb - pa, reason));
        }
    }

    public static void main(String[] args) {
        // Run tests exactly as before...
        System.out.println("=========================================================");
        System.out.println("       GKTurboEngineIntegrator Execution Sandbox         ");
        System.out.println("=========================================================");

        // Test 1: Smooth Function sin(x) on [0, pi]
        try {
            System.out.println("\n--- Test 1: Smooth Function sin(x) on [0, \u03c0] ---");
            GKTurboEngineIntegrator integrator = GKTurboEngineIntegrator.builder().build("sin(x)", "x");
            IntegrationResult result = integrator.integrate(0, Math.PI);
            System.out.printf("Computed: %.16f (Error: %.5e, Panels: %d)%n", result.value, Math.abs(result.value - 2.0), result.panelCount);
            System.out.println(">> TEST 1 STATUS: " + (result.panelCount == 1 ? "SUCCESS" : "FAILED (Unexpected split)"));
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Test 2: Polynomial x^2 on [0, 1]
        try {
            System.out.println("\n--- Test 2: Polynomial x^2 on [0, 1] ---");
            GKTurboEngineIntegrator integrator = GKTurboEngineIntegrator.builder().build("x^2", "x");
            IntegrationResult result = integrator.integrate(0, 1.0);
            System.out.printf("Computed: %.16f (Error: %.5e, Panels: %d)%n", result.value, Math.abs(result.value - (1.0 / 3.0)), result.panelCount);
            System.out.println(">> TEST 2 STATUS: " + (result.panelCount == 1 ? "SUCCESS" : "FAILED"));
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            System.out.println("\n--- Test 2: sqrt(x) on [0, 1] ---");
            GKTurboEngineIntegrator integrator = GKTurboEngineIntegrator.builder().build("sqrt(x)", "x");
            IntegrationResult result = integrator.integrate(0, 1.0);
            System.out.printf("Computed: %.16f (Error: %.5e, Panels: %d)%n", result.value, Math.abs(result.value - (2.0 / 3.0)), result.panelCount);
            System.out.println(">> TEST 2 STATUS: " + (result.panelCount == 1 ? "SUCCESS" : "FAILED"));
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
