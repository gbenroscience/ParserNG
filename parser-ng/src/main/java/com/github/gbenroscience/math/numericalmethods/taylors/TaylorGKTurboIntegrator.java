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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Highly resilient adaptive Gauss-Kronrod integrator with AD-based singularity subtraction.
 * Pre-computes the analytical power 'n' of local singularities and pre-compiles 
 * smoothed engines to ensure zero allocations and maximum performance on the hot path.
 */
public class TaylorGKTurboIntegrator {

    private static final Logger LOG = Logger.getLogger(TaylorGKTurboIntegrator.class.getName());

    // Core engines initialized exactly once
    private final GKTurboEngineIntegrator baseGK;
    private final GKTurboEngineIntegrator smoothGK;

    // Pre-computed singularity properties
    private final double singN;
    private final double singC;
    private final boolean hasIntegrableSingularity;
    private final boolean hasNonIntegrableSingularity;
    
    // Cached objects to prevent allocation during the integrate loop
    private final Singularity cachedSingularity;

    public interface ProgressListener {
        ProgressListener NO_OP = (panels, depth, value) -> {};
        void onProgress(int panelsProcessed, int currentMaxDepth, double currentEstimate);
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String rawExpression;
        private double absTol = 1e-12;
        private double relTol = 1e-12;
        private int maxDepth = 40;
        private int adCheckOrder = 3; 
        private boolean useADCrossCheck = true;
        private double crossCheckTolFactor = 10.0;
        private ProgressListener progressListener = ProgressListener.NO_OP;

        public Builder expression(String expr) { this.rawExpression = expr; return this; }
        public Builder absoluteTolerance(double tol) { this.absTol = tol; return this; }
        public Builder relativeTolerance(double tol) { this.relTol = tol; return this; }
        public Builder maxDepth(int depth) { this.maxDepth = depth; return this; }
        public Builder adCheckOrder(int order) { this.adCheckOrder = Math.max(2, order); return this; } 
        public Builder useADCrossCheck(boolean use) { this.useADCrossCheck = use; return this; }
        public Builder progressListener(ProgressListener listener) {
            this.progressListener = listener != null ? listener : ProgressListener.NO_OP;
            return this;
        }

        public TaylorGKTurboIntegrator build() {
            return new TaylorGKTurboIntegrator(rawExpression, absTol, relTol, maxDepth,
                    adCheckOrder, useADCrossCheck, crossCheckTolFactor, progressListener);
        }
    }

    private TaylorGKTurboIntegrator(String rawExpression, double absTol, double relTol, int maxDepth,
                                    int adCheckOrder, boolean useADCrossCheck, double crossCheckTolFactor,
                                    ProgressListener progressListener) {
        
        // 1. Build the standard base engine
        this.baseGK = GKTurboEngineIntegrator.builder()
                .absoluteTolerance(absTol)
                .relativeTolerance(relTol)
                .maxDepth(maxDepth)
                .progressListener((panels, depth) -> progressListener.onProgress(panels, depth, 0.0))
                .build(rawExpression, "x");

        // 2. Proactively probe for singularities to pre-compile the smooth engine
        boolean foundIntegrable = false;
        boolean foundNonIntegrable = false;
        double calcN = 0.0;
        double calcC = 0.0;
        GKTurboEngineIntegrator sGk = null;
        Singularity cSing = null;

        try {
            MathExpression me = new MathExpression(rawExpression, true);
            AutoDiffNEvaluator ad = new AutoDiffNEvaluator(me, adCheckOrder);
            double probe = 1e-5;
            double[] jet = ad.evaluateRPN("x", probe, 1);

            if (jet != null && jet.length >= 2 && Math.abs(jet[0]) > 1.0) {
                double f = jet[0];
                double fp = jet[1];
                double n = (probe * fp) / f; // Empirical power

                if (n < -0.05 && n > -0.99) {
                    LOG.info(String.format("Singularity detected during initialization. Power n ≈ %.4f. Pre-compiling smoothed engine.", n));
                    calcN = n;
                    calcC = f / Math.pow(probe, n);
                    
                    String smoothedExpr = String.format("(%s) - ((%s)*x^(%s))", rawExpression, calcC, calcN);
                    sGk = GKTurboEngineIntegrator.builder()
                            .absoluteTolerance(absTol)
                            .relativeTolerance(relTol)
                            .maxDepth(maxDepth)
                            .progressListener((panels, depth) -> progressListener.onProgress(panels, depth, 0.0))
                            .build(smoothedExpr, "x");

                    foundIntegrable = true;
                    cSing = new Singularity(0.0, probe, String.format("Power Singularity (n=%.3f) subtracted analytically", n));
                } else if (n <= -1.0) {
                    LOG.warning(String.format("Non-integrable singularity detected at x=0 (Power n ≈ %.4f). Integral diverges.", n));
                    foundNonIntegrable = true;
                }
            }
        } catch (Exception ignored) {
            // Failsafe: fallback to base engine if evaluation fails at probe
        }

        this.hasIntegrableSingularity = foundIntegrable;
        this.hasNonIntegrableSingularity = foundNonIntegrable;
        this.singN = calcN;
        this.singC = calcC;
        this.smoothGK = sGk;
        this.cachedSingularity = cSing;
    }

    public IntegrationResult integrate(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) throw new IllegalArgumentException("Bounds must be finite");
        if (a == b) return new IntegrationResult(0.0, 0.0, 0, 0, Collections.emptyList(), 0.0);

        // Subtracted fast-path (Zero internal object allocations!)
        if (Math.abs(a) <= 1e-8) {
            if (hasNonIntegrableSingularity) {
                throw new ArithmeticException("Non-integrable singularity (n <= -1). The integral diverges.");
            }
            
            if (hasIntegrableSingularity) {
                double analyticalIntegral = (singC / (singN + 1.0)) * (Math.pow(b, singN + 1.0) - Math.pow(Math.max(a, 0.0), singN + 1.0));
                
                GKTurboEngineIntegrator.IntegrationResult smoothGkResult = smoothGK.integrate(a, b);
                double finalValue = smoothGkResult.value + analyticalIntegral;

                List<Singularity> sings;
                int baseSize = smoothGkResult.singularities.size();
                if (baseSize == 0) {
                    sings = Collections.singletonList(cachedSingularity);
                } else {
                    sings = new ArrayList<>(baseSize + 1);
                    sings.add(cachedSingularity);
                    // Avoid iterator allocation via index loop
                    for (int i = 0; i < baseSize; i++) {
                        GKTurboEngineIntegrator.Singularity s = smoothGkResult.singularities.get(i);
                        sings.add(new Singularity(s.location, s.width, s.reason));
                    }
                }

                return new IntegrationResult(
                        finalValue, 
                        smoothGkResult.errorEstimate, 
                        smoothGkResult.panelCount, 
                        smoothGkResult.maxDepthReached, 
                        sings, 
                        smoothGkResult.originalWidth
                );
            }
        }

        // Standard fallback path
        GKTurboEngineIntegrator.IntegrationResult gk = baseGK.integrate(a, b);
        
        List<Singularity> sings;
        int size = gk.singularities.size();
        if (size == 0) {
            sings = Collections.emptyList();
        } else {
            sings = new ArrayList<>(size);
            // Avoid iterator allocation via index loop
            for (int i = 0; i < size; i++) {
                GKTurboEngineIntegrator.Singularity s = gk.singularities.get(i);
                sings.add(new Singularity(s.location, s.width, s.reason));
            }
        }

        return new IntegrationResult(gk.value, gk.errorEstimate, gk.panelCount, 
                                     gk.maxDepthReached, sings, gk.originalWidth);
    }

    public static class IntegrationResult {
        public final double value;
        public final double errorEstimate;
        public final int panelCount;
        public final int maxDepthReached;
        public final List<Singularity> singularities;
        private final double originalWidth;

        public IntegrationResult(double value, double errorEstimate, int panelCount, int maxDepthReached,
                                 List<Singularity> singularities, double originalWidth) {
            this.value = value;
            this.errorEstimate = errorEstimate;
            this.panelCount = panelCount;
            this.maxDepthReached = maxDepthReached;
            
            // Wrapped inline using unmodifiableList only if it's not already empty/singleton
            this.singularities = (singularities.isEmpty() || singularities.size() == 1) 
                    ? singularities 
                    : Collections.unmodifiableList(singularities);
            this.originalWidth = originalWidth;
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
    
    private static void test(String expr, double a, double b, double expected, ProgressListener listener) {
        try {
            TaylorGKTurboIntegrator integrator = TaylorGKTurboIntegrator.builder()
                    .expression(expr)
                    .absoluteTolerance(1e-12)
                    .relativeTolerance(1e-12)
                    .adCheckOrder(4)
                    .progressListener(listener)
                    .build();

            IntegrationResult r = integrator.integrate(a, b);
            double err = Math.abs(r.value - expected);

            System.out.printf("\n--- Test: %s on [%.4f, %.4f] ---%n", expr, a, b);
            System.out.printf("Computed: %.15f | Expected: %.15f | Error: %.3e%n", r.value, expected, err);
            System.out.printf("Panels: %d | Max Depth: %d%n", r.panelCount, r.maxDepthReached);
            System.out.println(err < 1e-10 ? ">>> PASS" : ">>> MARGINAL");
        } catch (Exception e) {
            System.out.println(">>> FAILED " + expr + ": " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        System.out.println("=========================================================");
        System.out.println(" TaylorGKTurboIntegrator Test Suite (GC-Free Optimized)");
        System.out.println("=========================================================");

        ProgressListener listener = (panels, depth, est) -> {
            if (panels % 10 == 0 && panels > 0) System.out.printf("  Progress → Panels: %d | Depth: %d%n", panels, depth);
        };

        test("sin(x)", 0.0, Math.PI, 2.0, listener);
        test("1/sqrt(x)", 0.0, 1.0, 2.0, listener); 
        test("1/sqrt(x) + cos(x)", 0.0, 1.0, 2.0 + Math.sin(1.0), listener); 
        test("1/(x^(1/3))", 0.0, 1.0, 1.5, listener); 
        test("exp(-x^2)", -3.0, 3.0, 1.772414696519202, listener); 
        
        
        
        
            test("sin(x)", 0, Math.PI, 2.0, listener);
            test("ln(x)", 0.001, 1.0, -0.992, listener);
            test("1/sqrt(x)", 0.001, 1.0, 1.937, listener);
            test("1/(x-0.5)", 0.1, 0.49, -3.6888794541139363, listener);
            test("(1/(x*sin(x)+3*x*cos(x)))", 0.5, 1.8, 0.7356995195194, listener);
    }
}