/*
 * Copyright 2026 oluwagbemirojiboye.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://apache.org
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

public class TaylorGKTurboIntegrator {

    private static final Logger LOG = Logger.getLogger(
        TaylorGKTurboIntegrator.class.getName()
    );

    private final String rawExpression;
    private final double absTol;
    private final double relTol;
    private final int maxDepth;
    private final int adCheckOrder;
    private final ProgressListener progressListener;
    private final GKTurboEngineIntegrator baseGK;

    public interface ProgressListener {
        ProgressListener NO_OP = (panels, depth, value) -> {};
        void onProgress(
            int panelsProcessed, 
            int currentMaxDepth, 
            double currentEstimate
        );
    }

    public static Builder builder() { 
        return new Builder(); 
    }

    public static final class Builder {
        private String rawExpression;
        private double absTol = 1e-12;
        private double relTol = 1e-12;
        private int maxDepth = 40;
        private int adCheckOrder = 3; 
        private ProgressListener progressListener = ProgressListener.NO_OP;

        public Builder expression(String expr) { 
            this.rawExpression = expr; 
            return this; 
        }
        public Builder absoluteTolerance(double tol) { 
            this.absTol = tol; 
            return this; 
        }
        public Builder relativeTolerance(double tol) { 
            this.relTol = tol; 
            return this; 
        }
        public Builder maxDepth(int depth) { 
            this.maxDepth = depth; 
            return this; 
        }
        public Builder adCheckOrder(int order) { 
            this.adCheckOrder = Math.max(2, order); 
            return this; 
        } 
        public Builder progressListener(ProgressListener listener) {
            this.progressListener = listener != null ? 
                listener : ProgressListener.NO_OP;
            return this;
        }
        public TaylorGKTurboIntegrator build() {
            return new TaylorGKTurboIntegrator(
                rawExpression, absTol, relTol, maxDepth,
                adCheckOrder, progressListener
            );
        }
    }

    private TaylorGKTurboIntegrator(
        String rawExpression, double absTol, double relTol, 
        int maxDepth, int adCheckOrder, ProgressListener progressListener
    ) {
        this.rawExpression = rawExpression;
        this.absTol = absTol;
        this.relTol = relTol;
        this.maxDepth = maxDepth;
        this.adCheckOrder = adCheckOrder;
        this.progressListener = progressListener;
        
        this.baseGK = GKTurboEngineIntegrator.builder()
                .absoluteTolerance(absTol)
                .relativeTolerance(relTol)
                .maxDepth(maxDepth)
                .progressListener((p, d) -> 
                    progressListener.onProgress(p, d, 0.0))
                .build(rawExpression, "x");
    }
    public IntegrationResult integrate(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            throw new IllegalArgumentException("Bounds must be finite");
        }
        if (a == b) {
            return new IntegrationResult(
                0.0, 0.0, 0, 0, Collections.emptyList(), 0.0
            );
        }

        double limitCheck = Math.abs(a);
        if (limitCheck <= 1e-3) { 
            double singN = 0.0;
            double singC = 0.0;
            boolean hasIntegrableSingularity = false;
            boolean hasNonIntegrableSingularity = false;

            try {
                MathExpression me = new MathExpression(rawExpression);
                int slot = me.getSlotByName("x");
                AutoDiffNEvaluator ad = new AutoDiffNEvaluator(
                    me, adCheckOrder
                );
                
                double sNear = 1e-7;
                double sFar = 1e-5;
                String v = me.getVariablesNames()[0];
                me.updateSlot(slot, sNear);
                double[] jetNear = ad.evaluateRPN(v, sNear, adCheckOrder);
                
                me.updateSlot(slot, sFar);
                double[] jetFar = ad.evaluateRPN(v, sFar, adCheckOrder);

                if (jetNear != null && jetNear.length >= 2 && 
                    jetFar != null && jetFar.length >= 2) {
                    
                    double fNear = jetNear[0];
                    double fpNear = jetNear[1];
                    double fFar = jetFar[0];
                    double fpFar = jetFar[1];
                    
                    double nNear = (sNear * fpNear) / fNear;
                    double nFar = (sFar * fpFar) / fFar;

                    if (nNear < -0.01 && Math.abs(nNear - nFar) < 0.15) {
                        double roundN = Math.round(nNear * 6.0) / 6.0;
                        if (Math.abs(nNear - (-0.5)) < 0.05) {
                            roundN = -0.5;
                        }
                        if (Math.abs(nNear - (-1.0)) < 0.05) {
                            roundN = -1.0;
                        }
                        
                        if (roundN <= -1.0) {
                            hasNonIntegrableSingularity = true;
                        } else {
                            singN = roundN;
                            singC = fNear / Math.pow(sNear, singN);
                            hasIntegrableSingularity = true;
                        }
                    }
                }
            } catch (Exception ignored) {}

            if (hasNonIntegrableSingularity && a == 0.0) {
                throw new ArithmeticException(
                    "Non-integrable singularity at boundary."
                );
            }
            
            if (hasIntegrableSingularity) {
                double startPoint = Math.max(a, 0.0);
                double analyticalIntegral = (singC / (singN + 1.0)) * 
                    (Math.pow(b, singN + 1.0) - 
                     Math.pow(startPoint, singN + 1.0));
                
                String smoothedExpr = String.format(
                    "(%s) - ((%.16f)*x^(%.16f))", 
                    rawExpression, singC, singN
                );
                
                GKTurboEngineIntegrator smoothGK = 
                    GKTurboEngineIntegrator.builder()
                        .absoluteTolerance(absTol)
                        .relativeTolerance(relTol)
                        .maxDepth(maxDepth)
                        .progressListener((p, d) -> 
                            progressListener.onProgress(p, d, 0.0))
                        .build(smoothedExpr, "x");

                GKTurboEngineIntegrator.IntegrationResult smoothGkResult = 
                    smoothGK.integrate(a, b);
                double finalValue = smoothGkResult.value + 
                    analyticalIntegral;

                List<Singularity> sings = new ArrayList<>();
                sings.add(new Singularity(
                    0.0, 1e-5, 
                    String.format("Aligned Pole (n=%.2f)", singN)
                ));
                
                int baseSize = smoothGkResult.singularities.size();
                for (int i = 0; i < baseSize; i++) {
                    GKTurboEngineIntegrator.Singularity s = 
                        smoothGkResult.singularities.get(i);
                    sings.add(new Singularity(s.location, s.width, s.reason));
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

        GKTurboEngineIntegrator.IntegrationResult gk = baseGK.integrate(a, b);
        List<Singularity> sings = new ArrayList<>(gk.singularities.size());
        for (int i = 0; i < gk.singularities.size(); i++) {
            GKTurboEngineIntegrator.Singularity s = gk.singularities.get(i);
            sings.add(new Singularity(s.location, s.width, s.reason));
        }

        return new IntegrationResult(
            gk.value, gk.errorEstimate, gk.panelCount, 
            gk.maxDepthReached, sings, gk.originalWidth
        );
    }

    public static class IntegrationResult {
        public final double value;
        public final double errorEstimate;
        public final int panelCount;
        public final int maxDepthReached;
        public final List<Singularity> singularities;
        private final double originalWidth;

        public IntegrationResult(
            double value, double errorEstimate, int panelCount, 
            int maxDepthReached, List<Singularity> singularities, 
            double originalWidth
        ) {
            this.value = value;
            this.errorEstimate = errorEstimate;
            this.panelCount = panelCount;
            this.maxDepthReached = maxDepthReached;
            this.singularities = (singularities.isEmpty() || 
                singularities.size() == 1) ? singularities : 
                Collections.unmodifiableList(singularities);
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
    
    private static void test(
        String expr, double a, double b, double expected, 
        ProgressListener listener
    ) {
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

            System.out.printf("\n--- Test: %s on [%.4f, %.4f] ---%n", 
                expr, a, b);
            System.out.printf(
                "Computed: %.15f | Expected: %.15f | Error: %.3e%n", 
                r.value, expected, err
            );
            System.out.printf("Panels: %d | Max Depth: %d%n", 
                r.panelCount, r.maxDepthReached);
            System.out.println(err < 1e-10 ? ">>> PASS" : ">>> MARGINAL");
        } catch (Exception e) {
            System.out.println(">>> FAILED " + expr + ": " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        System.out.println("=========================================");
        System.out.println(" TaylorGKTurboIntegrator Test Suite      ");
        System.out.println("=========================================");

        ProgressListener listener = (panels, depth, est) -> {};

        test("sin(x)", 0.0, Math.PI, 2.0, listener);
        test("1/sqrt(x)", 0.0, 1.0, 2.0, listener); 
        test("1/sqrt(x) + cos(x)", 0.0, 1.0, 2.0 + Math.sin(1.0), listener); 
        test("1/(x^(1/3))", 0.0, 1.0, 1.5, listener); 
        test("exp(-x^2)", -3.0, 3.0, 1.772414696519202, listener); 
        test("1/sin(x)", 0.0000001, 1.5, 16.740387290564932014, listener);
        test("ln(x)", 0.001, 1.0, -0.992092244720970, listener); 
        test("1/sqrt(x)", 0.001, 1.0, 1.936754446796540, listener);  
        test("1/(x-0.5)", 0.1, 0.49, -3.6888794541139363, listener);
        test("(1/(x*sin(x)+3*x*cos(x)))", 0.5, 1.8, 0.7356995195194, listener);
        test("sin(1/x)", 1, 10,  2.22213549121864979008, listener);
        test("sin(x^2)", 1, 10, 0.27340259820624224035, listener);
    }
}
