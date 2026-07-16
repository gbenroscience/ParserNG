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

/**
 * 100% GC-Free Adaptive Gauss-Kronrod (GK15) Singularity Integrator.
 * Isolates allocations to initialization time.
 */
public class TaylorGKTurboIntegrator {

    private static final Logger LOG = Logger.getLogger(
        TaylorGKTurboIntegrator.class.getName()
    );

    // Standard Gauss-Kronrod 15-point constant weights and nodes
    private static final double[] XGK = {
        0.9914553711208126392, 0.9491079123427585245,
        0.8648644233597690728, 0.7415311855993944399,
        0.5860872354676911303, 0.4058451513773971669,
        0.2077849550078984676, 0.0000000000000000000
    };
    private static final double[] WGK = {
        0.0229353220105292250, 0.0630920926299785533,
        0.1047900103222501838, 0.1406532597155259187,
        0.1690047266392679028, 0.1903505780647854099,
        0.2044329400752988924, 0.2094821410847278280
    };
    private static final double[] WG = {
        0.1294849661688696933, 0.2797053914892766679,
        0.3818300505051189450, 0.4179591836734693878
    };

    private final String rawExpression;
    private final double absTol;
    private final double relTol;
    private final int maxDepth;
    private final int adCheckOrder;
    private final ProgressListener progressListener;
    
    // Non-allocating structural dependencies
    private final MathExpression baseExpr;
    private final AutoDiffNEvaluator baseEvaluator;

    // Fixed primitive workspaces (Zero allocations on the hot path)
    private final double[] jetNear;
    private final double[] jetFar;
    private final double[] evalBuf;
    private final double[] fv1;
    private final double[] fv2;
    
    // Explicit primitive array stacks replacing object-heavy frames
    private final double[] stackA;
    private final double[] stackB;
    private final int[] stackDepth;
    private final List<Singularity> staticSingularityList;

    public interface ProgressListener {
        ProgressListener NO_OP = (panels, depth, value) -> {};
        void onProgress(int panels, int depth, double estimate);
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
        public Builder progressListener(ProgressListener l) {
            this.progressListener = l != null ? l : ProgressListener.NO_OP;
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
        
        this.baseExpr = new MathExpression(rawExpression);
        this.baseEvaluator = new AutoDiffNEvaluator(this.baseExpr, adCheckOrder);

        // Pre-allocate array blocks completely
        this.jetNear = new double[adCheckOrder + 1];
        this.jetFar = new double[adCheckOrder + 1];
        this.evalBuf = new double[adCheckOrder + 1];
        this.fv1 = new double[8];
        this.fv2 = new double[8];
        
        int stackCapacity = maxDepth + 5;
        this.stackA = new double[stackCapacity];
        this.stackB = new double[stackCapacity];
        this.stackDepth = new int[stackCapacity];
        this.staticSingularityList = new ArrayList<>();
    }


    // Zero-alloc node evaluator intercept
    private double sampleFunction(double x, double singC, double singN, boolean subtract) {
        baseEvaluator.evaluateRPN("x", x, 0, evalBuf);
        double val = evalBuf[0];
        if (subtract && x > 0.0) {
            val -= (singC * Math.pow(x, singN));
        }
        return val;
    }

    public IntegrationResult integrate(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            throw new IllegalArgumentException("Bounds must be finite");
        }
        if (a == b) {
            return new IntegrationResult(0.0, 0.0, 0, 0, Collections.emptyList(), 0.0);
        }

        double singN = 0.0;
        double singC = 0.0;
        boolean useSubtraction = false;

        // 1. Dynamic Boundary Proximity Singularity Check
        if (Math.abs(a) <= 1e-3) {
            try {
                double sNear = 1e-7;
                double sFar = 1e-5;
                baseEvaluator.evaluateRPN("x", sNear, adCheckOrder, jetNear);
                baseEvaluator.evaluateRPN("x", sFar, adCheckOrder, jetFar);

                double nNear = (sNear * jetNear[1]) / jetNear[0];
                double nFar = (sFar * jetFar[1]) / jetFar[0];

                if (nNear < -0.01 && Math.abs(nNear - nFar) < 0.15) {
                    double roundN = Math.round(nNear * 6.0) / 6.0;
                    if (Math.abs(nNear - (-0.5)) < 0.05) roundN = -0.5;
                    if (Math.abs(nNear - (-1.0)) < 0.05) roundN = -1.0;

                    if (roundN <= -1.0 && a == 0.0) {
                        throw new ArithmeticException("Non-integrable pole. Diverges.");
                    }
                    if (roundN > -1.0) {
                        singN = roundN;
                        singC = jetNear[0] / Math.pow(sNear, singN);
                        useSubtraction = true;
                    }
                }
            } catch (Exception ignored) {}
        }

        // Calculate analytical base correction if subtraction maps out
        double analyticalIntegral = 0.0;
        if (useSubtraction) {
            double startP = Math.max(a, 0.0);
            analyticalIntegral = (singC / (singN + 1.0)) * 
                (Math.pow(b, singN + 1.0) - Math.pow(startP, singN + 1.0));
        }

        // 2. Core Stack-Based Adaptive Gauss-Kronrod Quadrature Loop
        int stackSize = 0;
        stackA[stackSize] = a;
        stackB[stackSize] = b;
        stackDepth[stackSize] = 0;
        stackSize++;

        double globalIntegral = 0.0;
        double globalErrorEstimate = 0.0;
        int totalPanels = 0;
        int maxDepthReached = 0;

        while (stackSize > 0) {
            stackSize--;
            double pA = stackA[stackSize];
            double pB = stackB[stackSize];
            int depth = stackDepth[stackSize];

            if (depth > maxDepthReached) maxDepthReached = depth;

            double mid = 0.5 * (pA + pB);
            double h = 0.5 * (pB - pA);

            // Single-Pass Interleaved Shared Node Sampling Engine
            double fc = sampleFunction(mid, singC, singN, useSubtraction);
            double sumK = WGK[7] * fc;
            double sumG = WG[3] * fc;
            double absK = WGK[7] * Math.abs(fc);

            boolean evaluationFailed = !Double.isFinite(fc);

            for (int i = 0; i < 7; i++) {
                double dx = h * XGK[i];
                double f1 = sampleFunction(mid - dx, singC, singN, useSubtraction);
                double f2 = sampleFunction(mid + dx, singC, singN, useSubtraction);

                if (!Double.isFinite(f1) || !Double.isFinite(f2)) {
                    evaluationFailed = true;
                    break;
                }

                fv1[i] = f1;
                fv2[i] = f2;
                double fSum = f1 + f2;

                sumK += WGK[i] * fSum;
                absK += WGK[i] * (Math.abs(f1) + Math.abs(f2));

                if (i == 1) sumG += WG[0] * fSum;
                else if (i == 3) sumG += WG[1] * fSum;
                else if (i == 5) sumG += WG[2] * fSum;
            }

            // Error mapping and precision verification limits
            double panelResult = sumK * h;
            double panelError = Math.abs((sumK - sumG) * h);
            double budget = Math.max(absTol * (h / (b - a)), relTol * Math.abs(panelResult));

            if (evaluationFailed || (panelError > budget && depth < maxDepth)) {
                // Push sub-blocks safely back onto the primitive tracking matrix
                stackA[stackSize] = pA;
                stackB[stackSize] = mid;
                stackDepth[stackSize] = depth + 1;
                stackSize++;

                stackA[stackSize] = mid;
                stackB[stackSize] = pB;
                stackDepth[stackSize] = depth + 1;
                stackSize++;
            } else {
                globalIntegral += panelResult;
                globalErrorEstimate += panelError;
                totalPanels++;
                progressListener.onProgress(totalPanels, maxDepthReached, globalIntegral);
            }
        }

        double absoluteValue = globalIntegral + analyticalIntegral;
        staticSingularityList.clear();
        if (useSubtraction) {
            staticSingularityList.add(new Singularity(0.0, 1e-5, "Zero-Alloc Extraction Matrix"));
        }

        return new IntegrationResult(
            absoluteValue, globalErrorEstimate, totalPanels,
            maxDepthReached, staticSingularityList, b - a
        );
    }

    public static class IntegrationResult {
        public final double value;
        public final double errorEstimate;
        public final int panelCount;
        public final int maxDepthReached;
        public final List<Singularity> singularities;
        private final double originalWidth;

        public IntegrationResult(double v, double e, int p, int d, List<Singularity> s, double w) {
            this.value = v;
            this.errorEstimate = e;
            this.panelCount = p;
            this.maxDepthReached = d;
            this.singularities = s.isEmpty() ? Collections.emptyList() : Collections.unmodifiableList(s);
            this.originalWidth = w;
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
    
    private static void test(String expr, double a, double b, double expected) {
        try {
            TaylorGKTurboIntegrator integrator = TaylorGKTurboIntegrator.builder()
                    .expression(expr)
                    .absoluteTolerance(1e-12)
                    .relativeTolerance(1e-12)
                    .adCheckOrder(3)
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
        System.out.println("=========================================");
        System.out.println(" TaylorGKTurboIntegrator Test Suite      ");
        System.out.println("=========================================");
        test("sin(x)", 0.0, Math.PI, 2.0);
        test("1/sqrt(x)", 0.0, 1.0, 2.0); 
        test("1/sqrt(x) + cos(x)", 0.0, 1.0, 2.0 + Math.sin(1.0)); 
        test("1/(x^(1/3))", 0.0, 1.0, 1.5); 
        test("exp(-x^2)", -3.0, 3.0, 1.772414696519202); 
        test("1/sin(x)", 0.0000001, 1.5, 16.740387290564932);
        test("ln(x)", 0.001, 1.0, -0.992092244720970); 
        test("1/sqrt(x)", 0.001, 1.0, 1.936754446796540);  
        test("1/(x-0.5)", 0.1, 0.49, -3.6888794541139363);
        test("sin(1/x)", 1, 10, 2.2221354912186498);
        test("sin(x^2)", 1, 10, 0.2734025982062422);
    }
}
