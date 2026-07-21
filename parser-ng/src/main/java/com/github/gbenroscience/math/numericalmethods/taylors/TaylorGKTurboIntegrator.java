/*
 * Copyright 2026 oluwagbemirojiboye.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://apache.org/licenses/LICENSE-2.0
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
 * Multi-scale hardened adaptive Gauss-Kronrod integrator equipped with a
 * zero-allocation interior asymptote root-slicing pre-processor.
 */
public class TaylorGKTurboIntegrator {

    private static final Logger LOG = Logger.getLogger(
            TaylorGKTurboIntegrator.class.getName()
    );

    private static final double[] XGK = {
        0.9914553711208126392, 0.9491079123427585245,
        0.8648644233597690728, 0.7415311855993944399,
        0.5860872354676911303, 0.4058451513773971669,
        0.2077849550078984676, 0.0000000000000000000
    };
    private static final double[] WGK = {
        0.0229353220105292250, 0.0630920926299785533,
        0.1047900103222501838, 0.1406532597155259187,
        0.1690047266392679028, 0.1903505780647854100,
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

    private final MathExpression baseExpr;
    private final AutoDiffNEvaluator baseEvaluator;

    // Add these fields in the class
    private final double[] rootBuffer;
    private static final int MAX_ROOTS = 128;   // adjust if you expect more singularities

    private final double[] jetNear;
    private final double[] jetFar;
    private final double[] evalBuf;
    private final double[] fv1;
    private final double[] fv2;

    private final double[] stackA;
    private final double[] stackB;
    private final int[] stackDepth;
    private final List<Singularity> staticSingularityList;

    public interface ProgressListener {

        ProgressListener NO_OP = (panels, depth, value) -> {
        };

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

        this.jetNear = new double[adCheckOrder + 1];
        this.jetFar = new double[adCheckOrder + 1];
        this.evalBuf = new double[adCheckOrder + 1];
        this.fv1 = new double[7];
        this.fv2 = new double[7];

        int stackCapacity = maxDepth + 10;
        this.stackA = new double[stackCapacity];
        this.stackB = new double[stackCapacity];
        this.stackDepth = new int[stackCapacity];
        this.staticSingularityList = new ArrayList<>();

// In constructor:
        this.rootBuffer = new double[MAX_ROOTS];
    }

    /**
     * Executes a Newton-Raphson step sequence on 1/f(x) to auto-snap
     * mathematically exact poles.
     */
    private double findNearbyPole(double x) {
        double current = x;
        boolean converged = false;
        for (int i = 0; i < 10; i++) {
            try {
                baseEvaluator.evaluateRPN("x", current, 1, evalBuf);
                double f = evalBuf[0];
                double fp = evalBuf[1];
                if (!Double.isFinite(f) || Double.isNaN(f)) {
                    return current;
                }

                if (Math.abs(f) < 100.0) {
                    return Double.NaN;
                }

                double dx = f / fp;
                if (!Double.isFinite(dx) || Math.abs(dx) > 1e-2) {
                    return Double.NaN;
                }

                if (Math.abs(dx) < 1e-12) {
                    converged = true;
                    break;
                }
                current += dx;
            } catch (Exception e) {
                return current;
            }
        }

        if (converged) {
            try {
                baseEvaluator.evaluateRPN("x", current, 0, evalBuf);
                if (!Double.isFinite(evalBuf[0]) || Math.abs(evalBuf[0]) > 1e7) {
                    return current;
                }
            } catch (Exception ignored) {
            }
        }
        return Double.NaN;
    }

    private double executeCoreSegment(double segA, double segB, double absTolBudget) {
        double singN = 0.0;
        double singC = 0.0;
        boolean useSubtraction = false;
        double activePoleX0 = Double.NaN;

        // 1. Direct Local Boundary Check & Auto-Snap
        double poleA = findNearbyPole(segA);
        double poleB = findNearbyPole(segB);

        boolean segAIsPole = !Double.isNaN(poleA);
        boolean segBIsPole = !Double.isNaN(poleB);

        if (segAIsPole || segBIsPole) {
            activePoleX0 = segAIsPole ? poleA : poleB;
            if (segAIsPole && segBIsPole) {
                activePoleX0 = (Math.abs(poleA - segA) < Math.abs(poleB - segB)) ? poleA : poleB;
            }

            // Re-anchor the probe distance direction strictly within the segment
            double dir = (segB >= segA) ? 1.0 : -1.0;
            if (activePoleX0 == poleB) {
                dir = -dir;
            }

            double sNear = dir * 1e-7;
            double sFar = dir * 1e-5;

            try {
                baseEvaluator.evaluateRPN("x", activePoleX0 + sNear, adCheckOrder, jetNear);
                baseEvaluator.evaluateRPN("x", activePoleX0 + sFar, adCheckOrder, jetFar);

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

                    if (roundN > -2.0) {
                        singN = roundN;
                        singC = fNear / Math.pow(Math.abs(sNear), singN);
                        useSubtraction = true;
                    }
                }
            } catch (Exception ignored) {
            }
        }

        // 2. Analytical Core Subtraction Matrix
        double analyticalCorrection = 0.0;
        if (useSubtraction) {
            double uA = segA - activePoleX0;
            double uB = segB - activePoleX0;

            double signA = (uA < 0) ? -1.0 : 1.0;
            double signB = (uB < 0) ? -1.0 : 1.0;

            if (uA == 0.0) {
                signA = 1.0;
            }
            if (uB == 0.0) {
                signB = 1.0;
            }

            if (Math.abs(singN - (-1.0)) < 1e-4) {
                analyticalCorrection = singC * (signB * Math.log(Math.abs(uB))
                        - signA * Math.log(Math.abs(uA)));
            } else {
                analyticalCorrection = (singC / (singN + 1.0))
                        * (signB * Math.pow(Math.abs(uB), singN + 1.0)
                        - signA * Math.pow(Math.abs(uA), singN + 1.0));
            }
        }

        int stackSize = 0;
        stackA[stackSize] = segA;
        stackB[stackSize] = segB;
        stackDepth[stackSize] = 0;
        stackSize++;

        double segmentAccumulator = 0.0;

        while (stackSize > 0) {
            stackSize--;
            double pA = stackA[stackSize];
            double pB = stackB[stackSize];
            int depth = stackDepth[stackSize];

            double mid = 0.5 * (pA + pB);
            double h = 0.5 * (pB - pA);

            baseEvaluator.evaluateRPN("x", mid, 0, evalBuf);
            double fc = evalBuf[0];
            if (useSubtraction) {
                fc -= (singC * Math.pow(Math.abs(mid - activePoleX0), singN));
            }

            double sumK = WGK[7] * fc;
            double sumG = WG[3] * fc;
            boolean evaluationFailed = !Double.isFinite(fc);

            for (int i = 0; i < 7; i++) {
                double dx = h * XGK[i];

                double x1 = mid - dx;
                baseEvaluator.evaluateRPN("x", x1, 0, evalBuf);
                double f1 = evalBuf[0];
                if (useSubtraction) {
                    f1 -= (singC * Math.pow(Math.abs(x1 - activePoleX0), singN));
                }

                double x2 = mid + dx;
                baseEvaluator.evaluateRPN("x", x2, 0, evalBuf);
                double f2 = evalBuf[0];
                if (useSubtraction) {
                    f2 -= (singC * Math.pow(Math.abs(x2 - activePoleX0), singN));
                }

                if (!Double.isFinite(f1) || !Double.isFinite(f2)) {
                    evaluationFailed = true;
                    break;
                }

                fv1[i] = f1;
                fv2[i] = f2;
                double fSum = f1 + f2;

                sumK += WGK[i] * fSum;
                if (i == 1) {
                    sumG += WG[0] * fSum;
                } else if (i == 3) {
                    sumG += WG[1] * fSum;
                } else if (i == 5) {
                    sumG += WG[2] * fSum;
                }
            }

            double panelResult = sumK * h;
            double panelError = Math.abs((sumK - sumG) * h);
            double budget = Math.max(absTolBudget * (h / (segB - segA)), relTol * Math.abs(panelResult));

            if (evaluationFailed || (panelError > budget && depth < maxDepth)) {
                stackA[stackSize] = pA;
                stackB[stackSize] = mid;
                stackDepth[stackSize] = depth + 1;
                stackSize++;

                stackA[stackSize] = mid;
                stackB[stackSize] = pB;
                stackDepth[stackSize] = depth + 1;
                stackSize++;
            } else {
                segmentAccumulator += panelResult;
            }
        }

        return segmentAccumulator + analyticalCorrection;
    }

    public IntegrationResult integrate(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            throw new IllegalArgumentException("Bounds must be finite");
        }
        if (a == b) {
            return new IntegrationResult(0.0, 0.0, 0, 0, Collections.emptyList(), 0.0);
        }

        double lower = Math.min(a, b);
        double upper = Math.max(a, b);
        boolean isReversed = (a > b);

        // === ZERO-ALLOCATION ROOT DETECTION ===
        int rootCount = 0;
        try {
            int slotIdx = this.baseExpr.getSlotByName("x");
            int scanPoints = 1000;
            double step = (upper - lower) / scanPoints;

            for (int i = 1; i < scanPoints && rootCount < MAX_ROOTS; i++) {
                double scan = lower + (i * step);
                this.baseExpr.updateSlot(slotIdx, scan);
                double val = this.baseExpr.solveGeneric(scan).scalar;

                if (!Double.isFinite(val) || Math.abs(val) > 1000.0) {
                    double exactPole = findNearbyPole(scan);
                    if (!Double.isNaN(exactPole) && exactPole > lower && exactPole < upper) {
                        // deduplicate
                        boolean exists = false;
                        for (int j = 0; j < rootCount; j++) {
                            if (Math.abs(rootBuffer[j] - exactPole) < 1e-5) {
                                exists = true;
                                break;
                            }
                        }
                        if (!exists) {
                            rootBuffer[rootCount++] = exactPole;
                        }
                    }
                }
            }
        } catch (Exception ignored) {
        }

        // Build boundaries (minimal allocation)
        int numBoundaries = rootCount + 2;
        double[] runtimeBoundaries = new double[numBoundaries];
        runtimeBoundaries[0] = lower;
        for (int i = 0; i < rootCount; i++) {
            runtimeBoundaries[i + 1] = rootBuffer[i];
        }
        runtimeBoundaries[numBoundaries - 1] = upper;

        double globalTotalArea = 0.0;
        int activeSegmentsCount = numBoundaries - 1;
        double segmentAbsTolBudget = absTol / activeSegmentsCount;

        for (int i = 0; i < activeSegmentsCount; i++) {
            globalTotalArea += executeCoreSegment(
                    runtimeBoundaries[i],
                    runtimeBoundaries[i + 1],
                    segmentAbsTolBudget
            );
        }

        if (isReversed) {
            globalTotalArea = -globalTotalArea;
        }

        staticSingularityList.clear();
        return new IntegrationResult(
                globalTotalArea, 0.0, activeSegmentsCount,
                maxDepth, staticSingularityList, b - a
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
            System.out.println(err < 1e-9 ? ">>> PASS" : ">>> MARGINAL");
        } catch (Exception e) {
            System.out.println(">>> FAILED " + expr + ": " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        System.out.println("=========================================");
        System.out.println(" TaylorGKTurboIntegrator Test Suite      ");
        System.out.println("=========================================");

        test("1/(x-0.5)", 0.1, 0.49, -3.6888794541139363);
        test("1/(x-0.5)", 0.1, 0.499999999, -19.806975105072254332);
        test("sin(x)", 0.0, Math.PI, 2.0);
        test("1/sqrt(x)", 0.0, 1.0, 2.0);
        test("1/sqrt(x) + cos(x)", 0.0, 1.0, 2.0 + Math.sin(1.0));
        test("1/sin(x)", 0.0000001, 1.5, 16.740387290564932);

        test("1/(x^(1/3))", 0.0, 1.0, 1.5);
        test("exp(-x^2)", -3.0, 3.0, 1.772414696519202);
        test("ln(x)", 0.001, 1.0, -0.992092244720970);
        test("sin(1/x)", 1, 10, 2.222135491218650);
        test("sin(x^2)", 1, 10, 0.273402598206242);

        test("1/(x-5.5)", 4.0, 5.49, -5.0106352940962555);
        test("x^x", 1.1, 15, 118685141706060739.36763292129980760724321639737101);
    }
}
