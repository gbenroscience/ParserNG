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
package com.github.gbenroscience.math.numericalmethods.taylors.ffx;

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

import com.github.gbenroscience.math.differentialcalculus.autodiff.AutoDiffNEvaluator;
import com.github.gbenroscience.math.numericalmethods.taylors.TurboIntegratorEngine;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Composite adaptive integrator using the 7-15 Gauss-Kronrod pair as the
 * primary panel rule, {@link AutoDiffNEvaluator}-based domain-error
 * classification for genuine singularities, a low-order AD Taylor
 * cross-check that spends one extra derivative evaluation only on panels
 * Gauss-Kronrod already believes it has converged on, and analytic endpoint
 * power-law singularity subtraction.
 *
 * <h2>Endpoint singularity subtraction</h2>
 * At the start of every {@link #integrate(double, double)} call, both
 * bounds are probed for a local power-law behavior
 * {@code f(x) ~= C * s^n}, where {@code s} is distance into the domain from
 * that endpoint. The exponent is estimated two independent ways from a
 * single nearby AD probe -- {@code n1 = s*f'(x)/f(x)} and
 * {@code n2 = 1 + s*f''(x)/f'(x)} (both exact for a pure power term; see
 * derivation below) -- and only trusted if they agree to within {@link
 * Builder#singularityConsistencyTolerance}.
 *
 * <p><b>Derivation.</b> Let {@code s = dir*(x-x0)} where {@code dir} is
 * {@code +1} at the lower bound and {@code -1} at the upper bound, so
 * {@code s} is always positive moving from the endpoint into the domain.
 * If {@code f(x) ~= C*s^n} locally, then {@code df/ds = C*n*s^(n-1) =
 * n*f/s}, and since {@code df/ds = dir*f'(x)}, algebra gives
 * {@code n = s*dir*f'(x)/f(x)}. Differentiating again the same way gives
 * {@code n2 = 1 + s*dir*f''(x)/f'(x)}. Both are exact for a pure power law
 * and independently derived, so their disagreement is a direct, cheap test
 * of whether the local behavior really is a clean power law.
 *
 * <p><b>Non-integrable case.</b> If a confidently-estimated exponent is
 * {@code <= -1} at either bound, the integral does not converge there. This
 * always throws {@link TurboIntegratorEngine.NonIntegrableSingularityException},
 * regardless of {@link Builder#strictSingularities}.
 *
 * <p><b>Cancellation floor (important).</b> Subtraction computes
 * {@code f(x) - C*s^n} by subtracting two numbers of comparable, and near
 * {@code x0} very large, magnitude. If the true remainder is itself close
 * to zero (e.g. pure {@code 1/sqrt(x)}, whose subtracted remainder is
 * exactly 0), this costs almost nothing. But if {@code f} has a genuine
 * O(1) smooth part on top of the singular part (e.g.
 * {@code 1/sqrt(x) + cos(x)}, whose remainder is {@code cos(x)}, not 0),
 * recovering that O(1) remainder from two ~10^6-magnitude numbers caps the
 * achievable absolute precision at roughly
 * {@code machine_epsilon * (subtracted magnitude)} -- a floor that no
 * amount of bisection can push below, since it isn't a convergence problem,
 * it's a representable-precision problem. This class tracks the largest
 * subtracted magnitude seen on each panel and folds
 * {@code EPMACH * 100 * that magnitude} into <em>both</em> the reported
 * error estimate and the panel's acceptance budget consistently (see
 * {@link #computeQK15} and the {@code budget} calculation in {@link
 * #integrate}), so a panel that has hit this floor is accepted -- with an
 * honestly elevated error estimate reflected in {@link
 * #getLastErrorEstimate()} -- rather than bisected forever chasing a target
 * that was never reachable. Earlier revisions floored only the reported
 * error, which made the acceptance test strictly harder to satisfy and
 * caused exactly this: infinite bisection down to the panel-width floor,
 * followed by a strict-mode throw, on any subtracted singularity whose true
 * remainder wasn't itself near zero.
 *
 * <p><b>Scope.</b> This only ever looks at the two integration bounds, and
 * only for power-law ({@code C*s^n}) behavior -- not logarithmic
 * singularities, not singularities strictly interior to {@code (a,b)}.
 *
 * <p><b>Cross-check interaction.</b> When a bound is subtracted, the AD
 * cross-check is disabled for that whole call: it works from the
 * <b>original</b>, still-genuinely-singular {@code f}, so comparing it
 * against a Gauss-Kronrod result computed from the smoothed integrand would
 * compare two different functions.
 *
 * <h2>Why GK as the primary rule</h2>
 * Gauss-Kronrod is exact for polynomials up to degree 23 (K15) using only
 * 15 function evaluations, and its embedded {@code |K15-G7|} comparison
 * (refined the way QUADPACK's {@code qk15} does, via the {@code resasc}
 * scaling below) is a strong, self-contained error estimate that does not
 * require derivatives at all.
 *
 * <h2>Why GK alone is not enough</h2>
 * GK's error estimate is a comparison between two quadrature rules built
 * from the same 15 sample points. If a feature of {@code f} falls between
 * those points, both rules can agree closely with each other while being
 * wrong together -- verified empirically: a Gaussian bump
 * {@code exp(-3200*(x-1.3)^2)} on {@code [0,2]}, integrated by plain
 * GK-adaptive at {@code absTol=1e-10}, returns {@code 0.0} (true value
 * {@code 0.0313}) because the starting panel's reported error was
 * {@code ~5.6e-13}.
 *
 * <h2>The cross-check layer</h2>
 * On any panel where GK's own error estimate is already within budget, this
 * class spends one additional low, fixed-order Taylor-series estimate from
 * {@link AutoDiffNEvaluator#taylorCoefficients}, expanded about the panel
 * midpoint. Disagreement -- in value or in the cross-check's own
 * convergence signal -- forces a bisection anyway, overriding GK's claim.
 * This roughly triples to quadruples panel counts on integrands GK already
 * handles perfectly alone; disable via {@link Builder#crossCheckEnabled}.
 *
 * <h2>Progress reporting</h2>
 * {@link Builder#progressListener(ProgressListener)} is invoked once per
 * panel accepted into the running total, with the accepted-panel count so
 * far, the deepest bisection level reached so far, and the running estimate
 * accumulated so far (excluding the endpoint singularity correction term,
 * which is only known once the whole panel loop completes). Default is a
 * no-op. Not throttled internally.
 *
 * <h2>Domain-error and singularity policy</h2>
 * A genuine AD domain error that persists down to the minimum panel width
 * throws {@link TurboIntegratorEngine.NonIntegrableSingularityException} in
 * strict mode (the default), or is recorded and its contribution omitted in
 * lenient mode.
 *
 * <h2>Constants</h2>
 * The 15-point Kronrod / 7-point Gauss abscissae and weights are the
 * standard QUADPACK {@code dqk15.f} table (Fullerton, Bell Labs, 1981;
 * cross-checked against two independent mirrors of the Netlib source).
 *
 * <h2>Thread-safety</h2>
 * Not thread-safe. Give each thread its own instance built from the same
 * {@code rpnTokens} array (which is only read).
 */
public class Integrator {

    private static final Logger LOG = Logger.getLogger(Integrator.class.getName());

    // ------------------------------------------------------------------
    // Verified QK15 constants (Netlib/QUADPACK dqk15.f; Fullerton, 1981).
    // ------------------------------------------------------------------
    private static final double[] XGK = {
            0.991455371120812639206854697526328516642,
            0.949107912342758524526189684047851262401,
            0.864864423359769072789712788640926201211,
            0.741531185599394439863864773280788407074,
            0.586087235467691130294144838258729598437,
            0.405845151377397166906606412076961463347,
            0.207784955007898467600689403773244913480,
            0.000000000000000000000000000000000000000
    };
    private static final double[] WGK = {
            0.022935322010529224963732008058969592,
            0.063092092629978553290700663189204287,
            0.104790010322250183839876322541518017,
            0.140653259715525918745189590510237920,
            0.169004726639267902826583426598550284,
            0.190350578064785409913256402421013683,
            0.204432940075298892414161999234649085,
            0.209482141084727828012999174891714264
    };
    private static final double[] WG = {
            0.129484966168869693270611432679082018,
            0.279705391489276667901467771423779582,
            0.381830050505118944950369775488975134,
            0.417959183673469387755102040816326531
    };

    private static final double EPMACH = Math.ulp(1.0);
    private static final double UFLOW = Double.MIN_NORMAL;

    public static final int DEFAULT_CROSS_CHECK_ORDER = 8;
    public static final int MAX_CROSS_CHECK_ORDER = 40;
    public static final double DEFAULT_SINGULARITY_CONSISTENCY_TOL = 0.05;
    /** Multiplier on machine epsilon used to build the cancellation-noise floor near a subtracted endpoint. */
    public static final double CANCELLATION_FLOOR_FACTOR = 100.0;

    // ------------------------------------------------------------------
    // Progress reporting
    // ------------------------------------------------------------------

    public interface ProgressListener {
        ProgressListener NO_OP = (panelsAccepted, currentMaxDepth, currentEstimate) -> { };

        void onProgress(int panelsAccepted, int currentMaxDepth, double currentEstimate);
    }

    // ------------------------------------------------------------------
    // Result type
    // ------------------------------------------------------------------

    public static final class Result {
        public final double value;
        public final double estimatedError;
        public final int panelCount;
        public final int maxDepth;
        public final int crossCheckCount;
        public final int forcedSplitCount;
        public final String singularitySubtractionInfo;
        public final List<TurboIntegratorEngine.Singularity> singularities;

        Result(double value, double estimatedError, int panelCount, int maxDepth,
               int crossCheckCount, int forcedSplitCount, String singularitySubtractionInfo,
               List<TurboIntegratorEngine.Singularity> singularities) {
            this.value = value;
            this.estimatedError = estimatedError;
            this.panelCount = panelCount;
            this.maxDepth = maxDepth;
            this.crossCheckCount = crossCheckCount;
            this.forcedSplitCount = forcedSplitCount;
            this.singularitySubtractionInfo = singularitySubtractionInfo;
            this.singularities = singularities;
        }

        @Override
        public String toString() {
            return String.format(
                    "value=%.15g  errEst~%.3e  panels=%d  maxDepth=%d  crossChecks=%d  forcedSplits=%d%s  flags=%d%s",
                    value, estimatedError, panelCount, maxDepth, crossCheckCount, forcedSplitCount,
                    singularitySubtractionInfo == null ? "" : ("  subtracted=[" + singularitySubtractionInfo + "]"),
                    singularities.size(), singularities.isEmpty() ? "" : ("  " + singularities));
        }
    }

    // ------------------------------------------------------------------
    // Builder
    // ------------------------------------------------------------------

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int crossCheckOrder = DEFAULT_CROSS_CHECK_ORDER;
        private double absTol = 1e-11;
        private double relTol = 1e-11;
        private int maxDepth = 40;
        private boolean strictSingularities = true;
        private boolean crossCheckEnabled = true;
        private double crossCheckLooseness = 50.0;
        private boolean singularitySubtractionEnabled = true;
        private double singularityConsistencyTol = DEFAULT_SINGULARITY_CONSISTENCY_TOL;
        private ProgressListener progressListener = ProgressListener.NO_OP;

        public Builder crossCheckOrder(int order) { this.crossCheckOrder = order; return this; }
        public Builder absoluteTolerance(double absTol) { this.absTol = absTol; return this; }
        public Builder relativeTolerance(double relTol) { this.relTol = relTol; return this; }
        public Builder maxDepth(int maxDepth) { this.maxDepth = maxDepth; return this; }
        public Builder strictSingularities(boolean strict) { this.strictSingularities = strict; return this; }
        public Builder crossCheckEnabled(boolean enabled) { this.crossCheckEnabled = enabled; return this; }
        public Builder crossCheckLooseness(double looseness) { this.crossCheckLooseness = looseness; return this; }
        public Builder singularitySubtractionEnabled(boolean enabled) { this.singularitySubtractionEnabled = enabled; return this; }
        public Builder singularityConsistencyTolerance(double tol) { this.singularityConsistencyTol = tol; return this; }

        public Builder progressListener(ProgressListener listener) {
            this.progressListener = listener != null ? listener : ProgressListener.NO_OP;
            return this;
        }

        public Integrator build(MathExpression me, String wrtVar) {
            return new Integrator(me, wrtVar, crossCheckOrder, absTol, relTol,
                    maxDepth, strictSingularities, crossCheckEnabled, crossCheckLooseness,
                    singularitySubtractionEnabled, singularityConsistencyTol, progressListener);
        }

        public Integrator build(String expr, String wrtVar) {
            MathExpression me = new MathExpression(expr);
            return build(me, wrtVar);
        }
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    private final AutoDiffNEvaluator ad;
    private final String wrtVar;
    private final int crossCheckOrder;
    private final int crossCheckAdOrder;
    private final int maxDepth;
    private final double absTol;
    private final double relTol;
    private final boolean strictSingularities;
    private final boolean crossCheckEnabled;
    private final double crossCheckLooseness;
    private final boolean singularitySubtractionEnabled;
    private final double singularityConsistencyTol;
    private final ProgressListener progressListener;

    private final double[] fnBuf;
    private final double[] fv1;
    private final double[] fv2;
    private final double[] ccCoeffs;
    private final double[] probeCoeffs;

    private double[] stackA;
    private double[] stackB;
    private int[] stackDepth;
    private int stackSize;

    private double gkResult;
    private double gkAbsErr;
    private double lastCancellationFloor;
    private double lastSubtractedMagnitude;
    private double ccValue;
    private boolean ccConverging;
    private String lastDomainErrorMessage;

    private boolean leftActive, rightActive;
    private double leftX0, leftN, leftC;
    private double rightX0, rightN, rightC;
    private boolean crossCheckActiveThisCall;

    private double lastErrorEstimate;
    private int lastPanelCount;
    private int lastMaxDepthReached;
    private int lastCrossCheckCount;
    private int lastForcedSplitCount;
    private String lastSingularitySubtractionInfo;
    private final List<TurboIntegratorEngine.Singularity> singularities;

    public Integrator(MathExpression me, String wrtVar, int crossCheckOrder,
                                    double absTol, double relTol, int maxDepth) {
        this(me, wrtVar, crossCheckOrder, absTol, relTol, maxDepth, true, true, 50.0, true,
                DEFAULT_SINGULARITY_CONSISTENCY_TOL, ProgressListener.NO_OP);
    }

    public Integrator(MathExpression me, String wrtVar, int crossCheckOrder,
                                    double absTol, double relTol, int maxDepth,
                                    boolean strictSingularities, boolean crossCheckEnabled,
                                    double crossCheckLooseness, boolean singularitySubtractionEnabled,
                                    double singularityConsistencyTol, ProgressListener progressListener) {
        if (crossCheckOrder < 2 || (crossCheckOrder % 2) != 0) {
            throw new IllegalArgumentException(
                    "crossCheckOrder must be a positive even number (midpoint expansion cancels odd terms)");
        }
        if (crossCheckOrder > MAX_CROSS_CHECK_ORDER) {
            throw new IllegalArgumentException(
                    "crossCheckOrder must be <= " + MAX_CROSS_CHECK_ORDER
                            + "; beyond this the cross-check is no longer 'cheap' relative to Gauss-Kronrod --"
                            + " use TurboIntegratorEngine directly if you need a high-order Taylor rule");
        }
        if (maxDepth < 1) {
            throw new IllegalArgumentException("maxDepth must be >= 1");
        }
        if (absTol < 0 || relTol < 0) {
            throw new IllegalArgumentException("tolerances must be >= 0");
        }
        if (crossCheckLooseness < 1.0) {
            throw new IllegalArgumentException("crossCheckLooseness must be >= 1.0");
        }
        if (singularityConsistencyTol < 0) {
            throw new IllegalArgumentException("singularityConsistencyTol must be >= 0");
        }
        if (wrtVar == null || wrtVar.trim().isEmpty()) {
            throw new IllegalArgumentException("wrtVar must be a non-empty variable name");
        }

        this.wrtVar = wrtVar;
        this.crossCheckOrder = crossCheckOrder;
        this.crossCheckAdOrder = crossCheckOrder + 2;
        this.ad = new AutoDiffNEvaluator(me, crossCheckAdOrder);
        this.absTol = absTol;
        this.relTol = relTol;
        this.maxDepth = maxDepth;
        this.strictSingularities = strictSingularities;
        this.crossCheckEnabled = crossCheckEnabled;
        this.crossCheckLooseness = crossCheckLooseness;
        this.singularitySubtractionEnabled = singularitySubtractionEnabled;
        this.singularityConsistencyTol = singularityConsistencyTol;
        this.progressListener = progressListener != null ? progressListener : ProgressListener.NO_OP;

        this.fnBuf = new double[1];
        this.fv1 = new double[7];
        this.fv2 = new double[7];
        this.ccCoeffs = new double[crossCheckAdOrder + 1];
        this.probeCoeffs = new double[3];

        int cap = maxDepth + 4;
        this.stackA = new double[cap];
        this.stackB = new double[cap];
        this.stackDepth = new int[cap];
        this.stackSize = 0;

        this.singularities = new ArrayList<>();
    }

    public static Integrator forExpression(String expr, String wrtVar) {
        return builder().build(expr, wrtVar);
    }

    // ------------------------------------------------------------------
    // Fixed-capacity primitive stack helpers
    // ------------------------------------------------------------------

    private void stackPush(double a, double b, int depth) {
        if (stackSize == stackA.length) {
            growStack();
        }
        stackA[stackSize] = a;
        stackB[stackSize] = b;
        stackDepth[stackSize] = depth;
        stackSize++;
    }

    private void growStack() {
        int newCap = stackA.length * 2;
        stackA = Arrays.copyOf(stackA, newCap);
        stackB = Arrays.copyOf(stackB, newCap);
        stackDepth = Arrays.copyOf(stackDepth, newCap);
    }

    // ------------------------------------------------------------------
    // Integration
    // ------------------------------------------------------------------

    public double integrate(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            throw new IllegalArgumentException(
                    "bounds must be finite (got [" + a + ", " + b + "]); this integrator uses local panel"
                            + " rules and does not support infinite domains -- apply your own variable"
                            + " substitution first, or use a quadrature rule designed for"
                            + " semi-infinite/infinite intervals (e.g. tanh-sinh)");
        }

        singularities.clear();
        lastErrorEstimate = 0.0;
        lastPanelCount = 0;
        lastMaxDepthReached = 0;
        lastCrossCheckCount = 0;
        lastForcedSplitCount = 0;
        lastSingularitySubtractionInfo = null;
        leftActive = false;
        rightActive = false;

        if (a == b) {
            return 0.0;
        }
        boolean negate = b < a;
        double lo = negate ? b : a;
        double hi = negate ? a : b;
        double fullWidth = hi - lo;
        double minWidth = Math.max(fullWidth * 1e-13, Double.MIN_NORMAL);

        double correction = 0.0;
        if (singularitySubtractionEnabled) {
            SingularityProbe leftProbe = probeEndpointSingularity(lo, fullWidth, +1.0);
            SingularityProbe rightProbe = probeEndpointSingularity(hi, fullWidth, -1.0);

            if (leftProbe.found && leftProbe.n <= -1.0) {
                throw new TurboIntegratorEngine.NonIntegrableSingularityException(lo, 0.0,
                        "Detected a non-integrable power-law singularity at the lower bound"
                                + " (empirical exponent n=" + leftProbe.n + " <= -1); this integral does not converge",
                        null);
            }
            if (rightProbe.found && rightProbe.n <= -1.0) {
                throw new TurboIntegratorEngine.NonIntegrableSingularityException(hi, 0.0,
                        "Detected a non-integrable power-law singularity at the upper bound"
                                + " (empirical exponent n=" + rightProbe.n + " <= -1); this integral does not converge",
                        null);
            }

            StringBuilder info = null;
            if (leftProbe.found) {
                leftActive = true;
                leftX0 = lo; leftN = leftProbe.n; leftC = leftProbe.c;
                correction += leftC * Math.pow(fullWidth, leftN + 1.0) / (leftN + 1.0);
                info = new StringBuilder(String.format("lower bound n=%.6g C=%.6g", leftN, leftC));
            }
            if (rightProbe.found) {
                rightActive = true;
                rightX0 = hi; rightN = rightProbe.n; rightC = rightProbe.c;
                correction += rightC * Math.pow(fullWidth, rightN + 1.0) / (rightN + 1.0);
                if (info == null) info = new StringBuilder();
                else info.append("; ");
                info.append(String.format("upper bound n=%.6g C=%.6g", rightN, rightC));
            }
            lastSingularitySubtractionInfo = (info == null) ? null : info.toString();
        }
        crossCheckActiveThisCall = crossCheckEnabled && !leftActive && !rightActive;

        stackSize = 0;
        stackPush(lo, hi, 0);

        double sum = 0.0, comp = 0.0;
        double errAccum = 0.0;
        int panelCount = 0;
        int crossCheckCount = 0;
        int forcedSplitCount = 0;
        int maxDepthSeen = 0;

        while (stackSize > 0) {
            stackSize--;
            double pa = stackA[stackSize];
            double pb = stackB[stackSize];
            int pDepth = stackDepth[stackSize];
            if (pDepth > maxDepthSeen) {
                maxDepthSeen = pDepth;
            }

            double width = pb - pa;
            boolean canSplit = pDepth < maxDepth && width > minWidth;

            boolean gkOk = computeQK15(pa, pb);
            if (!gkOk) {
                if (canSplit) {
                    double m = 0.5 * (pa + pb);
                    stackPush(pa, m, pDepth + 1);
                    stackPush(m, pb, pDepth + 1);
                } else {
                    recordSingularityOrThrow(pa, pb,
                            "AD domain error evaluating Gauss-Kronrod nodes: " + lastDomainErrorMessage, null);
                }
                continue;
            }

            // Budget acknowledges the same cancellation-noise floor already folded into gkAbsErr
            // (see class javadoc) -- otherwise a panel that has hit its precision limit near a
            // subtracted singularity would be bisected forever chasing an unreachable target.
            double budget = Math.max(absTol * (width / fullWidth),
                    Math.max(relTol * Math.abs(gkResult), lastCancellationFloor));
            boolean gkConverged = gkAbsErr <= budget;

            if (!gkConverged) {
                if (canSplit) {
                    double m = 0.5 * (pa + pb);
                    stackPush(pa, m, pDepth + 1);
                    stackPush(m, pb, pDepth + 1);
                    continue;
                }
                recordSingularityOrThrow(pa, pb,
                        "Gauss-Kronrod did not converge to tolerance (reported error " + gkAbsErr
                                + " vs budget " + budget + ") after reaching the panel-depth/width limit",
                        null);
                double y = gkResult - comp;
                double t = sum + y; comp = (t - sum) - y; sum = t;
                errAccum += gkAbsErr;
                panelCount++;
                progressListener.onProgress(panelCount, maxDepthSeen, negate ? -sum : sum);
                continue;
            }

            if (!crossCheckActiveThisCall) {
                double y = gkResult - comp;
                double t = sum + y; comp = (t - sum) - y; sum = t;
                errAccum += gkAbsErr;
                panelCount++;
                progressListener.onProgress(panelCount, maxDepthSeen, negate ? -sum : sum);
                continue;
            }

            crossCheckCount++;
            boolean ccOk = crossCheckPanel(pa, pb);
            double crossTol = Math.max(budget, absTol) * crossCheckLooseness;
            boolean disagree = !ccOk || !ccConverging || (Math.abs(ccValue - gkResult) > crossTol);

            if (disagree) {
                forcedSplitCount++;
                if (canSplit) {
                    double m = 0.5 * (pa + pb);
                    stackPush(pa, m, pDepth + 1);
                    stackPush(m, pb, pDepth + 1);
                    continue;
                }
                recordSingularityOrThrow(pa, pb,
                        "Gauss-Kronrod/AD cross-check disagreement could not be resolved by further"
                                + " bisection" + (ccOk
                                ? (" (GK=" + gkResult + ", AD=" + ccValue + ")")
                                : " (AD cross-check itself failed to evaluate)"),
                        null);
                double y = gkResult - comp;
                double t = sum + y; comp = (t - sum) - y; sum = t;
                errAccum += gkAbsErr;
                panelCount++;
                progressListener.onProgress(panelCount, maxDepthSeen, negate ? -sum : sum);
                continue;
            }

            double y = gkResult - comp;
            double t = sum + y; comp = (t - sum) - y; sum = t;
            errAccum += gkAbsErr;
            panelCount++;
            progressListener.onProgress(panelCount, maxDepthSeen, negate ? -sum : sum);
        }

        sum += correction;

        lastErrorEstimate = errAccum;
        lastPanelCount = panelCount;
        lastMaxDepthReached = maxDepthSeen;
        lastCrossCheckCount = crossCheckCount;
        lastForcedSplitCount = forcedSplitCount;

        return negate ? -sum : sum;
    }

    public Result integrateResult(double a, double b) {
        double value = integrate(a, b);
        return new Result(value, lastErrorEstimate, lastPanelCount, lastMaxDepthReached,
                lastCrossCheckCount, lastForcedSplitCount, lastSingularitySubtractionInfo,
                new ArrayList<>(singularities));
    }

    public double getLastErrorEstimate() { return lastErrorEstimate; }
    public int getLastPanelCount() { return lastPanelCount; }
    public int getLastMaxDepth() { return lastMaxDepthReached; }
    public int getLastCrossCheckCount() { return lastCrossCheckCount; }
    public int getLastForcedSplitCount() { return lastForcedSplitCount; }
    public boolean isStrictSingularities() { return strictSingularities; }
    public String getLastSingularitySubtractionInfo() { return lastSingularitySubtractionInfo; }

    public List<TurboIntegratorEngine.Singularity> getSingularities() {
        return Collections.unmodifiableList(singularities);
    }

    private void recordSingularityOrThrow(double pa, double pb, String reason, Throwable cause) {
        double location = 0.5 * (pa + pb);
        double width = pb - pa;
        if (strictSingularities) {
            throw new TurboIntegratorEngine.NonIntegrableSingularityException(location, width, reason, cause);
        }
        LOG.warning(String.format(
                "Unresolved panel near x=%.15g (width %.3e): %s -- lenient mode", location, width, reason));
        singularities.add(new TurboIntegratorEngine.Singularity(location, width, reason));
    }

    // ------------------------------------------------------------------
    // Endpoint power-law singularity probing
    // ------------------------------------------------------------------

    private static final class SingularityProbe {
        final boolean found;
        final double n;
        final double c;

        SingularityProbe(boolean found, double n, double c) {
            this.found = found;
            this.n = n;
            this.c = c;
        }

        static final SingularityProbe NONE = new SingularityProbe(false, 0.0, 0.0);
    }

    private SingularityProbe probeEndpointSingularity(double x0, double intervalWidth, double dir) {
        double s = Math.max(intervalWidth * 1e-4, 1e-12);
        double probeX = x0 + dir * s;

        try {
            ad.taylorCoefficients(wrtVar, probeX, 2, probeCoeffs);
        } catch (ArithmeticException domainEx) {
            return SingularityProbe.NONE;
        }
        double f = probeCoeffs[0];
        double fp = probeCoeffs[1];
        double fpp = 2.0 * probeCoeffs[2];

        if (!Double.isFinite(f) || !Double.isFinite(fp) || !Double.isFinite(fpp) || f == 0.0 || fp == 0.0) {
            return SingularityProbe.NONE;
        }

        double t = dir * s;
        double n1 = t * fp / f;
        double n2 = 1.0 + t * fpp / fp;

        if (!Double.isFinite(n1) || !Double.isFinite(n2)) {
            return SingularityProbe.NONE;
        }
        if (Math.abs(n1 - n2) > singularityConsistencyTol) {
            return SingularityProbe.NONE;
        }

        double n = 0.5 * (n1 + n2);

        if (n <= -1.0) {
            return new SingularityProbe(true, n, Double.NaN);
        }
        if (Math.abs(n) < 0.05) {
            return SingularityProbe.NONE;
        }

        double c = f / Math.pow(s, n);
        if (!Double.isFinite(c)) {
            return SingularityProbe.NONE;
        }
        return new SingularityProbe(true, n, c);
    }

    /**
     * Samples the integrand: raw value via {@link #evalFnAt}, minus
     * whichever endpoint singular terms are active. Also records, in
     * {@link #lastSubtractedMagnitude}, how large a term was actually
     * subtracted this call (0 if none) -- fed into the cancellation floor
     * computed per-panel in {@link #computeQK15}.
     */
    private boolean sampleF(double xPoint) {
        if (!evalFnAt(xPoint)) {
            lastSubtractedMagnitude = 0.0;
            return false;
        }
        if (!leftActive && !rightActive) {
            lastSubtractedMagnitude = 0.0;
            return true;
        }
        double g = fnBuf[0];
        double subMag = 0.0;
        if (leftActive) {
            double s = xPoint - leftX0;
            if (s > 0) {
                double term = leftC * Math.pow(s, leftN);
                g -= term;
                subMag += Math.abs(term);
            }
        }
        if (rightActive) {
            double s = rightX0 - xPoint;
            if (s > 0) {
                double term = rightC * Math.pow(s, rightN);
                g -= term;
                subMag += Math.abs(term);
            }
        }
        lastSubtractedMagnitude = subMag;
        if (!Double.isFinite(g)) {
            lastDomainErrorMessage = "non-finite value after endpoint singularity subtraction";
            return false;
        }
        fnBuf[0] = g;
        return true;
    }

    // ------------------------------------------------------------------
    // Gauss-Kronrod 7-15 panel rule
    // ------------------------------------------------------------------

    private boolean computeQK15(double a, double b) {
        double centr = 0.5 * (a + b);
        double hlgth = 0.5 * (b - a);
        double maxSubtractedMag = 0.0;

        if (!sampleF(centr)) return false;
        double fc = fnBuf[0];
        maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);

        double resg = WG[3] * fc;
        double resk = WGK[7] * fc;
        double resabs = Math.abs(resk);

        for (int j = 0; j < 3; j++) {
            int jtw = 2 * (j + 1) - 1;
            double absc = hlgth * XGK[jtw];
            if (!sampleF(centr - absc)) return false;
            double f1 = fnBuf[0];
            maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);
            if (!sampleF(centr + absc)) return false;
            double f2 = fnBuf[0];
            maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);
            fv1[jtw] = f1; fv2[jtw] = f2;
            double fsum = f1 + f2;
            resg += WG[j] * fsum;
            resk += WGK[jtw] * fsum;
            resabs += WGK[jtw] * (Math.abs(f1) + Math.abs(f2));
        }
        for (int j = 0; j < 4; j++) {
            int jtwm1 = 2 * (j + 1) - 2;
            double absc = hlgth * XGK[jtwm1];
            if (!sampleF(centr - absc)) return false;
            double f1 = fnBuf[0];
            maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);
            if (!sampleF(centr + absc)) return false;
            double f2 = fnBuf[0];
            maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);
            fv1[jtwm1] = f1; fv2[jtwm1] = f2;
            double fsum = f1 + f2;
            resk += WGK[jtwm1] * fsum;
            resabs += WGK[jtwm1] * (Math.abs(f1) + Math.abs(f2));
        }

        double reskh = resk * 0.5;
        double resasc = WGK[7] * Math.abs(fc - reskh);
        for (int j = 0; j < 7; j++) {
            resasc += WGK[j] * (Math.abs(fv1[j] - reskh) + Math.abs(fv2[j] - reskh));
        }

        double result = resk * hlgth;
        resabs = resabs * Math.abs(hlgth);
        resasc = resasc * Math.abs(hlgth);
        double abserr = Math.abs((resk - resg) * hlgth);
        if (resasc != 0.0 && abserr != 0.0) {
            abserr = resasc * Math.min(1.0, Math.pow(200.0 * abserr / resasc, 1.5));
        }
        if (resabs > UFLOW / (50 * EPMACH)) {
            abserr = Math.max(EPMACH * 50 * resabs, abserr);
        }

        // Cancellation-noise floor: see class javadoc. Applied to the reported error AND (via
        // lastCancellationFloor, read by the caller when building `budget`) the acceptance test,
        // so both agree on what's actually achievable near a subtracted singularity.
        double cancellationFloor = EPMACH * CANCELLATION_FLOOR_FACTOR * maxSubtractedMag;
        if (cancellationFloor > abserr) {
            abserr = cancellationFloor;
        }
        lastCancellationFloor = cancellationFloor;

        gkResult = result;
        gkAbsErr = abserr;
        return true;
    }

    private boolean evalFnAt(double xPoint) {
        try {
            ad.evaluateRPN(wrtVar, xPoint, 0, fnBuf);
        } catch (ArithmeticException domainEx) {
            lastDomainErrorMessage = domainEx.getMessage();
            return false;
        }
        if (!Double.isFinite(fnBuf[0])) {
            lastDomainErrorMessage = "non-finite function value (overflow, or an unguarded near-pole)";
            return false;
        }
        return true;
    }

    // ------------------------------------------------------------------
    // Low-order AD cross-check (midpoint Taylor expansion, even orders only)
    // ------------------------------------------------------------------

    private boolean crossCheckPanel(double a, double b) {
        double m = 0.5 * (a + b);
        double h = 0.5 * (b - a);
        double h2 = h * h;

        try {
            ad.taylorCoefficients(wrtVar, m, crossCheckAdOrder, ccCoeffs);
        } catch (ArithmeticException domainEx) {
            return false;
        }
        for (double c : ccCoeffs) {
            if (!Double.isFinite(c)) return false;
        }

        double sum = 0.0;
        double lastTerm = 0.0, prevTerm = 0.0;
        double hp = h;
        for (int i = 0; i <= crossCheckOrder; i += 2) {
            double term = 2.0 * ccCoeffs[i] * hp / (i + 1);
            sum += term;
            prevTerm = lastTerm;
            lastTerm = term;
            hp *= h2;
        }
        double probeTerm = 2.0 * ccCoeffs[crossCheckOrder + 2] * hp / (crossCheckOrder + 3);

        boolean converging = (lastTerm == 0.0) || (Math.abs(probeTerm) <= Math.abs(lastTerm));
        if (prevTerm != 0.0 && Math.abs(lastTerm) > Math.abs(prevTerm)) {
            converging = false;
        }

        ccValue = sum;
        ccConverging = converging;
        return true;
    }

    // ------------------------------------------------------------------
    // Self-tests
    // ------------------------------------------------------------------

    private static int passCount = 0;
    private static int failCount = 0;

    private static void expect(String label, double actual, double expected, double tol) {
        double err = Math.abs(actual - expected);
        boolean ok = err <= tol;
        System.out.printf("[%s] %-55s actual=%.15g expected=%.15g err=%.3e (tol=%.1e)%n",
                ok ? "PASS" : "FAIL", label, actual, expected, err, tol);
        if (ok) passCount++; else failCount++;
    }

    public static void main(String[] args) {
        Integrator expI = builder().build("exp(x)", "x");
        expect("exp(x) on [0,10]", expI.integrate(0, 10), Math.exp(10) - 1, 1e-8);

        Integrator sinI = builder().build("sin(x)", "x");
        expect("sin(x) on [0,1.5]", sinI.integrate(0, 1.5), 1 - Math.cos(1.5), 1e-9);

        Integrator polyI = builder().build("x^3+3*x^2-5*x-8", "x");
        expect("x^3+3x^2-5x-8 on [-2,3]", polyI.integrate(-2, 3), -1.25, 1e-9);

        Integrator revI = builder().build("exp(x)", "x");
        expect("exp(x) on [1,0] (reversed)", revI.integrate(1, 0), -(Math.E - 1), 1e-9);

        Integrator bumpWithCC = builder()
                .absoluteTolerance(1e-10).relativeTolerance(1e-10)
                .build("exp(-3200*(x-1.3)^2)", "x");
        double bumpTrue = 0.03133285343288750;
        expect("Gaussian bump WITH cross-check", bumpWithCC.integrate(0, 2), bumpTrue, 1e-8);

        Integrator sqrtI = builder().build("sqrt(x)", "x");
        expect("sqrt(x) on [0,1] (n=0.5 endpoint)", sqrtI.integrate(0, 1), 2.0 / 3.0, 1e-9);

        Integrator invSqrtI = builder().build("1/sqrt(x)", "x");
        expect("1/sqrt(x) on [0,1] (n=-0.5 endpoint)", invSqrtI.integrate(0, 1), 2.0, 1e-9);

        // The case that previously threw: real O(1) remainder (cos(x)) sitting under the
        // subtracted singular term, at default (tight) tolerance.
        Integrator mixedI = builder().build("1/sqrt(x) + cos(x)", "x");
        expect("1/sqrt(x)+cos(x) on [0,1] (cancellation-floor case)",
                mixedI.integrate(0, 1), 2.0 + Math.sin(1.0), 1e-8);
        System.out.println("  achieved error estimate=" + mixedI.getLastErrorEstimate()
                + " (nominal request was 1e-11 -- elevated estimate is expected/honest here, not a failure)");

        Integrator cbrtI = builder().build("1/(x^(1/3))", "x");
        expect("1/x^(1/3) on [0,1] (n=-1/3 endpoint)", cbrtI.integrate(0, 1), 1.5, 1e-8);

        Integrator divergent = builder().build("1/x^1.5", "x");
        try {
            divergent.integrate(0, 1);
            System.out.println("[FAIL] 1/x^1.5 on [0,1] should have thrown (non-integrable)");
            failCount++;
        } catch (TurboIntegratorEngine.NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/x^1.5 on [0,1] correctly threw: " + expectedEx.getMessage());
            passCount++;
        }

        Integrator poleStrict = builder().build("1/(x-0.5)", "x");
        try {
            poleStrict.integrate(0.1, 0.9);
            System.out.println("[FAIL] 1/(x-0.5) through pole (strict) should have thrown");
            failCount++;
        } catch (TurboIntegratorEngine.NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/(x-0.5) through pole (strict) correctly threw: " + expectedEx.getMessage());
            passCount++;
        }

        int[] callCount = {0};
        Integrator progressI = builder()
                .progressListener((panels, depth, est) -> callCount[0]++)
                .build("exp(-3200*(x-1.3)^2)", "x");
        progressI.integrate(0, 2);
        boolean progressOk = callCount[0] == progressI.getLastPanelCount() && callCount[0] > 0;
        System.out.printf("[%s] progress listener: %d calls (panels=%d)%n",
                progressOk ? "PASS" : "FAIL", callCount[0], progressI.getLastPanelCount());
        if (progressOk) passCount++; else failCount++;

        try {
            builder().build("x", "x").integrate(0, Double.POSITIVE_INFINITY);
            System.out.println("[FAIL] infinite bound should have thrown IllegalArgumentException");
            failCount++;
        } catch (IllegalArgumentException expectedEx) {
            System.out.println("[PASS] infinite bound correctly rejected: " + expectedEx.getMessage());
            passCount++;
        }

        System.out.println();
        System.out.println(passCount + " passed, " + failCount + " failed");
        if (failCount > 0) {
            System.exit(1);
        }
    }
}