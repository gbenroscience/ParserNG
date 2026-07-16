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

import com.github.gbenroscience.math.differentialcalculus.autodiff.AutoDiffNEvaluator;
import com.github.gbenroscience.parser.MathExpression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Composite adaptive integrator using the 7-15 Gauss-Kronrod pair as the
 * primary panel rule, {@link AutoDiffNEvaluator}-based domain-error
 * classification, a low-order AD Taylor cross-check, and analytic endpoint
 * power-law singularity subtraction with a floating-point cancellation floor.
 * This revision adds a precision-floor acceptance path and an opt-in
 * cross-check depth cap -- see "Precision-floor acceptance" and "Cross-check
 * depth cap" below for what was and wasn't adopted from a round of external
 * review, and why.
 *
 * <h2>Precision-floor acceptance (new)</h2>
 * A panel that exhausts its bisection budget (hits the width/depth limit) while
 * still failing Gauss-Kronrod's own convergence test is normally a strict-mode
 * throw. This revision adds one exception: if the panel's reported error is
 * essentially equal to its own {@link
 * #lastCancellationFloor} -- i.e. the residual *is* the already-quantified
 * subtraction cancellation noise (see below), not an unknown or unresolved
 * failure -- it is accepted regardless of {@link Builder#strictSingularities},
 * flagged as a {@link TurboIntegratorXEngine.Singularity} for transparency.
 * This is deliberately narrow: further bisection provably cannot reduce a
 * cancellation floor, since it doesn't shrink with panel width, so refusing to
 * accept here would only ever produce an unrecoverable throw, never a more
 * accurate answer. It does <em>not</em> apply to genuinely unresolved failures
 * (real domain errors, oscillation-driven non-convergence, unresolved
 * cross-check disagreement) -- those still throw in strict mode, exactly as
 * before. An external reviewer's suggestion to instead "hand the micro-panel to
 * Taylor integration" was not adopted: {@code x^n} for non-integer {@code n}
 * has no valid Taylor series at the singular endpoint in the first place, which
 * is precisely why subtraction was needed there rather than a Taylor panel rule
 * -- the same reason {@link
 * TurboIntegratorXEngine} falls back to Simpson, not Taylor, in the analogous
 * situation.
 *
 * <h2>Cross-check depth cap (new, opt-in)</h2>
 * {@link Builder#crossCheckMaxDepth(int)} lets a caller restrict the AD
 * cross-check to panels at or above a given bisection depth (default:
 * unlimited, i.e. every Gauss-Kronrod-converged panel is cross-checked,
 * unchanged from prior behavior). This exists to give callers who understand
 * the tradeoff explicit control over the cross-check's real cost. It is
 * deliberately <em>not</em> gated on "Gauss-Kronrod failed to converge" -- the
 * cross-check exists specifically to catch panels where Gauss-Kronrod claims
 * success but is wrong (see the Gaussian-bump case below, whose defining
 * feature is a *low* reported GK error, not a high one); gating on GK failure
 * would silently disable the check on exactly the cases it exists to catch.
 * Lowering {@code crossCheckMaxDepth} trades the narrow-feature guarantee away
 * deliberately and explicitly, at whatever depth the caller chooses -- it is
 * not a default, and it does not change behavior unless set.
 *
 * <h2>Endpoint singularity subtraction</h2>
 * At the start of every {@link #integrate(double, double)} call, both bounds
 * are probed for local power-law behavior {@code f(x) ~= C*s^n} ({@code s} =
 * distance into the domain from that endpoint), estimated two independent ways
 * from a single AD probe -- {@code n1 = s*f'(x)/f(x)} and
 * {@code n2 = 1 + s*f''(x)/f'(x)}, both exact for a pure power term -- and only
 * trusted if they agree within {@link Builder#singularityConsistencyTolerance}.
 * A second probe at {@link Builder#singularityScaleFactor} times the distance
 * (default 100x) must agree with the near-scale exponent within
 * {@link Builder#crossScaleConsistencyTolerance}, catching contaminated cases
 * like {@code 1/sqrt(x) + 5000*x} that a single-point check alone would accept.
 * A confidently-estimated exponent {@code <= -1} always throws
 * {@link NonIntegrableSingularityException}, regardless of strict/lenient mode
 * -- there is no partial estimate for a divergent integral.
 *
 * <p>
 * <b>Cancellation floor.</b> Subtracting {@code C*s^n} from {@code f(x)} near a
 * singular endpoint means computing a difference of two comparably large
 * numbers; if the true remainder is itself O(1) (e.g.
 * {@code 1/sqrt(x) + cos(x)}), the achievable absolute precision is capped at
 * roughly {@code machine_epsilon * (subtracted magnitude)}, a
 * representable-precision floor, not a convergence problem. This class folds
 * that floor into both the reported error and the acceptance budget
 * consistently, and (new) recognizes a panel that hits exactly this floor as an
 * acceptable terminal state -- see "Precision-floor acceptance" above.
 *
 * <p>
 * <b>Cross-check stays live per-panel, not per-call.</b> Each panel re-enables
 * the cross-check individually once its own subtracted contribution
 * ({@link #lastPanelMaxSubtractedMag}) is negligible relative to its
 * Gauss-Kronrod result (threshold: {@link
 * Builder#crossCheckSubtractionNegligibility}), rather than disabling it for
 * the whole call the moment any endpoint is subtracted.
 *
 * <h2>Zero-allocation steady state</h2>
 * The per-panel loop -- {@link #computeQK15}, {@link #sampleF}, {@link
 * #evalFnAt}, {@link #crossCheckPanel} -- performs no {@code new}, no
 * autoboxing, no String work on its common (panel-accepted) path; all scratch
 * is pre-sized instance state. Endpoint probing writes scratch fields
 * ({@link #probeN}, {@link #probeC}, {@link #probeDomainFailure}) instead of
 * returning objects. {@link #getLastSingularitySubtractionInfo()} builds its
 * description lazily, only when read. What still legitimately allocates: the
 * panel stack only grows (rarely, amortized) past its initial capacity;
 * exception messages and lenient-mode singularity records build strings, but
 * only on already-exceptional/rare paths; {@link
 * #integrateResult} is the documented allocating convenience wrapper.
 *
 * <h2>Why GK as the primary rule / why GK alone is not enough</h2>
 * Gauss-Kronrod is exact for polynomials up to degree 23 (K15) using only 15
 * function evaluations, with a self-contained {@code resasc}-refined error
 * estimate needing no derivatives. But that estimate compares two rules built
 * from the same 15 points, so a feature of {@code f} that falls between them
 * can fool both together -- verified empirically: {@code exp(-3200*(x-1.3)^2)}
 * on {@code [0,2]}, plain GK-adaptive at {@code absTol=1e-10}, returns
 * {@code 0.0} (true value {@code 0.0313}) because the starting panel's reported
 * error was {@code ~5.6e-13} -- a *low*, tolerance-satisfying number, which is
 * exactly why gating the cross-check on "GK reported a high error" cannot work.
 *
 * <h2>Progress reporting</h2> {@link ProgressListener#onProgress} reports
 * accepted panels so far, deepest bisection level so far, the running estimate
 * (including the endpoint correction term throughout), and a width-resolved
 * completion fraction in {@code [0,1]}, monotonic, ending at exactly
 * {@code 1.0}.
 *
 * <h2>Constants</h2>
 * The 15-point Kronrod / 7-point Gauss abscissae and weights are the standard
 * QUADPACK {@code dqk15.f} table (Fullerton, Bell Labs, 1981; cross-checked
 * against two independent Netlib mirrors).
 *
 * <h2>Thread-safety</h2>
 * Not thread-safe. One instance per thread, built from the same {@code
 * rpnTokens} array (which is only read).
 */
public class IntegratorX {

    private static final Logger LOG = Logger.getLogger(IntegratorX.class.getName());

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
    public static final double DEFAULT_CROSS_SCALE_CONSISTENCY_TOL = 0.1;
    public static final double DEFAULT_SINGULARITY_SCALE_FACTOR = 100.0;
    public static final double DEFAULT_CROSS_CHECK_SUBTRACTION_NEGLIGIBILITY = 1e-6;
    public static final double CANCELLATION_FLOOR_FACTOR = 100.0;
    /**
     * How close (as a multiplier on lastCancellationFloor) an exhausted panel's
     * error must be to accept it regardless of strict mode.
     */
    public static final double PRECISION_FLOOR_ACCEPT_MARGIN = 2.0;
    public static final int DEFAULT_CROSS_CHECK_MAX_DEPTH = Integer.MAX_VALUE;

    // ------------------------------------------------------------------
    // Progress reporting
    // ------------------------------------------------------------------
    public interface ProgressListener {

        ProgressListener NO_OP = (panelsAccepted, currentMaxDepth, currentEstimate, fractionComplete) -> {
        };

        void onProgress(int panelsAccepted, int currentMaxDepth, double currentEstimate, double fractionComplete);
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
        public final List<Singularity> singularities;

        Result(double value, double estimatedError, int panelCount, int maxDepth,
                int crossCheckCount, int forcedSplitCount, String singularitySubtractionInfo,
                List<Singularity> singularities) {
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
        private int crossCheckMaxDepth = DEFAULT_CROSS_CHECK_MAX_DEPTH;
        private boolean singularitySubtractionEnabled = true;
        private double singularityConsistencyTol = DEFAULT_SINGULARITY_CONSISTENCY_TOL;
        private double crossScaleConsistencyTol = DEFAULT_CROSS_SCALE_CONSISTENCY_TOL;
        private double singularityScaleFactor = DEFAULT_SINGULARITY_SCALE_FACTOR;
        private double crossCheckSubtractionNegligibility = DEFAULT_CROSS_CHECK_SUBTRACTION_NEGLIGIBILITY;
        private ProgressListener progressListener = ProgressListener.NO_OP;

        public Builder crossCheckOrder(int order) {
            this.crossCheckOrder = order;
            return this;
        }

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

        public Builder crossCheckEnabled(boolean enabled) {
            this.crossCheckEnabled = enabled;
            return this;
        }

        public Builder crossCheckLooseness(double looseness) {
            this.crossCheckLooseness = looseness;
            return this;
        }

        /**
         * Restricts the AD cross-check to panels at or above this bisection
         * depth (default: unlimited). See class javadoc, "Cross-check depth
         * cap" -- this is an explicit, opt-in tradeoff of the narrow-feature
         * safety guarantee for speed, not a default behavior change.
         */
        public Builder crossCheckMaxDepth(int depth) {
            this.crossCheckMaxDepth = depth;
            return this;
        }

        public Builder singularitySubtractionEnabled(boolean enabled) {
            this.singularitySubtractionEnabled = enabled;
            return this;
        }

        public Builder singularityConsistencyTolerance(double tol) {
            this.singularityConsistencyTol = tol;
            return this;
        }

        public Builder crossScaleConsistencyTolerance(double tol) {
            this.crossScaleConsistencyTol = tol;
            return this;
        }

        public Builder singularityScaleFactor(double factor) {
            this.singularityScaleFactor = factor;
            return this;
        }

        public Builder crossCheckSubtractionNegligibility(double rel) {
            this.crossCheckSubtractionNegligibility = rel;
            return this;
        }

        public Builder progressListener(ProgressListener listener) {
            this.progressListener = listener != null ? listener : ProgressListener.NO_OP;
            return this;
        }

        public IntegratorX build(MathExpression me, String wrtVar) {
            return new IntegratorX(me, wrtVar, crossCheckOrder, absTol, relTol,
                    maxDepth, strictSingularities, crossCheckEnabled, crossCheckLooseness, crossCheckMaxDepth,
                    singularitySubtractionEnabled, singularityConsistencyTol, crossScaleConsistencyTol,
                    singularityScaleFactor, crossCheckSubtractionNegligibility, progressListener);
        }

        public IntegratorX build(String expr, String wrtVar) {
            MathExpression me = new MathExpression(expr);
            return build(me, wrtVar);
        }
    }

    public static class TaylorIntegrationException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        public TaylorIntegrationException(String message) {
            super(message);
        }

        public TaylorIntegrationException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    public static final class NonIntegrableSingularityException extends TaylorIntegrationException {

        private static final long serialVersionUID = 1L;
        public final double location;
        public final double panelWidth;

        public NonIntegrableSingularityException(double location, double panelWidth, String reason, Throwable cause) {
            super(String.format(
                    "Non-integrable singularity near x=%.15g (panel width %.3e): %s. "
                    + "If a principal-value-style partial result is acceptable, build with "
                    + "strictSingularities(false) instead.",
                    location, panelWidth, reason), cause);
            this.location = location;
            this.panelWidth = panelWidth;
        }
    }

    // ------------------------------------------------------------------
    // Public diagnostics types
    // ------------------------------------------------------------------
    /**
     * A panel flagged during integration: either omitted (lenient mode) or
     * resolved via fallback.
     */
    public static final class Singularity {

        public final double location;
        public final double width;
        public final String reason;

        public Singularity(double location, double width, String reason) {
            this.location = location;
            this.width = width;
            this.reason = reason;
        }

        @Override
        public String toString() {
            return String.format("flag near x=%.15g (panel width %.3e): %s", location, width, reason);
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
    private final int crossCheckMaxDepth;
    private final boolean singularitySubtractionEnabled;
    private final double singularityConsistencyTol;
    private final double crossScaleConsistencyTol;
    private final double singularityScaleFactor;
    private final double crossCheckSubtractionNegligibility;
    private final ProgressListener progressListener;

    // Scratch (zero-allocation steady state)
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
    private double lastPanelMaxSubtractedMag;
    private double ccValue;
    private boolean ccConverging;
    private String lastDomainErrorMessage;
    private double probeN;
    private double probeC;
    private boolean probeDomainFailure;

    private boolean leftActive, rightActive;
    private double leftX0, leftN, leftC;
    private double rightX0, rightN, rightC;

    private double lastErrorEstimate;
    private int lastPanelCount;
    private int lastMaxDepthReached;
    private int lastCrossCheckCount;
    private int lastForcedSplitCount;
    private final List<Singularity> singularities;

    //FRESH
    // New scratch, sized once, zero-alloc on the hot path:
    private final double[] binomBuf; // size crossCheckAdOrder + 1

    // Written by computeTaylorRescue instead of returning an object:
    private double rescueValue;
    private double rescueErrEst;
    private boolean rescueConverging;

    public IntegratorX(MathExpression me, String wrtVar, int crossCheckOrder,
            double absTol, double relTol, int maxDepth) {
        this(me, wrtVar, crossCheckOrder, absTol, relTol, maxDepth, true, true, 50.0,
                DEFAULT_CROSS_CHECK_MAX_DEPTH, true,
                DEFAULT_SINGULARITY_CONSISTENCY_TOL, DEFAULT_CROSS_SCALE_CONSISTENCY_TOL,
                DEFAULT_SINGULARITY_SCALE_FACTOR, DEFAULT_CROSS_CHECK_SUBTRACTION_NEGLIGIBILITY,
                ProgressListener.NO_OP);
    }

    public IntegratorX(MathExpression me, String wrtVar, int crossCheckOrder,
            double absTol, double relTol, int maxDepth,
            boolean strictSingularities, boolean crossCheckEnabled,
            double crossCheckLooseness, int crossCheckMaxDepth,
            boolean singularitySubtractionEnabled,
            double singularityConsistencyTol, double crossScaleConsistencyTol,
            double singularityScaleFactor, double crossCheckSubtractionNegligibility,
            ProgressListener progressListener) {
        if (crossCheckOrder < 2 || (crossCheckOrder % 2) != 0) {
            throw new IllegalArgumentException(
                    "crossCheckOrder must be a positive even number (midpoint expansion cancels odd terms)");
        }
        if (crossCheckOrder > MAX_CROSS_CHECK_ORDER) {
            throw new IllegalArgumentException(
                    "crossCheckOrder must be <= " + MAX_CROSS_CHECK_ORDER
                    + "; beyond this the cross-check is no longer 'cheap' relative to Gauss-Kronrod --"
                    + " use TurboIntegratorXEngine directly if you need a high-order Taylor rule");
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
        if (crossCheckMaxDepth < 0) {
            throw new IllegalArgumentException("crossCheckMaxDepth must be >= 0");
        }
        if (singularityConsistencyTol < 0 || crossScaleConsistencyTol < 0) {
            throw new IllegalArgumentException("consistency tolerances must be >= 0");
        }
        if (singularityScaleFactor <= 1.0) {
            throw new IllegalArgumentException("singularityScaleFactor must be > 1.0");
        }
        if (crossCheckSubtractionNegligibility < 0) {
            throw new IllegalArgumentException("crossCheckSubtractionNegligibility must be >= 0");
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
        this.crossCheckMaxDepth = crossCheckMaxDepth;
        this.singularitySubtractionEnabled = singularitySubtractionEnabled;
        this.singularityConsistencyTol = singularityConsistencyTol;
        this.crossScaleConsistencyTol = crossScaleConsistencyTol;
        this.singularityScaleFactor = singularityScaleFactor;
        this.crossCheckSubtractionNegligibility = crossCheckSubtractionNegligibility;
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

        this.binomBuf = new double[crossCheckAdOrder + 1];
    }

    public static IntegratorX forExpression(String expr, String wrtVar) {
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
            boolean leftFound = probeEndpointSingularityTwoScale(lo, fullWidth, +1.0);
            double foundLeftN = leftFound ? probeN : 0.0;
            double foundLeftC = leftFound ? probeC : 0.0;

            boolean rightFound = probeEndpointSingularityTwoScale(hi, fullWidth, -1.0);
            double foundRightN = rightFound ? probeN : 0.0;
            double foundRightC = rightFound ? probeC : 0.0;

            if (leftFound && foundLeftN <= -1.0) {
                throw new NonIntegrableSingularityException(lo, 0.0,
                        "Detected a non-integrable power-law singularity at the lower bound"
                        + " (empirical exponent n=" + foundLeftN + " <= -1); this integral does not converge",
                        null);
            }
            if (rightFound && foundRightN <= -1.0) {
                throw new NonIntegrableSingularityException(hi, 0.0,
                        "Detected a non-integrable power-law singularity at the upper bound"
                        + " (empirical exponent n=" + foundRightN + " <= -1); this integral does not converge",
                        null);
            }

            if (leftFound) {
                leftActive = true;
                leftX0 = lo;
                leftN = foundLeftN;
                leftC = foundLeftC;
                correction += leftC * Math.pow(fullWidth, leftN + 1.0) / (leftN + 1.0);
            }
            if (rightFound) {
                rightActive = true;
                rightX0 = hi;
                rightN = foundRightN;
                rightC = foundRightC;
                correction += rightC * Math.pow(fullWidth, rightN + 1.0) / (rightN + 1.0);
            }
        }

        stackSize = 0;
        stackPush(lo, hi, 0);

        double sum = 0.0, comp = 0.0; // Kahan compensated summation
        double errAccum = 0.0;
        double resolvedWidth = 0.0;
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

                boolean rescueConverging = computeTaylorRescue(pa, pb);
                if (rescueConverging) {
                   sum = rescueValue;
                   errAccum = rescueErrEst;
                } else {
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

            }

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
                // Precision-floor acceptance (see class javadoc): if the residual error IS the
                // already-characterized cancellation floor, accept regardless of strict mode --
                // further bisection provably cannot help, since the floor doesn't shrink with
                // panel width.
                boolean cappedByCancellationFloor = lastCancellationFloor > 0.0
                        && gkAbsErr <= lastCancellationFloor * PRECISION_FLOOR_ACCEPT_MARGIN;
                if (cappedByCancellationFloor) {
                    singularities.add(new Singularity(0.5 * (pa + pb), width,
                            "accepted at the floating-point cancellation floor near a subtracted"
                            + " singularity (reported error " + gkAbsErr + " is the characterized"
                            + " noise floor, not an unresolved convergence failure); further"
                            + " bisection cannot improve this"));
                    double y = gkResult - comp;
                    double t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                    errAccum += gkAbsErr;
                    panelCount++;
                    resolvedWidth += width;
                    progressListener.onProgress(panelCount, maxDepthSeen,
                            negate ? -(sum + correction) : (sum + correction), resolvedWidth / fullWidth);
                    continue;
                }
                recordSingularityOrThrow(pa, pb,
                        "Gauss-Kronrod did not converge to tolerance (reported error " + gkAbsErr
                        + " vs budget " + budget + ") after reaching the panel-depth/width limit",
                        null);
                double y = gkResult - comp;
                double t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                errAccum += gkAbsErr;
                panelCount++;
                resolvedWidth += width;
                progressListener.onProgress(panelCount, maxDepthSeen,
                        negate ? -(sum + correction) : (sum + correction), resolvedWidth / fullWidth);
                continue;
            }

            boolean subtractionNegligibleHere = (leftActive || rightActive)
                    && lastPanelMaxSubtractedMag <= crossCheckSubtractionNegligibility * Math.abs(gkResult);
            boolean withinCrossCheckDepth = pDepth <= crossCheckMaxDepth;
            boolean runCrossCheckHere = crossCheckEnabled && withinCrossCheckDepth
                    && (!(leftActive || rightActive) || subtractionNegligibleHere);

            if (!runCrossCheckHere) {
                double y = gkResult - comp;
                double t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                errAccum += gkAbsErr;
                panelCount++;
                resolvedWidth += width;
                progressListener.onProgress(panelCount, maxDepthSeen,
                        negate ? -(sum + correction) : (sum + correction), resolvedWidth / fullWidth);
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
                double t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                errAccum += gkAbsErr;
                panelCount++;
                resolvedWidth += width;
                progressListener.onProgress(panelCount, maxDepthSeen,
                        negate ? -(sum + correction) : (sum + correction), resolvedWidth / fullWidth);
                continue;
            }

            double y = gkResult - comp;
            double t = sum + y;
            comp = (t - sum) - y;
            sum = t;
            errAccum += gkAbsErr;
            panelCount++;
            resolvedWidth += width;
            progressListener.onProgress(panelCount, maxDepthSeen,
                    negate ? -(sum + correction) : (sum + correction), resolvedWidth / fullWidth);
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
                lastCrossCheckCount, lastForcedSplitCount, getLastSingularitySubtractionInfo(),
                new ArrayList<>(singularities));
    }

    public double getLastErrorEstimate() {
        return lastErrorEstimate;
    }

    public int getLastPanelCount() {
        return lastPanelCount;
    }

    public int getLastMaxDepth() {
        return lastMaxDepthReached;
    }

    public int getLastCrossCheckCount() {
        return lastCrossCheckCount;
    }

    public int getLastForcedSplitCount() {
        return lastForcedSplitCount;
    }

    public boolean isStrictSingularities() {
        return strictSingularities;
    }

    public String getLastSingularitySubtractionInfo() {
        if (!leftActive && !rightActive) {
            return null;
        }
        StringBuilder info = new StringBuilder();
        if (leftActive) {
            info.append(String.format("lower bound n=%.6g C=%.6g", leftN, leftC));
        }
        if (rightActive) {
            if (info.length() > 0) {
                info.append("; ");
            }
            info.append(String.format("upper bound n=%.6g C=%.6g", rightN, rightC));
        }
        return info.toString();
    }

    public List<Singularity> getSingularities() {
        return Collections.unmodifiableList(singularities);
    }

    private void recordSingularityOrThrow(double pa, double pb, String reason, Throwable cause) {
        double location = 0.5 * (pa + pb);
        double width = pb - pa;
        if (strictSingularities) {
            throw new NonIntegrableSingularityException(location, width, reason, cause);
        }
        LOG.warning(String.format(
                "Unresolved panel near x=%.15g (width %.3e): %s -- lenient mode", location, width, reason));
        singularities.add(new Singularity(location, width, reason));
    }

    // ------------------------------------------------------------------
    // Endpoint power-law singularity probing (zero-alloc scratch fields)
    // ------------------------------------------------------------------
    private boolean probeSinglePointExponent(double x0, double s, double dir) {
        probeDomainFailure = false;
        double probeX = x0 + dir * s;

        try {
            ad.taylorCoefficients(wrtVar, probeX, 2, probeCoeffs);
        } catch (ArithmeticException domainEx) {
            probeDomainFailure = true;
            return false;
        }
        double f = probeCoeffs[0];
        double fp = probeCoeffs[1];
        double fpp = 2.0 * probeCoeffs[2];

        if (!Double.isFinite(f) || !Double.isFinite(fp) || !Double.isFinite(fpp) || f == 0.0 || fp == 0.0) {
            probeDomainFailure = true;
            return false;
        }

        double t = dir * s;
        double n1 = t * fp / f;
        double n2 = 1.0 + t * fpp / fp;
        if (!Double.isFinite(n1) || !Double.isFinite(n2)) {
            probeDomainFailure = true;
            return false;
        }
        if (Math.abs(n1 - n2) > singularityConsistencyTol) {
            return false;
        }
        double n = 0.5 * (n1 + n2);

        if (n <= -1.0) {
            probeN = n;
            probeC = Double.NaN;
            return true;
        }
        if (Math.abs(n) < 0.05) {
            return false;
        }
        double c = f / Math.pow(s, n);
        if (!Double.isFinite(c)) {
            probeDomainFailure = true;
            return false;
        }
        probeN = n;
        probeC = c;
        return true;
    }

    private boolean probeEndpointSingularityTwoScale(double x0, double intervalWidth, double dir) {
        double sNear = Math.max(intervalWidth * 1e-4, 1e-12);
        if (!probeSinglePointExponent(x0, sNear, dir)) {
            return false;
        }
        double nNear = probeN;
        double cNear = probeC;

        if (nNear <= -1.0) {
            probeN = nNear;
            probeC = cNear;
            return true;
        }

        double sFar = Math.min(sNear * singularityScaleFactor, intervalWidth * 0.4);
        boolean farOk = probeSinglePointExponent(x0, sFar, dir);
        if (farOk) {
            if (Math.abs(nNear - probeN) > crossScaleConsistencyTol) {
                return false;
            }
        } else if (!probeDomainFailure) {
            return false;
        }

        probeN = nNear;
        probeC = cNear;
        return true;
    }

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

        if (!sampleF(centr)) {
            return false;
        }
        double fc = fnBuf[0];
        maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);

        double resg = WG[3] * fc;
        double resk = WGK[7] * fc;
        double resabs = Math.abs(resk);

        for (int j = 0; j < 3; j++) {
            int jtw = 2 * (j + 1) - 1;
            double absc = hlgth * XGK[jtw];
            if (!sampleF(centr - absc)) {
                return false;
            }
            double f1 = fnBuf[0];
            maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);
            if (!sampleF(centr + absc)) {
                return false;
            }
            double f2 = fnBuf[0];
            maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);
            fv1[jtw] = f1;
            fv2[jtw] = f2;
            double fsum = f1 + f2;
            resg += WG[j] * fsum;
            resk += WGK[jtw] * fsum;
            resabs += WGK[jtw] * (Math.abs(f1) + Math.abs(f2));
        }
        for (int j = 0; j < 4; j++) {
            int jtwm1 = 2 * (j + 1) - 2;
            double absc = hlgth * XGK[jtwm1];
            if (!sampleF(centr - absc)) {
                return false;
            }
            double f1 = fnBuf[0];
            maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);
            if (!sampleF(centr + absc)) {
                return false;
            }
            double f2 = fnBuf[0];
            maxSubtractedMag = Math.max(maxSubtractedMag, lastSubtractedMagnitude);
            fv1[jtwm1] = f1;
            fv2[jtwm1] = f2;
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

        double cancellationFloor = EPMACH * CANCELLATION_FLOOR_FACTOR * maxSubtractedMag;
        if (cancellationFloor > abserr) {
            abserr = cancellationFloor;
        }
        lastCancellationFloor = cancellationFloor;
        lastPanelMaxSubtractedMag = maxSubtractedMag;

        gkResult = result;
        gkAbsErr = abserr;
        return true;
    }

    //FALLBACK
    /**
     * Taylor-panel rescue: a genuine quadrature estimate (not just a comparison
     * probe) at the panel midpoint, valid even arbitrarily close to a
     * subtracted endpoint singularity, because the singular term's contribution
     * is removed via a closed-form binomial series rather than by re-evaluating
     * f near x0.
     */
    private boolean computeTaylorRescue(double a, double b) {
        double m = 0.5 * (a + b);
        double h = 0.5 * (b - a);
        double h2 = h * h;

        try {
            ad.taylorCoefficients(wrtVar, m, crossCheckAdOrder, ccCoeffs);
        } catch (ArithmeticException domainEx) {
            return false; // the center itself is bad; rescue can't help here
        }
        for (double c : ccCoeffs) {
            if (!Double.isFinite(c)) {
                return false;
            }
        }

        if (leftActive) {
            subtractBinomialSeries(m - leftX0, leftN, leftC, +1);
        }
        if (rightActive) {
            subtractBinomialSeries(rightX0 - m, rightN, rightC, -1);
        }
        for (double c : ccCoeffs) {
            if (!Double.isFinite(c)) {
                return false; // guard: pow(d,n) could itself overflow for tiny d
            }
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

        rescueValue = sum;
        rescueErrEst = Math.abs(probeTerm);
        rescueConverging = converging;
        return Double.isFinite(sum) && Double.isFinite(rescueErrEst);
    }

    /**
     * Subtracts C*(dist)^n's Taylor series (about the panel midpoint) from
     * ccCoeffs in place.
     */
    private void subtractBinomialSeries(double d, double n, double c, int sign) {
        binomBuf[0] = 1.0;
        double dPow = Math.pow(d, n); // d^(n-k), k=0 term
        for (int k = 0; k <= crossCheckAdOrder; k++) {
            if (k > 0) {
                binomBuf[k] = binomBuf[k - 1] * (n - k + 1) / k;
                dPow /= d;
            }
            double coeff = c * binomBuf[k] * dPow;
            if (sign < 0 && (k & 1) == 1) {
                coeff = -coeff;
            }
            ccCoeffs[k] -= coeff;
        }
    }
//FALLBACK ENDS

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
            if (!Double.isFinite(c)) {
                return false;
            }
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
        if (ok) {
            passCount++;
        } else {
            failCount++;
        }
    }

    public static void main(String[] args) {
        IntegratorX expI = builder().build("exp(x)", "x");
        expect("exp(x) on [0,10]", expI.integrate(0, 10), Math.exp(10) - 1, 1e-8);

        // Default (unlimited crossCheckMaxDepth) must still catch the bump -- proving the new
        // opt-in depth cap didn't quietly weaken the default safety property.
        IntegratorX bumpDefault = builder()
                .absoluteTolerance(1e-10).relativeTolerance(1e-10)
                .build("exp(-3200*(x-1.3)^2)", "x");
        double bumpTrue = 0.03133285343288750;
        expect("Gaussian bump, default crossCheckMaxDepth (still caught)", bumpDefault.integrate(0, 2), bumpTrue, 1e-8);

        // Explicit opt-in: crossCheckMaxDepth=0 means only the root panel gets cross-checked.
        // This is the caller's deliberate choice to trade the guarantee away, not a default.
        IntegratorX bumpShallow = builder()
                .absoluteTolerance(1e-10).relativeTolerance(1e-10)
                .crossCheckMaxDepth(0)
                .build("exp(-3200*(x-1.3)^2)", "x");
        double bumpShallowResult = bumpShallow.integrate(0, 2);//
        System.out.println("[INFO ] Gaussian bump, crossCheckMaxDepth=0 (opt-in, reduced safety): got="
                + bumpShallowResult + " -- illustrates the explicit tradeoff, not asserted correct or incorrect");

        IntegratorX sqrtI = builder().build("sqrt(x)", "x");
        expect("sqrt(x) on [0,1] (n=0.5 endpoint)", sqrtI.integrate(0, 1), 2.0 / 3.0, 1e-9);

        IntegratorX mixedI = builder().build("1/sqrt(x) + cos(x)", "x");
        expect("1/sqrt(x)+cos(x) on [0,1] (cancellation-floor case)",
                mixedI.integrate(0, 1), 2.0 + Math.sin(1.0), 1e-8);

        IntegratorX sinInvX = builder().build("1/sin(x)", "x");
        expect("1/sin(x) on [0.01,0.02] (cancellation-floor case)",
                sinInvX.integrate(0.01, 0.02), 0.6931721812891335, 1e-8);
        System.out.println("intg(1/sin(x), 0.1, 1.2) -> " + sinInvX.integrate(0.1, 1.2));
  
        expect("1/sin(x) on [0.0000001, 1.5] (cancellation-floor case)",
                sinInvX.integrate(0.0000001, 1.5), 15.5458014523992019, 1e-14);

        // Precision-floor acceptance: an intentionally unreachable tolerance must still succeed
        // in strict mode (not throw), by falling through to the new floor-acceptance path.
        IntegratorX impossibleTol = builder()
                .absoluteTolerance(1e-20).relativeTolerance(1e-20)
                .build("1/sqrt(x) + cos(x)", "x");
        try {
            double v = impossibleTol.integrate(0, 1);
            double err = Math.abs(v - (2.0 + Math.sin(1.0)));
            boolean ok = err < 1e-6; // can't hit 1e-20 near this singularity; should still be close
            System.out.printf("[%s] impossible tolerance (1e-20) on 1/sqrt(x)+cos(x): got=%.10g err=%.3e"
                    + " -- accepted at precision floor instead of throwing, flags=%d%n",
                    ok ? "PASS" : "FAIL", v, err, impossibleTol.getSingularities().size());
            if (ok) {
                passCount++;
            } else {
                failCount++;
            }
        } catch (NonIntegrableSingularityException ex) {
            System.out.println("[FAIL] impossible tolerance should have been accepted at the precision floor, not thrown: " + ex.getMessage());
            failCount++;
        }

        IntegratorX cbrtI = builder().build("1/(x^(1/3))", "x");
        expect("1/x^(1/3) on [0,1] (n=-1/3 endpoint)", cbrtI.integrate(0, 1), 1.5, 1e-8);

        IntegratorX divergent = builder().build("1/x^1.5", "x");
        try {
            divergent.integrate(0, 1);
            System.out.println("[FAIL] 1/x^1.5 on [0,1] should have thrown (non-integrable)");
            failCount++;
        } catch (NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/x^1.5 on [0,1] correctly threw: " + expectedEx.getMessage());
            passCount++;
        }
        try {
            IntegratorX contaminated = builder().build("1/sqrt(x) + 5000*x", "x");
            double contamTrue = 2.0 + 2500.0;
            expect("1/sqrt(x)+5000*x on [0,1] (two-scale-rejected, plain bisection instead)",
                    contaminated.integrate(0, 1), contamTrue, 1e-6);
        } catch (NonIntegrableSingularityException e) {
            e.printStackTrace();
        }
        IntegratorX poleStrict = builder().build("1/(x-0.5)", "x");
        try {
            poleStrict.integrate(0.1, 0.9);
            System.out.println("[FAIL] 1/(x-0.5) through pole (strict) should have thrown");
            failCount++;
        } catch (NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/(x-0.5) through pole (strict) correctly threw: " + expectedEx.getMessage());
            passCount++;
        }

        int[] callCount = {0};
        double[] lastFraction = {-1.0};
        boolean[] monotonic = {true};
        IntegratorX progressI = builder()
                .progressListener((panels, depth, est, frac) -> {
                    callCount[0]++;
                    if (frac < lastFraction[0]) {
                        monotonic[0] = false;
                    }
                    lastFraction[0] = frac;
                })
                .build("exp(-3200*(x-1.3)^2)", "x");
        progressI.integrate(0, 2);
        boolean progressOk = callCount[0] == progressI.getLastPanelCount() && callCount[0] > 0
                && monotonic[0] && Math.abs(lastFraction[0] - 1.0) < 1e-9;
        System.out.printf("[%s] progress listener: %d calls, monotonic=%b, final fraction=%.10f%n",
                progressOk ? "PASS" : "FAIL", callCount[0], monotonic[0], lastFraction[0]);
        if (progressOk) {
            passCount++;
        } else {
            failCount++;
        }

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
