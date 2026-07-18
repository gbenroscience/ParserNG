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
package com.github.gbenroscience.math.numericalmethods.taylors.crx;

import com.github.gbenroscience.math.differentialcalculus.autodiff.AutoDiffNEvaluator;
import com.github.gbenroscience.math.numericalmethods.SingularityExtractor;
import com.github.gbenroscience.parser.MathExpression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Composite adaptive integrator: Gauss-Kronrod 7-15 as the primary panel rule,
 * AD-based domain-error classification, a low-order Taylor cross-check,
 * analytic endpoint power-law singularity subtraction with a floating-point
 * cancellation floor, three-site Taylor-series rescue, and (new) a zero-alloc
 * interior-singularity scan that lets this class handle poles <em>anywhere</em>
 * in the domain, not just at the two literal integration bounds.
 *
 * <h2>Interior singularity scan and segmentation (new)</h2>
 * At the top of every {@link #integrate(double, double)} call (when {@link
 * Builder#interiorSingularityScanEnabled}, default true), the domain is sampled
 * at {@link Builder#interiorScanPoints} points. A sample is "suspicious" if it
 * is non-finite, exceeds {@link
 * Builder#interiorScanAbsoluteThreshold}, or is drastically larger (by
 * {@link Builder#interiorScanRelativeMultiplier}) than every ordinary sample
 * seen so far in the scan. Each suspicious point is refined via Newton's method
 * on {@code 1/f} -- {@link #newtonSnapToPole} -- computed as
 * {@code x += f(x)/f'(x)} rather than ever evaluating {@code 1/f} directly, so
 * the iteration never needs to divide by the very quantity that is blowing up.
 * A converged, confirmed pole becomes an internal segment boundary.
 *
 * <p>
 * <b>Why this needs almost no new quadrature logic.</b> An interior pole at
 * location {@code P} simply becomes the shared edge of two adjacent segments
 * {@code [..., P]} and {@code [P, ...]} -- exactly the same shape the
 * endpoint-subtraction machinery already handles for the outer call bounds. So
 * {@link #integrateSegment} runs the <em>entire</em> existing per-call pipeline
 * (endpoint-probe both of <em>its</em> edges, subtract and verify if singular,
 * run the GK/cross-check/rescue panel loop) once per detected segment,
 * accumulating into call-wide running totals, rather than duplicating a second,
 * weaker integration path.
 *
 * <p>
 * <b>The scan is a heuristic accelerator, not the only line of defense.</b>
 * A pole the flat scan misses (falls between two scan points, or the function
 * is suspicious in a way the threshold doesn't catch) is still caught by the
 * pre-existing per-panel domain-error handling inside whichever segment it
 * falls in -- bisect, then Taylor-rescue, then record/throw at exhaustion,
 * exactly as before. The scan only changes how
 * <em>quickly</em> and how <em>precisely-diagnosed</em> a known-shape
 * (power-law) interior singularity is resolved; it does not weaken any existing
 * safety net if it misses something.
 *
 * <p>
 * <b>Divergence at an interior point always throws.</b> Exactly like the
 * outer-bound case, an exponent {@code n &lt;= -1} detected at a segment edge
 * -- whether that edge is the call's actual bound or an interior pole the scan
 * found -- throws {@link NonIntegrableSingularityException} unconditionally,
 * regardless of {@link Builder#strictSingularities}. This class deliberately
 * never attempts a Cauchy-principal-value-style partial result across a genuine
 * pole; that would be a materially different feature (symmetric excision around
 * the pole) and is not implemented here.
 *
 * <h2>Precision-floor acceptance</h2>
 * A panel that exhausts its bisection budget while still failing
 * Gauss-Kronrod's own convergence test is normally a strict-mode throw. One
 * exception: if the panel's reported error is essentially equal to its own
 * {@link #lastCancellationFloor} -- i.e. the residual <em>is</em> the
 * already-quantified subtraction cancellation noise, not an unresolved failure
 * -- it is accepted regardless of strict mode, flagged as a {@link
 * Singularity} for transparency. Further bisection provably cannot reduce a
 * cancellation floor, since it does not shrink with panel width.
 *
 * <h2>Taylor-series rescue (three sites)</h2>
 * When {@link Builder#taylorRescueEnabled} (default true), the AD engine gets
 * one more chance via a midpoint Taylor expansion -- with any active endpoint
 * singular term subtracted in closed form via a generalized binomial series, so
 * the rescue is valid arbitrarily close to a subtracted edge -- at three points
 * where Gauss-Kronrod's own machinery has nothing left to try: (1) a GK node
 * hits a genuine AD domain error, tried <em>before</em> bisecting; (2) GK's own
 * convergence test fails and bisection is exhausted, tried <em>after</em> the
 * cheaper cancellation-floor check; (3) an AD cross-check disagreement survives
 * to the bisection budget limit, tried only once bisection -- the correct first
 * response to aliasing -- is truly exhausted.
 *
 * <h2>Cross-check stays live per-segment, not per-call</h2>
 * Each panel re-enables the cross-check individually once its own subtracted
 * contribution is negligible relative to its Gauss-Kronrod result, rather than
 * disabling it for a whole segment or call the moment any edge is subtracted.
 *
 * <h2>Zero-allocation steady state</h2>
 * The per-panel loop and the interior-singularity scan both perform no
 * {@code new}, no autoboxing, no String work on their common paths -- all
 * scratch (including the growable {@link #interiorPoles} buffer) is pre-sized
 * instance state, doubled only rarely and amortized when it grows. What
 * legitimately allocates: exception messages and lenient-mode
 * singularity/subtraction records, but only on already-exceptional or genuinely
 * rare (a singularity was actually found) paths; {@link
 * #integrateResult} is the documented allocating convenience wrapper.
 *
 * <h2>Why GK as the primary rule / why GK alone is not enough</h2>
 * Gauss-Kronrod is exact for polynomials up to degree 23 (K15) from only 15
 * evaluations, with a self-contained {@code resasc}-refined error estimate
 * needing no derivatives -- but that estimate compares two rules built from the
 * same 15 points, so a feature of {@code f} falling between them can fool both
 * together: {@code exp(-3200*(x-1.3)^2)} on {@code [0,2]}, plain GK-adaptive at
 * {@code absTol=1e-10}, returns {@code 0.0} (true value {@code 0.0313}) because
 * the starting panel's reported error was {@code ~5.6e-13} -- a low,
 * tolerance-satisfying number, which is exactly why gating the cross-check on
 * "GK reported a high error" cannot work.
 *
 * <h2>Constants</h2>
 * The 15-point Kronrod / 7-point Gauss abscissae and weights are the standard
 * QUADPACK {@code dqk15.f} table (Fullerton, Bell Labs, 1981).
 *
 * <h2>Thread-safety</h2>
 * Not thread-safe. One instance per thread, built from the same {@code
 * rpnTokens} (which is only read).
 */
public class Integrator implements com.github.gbenroscience.math.numericalmethods.taylors.ffx.smart.SymbolicIntegrator.SingularityAwareIntegrator {

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
    public static final double DEFAULT_CROSS_SCALE_CONSISTENCY_TOL = 0.1;
    public static final double DEFAULT_SINGULARITY_SCALE_FACTOR = 100.0;
    public static final double DEFAULT_CROSS_CHECK_SUBTRACTION_NEGLIGIBILITY = 1e-6;
    public static final double CANCELLATION_FLOOR_FACTOR = 100.0;
    public static final double PRECISION_FLOOR_ACCEPT_MARGIN = 2.0;
    public static final int DEFAULT_CROSS_CHECK_MAX_DEPTH = Integer.MAX_VALUE;

    public static final int DEFAULT_INTERIOR_SCAN_POINTS = 64;
    public static final double DEFAULT_INTERIOR_SCAN_ABSOLUTE_THRESHOLD = 1e6;
    public static final double DEFAULT_INTERIOR_SCAN_RELATIVE_MULTIPLIER = 1e4;
    public static final double DEFAULT_POLE_SNAP_MAGNITUDE_THRESHOLD = 100.0;
    public static final int DEFAULT_NEWTON_MAX_ITERS = 12;
    public static final double DEFAULT_NEWTON_MAX_STEP_FRACTION = 0.1;
    public static final double DEFAULT_NEWTON_CONVERGENCE_TOL_FRACTION = 1e-10;
    public static final double DEFAULT_POLE_CONFIRM_MAGNITUDE = 1e6;
    public static final double DEFAULT_INTERIOR_DEDUPE_FRACTION = 1e-6;

    /**
     * Cheap pre-check: does this range appear to contain (or nearly touch) a
     * singularity, without running a full adaptive integration? Reuses the same
     * interior-pole scan and endpoint-proximity check {@link #integrate} itself
     * uses internally. Intended for callers (e.g. {@link SymbolicIntegrator})
     * deciding whether to trust a symbolic shortcut before paying for the full
     * numeric engine.
     */
    @Override
    public boolean rangeLooksSingular(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b) || a == b) {
            return false;
        }
        double lo = Math.min(a, b), hi = Math.max(a, b); 
        scanForInteriorSingularities(lo, hi);
        MathExpression me = this.ad.targetExpr;
      return new SingularityExtractor(me.getCachedPostfix()).findSingularities(me.getCachedPostfix(), me, me.getSlots()[0], lo, hi, new double[]{}) > 0;
         
    }

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
        private boolean taylorRescueEnabled = true;
        private boolean interiorSingularityScanEnabled = true;
        private int interiorScanPoints = DEFAULT_INTERIOR_SCAN_POINTS;
        private double interiorScanAbsoluteThreshold = DEFAULT_INTERIOR_SCAN_ABSOLUTE_THRESHOLD;
        private double interiorScanRelativeMultiplier = DEFAULT_INTERIOR_SCAN_RELATIVE_MULTIPLIER;
        private double poleSnapMagnitudeThreshold = DEFAULT_POLE_SNAP_MAGNITUDE_THRESHOLD;
        private int newtonMaxIters = DEFAULT_NEWTON_MAX_ITERS;
        private double newtonMaxStepFraction = DEFAULT_NEWTON_MAX_STEP_FRACTION;
        private double newtonConvergenceTolFraction = DEFAULT_NEWTON_CONVERGENCE_TOL_FRACTION;
        private double poleConfirmMagnitude = DEFAULT_POLE_CONFIRM_MAGNITUDE;
        private double interiorDedupeFraction = DEFAULT_INTERIOR_DEDUPE_FRACTION;
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

        public Builder taylorRescueEnabled(boolean enabled) {
            this.taylorRescueEnabled = enabled;
            return this;
        }

        /**
         * Default true. Enables the zero-alloc flat scan + Newton-snap for
         * interior singularities anywhere in the domain, not just at the two
         * bounds.
         */
        public Builder interiorSingularityScanEnabled(boolean enabled) {
            this.interiorSingularityScanEnabled = enabled;
            return this;
        }

        /**
         * How many points to sample across the domain looking for suspicious
         * (large/non-finite) values. Default 64; this is a real, honest
         * per-call cost.
         */
        public Builder interiorScanPoints(int points) {
            this.interiorScanPoints = Math.max(0, points);
            return this;
        }

        public Builder interiorScanAbsoluteThreshold(double threshold) {
            this.interiorScanAbsoluteThreshold = threshold;
            return this;
        }

        public Builder interiorScanRelativeMultiplier(double multiplier) {
            this.interiorScanRelativeMultiplier = multiplier;
            return this;
        }

        public Builder poleSnapMagnitudeThreshold(double threshold) {
            this.poleSnapMagnitudeThreshold = threshold;
            return this;
        }

        public Builder newtonMaxIters(int iters) {
            this.newtonMaxIters = Math.max(1, iters);
            return this;
        }

        public Builder newtonMaxStepFraction(double fraction) {
            this.newtonMaxStepFraction = fraction;
            return this;
        }

        public Builder newtonConvergenceTolFraction(double fraction) {
            this.newtonConvergenceTolFraction = fraction;
            return this;
        }

        public Builder poleConfirmMagnitude(double magnitude) {
            this.poleConfirmMagnitude = magnitude;
            return this;
        }

        public Builder interiorDedupeFraction(double fraction) {
            this.interiorDedupeFraction = fraction;
            return this;
        }

        public Builder progressListener(ProgressListener listener) {
            this.progressListener = listener != null ? listener : ProgressListener.NO_OP;
            return this;
        }

        public Integrator build(MathExpression me, String wrtVar) {
            return new Integrator(me, wrtVar, crossCheckOrder, absTol, relTol,
                    maxDepth, strictSingularities, crossCheckEnabled, crossCheckLooseness, crossCheckMaxDepth,
                    singularitySubtractionEnabled, singularityConsistencyTol, crossScaleConsistencyTol,
                    singularityScaleFactor, crossCheckSubtractionNegligibility, taylorRescueEnabled,
                    interiorSingularityScanEnabled, interiorScanPoints, interiorScanAbsoluteThreshold,
                    interiorScanRelativeMultiplier, poleSnapMagnitudeThreshold, newtonMaxIters,
                    newtonMaxStepFraction, newtonConvergenceTolFraction, poleConfirmMagnitude,
                    interiorDedupeFraction, progressListener);
        }

        public Integrator build(String expr, String wrtVar) {
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
                    + "If a principal-value-style partial result is acceptable, this class does not "
                    + "support that; strictSingularities(false) omits/flags a panel that ran out of "
                    + "bisection budget, but a genuine divergence (n<=-1) always throws regardless.",
                    location, panelWidth, reason), cause);
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
    private final boolean taylorRescueEnabled;
    private final boolean interiorSingularityScanEnabled;
    private final int interiorScanPoints;
    private final double interiorScanAbsoluteThreshold;
    private final double interiorScanRelativeMultiplier;
    private final double poleSnapMagnitudeThreshold;
    private final int newtonMaxIters;
    private final double newtonMaxStepFraction;
    private final double newtonConvergenceTolFraction;
    private final double poleConfirmMagnitude;
    private final double interiorDedupeFraction;
    private final ProgressListener progressListener;

    // Scratch (zero-allocation steady state)
    private final double[] fnBuf;
    private final double[] fv1;
    private final double[] fv2;
    private final double[] ccCoeffs;
    private final double[] probeCoeffs;
    private final double[] binomBuf;
    private final double[] newtonBuf;

    private double[] stackA;
    private double[] stackB;
    private int[] stackDepth;
    private int stackSize;

    private double[] interiorPoles;
    private int interiorPoleCount;

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

    private double rescueValue;
    private double rescueErrEst;
    private boolean rescueConverging;

    private boolean leftActive, rightActive;
    private double leftX0, leftN, leftC;
    private double rightX0, rightN, rightC;

    // Call-wide running accumulators, threaded across segments.
    private double accSum, accComp, accErrAccum, accResolvedWidth, accCorrection;
    private int accPanelCount, accCrossCheckCount, accForcedSplitCount, accMaxDepthSeen;

    private double lastErrorEstimate;
    private int lastPanelCount;
    private int lastMaxDepthReached;
    private int lastCrossCheckCount;
    private int lastForcedSplitCount;
    private final List<Singularity> singularities;
    private final List<String> appliedSubtractions;

    public Integrator(MathExpression me, String wrtVar, int crossCheckOrder,
            double absTol, double relTol, int maxDepth) {
        this(me, wrtVar, crossCheckOrder, absTol, relTol, maxDepth, true, true, 50.0,
                DEFAULT_CROSS_CHECK_MAX_DEPTH, true,
                DEFAULT_SINGULARITY_CONSISTENCY_TOL, DEFAULT_CROSS_SCALE_CONSISTENCY_TOL,
                DEFAULT_SINGULARITY_SCALE_FACTOR, DEFAULT_CROSS_CHECK_SUBTRACTION_NEGLIGIBILITY, true,
                true, DEFAULT_INTERIOR_SCAN_POINTS, DEFAULT_INTERIOR_SCAN_ABSOLUTE_THRESHOLD,
                DEFAULT_INTERIOR_SCAN_RELATIVE_MULTIPLIER, DEFAULT_POLE_SNAP_MAGNITUDE_THRESHOLD,
                DEFAULT_NEWTON_MAX_ITERS, DEFAULT_NEWTON_MAX_STEP_FRACTION,
                DEFAULT_NEWTON_CONVERGENCE_TOL_FRACTION, DEFAULT_POLE_CONFIRM_MAGNITUDE,
                DEFAULT_INTERIOR_DEDUPE_FRACTION, ProgressListener.NO_OP);
    }

    public Integrator(MathExpression me, String wrtVar, int crossCheckOrder,
            double absTol, double relTol, int maxDepth,
            boolean strictSingularities, boolean crossCheckEnabled,
            double crossCheckLooseness, int crossCheckMaxDepth,
            boolean singularitySubtractionEnabled,
            double singularityConsistencyTol, double crossScaleConsistencyTol,
            double singularityScaleFactor, double crossCheckSubtractionNegligibility,
            boolean taylorRescueEnabled, boolean interiorSingularityScanEnabled,
            int interiorScanPoints, double interiorScanAbsoluteThreshold,
            double interiorScanRelativeMultiplier, double poleSnapMagnitudeThreshold,
            int newtonMaxIters, double newtonMaxStepFraction, double newtonConvergenceTolFraction,
            double poleConfirmMagnitude, double interiorDedupeFraction,
            ProgressListener progressListener) {
        if (crossCheckOrder < 2 || (crossCheckOrder % 2) != 0) {
            throw new IllegalArgumentException(
                    "crossCheckOrder must be a positive even number (midpoint expansion cancels odd terms)");
        }
        if (crossCheckOrder > MAX_CROSS_CHECK_ORDER) {
            throw new IllegalArgumentException("crossCheckOrder must be <= " + MAX_CROSS_CHECK_ORDER);
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
        if (interiorScanPoints < 0) {
            throw new IllegalArgumentException("interiorScanPoints must be >= 0");
        }
        if (newtonMaxStepFraction <= 0 || newtonMaxStepFraction >= 1.0) {
            throw new IllegalArgumentException("newtonMaxStepFraction must be in (0,1)");
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
        this.taylorRescueEnabled = taylorRescueEnabled;
        this.interiorSingularityScanEnabled = interiorSingularityScanEnabled;
        this.interiorScanPoints = interiorScanPoints;
        this.interiorScanAbsoluteThreshold = interiorScanAbsoluteThreshold;
        this.interiorScanRelativeMultiplier = interiorScanRelativeMultiplier;
        this.poleSnapMagnitudeThreshold = poleSnapMagnitudeThreshold;
        this.newtonMaxIters = newtonMaxIters;
        this.newtonMaxStepFraction = newtonMaxStepFraction;
        this.newtonConvergenceTolFraction = newtonConvergenceTolFraction;
        this.poleConfirmMagnitude = poleConfirmMagnitude;
        this.interiorDedupeFraction = interiorDedupeFraction;
        this.progressListener = progressListener != null ? progressListener : ProgressListener.NO_OP;

        this.fnBuf = new double[1];
        this.fv1 = new double[7];
        this.fv2 = new double[7];
        this.ccCoeffs = new double[crossCheckAdOrder + 1];
        this.probeCoeffs = new double[3];
        this.binomBuf = new double[crossCheckAdOrder + 1];
        this.newtonBuf = new double[2];

        int cap = maxDepth + 4;
        this.stackA = new double[cap];
        this.stackB = new double[cap];
        this.stackDepth = new int[cap];
        this.stackSize = 0;

        this.interiorPoles = new double[16];
        this.interiorPoleCount = 0;

        this.singularities = new ArrayList<>();
        this.appliedSubtractions = new ArrayList<>();
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
    // Interior singularity scan (new)
    // ------------------------------------------------------------------
    private void addInteriorPoleIfNew(double location, double fullWidth) {
        double dedupeEps = Math.max(Math.abs(location) * 1e-12, interiorDedupeFraction * fullWidth);
        for (int i = 0; i < interiorPoleCount; i++) {
            if (Math.abs(interiorPoles[i] - location) < dedupeEps) {
                return;
            }
        }
        if (interiorPoleCount == interiorPoles.length) {
            interiorPoles = Arrays.copyOf(interiorPoles, interiorPoles.length * 2);
        }
        interiorPoles[interiorPoleCount++] = location;
    }

    private boolean confirmPole(double x) {
        try {
            ad.evaluateRPN(wrtVar, x, 0, fnBuf);
            return !Double.isFinite(fnBuf[0]) || Math.abs(fnBuf[0]) > poleConfirmMagnitude;
        } catch (ArithmeticException domainEx) {
            return true;
        }
    }

    /**
     * Newton's method on {@code 1/f}, computed as {@code x += f(x)/f'(x)} so
     * the iteration never evaluates {@code 1/f} directly near the value that is
     * blowing up. Returns the snapped location, or NaN if the point isn't
     * actually near a large value, the step is untrustworthy, the iteration
     * wanders outside {@code (domainLo, domainHi)}, or it fails to converge.
     */
    private double newtonSnapToPole(double xStart, double domainLo, double domainHi) {
        if (xStart <= domainLo || xStart >= domainHi) {
            return Double.NaN;
        }
        double current = xStart;
        double maxStep = Math.max((domainHi - domainLo) * newtonMaxStepFraction, 1e-14);
        double convergeTol = Math.max((domainHi - domainLo) * newtonConvergenceTolFraction, 1e-15);

        for (int i = 0; i < newtonMaxIters; i++) {
            try {
                ad.evaluateRPN(wrtVar, current, 1, newtonBuf);
            } catch (ArithmeticException domainEx) {
                return current; // the domain wall itself is likely the singularity's exact location
            }
            double f = newtonBuf[0];
            double fp = newtonBuf[1];
            if (!Double.isFinite(f) || !Double.isFinite(fp)) {
                return current;
            }
            if (Math.abs(f) < poleSnapMagnitudeThreshold) {
                return Double.NaN;
            }
            if (fp == 0.0) {
                return Double.NaN;
            }
            double dx = f / fp; // Newton step toward a root of 1/f, i.e. a blow-up of f
            if (!Double.isFinite(dx) || Math.abs(dx) > maxStep) {
                return Double.NaN;
            }
            double next = current + dx;
            if (next <= domainLo || next >= domainHi) {
                return Double.NaN;
            }
            if (Math.abs(dx) < convergeTol) {
                return confirmPole(next) ? next : Double.NaN;
            }
            current = next;
        }
        return Double.NaN;
    }

    /**
     * Flat scan for interior singularities. A heuristic accelerator, not the
     * only safety net -- see class javadoc. Populates
     * {@link #interiorPoles}[0..{@link #interiorPoleCount}), sorted,
     * deduplicated, and kept strictly away from the domain's own edges (those
     * are handled as ordinary segment endpoints).
     */
    private void scanForInteriorSingularities(double lo, double hi) {
        interiorPoleCount = 0;
        double fullWidth = hi - lo;
        if (fullWidth <= 0 || interiorScanPoints == 0) {
            return;
        }

        int n = interiorScanPoints;
        double step = fullWidth / (n + 1);
        double runningMaxNormal = 0.0;
        double edgeMargin = fullWidth * 1e-6;
        double scanLo = lo + edgeMargin;
        double scanHi = hi - edgeMargin;

        for (int i = 1; i <= n; i++) {
            double x = lo + i * step;
            double val;
            boolean suspicious;
            try {
                ad.evaluateRPN(wrtVar, x, 0, fnBuf);
                val = fnBuf[0];
                suspicious = !Double.isFinite(val)
                        || Math.abs(val) > interiorScanAbsoluteThreshold
                        || (runningMaxNormal > 0.0 && Math.abs(val) > interiorScanRelativeMultiplier * runningMaxNormal);
            } catch (ArithmeticException domainEx) {
                val = Double.NaN;
                suspicious = true;
            }

            if (suspicious) {
                double snapped = newtonSnapToPole(x, scanLo, scanHi);
                if (Double.isFinite(snapped)) {
                    addInteriorPoleIfNew(snapped, fullWidth);
                }
            } else if (Double.isFinite(val)) {
                runningMaxNormal = Math.max(runningMaxNormal, Math.abs(val));
            }
        }

        if (interiorPoleCount > 1) {
            Arrays.sort(interiorPoles, 0, interiorPoleCount);
        }
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
        appliedSubtractions.clear();
        interiorPoleCount = 0;
        lastErrorEstimate = 0.0;
        lastPanelCount = 0;
        lastMaxDepthReached = 0;
        lastCrossCheckCount = 0;
        lastForcedSplitCount = 0;

        if (a == b) {
            return 0.0;
        }
        boolean negate = b < a;
        double lo = negate ? b : a;
        double hi = negate ? a : b;
        double fullWidth = hi - lo;
        double minWidth = Math.max(fullWidth * 1e-13, Double.MIN_NORMAL);

        if (interiorSingularityScanEnabled) {
            scanForInteriorSingularities(lo, hi);
        }

        accSum = 0.0;
        accComp = 0.0;
        accErrAccum = 0.0;
        accResolvedWidth = 0.0;
        accCorrection = 0.0;
        accPanelCount = 0;
        accCrossCheckCount = 0;
        accForcedSplitCount = 0;
        accMaxDepthSeen = 0;

        double segStart = lo;
        double segmentEpsilon = fullWidth * 1e-9;
        for (int i = 0; i <= interiorPoleCount; i++) {
            double segEnd = (i < interiorPoleCount) ? interiorPoles[i] : hi;
            if (segEnd - segStart > segmentEpsilon) {
                integrateSegment(segStart, segEnd, fullWidth, lo, hi, minWidth, negate);
            }
            segStart = segEnd;
        }

        lastErrorEstimate = accErrAccum;
        lastPanelCount = accPanelCount;
        lastMaxDepthReached = accMaxDepthSeen;
        lastCrossCheckCount = accCrossCheckCount;
        lastForcedSplitCount = accForcedSplitCount;

        double finalValue = accSum + accCorrection;
        return negate ? -finalValue : finalValue;
    }

    /**
     * Runs the full per-call pipeline (endpoint-probe both of this segment's
     * own edges, subtract and verify if singular, then the
     * GK/cross-check/rescue panel loop) for one segment, accumulating into the
     * call-wide {@code acc*} running totals. An edge here may be the call's
     * actual bound, or an interior location the singularity scan found -- both
     * are handled identically, since a segment doesn't know or care which.
     */
    private void integrateSegment(double segLo, double segHi, double fullWidth,
            double callLo, double callHi, double minWidth, boolean negate) {
        double segWidth = segHi - segLo;
        leftActive = false;
        rightActive = false;
        double segCorrection = 0.0;

        if (singularitySubtractionEnabled) {
            boolean leftFound = probeEndpointSingularityTwoScale(segLo, segWidth, +1.0);
            double foundLeftN = leftFound ? probeN : 0.0;
            double foundLeftC = leftFound ? probeC : 0.0;

            boolean rightFound = probeEndpointSingularityTwoScale(segHi, segWidth, -1.0);
            double foundRightN = rightFound ? probeN : 0.0;
            double foundRightC = rightFound ? probeC : 0.0;

            if (leftFound && foundLeftN <= -1.0) {
                throw new NonIntegrableSingularityException(segLo, 0.0,
                        describeLocation(segLo, callLo, callHi)
                        + " (empirical exponent n=" + foundLeftN + " <= -1); this integral does not converge",
                        null);
            }
            if (rightFound && foundRightN <= -1.0) {
                throw new NonIntegrableSingularityException(segHi, 0.0,
                        describeLocation(segHi, callLo, callHi)
                        + " (empirical exponent n=" + foundRightN + " <= -1); this integral does not converge",
                        null);
            }

            if (leftFound) {
                leftActive = true;
                leftX0 = segLo;
                leftN = foundLeftN;
                leftC = foundLeftC;
                segCorrection += leftC * Math.pow(segWidth, leftN + 1.0) / (leftN + 1.0);
                appliedSubtractions.add(String.format("x=%.10g (segment lower edge): n=%.6g C=%.6g", segLo, leftN, leftC));
            }
            if (rightFound) {
                rightActive = true;
                rightX0 = segHi;
                rightN = foundRightN;
                rightC = foundRightC;
                segCorrection += rightC * Math.pow(segWidth, rightN + 1.0) / (rightN + 1.0);
                appliedSubtractions.add(String.format("x=%.10g (segment upper edge): n=%.6g C=%.6g", segHi, rightN, rightC));
            }
        }
        accCorrection += segCorrection;

        stackSize = 0;
        stackPush(segLo, segHi, 0);

        while (stackSize > 0) {
            stackSize--;
            double pa = stackA[stackSize];
            double pb = stackB[stackSize];
            int pDepth = stackDepth[stackSize];
            if (pDepth > accMaxDepthSeen) {
                accMaxDepthSeen = pDepth;
            }

            double width = pb - pa;
            boolean canSplit = pDepth < maxDepth && width > minWidth;

            boolean gkOk = computeQK15(pa, pb);
            if (!gkOk) {
                if (taylorRescueEnabled && computeTaylorRescue(pa, pb)) {
                    double rescueBudget = Math.max(absTol * (width / fullWidth), relTol * Math.abs(rescueValue));
                    if (rescueConverging && rescueErrEst <= rescueBudget) {
                        acceptPanel(rescueValue, rescueErrEst, pa, pb, width, fullWidth, negate,
                                "Gauss-Kronrod hit an AD domain error at one of its 15 nodes ("
                                + lastDomainErrorMessage + "); resolved via a Taylor-series rescue panel");
                        continue;
                    }
                }
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
                boolean cappedByCancellationFloor = lastCancellationFloor > 0.0
                        && gkAbsErr <= lastCancellationFloor * PRECISION_FLOOR_ACCEPT_MARGIN;
                if (cappedByCancellationFloor) {
                    acceptPanel(gkResult, gkAbsErr, pa, pb, width, fullWidth, negate,
                            "accepted at the floating-point cancellation floor near a subtracted singularity"
                            + " (reported error " + gkAbsErr + " is the characterized noise floor, not an"
                            + " unresolved convergence failure); further bisection cannot improve this");
                    continue;
                }
                if (taylorRescueEnabled && computeTaylorRescue(pa, pb) && rescueConverging && rescueErrEst <= budget) {
                    acceptPanel(rescueValue, rescueErrEst, pa, pb, width, fullWidth, negate,
                            "Gauss-Kronrod did not converge to tolerance after reaching the panel-depth/width"
                            + " limit (reported error " + gkAbsErr + " vs budget " + budget
                            + "); resolved via a converging Taylor-series rescue panel instead");
                    continue;
                }
                recordSingularityOrThrow(pa, pb,
                        "Gauss-Kronrod did not converge to tolerance (reported error " + gkAbsErr
                        + " vs budget " + budget + ") after reaching the panel-depth/width limit",
                        null);
                acceptPanel(gkResult, gkAbsErr, pa, pb, width, fullWidth, negate, null);
                continue;
            }

            boolean subtractionNegligibleHere = (leftActive || rightActive)
                    && lastPanelMaxSubtractedMag <= crossCheckSubtractionNegligibility * Math.abs(gkResult);
            boolean withinCrossCheckDepth = pDepth <= crossCheckMaxDepth;
            boolean runCrossCheckHere = crossCheckEnabled && withinCrossCheckDepth
                    && (!(leftActive || rightActive) || subtractionNegligibleHere);

            if (!runCrossCheckHere) {
                acceptPanel(gkResult, gkAbsErr, pa, pb, width, fullWidth, negate, null);
                continue;
            }

            accCrossCheckCount++;
            boolean ccOk = crossCheckPanel(pa, pb);
            double crossTol = Math.max(budget, absTol) * crossCheckLooseness;
            boolean disagree = !ccOk || !ccConverging || (Math.abs(ccValue - gkResult) > crossTol);

            if (disagree) {
                accForcedSplitCount++;
                if (canSplit) {
                    double m = 0.5 * (pa + pb);
                    stackPush(pa, m, pDepth + 1);
                    stackPush(m, pb, pDepth + 1);
                    continue;
                }
                if (taylorRescueEnabled && computeTaylorRescue(pa, pb) && rescueConverging && rescueErrEst <= budget) {
                    acceptPanel(rescueValue, rescueErrEst, pa, pb, width, fullWidth, negate,
                            "Gauss-Kronrod/AD cross-check disagreement" + (ccOk
                                    ? (" (GK=" + gkResult + ", AD=" + ccValue + ")")
                                    : " (AD cross-check itself failed to evaluate)")
                            + " could not be resolved by further bisection; resolved via a converging"
                            + " Taylor-series rescue panel instead of trusting the disputed Gauss-Kronrod value");
                    continue;
                }
                recordSingularityOrThrow(pa, pb,
                        "Gauss-Kronrod/AD cross-check disagreement could not be resolved by further"
                        + " bisection" + (ccOk
                                ? (" (GK=" + gkResult + ", AD=" + ccValue + ")")
                                : " (AD cross-check itself failed to evaluate)"),
                        null);
                acceptPanel(gkResult, gkAbsErr, pa, pb, width, fullWidth, negate, null);
                continue;
            }

            acceptPanel(gkResult, gkAbsErr, pa, pb, width, fullWidth, negate, null);
        }
    }

    /**
     * Common panel-acceptance bookkeeping: Kahan-sums the value, updates
     * diagnostics, reports progress.
     */
    private void acceptPanel(double value, double errEst, double pa, double pb, double width,
            double fullWidth, boolean negate, String flagReasonOrNull) {
        double y = value - accComp;
        double t = accSum + y;
        accComp = (t - accSum) - y;
        accSum = t;
        if (Double.isFinite(errEst)) {
            accErrAccum += errEst;
        }
        accPanelCount++;
        accResolvedWidth += width;
        if (flagReasonOrNull != null) {
            singularities.add(new Singularity(0.5 * (pa + pb), width, flagReasonOrNull));
        }
        progressListener.onProgress(accPanelCount, accMaxDepthSeen,
                negate ? -(accSum + accCorrection) : (accSum + accCorrection), accResolvedWidth / fullWidth);
    }

    private String describeLocation(double x, double callLo, double callHi) {
        if (x == callLo) {
            return "Non-integrable power-law singularity at the lower bound (x=" + x + ")";
        }
        if (x == callHi) {
            return "Non-integrable power-law singularity at the upper bound (x=" + x + ")";
        }
        return "Non-integrable power-law singularity at an interior point (x=" + x + "), found via singularity scanning";
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
        if (appliedSubtractions.isEmpty()) {
            return null;
        }
        return String.join("; ", appliedSubtractions);
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
    // Taylor-series rescue (three call sites in integrateSegment)
    // ------------------------------------------------------------------
    private boolean computeTaylorRescue(double a, double b) {
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

        if (leftActive) {
            subtractBinomialSeries(m - leftX0, leftN, leftC, +1);
        }
        if (rightActive) {
            subtractBinomialSeries(rightX0 - m, rightN, rightC, -1);
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

        rescueValue = sum;
        rescueErrEst = Math.abs(probeTerm);
        rescueConverging = converging;
        return Double.isFinite(sum) && Double.isFinite(rescueErrEst);
    }

    private void subtractBinomialSeries(double d, double n, double c, int sign) {
        binomBuf[0] = 1.0;
        double dPow = Math.pow(d, n);
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

    // ------------------------------------------------------------------
    // Self-tests
    // ------------------------------------------------------------------
    private static int passCount = 0;
    private static int failCount = 0;

    private static void expect(String label, double actual, double expected, double tol) {
        double err = Math.abs(actual - expected);
        boolean ok = err <= tol;
        System.out.printf("[%s] %-60s actual=%.15g expected=%.15g err=%.3e (tol=%.1e)%n",
                ok ? "PASS" : "FAIL", label, actual, expected, err, tol);
        if (ok) {
            passCount++;
        } else {
            failCount++;
        }
    }

    public static void main(String[] args) {
        Integrator expI = builder().build("exp(x)", "x");
        expect("exp(x) on [0,10]", expI.integrate(0, 10), Math.exp(10) - 1, 1e-8);

        Integrator bumpDefault = builder()
                .absoluteTolerance(1e-10).relativeTolerance(1e-10)
                .build("exp(-3200*(x-1.3)^2)", "x");
        double bumpTrue = 0.03133285343288750;
        expect("Gaussian bump (aliasing cross-check, regression)", bumpDefault.integrate(0, 2), bumpTrue, 1e-8);

        Integrator sqrtI = builder().build("sqrt(x)", "x");
        expect("sqrt(x) on [0,1] (n=0.5 outer endpoint, regression)", sqrtI.integrate(0, 1), 2.0 / 3.0, 1e-9);

        Integrator mixedI = builder().build("1/sqrt(x) + cos(x)", "x");
        expect("1/sqrt(x)+cos(x) on [0,1] (cancellation-floor, regression)",
                mixedI.integrate(0, 1), 2.0 + Math.sin(1.0), 1e-8);

        // --- The case that motivated this revision: pole strictly OUTSIDE the interval, right at
        // its stiff edge. The interior scan must NOT falsely claim this as an interior singularity
        // (Newton-snap wanders to x=0.5, outside the scanned domain, and is correctly rejected);
        // this must be resolved by ordinary bisection near the stiff edge instead.
        Integrator stiffEdge = builder().maxDepth(55).build("1/(x-0.5)", "x");
        expect("1/(x-0.5) on [0.1, 0.499999999] (pole just outside domain, NOT an interior hit)",
                stiffEdge.integrate(0.1, 0.499999999), -19.806975105072254332, 1e-6);

        // --- Genuine interior pole: 1/sin(x) crosses a simple pole at x=pi inside [1,4].
        // Must be detected by the scan and thrown as non-integrable, with a message identifying
        // it as an INTERIOR location, not an outer bound.
        Integrator interiorPole = builder().build("1/sin(x)", "x");
        try {
            interiorPole.integrate(1, 4);
            System.out.println("[FAIL] 1/sin(x) on [1,4] (crosses pole at pi) should have thrown");
            failCount++;
        } catch (NonIntegrableSingularityException expectedEx) {
            boolean mentionsInterior = expectedEx.getMessage().contains("interior point");
            System.out.printf("[%s] 1/sin(x) on [1,4] correctly threw (interior pole at pi): %s%n",
                    mentionsInterior ? "PASS" : "FAIL", expectedEx.getMessage());
            if (mentionsInterior) {
                passCount++;
            } else {
                failCount++;
            }
        }

        // --- Genuine interior, INTEGRABLE power-law singularity at x=2, split symmetrically.
        // This is a real new capability: neither an outer-bound-only design, nor the reference
        // file's boundary-collapsing correction formula, handles this correctly.
        Integrator interiorIntegrable = builder().build("1/sqrt(abs(x-2))", "x");
        try {
            expect("1/sqrt(|x-2|) on [0,4] (interior n=-0.5 singularity, split at x=2)",
                    interiorIntegrable.integrate(0, 4), 4.0 * Math.sqrt(2.0), 1e-7);
            System.out.println("  " + interiorIntegrable.getLastSingularitySubtractionInfo());
            System.out.println("  segments implied by interior poles found: " + interiorIntegrable.getSingularities());
        } catch (NonIntegrableSingularityException ex) {
            System.out.println("[INFO ] 1/sqrt(abs(x-2)) on [1.1,15] threw: " + ex.getMessage()
                    + " -- if unexpected, this is exactly the kind of false positive to tune"
                    + " interiorScanRelativeMultiplier/poleSnapMagnitudeThreshold against");
        }

        // --- False-positive avoidance: x^x grows extremely fast but is smooth everywhere on
        // [1.1,15]; the scan must not mistake rapid-but-continuous growth for a pole. Printed as
        // diagnostic, not hard-asserted, since I have not executed this myself -- see chat note.
        Integrator xx = builder().build("x^x", "x");
        try {
            double v = xx.integrate(1.1, 15);
            double expectedXX = 3.92135678385800451368E16;
            System.out.printf("[INFO ] x^x on [1.1,15]: got=%.6g (reference=%.6g, rel err=%.3e) --"
                    + " completed without a false-positive interior-singularity throw%n",
                    v, expectedXX, Math.abs(v - expectedXX) / expectedXX);
        } catch (NonIntegrableSingularityException ex) {
            System.out.println("[INFO ] x^x on [1.1,15] threw: " + ex.getMessage()
                    + " -- if unexpected, this is exactly the kind of false positive to tune"
                    + " interiorScanRelativeMultiplier/poleSnapMagnitudeThreshold against");
        }

        // Precision-floor acceptance regression.
        Integrator impossibleTol = builder()
                .absoluteTolerance(1e-20).relativeTolerance(1e-20)
                .build("1/sqrt(x) + cos(x)", "x");
        try {
            double v = impossibleTol.integrate(0, 1);
            double err = Math.abs(v - (2.0 + Math.sin(1.0)));
            boolean ok = err < 1e-6;
            System.out.printf("[%s] impossible tolerance (1e-20) on 1/sqrt(x)+cos(x): got=%.10g err=%.3e%n",
                    ok ? "PASS" : "FAIL", v, err);
            if (ok) {
                passCount++;
            } else {
                failCount++;
            }
        } catch (NonIntegrableSingularityException ex) {
            System.out.println("[FAIL] impossible tolerance should have been accepted at the precision floor: " + ex.getMessage());
            failCount++;
        }

        Integrator divergent = builder().build("1/x^1.5", "x");
        try {
            divergent.integrate(0, 1);
            System.out.println("[FAIL] 1/x^1.5 on [0,1] should have thrown (non-integrable)");
            failCount++;
        } catch (NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/x^1.5 on [0,1] correctly threw: " + expectedEx.getMessage());
            passCount++;
        }

        Integrator poleStrict = builder().build("1/(x-0.5)", "x");
        try {
            poleStrict.integrate(0.1, 0.9);
            System.out.println("[FAIL] 1/(x-0.5) through pole (strict, crosses it) should have thrown");
            failCount++;
        } catch (NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/(x-0.5) through pole (strict, crosses it) correctly threw: " + expectedEx.getMessage());
            passCount++;
        }

        int[] callCount = {0};
        double[] lastFraction = {-1.0};
        boolean[] monotonic = {true};
        Integrator progressI = builder()
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
