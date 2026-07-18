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
 * Composite adaptive integrator: Gauss-Kronrod 7-15 as the primary panel rule,
 * an AD Taylor cross-check against GK's one structural blind spot, and analytic
 * singularity handling that locates the <em>true</em> nearby singularity (which
 * may lie outside the requested bounds) rather than assuming it coincides with
 * wherever the caller's interval happens to end.
 *
 * <h2>Why "assume the bound is the singularity" was wrong</h2>
 * An earlier revision modeled {@code f(x) ~= C*(x-x0)^n} with {@code x0} fixed
 * at the domain bound itself. For an integrand like {@code 1/sin(x)} on
 * {@code [1e-7, 1.5]}, the true singularity is at {@code x=0} -- not at
 * {@code lo=1e-7}, merely close to it. Subtracting {@code C*(x-lo)^n}
 * manufactures a divergence at {@code x=lo} that {@code f} itself does not have
 * there ({@code f(lo)} is large but perfectly finite). Worse: even an
 * infinitesimal error in the fitted exponent causes the subtraction mismatch to
 * diverge <em>multiplicatively</em> as panels shrink toward the (wrong)
 * reference point -- {@code s^d -> infinity} as {@code s -> 0} for any nonzero
 * {@code d}, however small -- which is why the failure looked like unbounded
 * numerical noise rather than bounded floating-point cancellation.
 *
 * <h2>The fix</h2>
 * <ol>
 * <li><b>Locate the true {@code x0}</b> via Newton iteration on
 * {@code f/f'} ({@link #findNearbySingularity}): for a pure power law
 * {@code f=A*(x-c)^n}, {@code f/f' = (x-c)/(-n)}, so {@code x - f/f'} converges
 * toward {@code c} -- and {@code c} is allowed to land outside {@code [a,b]},
 * which is the correct outcome for the {@code 1/sin(x)} case above.</li>
 * <li><b>A general two-term closed-form correction</b>
 * ({@link #correctionTerm}) that does not assume {@code x0} coincides with a
 * bound: {@code C*[upper^(n+1) - lower^(n+1)]/(n+1)} (or
 * {@code C*ln(upper/lower)} when {@code n=-1}), where {@code lower} and
 * {@code upper} are the two segment-relative distances from {@code x0}. The old
 * one-term formula was a special case of this that only happened to be correct
 * when {@code x0} really was the bound.</li>
 * <li><b>Non-integrability is now judged by proximity, not exponent alone.</b>
 * {@code 1/x} has {@code n=-1}, but {@code integral[1e-7,1.5] 1/x dx} is
 * perfectly finite -- non-convergence is a property of integrating
 * <em>through</em> the singularity, not of the exponent in isolation. This
 * class only throws {@link NonIntegrableSingularityException} when
 * {@code n&lt;=-1} <em>and</em> {@code x0} essentially coincides with the
 * segment edge being evaluated.</li>
 * <li><b>Exponent snapping</b> ({@link #snapExponent}, adapted from an external
 * reviewer's technique): once {@code x0} is correct, snapping the fitted
 * exponent to the nearest canonical value (a multiple of
 * {@code 1/singularityOrderSnapDenominator}) eliminates the residual
 * multiplicative-mismatch risk from (1) essentially entirely, since real
 * elementary-function singularities almost always have exactly such an
 * order.</li>
 * <li><b>A tight secondary probe for {@code C}</b> ({@link
 *       #probeEndpointSingularityTwoScale}): once the exponent is trusted, one more,
 * much closer probe refines the leading coefficient -- fitting {@code C} from a
 * probe at relative distance {@code 1e-4} leaves an {@code O(s^2)} bias (e.g.
 * {@code ~3.75e-9} relative for {@code 1/sin(x)}) that a probe at {@code 1e-8}
 * reduces to {@code ~1e-17}, i.e. machine precision.</li>
 * </ol>
 *
 * <h2>Domain-wide singularity handling (new)</h2> {@link #scanForInteriorPoles}
 * scans the whole requested interval (not just its two endpoints) for blow-ups
 * or sign-changes-through-infinity, refines each candidate via the same Newton
 * search, and splits the domain into segments at each confirmed interior
 * singularity. Each segment is then integrated independently, with the shared
 * endpoint of two adjacent segments already known exactly (no re-search needed)
 * rather than assumed.
 *
 * <h2>What is unchanged from the previous revision</h2>
 * Gauss-Kronrod 7-15 with the QUADPACK {@code resasc}-refined error estimate;
 * the AD Taylor cross-check and its per-panel re-enablement once subtraction is
 * locally negligible; the cancellation floor and precision-floor acceptance for
 * genuine floating-point noise (distinct from, and still needed alongside, the
 * fixes above); zero-allocation per-panel steady state; progress reporting.
 *
 * <h2>Thread-safety</h2>
 * Not thread-safe. One instance per thread, built from the same expression
 * (read-only).
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
    public static final double DEFAULT_CROSS_SCALE_CONSISTENCY_TOL = 0.1;
    public static final double DEFAULT_SINGULARITY_SCALE_FACTOR = 100.0;
    public static final double DEFAULT_CROSS_CHECK_SUBTRACTION_NEGLIGIBILITY = 1e-6;
    public static final double CANCELLATION_FLOOR_FACTOR = 100.0;
    public static final double PRECISION_FLOOR_ACCEPT_MARGIN = 2.0;
    public static final int DEFAULT_CROSS_CHECK_MAX_DEPTH = Integer.MAX_VALUE;

    public static final int DEFAULT_SINGULARITY_ORDER_SNAP_DENOMINATOR = 6;
    public static final double DEFAULT_SINGULARITY_ORDER_SNAP_TOLERANCE = 0.08;
    public static final double DEFAULT_NEARBY_SINGULARITY_TRIGGER_MAGNITUDE = 1e3;
    public static final double DEFAULT_BOUND_COINCIDENCE_TOLERANCE = 1e-9;
    public static final int DEFAULT_POLE_SEARCH_MAX_ITERATIONS = 40;
    public static final double DEFAULT_POLE_SEARCH_CONVERGENCE_TOL = 1e-11;
    public static final int DEFAULT_INTERIOR_SCAN_POINTS = 150;
    public static final double DEFAULT_INTERIOR_SCAN_MAGNITUDE_THRESHOLD = 1e6;
    public static final int DEFAULT_MAX_INTERIOR_POLES = 32;

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
        public final int interiorPolesFound;
        public final String singularitySubtractionInfo;
        public final List<Singularity> singularities;

        Result(double value, double estimatedError, int panelCount, int maxDepth,
                int crossCheckCount, int forcedSplitCount, int interiorPolesFound,
                String singularitySubtractionInfo, List<Singularity> singularities) {
            this.value = value;
            this.estimatedError = estimatedError;
            this.panelCount = panelCount;
            this.maxDepth = maxDepth;
            this.crossCheckCount = crossCheckCount;
            this.forcedSplitCount = forcedSplitCount;
            this.interiorPolesFound = interiorPolesFound;
            this.singularitySubtractionInfo = singularitySubtractionInfo;
            this.singularities = singularities;
        }

        @Override
        public String toString() {
            return String.format(
                    "value=%.15g  errEst~%.3e  panels=%d  maxDepth=%d  crossChecks=%d  forcedSplits=%d"
                    + "  interiorPoles=%d%s  flags=%d%s",
                    value, estimatedError, panelCount, maxDepth, crossCheckCount, forcedSplitCount,
                    interiorPolesFound,
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

        private int singularityOrderSnapDenominator = DEFAULT_SINGULARITY_ORDER_SNAP_DENOMINATOR;
        private double singularityOrderSnapTolerance = DEFAULT_SINGULARITY_ORDER_SNAP_TOLERANCE;
        private double nearbySingularityTriggerMagnitude = DEFAULT_NEARBY_SINGULARITY_TRIGGER_MAGNITUDE;
        private double boundCoincidenceTolerance = DEFAULT_BOUND_COINCIDENCE_TOLERANCE;
        private int poleSearchMaxIterations = DEFAULT_POLE_SEARCH_MAX_ITERATIONS;
        private double poleSearchConvergenceTol = DEFAULT_POLE_SEARCH_CONVERGENCE_TOL;
        private boolean interiorPoleScanEnabled = true;
        private int interiorScanPoints = DEFAULT_INTERIOR_SCAN_POINTS;
        private double interiorScanMagnitudeThreshold = DEFAULT_INTERIOR_SCAN_MAGNITUDE_THRESHOLD;
        private int maxInteriorPoles = DEFAULT_MAX_INTERIOR_POLES;

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

        public Builder progressListener(ProgressListener listener) {
            this.progressListener = listener != null ? listener : ProgressListener.NO_OP;
            return this;
        }

        public Builder singularityOrderSnapDenominator(int denom) {
            this.singularityOrderSnapDenominator = denom;
            return this;
        }

        public Builder singularityOrderSnapTolerance(double tol) {
            this.singularityOrderSnapTolerance = tol;
            return this;
        }

        public Builder nearbySingularityTriggerMagnitude(double mag) {
            this.nearbySingularityTriggerMagnitude = mag;
            return this;
        }

        public Builder boundCoincidenceTolerance(double tol) {
            this.boundCoincidenceTolerance = tol;
            return this;
        }

        public Builder poleSearchMaxIterations(int n) {
            this.poleSearchMaxIterations = n;
            return this;
        }

        public Builder poleSearchConvergenceTolerance(double tol) {
            this.poleSearchConvergenceTol = tol;
            return this;
        }

        public Builder interiorPoleScanEnabled(boolean enabled) {
            this.interiorPoleScanEnabled = enabled;
            return this;
        }

        public Builder interiorScanPoints(int points) {
            this.interiorScanPoints = points;
            return this;
        }

        public Builder interiorScanMagnitudeThreshold(double mag) {
            this.interiorScanMagnitudeThreshold = mag;
            return this;
        }

        public Builder maxInteriorPoles(int max) {
            this.maxInteriorPoles = max;
            return this;
        }

        public Integrator build(MathExpression me, String wrtVar) {
            return new Integrator(this, me, wrtVar);
        }

        public Integrator build(String expr, String wrtVar) {
            return build(new MathExpression(expr), wrtVar);
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

    private final int singularityOrderSnapDenominator;
    private final double singularityOrderSnapTolerance;
    private final double nearbySingularityTriggerMagnitude;
    private final double boundCoincidenceTolerance;
    private final int poleSearchMaxIterations;
    private final double poleSearchConvergenceTol;
    private final boolean interiorPoleScanEnabled;
    private final int interiorScanPoints;
    private final double interiorScanMagnitudeThreshold;
    private final int maxInteriorPoles;

    // Scratch (zero-allocation steady state)
    private final double[] fnBuf;
    private final double[] fv1;
    private final double[] fv2;
    private final double[] ccCoeffs;
    private final double[] probeCoeffs; // reused by both the endpoint probe and the pole search (sequential use only)
    private final double[] poleBuffer;
    private final double[] handledSingLoc;
    private final double[] handledSingN;
    private final double[] handledSingC;
    private int handledSingCount;
    private boolean lastInteriorPoleCapHit;

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

    // Scratch written by probeSinglePointExponent / probeEndpointSingularityTwoScale
    private double probeN;
    private double probeC;
    private double probeF;
    private double probeS;
    private boolean probeDomainFailure;

    // Per-segment active subtraction state (leftX0/rightX0 may lie outside the segment)
    private boolean leftActive, rightActive;
    private double leftX0, leftN, leftC;
    private double rightX0, rightN, rightC;

    private double lastErrorEstimate;
    private int lastPanelCount;
    private int lastMaxDepthReached;
    private int lastCrossCheckCount;
    private int lastForcedSplitCount;
    private int lastInteriorPolesFound;
    private final List<Singularity> singularities;

    private Integrator(Builder b, MathExpression me, String wrtVar) {
        if (b.crossCheckOrder < 2 || (b.crossCheckOrder % 2) != 0) {
            throw new IllegalArgumentException("crossCheckOrder must be a positive even number");
        }
        if (b.crossCheckOrder > MAX_CROSS_CHECK_ORDER) {
            throw new IllegalArgumentException("crossCheckOrder must be <= " + MAX_CROSS_CHECK_ORDER);
        }
        if (b.maxDepth < 1) {
            throw new IllegalArgumentException("maxDepth must be >= 1");
        }
        if (b.absTol < 0 || b.relTol < 0) {
            throw new IllegalArgumentException("tolerances must be >= 0");
        }
        if (b.crossCheckLooseness < 1.0) {
            throw new IllegalArgumentException("crossCheckLooseness must be >= 1.0");
        }
        if (b.singularityScaleFactor <= 1.0) {
            throw new IllegalArgumentException("singularityScaleFactor must be > 1.0");
        }
        if (wrtVar == null || wrtVar.trim().isEmpty()) {
            throw new IllegalArgumentException("wrtVar must be a non-empty variable name");
        }

        this.wrtVar = wrtVar;
        this.crossCheckOrder = b.crossCheckOrder;
        this.crossCheckAdOrder = b.crossCheckOrder + 2;
        this.ad = new AutoDiffNEvaluator(me, crossCheckAdOrder);
        this.absTol = b.absTol;
        this.relTol = b.relTol;
        this.maxDepth = b.maxDepth;
        this.strictSingularities = b.strictSingularities;
        this.crossCheckEnabled = b.crossCheckEnabled;
        this.crossCheckLooseness = b.crossCheckLooseness;
        this.crossCheckMaxDepth = b.crossCheckMaxDepth;
        this.singularitySubtractionEnabled = b.singularitySubtractionEnabled;
        this.singularityConsistencyTol = b.singularityConsistencyTol;
        this.crossScaleConsistencyTol = b.crossScaleConsistencyTol;
        this.singularityScaleFactor = b.singularityScaleFactor;
        this.crossCheckSubtractionNegligibility = b.crossCheckSubtractionNegligibility;
        this.progressListener = b.progressListener;

        this.singularityOrderSnapDenominator = b.singularityOrderSnapDenominator;
        this.singularityOrderSnapTolerance = b.singularityOrderSnapTolerance;
        this.nearbySingularityTriggerMagnitude = b.nearbySingularityTriggerMagnitude;
        this.boundCoincidenceTolerance = b.boundCoincidenceTolerance;
        this.poleSearchMaxIterations = b.poleSearchMaxIterations;
        this.poleSearchConvergenceTol = b.poleSearchConvergenceTol;
        this.interiorPoleScanEnabled = b.interiorPoleScanEnabled;
        this.interiorScanPoints = b.interiorScanPoints;
        this.interiorScanMagnitudeThreshold = b.interiorScanMagnitudeThreshold;
        this.maxInteriorPoles = b.maxInteriorPoles;

        this.fnBuf = new double[1];
        this.fv1 = new double[7];
        this.fv2 = new double[7];
        this.ccCoeffs = new double[crossCheckAdOrder + 1];
        this.probeCoeffs = new double[3];
        this.poleBuffer = new double[maxInteriorPoles];
        this.handledSingLoc = new double[maxInteriorPoles + 2];
        this.handledSingN = new double[maxInteriorPoles + 2];
        this.handledSingC = new double[maxInteriorPoles + 2];

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
    // Top-level integration: scan for interior singularities, split into
    // segments, integrate each segment, sum.
    // ------------------------------------------------------------------
    public double integrate(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            throw new IllegalArgumentException(
                    "bounds must be finite (got [" + a + ", " + b + "])");
        }

        singularities.clear();
        handledSingCount = 0;
        lastInteriorPoleCapHit = false;
        lastErrorEstimate = 0.0;
        lastPanelCount = 0;
        lastMaxDepthReached = 0;
        lastCrossCheckCount = 0;
        lastForcedSplitCount = 0;
        lastInteriorPolesFound = 0;

        if (a == b) {
            return 0.0;
        }
        boolean negate = b < a;
        double lo = negate ? b : a;
        double hi = negate ? a : b;

        int poleCount = 0;
        if (interiorPoleScanEnabled && singularitySubtractionEnabled) {
            poleCount = scanForInteriorPoles(lo, hi);
            lastInteriorPolesFound = poleCount;
        }

        double total = 0.0;
        double segStart = lo;
        for (int i = 0; i <= poleCount; i++) {
            double segEnd = (i < poleCount) ? poleBuffer[i] : hi;
            if (segEnd <= segStart) {
                continue; // degenerate/duplicate boundary, skip
            }
            boolean leftKnown = i > 0;
            boolean rightKnown = i < poleCount;
            total += integrateSegment(segStart, segEnd, hi - lo, leftKnown, segStart, rightKnown, segEnd);
            segStart = segEnd;
        }

        return negate ? -total : total;
    }

    public Result integrateResult(double a, double b) {
        double value = integrate(a, b);
        return new Result(value, lastErrorEstimate, lastPanelCount, lastMaxDepthReached,
                lastCrossCheckCount, lastForcedSplitCount, lastInteriorPolesFound,
                getLastSingularitySubtractionInfo(), new ArrayList<>(singularities));
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

    public int getLastInteriorPolesFound() {
        return lastInteriorPolesFound;
    }

    public boolean isStrictSingularities() {
        return strictSingularities;
    }

    public boolean isLastInteriorPoleCapHit() {
        return lastInteriorPoleCapHit;
    }

    /**
     * Lazily builds a summary of every singularity handled (endpoint or
     * interior) on the most recent call.
     */
    public String getLastSingularitySubtractionInfo() {
        if (handledSingCount == 0) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < handledSingCount; i++) {
            if (i > 0) {
                sb.append("; ");
            }
            sb.append(String.format("x0=%.10g n=%.6g C=%.6g", handledSingLoc[i], handledSingN[i], handledSingC[i]));
        }
        return sb.toString();
    }

    public List<Singularity> getSingularities() {
        return Collections.unmodifiableList(singularities);
    }

    private void recordHandledSingularity(double x0, double n, double c) {
        if (handledSingCount < handledSingLoc.length) {
            handledSingLoc[handledSingCount] = x0;
            handledSingN[handledSingCount] = n;
            handledSingC[handledSingCount] = c;
            handledSingCount++;
        }
    }

    // ------------------------------------------------------------------
    // Interior singularity scanning
    // ------------------------------------------------------------------
    /**
     * Coarse scan of {@code [lo,hi]} for blow-ups or sign-changes-through-
     * infinity, refined via {@link #findNearbySingularity}. Writes confirmed,
     * deduplicated, sorted locations into {@link #poleBuffer} and returns the
     * count. Capped at {@code maxInteriorPoles}; if hit,
     * {@link #lastInteriorPoleCapHit} is set (not silently dropped without a
     * signal, but not thrown either -- the poles found are still used).
     */
    private int scanForInteriorPoles(double lo, double hi) {
        int n = interiorScanPoints;
        double step = (hi - lo) / n;
        int found = 0;

        double prevX = lo;
        boolean prevOk = evalFnAt(prevX);
        double prevF = prevOk ? fnBuf[0] : Double.NaN;

        for (int i = 1; i <= n && found < maxInteriorPoles; i++) {
            double x = (i == n) ? hi : lo + i * step;
            boolean ok = evalFnAt(x);
            double f = ok ? fnBuf[0] : Double.NaN;

            boolean suspicious = !ok || !Double.isFinite(f)
                    || Math.abs(f) > interiorScanMagnitudeThreshold
                    || (prevOk && ok && Double.isFinite(prevF) && Double.isFinite(f)
                    && Math.signum(prevF) != Math.signum(f)
                    && (Math.abs(prevF) > interiorScanMagnitudeThreshold
                    || Math.abs(f) > interiorScanMagnitudeThreshold));

            if (suspicious && x > lo && prevX < hi) {
                double candidate = findNearbySingularity(0.5 * (prevX + x), prevX, x);
                if (!Double.isNaN(candidate) && candidate > lo && candidate < hi) {
                    boolean dup = false;
                    double dedupTol = (hi - lo) * 1e-9;
                    for (int j = 0; j < found; j++) {
                        if (Math.abs(poleBuffer[j] - candidate) < dedupTol) {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup && found < maxInteriorPoles) {
                        poleBuffer[found++] = candidate;
                    }
                }
            }
            prevX = x;
            prevOk = ok;
            prevF = f;
        }
        if (found == maxInteriorPoles) {
            lastInteriorPoleCapHit = true;
            LOG.warning("Interior pole scan hit maxInteriorPoles (" + maxInteriorPoles
                    + "); some singularities in [" + lo + "," + hi + "] may be unhandled.");
        }
        Arrays.sort(poleBuffer, 0, found);
        return found;
    }

    /**
     * Newton search for a genuine singularity near {@code seed}, bounded to
     * roughly {@code [searchLo, searchHi]} plus a small margin. For a pure
     * power law {@code f=A*(x-c)^n}, {@code f/f' = (x-c)/(-n)}, so
     * {@code x - f/f'} converges toward {@code c}. The result may fall outside
     * {@code [searchLo, searchHi]} -- that is correct and expected when the
     * nearest true singularity is just past a requested bound. Returns
     * {@code Double.NaN} if nothing is confirmed.
     */
    private double findNearbySingularity(double seed, double searchLo, double searchHi) {
        double span = Math.max(searchHi - searchLo, 1e-12);
        double maxJump = span; // don't let one Newton step run away arbitrarily far
        double current = seed;

        for (int iter = 0; iter < poleSearchMaxIterations; iter++) {
            double f, fp;
            try {
                ad.taylorCoefficients(wrtVar, current, 1, probeCoeffs);
            } catch (ArithmeticException domainEx) {
                return confirmSingularity(current) ? current : Double.NaN;
            }
            f = probeCoeffs[0];
            fp = probeCoeffs[1];
            if (!Double.isFinite(f)) {
                return confirmSingularity(current) ? current : Double.NaN;
            }
            if (fp == 0.0 || !Double.isFinite(fp)) {
                return Double.NaN;
            }
            double dx = f / fp;
            if (!Double.isFinite(dx)) {
                return Double.NaN;
            }
            if (Math.abs(dx) > maxJump) {
                dx = Math.copySign(maxJump, dx);
            }
            if (Math.abs(dx) < poleSearchConvergenceTol * span) {
                double candidate = current + dx;
                return confirmSingularity(candidate) ? candidate : Double.NaN;
            }
            current += dx;
        }
        return Double.NaN;
    }

    /**
     * A candidate is a genuine singularity if f itself is non-finite or throws
     * there.
     */
    private boolean confirmSingularity(double x0) {
        try {
            ad.evaluateRPN(wrtVar, x0, 0, fnBuf);
            return !Double.isFinite(fnBuf[0]);
        } catch (ArithmeticException domainEx) {
            return true;
        }
    }

    /**
     * Cheap pre-check: is it worth spending a Newton search near this point at
     * all?
     */
    private boolean looksNearSingular(double x) {
        try {
            ad.evaluateRPN(wrtVar, x, 0, fnBuf);
        } catch (ArithmeticException domainEx) {
            return true;
        }
        return !Double.isFinite(fnBuf[0]) || Math.abs(fnBuf[0]) > nearbySingularityTriggerMagnitude;
    }

    // ------------------------------------------------------------------
    // Per-segment integration
    // ------------------------------------------------------------------
    private double integrateSegment(double segLo, double segHi, double globalFullWidth,
            boolean leftIsKnownPole, double leftKnownX0,
            boolean rightIsKnownPole, double rightKnownX0) {
        double fullWidth = segHi - segLo;
        double minWidth = Math.max(fullWidth * 1e-13, Double.MIN_NORMAL);

        leftActive = false;
        rightActive = false;
        double correction = 0.0;

        if (singularitySubtractionEnabled) {
            double x0Left = Double.NaN;
            if (leftIsKnownPole) {
                x0Left = leftKnownX0;
            } else if (looksNearSingular(segLo)) {
                x0Left = findNearbySingularity(segLo, segLo - fullWidth, segHi);
            }
            if (!Double.isNaN(x0Left) && probeEndpointSingularityTwoScale(x0Left, fullWidth, +1.0)) {
                double n = probeN, c = probeC;
                double uLo = segLo - x0Left;
                boolean coincident = uLo < globalFullWidth * boundCoincidenceTolerance;
                if (n <= -1.0 && coincident) {
                    throw new NonIntegrableSingularityException(segLo, 0.0,
                            "Non-integrable power-law singularity at the lower bound x=" + segLo
                            + " (snapped exponent n=" + n + " <= -1); this integral does not converge",
                            null);
                }
                leftActive = true;
                leftX0 = x0Left;
                leftN = n;
                leftC = c;
                recordHandledSingularity(x0Left, n, c);
                double uHi = segHi - x0Left;
                correction += correctionTerm(c, n, uLo, uHi);
            }

            double x0Right = Double.NaN;
            if (rightIsKnownPole) {
                x0Right = rightKnownX0;
            } else if (looksNearSingular(segHi)) {
                x0Right = findNearbySingularity(segHi, segLo, segHi + fullWidth);
            }
            if (!Double.isNaN(x0Right) && probeEndpointSingularityTwoScale(x0Right, fullWidth, -1.0)) {
                double n = probeN, c = probeC;
                double uLo = x0Right - segHi;
                boolean coincident = uLo < globalFullWidth * boundCoincidenceTolerance;
                if (n <= -1.0 && coincident) {
                    throw new NonIntegrableSingularityException(segHi, 0.0,
                            "Non-integrable power-law singularity at the upper bound x=" + segHi
                            + " (snapped exponent n=" + n + " <= -1); this integral does not converge",
                            null);
                }
                rightActive = true;
                rightX0 = x0Right;
                rightN = n;
                rightC = c;
                recordHandledSingularity(x0Right, n, c);
                double uFar = x0Right - segLo;
                correction += correctionTerm(c, n, uLo, uFar);
            }
        }

        stackSize = 0;
        stackPush(segLo, segHi, 0);

        double sum = 0.0, comp = 0.0;
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
                    singularities.add(new Singularity(0.5 * (pa + pb), width,
                            "accepted at the floating-point cancellation floor (reported error " + gkAbsErr
                            + " is the characterized noise floor, not an unresolved failure)"));
                    sum = kahanAdd(sum, comp, gkResult);
                    comp = kahanComp;
                    errAccum += gkAbsErr;
                    panelCount++;
                    resolvedWidth += width;
                    reportProgress(panelCount, maxDepthSeen, sum + correction, resolvedWidth / fullWidth);
                    continue;
                }
                recordSingularityOrThrow(pa, pb,
                        "Gauss-Kronrod did not converge to tolerance (reported error " + gkAbsErr
                        + " vs budget " + budget + ") after reaching the panel-depth/width limit", null);
                sum = kahanAdd(sum, comp, gkResult);
                comp = kahanComp;
                errAccum += gkAbsErr;
                panelCount++;
                resolvedWidth += width;
                reportProgress(panelCount, maxDepthSeen, sum + correction, resolvedWidth / fullWidth);
                continue;
            }

            boolean subtractionNegligibleHere = (leftActive || rightActive)
                    && lastPanelMaxSubtractedMag <= crossCheckSubtractionNegligibility * Math.abs(gkResult);
            boolean withinCrossCheckDepth = pDepth <= crossCheckMaxDepth;
            boolean runCrossCheckHere = crossCheckEnabled && withinCrossCheckDepth
                    && (!(leftActive || rightActive) || subtractionNegligibleHere);

            if (!runCrossCheckHere) {
                sum = kahanAdd(sum, comp, gkResult);
                comp = kahanComp;
                errAccum += gkAbsErr;
                panelCount++;
                resolvedWidth += width;
                reportProgress(panelCount, maxDepthSeen, sum + correction, resolvedWidth / fullWidth);
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
                        "Gauss-Kronrod/AD cross-check disagreement could not be resolved by further bisection"
                        + (ccOk ? (" (GK=" + gkResult + ", AD=" + ccValue + ")")
                                : " (AD cross-check itself failed to evaluate)"), null);
                sum = kahanAdd(sum, comp, gkResult);
                comp = kahanComp;
                errAccum += gkAbsErr;
                panelCount++;
                resolvedWidth += width;
                reportProgress(panelCount, maxDepthSeen, sum + correction, resolvedWidth / fullWidth);
                continue;
            }

            sum = kahanAdd(sum, comp, gkResult);
            comp = kahanComp;
            errAccum += gkAbsErr;
            panelCount++;
            resolvedWidth += width;
            reportProgress(panelCount, maxDepthSeen, sum + correction, resolvedWidth / fullWidth);
        }

        sum += correction;

        lastErrorEstimate += errAccum;
        lastPanelCount += panelCount;
        lastMaxDepthReached = Math.max(lastMaxDepthReached, maxDepthSeen);
        lastCrossCheckCount += crossCheckCount;
        lastForcedSplitCount += forcedSplitCount;

        return sum;
    }

    private double kahanComp;

    private double kahanAdd(double sum, double comp, double value) {
        double y = value - comp;
        double t = sum + y;
        kahanComp = (t - sum) - y;
        return t;
    }

    private void reportProgress(int panels, int depth, double estimate, double fraction) {
        progressListener.onProgress(panels, depth, estimate, fraction);
    }

    /**
     * General closed-form correction for {@code integral C*s^n} over the range
     * represented by {@code [lower, upper]} (both distances from the
     * singularity, {@code lower < upper}; {@code lower} may be exactly 0 when
     * {@code x0} coincides with the near edge). Reduces to the old single-term
     * formula when {@code lower==0}.
     */
    private double correctionTerm(double c, double n, double lower, double upper) {
        if (Math.abs(n + 1.0) < 1e-9) {
            return c * (Math.log(upper) - Math.log(Math.max(lower, Double.MIN_NORMAL)));
        }
        return (c / (n + 1.0)) * (Math.pow(upper, n + 1.0) - Math.pow(lower, n + 1.0));
    }

    private void recordSingularityOrThrow(double pa, double pb, String reason, Throwable cause) {
        double location = 0.5 * (pa + pb);
        double width = pb - pa;
        if (strictSingularities) {
            throw new NonIntegrableSingularityException(location, width, reason, cause);
        }
        LOG.warning(String.format("Unresolved panel near x=%.15g (width %.3e): %s -- lenient mode", location, width, reason));
        singularities.add(new Singularity(location, width, reason));
    }

    // ------------------------------------------------------------------
    // Endpoint power-law exponent/coefficient fitting
    // ------------------------------------------------------------------
    /**
     * Snaps a raw exponent to the nearest multiple of 1/denominator, if close
     * enough to trust it.
     */
    private double snapExponent(double rawN) {
        double scaled = rawN * singularityOrderSnapDenominator;
        double candidate = Math.round(scaled) / (double) singularityOrderSnapDenominator;
        return (Math.abs(candidate - rawN) <= singularityOrderSnapTolerance) ? candidate : rawN;
    }

    /**
     * Single-point exponent estimate at distance {@code s} from {@code x0}
     * (which need not be inside the segment). Writes {@link #probeN} (raw,
     * unsnapped), {@link #probeF}, {@link #probeS} on success.
     */
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
        if (Math.abs(n) < 0.05) {
            return false;
        }
        probeN = n;
        probeF = f;
        probeS = s;
        return true;
    }

    /**
     * Given a known (or freshly-located) {@code x0}, fits (n, C): a near probe
     * and a far probe (100x farther, by default) must agree on the raw exponent
     * -- catching contaminated cases like {@code 1/sqrt(x) + 5000*x} -- after
     * which the exponent is snapped to the nearest canonical value, and a
     * third, much tighter probe refines {@code C} using the now-trusted
     * exponent (see class javadoc for why this matters at tight tolerances).
     * Writes {@link #probeN} (snapped) and {@link #probeC} on success.
     */
    private boolean probeEndpointSingularityTwoScale(double x0, double intervalWidth, double dir) {
        double sNear = Math.max(intervalWidth * 1e-4, 1e-12);
        if (!probeSinglePointExponent(x0, sNear, dir)) {
            return false;
        }
        double nNear = probeN;
        double fNear = probeF;
        double sNearActual = probeS;

        if (nNear > -1.0) {
            double sFar = Math.min(sNear * singularityScaleFactor, intervalWidth * 0.4);
            boolean farOk = probeSinglePointExponent(x0, sFar, dir);
            if (farOk) {
                if (Math.abs(nNear - probeN) > crossScaleConsistencyTol) {
                    return false;
                }
            } else if (!probeDomainFailure) {
                return false;
            }
        }
        // nNear <= -1.0 skips the far-scale check: whether integrable depends on proximity to a
        // bound, decided by the caller, not on this class's contamination heuristic.

        double nSnapped = snapExponent(nNear);

        double sTight = Math.max(sNear * 1e-4, 1e-14);
        double cFinal;
        if (probeSinglePointExponent(x0, sTight, dir) && Double.isFinite(Math.pow(sTight, nSnapped))) {
            cFinal = probeF / Math.pow(sTight, nSnapped);
        } else {
            cFinal = fNear / Math.pow(sNearActual, nSnapped);
        }
        if (!Double.isFinite(cFinal)) {
            return false;
        }

        probeN = nSnapped;
        probeC = cFinal;
        return true;
    }

    // ------------------------------------------------------------------
    // Sampling with active subtraction (unchanged in principle -- was
    // already correct for any x0, the bug was always upstream in what x0
    // got passed in)
    // ------------------------------------------------------------------
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
    // Self-tests
    // ------------------------------------------------------------------
    private static int passCount = 0;
    private static int failCount = 0;

    private static void expect(String label, double actual, double expected, double tol) {
        double err = Math.abs(actual - expected);
        boolean ok = err <= tol;
        System.out.printf("[%s] %-65s actual=%.15g expected=%.15g err=%.3e (tol=%.1e)%n",
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

        Integrator bump = builder().absoluteTolerance(1e-10).relativeTolerance(1e-10)
                .build("exp(-3200*(x-1.3)^2)", "x");
        expect("Gaussian bump (cross-check still catches it)", bump.integrate(0, 2), 0.03133285343288750, 1e-8);

        Integrator sqrtI = builder().build("sqrt(x)", "x");
        expect("sqrt(x) on [0,1]", sqrtI.integrate(0, 1), 2.0 / 3.0, 1e-9);

        Integrator mixedI = builder().build("1/sqrt(x) + cos(x)", "x");
        expect("1/sqrt(x)+cos(x) on [0,1]", mixedI.integrate(0, 1), 2.0 + Math.sin(1.0), 1e-8);

        // The case this whole revision exists for: true singularity (x=0) strictly outside
        // the domain (lo=1e-7), not coincident with it -- previously threw via a bogus
        // "n<=-1 always non-integrable" rule and a divergent correction formula.
        Integrator sinInvX = builder().build("1/sin(x)", "x");
        expect("1/sin(x) on [0.01,0.02] (nearby-not-coincident singularity)",
                sinInvX.integrate(0.01, 0.02), 0.6931721812891335, 1e-8);
        expect("1/sin(x) on [1e-7, 1.5] (the original failing case)",
                sinInvX.integrate(0.0000001, 1.5), 16.740387290564932, 1e-9);

        // Interior pole, genuinely non-integrable (n=-1 exactly at an interior point that
        // becomes a coincident segment edge after splitting) -- should still throw.
        Integrator poleStrict = builder().build("1/(x-0.5)", "x");
        try {
            poleStrict.integrate(0.1, 0.9);
            System.out.println("[FAIL] 1/(x-0.5) through interior pole should have thrown");
            failCount++;
        } catch (NonIntegrableSingularityException ex) {
            System.out.println("[PASS] 1/(x-0.5) through interior pole correctly threw: " + ex.getMessage());
            passCount++;
        }

        // Interior pole with an odd order that IS integrable in a principal-value-adjacent
        // sense is out of scope here (n=-1 is never integrable through); this instead checks
        // an interior removable-looking spike that should NOT be flagged non-integrable.
        Integrator cbrtI = builder().build("1/(x^(1/3))", "x");
        expect("1/x^(1/3) on [0,1] (n=-1/3 endpoint, integrable)", cbrtI.integrate(0, 1), 1.5, 1e-8);

        Integrator divergent = builder().build("1/x^1.5", "x");
        try {
            divergent.integrate(0, 1);
            System.out.println("[FAIL] 1/x^1.5 on [0,1] should have thrown (coincident, n<=-1)");
            failCount++;
        } catch (NonIntegrableSingularityException ex) {
            System.out.println("[PASS] 1/x^1.5 on [0,1] correctly threw: " + ex.getMessage());
            passCount++;
        }

        Integrator contaminated = builder().build("1/sqrt(x) + 5000*x", "x");
        expect("1/sqrt(x)+5000*x on [0,1] (cross-scale rejects the fit, falls back to plain bisection)",
                contaminated.integrate(0, 1), 2.0 + 2500.0, 1e-6);

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
        boolean progressOk = callCount[0] > 0 && monotonic[0] && Math.abs(lastFraction[0] - 1.0) < 1e-9;
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
        } catch (IllegalArgumentException ex) {
            System.out.println("[PASS] infinite bound correctly rejected: " + ex.getMessage());
            passCount++;
        }

        System.out.println();
        System.out.println(passCount + " passed, " + failCount + " failed");
        if (failCount > 0) {
            System.exit(1);
        }
    }
}
