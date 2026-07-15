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
import com.github.gbenroscience.parser.MathExpression;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Adaptive Gauss-Kronrod (7-15) integrator with an AD aliasing cross-check
 * and verified analytic subtraction of endpoint power-law singularities --
 * engineered so the per-panel hot path performs <b>zero heap allocations</b>,
 * built directly on {@link AutoDiffNEvaluator#evaluateRPN}, whose own
 * scratch buffers are sized once at construction and never reallocated.
 *
 * <h2>What "zero-allocation hot path" means here, precisely</h2>
 * The claim is scoped exactly, not asserted loosely:
 * <ul>
 *   <li><b>The per-panel loop</b> (Gauss-Kronrod evaluation at 15 nodes,
 *       the AD cross-check, Simpson fallback, the fixed-capacity worklist
 *       stack, Kahan summation, progress reporting) allocates nothing on
 *       the heap, for any number of panels, for the entire common case of
 *       a well-behaved integrand. Every scratch buffer it touches
 *       ({@code edgeBuf}, {@code crossDeriv}, {@code gkFv1}/{@code gkFv2},
 *       {@code exponentProbeBuf}, the stack arrays) is sized once in the
 *       constructor.</li>
 *   <li><b>Endpoint singularity detection</b> (at most twice per
 *       {@link #integrate(double, double)} call, never per panel) writes
 *       its result into instance fields ({@link #probeValid}, {@link
 *       #probeN}, {@link #probeC}) instead of returning a boxed object, so
 *       even this call-level step allocates nothing.</li>
 *   <li><b>The one honest exception:</b> if a singularity is actually
 *       detected and verified, building the smoothed expression (a new
 *       {@code String}, a re-parsed {@code Token[]}, a second {@code
 *       AutoDiffNEvaluator} with its own scratch arrays) happens exactly
 *       once per call, not per panel. This is a bounded, one-time setup
 *       cost, stated plainly rather than hidden -- it is what makes it
 *       possible to keep every panel afterward at true zero allocation
 *       while still integrating a smoothed, well-behaved function.</li>
 *   <li>Diagnostic/error paths (a genuine AD domain error, an aliasing
 *       flag, a {@link Singularity} being recorded, an exception being
 *       thrown) may allocate -- {@code String.format}, list adds, the
 *       exception object itself. These are, by construction, rare events,
 *       not the steady-state path, and are documented as such rather than
 *       silently exempted.</li>
 * </ul>
 *
 * <h2>Endpoint power-law singularity subtraction</h2>
 * At the start of every {@link #integrate(double, double)} call, both
 * bounds are probed for local behavior {@code f(x) ~= C*s^n}, where
 * {@code s} is the (positive) distance into the domain from that endpoint.
 * Detection combines two independent checks, each of which must pass:
 * <ol>
 *   <li><b>Two-formula consistency</b> at a single probe point: from one
 *       {@code evaluateRPN(..., order=2, ...)} call giving {@code f, f',
 *       f''}, {@code n1 = s*dir*f'/f} and {@code n2 = 1 + s*dir*f''/f'}
 *       are both exact for a pure power law and independently derived
 *       (dir = +1 at the lower bound, -1 at the upper). Their disagreement
 *       is a cheap, direct test of whether the local behavior really is a
 *       clean power law at that scale.</li>
 *   <li><b>Two-scale consistency</b>: the above is repeated at a second
 *       probe distance two orders of magnitude closer to the endpoint, and
 *       the resulting exponent estimates from the two scales must also
 *       agree. This catches what a single-scale check cannot: a function
 *       that is a sum of several power-law terms can look like a clean
 *       single power law at one particular distance while disagreeing at
 *       another.</li>
 * </ol>
 * Any exponent {@code n <= -1} (within numerical slack) means the ordinary
 * integral diverges there; this always throws {@link
 * EarlyDivergenceDetectedException} immediately in strict mode, before a
 * single panel is bisected. {@code |n| < 0.05} is treated as not
 * meaningfully singular. Otherwise -- for <em>either</em> sign of
 * {@code n}, not just blow-ups: a bounded-but-non-polynomial exponent like
 * {@code sqrt(x)}'s {@code n=0.5} still hurts Gauss-Kronrod's near-exact
 * convergence on analytic functions, and subtracting it is just as valid --
 * a new expression {@code (f) - C*(x-a)^n} (and/or the upper-bound mirror)
 * is built and <b>re-probed</b> before being trusted: if the residual is
 * still as singular or worse, the subtraction is discarded and the
 * original expression is integrated directly instead.
 *
 * <h2>Cancellation floor</h2>
 * Recovering an O(1) remainder from a subtraction of two numbers that are
 * individually huge near the endpoint (e.g. {@code 1/sqrt(x) + cos(x)}:
 * subtracting a ~10^8-magnitude term to recover {@code cos(x)} near
 * {@code x=1e-16}) has a hard floor of roughly
 * {@code machine_epsilon * (subtracted magnitude)} that no amount of
 * further bisection can push below -- it is a representable-precision
 * problem, not a convergence problem. Every Gauss-Kronrod node evaluated
 * while a subtraction is active also computes the magnitude of the term
 * that was subtracted there (pure {@code Math.pow} arithmetic against
 * already-tracked endpoint state -- no extra allocation), and the largest
 * such magnitude on a panel floors <em>both</em> the reported error
 * estimate and that panel's acceptance budget consistently. A panel that
 * has hit this floor is accepted, with an honestly elevated error estimate,
 * rather than bisected forever chasing a target that was never reachable.
 *
 * <h2>Why the aliasing cross-check stays live even during subtraction</h2>
 * Subtraction here works by re-parsing into a single, self-consistent
 * smoothed expression -- both the Gauss-Kronrod samples and the AD
 * cross-check's derivatives come from the <em>same</em> evaluator
 * throughout a call. There is no risk of comparing a Gauss-Kronrod result
 * against a cross-check computed from a different (still-singular)
 * function, so unlike a design that evaluates GK on manually-adjusted
 * sample values while keeping AD pointed at the original expression, there
 * is no need to disable the cross-check for the whole call just because one
 * endpoint was subtracted.
 *
 * <h2>Progress reporting</h2>
 * {@link ProgressListener#onProgress} reports the fraction of the original
 * interval's <em>width</em> that has been finalized so far -- panels are
 * never revisited once finalized, so this is monotonically increasing and
 * reaches exactly 1.0 on completion, an honest completion estimate rather
 * than a guessed percentage.
 *
 * <h2>Constants</h2>
 * The 15-point Kronrod / 7-point Gauss abscissae and weights are the
 * standard QUADPACK {@code dqk15.f} table (Fullerton, Bell Labs, 1981).
 *
 * <h2>Thread-safety</h2>
 * A single instance is stateful (reused scratch buffers, a reused worklist
 * stack) and not thread-safe. Give each worker thread its own instance
 * built from the same expression string.
 */
public class Integrator {

    private static final Logger LOG = Logger.getLogger(Integrator.class.getName());

    // ------------------------------------------------------------------
    // QUADPACK dqk15 constants (Netlib/QUADPACK dqk15.f; Fullerton, 1981).
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

    public static final int MAX_CROSS_CHECK_ORDER = 60;
    public static final double DEFAULT_SINGULARITY_CONSISTENCY_TOL = 0.05;
    public static final double CANCELLATION_FLOOR_FACTOR = 100.0;

    // ------------------------------------------------------------------
    // Exceptions
    // ------------------------------------------------------------------

    public static class TaylorIntegrationException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        public TaylorIntegrationException(String message) { super(message); }
        public TaylorIntegrationException(String message, Throwable cause) { super(message, cause); }
    }

    public static final class NonIntegrableSingularityException extends TaylorIntegrationException {
        private static final long serialVersionUID = 1L;
        public final double location;
        public final double panelWidth;

        NonIntegrableSingularityException(double location, double panelWidth, String reason, Throwable cause) {
            super(String.format(
                    "Non-integrable singularity near x=%.15g (panel width %.3e): %s. "
                            + "If a partial result is acceptable, build with strictSingularities(false).",
                    location, panelWidth, reason), cause);
            this.location = location;
            this.panelWidth = panelWidth;
        }
    }

    public static final class EarlyDivergenceDetectedException extends TaylorIntegrationException {
        private static final long serialVersionUID = 1L;
        public final double location;
        public final double empiricalPower;

        EarlyDivergenceDetectedException(double location, double empiricalPower, String whichEndpoint) {
            super(String.format(
                    "Likely divergent integral detected at the %s (x=%.15g): empirical power n≈%.4f "
                            + "(n<=-1 means the ordinary integral does not converge there). Detected via a "
                            + "two-scale, two-formula probe before any bisection -- if this is a false "
                            + "positive, retry with strictSingularities(false) or singularitySubtraction(false).",
                    whichEndpoint, location, empiricalPower));
            this.location = location;
            this.empiricalPower = empiricalPower;
        }
    }

    // ------------------------------------------------------------------
    // Public diagnostics types
    // ------------------------------------------------------------------

    public static final class Singularity {
        public final double location;
        public final double width;
        public final String reason;

        Singularity(double location, double width, String reason) {
            this.location = location;
            this.width = width;
            this.reason = reason;
        }

        @Override
        public String toString() {
            return String.format("flag near x=%.15g (panel width %.3e): %s", location, width, reason);
        }
    }

    public static final class Result {
        public final double value;
        public final double estimatedError;
        public final int panelCount;
        public final int maxDepth;
        public final List<Singularity> singularities;
        public final List<String> appliedSubtractions;

        Result(double value, double estimatedError, int panelCount, int maxDepth,
               List<Singularity> singularities, List<String> appliedSubtractions) {
            this.value = value;
            this.estimatedError = estimatedError;
            this.panelCount = panelCount;
            this.maxDepth = maxDepth;
            this.singularities = singularities;
            this.appliedSubtractions = appliedSubtractions;
        }

        @Override
        public String toString() {
            return String.format(
                    "value=%.15g  errEst~%.3e  panels=%d  maxDepth=%d  flags=%d%s%s",
                    value, estimatedError, panelCount, maxDepth, singularities.size(),
                    singularities.isEmpty() ? "" : ("  " + singularities),
                    appliedSubtractions.isEmpty() ? "" : ("  subtractions=" + appliedSubtractions));
        }
    }

    /** {@code fractionComplete}: fraction of the original interval's width finalized so far -- see class javadoc. */
    public interface ProgressListener {
        ProgressListener NO_OP = (panels, depth, estimate, fractionComplete) -> { };

        void onProgress(int panelsProcessed, int currentMaxDepth, double currentEstimate, double fractionComplete);
    }

    // ------------------------------------------------------------------
    // Builder
    // ------------------------------------------------------------------

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String expression;
        private String wrtVar = "x";
        private int crossCheckOrder = 8;
        private double absTol = 1e-11;
        private double relTol = 1e-11;
        private int maxDepth = 40;
        private boolean strictSingularities = true;
        private boolean crossCheckEnabled = true;
        private double crossCheckLooseness = 50.0;
        private boolean singularitySubtractionEnabled = true;
        private double singularityConsistencyTol = DEFAULT_SINGULARITY_CONSISTENCY_TOL;
        private ProgressListener progressListener = ProgressListener.NO_OP;
        private int progressEveryNPanels = 1;

        public Builder expression(String expr) { this.expression = expr; return this; }
        public Builder wrtVar(String wrtVar) { this.wrtVar = wrtVar; return this; }
        public Builder crossCheckOrder(int order) { this.crossCheckOrder = order; return this; }
        public Builder absoluteTolerance(double absTol) { this.absTol = absTol; return this; }
        public Builder relativeTolerance(double relTol) { this.relTol = relTol; return this; }
        public Builder maxDepth(int maxDepth) { this.maxDepth = maxDepth; return this; }
        public Builder strictSingularities(boolean strict) { this.strictSingularities = strict; return this; }
        public Builder crossCheckEnabled(boolean enabled) { this.crossCheckEnabled = enabled; return this; }
        public Builder crossCheckLooseness(double looseness) { this.crossCheckLooseness = looseness; return this; }
        public Builder singularitySubtraction(boolean enabled) { this.singularitySubtractionEnabled = enabled; return this; }
        public Builder singularityConsistencyTolerance(double tol) { this.singularityConsistencyTol = tol; return this; }

        public Builder progressListener(ProgressListener listener) {
            this.progressListener = listener != null ? listener : ProgressListener.NO_OP;
            return this;
        }

        public Builder progressEveryNPanels(int n) {
            this.progressEveryNPanels = Math.max(1, n);
            return this;
        }

        public Integrator build() {
            if (expression == null || expression.trim().isEmpty()) {
                throw new IllegalArgumentException("expression(...) must be set before build()");
            }
            return new Integrator(expression, wrtVar, crossCheckOrder, absTol, relTol, maxDepth,
                    strictSingularities, crossCheckEnabled, crossCheckLooseness, singularitySubtractionEnabled,
                    singularityConsistencyTol, progressListener, progressEveryNPanels);
        }
    }

    // ------------------------------------------------------------------
    // Configuration and pre-sized (zero-allocation-hot-path) state
    // ------------------------------------------------------------------

    private final String rawExpression;
    private final String wrtVar;
    private final int crossCheckOrder;
    private final int maxDepth;
    private final double absTol;
    private final double relTol;
    private final boolean strictSingularities;
    private final boolean crossCheckEnabled;
    private final double crossCheckLooseness;
    private final boolean singularitySubtractionEnabled;
    private final double singularityConsistencyTol;
    private final ProgressListener progressListener;
    private final int progressEveryNPanels;

    private final AutoDiffNEvaluator baseAd;

    private final double[] edgeBuf;          // single function value, order-0 AD result
    private final double[] exponentProbeBuf; // [f, f', f''] for endpoint singularity probing (order 2)
    private final double[] crossDeriv;       // Taylor cross-check derivatives, size crossCheckOrder+1
    private final double[] crossFactorial;   // factorial table, size crossCheckOrder+2
    private final double[] gkFv1;            // GK: f(centr-absc) at the 7 non-center nodes
    private final double[] gkFv2;            // GK: f(centr+absc) at the 7 non-center nodes

    private double[] stackA;
    private double[] stackB;
    private double[] stackTol;
    private int[] stackDepth;
    private int stackSize;

    // Written by evaluateGK; read by the caller. No object allocation.
    private double lastCenterValue;
    private double lastGKValue;
    private double lastGKErrEst;
    private double lastCancellationFloor;

    // Written by probeExponentAt instead of returning a boxed result object.
    private boolean probeValid;
    private double probeN;
    private double probeC;

    // Active endpoint-subtraction state for the current integrate() call.
    private boolean leftActive, rightActive;
    private double leftX0, leftN, leftC;
    private double rightX0, rightN, rightC;
    private AutoDiffNEvaluator pendingSmoothedEvaluator;

    private double lastErrorEstimate;
    private int lastPanelCount;
    private int lastMaxDepthReached;
    private final List<Singularity> singularities;
    private final List<String> appliedSubtractions;

    private Integrator(String rawExpression, String wrtVar, int crossCheckOrder,
                                         double absTol, double relTol, int maxDepth,
                                         boolean strictSingularities, boolean crossCheckEnabled,
                                         double crossCheckLooseness, boolean singularitySubtractionEnabled,
                                         double singularityConsistencyTol, ProgressListener progressListener,
                                         int progressEveryNPanels) {
        if (crossCheckOrder < 2 || (crossCheckOrder % 2) != 0) {
            throw new IllegalArgumentException("crossCheckOrder must be a positive even number");
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
        if (singularityConsistencyTol < 0) {
            throw new IllegalArgumentException("singularityConsistencyTol must be >= 0");
        }
        if (wrtVar == null || wrtVar.trim().isEmpty()) {
            throw new IllegalArgumentException("wrtVar must be a non-empty variable name");
        }
        if (crossCheckLooseness <= 0) {
            throw new IllegalArgumentException("crossCheckLooseness must be > 0");
        }

        this.rawExpression = rawExpression;
        this.wrtVar = wrtVar;
        this.crossCheckOrder = crossCheckOrder;
        this.absTol = absTol;
        this.relTol = relTol;
        this.maxDepth = maxDepth;
        this.strictSingularities = strictSingularities;
        this.crossCheckEnabled = crossCheckEnabled;
        this.crossCheckLooseness = crossCheckLooseness;
        this.singularitySubtractionEnabled = singularitySubtractionEnabled;
        this.singularityConsistencyTol = singularityConsistencyTol;
        this.progressListener = progressListener;
        this.progressEveryNPanels = progressEveryNPanels;

        MathExpression me = new MathExpression(rawExpression);
        this.baseAd = new AutoDiffNEvaluator(me, crossCheckOrder);

        this.edgeBuf = new double[1];
        this.exponentProbeBuf = new double[3];
        this.crossDeriv = new double[crossCheckOrder + 1];
        this.gkFv1 = new double[7];
        this.gkFv2 = new double[7];

        this.crossFactorial = new double[crossCheckOrder + 2];
        crossFactorial[0] = 1.0;
        for (int i = 1; i < crossFactorial.length; i++) {
            crossFactorial[i] = crossFactorial[i - 1] * i;
        }

        int cap = maxDepth + 4;
        this.stackA = new double[cap];
        this.stackB = new double[cap];
        this.stackTol = new double[cap];
        this.stackDepth = new int[cap];
        this.stackSize = 0;

        this.singularities = new ArrayList<>();
        this.appliedSubtractions = new ArrayList<>();
    }

    // ------------------------------------------------------------------
    // Fixed-capacity primitive stack helpers
    // ------------------------------------------------------------------

    private void stackPush(double a, double b, double tol, int depth) {
        if (stackSize == stackA.length) {
            growStack(); // defensive; provably unreachable given capacity maxDepth+4
        }
        stackA[stackSize] = a;
        stackB[stackSize] = b;
        stackTol[stackSize] = tol;
        stackDepth[stackSize] = depth;
        stackSize++;
    }

    private void growStack() {
        int newCap = stackA.length * 2;
        stackA = Arrays.copyOf(stackA, newCap);
        stackB = Arrays.copyOf(stackB, newCap);
        stackTol = Arrays.copyOf(stackTol, newCap);
        stackDepth = Arrays.copyOf(stackDepth, newCap);
    }

    // ------------------------------------------------------------------
    // Top-level integrate()
    // ------------------------------------------------------------------

    public double integrate(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            throw new IllegalArgumentException(
                    "bounds must be finite (got [" + a + ", " + b + "]); this integrator does not support "
                            + "infinite domains -- apply your own variable substitution first, or use a "
                            + "quadrature rule designed for semi-infinite/infinite intervals");
        }

        singularities.clear();
        appliedSubtractions.clear();
        lastErrorEstimate = 0.0;
        lastPanelCount = 0;
        lastMaxDepthReached = 0;
        leftActive = false;
        rightActive = false;
        pendingSmoothedEvaluator = null;

        if (a == b) {
            return 0.0;
        }
        boolean negate = b < a;
        double lo = negate ? b : a;
        double hi = negate ? a : b;
        double width0 = hi - lo;

        double analyticCorrection = 0.0;
        if (singularitySubtractionEnabled) {
            analyticCorrection = detectAndSubtract(lo, hi, width0);
        }
        AutoDiffNEvaluator activeEvaluator = pendingSmoothedEvaluator != null ? pendingSmoothedEvaluator : baseAd;
        boolean crossCheckActiveThisCall = crossCheckEnabled; // stays on even during subtraction; see class javadoc

        double rawSum = runAdaptiveLoop(activeEvaluator, lo, hi, negate, analyticCorrection, crossCheckActiveThisCall);
        double finalValue = rawSum + analyticCorrection;
        return negate ? -finalValue : finalValue;
    }

    public Result integrateResult(double a, double b) {
        double value = integrate(a, b);
        return new Result(value, lastErrorEstimate, lastPanelCount, lastMaxDepthReached,
                new ArrayList<>(singularities), new ArrayList<>(appliedSubtractions));
    }

    public double getLastErrorEstimate() { return lastErrorEstimate; }
    public int getLastPanelCount() { return lastPanelCount; }
    public int getLastMaxDepth() { return lastMaxDepthReached; }
    public List<Singularity> getSingularities() { return Collections.unmodifiableList(singularities); }
    public List<String> getAppliedSubtractions() { return Collections.unmodifiableList(appliedSubtractions); }

    // ------------------------------------------------------------------
    // Endpoint singularity detection, subtraction, verification
    // (call-level: at most twice per integrate() call, never per panel)
    // ------------------------------------------------------------------

    /**
     * Single-probe, two-formula exponent estimate at one distance {@code s} from an endpoint,
     * from one order-2 AD evaluation. Writes {@link #probeValid}, {@link #probeN}, {@link #probeC}
     * -- no object allocation. probeValid=false if evaluation failed, values were non-finite, or
     * the two formulas (n1 from f'/f, n2 from f''/f') disagree beyond {@code singularityConsistencyTol}.
     */
    private void probeExponentAt(AutoDiffNEvaluator evaluator, double endpoint, double dir, double s) {
        probeValid = false;
        double x = endpoint + dir * s;
        try {
            evaluator.evaluateRPN(wrtVar, x, 2, exponentProbeBuf);
        } catch (ArithmeticException domainEx) {
            return;
        }
        double f = exponentProbeBuf[0];
        double fp = exponentProbeBuf[1];
        double fpp = exponentProbeBuf[2];
        if (!Double.isFinite(f) || !Double.isFinite(fp) || !Double.isFinite(fpp) || f == 0.0 || fp == 0.0) {
            return;
        }
        double t = dir * s;
        double n1 = t * fp / f;
        double n2 = 1.0 + t * fpp / fp;
        if (!Double.isFinite(n1) || !Double.isFinite(n2) || Math.abs(n1 - n2) > singularityConsistencyTol) {
            return;
        }
        double n = 0.5 * (n1 + n2);
        double c = f / Math.pow(s, n);
        if (!Double.isFinite(c)) {
            return;
        }
        probeN = n;
        probeC = c;
        probeValid = true;
    }

    /**
     * Full endpoint check: two-formula consistency (in {@link #probeExponentAt}) at each of two
     * scales two orders of magnitude apart, plus agreement across those two scales. Writes the
     * same {@link #probeValid}/{@link #probeN}/{@link #probeC} fields as the final combined result.
     */
    private void probeEndpoint(AutoDiffNEvaluator evaluator, double endpoint, double dir, double intervalWidth) {
        double s1 = Math.max(intervalWidth * 1e-4, 1e-12);
        double s2 = Math.max(intervalWidth * 1e-6, 1e-14);
        if (s2 >= s1) {
            s2 = s1 * 1e-2;
        }

        probeExponentAt(evaluator, endpoint, dir, s1);
        boolean valid1 = probeValid;
        double n1 = probeN, c1 = probeC;

        probeExponentAt(evaluator, endpoint, dir, s2);
        boolean valid2 = probeValid;
        double n2 = probeN, c2 = probeC;

        probeValid = false;
        if (!valid1 || !valid2 || Math.abs(n1 - n2) > singularityConsistencyTol) {
            return;
        }
        double n = 0.5 * (n1 + n2);
        if (n <= -1.0 + 1e-6) {
            probeN = n;
            probeValid = true; // nonIntegrable case; caller checks n<=-1 itself, probeC unused
            return;
        }
        if (Math.abs(n) < 0.05) {
            return;
        }
        probeN = n;
        probeC = c2; // closer probe's constant estimate
        probeValid = true;
    }

    private static String fmt(double d) {
        if (!Double.isFinite(d)) {
            throw new IllegalStateException("cannot embed non-finite constant " + d + " in an expression");
        }
        return new BigDecimal(Double.toString(d)).toPlainString();
    }

    /**
     * Detects and, if verified, applies endpoint singularity subtraction. Sets {@link
     * #pendingSmoothedEvaluator}/{@link #leftActive}/{@link #rightActive} (and their C/n/x0
     * fields) as side effects on success; leaves them at their reset defaults (base evaluator
     * stays active) otherwise. Returns the analytic correction to add back. Never lets an
     * internal issue with this optimization abort the call -- except a confidently-detected
     * non-integrable singularity in strict mode, which is the one deliberate early throw.
     */
    private double detectAndSubtract(double lo, double hi, double width0) {
        try {
            probeEndpoint(baseAd, lo, +1.0, width0);
            boolean leftFound = probeValid;
            double leftFoundN = probeN, leftFoundC = probeC;

            probeEndpoint(baseAd, hi, -1.0, width0);
            boolean rightFound = probeValid;
            double rightFoundN = probeN, rightFoundC = probeC;

            if (leftFound && leftFoundN <= -1.0 + 1e-6) {
                if (strictSingularities) {
                    throw new EarlyDivergenceDetectedException(lo, leftFoundN, "lower bound");
                }
                LOG.warning(String.format("Likely divergent integral at lower bound x=%.15g (n≈%.4f); "
                        + "proceeding in lenient mode without subtraction there", lo, leftFoundN));
                leftFound = false;
            }
            if (rightFound && rightFoundN <= -1.0 + 1e-6) {
                if (strictSingularities) {
                    throw new EarlyDivergenceDetectedException(hi, rightFoundN, "upper bound");
                }
                LOG.warning(String.format("Likely divergent integral at upper bound x=%.15g (n≈%.4f); "
                        + "proceeding in lenient mode without subtraction there", hi, rightFoundN));
                rightFound = false;
            }

            if (!leftFound && !rightFound) {
                return 0.0;
            }

            StringBuilder expr = new StringBuilder("(").append(rawExpression).append(")");
            double correction = 0.0;
            if (leftFound) {
                expr.append(" - (").append(fmt(leftFoundC)).append(")*(").append(wrtVar)
                        .append("-(").append(fmt(lo)).append("))^(").append(fmt(leftFoundN)).append(")");
                correction += leftFoundC / (leftFoundN + 1.0) * Math.pow(width0, leftFoundN + 1.0);
            }
            if (rightFound) {
                expr.append(" - (").append(fmt(rightFoundC)).append(")*((").append(fmt(hi)).append(")-")
                        .append(wrtVar).append(")^(").append(fmt(rightFoundN)).append(")");
                correction += rightFoundC / (rightFoundN + 1.0) * Math.pow(width0, rightFoundN + 1.0);
            }

            MathExpression smoothedMe = new MathExpression(expr.toString());
            AutoDiffNEvaluator smoothedAd = new AutoDiffNEvaluator(smoothedMe, crossCheckOrder);

            String failReason = null;
            if (leftFound) {
                probeEndpoint(smoothedAd, lo, +1.0, width0);
                if (probeValid && probeN <= leftFoundN + 0.01) {
                    failReason = "lower bound residual still singular after subtraction (n≈" + probeN + ")";
                }
            }
            if (failReason == null && rightFound) {
                probeEndpoint(smoothedAd, hi, -1.0, width0);
                if (probeValid && probeN <= rightFoundN + 0.01) {
                    failReason = "upper bound residual still singular after subtraction (n≈" + probeN + ")";
                }
            }

            if (failReason != null) {
                LOG.fine("singularity subtraction attempted but failed verification (" + failReason
                        + "); integrating the original expression instead");
                return 0.0;
            }

            pendingSmoothedEvaluator = smoothedAd;
            if (leftFound) {
                leftActive = true;
                leftX0 = lo; leftN = leftFoundN; leftC = leftFoundC;
                appliedSubtractions.add(String.format(
                        "lower bound x=%.10g: subtracted %.10g*(x-a)^%.6f analytically (verified)",
                        lo, leftFoundC, leftFoundN));
            }
            if (rightFound) {
                rightActive = true;
                rightX0 = hi; rightN = rightFoundN; rightC = rightFoundC;
                appliedSubtractions.add(String.format(
                        "upper bound x=%.10g: subtracted %.10g*(b-x)^%.6f analytically (verified)",
                        hi, rightFoundC, rightFoundN));
            }
            return correction;

        } catch (EarlyDivergenceDetectedException propagate) {
            throw propagate;
        } catch (RuntimeException detectionIssue) {
            LOG.fine("singularity detection/subtraction skipped due to: " + detectionIssue);
            pendingSmoothedEvaluator = null;
            leftActive = false;
            rightActive = false;
            return 0.0;
        }
    }

    /** Magnitude of whatever endpoint singular term(s) were subtracted at x. Pure arithmetic, no allocation. */
    private double subtractedMagnitudeAt(double x) {
        double mag = 0.0;
        if (leftActive) {
            double s = x - leftX0;
            if (s > 0) {
                mag += Math.abs(leftC) * Math.pow(s, leftN);
            }
        }
        if (rightActive) {
            double s = rightX0 - x;
            if (s > 0) {
                mag += Math.abs(rightC) * Math.pow(s, rightN);
            }
        }
        return mag;
    }

    // ------------------------------------------------------------------
    // Adaptive Gauss-Kronrod + cross-check loop -- the true hot path.
    // Zero heap allocation for the entire common (no domain error, no
    // aliasing flag) case, for any number of panels.
    // ------------------------------------------------------------------

    private double runAdaptiveLoop(AutoDiffNEvaluator evaluator, double lo, double hi,
                                    boolean negateForDisplay, double analyticCorrectionForDisplay,
                                    boolean crossCheckActiveThisCall) {
        double width0 = hi - lo;
        double minWidth = Math.max(width0 * 1e-13, Double.MIN_NORMAL);
        double scale = estimateScale(evaluator, lo, hi);
        double effectiveAbsTol = Math.max(absTol, relTol * scale);

        stackSize = 0;
        stackPush(lo, hi, effectiveAbsTol, 0);

        double sum = 0.0, comp = 0.0;
        double errAccum = 0.0;
        double resolvedWidth = 0.0;
        int panelCount = 0;
        int maxDepthSeen = 0;

        while (stackSize > 0) {
            stackSize--;
            double pa = stackA[stackSize];
            double pb = stackB[stackSize];
            double pTol = stackTol[stackSize];
            int pDepth = stackDepth[stackSize];
            if (pDepth > maxDepthSeen) {
                maxDepthSeen = pDepth;
            }

            boolean threwDomainError;
            String domainErrorMsg = null;
            Throwable domainErrorCause = null;
            try {
                evaluateGK(evaluator, pa, pb, width0);
                threwDomainError = false;
            } catch (ArithmeticException domainEx) {
                threwDomainError = true;
                domainErrorMsg = domainEx.getMessage();
                domainErrorCause = domainEx;
            }

            boolean widthExhausted = (pb - pa) <= minWidth;
            boolean depthExhausted = pDepth >= maxDepth;

            if (threwDomainError) {
                if (widthExhausted || depthExhausted) {
                    recordSingularityOrThrow(pa, pb, "AD domain error: " + domainErrorMsg, domainErrorCause);
                    resolvedWidth += (pb - pa);
                    panelCount++;
                    reportProgress(panelCount, maxDepthSeen, sum, resolvedWidth, width0,
                            negateForDisplay, analyticCorrectionForDisplay);
                    continue;
                }
                double m = 0.5 * (pa + pb);
                stackPush(pa, m, pTol * 0.5, pDepth + 1);
                stackPush(m, pb, pTol * 0.5, pDepth + 1);
                continue;
            }

            double budget = Math.max(pTol, Math.max(relTol * Math.abs(lastGKValue), lastCancellationFloor));
            boolean gkOk = Double.isFinite(lastGKValue) && Double.isFinite(lastGKErrEst) && lastGKErrEst <= budget;

            boolean aliasFlagged = false;
            String aliasReason = null;
            if (gkOk && crossCheckActiveThisCall) {
                try {
                    double crossVal = taylorCrossCheck(evaluator, pa, pb);
                    if (Double.isFinite(crossVal)) {
                        double disagreement = Math.abs(lastGKValue - crossVal);
                        double crossTol = Math.max(budget, absTol) * crossCheckLooseness;
                        if (disagreement > crossTol) {
                            aliasFlagged = true;
                            aliasReason = String.format(
                                    "Gauss-Kronrod (%.15g) and an independent order-%d Taylor cross-check (%.15g) "
                                            + "disagree by %.3e, past the %.1fx aliasing-guard threshold",
                                    lastGKValue, crossCheckOrder, crossVal, disagreement, crossCheckLooseness);
                        }
                    }
                } catch (RuntimeException ignore) {
                    // fail open: GK's own result stands, no cross-check signal available
                }
            }

            boolean accepted = gkOk && !aliasFlagged;

            if (accepted) {
                double y = lastGKValue - comp;
                double t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                errAccum += lastGKErrEst;
                panelCount++;
                resolvedWidth += (pb - pa);
                reportProgress(panelCount, maxDepthSeen, sum, resolvedWidth, width0,
                        negateForDisplay, analyticCorrectionForDisplay);
            } else if (widthExhausted || depthExhausted) {
                if (Double.isFinite(lastGKValue)) {
                    double y = lastGKValue - comp;
                    double t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                    if (Double.isFinite(lastGKErrEst)) {
                        errAccum += lastGKErrEst;
                    }
                    panelCount++;
                    String reason = aliasFlagged ? aliasReason
                            : "Gauss-Kronrod error estimate exceeded tolerance and could not be reduced further";
                    reason += widthExhausted ? " (minimum panel width reached)" : " (maximum recursion depth reached)";
                    singularities.add(new Singularity(0.5 * (pa + pb), pb - pa, reason));
                    resolvedWidth += (pb - pa);
                    reportProgress(panelCount, maxDepthSeen, sum, resolvedWidth, width0,
                            negateForDisplay, analyticCorrectionForDisplay);
                } else {
                    try {
                        double fb = simpsonFallback(evaluator, pa, pb);
                        if (!Double.isFinite(fb)) {
                            throw new ArithmeticException("Simpson fallback produced a non-finite value");
                        }
                        double y = fb - comp;
                        double t = sum + y;
                        comp = (t - sum) - y;
                        sum = t;
                        panelCount++;
                        singularities.add(new Singularity(0.5 * (pa + pb), pb - pa,
                                "Gauss-Kronrod result was non-finite; resolved via emergency Simpson fallback"));
                        resolvedWidth += (pb - pa);
                        reportProgress(panelCount, maxDepthSeen, sum, resolvedWidth, width0,
                                negateForDisplay, analyticCorrectionForDisplay);
                    } catch (ArithmeticException fallbackEx) {
                        recordSingularityOrThrow(pa, pb,
                                "AD domain error during emergency fallback: " + fallbackEx.getMessage(), fallbackEx);
                        resolvedWidth += (pb - pa);
                        panelCount++;
                        reportProgress(panelCount, maxDepthSeen, sum, resolvedWidth, width0,
                                negateForDisplay, analyticCorrectionForDisplay);
                    }
                }
            } else {
                double m = 0.5 * (pa + pb);
                stackPush(pa, m, pTol * 0.5, pDepth + 1);
                stackPush(m, pb, pTol * 0.5, pDepth + 1);
            }
        }

        lastErrorEstimate = errAccum;
        lastPanelCount = panelCount;
        lastMaxDepthReached = maxDepthSeen;
        return sum;
    }

    private void reportProgress(int panelCount, int maxDepthSeen, double rawSum, double resolvedWidth,
                                 double width0, boolean negateForDisplay, double analyticCorrectionForDisplay) {
        if (progressListener == ProgressListener.NO_OP) {
            return;
        }
        if (panelCount % progressEveryNPanels != 0) {
            return;
        }
        double liveEstimate = rawSum + analyticCorrectionForDisplay;
        if (negateForDisplay) {
            liveEstimate = -liveEstimate;
        }
        double fraction = width0 > 0 ? Math.min(1.0, resolvedWidth / width0) : 1.0;
        progressListener.onProgress(panelCount, maxDepthSeen, liveEstimate, fraction);
    }

    private void recordSingularityOrThrow(double pa, double pb, String reason, Throwable cause) {
        double location = 0.5 * (pa + pb);
        double width = pb - pa;
        if (strictSingularities) {
            throw new NonIntegrableSingularityException(location, width, reason, cause);
        }
        LOG.warning(String.format(
                "Unresolvable singularity near x=%.15g (panel width %.3e): %s -- lenient mode: "
                        + "this panel's contribution is omitted from the result",
                location, width, reason));
        singularities.add(new Singularity(location, width, reason));
    }

    /**
     * 7-point Gauss / 15-point Kronrod evaluation with the QUADPACK resasc/resabs-scaled error
     * estimate, additionally floored by the cancellation-noise term when an endpoint subtraction
     * is active (see class javadoc). Writes {@link #lastCenterValue}, {@link #lastGKValue},
     * {@link #lastGKErrEst}, {@link #lastCancellationFloor}. Pure arithmetic plus 15 calls to
     * {@link #evalAt} -- no heap allocation on the success path.
     */
    private void evaluateGK(AutoDiffNEvaluator evaluator, double a, double b, double fullWidth) {
        double centr = 0.5 * (a + b);
        double hlgth = 0.5 * (b - a);
        boolean trackSubtraction = leftActive || rightActive;
        double maxSubtractedMag = 0.0;

        double fc = evalAt(evaluator, centr);
        if (trackSubtraction) {
            maxSubtractedMag = Math.max(maxSubtractedMag, subtractedMagnitudeAt(centr));
        }
        double resg = WG[3] * fc;
        double resk = WGK[7] * fc;
        double resabs = Math.abs(resk);

        for (int j = 0; j < 3; j++) {
            int jtw = 2 * (j + 1) - 1;
            double absc = hlgth * XGK[jtw];
            double f1 = evalAt(evaluator, centr - absc);
            double f2 = evalAt(evaluator, centr + absc);
            if (trackSubtraction) {
                maxSubtractedMag = Math.max(maxSubtractedMag, subtractedMagnitudeAt(centr - absc));
                maxSubtractedMag = Math.max(maxSubtractedMag, subtractedMagnitudeAt(centr + absc));
            }
            gkFv1[jtw] = f1;
            gkFv2[jtw] = f2;
            double fsum = f1 + f2;
            resg += WG[j] * fsum;
            resk += WGK[jtw] * fsum;
            resabs += WGK[jtw] * (Math.abs(f1) + Math.abs(f2));
        }
        for (int j = 0; j < 4; j++) {
            int jtwm1 = 2 * (j + 1) - 2;
            double absc = hlgth * XGK[jtwm1];
            double f1 = evalAt(evaluator, centr - absc);
            double f2 = evalAt(evaluator, centr + absc);
            if (trackSubtraction) {
                maxSubtractedMag = Math.max(maxSubtractedMag, subtractedMagnitudeAt(centr - absc));
                maxSubtractedMag = Math.max(maxSubtractedMag, subtractedMagnitudeAt(centr + absc));
            }
            gkFv1[jtwm1] = f1;
            gkFv2[jtwm1] = f2;
            double fsum = f1 + f2;
            resk += WGK[jtwm1] * fsum;
            resabs += WGK[jtwm1] * (Math.abs(f1) + Math.abs(f2));
        }

        double reskh = resk * 0.5;
        double resasc = WGK[7] * Math.abs(fc - reskh);
        for (int j = 0; j < 7; j++) {
            resasc += WGK[j] * (Math.abs(gkFv1[j] - reskh) + Math.abs(gkFv2[j] - reskh));
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

        double cancellationFloor = trackSubtraction ? EPMACH * CANCELLATION_FLOOR_FACTOR * maxSubtractedMag : 0.0;
        if (cancellationFloor > abserr) {
            abserr = cancellationFloor;
        }

        lastCenterValue = fc;
        lastGKValue = result;
        lastGKErrEst = abserr;
        lastCancellationFloor = cancellationFloor;
    }

    /** Independent low-order Taylor cross-check about the panel midpoint. No allocation. */
    private double taylorCrossCheck(AutoDiffNEvaluator evaluator, double a, double b) {
        double m = 0.5 * (a + b);
        double h = 0.5 * (b - a);
        double h2 = h * h;

        evaluator.evaluateRPN(wrtVar, m, crossCheckOrder, crossDeriv);

        double sum = 0.0;
        double hp = h;
        for (int i = 0; i <= crossCheckOrder; i += 2) {
            sum += 2.0 * crossDeriv[i] * hp / crossFactorial[i + 1];
            hp *= h2;
        }
        return sum;
    }

    private double simpsonFallback(AutoDiffNEvaluator evaluator, double a, double b) {
        double fm = lastCenterValue;
        double fa = evalAt(evaluator, a);
        double fb = evalAt(evaluator, b);
        return (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    }

    /** Order-0 AD evaluation of f(x); throws ArithmeticException with a named cause on domain errors. No allocation on success. */
    private double evalAt(AutoDiffNEvaluator evaluator, double x) {
        evaluator.evaluateRPN(wrtVar, x, 0, edgeBuf);
        return edgeBuf[0];
    }

    /** Rough magnitude scale of f, used to turn relTol into an absolute budget. No allocation (unrolled, no array literal). */
    private double estimateScale(AutoDiffNEvaluator evaluator, double a, double b) {
        double m = 0.5 * (a + b);
        double total = 0.0;
        int count = 0;
        try {
            total += Math.abs(evalAt(evaluator, a));
            count++;
        } catch (RuntimeException ignore) { }
        try {
            total += Math.abs(evalAt(evaluator, m));
            count++;
        } catch (RuntimeException ignore) { }
        try {
            total += Math.abs(evalAt(evaluator, b));
            count++;
        } catch (RuntimeException ignore) { }
        return count > 0 ? total / count : 1.0;
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

    private static Integrator make(String expr) {
        return builder().expression(expr).absoluteTolerance(1e-10).relativeTolerance(1e-10).maxDepth(40).build();
    }

    public static void main(String[] args) {
        expect("sin(x) on [0,pi]", make("sin(x)").integrate(0, Math.PI), 2.0, 1e-9);
        expect("exp(x) on [0,10]", make("exp(x)").integrate(0, 10), Math.exp(10) - 1, 1e-8);

        // The blind spot this whole lineage exists to close.
        Integrator bump = builder().expression("exp(-3200*(x-1.3)^2)")
                .absoluteTolerance(1e-10).relativeTolerance(1e-10).build();
        expect("Gaussian bump (aliasing cross-check)", bump.integrate(0, 2), 0.03133285343288750, 1e-8);

        // Left endpoint, negative exponent.
        Integrator leftSing = make("1/sqrt(x)");
        expect("1/sqrt(x) on [0,1] (left, n=-0.5)", leftSing.integrate(0, 1), 2.0, 1e-9);
        System.out.println("  " + leftSing.getAppliedSubtractions());

        // Right endpoint, negative exponent.
        Integrator rightSing = make("1/sqrt(1-x)");
        expect("1/sqrt(1-x) on [0,1] (right, n=-0.5)", rightSing.integrate(0, 1), 2.0, 1e-9);
        System.out.println("  " + rightSing.getAppliedSubtractions());

        // Positive exponent: bounded, but still worth subtracting (new vs. negative-only designs).
        Integrator posSing = make("sqrt(x)");
        expect("sqrt(x) on [0,1] (n=+0.5, positive-exponent subtraction)", posSing.integrate(0, 1), 2.0 / 3.0, 1e-9);
        System.out.println("  " + posSing.getAppliedSubtractions());

        // Both endpoints singular simultaneously.
        Integrator bothEnds = make("1/sqrt(x)+1/sqrt(1-x)");
        expect("1/sqrt(x)+1/sqrt(1-x) on [0,1] (both endpoints)", bothEnds.integrate(0, 1), 4.0, 1e-8);
        System.out.println("  " + bothEnds.getAppliedSubtractions());

        // The cancellation-floor regression test: real O(1) remainder under a subtracted singularity.
        Integrator mixed = make("1/sqrt(x) + cos(x)");
        expect("1/sqrt(x)+cos(x) on [0,1] (cancellation-floor case)", mixed.integrate(0, 1), 2.0 + Math.sin(1.0), 1e-8);
        System.out.println("  achieved errEst=" + mixed.getLastErrorEstimate()
                + " (elevated vs the 1e-10 request is expected/honest here, not a failure)");

        Integrator cbrtI = make("1/(x^(1/3))");
        expect("1/x^(1/3) on [0,1] (n=-1/3)", cbrtI.integrate(0, 1), 1.5, 1e-8);

        // Non-integrable: strict mode fails before any bisection.
        Integrator divergent = make("1/x^1.5");
        try {
            divergent.integrate(0, 1);
            System.out.println("[FAIL] 1/x^1.5 on [0,1] should have thrown early");
            failCount++;
        } catch (EarlyDivergenceDetectedException expectedEx) {
            System.out.println("[PASS] 1/x^1.5 on [0,1] correctly threw early: " + expectedEx.getMessage());
            passCount++;
        }

        // Interior pole: subtraction machinery must not interfere (neither endpoint is singular).
        Integrator poleStrict = make("1/(x-0.5)");
        try {
            poleStrict.integrate(0.1, 0.9);
            System.out.println("[FAIL] 1/(x-0.5) through interior pole should have thrown");
            failCount++;
        } catch (NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/(x-0.5) through interior pole correctly threw: " + expectedEx.getMessage());
            passCount++;
        }

        // Progress listener: real one, honest width-resolved fraction metric.
        System.out.println();
        System.out.println("Progress-listener demo: sin(300x) on [0,5]");
        Integrator.ProgressListener listener = (panels, depth, estimate, fraction) -> {
            if (panels % 25 == 0) {
                System.out.printf("  panels=%-5d depth=%-3d estimate=%.10g  %5.1f%% of domain resolved%n",
                        panels, depth, estimate, fraction * 100.0);
            }
        };
        Integrator oscillatory = builder().expression("sin(300*x)")
                .absoluteTolerance(1e-9).relativeTolerance(1e-9).progressListener(listener).build();
        double oscVal = oscillatory.integrate(0, 5);
        expect("sin(300x) on [0,5] (with progress listener)", oscVal, (1 - Math.cos(1500.0)) / 300.0, 1e-6);

        try {
            make("x").integrate(0, Double.POSITIVE_INFINITY);
            System.out.println("[FAIL] infinite bound should have thrown");
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