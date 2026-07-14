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
import com.github.gbenroscience.parser.MathExpression.Token;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Composite definite integrator built directly on top of {@link AutoDiffNEvaluator}.
 *
 * <h2>Mathematical basis</h2>
 * For a panel [a, b], expand f as a Taylor series about the panel's own
 * midpoint m = (a+b)/2 and integrate the series term by term. With
 * h = (b-a)/2 and t = x - m:
 *
 * <pre>
 *   int_{a}^{b} f(x) dx = int_{-h}^{h} f(m + t) dt
 *                       = sum_{i=0}^{infinity}  f^(i)(m) * ( h^(i+1) - (-h)^(i+1) ) / (i+1)!
 * </pre>
 *
 * which is exactly the series in the reference image
 * ( sum_{i=0}^{infinity} f^(i)(x1) . dx^(i+1) / (i+1)! ), just anchored at
 * the panel midpoint instead of its left edge. For odd i, (i+1) is even, so
 * (-h)^(i+1) = h^(i+1) and the term cancels exactly -- every odd-order
 * derivative drops out for free, so a Taylor order of N carries the
 * truncation accuracy of order N+1 at no extra cost. Only even i survive:
 *
 * <pre>
 *   int_{a}^{b} f(x) dx = sum_{i even}  2 * f^(i)(m) * h^(i+1) / (i+1)!
 * </pre>
 *
 * <h2>Scope: this is a finite-interval, local-Taylor quadrature</h2>
 * This class deliberately does <b>not</b> attempt semi-infinite or infinite
 * domains. Doing that well needs a fundamentally different local rule
 * (e.g. tanh-sinh / double-exponential quadrature); bolting an infinite-domain
 * substitution onto a local Taylor-series panel rule just relocates the
 * problem to a Jacobian that blows up at the mapped endpoint, which this
 * integrator would then have no better a tool for than the singularity
 * handling described below. {@link #integrate(double, double)} rejects
 * non-finite bounds outright rather than silently producing a number from a
 * mismatched technique.
 *
 * <h2>Singularity policy: strict by default</h2>
 * Two situations can happen inside a panel, and they are handled differently
 * on purpose:
 * <ul>
 *   <li><b>The function is genuinely undefined</b> (division by zero, a log
 *       domain error, etc: an {@link ArithmeticException} from the AD
 *       evaluator). If this persists down to the minimum panel width, the
 *       integral may not even converge there (e.g. a true pole). By default
 *       ({@code strictSingularities = true}) this throws
 *       {@link NonIntegrableSingularityException} rather than silently
 *       dropping that panel's contribution -- the way a fixed epsilon-width
 *       pole excision would. Call {@link Builder#strictSingularities(boolean)}
 *       with {@code false} to opt into the lenient behavior instead: the
 *       panel is flagged as a {@link Singularity} and its contribution is
 *       omitted, and integration continues.</li>
 *   <li><b>The function is perfectly well-defined but the local Taylor series
 *       does not converge on this panel</b> (h exceeds the local radius of
 *       convergence -- common right next to, but not exactly at, a
 *       singularity, e.g. near x=0 for {@code sqrt(x)}). Dropping this
 *       contribution to zero would bias the result for no good reason, since
 *       f is computable here. Instead the integrator falls back to a plain
 *       Simpson's-rule estimate on that one small panel, which needs only
 *       three function values and no series convergence at all. This is
 *       still recorded as a {@link Singularity} for transparency (the caller
 *       can see exactly where the accuracy guarantee was downgraded), but it
 *       does not stop integration and does not throw.</li>
 * </ul>
 *
 * <h2>Other production hardening</h2>
 * <ul>
 *   <li><b>No silent factorial overflow.</b> Factorials are stored as
 *       {@code double}, not {@code long}: {@code long} factorials silently
 *       wrap around ~21! with no warning, which would make every division
 *       past that order silently wrong. {@code order} is capped at
 *       {@link #MAX_ORDER} (163! stays comfortably inside double range;
 *       171! would overflow to {@code Infinity}).</li>
 *   <li><b>No NaN/Infinity poisoning the running sum.</b> A panel outcome is
 *       only ever added to the total if it is finite. Anything else is
 *       routed through the singularity/fallback policy above instead of
 *       being summed as-is.</li>
 *   <li><b>Fail-fast on malformed input.</b> {@link IllegalStateException}
 *       (malformed RPN) and {@link UnsupportedOperationException} (an
 *       operator the AD evaluator doesn't implement) are programming/config
 *       errors, not recoverable numerical situations -- they are
 *       deliberately <i>not</i> caught here and propagate immediately rather
 *       than triggering endless bisection against an error that will never
 *       go away.</li>
 *   <li><b>Integer powers of h computed iteratively</b>, not via
 *       {@code Math.pow}, which is both faster and avoids extra
 *       transcendental rounding noise feeding into a sum that is supposed to
 *       be accurate to many digits.</li>
 * </ul>
 *
 * <h2>Zero-allocation steady state</h2>
 * As in the earlier revision: the panel worklist is four fixed-size
 * primitive arrays sized {@code maxDepth + 4} (provably sufficient -- an
 * explicit stack that pops one node and, on split, pushes exactly two
 * children holds at most one pending sibling per tree level, bounding stack
 * size by tree height, not panel count); {@code evaluatePanel} writes into
 * instance fields instead of returning a boxed outcome; {@code deriv},
 * {@code factorial}, and the Simpson-fallback scratch buffer are allocated
 * once in the constructor. {@link Singularity} objects remain the one
 * deliberate exception, since they only occur on the rare diagnostic path.
 *
 * <h2>Thread-safety</h2>
 * A single instance is stateful and reused across panels; it is not
 * thread-safe. Give each worker thread its own instance built from the same
 * {@code rpnTokens} array (which is only read).
 */
public class TurboIntegratorEngine {

    private static final Logger LOG = Logger.getLogger(TurboIntegratorEngine.class.getName());

    /** Largest allowed Taylor order: keeps factorial(order+3) comfortably inside double range. */
    public static final int MAX_ORDER = 160;

    // ------------------------------------------------------------------
    // Exceptions
    // ------------------------------------------------------------------

    /** Base type for integration failures that are not programming errors. */
    public static class TaylorIntegrationException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        public TaylorIntegrationException(String message) {
            super(message);
        }

        public TaylorIntegrationException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Thrown (in strict mode, the default) when a panel could not be safely
     * resolved down to the minimum panel width or maximum recursion depth
     * because the integrand is genuinely undefined there -- e.g. an actual
     * pole. Thrown instead of silently omitting that panel's contribution,
     * since a pole inside the interval generally means the ordinary
     * (non-principal-value) integral simply does not converge.
     */
    public static final class NonIntegrableSingularityException extends TaylorIntegrationException {
        private static final long serialVersionUID = 1L;
        public final double location;
        public final double panelWidth;

        NonIntegrableSingularityException(double location, double panelWidth, String reason, Throwable cause) {
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

    /** A panel flagged during integration: either omitted (lenient mode) or resolved via fallback. */
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

    /** Immutable snapshot returned by the allocating {@link #integrateResult} convenience API. */
    public static final class Result {
        public final double value;
        public final double estimatedError;
        public final int panelCount;
        public final int maxDepth;
        public final List<Singularity> singularities;

        Result(double value, double estimatedError, int panelCount, int maxDepth, List<Singularity> singularities) {
            this.value = value;
            this.estimatedError = estimatedError;
            this.panelCount = panelCount;
            this.maxDepth = maxDepth;
            this.singularities = singularities;
        }

        @Override
        public String toString() {
            return String.format(
                    "value=%.15g  errEst~%.3e  panels=%d  maxDepth=%d  flags=%d%s",
                    value, estimatedError, panelCount, maxDepth, singularities.size(),
                    singularities.isEmpty() ? "" : ("  " + singularities));
        }
    }

    // ------------------------------------------------------------------
    // Builder
    // ------------------------------------------------------------------

    public static Builder builder() {
        return new Builder();
    }

    /** Fluent construction; the recommended way to configure a new integrator. */
    public static final class Builder {
        private int order = 8;
        private double absTol = 1e-10;
        private double relTol = 1e-10;
        private int maxDepth = 40;
        private boolean strictSingularities = true;

        public Builder order(int order) {
            this.order = order;
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

        /** Default true: unresolvable singularities throw rather than silently truncating the result. */
        public Builder strictSingularities(boolean strict) {
            this.strictSingularities = strict;
            return this;
        }

        public TurboIntegratorEngine build(Token[] rpnTokens, String wrtVar) {
            return new TurboIntegratorEngine(rpnTokens, wrtVar, order, absTol, relTol, maxDepth, strictSingularities);
        }

        public TurboIntegratorEngine build(String expr, String wrtVar) {
            MathExpression me = new MathExpression(expr);
            return build(me.getCachedPostfix(), wrtVar);
        }
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    private final AutoDiffNEvaluator ad;
    private final String wrtVar;
    private final int order;
    private final int adOrder;
    private final int maxDepth;
    private final double absTol;
    private final double relTol;
    private final boolean strictSingularities;

    private final double[] deriv;    // reused scratch for AD output at panel midpoints, size adOrder+1
    private final double[] edgeBuf;  // reused scratch for single-value (order 0) edge samples
    private final double[] factorial; // factorial[i] = i! as double, sized through adOrder+3

    private double[] stackA;
    private double[] stackB;
    private double[] stackTol;
    private int[] stackDepth;
    private int stackSize;

    private double lastValue;
    private double lastErrEst;
    private boolean lastConverging;

    private double lastErrorEstimate;
    private int lastPanelCount;
    private int lastMaxDepthReached;
    private final List<Singularity> singularities;

    public TurboIntegratorEngine(Token[] rpnTokens, String wrtVar, int order,
                                  double absTol, double relTol, int maxDepth) {
        this(rpnTokens, wrtVar, order, absTol, relTol, maxDepth, true);
    }

    public TurboIntegratorEngine(Token[] rpnTokens, String wrtVar, int order,
                                  double absTol, double relTol, int maxDepth, boolean strictSingularities) {
        if (order < 2 || (order % 2) != 0) {
            throw new IllegalArgumentException("order must be a positive even number (midpoint expansion cancels odd terms)");
        }
        if (order > MAX_ORDER) {
            throw new IllegalArgumentException("order must be <= " + MAX_ORDER + " (factorial(order+3) would approach double overflow beyond this)");
        }
        if (maxDepth < 1) {
            throw new IllegalArgumentException("maxDepth must be >= 1");
        }
        if (absTol < 0 || relTol < 0) {
            throw new IllegalArgumentException("tolerances must be >= 0");
        }
        if (wrtVar == null || wrtVar.trim().isEmpty()) {
            throw new IllegalArgumentException("wrtVar must be a non-empty variable name");
        }

        this.order = order;
        this.adOrder = order + 2;
        this.ad = new AutoDiffNEvaluator(rpnTokens, adOrder);
        this.wrtVar = wrtVar;
        this.absTol = absTol;
        this.relTol = relTol;
        this.maxDepth = maxDepth;
        this.strictSingularities = strictSingularities;

        this.deriv = new double[adOrder + 1];
        this.edgeBuf = new double[1];

        this.factorial = new double[adOrder + 4];
        factorial[0] = 1.0;
        for (int i = 1; i < factorial.length; i++) {
            factorial[i] = factorial[i - 1] * i;
        }

        int cap = maxDepth + 4;
        this.stackA = new double[cap];
        this.stackB = new double[cap];
        this.stackTol = new double[cap];
        this.stackDepth = new int[cap];
        this.stackSize = 0;

        this.singularities = new ArrayList<>();
    }

    public static TurboIntegratorEngine forExpression(String expr, String wrtVar, int order,
                                                        double absTol, double relTol, int maxDepth) {
        return forExpression(expr, wrtVar, order, absTol, relTol, maxDepth, true);
    }

    public static TurboIntegratorEngine forExpression(String expr, String wrtVar, int order,
                                                        double absTol, double relTol, int maxDepth,
                                                        boolean strictSingularities) {
        MathExpression me = new MathExpression(expr);
        return new TurboIntegratorEngine(me.getCachedPostfix(), wrtVar, order, absTol, relTol, maxDepth, strictSingularities);
    }

    // ------------------------------------------------------------------
    // Fixed-capacity primitive stack helpers
    // ------------------------------------------------------------------

    private void stackPush(double a, double b, double tol, int depth) {
        if (stackSize == stackA.length) {
            growStack(); // defensive; should be unreachable, see class javadoc
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
    // Integration
    // ------------------------------------------------------------------

    /**
     * Integrates f from a to b, splitting panels adaptively.
     *
     * @throws IllegalArgumentException          if a or b is not finite, or a == b is not intended
     *                                            (a == b is allowed and simply returns 0)
     * @throws NonIntegrableSingularityException in strict mode (the default), if some panel's
     *                                            contribution could not be safely resolved
     */
    public double integrate(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            throw new IllegalArgumentException(
                    "bounds must be finite (got [" + a + ", " + b + "]); this integrator uses a local "
                            + "Taylor expansion and does not support infinite domains -- apply your own "
                            + "variable substitution first, or use a quadrature rule designed for "
                            + "semi-infinite/infinite intervals (e.g. tanh-sinh)");
        }

        singularities.clear();
        lastErrorEstimate = 0.0;
        lastPanelCount = 0;
        lastMaxDepthReached = 0;

        if (a == b) {
            return 0.0;
        }
        boolean negate = b < a;
        double lo = negate ? b : a;
        double hi = negate ? a : b;

        double width0 = hi - lo;
        double minWidth = Math.max(width0 * 1e-13, Double.MIN_NORMAL);
        double scale = estimateScale(lo, hi);
        double effectiveAbsTol = Math.max(absTol, relTol * scale);

        stackSize = 0;
        stackPush(lo, hi, effectiveAbsTol, 0);

        double sum = 0.0, comp = 0.0; // Kahan compensated summation
        double errAccum = 0.0;
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
                evaluatePanel(pa, pb);
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
                    continue; // lenient mode: nothing safely computable here, contribute nothing
                }
                double m = 0.5 * (pa + pb);
                stackPush(pa, m, pTol * 0.5, pDepth + 1);
                stackPush(m, pb, pTol * 0.5, pDepth + 1);
                continue;
            }

            boolean accepted = lastConverging
                    && Double.isFinite(lastValue)
                    && Double.isFinite(lastErrEst)
                    && lastErrEst <= pTol;

            if (accepted) {
                double y = lastValue - comp;
                double t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                errAccum += lastErrEst;
                panelCount++;
            } else if (widthExhausted || depthExhausted) {
                // Series did not converge here, but f itself evaluated fine: fall back to a
                // low-order, always-safe estimate instead of trusting a possibly-divergent
                // Taylor sum, and instead of silently contributing zero.
                double fallback;
                try {
                    fallback = simpsonFallback(pa, pb);
                    if (!Double.isFinite(fallback)) {
                        throw new ArithmeticException("Simpson fallback produced a non-finite value");
                    }
                } catch (ArithmeticException fallbackEx) {
                    recordSingularityOrThrow(pa, pb, "AD domain error during fallback sampling: " + fallbackEx.getMessage(), fallbackEx);
                    continue;
                }

                double y = fallback - comp;
                double t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                if (Double.isFinite(lastErrEst)) {
                    errAccum += lastErrEst;
                }
                panelCount++;

                singularities.add(new Singularity(0.5 * (pa + pb), pb - pa,
                        (widthExhausted
                                ? "minimum panel width reached without series convergence"
                                : "maximum recursion depth reached without series convergence")
                                + " -- resolved with a Simpson fallback instead of the Taylor series"));
                LOG.fine(() -> String.format(
                        "Taylor series did not converge on [%.15g, %.15g]; used Simpson fallback", pa, pb));
            } else {
                double m = 0.5 * (pa + pb);
                stackPush(pa, m, pTol * 0.5, pDepth + 1);
                stackPush(m, pb, pTol * 0.5, pDepth + 1);
            }
        }

        lastErrorEstimate = errAccum;
        lastPanelCount = panelCount;
        lastMaxDepthReached = maxDepthSeen;

        return negate ? -sum : sum;
    }

    /** Allocating convenience overload: same computation, packaged as one immutable {@link Result}. */
    public Result integrateResult(double a, double b) {
        double value = integrate(a, b);
        return new Result(value, lastErrorEstimate, lastPanelCount, lastMaxDepthReached,
                new ArrayList<>(singularities));
    }

    public double getLastErrorEstimate() { return lastErrorEstimate; }
    public int getLastPanelCount() { return lastPanelCount; }
    public int getLastMaxDepth() { return lastMaxDepthReached; }
    public boolean isStrictSingularities() { return strictSingularities; }

    /** Live view of the flags recorded during the most recent {@link #integrate} call. */
    public List<Singularity> getSingularities() {
        return Collections.unmodifiableList(singularities);
    }

    /**
     * Either throws (strict mode) or logs + records a {@link Singularity} and returns normally
     * (lenient mode), leaving the panel's contribution omitted from the sum.
     */
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
     * Expands f about the panel midpoint and integrates its Taylor series
     * term-by-term, writing the outcome into {@link #lastValue},
     * {@link #lastErrEst}, {@link #lastConverging} (no object allocation).
     * Odd-order terms vanish identically by symmetry. The two probe
     * derivatives beyond {@code order} (fetched via {@code adOrder}) give a
     * genuine next-term error estimate and feed the ratio/divergence test.
     * Integer powers of h are advanced iteratively (h *= h^2 each step)
     * rather than via {@code Math.pow}.
     */
    private void evaluatePanel(double a, double b) {
        double m = 0.5 * (a + b);
        double h = 0.5 * (b - a);
        double h2 = h * h;

        ad.evaluateRPN(wrtVar, m, adOrder, deriv);

        double sum = 0.0;
        double lastTerm = 0.0;  // term at i = order      (last accepted even term)
        double prevTerm = 0.0;  // term at i = order - 2
        double hp = h;          // h^(i+1) for the current i, starting at i=0 => h^1

        for (int i = 0; i <= order; i += 2) {
            double term = 2.0 * deriv[i] * hp / factorial[i + 1];
            sum += term;
            prevTerm = lastTerm;
            lastTerm = term;
            hp *= h2; // advance from h^(i+1) to h^(i+3), i.e. to (i+2)+1
        }

        // hp is now h^(order+3): exactly what the next even-order term (i = order+2) needs.
        // (order+1 is odd, so it contributes nothing -- that's why adOrder = order+2.)
        double probeTerm = 2.0 * deriv[order + 2] * hp / factorial[order + 3];
        double errEst = Math.abs(probeTerm);

        boolean converging = (lastTerm == 0.0) || (Math.abs(probeTerm) <= Math.abs(lastTerm));
        if (prevTerm != 0.0 && Math.abs(lastTerm) > Math.abs(prevTerm)) {
            converging = false;
        }

        lastValue = sum;
        lastErrEst = errEst;
        lastConverging = converging;
    }

    /**
     * Low-order, always-safe fallback for a panel whose Taylor series didn't converge but whose
     * function value is well-defined. Reuses deriv[0] (f at the panel midpoint, just computed by
     * the evaluatePanel call that preceded this one) plus two fresh order-0 edge samples.
     */
    private double simpsonFallback(double a, double b) {
        double fm = deriv[0];
        ad.evaluateRPN(wrtVar, a, 0, edgeBuf);
        double fa = edgeBuf[0];
        ad.evaluateRPN(wrtVar, b, 0, edgeBuf);
        double fb = edgeBuf[0];
        return (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    }

    /** Rough magnitude scale of f over [a,b], used to turn relTol into an absolute budget. No allocation. */
    private double estimateScale(double a, double b) {
        double m = 0.5 * (a + b);
        double total = 0.0;
        int count = 0;

        if (sampleAbs(a) == 1) { total += Math.abs(edgeBuf[0]); count++; }
        if (sampleAbs(m) == 1) { total += Math.abs(edgeBuf[0]); count++; }
        if (sampleAbs(b) == 1) { total += Math.abs(edgeBuf[0]); count++; }

        return count > 0 ? total / count : 1.0;
    }

    /** Evaluates f(x) (order 0) into edgeBuf[0]; returns 1 on success, 0 if x hit a domain error. */
    private int sampleAbs(double x) {
        try {
            ad.evaluateRPN(wrtVar, x, 0, edgeBuf);
            return 1;
        } catch (RuntimeException ignore) {
            return 0;
        }
    }

    // ------------------------------------------------------------------
    // Self-tests (verified closed-form references, not just printed output)
    // ------------------------------------------------------------------

    private static int passCount = 0;
    private static int failCount = 0;

    private static void expect(String label, double actual, double expected, double tol) {
        double err = Math.abs(actual - expected);
        boolean ok = err <= tol;
        System.out.printf("[%s] %-40s actual=%.15g expected=%.15g err=%.3e (tol=%.1e)%n",
                ok ? "PASS" : "FAIL", label, actual, expected, err, tol);
        if (ok) passCount++; else failCount++;
    }

    public static void main(String[] args) {
        TurboIntegratorEngine sinI = forExpression("sin(x)", "x", 8, 1e-10, 1e-10, 40);
        expect("sin(x) on [0,pi]", sinI.integrate(0, Math.PI), 2.0, 1e-9);

        TurboIntegratorEngine x2 = forExpression("x^2", "x", 8, 1e-10, 1e-10, 40);
        expect("x^2 on [0,1]", x2.integrate(0, 1), 1.0 / 3.0, 1e-12);

        TurboIntegratorEngine invX = forExpression("1/x", "x", 8, 1e-10, 1e-10, 40);
        expect("1/x on [1,2]", invX.integrate(1, 2), Math.log(2), 1e-9);

        TurboIntegratorEngine sqrtX = forExpression("sqrt(x)", "x", 8, 1e-9, 1e-9, 60);
        double sqrtVal = sqrtX.integrate(0, 4);
        expect("sqrt(x) on [0,4]", sqrtVal, 16.0 / 3.0, 1e-8);
        if (!sqrtX.getSingularities().isEmpty()) {
            System.out.println("  used fallback: " + sqrtX.getSingularities());
        }

        TurboIntegratorEngine tanX = forExpression("tan(x)", "x", 10, 1e-10, 1e-10, 40);
        expect("tan(x) on [0,pi/4]", tanX.integrate(0, Math.PI / 4), 0.5 * Math.log(2), 1e-9);

        // Strict mode (default): integrating straight through a genuine pole must fail loudly.
        TurboIntegratorEngine poleStrict = forExpression("1/(x-0.5)", "x", 8, 1e-10, 1e-10, 40, true);
        try {
            poleStrict.integrate(0.1, 0.9);
            System.out.println("[FAIL] 1/(x-0.5) through pole (strict) should have thrown");
            failCount++;
        } catch (NonIntegrableSingularityException expectedEx) {
            System.out.println("[PASS] 1/(x-0.5) through pole (strict) correctly threw: " + expectedEx.getMessage());
            passCount++;
        }

        // Lenient mode: same integral instead returns a partial result, explicitly flagged.
        TurboIntegratorEngine poleLenient = forExpression("1/(x-0.5)", "x", 8, 1e-10, 1e-10, 40, false);
        double lenientResult = poleLenient.integrate(0.1, 0.9);
        boolean lenientOk = Double.isFinite(lenientResult) && !poleLenient.getSingularities().isEmpty();
        System.out.printf("[%s] 1/(x-0.5) through pole (lenient) returned %.6g flagged with %d flag(s)%n",
                lenientOk ? "PASS" : "FAIL", lenientResult, poleLenient.getSingularities().size());
        if (lenientOk) passCount++; else failCount++;

        // Reject non-finite bounds rather than pretending to support them.
        try {
            forExpression("x", "x", 8, 1e-10, 1e-10, 40).integrate(0, Double.POSITIVE_INFINITY);
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