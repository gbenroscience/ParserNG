/*
 * Copyright 2026 oluwagbemirojiboye.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
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
import java.util.Collections;
import java.util.List;

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
 * <h2>Resilience features</h2>
 * <ul>
 *   <li><b>Real error estimate.</b> The AD evaluator is run two orders past
 *       what is kept, so the "next" term used for the error bound is an
 *       actually-computed quantity, not an extrapolated guess.</li>
 *   <li><b>Divergence / ratio test.</b> If the newest even-order term is not
 *       smaller than the one before it, the panel is wider than the local
 *       radius of convergence, and only shrinking the panel can help --
 *       that forces an immediate split, independent of the raw error
 *       estimate.</li>
 *   <li><b>Tolerance-splitting bisection.</b> Each split halves the
 *       tolerance budget handed to its two children, so the global error
 *       stays bounded as the panel count grows.</li>
 *   <li><b>Hard termination guarantees.</b> A minimum panel width and a
 *       maximum recursion depth are enforced; any panel given up on is
 *       reported back as a flagged {@link Singularity}.</li>
 *   <li><b>Domain-exception resilience.</b> {@link ArithmeticException}s
 *       thrown by the AD evaluator (ln/sqrt/asin/atanh/pow domain errors,
 *       division by zero, atan2 at the origin) are caught mid-panel and
 *       treated as "there is a singularity here" -- triggering a bisection
 *       instead of aborting.</li>
 *   <li><b>Kahan compensated summation</b> across all accepted panels.</li>
 * </ul>
 *
 * <h2>Zero-allocation steady state</h2>
 * Like {@code AutoDiffNEvaluator} itself, the hot path here does not
 * allocate once the instance is constructed:
 * <ul>
 *   <li>The panel worklist is an explicit stack over four fixed-size
 *       primitive arrays ({@code double[]}/{@code int[]}), not a
 *       {@code Deque<Panel>} of boxed objects. Its capacity is
 *       {@code maxDepth + 4}, which is provably sufficient: an explicit
 *       stack that pops one node and (on split) pushes exactly two
 *       children can hold at most one pending sibling per tree level, so
 *       its size never exceeds the tree height. (A defensive geometric
 *       grow-via-arraycopy path exists in case that reasoning is ever
 *       violated by a future change, but it should never trigger.)</li>
 *   <li>{@code evaluatePanel} writes its result into three instance fields
 *       ({@code lastValue}, {@code lastErrEst}, {@code lastConverging})
 *       instead of returning a boxed outcome object.</li>
 *   <li>{@code deriv}, {@code factorial} are allocated once, in the
 *       constructor, and reused for the life of the instance -- same as
 *       the underlying {@code AutoDiffNEvaluator}'s own scratch arrays.</li>
 *   <li>The {@link #integrate(double, double)} entry point returns a
 *       primitive {@code double} and stores diagnostics
 *       ({@link #getLastErrorEstimate()}, {@link #getLastPanelCount()},
 *       {@link #getLastMaxDepth()}, {@link #getSingularities()}) in
 *       reused instance state -- no per-call {@code Result} object.</li>
 *   <li>The one intentional exception is {@link Singularity}: these are
 *       only allocated on the rare, exceptional path where a panel could
 *       not be resolved, so they don't affect steady-state throughput. The
 *       backing {@code List<Singularity>} itself is allocated once in the
 *       constructor and {@code clear()}-ed (not reallocated) between
 *       calls.</li>
 * </ul>
 * A convenience {@link #integrateResult(double, double)} overload is also
 * provided for callers who want a single immutable snapshot object instead
 * (mirroring {@code AutoDiffNEvaluator}'s own dual API of a fill-buffer
 * {@code evaluateRPN(..., resultOut)} plus an allocating convenience
 * overload) -- that path does allocate one {@link Result} per call.
 *
 * <h2>Thread-safety</h2>
 * A single {@code TurboIntegratorEngine} instance is stateful (reused
 * scratch buffers, reused worklist arrays) and therefore not thread-safe.
 * To integrate in parallel, give each worker thread its own instance; they
 * can safely share the same {@code rpnTokens} array, which is only read.
 */
public class TurboIntegratorEngine {

    // ------------------------------------------------------------------
    // Public diagnostics types
    // ------------------------------------------------------------------

    /** A panel the integrator could not resolve to tolerance and had to give up on. */
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
            return String.format("singularity near x=%.15g (panel width %.3e): %s", location, width, reason);
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
                    "value=%.15g  errEst~%.3e  panels=%d  maxDepth=%d  singularities=%d%s",
                    value, estimatedError, panelCount, maxDepth, singularities.size(),
                    singularities.isEmpty() ? "" : ("  " + singularities));
        }
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    private final AutoDiffNEvaluator ad;
    private final String wrtVar;
    private final int order;    // even Taylor order kept per panel (effective accuracy order+1)
    private final int adOrder;  // order + 2: gives two probe terms for error / divergence checks
    private final int maxDepth;
    private final double absTol;
    private final double relTol;

    private final double[] deriv;   // reused scratch for AD output, size adOrder+1
    private final long[] factorial; // factorial[i] = i!, sized through adOrder+3

    // ---- fixed-capacity primitive worklist stack (replaces Deque<Panel>) ----
    private double[] stackA;
    private double[] stackB;
    private double[] stackTol;
    private int[] stackDepth;
    private int stackSize;

    // ---- fields evaluatePanel() writes into instead of returning an object ----
    private double lastValue;
    private double lastErrEst;
    private boolean lastConverging;

    // ---- reused diagnostics from the most recent integrate() call ----
    private double lastErrorEstimate;
    private int lastPanelCount;
    private int lastMaxDepthReached;
    private final List<Singularity> singularities; // allocated once, cleared between calls

    /**
     * @param rpnTokens compiled expression, as produced by {@code MathExpression.getCachedPostfix()}
     * @param wrtVar    the integration variable's name
     * @param order     even Taylor order to keep per panel (e.g. 8); higher orders
     *                  converge faster on smooth panels but cost more per AD call
     * @param absTol    absolute error tolerance for the whole integral
     * @param relTol    relative error tolerance, scaled against a rough magnitude
     *                  estimate of f over [a, b]
     * @param maxDepth  hard cap on bisection depth per panel (guarantees termination
     *                  and bounds the fixed worklist stack size)
     */
    public TurboIntegratorEngine(Token[] rpnTokens, String wrtVar, int order,
                                  double absTol, double relTol, int maxDepth) {
        if (order < 2 || (order % 2) != 0) {
            throw new IllegalArgumentException("order must be a positive even number (midpoint expansion cancels odd terms)");
        }
        if (maxDepth < 1) {
            throw new IllegalArgumentException("maxDepth must be >= 1");
        }
        if (absTol < 0 || relTol < 0) {
            throw new IllegalArgumentException("tolerances must be >= 0");
        }

        this.order = order;
        this.adOrder = order + 2;
        this.ad = new AutoDiffNEvaluator(rpnTokens, adOrder);
        this.wrtVar = wrtVar;
        this.absTol = absTol;
        this.relTol = relTol;
        this.maxDepth = maxDepth;

        this.deriv = new double[adOrder + 1];

        this.factorial = new long[adOrder + 4];
        factorial[0] = 1;
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
        MathExpression me = new MathExpression(expr);
        return new TurboIntegratorEngine(me.getCachedPostfix(), wrtVar, order, absTol, relTol, maxDepth);
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
        stackA = java.util.Arrays.copyOf(stackA, newCap);
        stackB = java.util.Arrays.copyOf(stackB, newCap);
        stackTol = java.util.Arrays.copyOf(stackTol, newCap);
        stackDepth = java.util.Arrays.copyOf(stackDepth, newCap);
    }

    // ------------------------------------------------------------------
    // Integration
    // ------------------------------------------------------------------

    /**
     * Integrates f from a to b, splitting panels adaptively. Zero-allocation
     * in steady state (see class javadoc). Handles a &gt; b by negating.
     * Diagnostics from this call are available afterward via
     * {@link #getLastErrorEstimate()}, {@link #getLastPanelCount()},
     * {@link #getLastMaxDepth()}, and {@link #getSingularities()}.
     */
    public double integrate(double a, double b) {
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

            boolean singular;
            try {
                evaluatePanel(pa, pb);
                singular = false;
            } catch (ArithmeticException singularEx) {
                singular = true;
                double w = pb - pa;
                if (w <= minWidth || pDepth >= maxDepth) {
                    singularities.add(new Singularity(0.5 * (pa + pb), w,
                            "AD domain error: " + singularEx.getMessage()));
                    continue; // sliver too small to safely resolve; contributes nothing further
                }
                double m = 0.5 * (pa + pb);
                stackPush(pa, m, pTol * 0.5, pDepth + 1);
                stackPush(m, pb, pTol * 0.5, pDepth + 1);
                continue;
            }

            boolean widthExhausted = (pb - pa) <= minWidth;
            boolean depthExhausted = pDepth >= maxDepth;
            boolean accepted = lastErrEst <= pTol && lastConverging;

            if (accepted || widthExhausted || depthExhausted) {
                double y = lastValue - comp;
                double t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                errAccum += lastErrEst;
                panelCount++;

                if (!accepted) {
                    singularities.add(new Singularity(0.5 * (pa + pb), pb - pa,
                            widthExhausted
                                    ? "minimum panel width reached without series convergence"
                                    : "maximum recursion depth reached without series convergence"));
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

    /** Live view of the singularities found during the most recent {@link #integrate} call. */
    public List<Singularity> getSingularities() {
        return Collections.unmodifiableList(singularities);
    }

    /**
     * Expands f about the panel midpoint and integrates its Taylor series
     * term-by-term, writing the outcome into {@link #lastValue},
     * {@link #lastErrEst}, {@link #lastConverging} (no object allocation).
     * Odd-order terms vanish identically by symmetry. The two probe
     * derivatives beyond {@code order} (fetched via {@code adOrder}) give a
     * genuine next-term error estimate and feed the ratio/divergence test,
     * rather than a heuristic truncation guess.
     */
    private void evaluatePanel(double a, double b) {
        double m = 0.5 * (a + b);
        double h = 0.5 * (b - a);

        ad.evaluateRPN(wrtVar, m, adOrder, deriv);

        double sum = 0.0;
        double lastTerm = 0.0;  // term at i = order      (last accepted even term)
        double prevTerm = 0.0;  // term at i = order - 2

        for (int i = 0; i <= order; i += 2) {
            double term = 2.0 * deriv[i] * Math.pow(h, i + 1) / factorial[i + 1];
            sum += term;
            prevTerm = lastTerm;
            lastTerm = term;
        }

        // Next even-order term (order+1 is odd => zero contribution; the
        // real next nonzero term sits at order+2, which is why adOrder = order+2).
        double probeTerm = 2.0 * deriv[order + 2] * Math.pow(h, order + 3) / factorial[order + 3];
        double errEst = Math.abs(probeTerm);

        boolean converging = (lastTerm == 0.0) || (Math.abs(probeTerm) <= Math.abs(lastTerm));
        if (prevTerm != 0.0 && Math.abs(lastTerm) > Math.abs(prevTerm)) {
            // successive kept terms are growing, not shrinking: h is beyond
            // the local radius of convergence regardless of what the probe says
            converging = false;
        }

        lastValue = sum;
        lastErrEst = errEst;
        lastConverging = converging;
    }

    /** Rough magnitude scale of f over [a,b], used to turn relTol into an absolute budget. No allocation. */
    private double estimateScale(double a, double b) {
        double m = 0.5 * (a + b);
        double total = 0.0;
        int count = 0;

        if (sampleAbs(a, deriv) == 1) { total += Math.abs(deriv[0]); count++; }
        if (sampleAbs(m, deriv) == 1) { total += Math.abs(deriv[0]); count++; }
        if (sampleAbs(b, deriv) == 1) { total += Math.abs(deriv[0]); count++; }

        return count > 0 ? total / count : 1.0;
    }

    /** Evaluates f(x) (order 0) into out[0]; returns 1 on success, 0 if x hit a domain error. No allocation. */
    private int sampleAbs(double x, double[] out) {
        try {
            ad.evaluateRPN(wrtVar, x, 0, out);
            return 1;
        } catch (RuntimeException ignore) {
            return 0;
        }
    }

    // ------------------------------------------------------------------
    // Demo / smoke test
    // ------------------------------------------------------------------

    private static void demo(String expr, String var, double a, double b, int order) {
        TurboIntegratorEngine integ = TurboIntegratorEngine.forExpression(
                expr, var, order, 1e-10, 1e-10, 40);
        double value = integ.integrate(a, b);
        System.out.printf("integral of %-24s over [%s, %s] : value=%.15g errEst~%.3e panels=%d maxDepth=%d singularities=%d%n",
                expr, a, b, value, integ.getLastErrorEstimate(), integ.getLastPanelCount(),
                integ.getLastMaxDepth(), integ.getSingularities().size());
        if (!integ.getSingularities().isEmpty()) {
            System.out.println("  " + integ.getSingularities());
        }
    }

    public static void main(String[] args) {
        demo("sin(x)", "x", 0, Math.PI, 8);                 // smooth, known answer = 2
        demo("x^3+3*x^2-5*x-8*atan2(2*x,3)", "x", -1, 2, 8); // matches AutoDiffNEvaluator's own test case
        demo("1/x", "x", 0.1, 5, 8);                         // benign but strongly curved near 0.1
        demo("sqrt(x)", "x", 0, 4, 8);                       // derivative singular at the left endpoint
        demo("tan(x)", "x", 0, 1.4, 10);                     // approaches the pole at pi/2 ~ 1.5708
        demo("sin(50*x)", "x", 0, 2, 8);                     // highly oscillatory: forces many panels
    }
}