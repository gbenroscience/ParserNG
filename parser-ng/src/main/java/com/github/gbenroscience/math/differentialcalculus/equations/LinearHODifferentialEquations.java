package com.github.gbenroscience.math.differentialcalculus.equations;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

/**
 * Compilation layer backing diffeqnLinearHO(...)/diffeqnPathLinearHO(...) — the
 * ParserNG functional form for a general linear n-th order equation:
 *
 * A(t)*y_n + B(t)*y_(n-1) + C(t)*y_(n-2) + ... + F(t)*y_0 = g(t)
 *
 * where A, B, C, ..., F are each functions of t (or constants), and g(t) is an
 * optional forcing term (omit it, or pass null, for the homogeneous case where
 * the right-hand side is 0).
 *
 * This class does no new numerical work of its own. It divides the equation
 * through by the leading coefficient A(t), builds the resulting top-derivative
 * expression as a single MethodHandle, and hands that straight to the existing
 * HigherOrderODE / CompanionSystemHandles machinery — exactly the same
 * companion-system reduction diffeqnHO already uses. The only thing this class
 * adds is the coefficient-list convenience and the leading-coefficient domain
 * check described in buildTopDerivative's javadoc.
 *
 * <h2>Coefficient handle convention</h2>
 * Each coefficient (and the optional forcing term) is a MethodHandle matching
 * the standard (double(), double())void descriptor, reading only from
 * vars(tSlot) (coefficients are functions of t, never of the state Y — that is
 * what makes the equation linear) and writing its single value to outValue(0).
 *
 * <h2>The one thing this class cannot fully guarantee</h2>
 * A(t) may cross zero somewhere strictly inside (t0, tEnd) without doing so at
 * either endpoint — a mid-interval singular point. buildTopDerivative only ever
 * checks the value actually reached during integration (a division by a
 * near-zero A(t) throws immediately, at the point it happens, with a clear
 * message), and the entry points below additionally pre-check both endpoints
 * before starting. Neither is a substitute for a full interior root scan of
 * A(t) across (t0, tEnd) — that would need something like
 * TaylorGKTurboIntegrator's pole-finder, and is a reasonable next addition if
 * mid-interval singularities turn out to matter for your equations, but is out
 * of scope here.
 */
public final class LinearHODifferentialEquations {

    private static final MethodType COEFF_TYPE
            = MethodType.methodType(void.class, double[].class, double[].class);

    /**
     * Below this magnitude, A(t) is treated as singular rather than merely
     * small.
     */
    private static final double MIN_LEADING_COEFFICIENT_MAGNITUDE = 1e-10;

    private LinearHODifferentialEquations() {
    }

    // ------------------------------------------------------------------
    // diffeqnLinearHO(...) — endpoint-only
    // ------------------------------------------------------------------
    
    /**
     * 
     * @param coefficients ordered highest-order first: coefficients(0) is A(t)
     * (the y_n coefficient), coefficients(1) is B(t) (the y_(n-1) coefficient),
     * ..., the last entry is F(t) (the y_0 coefficient). Length must be
     * order+1, where order == y0.length.
     * @param forcingOrNull g(t) for the non-homogeneous case, or null for the
     * homogeneous case (right-hand side 0)
     * @param tSlot
     * @param ySlotStart
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param initialStep
     * @param method
     * @return
     * @throws Throwable 
     */
    public static double executeTurboODELinearHO(MethodHandle[] coefficients,
            MethodHandle forcingOrNull,
            int tSlot,
            int ySlotStart,
            int frameSize,
            double t0,
            double[] y0,
            double tEnd,
            double initialStep,
            DifferentialEquations.ODESolverMethod method) throws Throwable {
        return executeTurboODELinearHO(coefficients, forcingOrNull, tSlot, ySlotStart, frameSize,
                t0, y0, tEnd, initialStep, method, null);
    }

 
    /**
     * Same as {@link #executeTurboODELinearHO}, with an optional analytic
     * Jacobian strategy.
     * @param coefficients
     * @param forcingOrNull
     * @param tSlot
     * @param ySlotStart
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param initialStep
     * @param method
     * @param jacobianStrategy
     * @return
     * @throws Throwable 
     */
    public static double executeTurboODELinearHO(MethodHandle[] coefficients,
            MethodHandle forcingOrNull,
            int tSlot,
            int ySlotStart,
            int frameSize,
            double t0,
            double[] y0,
            double tEnd,
            double initialStep,
            DifferentialEquations.ODESolverMethod method,
            DifferentialEquations.JacobianStrategy jacobianStrategy) throws Throwable {

        MethodHandle topDerivative = buildTopDerivative(coefficients, forcingOrNull, tSlot, ySlotStart, frameSize, y0.length);
        checkLeadingCoefficientAtEndpoints(coefficients[0], tSlot, frameSize, t0, tEnd);

        return HigherOrderODE.executeTurboODEHO(
                topDerivative, tSlot, ySlotStart, frameSize, t0, y0, tEnd, initialStep, method, jacobianStrategy);
    }

    // ------------------------------------------------------------------
    // diffeqnPathLinearHO(...) — trajectory output, (t, y) only
    // ------------------------------------------------------------------
    public static double[][] executeTurboODEPathLinearHO(MethodHandle[] coefficients,
            MethodHandle forcingOrNull,
            int tSlot,
            int ySlotStart,
            int frameSize,
            double t0,
            double[] y0,
            double tEnd,
            double h,
            DifferentialEquations.ODESolverMethod method,
            int points) throws Throwable {
        return executeTurboODEPathLinearHO(coefficients, forcingOrNull, tSlot, ySlotStart, frameSize,
                t0, y0, tEnd, h, method, points, null);
    }

 
    /**
     * Same as {@link #executeTurboODEPathLinearHO}, with an optional analytic
     * Jacobian strategy.
     * @param coefficients
     * @param forcingOrNull
     * @param tSlot
     * @param ySlotStart
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param h
     * @param method
     * @param points
     * @param jacobianStrategy
     * @return
     * @throws Throwable 
     */
    public static double[][] executeTurboODEPathLinearHO(MethodHandle[] coefficients,
            MethodHandle forcingOrNull,
            int tSlot,
            int ySlotStart,
            int frameSize,
            double t0,
            double[] y0,
            double tEnd,
            double h,
            DifferentialEquations.ODESolverMethod method,
            int points,
            DifferentialEquations.JacobianStrategy jacobianStrategy) throws Throwable {

        MethodHandle topDerivative = buildTopDerivative(coefficients, forcingOrNull, tSlot, ySlotStart, frameSize, y0.length);
        checkLeadingCoefficientAtEndpoints(coefficients[0], tSlot, frameSize, t0, tEnd);

        return HigherOrderODE.executeTurboODEPathHO(
                topDerivative, tSlot, ySlotStart, frameSize, t0, y0, tEnd, h, method, points, jacobianStrategy);
    }

    // ------------------------------------------------------------------
    // Compilation core
    // ------------------------------------------------------------------
    /**
     * Builds the divided top-derivative handle:
     *
     * y_n = ( g(t) - sum over k=1..order of coefficients(k)*Y(order-k) ) /
     * coefficients(0)
     *
     * (with g(t) taken as 0 when forcingOrNull is null), ready to hand to
     * CompanionSystemHandles/HigherOrderODE exactly as if it had been written
     * directly as a diffeqnHO lambda body.
     *
     * A(t) == coefficients(0) is checked for a near-zero value on every single
     * call, at the point actually reached — an ArithmeticException is thrown
     * immediately, naming the offending t, rather than letting a division blow
     * up into Infinity/NaN and surface three layers down as an opaque Newton
     * non-convergence warning or a thrashing adaptive step size.
     * @param coefficients
     * @param forcingOrNull
     * @param tSlot
     * @param ySlotStart
     * @param frameSize
     * @param order
     * @return 
     */
    public static MethodHandle buildTopDerivative(MethodHandle[] coefficients,
            MethodHandle forcingOrNull,
            int tSlot,
            int ySlotStart,
            int frameSize,
            int order) {
        if (coefficients == null || coefficients.length != order + 1) {
            throw new IllegalArgumentException(
                    "coefficients must have length order+1 (highest order first through the y_0 coefficient), "
                    + "got " + (coefficients == null ? "null" : coefficients.length) + " for order=" + order);
        }
        if (order <= 0) {
            throw new IllegalArgumentException("order must be positive, got " + order);
        }
        if (ySlotStart < 0 || ySlotStart + order > frameSize) {
            throw new IllegalArgumentException(
                    "ySlotStart=" + ySlotStart + " with order=" + order
                    + " does not fit inside frameSize=" + frameSize);
        }
        for (int i = 0; i < coefficients.length; i++) {
            requireCoeffType(coefficients[i], "coefficients(" + i + ")");
        }
        if (forcingOrNull != null) {
            requireCoeffType(forcingOrNull, "forcing term g(t)");
        }

        Adapter adapter = new Adapter(coefficients, forcingOrNull, tSlot, ySlotStart, order);
        try {
            return MethodHandles.lookup().bind(adapter, "topDerivative", COEFF_TYPE);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException("Failed to bind linear-HO top-derivative adapter", e);
        }
    }

    private static void requireCoeffType(MethodHandle h, String label) {
        if (h == null) {
            throw new IllegalArgumentException(label + " must not be null");
        }
        if (!h.type().equals(COEFF_TYPE)) {
            throw new IllegalArgumentException(
                    label + " has incompatible signature. Expected " + COEFF_TYPE + " but got " + h.type());
        }
    }

    /**
     * Pre-flight check at both interval endpoints, run once before integration
     * starts, so a badly chosen interval fails immediately with a clear message
     * instead of after however many steps it takes the solver to reach the
     * singular point.
     */
    private static void checkLeadingCoefficientAtEndpoints(MethodHandle leadingCoefficient,
            int tSlot,
            int frameSize,
            double t0,
            double tEnd) throws Throwable {
        double[] probe = new double[frameSize];
        double[] out = new double[1];
        for (double t : new double[]{t0, tEnd}) {
            probe[tSlot] = t;
            leadingCoefficient.invokeExact(probe, out);
            if (!Double.isFinite(out[0]) || Math.abs(out[0]) < MIN_LEADING_COEFFICIENT_MAGNITUDE) {
                throw new IllegalArgumentException(
                        "Leading coefficient A(t) is zero, near-zero, or non-finite at t=" + t
                        + " (value=" + out[0] + "). The equation is singular there — choose an "
                        + "interval where A(t) does not vanish (this checks only the two endpoints; "
                        + "an interior zero crossing of A(t) is not caught here).");
            }
        }
    }

    /**
     * Holds the per-equation coefficient handles backing one companion
     * top-derivative MethodHandle.
     */
    private static final class Adapter {

        private final MethodHandle[] coefficients;
        private final MethodHandle forcingOrNull;
        private final int ySlotStart;
        private final int order;
        // Reusable single-value scratch buffer; not thread-safe by design —
        // one adapter backs exactly one solve call, same contract as
        // CompanionSystemHandles' adapter.
        private final double[] scratch = new double[1];

        Adapter(MethodHandle[] coefficients, MethodHandle forcingOrNull, int tSlot, int ySlotStart, int order) {
            this.coefficients = coefficients;
            this.forcingOrNull = forcingOrNull;
            this.ySlotStart = ySlotStart;
            this.order = order;
        }

        // Signature must exactly match (double(), double())void for invokeExact/bind.
        void topDerivative(double[] vars, double[] outDerivatives) throws Throwable {
            coefficients[0].invokeExact(vars, scratch);
            double a = scratch[0];
            if (!Double.isFinite(a) || Math.abs(a) < MIN_LEADING_COEFFICIENT_MAGNITUDE) {
                throw new ArithmeticException(
                        "Leading coefficient A(t) is zero, near-zero, or non-finite at t=" + vars[0]
                        + " (value=" + a + ") -- equation is singular at this point");
            }

            double sum = 0.0;
            for (int k = 1; k <= order; k++) {
                coefficients[k].invokeExact(vars, scratch);
                sum += scratch[0] * vars[ySlotStart + (order - k)];
            }

            double numerator;
            if (forcingOrNull != null) {
                forcingOrNull.invokeExact(vars, scratch);
                numerator = scratch[0] - sum;
            } else {
                numerator = -sum;
            }

            outDerivatives[0] = numerator / a;
        }
    }
}
