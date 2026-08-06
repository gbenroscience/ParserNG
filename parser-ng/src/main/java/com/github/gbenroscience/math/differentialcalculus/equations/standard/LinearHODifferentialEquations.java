package com.github.gbenroscience.math.differentialcalculus.equations.standard;

/**
 * Compilation utility backing the coefficient-list form of
 * diffeqnHO(...)/diffeqnPathHO(...) — the general linear n-th order equation:
 *
 *   A(t)*y_n + B(t)*y_(n-1) + C(t)*y_(n-2) + ... + F(t)*y_0 = g(t)
 *
 * where A, B, C, ..., F are each functions of t (or constants), and g(t) is
 * an optional forcing term (omit it, or pass null, for the homogeneous case
 * where the right-hand side is 0).
 *
 * There is no separate diffeqnLinearHO name — a coefficient list is simply a
 * second way to supply diffeqnHO's top-derivative argument, alongside the
 * usual lambda. ParserNG's compiler distinguishes the two by the Token kind
 * at that argument position (MATRIX for a coefficient list, FUNCTION_HANDLE
 * for a lambda) and, for the MATRIX case, calls buildTopDerivative below to
 * get the same kind of ODEFunction a lambda would have compiled to — from
 * that point on, HigherOrderODE/CompanionSystemHandles cannot tell the two
 * apart, and don't need to.
 *
 * <h2>Coefficient function convention</h2>
 * Each coefficient (and the optional forcing term) is an ODEFunction reading
 * only from vars(tSlot) (coefficients are functions of t, never of the state
 * Y — that is what makes the equation linear) and writing its single value
 * to outValue(0).
 *
 * <h2>The one thing this class cannot fully guarantee</h2>
 * A(t) may cross zero somewhere strictly inside (t0, tEnd) without doing so
 * at either endpoint — a mid-interval singular point. buildTopDerivative only
 * ever checks the value actually reached during integration (a division by a
 * near-zero A(t) throws immediately, at the point it happens, with a clear
 * message), and checkLeadingCoefficientAtEndpoints additionally pre-checks
 * both endpoints before starting. Neither is a substitute for a full interior
 * root scan of A(t) across (t0, tEnd) — that would need something like
 * TaylorGKTurboIntegrator's pole-finder, and is a reasonable next addition if
 * mid-interval singularities turn out to matter for your equations, but is
 * out of scope here.
 *
 * ParserNG Standard note: buildTopDerivative needs no reflection here either
 * — the composed top-derivative is just a plain ODEFunction implementation
 * holding the coefficient functions and a small reusable scratch buffer.
 */
public final class LinearHODifferentialEquations {

    /** Below this magnitude, A(t) is treated as singular rather than merely small. */
    private static final double MIN_LEADING_COEFFICIENT_MAGNITUDE = 1e-10;

    private LinearHODifferentialEquations() {
    }

    // ------------------------------------------------------------------
    // Compilation core
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // Compilation core
    // ------------------------------------------------------------------

    /**
     * Builds the divided top-derivative function:
     *
     *   y_n = ( g(t) - sum over k=1..order of coefficients(k)*Y(order-k) ) / coefficients(0)
     *
     * (with g(t) taken as 0 when forcingOrNull is null), ready to hand to
     * CompanionSystemHandles/HigherOrderODE exactly as if it had been written
     * directly as a diffeqnHO lambda body.
     *
     * A(t) == coefficients(0) is checked for a near-zero value on every
     * single call, at the point actually reached — an ArithmeticException is
     * thrown immediately, naming the offending t, rather than letting a
     * division blow up into Infinity/NaN and surface three layers down as an
     * opaque Newton non-convergence warning or a thrashing adaptive step size.
     */
    public static ODEFunction buildTopDerivative(ODEFunction[] coefficients,
                                                  ODEFunction forcingOrNull,
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
            if (coefficients[i] == null) {
                throw new IllegalArgumentException("coefficients(" + i + ") must not be null");
            }
        }

        return new LinearTopDerivative(coefficients, forcingOrNull, ySlotStart, order);
    }

    /**
     * Pre-flight check at both interval endpoints, run once before
     * integration starts, so a badly chosen interval fails immediately with
     * a clear message instead of after however many steps it takes the
     * solver to reach the singular point. Call this once, right after
     * buildTopDerivative, before handing the result to HigherOrderODE.
     */
    public static void checkLeadingCoefficientAtEndpoints(ODEFunction leadingCoefficient,
                                                            int tSlot,
                                                            int frameSize,
                                                            double t0,
                                                            double tEnd) {
        double[] probe = new double[frameSize];
        double[] out = new double[1];
        for (double t : new double[]{t0, tEnd}) {
            probe[tSlot] = t;
            leadingCoefficient.apply(probe, out);
            if (!Double.isFinite(out[0]) || Math.abs(out[0]) < MIN_LEADING_COEFFICIENT_MAGNITUDE) {
                throw new IllegalArgumentException(
                        "Leading coefficient A(t) is zero, near-zero, or non-finite at t=" + t
                        + " (value=" + out[0] + "). The equation is singular there — choose an "
                        + "interval where A(t) does not vanish (this checks only the two endpoints; "
                        + "an interior zero crossing of A(t) is not caught here).");
            }
        }
    }

    /** Plain ODEFunction implementation — no reflection, no MethodHandles. */
    private static final class LinearTopDerivative implements ODEFunction {
        private final ODEFunction[] coefficients;
        private final ODEFunction forcingOrNull;
        private final int ySlotStart;
        private final int order;
        // Reusable single-value scratch buffer; not thread-safe by design —
        // one instance backs exactly one solve call, same contract as
        // CompanionSystemHandles' companion function.
        private final double[] scratch = new double[1];

        LinearTopDerivative(ODEFunction[] coefficients, ODEFunction forcingOrNull, int ySlotStart, int order) {
            this.coefficients = coefficients;
            this.forcingOrNull = forcingOrNull;
            this.ySlotStart = ySlotStart;
            this.order = order;
        }

        @Override
        public void apply(double[] vars, double[] outDerivatives) {
            coefficients[0].apply(vars, scratch);
            double a = scratch[0];
            if (!Double.isFinite(a) || Math.abs(a) < MIN_LEADING_COEFFICIENT_MAGNITUDE) {
                throw new ArithmeticException(
                        "Leading coefficient A(t) is zero, near-zero, or non-finite at t=" + vars[0]
                        + " (value=" + a + ") -- equation is singular at this point");
            }

            double sum = 0.0;
            for (int k = 1; k <= order; k++) {
                coefficients[k].apply(vars, scratch);
                sum += scratch[0] * vars[ySlotStart + (order - k)];
            }

            double numerator;
            if (forcingOrNull != null) {
                forcingOrNull.apply(vars, scratch);
                numerator = scratch[0] - sum;
            } else {
                numerator = -sum;
            }

            outDerivatives[0] = numerator / a;
        }
    }
}