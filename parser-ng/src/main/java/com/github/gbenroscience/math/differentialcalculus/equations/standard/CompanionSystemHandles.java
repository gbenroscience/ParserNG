package com.github.gbenroscience.math.differentialcalculus.equations.standard;

/**
 * Builds the companion first-order system ODEFunction for a higher-order
 * equation, from an ODEFunction that computes only the top-order derivative.
 *
 * For an order-n equation y^(n) = f(t, Y) where Y = (y, y', ..., y^(n-1))
 * occupies frame slots (ySlotStart, ySlotStart+n), the companion system is:
 *
 *   d/dt Y(0)   = Y(1)
 *   d/dt Y(1)   = Y(2)
 *   ...
 *   d/dt Y(n-2) = Y(n-1)
 *   d/dt Y(n-1) = f(t, Y)
 *
 * This is the reduction ParserNG hides from the user for diffeqnHO/diffeqnPathHO:
 * the caller only ever compiles the top-derivative expression; this class builds
 * the full systemSize=n dy_dt ODEFunction expected by the vector solvers.
 *
 * ParserNG Standard note: unlike the Turbo tier, this needs no reflection —
 * the companion is just a plain object implementing ODEFunction, holding the
 * top-derivative ODEFunction and a small reusable scratch buffer.
 */
public final class CompanionSystemHandles {

    private CompanionSystemHandles() {
    }

    /**
     * @param topDerivative ODEFunction computing only the top-order derivative,
     *                      f(t, Y), writing its single result to outDerivatives(0).
     * @param tSlot         frame index holding t
     * @param ySlotStart    frame index where the order-n state block (y..y^(n-1)) starts
     * @param order         n, the order of the equation (== y0.length for the caller)
     * @param frameSize     total width of the vars frame
     * @return a new ODEFunction with an outDerivatives array of length
     *         `order`, ready to hand to the vector solvers exactly like any
     *         hand-written system dy_dt.
     *
     * Note: the returned ODEFunction is stateful per equation instance (it
     * carries a small reusable scratch buffer) and is intended to back one
     * solve call — do not share a single built instance across concurrent
     * solves.
     */
    public static ODEFunction buildCompanion(ODEFunction topDerivative,
                                              int tSlot,
                                              int ySlotStart,
                                              int order,
                                              int frameSize) {
        if (topDerivative == null) {
            throw new IllegalArgumentException("topDerivative ODEFunction must not be null");
        }
        if (order <= 0) {
            throw new IllegalArgumentException("order must be positive, got " + order);
        }
        if (ySlotStart < 0 || ySlotStart + order > frameSize) {
            throw new IllegalArgumentException(
                    "ySlotStart=" + ySlotStart + " with order=" + order
                    + " does not fit inside frameSize=" + frameSize);
        }

        return new CompanionFunction(topDerivative, ySlotStart, order);
    }

    /** Plain ODEFunction implementation — no reflection, no MethodHandles. */
    private static final class CompanionFunction implements ODEFunction {
        private final ODEFunction topDerivative;
        private final int ySlotStart;
        private final int order;
        // Reusable scratch buffer for the top-derivative call; not thread-safe by
        // design — one instance backs exactly one solve call.
        private final double[] topOut = new double[1];

        CompanionFunction(ODEFunction topDerivative, int ySlotStart, int order) {
            this.topDerivative = topDerivative;
            this.ySlotStart = ySlotStart;
            this.order = order;
        }

        @Override
        public void apply(double[] vars, double[] outDerivatives) {
            for (int i = 0; i < order - 1; i++) {
                outDerivatives[i] = vars[ySlotStart + i + 1];
            }
            topDerivative.apply(vars, topOut);
            outDerivatives[order - 1] = topOut[0];
        }
    }
}