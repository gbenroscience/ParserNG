package com.github.gbenroscience.math.differentialcalculus.equations;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

/**
 * Builds the companion first-order system MethodHandle for a higher-order
 * equation, from a MethodHandle that computes only the top-order derivative.
 *
 * For an order-n equation y^(n) = f(t, Y) where Y = [y, y', ..., y^(n-1)]
 * occupies frame slots [ySlotStart, ySlotStart+n), the companion system is:
 *
 * d/dt Y[0] = Y[1] d/dt Y[1] = Y[2] ... d/dt Y[n-2] = Y[n-1] d/dt Y[n-1] = f(t,
 * Y)
 *
 * This is the reduction ParserNG hides from the user for
 * diffeqnHO/diffeqnPathHO: the caller only ever compiles the top-derivative
 * expression; this class builds the full systemSize=n dy_dt handle expected by
 * the vector solvers.
 */
public final class CompanionSystemHandles {

    private static final MethodType DY_DT_TYPE
            = MethodType.methodType(void.class, double[].class, double[].class);

    private CompanionSystemHandles() {
    }

    /**
     * @param topDerivative MethodHandle computing only the top-order
     * derivative, f(t, Y), writing its single result to outDerivatives[0]. Must
     * match the standard (double[], double[])void descriptor.
     * @param tSlot frame index holding t
     * @param ySlotStart frame index where the order-n state block [y..y^(n-1)]
     * starts
     * @param order n, the order of the equation (== y0.length for the caller)
     * @param frameSize total width of the vars frame
     * @return a new MethodHandle, matching (double[], double[])void with an
     * outDerivatives array of length `order`, ready to hand to the vector
     * solvers exactly like any hand-written system dy_dt.
     *
     * Note: the returned handle is stateful per equation instance (it carries a
     * small reusable scratch buffer) and is intended to back one solve call —
     * do not share a single built handle across concurrent solves.
     */
    public static MethodHandle buildCompanion(MethodHandle topDerivative,
            int tSlot,
            int ySlotStart,
            int order,
            int frameSize) {
        if (topDerivative == null) {
            throw new IllegalArgumentException("topDerivative MethodHandle must not be null");
        }
        if (order <= 0) {
            throw new IllegalArgumentException("order must be positive, got " + order);
        }
        if (ySlotStart < 0 || ySlotStart + order > frameSize) {
            throw new IllegalArgumentException(
                    "ySlotStart=" + ySlotStart + " with order=" + order
                    + " does not fit inside frameSize=" + frameSize);
        }
        if (!topDerivative.type().equals(DY_DT_TYPE)) {
            throw new IllegalArgumentException(
                    "topDerivative MethodHandle has incompatible signature. Expected " + DY_DT_TYPE
                    + " but got " + topDerivative.type());
        }

        Adapter adapter = new Adapter(topDerivative, ySlotStart, order);
        try {
            return MethodHandles.lookup().bind(adapter, "companionDerivative", DY_DT_TYPE);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException("Failed to bind companion system adapter", e);
        }
    }

    /**
     * Holds the per-equation shift/delegate state backing one companion system
     * MethodHandle.
     */
    private static final class Adapter {

        private final MethodHandle topDerivative;
        private final int ySlotStart;
        private final int order;
        // Reusable scratch buffer for the top-derivative call; not thread-safe by design —
        // one adapter (and therefore one companion handle) backs exactly one solve call.
        private final double[] topOut = new double[1];

        Adapter(MethodHandle topDerivative, int ySlotStart, int order) {
            this.topDerivative = topDerivative;
            this.ySlotStart = ySlotStart;
            this.order = order;
        }

        /**
         * Signature must exactly match (double[], double[])void for
         * invokeExact/bind.
         *
         * @param vars
         * @param outDerivatives
         * @throws Throwable
         */
        void companionDerivative(double[] vars, double[] outDerivatives) throws Throwable {
            for (int i = 0; i < order - 1; i++) {
                outDerivatives[i] = vars[ySlotStart + i + 1];
            }
            topDerivative.invokeExact(vars, topOut);
            outDerivatives[order - 1] = topOut[0];
        }
    }
}
