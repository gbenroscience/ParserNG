package com.github.gbenroscience.math.differentialcalculus.equations.standard;


/**
 * Higher-order equation entry points — the ParserNG runtime targets for
 * diffeqnHO(...) and diffeqnPathHO(...).
 *
 * The caller supplies only the top-order derivative, f(t, Y) where
 * Y = (y, y', ..., y^(n-1)), and an initial state y0 = (y(t0), y'(t0), ...)
 * whose length fixes the order n. CompanionSystemHandles builds the full
 * companion first-order system from that single handle, and VectorODE drives
 * it — the order-n reduction is invisible to the caller, exactly like the
 * manual reduction step is invisible in the diffeqnHO(...) surface syntax.
 *
 * Both entry points return only y(t) (state component 0), not the derivative
 * components, to match diffeqn(...)/diffeqnPath(...)'s "one physical quantity
 * out" contract.
 */
public class HigherOrderODE {

    // ------------------------------------------------------------------
    // diffeqnHO(...) — endpoint-only
    // ------------------------------------------------------------------

    public static double executeTurboODEHO(ODEFunction topDerivative,
                                            int tSlot,
                                            int ySlotStart,
                                            int frameSize,
                                            double t0,
                                            double[] y0,
                                            double tEnd,
                                            double initialStep,
                                            DifferentialEquations.ODESolverMethod method) {
        return executeTurboODEHO(topDerivative, tSlot, ySlotStart, frameSize, t0, y0, tEnd, initialStep, method, null);
    }

    /**
     * Same as {@link #executeTurboODEHO}, but accepts an optional
     * {@link DifferentialEquations.JacobianStrategy} for the companion
     * system, consulted only when method is IMPLICIT_EULER (see
     * {@link VectorODE#executeVectorODE} for the same note on other methods
     * ignoring it). Note the Jacobian here is of the companion system, not of
     * the scalar top-derivative expression directly — its state indices are
     * ySlotStart through ySlotStart+order-1.
     */
    public static double executeTurboODEHO(ODEFunction topDerivative,
                                            int tSlot,
                                            int ySlotStart,
                                            int frameSize,
                                            double t0,
                                            double[] y0,
                                            double tEnd,
                                            double initialStep,
                                            DifferentialEquations.ODESolverMethod method,
                                            DifferentialEquations.JacobianStrategy jacobianStrategy) {

        int order = y0.length;
        ODEFunction companion = CompanionSystemHandles.buildCompanion(
                topDerivative, tSlot, ySlotStart, order, frameSize);

        double[] finalState = VectorODE.executeVectorODE(
                companion, tSlot, ySlotStart, frameSize, t0, y0, tEnd, initialStep, method, jacobianStrategy);

        return finalState[0]; // y(tEnd); finalState[1..] hold y', y'', ... if ever needed
    }

    // ------------------------------------------------------------------
    // diffeqnPathHO(...) — trajectory output, [t, y] only
    // ------------------------------------------------------------------

    public static double[][] executeTurboODEPathHO(ODEFunction topDerivative,
                                                    int tSlot,
                                                    int ySlotStart,
                                                    int frameSize,
                                                    double t0,
                                                    double[] y0,
                                                    double tEnd,
                                                    double h,
                                                    DifferentialEquations.ODESolverMethod method,
                                                    int points) {
        return executeTurboODEPathHO(topDerivative, tSlot, ySlotStart, frameSize, t0, y0, tEnd, h, method, points, null);
    }

    /**
     * Same as {@link #executeTurboODEPathHO}, with an optional
     * {@link DifferentialEquations.JacobianStrategy}, consulted only when
     * method is IMPLICIT_EULER.
     *
     * @param h      integration step (fixed methods) or initial step (rk45)
     * @param points requested number of uniformly-spaced samples, or less
     *               than or equal to 0 for the solver's natural steps
     */
    public static double[][] executeTurboODEPathHO(ODEFunction topDerivative,
                                                    int tSlot,
                                                    int ySlotStart,
                                                    int frameSize,
                                                    double t0,
                                                    double[] y0,
                                                    double tEnd,
                                                    double h,
                                                    DifferentialEquations.ODESolverMethod method,
                                                    int points,
                                                    DifferentialEquations.JacobianStrategy jacobianStrategy) {

        int order = y0.length;
        ODEFunction companion = CompanionSystemHandles.buildCompanion(
                topDerivative, tSlot, ySlotStart, order, frameSize);

        double[][] fullHistory = VectorODE.executeVectorODEPath(
                companion, tSlot, ySlotStart, frameSize, t0, y0, tEnd, h, method, points, jacobianStrategy);

        double[][] path = new double[fullHistory.length][2];
        for (int i = 0; i < fullHistory.length; i++) {
            path[i][0] = fullHistory[i][0]; // t
            path[i][1] = fullHistory[i][1]; // y (state component 0)
        }
        return path;
    }
}