package com.github.gbenroscience.math.differentialcalculus.equations.standard;


/**
 * Vector-system entry points: systemSize is derived directly from y0.length
 * (unlike TurboODE's scalar path, no packing into a 1-element array is needed).
 *
 * Backs diffeqn(...)/diffeqnPath(...) when called with a vector y0 (a genuine
 * system of ODEs), and is reused by HigherOrderODE to drive the companion
 * system built from a higher-order equation's top-derivative handle.
 */
public class VectorODE {

    // ------------------------------------------------------------------
    // Endpoint-only system solve
    // ------------------------------------------------------------------

    public static double[] executeVectorODE(ODEFunction dy_dt,
                                             int tSlot,
                                             int ySlotStart,
                                             int frameSize,
                                             double t0,
                                             double[] y0,
                                             double tEnd,
                                             double initialStep,
                                             DifferentialEquations.ODESolverMethod method) {
        return executeVectorODE(dy_dt, tSlot, ySlotStart, frameSize, t0, y0, tEnd, initialStep, method, null);
    }

    /**
     * Same as {@link #executeVectorODE}, but accepts an optional
     * {@link DifferentialEquations.JacobianStrategy} — e.g. an
     * AnalyticJacobian built from forward-mode AD — to replace the default
     * central-difference Jacobian used by the IMPLICIT_EULER path.
     *
     * jacobianStrategy is only consulted when method is IMPLICIT_EULER; for
     * every other method it is accepted but ignored, since explicit methods
     * never build a Jacobian. That keeps a single call site workable
     * regardless of which method a caller ultimately selects.
     */
    public static double[] executeVectorODE(ODEFunction dy_dt,
                                             int tSlot,
                                             int ySlotStart,
                                             int frameSize,
                                             double t0,
                                             double[] y0,
                                             double tEnd,
                                             double initialStep,
                                             DifferentialEquations.ODESolverMethod method,
                                             DifferentialEquations.JacobianStrategy jacobianStrategy) {

        if (initialStep <= 0.0) {
            throw new IllegalArgumentException("initialStep must be positive (a magnitude), got " + initialStep);
        }
        if (t0 == tEnd) {
            return y0.clone();
        }

        int systemSize = y0.length;

        switch (method) {
            case EULER: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, initialStep);
                return DifferentialEquations.stepEuler(
                        dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps);
            }
            case RK4: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, initialStep);
                return DifferentialEquations.stepRK4(
                        dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps);
            }
            case RK45_DORMAND_PRINCE:
                return DifferentialEquations.stepRK45Adaptive(
                        dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, initialStep);
            case IMPLICIT_EULER: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, initialStep);
                return DifferentialEquations.stepImplicitEuler(
                        dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, jacobianStrategy);
            }
            default:
                throw new IllegalArgumentException("Unsupported ODE method: " + method);
        }
    }

    // ------------------------------------------------------------------
    // Trajectory system solve
    // ------------------------------------------------------------------

    public static double[][] executeVectorODEPath(ODEFunction dy_dt,
                                                   int tSlot,
                                                   int ySlotStart,
                                                   int frameSize,
                                                   double t0,
                                                   double[] y0,
                                                   double tEnd,
                                                   double h,
                                                   DifferentialEquations.ODESolverMethod method,
                                                   int points) {
        return executeVectorODEPath(dy_dt, tSlot, ySlotStart, frameSize, t0, y0, tEnd, h, method, points, null);
    }

    /**
     * Same as {@link #executeVectorODEPath}, with an optional
     * {@link DifferentialEquations.JacobianStrategy}, consulted only when
     * method is IMPLICIT_EULER (see {@link #executeVectorODE} for the same
     * note on the other methods ignoring it).
     */
    public static double[][] executeVectorODEPath(ODEFunction dy_dt,
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

        if (h <= 0.0) {
            throw new IllegalArgumentException("h must be positive (a magnitude), got " + h);
        }
        if (t0 == tEnd) {
            double[] row0 = new double[1 + y0.length];
            row0[0] = t0;
            System.arraycopy(y0, 0, row0, 1, y0.length);
            return new double[][]{row0};
        }

        int systemSize = y0.length;
        double[][] history;
        boolean naturallyUniform;

        switch (method) {
            case EULER: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, h);
                history = DifferentialEquations.stepEulerWithHistory(
                        dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps);
                naturallyUniform = true;
                break;
            }
            case RK4: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, h);
                history = DifferentialEquations.stepRK4WithHistory(
                        dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps);
                naturallyUniform = true;
                break;
            }
            case RK45_DORMAND_PRINCE: {
                history = DifferentialEquations.stepRK45AdaptiveWithHistory(
                        dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, h);
                naturallyUniform = false; // accepted steps are irregularly spaced
                break;
            }
            case IMPLICIT_EULER: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, h);
                history = DifferentialEquations.stepImplicitEulerWithHistory(
                        dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, jacobianStrategy);
                naturallyUniform = true;
                break;
            }
            default:
                throw new IllegalArgumentException("Unsupported ODE method: " + method);
        }

        int requestedPoints = points > 0 ? points : OdeSupport.DEFAULT_PATH_POINTS;

        if (naturallyUniform && points <= 0) {
            return history;
        }
        if (naturallyUniform && history.length == requestedPoints) {
            return history;
        }

        return DifferentialEquations.resample(history, requestedPoints);
    }
}