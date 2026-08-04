package com.github.gbenroscience.math.differentialcalculus.equations.standard;


/**
 * Scalar "Turbo" entry points over the vectorized ODE solvers — the ParserNG
 * runtime targets for the diffeqn(...) and diffeqnPath(...) functional forms.
 *
 * Both are the degenerate scalar case (systemSize = 1) of the vectorized
 * solvers in {@link DifferentialEquations}. They are NOT ensemble-parallel
 * batch APIs — for SIMD/GPU ensemble dispatch, use systemSize = N (lanes)
 * directly against the vector solvers instead of looping these N times.
 */
public class TurboODE {

    // ------------------------------------------------------------------
    // diffeqn(...) — endpoint-only
    // ------------------------------------------------------------------

    public static double executeTurboODE(ODEFunction dy_dt,
                                          int tSlot,
                                          int ySlot,
                                          int frameSize,
                                          double t0,
                                          double y0,
                                          double tEnd,
                                          double initialStep,
                                          DifferentialEquations.ODESolverMethod method) {
        return executeTurboODE(dy_dt, tSlot, ySlot, frameSize, t0, y0, tEnd, initialStep, method, null);
    }

    /**
     * Same as {@link #executeTurboODE}, but accepts an optional
     * {@link DifferentialEquations.JacobianStrategy}, consulted only when
     * method is IMPLICIT_EULER and ignored (accepted, unused) for every other
     * method — see {@link VectorODE#executeVectorODE} for the same note.
     */
    public static double executeTurboODE(ODEFunction dy_dt,
                                          int tSlot,
                                          int ySlot,
                                          int frameSize,
                                          double t0,
                                          double y0,
                                          double tEnd,
                                          double initialStep,
                                          DifferentialEquations.ODESolverMethod method,
                                          DifferentialEquations.JacobianStrategy jacobianStrategy) {

        if (initialStep <= 0.0) {
            throw new IllegalArgumentException("initialStep must be positive (a magnitude), got " + initialStep);
        }
        if (t0 == tEnd) {
            return y0;
        }

        double[] y0Vector = new double[]{y0};
        int systemSize = 1;
        double[] resultVector;

        switch (method) {
            case EULER: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, initialStep);
                resultVector = DifferentialEquations.stepEuler(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                break;
            }
            case RK4: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, initialStep);
                resultVector = DifferentialEquations.stepRK4(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                break;
            }
            case RK45_DORMAND_PRINCE: {
                resultVector = DifferentialEquations.stepRK45Adaptive(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, initialStep);
                break;
            }
            case IMPLICIT_EULER: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, initialStep);
                resultVector = DifferentialEquations.stepImplicitEuler(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps, jacobianStrategy);
                break;
            }
            default:
                throw new IllegalArgumentException("Unsupported ODE method: " + method);
        }

        return resultVector[0];
    }

    // ------------------------------------------------------------------
    // diffeqnPath(...) — trajectory output
    // ------------------------------------------------------------------

    public static double[][] executeTurboODEPath(ODEFunction dy_dt,
                                                  int tSlot,
                                                  int ySlot,
                                                  int frameSize,
                                                  double t0,
                                                  double y0,
                                                  double tEnd,
                                                  double h,
                                                  DifferentialEquations.ODESolverMethod method,
                                                  int points) {
        return executeTurboODEPath(dy_dt, tSlot, ySlot, frameSize, t0, y0, tEnd, h, method, points, null);
    }

    /**
     * Same as {@link #executeTurboODEPath}, with an optional
     * {@link DifferentialEquations.JacobianStrategy}, consulted only when
     * method is IMPLICIT_EULER.
     *
     * @param h      integration step (fixed methods) or initial step (rk45). Must be a positive magnitude.
     * @param method solver to use
     * @param points requested number of uniformly-spaced (t, y) samples in the output,
     *               or less than or equal to 0 to mean "use the solver's natural steps, no resampling"
     */
    public static double[][] executeTurboODEPath(ODEFunction dy_dt,
                                                  int tSlot,
                                                  int ySlot,
                                                  int frameSize,
                                                  double t0,
                                                  double y0,
                                                  double tEnd,
                                                  double h,
                                                  DifferentialEquations.ODESolverMethod method,
                                                  int points,
                                                  DifferentialEquations.JacobianStrategy jacobianStrategy) {

        if (h <= 0.0) {
            throw new IllegalArgumentException("h must be positive (a magnitude), got " + h);
        }
        if (t0 == tEnd) {
            return new double[][]{{t0, y0}};
        }

        double[] y0Vector = new double[]{y0};
        int systemSize = 1;
        double[][] history;
        boolean naturallyUniform;

        switch (method) {
            case EULER: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, h);
                history = DifferentialEquations.stepEulerWithHistory(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                naturallyUniform = true;
                break;
            }
            case RK4: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, h);
                history = DifferentialEquations.stepRK4WithHistory(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                naturallyUniform = true;
                break;
            }
            case RK45_DORMAND_PRINCE: {
                history = DifferentialEquations.stepRK45AdaptiveWithHistory(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, h);
                naturallyUniform = false;
                break;
            }
            case IMPLICIT_EULER: {
                int steps = OdeSupport.fixedStepCount(t0, tEnd, h);
                history = DifferentialEquations.stepImplicitEulerWithHistory(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps, jacobianStrategy);
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