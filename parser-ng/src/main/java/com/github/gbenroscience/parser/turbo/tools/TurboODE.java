/*
 * Copyright 2026 GBEMIRO.
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
package com.github.gbenroscience.parser.turbo.tools;
 
import com.github.gbenroscience.math.differentialcalculus.equations.turbo.DifferentialEquations;
import java.lang.invoke.MethodHandle;


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

    /** Default output resolution for diffeqnPath when the caller doesn't specify one. */
    private static final int DEFAULT_PATH_POINTS = 100;

    // ------------------------------------------------------------------
    // diffeqn(...) — endpoint-only, unchanged contract from before
    // ------------------------------------------------------------------

    public static double executeTurboODE(MethodHandle dy_dt,
                                          int tSlot,
                                          int ySlot,
                                          int frameSize,
                                          double t0,
                                          double y0,
                                          double tEnd,
                                          double initialStep,
                                          DifferentialEquations.ODESolverMethod method) throws Throwable {

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
                int steps = fixedStepCount(t0, tEnd, initialStep);
                resultVector = DifferentialEquations.stepEuler(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                break;
            }
            case RK4: {
                int steps = fixedStepCount(t0, tEnd, initialStep);
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
                int steps = fixedStepCount(t0, tEnd, initialStep);
                resultVector = DifferentialEquations.stepImplicitEuler(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                break;
            }
            default:
                throw new IllegalArgumentException("Unsupported ODE method: " + method);
        }

        return resultVector[0];
    }

    // ------------------------------------------------------------------
    // diffeqnPath(...) — trajectory output, honoring an explicit `points` count
    // ------------------------------------------------------------------

    /**
     * Scalar trajectory entry point for diffeqnPath(@(t,y) f(t,y), t0, y0, tEnd, h, method, points).
     *
     * @param h      integration step (fixed methods) or initial step (rk45). Must be a positive magnitude.
     * @param method solver to use
     * @param points requested number of uniformly-spaced (t, y) samples in the output,
     *               or &lt;= 0 to mean "use the solver's natural steps, no resampling"
     * @return [rows][2] matrix: column 0 = t, column 1 = y
     */
    public static double[][] executeTurboODEPath(MethodHandle dy_dt,
                                                  int tSlot,
                                                  int ySlot,
                                                  int frameSize,
                                                  double t0,
                                                  double y0,
                                                  double tEnd,
                                                  double h,
                                                  DifferentialEquations.ODESolverMethod method,
                                                  int points) throws Throwable {

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
                int steps = fixedStepCount(t0, tEnd, h);
                history = DifferentialEquations.stepEulerWithHistory(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                naturallyUniform = true;
                break;
            }
            case RK4: {
                int steps = fixedStepCount(t0, tEnd, h);
                history = DifferentialEquations.stepRK4WithHistory(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                naturallyUniform = true;
                break;
            }
            case RK45_DORMAND_PRINCE: {
                history = DifferentialEquations.stepRK45AdaptiveWithHistory(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, h);
                naturallyUniform = false; // accepted steps are irregularly spaced
                break;
            }
            case IMPLICIT_EULER: {
                int steps = fixedStepCount(t0, tEnd, h);
                history = DifferentialEquations.stepImplicitEulerWithHistory(
                        dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                naturallyUniform = true;
                break;
            }
            default:
                throw new IllegalArgumentException("Unsupported ODE method: " + method);
        }

        int requestedPoints = points > 0 ? points : DEFAULT_PATH_POINTS;

        // Fixed-step methods: only resample if the caller asked for a different
        // resolution than what the natural step count already produced.
        if (naturallyUniform && points <= 0) {
            return history;
        }
        if (naturallyUniform && history.length == requestedPoints) {
            return history;
        }

        return DifferentialEquations.resample(history, requestedPoints);
    }

    private static int fixedStepCount(double t0, double tEnd, double stepMagnitude) {
        int steps = (int) Math.round(Math.abs(tEnd - t0) / stepMagnitude);
        return Math.max(steps, 1);
    }
}