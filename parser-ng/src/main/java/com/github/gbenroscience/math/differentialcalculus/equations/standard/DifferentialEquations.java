package com.github.gbenroscience.math.differentialcalculus.equations.standard;

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.JacobianStrategy;
import java.util.ArrayList;
import java.util.List;

/**
 * High-performance, JIT-optimized Vectorized Ordinary Differential Equation
 * (ODE) solvers. Supports arbitrary n-th order differential equations by
 * reduction to first-order vector systems.
 *
 * Target ODEFunction signature MUST strictly match: void f(double[] vars,
 * double[] outDerivatives)
 *
 * @author GBEMIRO
 */
public class DifferentialEquations {
    

    // Dormand-Prince 5(4) Coefficients
    private static final double DP_A21 = 1.0 / 5.0,
            DP_A31 = 3.0 / 40.0, DP_A32 = 9.0 / 40.0,
            DP_A41 = 44.0 / 45.0, DP_A42 = -56.0 / 15.0, DP_A43 = 32.0 / 9.0,
            DP_A51 = 19372.0 / 6561.0, DP_A52 = -25360.0 / 2187.0, DP_A53 = 64448.0 / 6561.0, DP_A54 = -212.0 / 729.0,
            DP_A61 = 9017.0 / 3168.0, DP_A62 = -355.0 / 33.0, DP_A63 = 46732.0 / 5247.0, DP_A64 = 49.0 / 176.0, DP_A65 = -5103.0 / 18656.0,
            DP_A71 = 35.0 / 384.0, DP_A72 = 0.0, DP_A73 = 500.0 / 1113.0, DP_A74 = 125.0 / 192.0, DP_A75 = -2187.0 / 6784.0, DP_A76 = 11.0 / 84.0;

    // 5th Order Weights (Matches A7 row due to FSAL property)
    private static final double DP_B51 = 35.0 / 384.0, DP_B53 = 500.0 / 1113.0, DP_B54 = 125.0 / 192.0,
            DP_B55 = -2187.0 / 6784.0, DP_B56 = 11.0 / 84.0;

    // 4th Order Weights 
    private static final double DP_B41 = 5179.0 / 57600.0, DP_B43 = 7571.0 / 16695.0, DP_B44 = 393.0 / 640.0,
            DP_B45 = -92097.0 / 339200.0, DP_B46 = 187.0 / 2100.0, DP_B47 = 1.0 / 40.0;

    // Nodes (Time fractions)
    private static final double DP_C2 = 1.0 / 5.0, DP_C3 = 3.0 / 10.0, DP_C4 = 4.0 / 5.0,
            DP_C5 = 8.0 / 9.0, DP_C6 = 1.0;

    /**
     * Callback invoked once per accepted state, (t, y). Used to record a
     * trajectory without forking a second copy of each solver. y is only valid
     * for the duration of the call — implementations that need to retain it
     * must clone it.
     */
    @FunctionalInterface
    public interface StepListener {

        void onStep(double t, double[] y);
    }

    // ------------------------------------------------------------------
    // Shared validation helpers
    // ------------------------------------------------------------------
    private static void validateHandle(ODEFunction dy_dt) {
        if (dy_dt == null) {
            throw new IllegalArgumentException("dy_dt ODEFunction must not be null");
        }
    }

    private static void validateSlots(int tSlot, int ySlotStart, int systemSize, int frameSize) {
        if (frameSize <= 0) {
            throw new IllegalArgumentException("frameSize must be positive, got " + frameSize);
        }
        if (systemSize <= 0) {
            throw new IllegalArgumentException("systemSize must be positive, got " + systemSize);
        }
        if (tSlot < 0 || tSlot >= frameSize) {
            throw new IllegalArgumentException(
                    "tSlot=" + tSlot + " out of bounds for frameSize=" + frameSize);
        }
        if (ySlotStart < 0 || ySlotStart + systemSize > frameSize) {
            throw new IllegalArgumentException(
                    "ySlotStart=" + ySlotStart + " with systemSize=" + systemSize
                    + " does not fit inside frameSize=" + frameSize);
        }
        if (tSlot >= ySlotStart && tSlot < ySlotStart + systemSize) {
            throw new IllegalArgumentException(
                    "tSlot=" + tSlot + " overlaps state block [" + ySlotStart + ", "
                    + (ySlotStart + systemSize) + ")");
        }
    }

    private static double clampStep(double h, double direction, double minMag, double maxMag) {
        double mag = Math.max(minMag, Math.min(Math.abs(h), maxMag));
        return direction * mag;
    }

    private static boolean reachedEnd(double t, double tEnd, double direction) {
        return direction > 0 ? (t >= tEnd) : (t <= tEnd);
    }

    /**
     * Convenience no-op listener used internally so the recording and
     * non-recording paths share one core loop.
     */
    private static final StepListener NO_OP = (t, y) -> {
    };

    // ------------------------------------------------------------------
    // Euler
    // ------------------------------------------------------------------
    public static double[] stepEuler(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps) {
        return stepEulerCore(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, NO_OP);
    }

    /**
     * Same as {@link #stepEuler}, but records (t, y) at t0 and after every
     * step. Returns a [steps+1][1+systemSize] matrix: column 0 is t, columns
     * 1..systemSize are y.
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @return
     */
    public static double[][] stepEulerWithHistory(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps) {
        validateHandle(dy_dt);
        validateSlots(tSlot, ySlotStart, systemSize, frameSize);
        if (steps <= 0) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }

        double[][] history = new double[steps + 1][1 + systemSize];
        int[] rowIndex = {0};
        StepListener recorder = (t, y) -> {
            double[] row = history[rowIndex[0]++];
            row[0] = t;
            System.arraycopy(y, 0, row, 1, systemSize);
        };
        stepEulerCore(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, recorder);
        return history;
    }

    /**
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @param listener
     * @return
     */
    private static double[] stepEulerCore(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps,
            StepListener listener) {
        validateHandle(dy_dt);
        validateSlots(tSlot, ySlotStart, systemSize, frameSize);
        if (steps <= 0) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }

        double[] vars = new double[frameSize];
        double[] currentY = y0.clone();
        double[] slopes = new double[systemSize];
        double h = (tEnd - t0) / steps;
        double t = t0;

        listener.onStep(t, currentY);

        for (int i = 0; i < steps; i++) {
            vars[tSlot] = t;
            System.arraycopy(currentY, 0, vars, ySlotStart, systemSize);

            dy_dt.apply(vars, slopes);

            for (int j = 0; j < systemSize; j++) {
                currentY[j] += h * slopes[j];
            }
            t += h;
            listener.onStep(t, currentY);
        }
        return currentY;
    }

    // ------------------------------------------------------------------
    // RK4
    // ------------------------------------------------------------------
    /**
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @return
     */
    public static double[] stepRK4(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps) {
        return stepRK4Core(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, NO_OP);
    }

    /**
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @return
     */
    public static double[][] stepRK4WithHistory(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps) {
        List<double[]> rows = new ArrayList<>(steps + 1);
        StepListener recorder = (t, y) -> {
            double[] row = new double[1 + systemSize];
            row[0] = t;
            System.arraycopy(y, 0, row, 1, systemSize);
            rows.add(row);
        };
        stepRK4Core(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, recorder);
        return rows.toArray(new double[0][]);
    }

    private static double[] stepRK4Core(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps,
            StepListener listener) {
        validateHandle(dy_dt);
        validateSlots(tSlot, ySlotStart, systemSize, frameSize);
        if (steps <= 0) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }

        double[] vars = new double[frameSize];
        double[] currentY = y0.clone();
        double[][] k = new double[4][systemSize];
        double h = (tEnd - t0) / steps;
        double t = t0;

        listener.onStep(t, currentY);

        for (int i = 0; i < steps; i++) {
            vars[tSlot] = t;
            System.arraycopy(currentY, 0, vars, ySlotStart, systemSize);
            dy_dt.apply(vars, k[0]);

            vars[tSlot] = t + h * 0.5;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * 0.5 * k[0][j];
            }
            dy_dt.apply(vars, k[1]);

            vars[tSlot] = t + h * 0.5;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * 0.5 * k[1][j];
            }
            dy_dt.apply(vars, k[2]);

            vars[tSlot] = t + h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * k[2][j];
            }
            dy_dt.apply(vars, k[3]);

            for (int j = 0; j < systemSize; j++) {
                currentY[j] += (h / 6.0) * (k[0][j] + 2 * k[1][j] + 2 * k[2][j] + k[3][j]);
            }
            t += h;
            listener.onStep(t, currentY);
        }
        return currentY;
    }

    /**
     * RK45 Dormand-Prince (adaptive, direction-aware)
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param initialH
     * @return
     */
    public static double[] stepRK45Adaptive(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, double initialH) {
        return stepRK45AdaptiveCore(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, initialH, NO_OP);
    }

    /**
     * Same as {@link #stepRK45Adaptive}, but records (t, y) at t0 and after
     * every ACCEPTED step. Because this solver is adaptive, the resulting t
     * values are irregularly spaced — resample() can be used afterward to
     * interpolate onto a uniform grid for plotting.
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param initialH
     * @return
     */
    public static double[][] stepRK45AdaptiveWithHistory(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd,
            double initialH) {
        List<double[]> rows = new ArrayList<>();
        StepListener recorder = (t, y) -> {
            double[] row = new double[1 + systemSize];
            row[0] = t;
            System.arraycopy(y, 0, row, 1, systemSize);
            rows.add(row);
        };
        stepRK45AdaptiveCore(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, initialH, recorder);
        return rows.toArray(new double[0][]);
    }

    private static double[] stepRK45AdaptiveCore(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd,
            double initialH, StepListener listener) {
        validateHandle(dy_dt);
        validateSlots(tSlot, ySlotStart, systemSize, frameSize);
        if (initialH == 0.0) {
            throw new IllegalArgumentException("initialH must be non-zero");
        }

        double[] vars = new double[frameSize];
        double[] currentY = y0.clone();

        final double ATOL = 1e-8;
        final double RTOL = 1e-6;
        final double MIN_H = 1e-12;

        double span = tEnd - t0;
        if (span == 0.0) {
            listener.onStep(t0, currentY);
            return currentY;
        }
        double direction = Math.signum(span);
        double MAX_H = Math.max(Math.abs(span), 1.0);

        double h = clampStep(initialH, direction, MIN_H, MAX_H);
        double t = t0;

        int maxSteps = 100000;
        int steps = 0;

        double[][] k = new double[7][systemSize];
        double[] y4 = new double[systemSize];
        double[] y5 = new double[systemSize];

        listener.onStep(t, currentY);

        while (!reachedEnd(t, tEnd, direction) && steps++ < maxSteps) {
            if (direction > 0 ? (t + h > tEnd) : (t + h < tEnd)) {
                h = tEnd - t;
            }

            vars[tSlot] = t;
            System.arraycopy(currentY, 0, vars, ySlotStart, systemSize);
            dy_dt.apply(vars, k[0]);

            vars[tSlot] = t + DP_C2 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A21 * k[0][j]);
            }
            dy_dt.apply(vars, k[1]);

            vars[tSlot] = t + DP_C3 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A31 * k[0][j] + DP_A32 * k[1][j]);
            }
            dy_dt.apply(vars, k[2]);

            vars[tSlot] = t + DP_C4 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A41 * k[0][j] + DP_A42 * k[1][j] + DP_A43 * k[2][j]);
            }
            dy_dt.apply(vars, k[3]);

            vars[tSlot] = t + DP_C5 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A51 * k[0][j] + DP_A52 * k[1][j] + DP_A53 * k[2][j] + DP_A54 * k[3][j]);
            }
            dy_dt.apply(vars, k[4]);

            vars[tSlot] = t + DP_C6 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A61 * k[0][j] + DP_A62 * k[1][j] + DP_A63 * k[2][j] + DP_A64 * k[3][j] + DP_A65 * k[4][j]);
            }
            dy_dt.apply(vars, k[5]);

            vars[tSlot] = t + h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A71 * k[0][j] + DP_A73 * k[2][j] + DP_A74 * k[3][j] + DP_A75 * k[4][j] + DP_A76 * k[5][j]);
            }
            dy_dt.apply(vars, k[6]);

            for (int j = 0; j < systemSize; j++) {
                y5[j] = currentY[j] + h * (DP_B51 * k[0][j] + DP_B53 * k[2][j] + DP_B54 * k[3][j] + DP_B55 * k[4][j] + DP_B56 * k[5][j]);
                y4[j] = currentY[j] + h * (DP_B41 * k[0][j] + DP_B43 * k[2][j] + DP_B44 * k[3][j] + DP_B45 * k[4][j] + DP_B46 * k[5][j] + DP_B47 * k[6][j]);
            }

            double maxErrorRatio = 0.0;
            for (int j = 0; j < systemSize; j++) {
                double error = Math.abs(y5[j] - y4[j]);
                double tol = ATOL + RTOL * Math.max(Math.abs(currentY[j]), Math.abs(y5[j]));
                maxErrorRatio = Math.max(maxErrorRatio, error / tol);
            }

            boolean stepAccepted = maxErrorRatio <= 1.0 || Math.abs(h) <= MIN_H;

            if (stepAccepted) {
                t += h;
                System.arraycopy(y5, 0, currentY, 0, systemSize);
                listener.onStep(t, currentY);

                double scale = (maxErrorRatio < 1e-20) ? 5.0 : 0.9 * Math.pow(1.0 / maxErrorRatio, 0.2);
                scale = Math.max(0.2, Math.min(scale, 5.0));
                h = clampStep(h * scale, direction, MIN_H, MAX_H);
            } else {
                double scale = 0.9 * Math.pow(1.0 / maxErrorRatio, 0.2);
                scale = Math.max(0.2, Math.min(scale, 0.9));
                h = clampStep(h * scale, direction, MIN_H, MAX_H);
            }
        }

        if (steps >= maxSteps) {
            System.err.println("Warning: RK45 reached iteration limit (" + maxSteps + ")");
        }

        return currentY;
    }

    /**
     * Implicit Euler
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @return
     */
    public static double[] stepImplicitEuler(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps) {
        return stepImplicitEulerCore(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, NO_OP, null);
    }

    /**
     * Same as {@link #stepImplicitEuler}, but replaces the default central-
     * difference Jacobian with the supplied {@link JacobianStrategy} — e.g. an
     * AnalyticJacobian for an exact forward-mode-AD Jacobian.
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @param jacobianStrategy
     * @return
     */
    public static double[] stepImplicitEuler(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps,
            JacobianStrategy jacobianStrategy) {
        return stepImplicitEulerCore(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, NO_OP, jacobianStrategy);
    }

    /**
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @return
     */
    public static double[][] stepImplicitEulerWithHistory(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps) {
        return stepImplicitEulerWithHistory(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, null);
    }

    /**
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @param jacobianStrategy
     * @return
     */
    public static double[][] stepImplicitEulerWithHistory(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps,
            JacobianStrategy jacobianStrategy) {
        List<double[]> rows = new ArrayList<>(steps + 1);
        StepListener recorder = (t, y) -> {
            double[] row = new double[1 + systemSize];
            row[0] = t;
            System.arraycopy(y, 0, row, 1, systemSize);
            rows.add(row);
        };
        stepImplicitEulerCore(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, recorder, jacobianStrategy);
        return rows.toArray(new double[0][]);
    }

    private static double[] stepImplicitEulerCore(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps,
            StepListener listener, JacobianStrategy jacobianStrategyOrNull) {
        validateHandle(dy_dt);
        validateSlots(tSlot, ySlotStart, systemSize, frameSize);
        if (steps <= 0) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }

        double[] vars = new double[frameSize];
        double[] currentY = y0.clone();
        double h = (tEnd - t0) / steps;
        double t = t0;

        final int MAX_NEWTON_ITER = 30;
        final double NEWTON_TOLERANCE = 1e-9;

        JacobianStrategy jacobianStrategy = jacobianStrategyOrNull != null
                ? jacobianStrategyOrNull
                : finiteDifferenceStrategy(dy_dt, ySlotStart, systemSize);

        double[] nextYGuess = new double[systemSize];
        double[] f_guess = new double[systemSize];
        double[] G = new double[systemSize];
        double[][] dfDy = new double[systemSize][systemSize];
        double[][] jacobian = new double[systemSize][systemSize];
        double[] deltaY = new double[systemSize];
        double[][] linSysScratch = new double[systemSize][systemSize + 1];

        listener.onStep(t, currentY);

        for (int i = 0; i < steps; i++) {
            double nextT = t + h;

            vars[tSlot] = t;
            System.arraycopy(currentY, 0, vars, ySlotStart, systemSize);
            dy_dt.apply(vars, f_guess);

            for (int j = 0; j < systemSize; j++) {
                nextYGuess[j] = currentY[j] + h * f_guess[j];
            }

            boolean converged = false;
            for (int k = 0; k < MAX_NEWTON_ITER; k++) {
                vars[tSlot] = nextT;
                System.arraycopy(nextYGuess, 0, vars, ySlotStart, systemSize);
                dy_dt.apply(vars, f_guess);

                double gNorm = 0.0;
                for (int j = 0; j < systemSize; j++) {
                    G[j] = nextYGuess[j] - currentY[j] - h * f_guess[j];
                    gNorm += G[j] * G[j];
                }
                gNorm = Math.sqrt(gNorm);

                if (gNorm < NEWTON_TOLERANCE) {
                    converged = true;
                    break;
                }

                // vars already holds (nextT, nextYGuess) from the residual evaluation above —
                // the Jacobian strategy (finite-difference or analytic) evaluates at that point.
                jacobianStrategy.computeDfDy(vars, dfDy);
                for (int row = 0; row < systemSize; row++) {
                    for (int col = 0; col < systemSize; col++) {
                        jacobian[row][col] = (row == col ? 1.0 : 0.0) - h * dfDy[row][col];
                    }
                }

                if (!solveLinearSystem(jacobian, G, deltaY, systemSize, linSysScratch)) {
                    break;
                }

                for (int j = 0; j < systemSize; j++) {
                    nextYGuess[j] -= deltaY[j];
                }
            }

            if (!converged) {
                System.err.println("Warning: Newton-Raphson failed to converge at t = " + nextT);
            }

            System.arraycopy(nextYGuess, 0, currentY, 0, systemSize);
            t = nextT;
            listener.onStep(t, currentY);
        }

        return currentY;
    }

    // ------------------------------------------------------------------
// BDF2 (2nd-order Backward Differentiation Formula)
// ------------------------------------------------------------------
    /**
     * BDF2: 2nd-order-accurate, L-stable implicit multistep method. L-stability
     * makes it damp fast transients cleanly (no oscillation), unlike
     * Crank-Nicolson — the preferred choice when users report oscillatory
     * artifacts on very stiff systems with implicit Euler's accuracy no longer
     * being sufficient.
     *
     * The first step is bootstrapped with backward (implicit) Euler, since BDF2
     * requires both y_n and y_{n-1}; from step 2 onward it uses the full BDF2
     * formula: y_{n+1} - (4/3)y_n + (1/3)y_{n-1} = (2h/3) f(t_{n+1}, y_{n+1})
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @return
     */
    public static double[] stepBDF2(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps) {
        return stepBDF2Core(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, NO_OP, null);
    }

    /**
     * Same as {@link #stepBDF2}, but replaces the default central-difference
     * Jacobian with the supplied {@link JacobianStrategy} — e.g. an
     * AnalyticJacobian backed by an nth-order automatic differentiator.
     *
     * @param dy_dt
     * @param tSlot
     * @param ySlotStart
     * @param systemSize
     * @param frameSize
     * @param t0
     * @param y0
     * @param tEnd
     * @param steps
     * @param jacobianStrategy
     * @return
     */
    public static double[] stepBDF2(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps,
            JacobianStrategy jacobianStrategy) {
        return stepBDF2Core(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, NO_OP, jacobianStrategy);
    }

    public static double[][] stepBDF2WithHistory(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps) {
        return stepBDF2WithHistory(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, null);
    }

    public static double[][] stepBDF2WithHistory(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps,
            JacobianStrategy jacobianStrategy) {
        List<double[]> rows = new ArrayList<>(steps + 1);
        StepListener recorder = (t, y) -> {
            double[] row = new double[1 + systemSize];
            row[0] = t;
            System.arraycopy(y, 0, row, 1, systemSize);
            rows.add(row);
        };
        stepBDF2Core(dy_dt, tSlot, ySlotStart, systemSize, frameSize, t0, y0, tEnd, steps, recorder, jacobianStrategy);
        return rows.toArray(new double[0][]);
    }

    private static double[] stepBDF2Core(ODEFunction dy_dt, int tSlot, int ySlotStart, int systemSize,
            int frameSize, double t0, double[] y0, double tEnd, int steps,
            StepListener listener, JacobianStrategy jacobianStrategyOrNull) {
        validateHandle(dy_dt);
        validateSlots(tSlot, ySlotStart, systemSize, frameSize);
        if (steps <= 0) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }

        double[] vars = new double[frameSize];
        double[] prevY = new double[systemSize];    // y_{n-1}, unused until step index 1
        double[] currentY = y0.clone();              // y_n
        double h = (tEnd - t0) / steps;
        double t = t0;

        final int MAX_NEWTON_ITER = 30;
        final double NEWTON_TOLERANCE = 1e-9;

        JacobianStrategy jacobianStrategy = jacobianStrategyOrNull != null
                ? jacobianStrategyOrNull
                : finiteDifferenceStrategy(dy_dt, ySlotStart, systemSize);

        double[] nextYGuess = new double[systemSize];
        double[] f_guess = new double[systemSize];
        double[] G = new double[systemSize];
        double[][] dfDy = new double[systemSize][systemSize];
        double[][] jacobian = new double[systemSize][systemSize];
        double[] deltaY = new double[systemSize];
        double[][] linSysScratch = new double[systemSize][systemSize + 1];

        listener.onStep(t, currentY);

        for (int i = 0; i < steps; i++) {
            double nextT = t + h;
            boolean isBootstrapStep = (i == 0);

            vars[tSlot] = t;
            System.arraycopy(currentY, 0, vars, ySlotStart, systemSize);
            dy_dt.apply(vars, f_guess);
            for (int j = 0; j < systemSize; j++) {
                nextYGuess[j] = currentY[j] + h * f_guess[j];
            }

            boolean converged = false;
            for (int k = 0; k < MAX_NEWTON_ITER; k++) {
                vars[tSlot] = nextT;
                System.arraycopy(nextYGuess, 0, vars, ySlotStart, systemSize);
                dy_dt.apply(vars, f_guess);

                double gNorm = 0.0;
                if (isBootstrapStep) {
                    for (int j = 0; j < systemSize; j++) {
                        G[j] = nextYGuess[j] - currentY[j] - h * f_guess[j];
                        gNorm += G[j] * G[j];
                    }
                } else {
                    for (int j = 0; j < systemSize; j++) {
                        G[j] = nextYGuess[j] - (4.0 / 3.0) * currentY[j] + (1.0 / 3.0) * prevY[j]
                                - (2.0 * h / 3.0) * f_guess[j];
                        gNorm += G[j] * G[j];
                    }
                }
                gNorm = Math.sqrt(gNorm);

                if (gNorm < NEWTON_TOLERANCE) {
                    converged = true;
                    break;
                }

                jacobianStrategy.computeDfDy(vars, dfDy);
                double jacobianScale = isBootstrapStep ? h : (2.0 * h / 3.0);
                for (int row = 0; row < systemSize; row++) {
                    for (int col = 0; col < systemSize; col++) {
                        jacobian[row][col] = (row == col ? 1.0 : 0.0) - jacobianScale * dfDy[row][col];
                    }
                }

                if (!solveLinearSystem(jacobian, G, deltaY, systemSize, linSysScratch)) {
                    break;
                }

                for (int j = 0; j < systemSize; j++) {
                    nextYGuess[j] -= deltaY[j];
                }
            }

            if (!converged) {
                System.err.println("Warning: Newton-Raphson failed to converge at t = " + nextT);
            }

            System.arraycopy(currentY, 0, prevY, 0, systemSize);
            System.arraycopy(nextYGuess, 0, currentY, 0, systemSize);
            t = nextT;
            listener.onStep(t, currentY);
        }

        return currentY;
    }

    /**
     * Default JacobianStrategy: central-difference approximation with dynamic
     * epsilon scaling (relative to variable magnitude) for better accuracy
     * across variables of different scale. f_plus/f_minus are captured once per
     * solve, not reallocated per Newton iteration.
     */
    private static JacobianStrategy finiteDifferenceStrategy(ODEFunction dy_dt, int ySlotStart, int systemSize) {
        final double BASE_EPSILON = 1e-7;
        final double[] f_plus = new double[systemSize];
        final double[] f_minus = new double[systemSize];
        return (vars, outDfDy) -> {
            for (int col = 0; col < systemSize; col++) {
                int targetIndex = ySlotStart + col;
                double originalValue = vars[targetIndex];
                double epsilon = BASE_EPSILON * Math.max(1.0, Math.abs(originalValue));
                try {
                    vars[targetIndex] = originalValue + epsilon;
                    dy_dt.apply(vars, f_plus);

                    vars[targetIndex] = originalValue - epsilon;
                    dy_dt.apply(vars, f_minus);

                    double doubleEpsilonInverse = 1.0 / (2.0 * epsilon);
                    for (int row = 0; row < systemSize; row++) {
                        outDfDy[row][col] = (f_plus[row] - f_minus[row]) * doubleEpsilonInverse;
                    }
                } finally {
                    // Guarantee vars is restored even if dy_dt.apply throws mid-column,
                    // so shared state is never left perturbed for the next column/caller.
                    vars[targetIndex] = originalValue;
                }
            }
        };
    }

    private static boolean solveLinearSystem(double[][] A, double[] b, double[] x, int n, double[][] M) {
        // M is caller-owned scratch space, sized [n][n+1]. No allocation here.
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, M[i], 0, n);
            M[i][n] = b[i];
        }

        for (int p = 0; p < n; p++) {
            int max = p;
            for (int i = p + 1; i < n; i++) {
                if (Math.abs(M[i][p]) > Math.abs(M[max][p])) {
                    max = i;
                }
            }

            double[] temp = M[p];
            M[p] = M[max];
            M[max] = temp;

            if (Math.abs(M[p][p]) < 1e-12) {
                return false;
            }

            for (int i = p + 1; i < n; i++) {
                double alpha = M[i][p] / M[p][p];
                for (int j = p; j <= n; j++) {
                    M[i][j] -= alpha * M[p][j];
                }
            }
        }

        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += M[i][j] * x[j];
            }
            x[i] = (M[i][n] - sum) / M[i][i];
        }

        return true;
    }

    /**
     * Resampling — used to honor a requested `points` count against either a
     * fixed-step history (already uniform) or an adaptive RK45 history
     * (irregularly spaced accepted steps). Resamples a [t, y1..yn] history
     * matrix onto `points` uniformly spaced t values spanning history's first
     * and last t, via piecewise-linear interpolation between bracketing rows.
     * history must have at least 2 rows and be monotonic in t (either
     * increasing or decreasing — matches whichever integration direction
     * produced it).
     *
     * @param history
     * @param points
     * @return
     */
    public static double[][] resample(double[][] history, int points) {
        if (history == null || history.length < 2) {
            throw new IllegalArgumentException("history must contain at least 2 rows");
        }
        if (points < 2) {
            throw new IllegalArgumentException("points must be >= 2, got " + points);
        }

        int cols = history[0].length;
        double t0 = history[0][0];
        double tEnd = history[history.length - 1][0];
        double direction = Math.signum(tEnd - t0);
        if (direction == 0.0) {
            // Degenerate: single instant repeated `points` times.
            double[][] out = new double[points][cols];
            for (int i = 0; i < points; i++) {
                System.arraycopy(history[0], 0, out[i], 0, cols);
            }
            return out;
        }

        double[][] out = new double[points][cols];
        int searchFrom = 0;

        for (int p = 0; p < points; p++) {
            double frac = p / (double) (points - 1);
            double targetT = t0 + frac * (tEnd - t0);

            // Advance to the bracketing segment [history[i], history[i+1]] containing targetT.
            while (searchFrom < history.length - 2
                    && (direction > 0 ? history[searchFrom + 1][0] < targetT
                            : history[searchFrom + 1][0] > targetT)) {
                searchFrom++;
            }

            double[] lo = history[searchFrom];
            double[] hi = history[searchFrom + 1];
            double span = hi[0] - lo[0];
            double alpha = (span == 0.0) ? 0.0 : (targetT - lo[0]) / span;

            out[p][0] = targetT;
            for (int c = 1; c < cols; c++) {
                out[p][c] = lo[c] + alpha * (hi[c] - lo[c]);
            }
        }

        return out;
    }
}
