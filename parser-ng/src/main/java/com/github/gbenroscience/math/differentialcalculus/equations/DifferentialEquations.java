package com.github.gbenroscience.math.differentialcalculus.equations;

import java.lang.invoke.MethodHandle;

/**
 * High-performance, JIT-optimized Vectorized Ordinary Differential Equation (ODE) solvers.
 * Supports arbitrary n-th order differential equations by reduction to first-order vector systems.
 * 
 * Target MethodHandle signature MUST strictly match: void f(double[] vars, double[] outDerivatives)
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
            DP_B45 = -212666291.0 / 524700000.0, DP_B46 = 187.0 / 2100.0, DP_B47 = 1.0 / 40.0;

    // Nodes (Time fractions)
    private static final double DP_C2 = 1.0 / 5.0, DP_C3 = 3.0 / 10.0, DP_C4 = 4.0 / 5.0, 
            DP_C5 = 8.0 / 9.0, DP_C6 = 1.0;

    public enum ODESolverMethod {
        EULER,              // Fast, O(h) error. Best for real-time graphics/particles.
        RK4,                // Classical 4th Order fixed-step system workhorse.
        RK45_DORMAND_PRINCE,// Adaptive-step size system engine (Industry standard).
        IMPLICIT_EULER      // Backwards implicit setup optimized for stiff vector spaces.
    }

    /**
     * Vectorized Explicit Euler solver for systems of equations.
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
     * @throws Throwable 
     */
    public static double[] stepEuler(MethodHandle dy_dt,
                                     int tSlot,
                                     int ySlotStart,
                                     int systemSize,
                                     int frameSize,
                                     double t0,
                                     double[] y0,
                                     double tEnd,
                                     int steps) throws Throwable {

        double[] vars = new double[frameSize];
        double[] currentY = y0.clone();
        double[] slopes = new double[systemSize];
        double h = (tEnd - t0) / steps;
        double t = t0;

        for (int i = 0; i < steps; i++) {
            vars[tSlot] = t;
            System.arraycopy(currentY, 0, vars, ySlotStart, systemSize);

            dy_dt.invokeExact(vars, slopes);

            for (int j = 0; j < systemSize; j++) {
                currentY[j] += h * slopes[j];
            }
            t += h;
        }
        return currentY;
    }

    /**
     * Vectorized 4th Order Classical Runge-Kutta (RK4) solver for systems of equations.
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
     * @throws Throwable 
     */
    public static double[] stepRK4(MethodHandle dy_dt,
                                   int tSlot,
                                   int ySlotStart,
                                   int systemSize,
                                   int frameSize,
                                   double t0,
                                   double[] y0,
                                   double tEnd,
                                   int steps) throws Throwable {

        double[] vars = new double[frameSize];
        double[] currentY = y0.clone();
        double[][] k = new double[4][systemSize];
        double h = (tEnd - t0) / steps;
        double t = t0;

        for (int i = 0; i < steps; i++) {
            // k1 = f(t, Y)
            vars[tSlot] = t;
            System.arraycopy(currentY, 0, vars, ySlotStart, systemSize);
            dy_dt.invokeExact(vars, k[0]);

            // k2 = f(t + h/2, Y + h/2 * k1)
            vars[tSlot] = t + h * 0.5;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * 0.5 * k[0][j];
            }
            dy_dt.invokeExact(vars, k[1]);

            // k3 = f(t + h/2, Y + h/2 * k2)
            vars[tSlot] = t + h * 0.5;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * 0.5 * k[1][j];
            }
            dy_dt.invokeExact(vars, k[2]);

            // k4 = f(t + h, Y + h * k3)
            vars[tSlot] = t + h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * k[2][j];
            }
            dy_dt.invokeExact(vars, k[3]);

            // Update Y
            for (int j = 0; j < systemSize; j++) {
                currentY[j] += (h / 6.0) * (k[0][j] + 2 * k[1][j] + 2 * k[2][j] + k[3][j]);
            }
            t += h;
        }
        return currentY;
    }

    /**
     * Vectorized Adaptive-step Dormand-Prince 5(4) solver for systems of equations.
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
     * @throws Throwable 
     */
    public static double[] stepRK45Adaptive(MethodHandle dy_dt,
                                            int tSlot,
                                            int ySlotStart,
                                            int systemSize,
                                            int frameSize,
                                            double t0,
                                            double[] y0,
                                            double tEnd,
                                            double initialH) throws Throwable {

        double[] vars = new double[frameSize];
        double[] currentY = y0.clone();
        double h = initialH;
        double t = t0;

        final double ATOL = 1e-8;
        final double RTOL = 1e-6;
        final double MIN_H = 1e-12;
        final double MAX_H = Math.max(tEnd - t0, 1.0);

        int maxSteps = 20000;
        int steps = 0;

        double[][] k = new double[7][systemSize];
        double[] y4 = new double[systemSize];
        double[] y5 = new double[systemSize];

        while (t < tEnd && steps++ < maxSteps) {
            if (t + h > tEnd) {
                h = tEnd - t;
            }

            // Stage 1
            vars[tSlot] = t;
            System.arraycopy(currentY, 0, vars, ySlotStart, systemSize);
            dy_dt.invokeExact(vars, k[0]);

            // Stage 2
            vars[tSlot] = t + DP_C2 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A21 * k[0][j]);
            }
            dy_dt.invokeExact(vars, k[1]);

            // Stage 3
            vars[tSlot] = t + DP_C3 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A31 * k[0][j] + DP_A32 * k[1][j]);
            }
            dy_dt.invokeExact(vars, k[2]);

            // Stage 4
            vars[tSlot] = t + DP_C4 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A41 * k[0][j] + DP_A42 * k[1][j] + DP_A43 * k[2][j]);
            }
            dy_dt.invokeExact(vars, k[3]);

            // Stage 5
            vars[tSlot] = t + DP_C5 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A51 * k[0][j] + DP_A52 * k[1][j] + DP_A53 * k[2][j] + DP_A54 * k[3][j]);
            }
            dy_dt.invokeExact(vars, k[4]);

            // Stage 6
            vars[tSlot] = t + DP_C6 * h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A61 * k[0][j] + DP_A62 * k[1][j] + DP_A63 * k[2][j] + DP_A64 * k[3][j] + DP_A65 * k[4][j]);
            }
            dy_dt.invokeExact(vars, k[5]);

            // Stage 7
            vars[tSlot] = t + h;
            for (int j = 0; j < systemSize; j++) {
                vars[ySlotStart + j] = currentY[j] + h * (DP_A71 * k[0][j] + DP_A73 * k[2][j] + DP_A74 * k[3][j] + DP_A75 * k[4][j] + DP_A76 * k[5][j]);
            }
            dy_dt.invokeExact(vars, k[6]);

            // Compute y5 and y4
            for (int j = 0; j < systemSize; j++) {
                y5[j] = currentY[j] + h * (DP_B51 * k[0][j] + DP_B53 * k[2][j] + DP_B54 * k[3][j] + DP_B55 * k[4][j] + DP_B56 * k[5][j]);
                y4[j] = currentY[j] + h * (DP_B41 * k[0][j] + DP_B43 * k[2][j] + DP_B44 * k[3][j] + DP_B45 * k[4][j] + DP_B46 * k[5][j] + DP_B47 * k[6][j]);
            }

            // Vector error analysis using max error ratio
            double maxErrorRatio = 0.0;
            for (int j = 0; j < systemSize; j++) {
                double error = Math.abs(y5[j] - y4[j]);
                double tol = ATOL + RTOL * Math.max(Math.abs(currentY[j]), Math.abs(y5[j]));
                maxErrorRatio = Math.max(maxErrorRatio, error / tol);
            }

            boolean stepAccepted = maxErrorRatio <= 1.0 || h <= MIN_H;

            if (stepAccepted) {
                t += h;
                System.arraycopy(y5, 0, currentY, 0, systemSize);

                double scale = (maxErrorRatio < 1e-20) ? 5.0 : 0.9 * Math.pow(1.0 / maxErrorRatio, 0.2);
                scale = Math.max(0.2, Math.min(scale, 5.0));
                h *= scale;
                h = Math.max(MIN_H, Math.min(h, MAX_H));
            } else {
                double scale = 0.9 * Math.pow(1.0 / maxErrorRatio, 0.2);
                scale = Math.max(0.2, Math.min(scale, 0.9));
                h *= scale;

                if (h < MIN_H) {
                    h = MIN_H;
                }
            }
        }

        if (steps >= maxSteps) {
            System.err.println("Warning: RK45 reached iteration limit (" + maxSteps + ")");
        }

        return currentY;
    }

    /**
     * Vectorized Implicit Backward Euler solver using multivariable Newton-Raphson
     * with numerical central-difference Jacobian approximations. Ideal for stiff systems.
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
     * @throws Throwable 
     */
    public static double[] stepImplicitEuler(MethodHandle dy_dt,
                                             int tSlot,
                                             int ySlotStart,
                                             int systemSize,
                                             int frameSize,
                                             double t0,
                                             double[] y0,
                                             double tEnd,
                                             int steps) throws Throwable {

        double[] vars = new double[frameSize];
        double[] currentY = y0.clone();
        double h = (tEnd - t0) / steps;
        double t = t0;

        final int MAX_NEWTON_ITER = 30;
        final double NEWTON_TOLERANCE = 1e-9;
        final double EPSILON = 1e-7;

        double[] nextYGuess = new double[systemSize];
        double[] f_guess = new double[systemSize];
        double[] G = new double[systemSize];
        double[] f_plus = new double[systemSize];
        double[] f_minus = new double[systemSize];
        double[][] jacobian = new double[systemSize][systemSize];
        double[] deltaY = new double[systemSize];

        for (int i = 0; i < steps; i++) {
            double nextT = t + h;

            // Predictor Step: Explicit Euler for initial guess
            vars[tSlot] = t;
            System.arraycopy(currentY, 0, vars, ySlotStart, systemSize);
            dy_dt.invokeExact(vars, f_guess);

            for (int j = 0; j < systemSize; j++) {
                nextYGuess[j] = currentY[j] + h * f_guess[j];
            }

            boolean converged = false;
            for (int k = 0; k < MAX_NEWTON_ITER; k++) {
                // Evaluate residual G(Y) = Y - Y_old - h * f(t_next, Y)
                vars[tSlot] = nextT;
                System.arraycopy(nextYGuess, 0, vars, ySlotStart, systemSize);
                dy_dt.invokeExact(vars, f_guess);

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

                // Build Jacobian using central differences
                for (int col = 0; col < systemSize; col++) {
                    double originalValue = nextYGuess[col];

                    // Forward
                    nextYGuess[col] = originalValue + EPSILON;
                    System.arraycopy(nextYGuess, 0, vars, ySlotStart, systemSize);
                    dy_dt.invokeExact(vars, f_plus);

                    // Backward
                    nextYGuess[col] = originalValue - EPSILON;
                    System.arraycopy(nextYGuess, 0, vars, ySlotStart, systemSize);
                    dy_dt.invokeExact(vars, f_minus);

                    nextYGuess[col] = originalValue; // Reset

                    for (int row = 0; row < systemSize; row++) {
                        double df_dy = (f_plus[row] - f_minus[row]) / (2.0 * EPSILON);
                        jacobian[row][col] = (row == col ? 1.0 : 0.0) - h * df_dy;
                    }
                }

                // Solve Jacobian * deltaY = -G
                if (!solveLinearSystem(jacobian, G, deltaY, systemSize)) {
                    break; // Singular matrix
                }

                // Apply correction
                for (int j = 0; j < systemSize; j++) {
                    nextYGuess[j] -= deltaY[j];
                }
            }

            if (!converged) {
                System.err.println("Warning: Newton-Raphson failed to converge at t = " + nextT);
            }

            System.arraycopy(nextYGuess, 0, currentY, 0, systemSize);
            t = nextT;
        }

        return currentY;
    }

    /**
     * Internal helper: Gaussian Elimination with partial pivoting.
     */
    private static boolean solveLinearSystem(double[][] A, double[] b, double[] x, int n) {
        double[][] M = new double[n][n + 1];

        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, M[i], 0, n);
            M[i][n] = b[i];
        }

        // Forward elimination with partial pivoting
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
                return false; // Singular matrix
            }

            for (int i = p + 1; i < n; i++) {
                double alpha = M[i][p] / M[p][p];
                for (int j = p; j <= n; j++) {
                    M[i][j] -= alpha * M[p][j];
                }
            }
        }

        // Back substitution
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += M[i][j] * x[j];   // Note: x should be initialized to 0 or handled properly
            }
            x[i] = (M[i][n] - sum) / M[i][i];
        }

        return true;
    }
}