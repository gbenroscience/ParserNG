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
 */ package com.github.gbenroscience.math.numericalmethods;

import java.lang.invoke.MethodHandle;
import static java.lang.Math.abs;

/**
 * High-performance root finder utilizing MethodHandles. Fallback Chain:
 * Newtonian -> Brent's -> Secant -> Bisection -> Self-Evaluator
 *
 * * @author GBEMIRO
 */
public class TurboRootFinder {

    private final MethodHandle targetHandle;
    private final MethodHandle derivativeHandle;
    private final int varIndex;

    private double x1;
    private double x2;
    /**
     * The number of iterations with any of the methods before switching to
     * another method
     */
    private int iterations = DEFAULT_ITERATIONS;
    public static final int DEFAULT_ITERATIONS = 2000;
    private final double[] dataFrame = new double[256];
    private static final double tolerances[] = {5e-16, 1e-15, 5e-15, 1e-14, 5e-14, 1e-13, 5e-13, 1e-12, 5e-12, 1e-11};

    public TurboRootFinder(MethodHandle targetHandle, MethodHandle derivativeHandle,
            int varIndex, double x1, double x2, int iterations) {
        this.targetHandle = targetHandle;
        this.derivativeHandle = derivativeHandle;
        this.varIndex = varIndex;
        this.x1 = x1;
        this.x2 = x2;
        this.iterations = iterations;
    }

    private double eval(double x) throws Throwable {
        dataFrame[varIndex] = x;
        // Direct invocation: assumes (double[])double or (double[])Object
        Object result = targetHandle.invoke(dataFrame);
        return (result instanceof Double) ? (Double) result : ((double[]) result)[0];
    }

    private double evalDeriv(double x) throws Throwable {
        if (derivativeHandle == null) {
            return Double.NaN;
        }

        dataFrame[varIndex] = x;
        Object result = derivativeHandle.invoke(dataFrame);
        return (result instanceof Double) ? (Double) result : ((double[]) result)[0];
    }

    private double evalX(double x) throws Throwable {
        dataFrame[varIndex] = x;

        // The turbo-compiled handle is already (double[])double or (double[])Object
        // Check its actual signature and invoke accordingly
        if (targetHandle.type().parameterCount() == 1
                && targetHandle.type().parameterType(0) == double[].class) {
            // Already accepts array - invoke directly
            Object result = targetHandle.invoke(dataFrame);

            // Extract scalar from result (may be double or double[])
            if (result instanceof Number) {
                return ((Number) result).doubleValue();
            } else if (result instanceof double[]) {
                double[] arr = (double[]) result;
                return arr.length > 0 ? arr[0] : Double.NaN;
            }
            return (double) result;
        } else if (targetHandle.type().parameterCount() > 1) {
            // Legacy handle in (double, double, ...)double form - needs spreading
            int paramCount = targetHandle.type().parameterCount();
            MethodHandle spreader = targetHandle.asSpreader(double[].class, paramCount);
            return (double) spreader.invoke(dataFrame);
        } else {
            // Zero-arity constant - just invoke
            return (double) targetHandle.invoke();
        }
    }

    private double evalDerivX(double x) throws Throwable {
        if (derivativeHandle == null) {
            return 0.0;
        }

        dataFrame[varIndex] = x;

        // Same logic as eval()
        if (derivativeHandle.type().parameterCount() == 1
                && derivativeHandle.type().parameterType(0) == double[].class) {
            Object result = derivativeHandle.invoke(dataFrame);

            if (result instanceof Number) {
                return ((Number) result).doubleValue();
            } else if (result instanceof double[]) {
                double[] arr = (double[]) result;
                return arr.length > 0 ? arr[0] : Double.NaN;
            }
            return (double) result;
        } else if (derivativeHandle.type().parameterCount() > 1) {
            int paramCount = derivativeHandle.type().parameterCount();
            MethodHandle spreader = derivativeHandle.asSpreader(double[].class, paramCount);
            return (double) spreader.invoke(dataFrame);
        } else {
            return (double) derivativeHandle.invoke();
        }
    }

    private boolean verifyRoot(double ans) {
        if (Double.isNaN(ans) || Double.isInfinite(ans)) {
            return false;
        }
        try {
            double val = eval(ans);
            return abs(val) <= 5.0e-11; // Lenient check for fallback triggers
        } catch (Throwable t) {
            return false;
        }
    }

    public double findRoots() {
        double ans;

        // 1. Newtonian
        if (derivativeHandle != null) {
            ans = runNewtonian();
            if (verifyRoot(ans)) {
                return ans;
            }
        }

        // 2. Brent's Method (New)
        ans = runBrentsMethod();
        if (verifyRoot(ans)) {
            return ans;
        }

        // 3. Secant
        ans = runSecant();
        if (verifyRoot(ans)) {
            return ans;
        }

        // 4. Bisection
        ans = runBisection();
        if (verifyRoot(ans)) {
            return ans;
        }

        // 5. Self-Evaluator (Final Resort)
        ans = runSelfEvaluator();
        if (verifyRoot(ans)) {
            return ans;
        }

        return Double.NaN;
    }

    /**
     * Brent's Method: Combines Bisection, Secant, and Inverse Quadratic
     * Interpolation.
     */
    private double runBrentsMethod() {
        double a = x1, b = x2, c = x1, d = 0, e = 0;
        try {
            double fa = eval(a);
            double fb = eval(b);

            if (fa * fb > 0) {
                return Double.NaN; // Requires opposite signs
            }
            double fc = fa;

            for (int i = 0; i < iterations; i++) {
                if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
                    c = a;
                    fc = fa;
                    d = b - a;
                    e = d;
                }
                if (abs(fc) < abs(fb)) {
                    a = b;
                    b = c;
                    c = a;
                    fa = fb;
                    fb = fc;
                    fc = fa;
                }

                double tol = 2 * 1e-15 * abs(b) + 0.5 * 1e-15;
                double xm = 0.5 * (c - b);

                if (abs(xm) <= tol || fb == 0) {
                    return b;
                }

                if (abs(e) >= tol && abs(fa) > abs(fb)) {
                    double s = fb / fa;
                    double p, q;
                    if (a == c) {
                        p = 2.0 * xm * s;
                        q = 1.0 - s;
                    } else {
                        q = fa / fc;
                        double r = fb / fc;
                        p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                    }
                    if (p > 0) {
                        q = -q;
                    }
                    p = abs(p);
                    if (2.0 * p < Math.min(3.0 * xm * q - abs(tol * q), abs(e * q))) {
                        e = d;
                        d = p / q;
                    } else {
                        d = xm;
                        e = d;
                    }
                } else {
                    d = xm;
                    e = d;
                }
                a = b;
                fa = fb;
                if (abs(d) > tol) {
                    b += d;
                } else {
                    b += Math.copySign(tol, xm);
                }
                fb = eval(b);
            }
        } catch (Throwable t) {
        }
        return Double.NaN;
    }

    private double runNewtonian() {
        double x = x1;

        try {
            for (int i = 0; i < iterations; i++) {
                double fx = eval(x);
                double dfx = evalDeriv(x);
                if (abs(dfx) < 1e-12) {
                    break;
                }
                double ratio = fx / dfx;
                x -= ratio;
                if (abs(ratio) <= 5.0e-16) {
                    return x;
                }
            }
        } catch (Throwable t) {
            t.printStackTrace();
        }
        return Double.NaN;
    }

    private double runSecant() {
        double xOne = x1, xTwo = x2;
        try {
            double f1 = eval(xOne), f2 = eval(xTwo);
            for (int i = 0; i < iterations; i++) {
                if (abs(f2 - f1) < 1e-14) {
                    break;
                }
                double x = xTwo - (f2 * ((xTwo - xOne) / (f2 - f1)));
                xOne = xTwo;
                xTwo = x;
                f1 = f2;
                f2 = eval(xTwo);
                if (abs(f2) <= 5.0e-16) {
                    return xTwo;
                }
            }
        } catch (Throwable t) {
        }
        return Double.NaN;

    }

    private double runBisection() {
        double a = x1, b = x2;
        try {
            double fa = eval(a), fb = eval(b);
            if (fa * fb > 0) {
                return Double.NaN;
            }
            for (int i = 0; i < iterations; i++) {
                double mid = 0.5 * (a + b);
                double fMid = eval(mid);
                if (abs(fMid) <= 5.0e-16 || (b - a) < 1e-15) {
                    return mid;
                }
                if (Math.signum(fMid) == Math.signum(fa)) {
                    a = mid;
                    fa = fMid;
                } else {
                    b = mid;
                }
            }
        } catch (Throwable t) {
        }
        return Double.NaN;
    }

    private double runSelfEvaluator() {
        double xOriginal = x1, currentX = x1;
        try {
            double f = eval(currentX);
            // Flavor 1
            for (int i = 0; i < iterations && abs(f) >= 5.0E-16; i++) {
                currentX -= (f / (f - 8));
                f = eval(currentX);
            }
            if (abs(f) < 5.0E-16) {
                return currentX;
            }
            // Flavor 2
            currentX = xOriginal;
            f = eval(currentX);
            for (int i = 0; i < iterations && abs(f) >= 5.0E-16; i++) {
                currentX -= (f / (f + 8));
                f = eval(currentX);
            }
            return currentX;
        } catch (Throwable t) {
        }
        return Double.NaN;
    }
}
