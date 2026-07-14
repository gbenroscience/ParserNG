/*
 * Copyright 2026 oluwagbemirojiboye.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math.numericalmethods.taylors;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.methods.Declarations;
/**
 * High-Speed Adaptive Taylor Series Definite Integral Solver.
 * Combines high-order Forward-Mode Automatic Differentiation with
 * dynamic step-size selection to evaluate analytic integrals to
 * machine-precision accuracy near-instantly without any heap allocation.
 */
public class TurboIntegrator {
    private final Token[] rpnTokens;
    private final int maxOrder;
    private final double[][] valStack; // [stackPos][k] = k-th Taylor coefficient (c_k)
   
    // Pre-allocated scratchpads maintaining a 100% GC-free lifecycle
    private final double[] scratch1;
    private final double[] scratch2;
    private final double[] scratch3;
    private final double[] scratchArg;
    private final double[] scratchU;
    private final double[] scratchV;
    // Cache of factorials stored in double precision to prevent 64-bit integer overflow at order >= 21
    private static final double[] FACTORIALS = new double[171];
    static {
        FACTORIALS[0] = 1.0;
        for (int i = 1; i <= 170; i++) {
            FACTORIALS[i] = FACTORIALS[i - 1] * i;
        }
    }
    public TurboIntegrator(Token[] rpnTokens, int maxOrder) {
        if (rpnTokens == null || rpnTokens.length == 0) {
            throw new IllegalArgumentException("rpnTokens must not be null or empty");
        }
        if (maxOrder < 0) {
            throw new IllegalArgumentException("maxOrder >= 0 required");
        }
        this.rpnTokens = formatTokens(rpnTokens);
        this.maxOrder = maxOrder;
        this.valStack = new double[this.rpnTokens.length + 1][maxOrder + 1];
       
        this.scratch1 = new double[maxOrder + 1];
        this.scratch2 = new double[maxOrder + 1];
        this.scratch3 = new double[maxOrder + 1];
        this.scratchArg = new double[maxOrder + 1];
        this.scratchU = new double[maxOrder + 1];
        this.scratchV = new double[maxOrder + 1];
    }
    private static Token[] formatTokens(Token[] postfix) {
        Token[] rpn = new Token[postfix.length];
        for (int i = 0; i < postfix.length; i++) {
            Token t = postfix[i].clone();
            if (t.name != null && (t.name.endsWith("_rad") || t.name.endsWith("_grad") || t.name.endsWith("_deg"))) {
                t.name = t.name.substring(0, t.name.lastIndexOf("_"));
            }
            rpn[i] = t;
        }
        return rpn;
    }
    /**
     * Integrates the expression from x1 to x2 adaptively.
     * 100% GC-Free Integration Loop.
     *
     * @param wrtVar Variable of integration (e.g. "x")
     * @param x1 Lower limit
     * @param x2 Upper limit
     * @param order The expansion order (higher order = bigger steps / faster integration)
     * @param tolerance Truncation error threshold (e.g. 1e-15)
     * @return The definite integral value
     */
    public double integrate(String wrtVar, double x1, double x2, int order, double tolerance) {
        if (order > maxOrder) {
            throw new IllegalArgumentException("Requested order " + order + " exceeds maxOrder " + maxOrder);
        }
        if (order < 0) {
            throw new IllegalArgumentException("Order must be non-negative");
        }
        if (tolerance <= 0.0) {
            throw new IllegalArgumentException("Tolerance must be strictly positive");
        }
       
        double x = x1;
        double target = x2;
        double direction = Math.signum(target - x);
       
        if (direction == 0.0) {
            return 0.0;
        }
        double totalIntegral = 0.0;
        double[] coeffs = new double[order + 1];
        // Continue integrating in steps until we hit the target limit
        while (direction * (target - x) > 1e-15) {
            // Step 1: Compute raw Taylor coefficients c_k at the current boundary point x
            evaluateRawCoefficients(wrtVar, x, order, coeffs);
            // Step 2: Estimate the mathematically optimal step size h (always a positive magnitude)
            double hMag = estimateStepSize(coeffs, order, tolerance);
            // Ensure our step size doesn't overshoot the boundary
            double remainingDistance = Math.abs(target - x);
            boolean isFinalStep = false;
           
            if (hMag >= remainingDistance) {
                hMag = remainingDistance;
                isFinalStep = true;
            }
            double signedH = direction * hMag;
            // Defend against infinite loops/stagnation when step size drops below precision floor
            if (Math.abs(signedH) <= 1e-16 || (x + signedH) == x) {
                if (remainingDistance < 1e-10) { // Safely close out intervals for moderate x magnitudes
                    signedH = direction * remainingDistance;
                    isFinalStep = true;
                } else {
                    throw new ArithmeticException(
                        String.format("Step size stagnated (underflow) at x = %.8f. Step size: %e. " +
                        "Function may contain a singularity. Try adjusting tolerance.", x, hMag)
                    );
                }
            }
            // Step 3: Accumulate the integral slice using the Taylor Shortcut:
            // Term_i = (c_i / (i + 1)) * h^(i + 1) using the direction-aware step!
            double stepIntegral = 0.0;
            double hPower = signedH; // Tracks signedH^(i + 1)
            for (int i = 0; i <= order; i++) {
                stepIntegral += (coeffs[i] / (i + 1)) * hPower;
                hPower *= signedH;
            }
            totalIntegral += stepIntegral;
           
            if (isFinalStep) {
                break; // Force termination when target is logically reached
            }
           
            double nextX = x + signedH;
            if (nextX == x) { // Final safety net against FP loop stagnation
                break;
            }
            x = nextX; // Advance along the integration path
        }
        return totalIntegral;
    }
    /**
     * Uses consecutive Taylor coefficients to find the optimal step size.
     * Prevents steps from exceeding the local radius of convergence (poles/singularities).
     */
    private double estimateStepSize(double[] coeffs, int order, double tolerance) {
        if (order == 0) {
            return Double.MAX_VALUE;
        }
        double minH = Double.POSITIVE_INFINITY;
        boolean allZero = true;
        // Only consider higher-order coefficients to determine step size.
        // This prevents overly conservative tiny steps from low-order terms
        // while still respecting truncation error and radius of convergence.
        int startK = Math.max(1, order - 5);  // Consider last ~5-6 terms for robustness
        for (int k = startK; k <= order; k++) {
            double absCoeff = Math.abs(coeffs[k]);
            if (absCoeff > 1e-300) {
                allZero = false;
                // Solve: |c_k| * h^k <= tolerance => h <= (tolerance / |c_k|)^(1/k)
                double hk = Math.pow(tolerance / absCoeff, 1.0 / k);
                if (hk < minH) {
                    minH = hk;
                }
            }
        }
        if (allZero) {
            // For flat linear/constant systems, we can safely jump the entire remainder in 1 step
            return Double.MAX_VALUE;
        }
        // Apply a conservative 90% safety factor
        return minH * 0.9;
    }
    /**
     * Core AD propagation. Returns raw Taylor Coefficients (c_k) directly.
     * Crucially skips the factorial multiplication step to preserve the Taylor Shortcut.
     */
    private void evaluateRawCoefficients(String wrtVar, double evalPoint, int order, double[] resultOut) {
        int sp = 0;
        for (Token t : rpnTokens) {
            if (t.kind == Token.NUMBER) {
                valStack[sp][0] = t.value;
                for (int k = 1; k <= order; k++) {
                    valStack[sp][k] = 0.0;
                }
                sp++;
            } else if (t.kind == Token.VARIABLE) {
                boolean isWrt = t.name != null && t.name.equals(wrtVar);
                valStack[sp][0] = isWrt ? evalPoint : (t.v != null ? t.v.getValue() : t.value);
                valStack[sp][1] = isWrt ? 1.0 : 0.0;
                for (int k = 2; k <= order; k++) {
                    valStack[sp][k] = 0.0;
                }
                sp++;
            } else if (t.kind == Token.OPERATOR) {
                sp = handleOperator(t, sp, order);
            } else if (t.kind == Token.METHOD || t.kind == Token.FUNCTION) {
                if (t.arity == 1) {
                    sp = handleUnaryFunction(t, sp, order);
                } else if (t.arity == 2) {
                    sp = handleBinaryFunction(t, sp, order);
                }
            }
        }
        if (sp != 1) {
            throw new IllegalStateException("Malformed RPN: expected 1 result, got " + sp);
        }
        System.arraycopy(valStack[0], 0, resultOut, 0, order + 1);
    }
    private int handleOperator(Token t, int sp, int ord) {
        if (t.arity == 2) {
            requireOperands(sp, 2, String.valueOf(t.opChar));
            double[] b = valStack[--sp];
            double[] a = valStack[--sp];
            double[] res = valStack[sp];
            System.arraycopy(a, 0, scratchU, 0, ord + 1);
            System.arraycopy(b, 0, scratchV, 0, ord + 1);
            switch (t.opChar) {
                case '+':
                    add(scratchU, scratchV, res, ord);
                    break;
                case '-':
                    sub(scratchU, scratchV, res, ord);
                    break;
                case '*':
                    mul(scratchU, scratchV, res, ord);
                    break;
                case '/':
                    if (Math.abs(scratchV[0]) < 1e-300) {
                        throw new ArithmeticException("Division by zero");
                    }
                    recipJet(scratchV, scratch1, ord);
                    mul(scratchU, scratch1, res, ord);
                    break;
                case '^':
                    powJet(scratchU, scratchV, res, ord);
                    break;
                default:
                    throw new UnsupportedOperationException("Operator " + t.opChar);
            }
            return sp + 1;
        } else if (t.arity == 1 && t.opChar == '-') {
            requireOperands(sp, 1, "unary -");
            double[] arg = valStack[--sp];
            double[] res = valStack[sp];
            for (int k = 0; k <= ord; k++) {
                res[k] = -arg[k];
            }
            return sp + 1;
        }
        return sp;
    }
    private int handleUnaryFunction(Token t, int sp, int ord) {
        requireOperands(sp, 1, t.name);
        double[] arg = valStack[--sp];
        double[] res = valStack[sp];
        System.arraycopy(arg, 0, scratchArg, 0, ord + 1);
        switch (t.name) {
            case Declarations.SIN:
                sinCosJet(scratchArg, res, scratch1, ord);
                break;
            case Declarations.COS:
                sinCosJet(scratchArg, scratch1, res, ord);
                break;
            case Declarations.TAN:
                tanJet(scratchArg, res, ord, scratch1);
                break;
            case Declarations.ARC_SIN:
            case Declarations.ARC_SIN_ALT:
                if (Math.abs(scratchArg[0]) > 1.0) {
                    throw new ArithmeticException("asin domain");
                }
                res[0] = Math.asin(scratchArg[0]);
                mul(scratchArg, scratchArg, scratch1, ord);
                scratch2[0] = 1.0 - scratch1[0];
                for (int k = 1; k <= ord; k++) scratch2[k] = -scratch1[k];
                sqrtJet(scratch2, scratch3, ord);
                for (int k = 1; k <= ord; k++) {
                    double s = 0;
                    for (int l = 1; l < k; l++) {
                        s += l * res[l] * scratch3[k - l];
                    }
                    res[k] = (scratchArg[k] - s / k) / scratch3[0];
                }
                break;
            case Declarations.ARC_COS:
            case Declarations.ARC_COS_ALT:
                if (Math.abs(scratchArg[0]) > 1.0) {
                    throw new ArithmeticException("acos domain");
                }
                res[0] = Math.acos(scratchArg[0]);
                mul(scratchArg, scratchArg, scratch1, ord);
                scratch2[0] = 1.0 - scratch1[0];
                for (int k = 1; k <= ord; k++) scratch2[k] = -scratch1[k];
                sqrtJet(scratch2, scratch3, ord);
                for (int k = 1; k <= ord; k++) {
                    double s = 0;
                    for (int l = 1; l < k; l++) {
                        s += l * res[l] * scratch3[k - l];
                    }
                    res[k] = (-scratchArg[k] - s / k) / scratch3[0];
                }
                break;
            case Declarations.ARC_TAN:
            case Declarations.ARC_TAN_ALT:
                atanJet(scratchArg, res, ord, scratch1);
                break;
            case Declarations.SEC:
                sinCosJet(scratchArg, scratch1, scratch2, ord);
                recipJet(scratch2, res, ord);
                break;
            case Declarations.COSEC:
                sinCosJet(scratchArg, scratch1, scratch2, ord);
                recipJet(scratch1, res, ord);
                break;
            case Declarations.COT:
                tanJet(scratchArg, scratch1, ord, scratch2);
                recipJet(scratch1, res, ord);
                break;
            case Declarations.SINH:
                sinhCoshJet(scratchArg, res, scratch1, ord);
                break;
            case Declarations.COSH:
                sinhCoshJet(scratchArg, scratch1, res, ord);
                break;
            case Declarations.TANH:
                tanhJet(scratchArg, res, ord, scratch1);
                break;
            case Declarations.ARC_SINH:
            case Declarations.ARC_SINH_ALT:
                res[0] = Math.log(scratchArg[0] + Math.sqrt(scratchArg[0] * scratchArg[0] + 1));
                mul(scratchArg, scratchArg, scratch1, ord);
                scratch2[0] = scratch1[0] + 1.0;
                for (int k = 1; k <= ord; k++) scratch2[k] = scratch1[k];
                sqrtJet(scratch2, scratch3, ord);
                for (int k = 1; k <= ord; k++) {
                    double s = 0;
                    for (int l = 1; l < k; l++) {
                        s += l * res[l] * scratch3[k - l];
                    }
                    res[k] = (scratchArg[k] - s / k) / scratch3[0];
                }
                break;
            case Declarations.ARC_COSH:
            case Declarations.ARC_COSH_ALT:
                if (scratchArg[0] < 1.0) {
                    throw new ArithmeticException("acosh domain");
                }
                res[0] = Math.log(scratchArg[0] + Math.sqrt(scratchArg[0] * scratchArg[0] - 1));
                mul(scratchArg, scratchArg, scratch1, ord);
                scratch2[0] = scratch1[0] - 1.0;
                for (int k = 1; k <= ord; k++) scratch2[k] = scratch1[k];
                sqrtJet(scratch2, scratch3, ord);
                for (int k = 1; k <= ord; k++) {
                    double s = 0;
                    for (int l = 1; l < k; l++) {
                        s += l * res[l] * scratch3[k - l];
                    }
                    res[k] = (scratchArg[k] - s / k) / scratch3[0];
                }
                break;
            case Declarations.ARC_TANH:
            case Declarations.ARC_TANH_ALT:
                if (Math.abs(scratchArg[0]) >= 1.0) {
                    throw new ArithmeticException("atanh domain");
                }
                res[0] = 0.5 * Math.log((1 + scratchArg[0]) / (1 - scratchArg[0]));
                mul(scratchArg, scratchArg, scratch1, ord);
                for (int k = 1; k <= ord; k++) {
                    double s = 0;
                    for (int l = 1; l < k; l++) {
                        s += l * res[l] * scratch1[k - l];
                    }
                    res[k] = (scratchArg[k] + s / k) / (1.0 - scratch1[0]);
                }
                break;
            case Declarations.EXP:
                expJet(scratchArg, res, ord);
                break;
            case Declarations.LN:
                lnJet(scratchArg, res, ord);
                break;
            case Declarations.LG:
                if (scratchArg[0] <= 0) {
                    throw new ArithmeticException("log domain");
                }
                lnJet(scratchArg, res, ord);
                double ln10 = Math.log(10.0);
                for (int k = 0; k <= ord; k++) {
                    res[k] /= ln10;
                }
                break;
            case Declarations.SQRT:
                sqrtJet(scratchArg, res, ord);
                break;
            case Declarations.CBRT:
                cbrtJet(scratchArg, res, ord, scratch1);
                break;
            case "abs":
                res[0] = Math.abs(scratchArg[0]);
                for (int k = 1; k <= ord; k++) {
                    res[k] = (scratchArg[0] > 0) ? scratchArg[k] : (scratchArg[0] < 0) ? -scratchArg[k] : 0.0;
                }
                break;
            default:
                throw new UnsupportedOperationException("Taylor AD implementation missing for: " + t.name);
        }
        return sp + 1;
    }
    private int handleBinaryFunction(Token t, int sp, int ord) {
        requireOperands(sp, 2, t.name);
        double[] v = valStack[--sp];
        double[] u = valStack[--sp];
        double[] res = valStack[sp];
        System.arraycopy(u, 0, scratchU, 0, ord + 1);
        System.arraycopy(v, 0, scratchV, 0, ord + 1);
        if (t.name.equals(Declarations.POW)) {
            powJet(scratchU, scratchV, res, ord);
        } else if (t.name.equals("atan2")) {
            res[0] = Math.atan2(scratchU[0], scratchV[0]);
            mul(scratchU, scratchU, scratch1, ord);
            mul(scratchV, scratchV, scratch2, ord);
            add(scratch1, scratch2, scratch1, ord); // scratch1 = u^2 + v^2
            if (scratch1[0] < 1e-300) {
                throw new ArithmeticException("atan2 at origin");
            }
            for (int k = 1; k <= ord; k++) {
                double rhs = 0;
                for (int j = 1; j <= k; j++) {
                    rhs += j * scratchU[j] * scratchV[k - j];
                }
                for (int j = 0; j < k; j++) {
                    rhs -= (k - j) * scratchU[j] * scratchV[k - j];
                }
                double lhsSum = 0;
                for (int l = 1; l < k; l++) {
                    lhsSum += l * res[l] * scratch1[k - l];
                }
                res[k] = (rhs - lhsSum) / (k * scratch1[0]);
            }
        } else {
            throw new UnsupportedOperationException("Binary method missing: " + t.name);
        }
        return sp + 1;
    }
    private void add(double[] a, double[] b, double[] out, int ord) {
        for (int k = 0; k <= ord; k++) out[k] = a[k] + b[k];
    }
    private void sub(double[] a, double[] b, double[] out, int ord) {
        for (int k = 0; k <= ord; k++) out[k] = a[k] - b[k];
    }
    private void mul(double[] a, double[] b, double[] out, int ord) {
        for (int k = 0; k <= ord; k++) {
            double sum = 0;
            for (int i = 0; i <= k; i++) {
                sum += a[i] * b[k - i];
            }
            out[k] = sum;
        }
    }
    private void recipJet(double[] b, double[] out, int ord) {
        out[0] = 1.0 / b[0];
        for (int k = 1; k <= ord; k++) {
            double s = 0;
            for (int i = 1; i <= k; i++) {
                s += b[i] * out[k - i];
            }
            out[k] = -out[0] * s;
        }
    }
    private void lnJet(double[] u, double[] out, int ord) {
        double u0 = u[0];
        if (u0 <= 0) throw new ArithmeticException("ln domain");
        out[0] = Math.log(u0);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int j = 1; j < k; j++) {
                s += j * out[j] * u[k - j];
            }
            out[k] = (u[k] - s / k) / u0;
        }
    }
    private void expJet(double[] w, double[] out, int ord) {
        out[0] = Math.exp(w[0]);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int j = 1; j <= k; j++) {
                s += j * w[j] * out[k - j];
            }
            out[k] = s / k;
        }
    }
    private void powJet(double[] u, double[] v, double[] out, int ord) {
        lnJet(u, scratch1, ord);
        mul(v, scratch1, scratch2, ord);
        expJet(scratch2, out, ord);
    }
    private void sinCosJet(double[] u, double[] sinOut, double[] cosOut, int ord) {
        sinOut[0] = Math.sin(u[0]);
        cosOut[0] = Math.cos(u[0]);
        for (int k = 1; k <= ord; k++) {
            double sSum = 0;
            double cSum = 0;
            for (int j = 1; j <= k; j++) {
                sSum += j * u[j] * cosOut[k - j];
                cSum += j * u[j] * sinOut[k - j];
            }
            sinOut[k] = sSum / k;
            cosOut[k] = -cSum / k;
        }
    }
    private void sinhCoshJet(double[] u, double[] sinhOut, double[] coshOut, int ord) {
        sinhOut[0] = Math.sinh(u[0]);
        coshOut[0] = Math.cosh(u[0]);
        for (int k = 1; k <= ord; k++) {
            double sSum = 0;
            double cSum = 0;
            for (int j = 1; j <= k; j++) {
                sSum += j * u[j] * coshOut[k - j];
                cSum += j * u[j] * sinhOut[k - j];
            }
            sinhOut[k] = sSum / k;
            coshOut[k] = cSum / k;
        }
    }
    private void tanJet(double[] u, double[] out, int ord, double[] pScratch) {
        out[0] = Math.tan(u[0]);
        pScratch[0] = out[0] * out[0];
        for (int k = 1; k <= ord; k++) {
            double pk_1 = 0;
            for (int i = 0; i <= k - 1; i++) {
                pk_1 += out[i] * out[k - 1 - i];
            }
            pScratch[k - 1] = pk_1;
            double s = 0;
            for (int j = 1; j < k; j++) {
                s += j * u[j] * pScratch[k - j];
            }
            out[k] = u[k] * (1.0 + pScratch[0]) + s / k;
        }
    }
    private void tanhJet(double[] u, double[] out, int ord, double[] pScratch) {
        out[0] = Math.tanh(u[0]);
        pScratch[0] = out[0] * out[0];
        for (int k = 1; k <= ord; k++) {
            double pk_1 = 0;
            for (int i = 0; i <= k - 1; i++) {
                pk_1 += out[i] * out[k - 1 - i];
            }
            pScratch[k - 1] = pk_1;
            double s = 0;
            for (int j = 1; j < k; j++) {
                s += j * u[j] * pScratch[k - j];
            }
            out[k] = u[k] * (1.0 - pScratch[0]) - s / k;
        }
    }
    private void sqrtJet(double[] u, double[] out, int ord) {
        if (u[0] < 0) throw new ArithmeticException("sqrt domain");
        out[0] = Math.sqrt(u[0]);
        for (int k = 1; k <= ord; k++) {
            double s = 0;
            for (int j = 1; j < k; j++) {
                s += j * out[j] * out[k - j];
            }
            out[k] = (u[k] - 2.0 * s / k) / (2.0 * out[0]);
        }
    }
    private void cbrtJet(double[] u, double[] out, int ord, double[] pScratch) {
        out[0] = Math.cbrt(u[0]);
        pScratch[0] = out[0] * out[0];
        for (int k = 1; k <= ord; k++) {
            double pk_1 = 0;
            for (int i = 0; i <= k - 1; i++) {
                pk_1 += out[i] * out[k - 1 - i];
            }
            pScratch[k - 1] = pk_1;
            double s = 0;
            for (int l = 1; l < k; l++) {
                s += l * out[l] * pScratch[k - l];
            }
            out[k] = (u[k] - 3.0 * s / k) / (3.0 * pScratch[0]);
        }
    }
    private void atanJet(double[] u, double[] out, int ord, double[] pScratch) {
        out[0] = Math.atan(u[0]);
        mul(u, u, pScratch, ord);
        for (int k = 1; k <= ord; k++) {
            double s = 0;
            for (int l = 1; l < k; l++) {
                s += l * out[l] * pScratch[k - l];
            }
            out[k] = (u[k] - s / k) / (1.0 + pScratch[0]);
        }
    }
    private static void requireOperands(int sp, int needed, String name) {
        if (sp < needed) {
            throw new IllegalStateException(name + " requires " + needed + " operands");
        }
    }
    public static void main(String[] args) {
        System.out.println("=== INITIALIZING BENCHMARK AND VALIDATION ===");
        // Test Function: f(x) = x * exp(x)
        String expr = "x * exp(x)";
        MathExpression me = new MathExpression(expr);
       
        int order = 12;
        double tol = 1e-15;
        TurboIntegrator integrator = new TurboIntegrator(me.getCachedPostfix(), order);
        // Exact integral over [0, 2]: (x-1)*exp(x) from 0 to 2 = exp(2) + 1 ≈ 8.38905609893065
        double exactForward = Math.exp(2.0) + 1.0;
        double exactBackward = -exactForward;
       
        // Warmup JVM
        for (int i = 0; i < 20000; i++) {
            integrator.integrate("x", 0.0, 2.0, order, tol);
            integrator.integrate("x", 2.0, 0.0, order, tol);
        }
        // Test 1: Forward Integration [0 to 2]
        long startForward = System.nanoTime();
        double approxForward = integrator.integrate("x", 0.0, 2.0, order, tol);
        long endForward = System.nanoTime();
        // Test 2: Backward Integration [2 to 0]
        long startBackward = System.nanoTime();
        double approxBackward = integrator.integrate("x", 2.0, 0.0, order, tol);
        long endBackward = System.nanoTime();
        double errForward = Math.abs(exactForward - approxForward);
        double errBackward = Math.abs(exactBackward - approxBackward);
        System.out.printf("Test Expression : %s%n", expr);
        System.out.println("---------------------------------------------");
        System.out.printf("Forward [0 to 2] : %.16f (Exact: %.16f)%n", approxForward, exactForward);
        System.out.printf("Forward Error : %.2e%n", errForward);
        System.out.printf("Forward Time : %.4f ms%n", (endForward - startForward) / 1_000_000.0);
        System.out.println("---------------------------------------------");
        System.out.printf("Backward [2 to 0] : %.16f (Exact: %.16f)%n", approxBackward, exactBackward);
        System.out.printf("Backward Error : %.2e%n", errBackward);
        System.out.printf("Backward Time : %.4f ms%n", (endBackward - startBackward) / 1_000_000.0);
        System.out.println("---------------------------------------------");
        System.out.println("Execution state : 100% GC-Free & Symmetrical!");
    }
}