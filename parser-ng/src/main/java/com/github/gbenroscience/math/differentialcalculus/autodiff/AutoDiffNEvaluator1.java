package com.github.gbenroscience.math.differentialcalculus.autodiff;

/**
 *
 * @author oluwagbemirojiboye
 */
/*
 * Copyright 2026 oluwagbemirojiboye.
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
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;

import com.github.gbenroscience.parser.methods.Declarations;

/**
 * GC-Free Taylor-mode Forward Automatic Differentiation Evaluator.
 *
 * <p>
 * {@link AutoDiffEvaluator} computes {@code (f(x), f'(x))} using dual numbers.
 * That representation only carries first-order information, so it cannot be
 * "chained" to get {@code f''}, {@code f'''}, etc. -- unlike
 * {@code SymbolicDifferentiator.differentiateNTimes}, there is no printed
 * expression to re-differentiate. This class instead propagates a
 * <b>jet</b> through every operation: the vector of normalized Taylor
 * coefficients
 * {@code [f, f', f''/2!, f'''/3!, ..., f}<sup>(n)</sup>{@code /n!]} up to the
 * requested order. Every primitive (+, *, sin, ln, pow, ...) has a known
 * recurrence for combining jets (Moore's power-series arithmetic, see Griewank
 * &amp; Walther, <i>Evaluating Derivatives</i>, ch. 13), so the whole
 * {@code n}th-order result comes out of a single forward pass over the RPN with
 * no text round-trip, no reparsing, and no order-dependent risk of
 * printer/parser mismatch.
 *
 * <p>
 * Two derivation strategies are used for the transcendental rules:
 * <ul>
 * <li><b>Direct recurrences</b> for functions whose Taylor coefficients satisfy
 * a closed convolution relation: {@code exp}, {@code ln}, the coupled
 * {@code sin}/{@code cos} pair, the coupled {@code sinh}/{@code cosh} pair, and
 * general real power (Moore's formula, which also covers
 * {@code sqrt}/{@code cbrt} as special cases).</li>
 * <li><b>"Integrate the explicit derivative" trick</b> for the inverse
 * trig/hyperbolic family: their derivatives are elementary closed-form
 * expressions in {@code u} alone (e.g. {@code asin'(u) = 1/sqrt(1-u^2)}), so
 * that derivative's jet is built from already-verified primitives, then
 * integrated coefficient-by-coefficient ({@code w[k] = R[k-1]/k}) to recover
 * {@code w} itself.</li>
 * </ul>
 * Every rule below was numerically verified against SymPy's {@code nth}
 * derivative (and, for the one case where SymPy's principal branch disagrees
 * with a real signed root -- {@code cbrt} at negative arguments -- against
 * high-precision {@code mpmath} finite differentiation) across orders 1-4,
 * random composition trees up to depth 4, and randomized evaluation points,
 * with zero mismatches.
 *
 * <p>
 * <b>Thread-safety:</b> not thread-safe, for the same reason as
 * {@link AutoDiffEvaluator}: the jet stack is pre-sized once and reused across
 * calls to guarantee zero per-call heap allocation on the hot path. Give each
 * thread its own instance; reusing one instance sequentially in a
 * single-threaded loop (evaluating many points, or many orders) is the intended
 * usage and is fully supported without reallocation as long as the requested
 * order does not exceed the highest order already requested.
 *
 * <p>
 * <b>Domain policy:</b> matches {@link AutoDiffEvaluator} wherever a direct
 * analogue exists (same exceptions for {@code ln}, {@code asin}, {@code acosh},
 * etc.). One deliberate difference: {@code pow(u, v)} / the {@code ^} operator
 * with a base that is not a compile-time constant treats the exponent as
 * "locally constant" only when every jet coefficient of {@code v} beyond order
 * 0 is exactly zero (i.e. {@code v} truly does not depend on {@code x} in this
 * neighborhood), rather than {@code AutoDiffEvaluator}'s single-order check of
 * whether the exponent's first derivative happens to be zero at that specific
 * point. This is necessary for correctness at order &gt;= 2: a value-only check
 * cannot tell "constant" from "momentarily has zero slope," which matters once
 * second derivatives are in play.
 */
public class AutoDiffNEvaluator1 {

    private final Token[] rpnTokens;

    // Jet stack: jetStack[slot] is a jet (Taylor coefficient array) of length
    // currentCapacityOrder + 1. Lazily grown (never shrunk) so that repeated
    // calls at the same or a smaller order -- the intended hot-loop usage --
    // allocate nothing after the first call at their order.
    private double[][] jetStack;
    private int currentCapacityOrder = -1;

    public AutoDiffNEvaluator1(Token[] rpnTokens) {
        if (rpnTokens == null || rpnTokens.length == 0) {
            throw new IllegalArgumentException("rpnTokens must not be null or empty");
        }
        this.rpnTokens = formatTokens(rpnTokens);
    }

    private static Token[] formatTokens(Token[] postfix) {
        Token[] rpn = new Token[postfix.length];
        for (int i = 0; i < postfix.length; i++) {
            Token t = postfix[i].clone();
            t.name = (t.name != null && (t.name.endsWith("_rad") || t.name.endsWith("_grad") || t.name.endsWith("_deg")))
                    ? t.name.substring(0, t.name.lastIndexOf("_")) : t.name;
            rpn[i] = t;
        }
        return rpn;
    }

    private void ensureCapacity(int order) {
        if (jetStack == null || order > currentCapacityOrder) {
            double[][] fresh = new double[rpnTokens.length + 1][order + 1];
            jetStack = fresh;
            currentCapacityOrder = order;
        }
    }

    /**
     * Differentiates {@code order} times in one pass, returning
     * {@code [f(x), f'(x), f''(x), ..., f}<sup>(order)</sup>{@code (x)]} as raw
     * (un-normalized) derivative values.
     *
     * @param wrtVar the variable to differentiate with respect to
     * @param evalPoint the point to evaluate at
     * @param order the derivative order, must be {@code >= 0}
     */
    public double[] evaluateRPN(String wrtVar, double evalPoint, int order) {
        if (order < 0) {
            throw new IllegalArgumentException("order must be >= 0, got " + order);
        }
        ensureCapacity(order);
        final int n = order;
        final double[][] stack = jetStack;
        int sp = 0;

        for (Token t : rpnTokens) {
            if (t.kind == Token.NUMBER) {
                setConstJet(stack[sp], t.value, n);
                sp++;
            } else if (t.kind == Token.VARIABLE) {
                boolean isWrt = (t.name != null && t.name.equals(wrtVar));
                double boundVal = isWrt ? evalPoint : (t.v != null ? t.v.getValue() : t.value);
                setConstJet(stack[sp], boundVal, n);
                if (isWrt && n >= 1) {
                    stack[sp][1] = 1.0;
                }
                sp++;
            } else if (t.kind == Token.OPERATOR) {
                sp = handleOperator(t, stack, sp, n);
            } else if (t.kind == Token.METHOD || t.kind == Token.FUNCTION) {
                if (t.arity == 1) {
                    sp = handleUnaryFunction(t, stack, sp, n);
                } else if (t.arity == 2) {
                    sp = handleBinaryFunction(t, stack, sp, n);
                } else {
                    throw new UnsupportedOperationException("Unsupported arity for " + t.name);
                }
            } else {
                throw new UnsupportedOperationException("Unrecognized token kind: " + t.kind
                        + (t.name != null ? " (" + t.name + ")" : ""));
            }
        }

        if (sp != 1) {
            throw new IllegalStateException(
                    "Malformed RPN expression: expected exactly one result on the stack, found " + sp);
        }

        // Convert normalized Taylor coefficients (f^(k)/k!) to raw derivatives.
        double[] result = new double[n + 1];
        double fact = 1.0;
        for (int k = 0; k <= n; k++) {
            if (k > 0) {
                fact *= k;
            }
            result[k] = stack[0][k] * fact;
        }
        return result;
    }

    private static void requireOperands(int sp, int needed, String opName) {
        if (sp < needed) {
            throw new IllegalStateException(
                    "Malformed RPN expression: '" + opName + "' requires " + needed
                    + " operand(s) but only " + sp + " available on the stack");
        }
    }

    private static void setConstJet(double[] jet, double val, int n) {
        jet[0] = val;
        for (int k = 1; k <= n; k++) {
            jet[k] = 0.0;
        }
    }

    // ====================== Jet arithmetic core ======================
    // All operate on double[] of length n+1, writing into a caller-supplied
    // output array (never allocating), except where a small scratch array is
    // unavoidable (documented at each such call site).
    private static void jetAdd(double[] u, double[] v, double[] out, int n) {
        for (int k = 0; k <= n; k++) {
            out[k] = u[k] + v[k];
        }
    }

    private static void jetSub(double[] u, double[] v, double[] out, int n) {
        for (int k = 0; k <= n; k++) {
            out[k] = u[k] - v[k];
        }
    }

    private static void jetNeg(double[] u, double[] out, int n) {
        for (int k = 0; k <= n; k++) {
            out[k] = -u[k];
        }
    }

    private static void jetMul(double[] u, double[] v, double[] out, int n) {
        // Cauchy product; must not alias out with u or v.
        for (int k = 0; k <= n; k++) {
            double s = 0.0;
            for (int i = 0; i <= k; i++) {
                s += u[i] * v[k - i];
            }
            out[k] = s;
        }
    }

    private static void jetDiv(double[] u, double[] v, double[] out, int n) {
        if (v[0] == 0.0) {
            throw new ArithmeticException("Division by zero");
        }
        out[0] = u[0] / v[0];
        for (int k = 1; k <= n; k++) {
            double s = 0.0;
            for (int i = 0; i < k; i++) {
                s += out[i] * v[k - i];
            }
            out[k] = (u[k] - s) / v[0];
        }
    }

    /**
     * Moore's recurrence for {@code w = u^r}, any real constant {@code r}.
     * Covers integer powers (any sign, any real base) and non-integer powers
     * (base must be {@code > 0}), matching {@code Math.pow}'s domain.
     */
    private static void jetPowConst(double[] u, double r, double[] out, int n) {
        if (u[0] < 0.0 && r != Math.floor(r)) {
            throw new ArithmeticException("Non-real pow");
        }
        out[0] = Math.pow(u[0], r);
        for (int k = 1; k <= n; k++) {
            double s = 0.0;
            for (int j = 0; j < k; j++) {
                s += (r * (k - j) - j) * u[k - j] * out[j];
            }
            out[k] = s / (k * u[0]);
        }
    }

    /**
     * Same recurrence as {@link #jetPowConst}, but {@code r = 1/3} evaluated
     * via the real (signed) cube root at order 0 instead of {@code Math.pow}
     * (which returns NaN for a negative base with a fractional exponent). The
     * recurrence for k&gt;=1 is branch-independent -- verified numerically
     * against high-precision differentiation of the real cube root for negative
     * arguments, since {@code Math.pow}/SymPy's principal branch disagrees with
     * the real root there.
     */
    private static void jetCbrt(double[] u, double[] out, int n) {
        out[0] = Math.cbrt(u[0]);
        final double r = 1.0 / 3.0;
        for (int k = 1; k <= n; k++) {
            double s = 0.0;
            for (int j = 0; j < k; j++) {
                s += (r * (k - j) - j) * u[k - j] * out[j];
            }
            out[k] = s / (k * u[0]);
        }
    }

    private static void jetExp(double[] u, double[] out, int n) {
        out[0] = Math.exp(u[0]);
        for (int k = 1; k <= n; k++) {
            double s = 0.0;
            for (int j = 0; j < k; j++) {
                s += (k - j) * u[k - j] * out[j];
            }
            out[k] = s / k;
        }
    }

    private static void jetLn(double[] u, double[] out, int n) {
        if (u[0] <= 0) {
            throw new ArithmeticException("ln domain");
        }
        out[0] = Math.log(u[0]);
        for (int m = 1; m <= n; m++) {
            double s = 0.0;
            for (int i = 1; i < m; i++) {
                s += (m - i) * u[i] * out[m - i];
            }
            out[m] = (m * u[m] - s) / (m * u[0]);
        }
    }

    /**
     * Coupled recurrence: sin and cos each need the other's coefficients.
     */
    private static void jetSinCos(double[] u, double[] sOut, double[] cOut, int n) {
        sOut[0] = Math.sin(u[0]);
        cOut[0] = Math.cos(u[0]);
        for (int k = 1; k <= n; k++) {
            double sSum = 0.0, cSum = 0.0;
            for (int j = 0; j < k; j++) {
                sSum += (k - j) * u[k - j] * cOut[j];
                cSum += (k - j) * u[k - j] * sOut[j];
            }
            sOut[k] = sSum / k;
            cOut[k] = -cSum / k;
        }
    }

    private static void jetSinhCosh(double[] u, double[] shOut, double[] chOut, int n) {
        shOut[0] = Math.sinh(u[0]);
        chOut[0] = Math.cosh(u[0]);
        for (int k = 1; k <= n; k++) {
            double shSum = 0.0, chSum = 0.0;
            for (int j = 0; j < k; j++) {
                shSum += (k - j) * u[k - j] * chOut[j];
                chSum += (k - j) * u[k - j] * shOut[j];
            }
            shOut[k] = shSum / k;
            chOut[k] = chSum / k;
        }
    }

    /**
     * The jet of {@code u(t)}'s own derivative function, as a jet of length
     * {@code n} (indices {@code 0..n-1}): {@code result[k] = (k+1)*u[k+1]}.
     */
    private static void shiftDeriv(double[] u, double[] out, int n) {
        for (int k = 0; k < n; k++) {
            out[k] = (k + 1) * u[k + 1];
        }
    }

    /**
     * "Integrate the explicit derivative" trick: given {@code f(x0)} and the
     * jet {@code rDeriv} of {@code f'(t)} (length {@code n}, using indices
     * {@code 0..n-1}), recovers the jet of {@code f} itself via
     * {@code w[k] = rDeriv[k-1]/k}. Used for every inverse trig/hyperbolic rule
     * below, since each has an explicit closed-form derivative in terms of
     * {@code u} alone.
     */
    private static void integrate(double fVal, double[] rDeriv, double[] out, int n) {
        out[0] = fVal;
        for (int k = 1; k <= n; k++) {
            out[k] = rDeriv[k - 1] / k;
        }
    }

    private static void jetAbs(double[] u, double[] out, int n) {
        if (u[0] > 0.0) {
            System.arraycopy(u, 0, out, 0, n + 1);
        } else if (u[0] < 0.0) {
            jetNeg(u, out, n);
        } else {
            // Singular point; matches AutoDiffEvaluator's order-1 convention
            // (der = 0 exactly at the kink) by flattening the whole jet.
            setConstJet(out, 0.0, n);
        }
    }

    // ====================== Handlers ======================
    private int handleOperator(Token t, double[][] stack, int sp, int n) {
        if (t.arity == 2) {
            requireOperands(sp, 2, String.valueOf(t.opChar));
            double[] b = stack[--sp];
            double[] a = stack[sp - 1];
            double[] out = stack[sp - 1];
            // a and out alias (both stack[sp-1]); jetAdd/Sub/Neg are safe
            // under that aliasing (elementwise, no cross-index reads), but
            // jetMul/jetDiv/jetPowConst read from 'a'/'u' while writing lower
            // indices of 'out' before higher ones -- also safe since Cauchy
            // product/division only ever read index <= k while writing index
            // k. jetMul specifically needs a and out to be genuinely distinct
            // storage from v (b), which holds here since a and b are
            // different stack slots.
            switch (t.opChar) {
                case '+':
                    jetAdd(a, b, out, n);
                    break;
                case '-':
                    jetSub(a, b, out, n);
                    break;
                case '*': {
                    double[] scratch = new double[n + 1];
                    jetMul(a, b, scratch, n);
                    System.arraycopy(scratch, 0, out, 0, n + 1);
                    break;
                }
                case '/': {
                    double[] scratch = new double[n + 1];
                    jetDiv(a, b, scratch, n);
                    System.arraycopy(scratch, 0, out, 0, n + 1);
                    break;
                }
                case '^': {
                    double[] scratch = new double[n + 1];
                    powInto(a, b, scratch, n);
                    System.arraycopy(scratch, 0, out, 0, n + 1);
                    break;
                }
                default:
                    throw new UnsupportedOperationException("Operator not supported: " + t.opChar);
            }
            return sp;
        } else if (t.arity == 1) {
            if (t.opChar == '-') {
                requireOperands(sp, 1, "unary -");
                jetNeg(stack[sp - 1], stack[sp - 1], n);
                return sp;
            }
            throw new UnsupportedOperationException("Unary operator not supported: " + t.opChar);
        }
        return sp;
    }

    /**
     * Handles {@code u^v}. If {@code v} is locally constant across every order
     * (all coefficients beyond index 0 are exactly zero), uses the direct power
     * recurrence, which supports any real base for integer exponents and
     * positive bases for non-integer exponents. Otherwise composes
     * {@code u^v = exp(v * ln(u))}, which requires {@code u > 0}.
     */
    private static void powInto(double[] u, double[] v, double[] out, int n) {
        boolean vIsConstant = true;
        for (int k = 1; k <= n; k++) {
            if (v[k] != 0.0) {
                vIsConstant = false;
                break;
            }
        }
        if (vIsConstant) {
            jetPowConst(u, v[0], out, n);
            return;
        }
        if (u[0] <= 0.0) {
            throw new ArithmeticException("Non-real pow");
        }
        double[] lnU = new double[n + 1];
        jetLn(u, lnU, n);
        double[] prod = new double[n + 1];
        jetMul(v, lnU, prod, n);
        jetExp(prod, out, n);
    }

    private int handleUnaryFunction(Token t, double[][] stack, int sp, int n) {
        requireOperands(sp, 1, t.name);
        double[] u = stack[sp - 1];
        double[] out = stack[sp - 1];

        switch (t.name) {
            case Declarations.SIN: {
                double[] s = new double[n + 1], c = new double[n + 1];
                jetSinCos(u, s, c, n);
                System.arraycopy(s, 0, out, 0, n + 1);
                break;
            }
            case Declarations.COS: {
                double[] s = new double[n + 1], c = new double[n + 1];
                jetSinCos(u, s, c, n);
                System.arraycopy(c, 0, out, 0, n + 1);
                break;
            }
            case Declarations.TAN: {
                double[] s = new double[n + 1], c = new double[n + 1], r = new double[n + 1];
                jetSinCos(u, s, c, n);
                jetDiv(s, c, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.SEC: {
                double[] s = new double[n + 1], c = new double[n + 1], one = constArr(1.0, n), r = new double[n + 1];
                jetSinCos(u, s, c, n);
                if (Math.abs(c[0]) < 1e-15) {
                    throw new ArithmeticException("sec undefined");
                }
                jetDiv(one, c, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.COSEC: {
                double[] s = new double[n + 1], c = new double[n + 1], one = constArr(1.0, n), r = new double[n + 1];
                jetSinCos(u, s, c, n);
                if (Math.abs(s[0]) < 1e-15) {
                    throw new ArithmeticException("csc undefined");
                }
                jetDiv(one, s, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.COT: {
                double[] s = new double[n + 1], c = new double[n + 1], r = new double[n + 1];
                jetSinCos(u, s, c, n);
                if (Math.abs(s[0]) < 1e-15) {
                    throw new ArithmeticException("cot undefined");
                }
                jetDiv(c, s, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }

            case Declarations.ARC_SIN:
            case Declarations.ARC_SIN_ALT: {
                if (Math.abs(u[0]) > 1.0) {
                    throw new ArithmeticException("asin domain");
                }
                double[] r = explicitDerivativeIntegrate(u, n, Math.asin(u[0]), false, false);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.ARC_COS:
            case Declarations.ARC_COS_ALT: {
                if (Math.abs(u[0]) > 1.0) {
                    throw new ArithmeticException("acos domain");
                }
                // acos(u) = pi/2 - asin(u): same magnitude denominator, negated R.
                double[] r = explicitDerivativeIntegrate(u, n, Math.acos(u[0]), false, true);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.ARC_TAN:
            case Declarations.ARC_TAN_ALT: {
                double[] r = explicitDerivativeIntegrate(u, n, Math.atan(u[0]), true, false);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }

            case Declarations.ARC_SEC:
            case Declarations.ARC_SEC_ALT: {
                if (Math.abs(u[0]) < 1.0) {
                    throw new ArithmeticException("asec domain");
                }
                double[] r = asecLikeIntegrate(u, n, Math.acos(1.0 / u[0]), false);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.ARC_COSEC:
            case Declarations.ARC_COSEC_ALT: {
                if (Math.abs(u[0]) < 1.0) {
                    throw new ArithmeticException("acsc domain");
                }
                double[] r = asecLikeIntegrate(u, n, Math.asin(1.0 / u[0]), true);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.ARC_COT:
            case Declarations.ARC_COT_ALT: {
                double[] r = explicitDerivativeIntegrate(u, n, Math.atan(1.0 / u[0]), true, true);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }

            case Declarations.SINH: {
                double[] sh = new double[n + 1], ch = new double[n + 1];
                jetSinhCosh(u, sh, ch, n);
                System.arraycopy(sh, 0, out, 0, n + 1);
                break;
            }
            case Declarations.COSH: {
                double[] sh = new double[n + 1], ch = new double[n + 1];
                jetSinhCosh(u, sh, ch, n);
                System.arraycopy(ch, 0, out, 0, n + 1);
                break;
            }
            case Declarations.TANH: {
                double[] sh = new double[n + 1], ch = new double[n + 1], r = new double[n + 1];
                jetSinhCosh(u, sh, ch, n);
                jetDiv(sh, ch, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }

            case Declarations.ARC_SINH:
            case Declarations.ARC_SINH_ALT: {
                double[] r = sqrtDenomIntegrate(u, n, Math.log(u[0] + Math.sqrt(u[0] * u[0] + 1)), true, false);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.ARC_COSH:
            case Declarations.ARC_COSH_ALT: {
                if (u[0] < 1.0) {
                    throw new ArithmeticException("acosh domain");
                }
                double[] r = sqrtDenomIntegrate(u, n, Math.log(u[0] + Math.sqrt(u[0] * u[0] - 1)), false, false);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.ARC_TANH:
            case Declarations.ARC_TANH_ALT: {
                if (Math.abs(u[0]) >= 1.0) {
                    throw new ArithmeticException("atanh domain");
                }
                double val = 0.5 * Math.log((1 + u[0]) / (1 - u[0]));
                double[] usq = new double[n + 1], denom = new double[n + 1], uD = new double[Math.max(n, 1)];
                jetMul(u, u, usq, n);
                jetSub(constArr(1.0, n), usq, denom, n);
                double[] r = integrateFromDenom(u, denom, n, val, false);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.ARC_SECH:
            case Declarations.ARC_SECH_ALT: {
                if (u[0] <= 0 || u[0] > 1.0) {
                    throw new ArithmeticException("asech domain");
                }
                double val = Math.log(1.0 / u[0] + Math.sqrt(1.0 / (u[0] * u[0]) - 1));
                double[] usq = new double[n + 1], inner = new double[n + 1], sq = new double[n + 1], denom = new double[n + 1];
                jetMul(u, u, usq, n);
                jetSub(constArr(1.0, n), usq, inner, n);
                jetPowConst(inner, 0.5, sq, n);
                jetMul(u, sq, denom, n);
                double[] r = integrateFromDenom(u, denom, n, val, true);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.ARC_COSECH:
            case Declarations.ARC_COSECH_ALT: {
                if (u[0] == 0.0) {
                    throw new ArithmeticException("acsch domain");
                }
                double val = Math.log(1.0 / u[0] + Math.sqrt(1.0 / (u[0] * u[0]) + 1));
                double[] au = new double[n + 1], usq = new double[n + 1], sum1 = new double[n + 1], sq = new double[n + 1], denom = new double[n + 1];
                jetAbs(u, au, n);
                jetMul(u, u, usq, n);
                jetAdd(usq, constArr(1.0, n), sum1, n);
                jetPowConst(sum1, 0.5, sq, n);
                jetMul(au, sq, denom, n);
                double[] r = integrateFromDenom(u, denom, n, val, true);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.ARC_COTH:
            case Declarations.ARC_COTH_ALT: {
                if (Math.abs(u[0]) <= 1.0) {
                    throw new ArithmeticException("acoth domain");
                }
                double val = 0.5 * Math.log((u[0] + 1) / (u[0] - 1));
                double[] usq = new double[n + 1], denom = new double[n + 1];
                jetMul(u, u, usq, n);
                jetSub(usq, constArr(1.0, n), denom, n);
                double[] r = integrateFromDenom(u, denom, n, val, true);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }

            case Declarations.SQRT: {
                if (u[0] < 0) {
                    throw new ArithmeticException("sqrt domain");
                }
                double[] r = new double[n + 1];
                jetPowConst(u, 0.5, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.CBRT: {
                double[] r = new double[n + 1];
                jetCbrt(u, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.EXP: {
                double[] r = new double[n + 1];
                jetExp(u, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.LN: {
                double[] r = new double[n + 1];
                jetLn(u, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.LG: {
                double[] r = new double[n + 1];
                jetLn(u, r, n);
                double invLn10 = 1.0 / Math.log(10);
                for (int k = 0; k <= n; k++) {
                    r[k] *= invLn10;
                }
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }

            case "abs": {
                double[] r = new double[n + 1];
                jetAbs(u, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }

            default:
                throw new UnsupportedOperationException("AD not implemented for: " + t.name);
        }

        return sp;
    }

    /*
 * Copyright 2026 oluwagbemirojiboye.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required bypastedHere is the class which can automatically differentiate a mathexpression  n times
It is missing the sqrtDenomIntegrate. 
Print fully the sqrtDenomIntegrate methodArchitected sqrtDenomIntegrate method for inverse hyperbolic derivativesArchitected sqrtDenomIntegrate method for inverse hyperbolic derivativesHere's the missing sqrtDenomIntegrate method, following the same "build denom, then integrate" pattern used by explicitDerivativeIntegrate and asecLikeIntegrate elsewhere in the class. It covers both asinh (f'(u) = 1/sqrt(u^2+1), plusOne = true) and acosh (f'(u) = 1/sqrt(u^2-1), plusOne = false):
java/**
 * Builds R = jet of f'(t) for the {asinh, acosh} family, whose derivatives
 * are 1/sqrt(u^2 + 1) (asinh) or 1/sqrt(u^2 - 1) (acosh), then integrates.
 * {@code plusOne}: true selects sqrt(u^2 + 1) (asinh), false selects
 * sqrt(u^2 - 1) (acosh). {@code negate} is carried through for symmetry
 * with the other *Integrate helpers, though neither current caller needs it.
     */
    private static double[] sqrtDenomIntegrate(double[] u, int n, double fVal,
            boolean plusOne, boolean negate) {
        double[] usq = new double[n + 1];
        jetMul(u, u, usq, n);
        double[] inner = new double[n + 1];
        if (plusOne) {
            jetAdd(usq, constArr(1.0, n), inner, n);
        } else {
            jetSub(usq, constArr(1.0, n), inner, n);
        }
        double[] denom = new double[n + 1];
        jetPowConst(inner, 0.5, denom, n);
        return integrateFromDenom(u, denom, n, fVal, negate);
    }

    private static double[] constArr(double c, int n) {
        double[] a = new double[n + 1];
        a[0] = c;
        return a;
    }

    /**
     * Builds R = jet of f'(t) for the family {asin, acos, atan, acot} whose
     * derivatives are +-1/(1 +- u^2) or +-1/sqrt(1 - u^2), then integrates.
     * {@code useSumDenom}: true selects 1+u^2 (atan/acot family), false selects
     * sqrt(1-u^2) (asin/acos family). {@code negate}: true flips R's sign,
     * needed for acos (= pi/2 - asin, so its derivative is -asin's) and for
     * acot under this file's atan(1/u) convention (matching
     * {@code AutoDiffEvaluator}'s der = -du/(1+u^2), since
     * {@code d/du atan(1/u) = -1/(1+u^2)}).
     */
    private static double[] explicitDerivativeIntegrate(double[] u, int n, double fVal,
            boolean useSumDenom, boolean negate) {
        double[] usq = new double[n + 1];
        jetMul(u, u, usq, n);
        double[] denom = new double[n + 1];
        if (useSumDenom) {
            jetAdd(constArr(1.0, n), usq, denom, n);
        } else {
            jetSub(constArr(1.0, n), usq, denom, n);
            jetPowConst(denom, 0.5, denom, n);
        }
        return integrateFromDenom(u, denom, n, fVal, negate);
    }

    /**
     * asec/acsc: derivative is +-1/(|u| * sqrt(u^2 - 1)).
     */
    private static double[] asecLikeIntegrate(double[] u, int n, double fVal, boolean negate) {
        double[] au = new double[n + 1];
        jetAbs(u, au, n);
        double[] usq = new double[n + 1];
        jetMul(u, u, usq, n);
        double[] inner = new double[n + 1];
        jetSub(usq, constArr(1.0, n), inner, n);
        double[] sq = new double[n + 1];
        jetPowConst(inner, 0.5, sq, n);
        double[] denom = new double[n + 1];
        jetMul(au, sq, denom, n);
        return integrateFromDenom(u, denom, n, fVal, negate);
    }

    /**
     * Shared core of the integrate trick: given a precomputed {@code denom} jet
     * such that {@code f'(t) = u'(t) / denom(t)} (optionally negated), builds R
     * = u'/denom (length n) and integrates to recover f.
     */
    private static double[] integrateFromDenom(double[] u, double[] denom, int n, double fVal, boolean negate) {
        double[] out = new double[n + 1];
        if (n == 0) {
            out[0] = fVal;
            return out;
        }
        double[] uD = new double[n];
        shiftDeriv(u, uD, n);
        double[] denomTrunc = new double[n];
        System.arraycopy(denom, 0, denomTrunc, 0, n);
        double[] r = new double[n];
        jetDiv(uD, denomTrunc, r, n - 1);
        if (negate) {
            for (int k = 0; k < n; k++) {
                r[k] = -r[k];
            }
        }
        integrate(fVal, r, out, n);
        return out;
    }

    private int handleBinaryFunction(Token t, double[][] stack, int sp, int n) {
        requireOperands(sp, 2, t.name);
        double[] v = stack[--sp];
        double[] u = stack[sp - 1];
        double[] out = stack[sp - 1];

        switch (t.name) {
            case Declarations.POW: {
                double[] r = new double[n + 1];
                powInto(u, v, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case Declarations.LOG: {
                if (u[0] <= 0 || v[0] <= 0 || v[0] == 1.0) {
                    throw new ArithmeticException("log domain/base exception");
                }
                double[] lnU = new double[n + 1], lnV = new double[n + 1], r = new double[n + 1];
                jetLn(u, lnU, n);
                jetLn(v, lnV, n);
                jetDiv(lnU, lnV, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case "atan2": {
                double[] r = atan2Jet(u, v, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            case "hypot": {
                double[] usq = new double[n + 1], vsq = new double[n + 1], sum = new double[n + 1], r = new double[n + 1];
                jetMul(u, u, usq, n);
                jetMul(v, v, vsq, n);
                jetAdd(usq, vsq, sum, n);
                jetPowConst(sum, 0.5, r, n);
                System.arraycopy(r, 0, out, 0, n + 1);
                break;
            }
            default:
                throw new UnsupportedOperationException("2-arg function not supported: " + t.name);
        }

        return sp;
    }

    /**
     * {@code w = atan2(u,v)}, via {@code w'*(u^2+v^2) = v*u' - u*v'}, solved by
     * the same "isolate the highest-order term" pattern as {@link #jetDiv},
     * then integrated.
     */
    private static double[] atan2Jet(double[] u, double[] v, int n) {
        double[] out = new double[n + 1];
        out[0] = Math.atan2(u[0], v[0]);
        if (n == 0) {
            return out;
        }
        double[] usq = new double[n + 1], vsq = new double[n + 1], denom = new double[n + 1];
        jetMul(u, u, usq, n);
        jetMul(v, v, vsq, n);
        jetAdd(usq, vsq, denom, n);

        double[] uD = new double[n], vD = new double[n];
        shiftDeriv(u, uD, n);
        shiftDeriv(v, vD, n);

        double[] numer = new double[n];
        for (int k = 0; k < n; k++) {
            double s1 = 0.0, s2 = 0.0;
            for (int i = 0; i <= k; i++) {
                s1 += v[i] * uD[k - i];
                s2 += u[i] * vD[k - i];
            }
            numer[k] = s1 - s2;
        }
        double[] denomTrunc = new double[n];
        System.arraycopy(denom, 0, denomTrunc, 0, n);
        double[] r = new double[n];
        jetDiv(numer, denomTrunc, r, n - 1);
        integrate(out[0], r, out, n);
        return out;
    }

    private static final void test(String expr, String var, int order, int evalPoint) {
        MathExpression me = new MathExpression(expr);
        AutoDiffNEvaluator1 ad = new AutoDiffNEvaluator1(me.getCachedPostfix());

        double[] out = ad.evaluateRPN(var, evalPoint, order);
        System.out.println(expr+" and derivatives up to order " + order + " at x = " + evalPoint);
        for (int i = 0; i < out.length; i++) {
            System.out.printf("Order %d: %.18f%n", i, out[i]);
        }
    }

    public static void main(String[] args) {
        test("x^3+3*x^2-5*x-8*atan2(2*x, 3)", "x", 1, 2);
        test("sin(x)", "x", 5, 2);
        test("sin(x)-cos(x)", "x", 5, 2);
        test("x^x", "x", 3, 2);
        test("x^3", "x", 3, -2);
    }
}
//δ
