package com.github.gbenroscience.math.differentialcalculus.autodiff;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.methods.Declarations;

/**
 * Strictly GC-free + inherently thread-safe Forward Mode Automatic Differentiation Evaluator.
 * 
 * <p>All heavy arrays are pre-allocated per-thread to the exact maximum size required by this evaluator.
 * No further allocations or resizes occur after the first evaluation on each thread.
 */
public class AutoDiffNEvaluator {

    private final Token[] rpnTokens;
    private final int maxOrder;
    private final int maxStackSize;

    // Thread-local storage for heavy mutable arrays (allocated once per thread to exact max size)
    private static final ThreadLocal<EvalState> THREAD_LOCAL_STATE = new ThreadLocal<>();

    public AutoDiffNEvaluator(Token[] rpnTokens) {
        this(rpnTokens, 1);
    }

    public AutoDiffNEvaluator(Token[] rpnTokens, int maxOrder) {
        if (rpnTokens == null || rpnTokens.length == 0) {
            throw new IllegalArgumentException("rpnTokens must not be null or empty");
        }
        if (maxOrder < 0) {
            throw new IllegalArgumentException("maxOrder >= 0 required");
        }
        this.rpnTokens = formatTokens(rpnTokens);
        this.maxOrder = maxOrder;
        this.maxStackSize = rpnTokens.length + 1; // fixed for this expression
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

    public void evaluateRPN(String wrtVar, double evalPoint, int order, double[] resultOut) {
        if (order > maxOrder) {
            throw new IllegalArgumentException("order > maxOrder");
        }
        if (resultOut == null || resultOut.length < order + 1) {
            throw new IllegalArgumentException("resultOut too small");
        }

        // Get or create thread-local state with exact pre-allocated sizes (GC-free after first use)
        EvalState state = THREAD_LOCAL_STATE.get();
        if (state == null || state.currentMaxStackSize < maxStackSize || state.currentMaxOrder < maxOrder) {
            state = new EvalState(maxStackSize, maxOrder);
            THREAD_LOCAL_STATE.set(state);
        }

        final double[][] valStack = state.valStack;
        final double[] scratch1 = state.scratch1;
        final double[] scratch2 = state.scratch2;
        final double[] scratch3 = state.scratch3;
        final double[] scratchArg = state.scratchArg;
        final double[] scratchU = state.scratchU;
        final double[] scratchV = state.scratchV;

        int sp = 0;
        for (Token t : rpnTokens) {
            if (t.kind == Token.NUMBER) {
                double[] jet = valStack[sp++];
                jet[0] = t.value;
                for (int k = 1; k <= order; k++) jet[k] = 0.0;
            } else if (t.kind == Token.VARIABLE) {
                double[] jet = valStack[sp++];
                boolean isWrt = t.name != null && t.name.equals(wrtVar);
                jet[0] = isWrt ? evalPoint : (t.v != null ? t.v.getValue() : t.value);
                jet[1] = isWrt ? 1.0 : 0.0;
                for (int k = 2; k <= order; k++) jet[k] = 0.0;
            } else if (t.kind == Token.OPERATOR) {
                sp = handleOperator(t, sp, order, valStack, scratch1, scratch2, scratch3, scratchU, scratchV);
            } else if (t.kind == Token.METHOD || t.kind == Token.FUNCTION) {
                if (t.arity == 1) {
                    sp = handleUnaryFunction(t, sp, order, valStack,
                            scratch1, scratch2, scratch3, scratchArg, scratchU, scratchV);
                } else if (t.arity == 2) {
                    sp = handleBinaryFunction(t, sp, order, valStack,
                            scratch1, scratch2, scratch3, scratchU, scratchV);
                } else {
                    throw new UnsupportedOperationException("Unsupported arity for " + t.name);
                }
            }
        }

        if (sp != 1) {
            throw new IllegalStateException("Malformed RPN: expected 1 result, got " + sp);
        }

        System.arraycopy(valStack[0], 0, resultOut, 0, order + 1);
        for (int k = 1; k <= order; k++) {
            resultOut[k] *= factorial(k);
        }
    }

    public double[] evaluateRPN(String wrtVar, double evalPoint, int order) {
        double[] out = new double[order + 1];
        evaluateRPN(wrtVar, evalPoint, order, out);
        return out;
    }

    // ===================================================================
    // Per-thread pre-allocated state (strictly GC-free after init)
    // ===================================================================
    private static final class EvalState {
        final double[][] valStack;
        final double[] scratch1, scratch2, scratch3, scratchArg, scratchU, scratchV;
        final int currentMaxStackSize;
        final int currentMaxOrder;

        EvalState(int stackSize, int order) {
            this.valStack = new double[stackSize][order + 1];
            this.scratch1 = new double[order + 1];
            this.scratch2 = new double[order + 1];
            this.scratch3 = new double[order + 1];
            this.scratchArg = new double[order + 1];
            this.scratchU = new double[order + 1];
            this.scratchV = new double[order + 1];
            this.currentMaxStackSize = stackSize;
            this.currentMaxOrder = order;
        }
    }

    // ===================================================================
    // Math helpers
    // ===================================================================
    private static long factorial(int n) {
        long f = 1;
        for (int i = 2; i <= n; i++) f *= i;
        return f;
    }

    private static void requireOperands(int sp, int needed, String name) {
        if (sp < needed) {
            throw new IllegalStateException(name + " requires " + needed + " operands");
        }
    }

    private void add(double[] a, double[] b, double[] out, int ord) {
        for (int k = 0; k <= ord; k++) out[k] = a[k] + b[k];
    }

    private void sub(double[] a, double[] b, double[] out, int ord) {
        for (int k = 0; k <= ord; k++) out[k] = a[k] - b[k];
    }

    private void mul(double[] a, double[] b, double[] out, int ord) {
        for (int k = 0; k <= ord; k++) {
            double sum = 0.0;
            for (int i = 0; i <= k; i++) {
                sum += a[i] * b[k - i];
            }
            out[k] = sum;
        }
    }

    private void recipJet(double[] b, double[] out, int ord) {
        out[0] = 1.0 / b[0];
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int i = 1; i <= k; i++) s += b[i] * out[k - i];
            out[k] = -out[0] * s;
        }
    }

    private void lnJet(double[] u, double[] out, int ord) {
        double u0 = u[0];
        if (u0 <= 0) throw new ArithmeticException("ln domain");
        out[0] = Math.log(u0);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int j = 1; j < k; j++) s += j * out[j] * u[k - j];
            out[k] = (u[k] - s / k) / u0;
        }
    }

    private void expJet(double[] w, double[] out, int ord) {
        out[0] = Math.exp(w[0]);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int j = 1; j <= k; j++) s += j * w[j] * out[k - j];
            out[k] = s / k;
        }
    }

    private void intPowJet(double[] u, int n, double[] out, int ord, double[] scratch1) {
        if (n == 0) {
            out[0] = 1.0;
            for (int k = 1; k <= ord; k++) out[k] = 0.0;
            return;
        }
        System.arraycopy(u, 0, out, 0, ord + 1);
        for (int p = 2; p <= n; p++) {
            mul(out, u, scratch1, ord);
            System.arraycopy(scratch1, 0, out, 0, ord + 1);
        }
    }

    private void powJet(double[] u, double[] v, double[] out, int ord,
                        double[] scratch1, double[] scratch2, double[] scratch3) {
        boolean vConstant = true;
        for (int k = 1; k <= ord; k++) {
            if (v[k] != 0.0) {
                vConstant = false;
                break;
            }
        }
        if (vConstant) {
            double r = v[0];
            if (!Double.isNaN(r) && !Double.isInfinite(r) && r == Math.floor(r)) {
                int n = (int) r;
                if (n >= 0) {
                    intPowJet(u, n, out, ord, scratch1);
                } else {
                    if (u[0] == 0.0) throw new ArithmeticException("pow domain: zero base with negative exponent");
                    intPowJet(u, -n, scratch3, ord, scratch1);
                    recipJet(scratch3, out, ord);
                }
                return;
            }
        }
        if (u[0] <= 0.0) throw new ArithmeticException("pow domain: non-real result");
        lnJet(u, scratch1, ord);
        mul(v, scratch1, scratch2, ord);
        expJet(scratch2, out, ord);
    }

    private void sinCosJet(double[] u, double[] sinOut, double[] cosOut, int ord) {
        sinOut[0] = Math.sin(u[0]);
        cosOut[0] = Math.cos(u[0]);
        for (int k = 1; k <= ord; k++) {
            double sSum = 0.0, cSum = 0.0;
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
            double sSum = 0.0, cSum = 0.0;
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
            double pk_1 = 0.0;
            for (int i = 0; i <= k - 1; i++) pk_1 += out[i] * out[k - 1 - i];
            pScratch[k - 1] = pk_1;
            double s = 0.0;
            for (int j = 1; j < k; j++) s += j * u[j] * pScratch[k - j];
            out[k] = u[k] * (1.0 + pScratch[0]) + s / k;
        }
    }

    private void tanhJet(double[] u, double[] out, int ord, double[] pScratch) {
        out[0] = Math.tanh(u[0]);
        pScratch[0] = out[0] * out[0];
        for (int k = 1; k <= ord; k++) {
            double pk_1 = 0.0;
            for (int i = 0; i <= k - 1; i++) pk_1 += out[i] * out[k - 1 - i];
            pScratch[k - 1] = pk_1;
            double s = 0.0;
            for (int j = 1; j < k; j++) s += j * u[j] * pScratch[k - j];
            out[k] = u[k] * (1.0 - pScratch[0]) - s / k;
        }
    }

    private void sqrtJet(double[] u, double[] out, int ord) {
        if (u[0] < 0) throw new ArithmeticException("sqrt domain");
        out[0] = Math.sqrt(u[0]);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int j = 1; j < k; j++) s += j * out[j] * out[k - j];
            out[k] = (u[k] - 2.0 * s / k) / (2.0 * out[0]);
        }
    }

    private void cbrtJet(double[] u, double[] out, int ord, double[] pScratch) {
        out[0] = Math.cbrt(u[0]);
        pScratch[0] = out[0] * out[0];
        for (int k = 1; k <= ord; k++) {
            double pk_1 = 0.0;
            for (int i = 0; i <= k - 1; i++) pk_1 += out[i] * out[k - 1 - i];
            pScratch[k - 1] = pk_1;
            double s = 0.0;
            for (int l = 1; l < k; l++) s += l * out[l] * pScratch[k - l];
            out[k] = (u[k] - 3.0 * s / k) / (3.0 * pScratch[0]);
        }
    }

    private void atanJet(double[] u, double[] out, int ord, double[] pScratch) {
        out[0] = Math.atan(u[0]);
        mul(u, u, pScratch, ord);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int l = 1; l < k; l++) s += l * out[l] * pScratch[k - l];
            out[k] = (u[k] - s / k) / (1.0 + pScratch[0]);
        }
    }

    // ===================================================================
    // Handlers
    // ===================================================================
    private int handleOperator(Token t, int sp, int ord, double[][] valStack,
                               double[] scratch1, double[] scratch2, double[] scratch3,
                               double[] scratchU, double[] scratchV) {
        if (t.arity == 2) {
            requireOperands(sp, 2, String.valueOf(t.opChar));
            double[] b = valStack[--sp];
            double[] a = valStack[--sp];
            double[] res = valStack[sp];

            System.arraycopy(a, 0, scratchU, 0, ord + 1);
            System.arraycopy(b, 0, scratchV, 0, ord + 1);

            switch (t.opChar) {
                case '+': add(scratchU, scratchV, res, ord); break;
                case '-': sub(scratchU, scratchV, res, ord); break;
                case '*': mul(scratchU, scratchV, res, ord); break;
                case '/':
                    if (Math.abs(scratchV[0]) < 1e-300) throw new ArithmeticException("Division by zero");
                    recipJet(scratchV, scratch1, ord);
                    mul(scratchU, scratch1, res, ord);
                    break;
                case '^':
                    powJet(scratchU, scratchV, res, ord, scratch1, scratch2, scratch3);
                    break;
                default:
                    throw new UnsupportedOperationException("Operator " + t.opChar);
            }
            return sp + 1;
        } else if (t.arity == 1 && t.opChar == '-') {
            requireOperands(sp, 1, "unary -");
            double[] arg = valStack[--sp];
            double[] res = valStack[sp];
            for (int k = 0; k <= ord; k++) res[k] = -arg[k];
            return sp + 1;
        }
        return sp;
    }

    private int handleUnaryFunction(Token t, int sp, int ord, double[][] valStack,
                                    double[] scratch1, double[] scratch2, double[] scratch3,
                                    double[] scratchArg, double[] scratchU, double[] scratchV) {
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
                if (Math.abs(scratchArg[0]) > 1.0) throw new ArithmeticException("asin domain");
                res[0] = Math.asin(scratchArg[0]);
                mul(scratchArg, scratchArg, scratch1, ord);
                scratch2[0] = 1.0 - scratch1[0];
                for (int k = 1; k <= ord; k++) scratch2[k] = -scratch1[k];
                sqrtJet(scratch2, scratch3, ord);
                for (int k = 1; k <= ord; k++) {
                    double s = 0.0;
                    for (int l = 1; l < k; l++) s += l * res[l] * scratch3[k - l];
                    res[k] = (scratchArg[k] - s / k) / scratch3[0];
                }
                break;
            case Declarations.ARC_COS:
            case Declarations.ARC_COS_ALT:
                if (Math.abs(scratchArg[0]) > 1.0) throw new ArithmeticException("acos domain");
                res[0] = Math.acos(scratchArg[0]);
                mul(scratchArg, scratchArg, scratch1, ord);
                scratch2[0] = 1.0 - scratch1[0];
                for (int k = 1; k <= ord; k++) scratch2[k] = -scratch1[k];
                sqrtJet(scratch2, scratch3, ord);
                for (int k = 1; k <= ord; k++) {
                    double s = 0.0;
                    for (int l = 1; l < k; l++) s += l * res[l] * scratch3[k - l];
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
                    double s = 0.0;
                    for (int l = 1; l < k; l++) s += l * res[l] * scratch3[k - l];
                    res[k] = (scratchArg[k] - s / k) / scratch3[0];
                }
                break;
            case Declarations.ARC_COSH:
            case Declarations.ARC_COSH_ALT:
                if (scratchArg[0] < 1.0) throw new ArithmeticException("acosh domain");
                res[0] = Math.log(scratchArg[0] + Math.sqrt(scratchArg[0] * scratchArg[0] - 1));
                mul(scratchArg, scratchArg, scratch1, ord);
                scratch2[0] = scratch1[0] - 1.0;
                for (int k = 1; k <= ord; k++) scratch2[k] = scratch1[k];
                sqrtJet(scratch2, scratch3, ord);
                for (int k = 1; k <= ord; k++) {
                    double s = 0.0;
                    for (int l = 1; l < k; l++) s += l * res[l] * scratch3[k - l];
                    res[k] = (scratchArg[k] - s / k) / scratch3[0];
                }
                break;
            case Declarations.ARC_TANH:
            case Declarations.ARC_TANH_ALT:
                if (Math.abs(scratchArg[0]) >= 1.0) throw new ArithmeticException("atanh domain");
                res[0] = 0.5 * Math.log((1 + scratchArg[0]) / (1 - scratchArg[0]));
                mul(scratchArg, scratchArg, scratch1, ord);
                for (int k = 1; k <= ord; k++) {
                    double s = 0.0;
                    for (int l = 1; l < k; l++) s += l * res[l] * scratch1[k - l];
                    res[k] = (scratchArg[k] + s / k) / (1.0 - scratch1[0]);
                }
                break;
            case Declarations.SQRT:
                sqrtJet(scratchArg, res, ord);
                break;
            case Declarations.CBRT:
                cbrtJet(scratchArg, res, ord, scratch1);
                break;
            case Declarations.EXP:
                expJet(scratchArg, res, ord);
                break;
            case Declarations.LN:
                lnJet(scratchArg, res, ord);
                break;
            case Declarations.LG:
                if (scratchArg[0] <= 0) throw new ArithmeticException("log domain");
                lnJet(scratchArg, res, ord);
                double ln10 = Math.log(10.0);
                for (int k = 0; k <= ord; k++) res[k] /= ln10;
                break;
            case "abs":
                res[0] = Math.abs(scratchArg[0]);
                for (int k = 1; k <= ord; k++) {
                    res[k] = (scratchArg[0] > 0) ? scratchArg[k] :
                             (scratchArg[0] < 0) ? -scratchArg[k] : 0.0;
                }
                break;
            default:
                throw new UnsupportedOperationException("Higher-order AD not implemented for: " + t.name);
        }
        return sp + 1;
    }

    private int handleBinaryFunction(Token t, int sp, int ord, double[][] valStack,
                                     double[] scratch1, double[] scratch2, double[] scratch3,
                                     double[] scratchU, double[] scratchV) {
        requireOperands(sp, 2, t.name);
        double[] v = valStack[--sp];
        double[] u = valStack[--sp];
        double[] res = valStack[sp];

        System.arraycopy(u, 0, scratchU, 0, ord + 1);
        System.arraycopy(v, 0, scratchV, 0, ord + 1);

        switch (t.name) {
            case Declarations.POW:
                powJet(scratchU, scratchV, res, ord, scratch1, scratch2, scratch3);
                break;
            case "atan2":
                res[0] = Math.atan2(scratchU[0], scratchV[0]);
                mul(scratchU, scratchU, scratch1, ord);
                mul(scratchV, scratchV, scratch2, ord);
                add(scratch1, scratch2, scratch1, ord);
                if (scratch1[0] < 1e-300) throw new ArithmeticException("atan2 at origin");
                for (int k = 1; k <= ord; k++) {
                    double rhs = 0.0;
                    for (int j = 1; j <= k; j++) rhs += j * scratchU[j] * scratchV[k - j];
                    for (int j = 0; j < k; j++) rhs -= (k - j) * scratchU[j] * scratchV[k - j];
                    double lhsSum = 0.0;
                    for (int l = 1; l < k; l++) lhsSum += l * res[l] * scratch1[k - l];
                    res[k] = (rhs - lhsSum) / (k * scratch1[0]);
                }
                break;
            default:
                throw new UnsupportedOperationException("2-arg not supported: " + t.name);
        }
        return sp + 1;
    }

    private static void test(String expr, String var, int order, double evalPoint) {
        MathExpression me = new MathExpression(expr);
        AutoDiffNEvaluator ad = new AutoDiffNEvaluator(me.getCachedPostfix(), 10);
        double[] out = ad.evaluateRPN(var, evalPoint, order);
        System.out.println(expr+": and derivatives up to order " + order + " at x = " + evalPoint);
        for (int i = 0; i < out.length; i++) {
            System.out.printf("Order %d: %.18f%n", i, out[i]);
        }
    }

    public static void main(String[] args) {
        test("x^3+3*x^2-5*x-8*atan2(2*x, 3)", "x", 1, 2);
        test("sin(x)", "x", 5, 2);
        test("sin(x)-cos(x)", "x", 5, 2);
        test("x^x", "x", 3, 2);
        test("x^2", "x", 2, 0);   // regression check: integer power at u0 == 0
        test("x^3", "x", 3, -1);  // regression check: integer power at negative base
    }
}