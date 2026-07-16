package com.github.gbenroscience.math.differentialcalculus.autodiff;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.methods.Declarations;

/**
 * Strictly GC-free + inherently thread-safe Forward Mode Automatic
 * Differentiation Evaluator, opcode-flattened.
 *
 * <h2>Opcode flattening</h2>
 * The per-token dispatch that used to walk an array of heap-allocated
 * {@link Token} objects and branch on {@code kind}/{@code name}/{@code opChar}
 * (a {@code switch} on {@code String} for every function call, every
 * evaluation, every token) is precomputed once at construction into a dense
 * {@code byte[] opcodes} array -- the thing actually walked once per token per
 * evaluation -- so the hot loop is a tight, cache-friendly, branch- predictable
 * {@code switch(byte)} over a primitive array instead of pointer-chasing
 * through objects and hashing strings. {@code Token[]} is still retained, but
 * touched only for the one thing that genuinely can't be precomputed into a
 * primitive: a non-differentiation-variable's dynamic binding
 * ({@code Token.v}), which can change value between calls.
 *
 * <h2>Aliasing-safe copy elision</h2> {@code res} for every operator is the
 * *same array object* as its first operand (a consequence of how the jet stack
 * pops-then-reuses a slot), so every helper that builds {@code res[k]} from a
 * convolution over lower indices of its own inputs (mul, div, pow, log, atan2)
 * still needs its operand defensively copied before {@code res} is overwritten
 * -- verified by tracing an actual corruption case in the repeated-squaring
 * loop of {@link #intPowJet} if that copy were skipped. Where {@code res[k]}
 * depends only on index-{@code k} reads that already happened (add, sub, unary
 * negate), no copy is needed at all, and none is done. Where only one of two
 * operands is ever written through {@code res}'s aliasing (true for every
 * binary op here, since {@code res} only ever aliases the *first* popped
 * operand), only that one operand is copied, not both.
 *
 * <p>
 * All heavy arrays are pre-allocated per-thread to the exact maximum size
 * required by this evaluator. No further allocations or resizes occur after the
 * first evaluation on each thread.
 */
public class AutoDiffNEvaluator implements Cloneable {

    @Override
    protected AutoDiffNEvaluator clone() throws CloneNotSupportedException {
        return new AutoDiffNEvaluator(formatTokens(this.rpnTokens), opcodes.clone(), constants.clone(), maxOrder, maxStackSize);
    }

    // ------------------------------------------------------------------
    // Opcodes
    // ------------------------------------------------------------------
    private static final int OP_NUMBER = 0;
    private static final int OP_VARIABLE = 1;
    private static final int OP_ADD = 2;
    private static final int OP_SUB = 3;
    private static final int OP_MUL = 4;
    private static final int OP_DIV = 5;
    private static final int OP_POW = 6;
    private static final int OP_NEG = 7;
    private static final int OP_SIN = 8;
    private static final int OP_COS = 9;
    private static final int OP_TAN = 10;
    private static final int OP_ASIN = 11;
    private static final int OP_ACOS = 12;
    private static final int OP_ATAN = 13;
    private static final int OP_SEC = 14;
    private static final int OP_COSEC = 15;
    private static final int OP_COT = 16;
    private static final int OP_SINH = 17;
    private static final int OP_COSH = 18;
    private static final int OP_TANH = 19;
    private static final int OP_ASINH = 20;
    private static final int OP_ACOSH = 21;
    private static final int OP_ATANH = 22;
    private static final int OP_SQRT = 23;
    private static final int OP_CBRT = 24;
    private static final int OP_EXP = 25;
    private static final int OP_LN = 26;
    private static final int OP_LG = 27;
    private static final int OP_ABS = 28;
    private static final int OP_ATAN2 = 29;
    private static final int OP_LOG_BASE = 30;

    private final Token[] rpnTokens;   // retained only for Token.v / Token.name lookups -- see class javadoc
    private final byte[] opcodes;      // the dense instruction stream actually walked every evaluation
    private final double[] constants;  // NUMBER values, and non-wrt VARIABLE fallback values
    private final int maxOrder;
    private final int maxStackSize;

    // Thread-local storage for heavy mutable arrays (allocated once per thread to exact max size).
    // Static and therefore potentially shared across different AutoDiffN instances on the
    // same thread -- see EvalState.cachedForInstance for why that matters for the wrtVar-slot cache.
    private static final ThreadLocal<EvalState> THREAD_LOCAL_STATE = new ThreadLocal<>();

    public AutoDiffNEvaluator(MathExpression me) {
        this(me, 1);
    }

    private AutoDiffNEvaluator(Token[] rpnTokens, byte[] opcodes, double[] constants, int maxOrder, int maxStackSize) {
        this.rpnTokens = rpnTokens;
        this.opcodes = opcodes;
        this.constants = constants;
        this.maxOrder = maxOrder;
        this.maxStackSize = maxStackSize;
    }

    public AutoDiffNEvaluator(MathExpression me, int maxOrder) {
        if (me == null || me.getCachedPostfix() == null || me.getCachedPostfix().length == 0) {
            throw new IllegalArgumentException("rpnTokens must not be null or empty");
        }
        if (maxOrder < 0) {
            throw new IllegalArgumentException("maxOrder >= 0 required");
        }
        this.rpnTokens = formatTokens(me.getCachedPostfix());
        this.maxOrder = maxOrder;
        this.maxStackSize = rpnTokens.length + 1;

        int n = rpnTokens.length;
        this.opcodes = new byte[n];
        this.constants = new double[n];
        for (int i = 0; i < n; i++) {
            Token t = rpnTokens[i];
            opcodes[i] = (byte) opcodeFor(t);
            constants[i] = t.value;
        }
    }

    public int getMaxOrder() {
        return maxOrder;
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
     * One-time (construction only) translation of a Token's kind/name/opChar
     * into a flat opcode.
     */
    private static int opcodeFor(Token t) {
        if (t.kind == Token.NUMBER) {
            return OP_NUMBER;
        }
        if (t.kind == Token.VARIABLE) {
            return OP_VARIABLE;
        }
        if (t.kind == Token.OPERATOR) {
            if (t.arity == 2) {
                switch (t.opChar) {
                    case '+':
                        return OP_ADD;
                    case '-':
                        return OP_SUB;
                    case '*':
                        return OP_MUL;
                    case '/':
                        return OP_DIV;
                    case '^':
                        return OP_POW;
                    default:
                        throw new UnsupportedOperationException("Operator " + t.opChar);
                }
            } else if (t.arity == 1 && t.opChar == '-') {
                return OP_NEG;
            }
            throw new UnsupportedOperationException("Unary operator not supported: " + t.opChar);
        }
        if (t.kind == Token.METHOD || t.kind == Token.FUNCTION) {
            if (t.arity == 1) {
                switch (t.name) {
                    case Declarations.SIN:
                        return OP_SIN;
                    case Declarations.COS:
                        return OP_COS;
                    case Declarations.TAN:
                        return OP_TAN;
                    case Declarations.ARC_SIN:
                    case Declarations.ARC_SIN_ALT:
                        return OP_ASIN;
                    case Declarations.ARC_COS:
                    case Declarations.ARC_COS_ALT:
                        return OP_ACOS;
                    case Declarations.ARC_TAN:
                    case Declarations.ARC_TAN_ALT:
                        return OP_ATAN;
                    case Declarations.SEC:
                        return OP_SEC;
                    case Declarations.COSEC:
                        return OP_COSEC;
                    case Declarations.COT:
                        return OP_COT;
                    case Declarations.SINH:
                        return OP_SINH;
                    case Declarations.COSH:
                        return OP_COSH;
                    case Declarations.TANH:
                        return OP_TANH;
                    case Declarations.ARC_SINH:
                    case Declarations.ARC_SINH_ALT:
                        return OP_ASINH;
                    case Declarations.ARC_COSH:
                    case Declarations.ARC_COSH_ALT:
                        return OP_ACOSH;
                    case Declarations.ARC_TANH:
                    case Declarations.ARC_TANH_ALT:
                        return OP_ATANH;
                    case Declarations.SQRT:
                        return OP_SQRT;
                    case Declarations.CBRT:
                        return OP_CBRT;
                    case Declarations.EXP:
                        return OP_EXP;
                    case Declarations.LN:
                        return OP_LN;
                    case Declarations.LG:
                        return OP_LG;
                    case "abs":
                        return OP_ABS;
                    default:
                        throw new UnsupportedOperationException("Higher-order AD not implemented for: " + t.name);
                }
            } else if (t.arity == 2) {
                switch (t.name) {
                    case Declarations.POW:
                        return OP_POW; // same math as the '^' operator
                    case Declarations.LOG:
                        return OP_LOG_BASE;
                    case Declarations.ATAN2:
                        return OP_ATAN2;
                    default:
                        throw new UnsupportedOperationException("2-arg not supported: " + t.name);
                }
            }
            throw new UnsupportedOperationException("Unsupported arity for " + t.name);
        }
        throw new UnsupportedOperationException("Unrecognized token kind: " + t.kind);
    }

    /**
     * Fills {@code resultOut[0..order]} with the ordinary derivatives
     * {@code f(x), f'(x), ..., f^(order)(x)}.
     */
    public void evaluateRPN(String wrtVar, double evalPoint, int order, double[] resultOut) {
        if (order > maxOrder) {
            throw new IllegalArgumentException("order > maxOrder");
        }
        if (resultOut == null || resultOut.length < order + 1) {
            throw new IllegalArgumentException("resultOut too small");
        }

        EvalState state = computeJet(wrtVar, evalPoint, order);
        System.arraycopy(state.valStack[0], 0, resultOut, 0, order + 1);
        for (int k = 1; k <= order; k++) {
            resultOut[k] *= factorial(k);
        }
    }

    public double[] evaluateRPN(String wrtVar, double evalPoint, int order) {
        double[] out = new double[order + 1];
        evaluateRPN(wrtVar, evalPoint, order, out);
        return out;
    }

    /**
     * Fills {@code resultOut[0..order]} with the normalized Taylor coefficients
     * {@code c_k = f^(k)(evalPoint)/k!}.
     */
    public void taylorCoefficients(String wrtVar, double evalPoint, int order, double[] resultOut) {
        if (order > maxOrder) {
            throw new IllegalArgumentException("order > maxOrder");
        }
        if (resultOut == null || resultOut.length < order + 1) {
            throw new IllegalArgumentException("resultOut too small");
        }

        EvalState state = computeJet(wrtVar, evalPoint, order);
        System.arraycopy(state.valStack[0], 0, resultOut, 0, order + 1);
    }

    public double[] taylorCoefficients(String wrtVar, double evalPoint, int order) {
        double[] out = new double[order + 1];
        taylorCoefficients(wrtVar, evalPoint, order, out);
        return out;
    }

    /**
     * Shared opcode walk: gets/creates this thread's {@link EvalState},
     * refreshes its wrt-variable-slot cache only if it's stale for this
     * instance/wrtVar combination (see {@link EvalState#cachedForInstance}),
     * drives the jet stack to a single result slot holding the normalized
     * Taylor coefficients, and returns that state.
     */
    private EvalState computeJet(String wrtVar, double evalPoint, int order) {
        EvalState state = THREAD_LOCAL_STATE.get();
        if (state == null || state.currentMaxStackSize < maxStackSize || state.currentMaxOrder < maxOrder) {
            state = new EvalState(maxStackSize, maxOrder);
            THREAD_LOCAL_STATE.set(state);
        }

        // Fast path: same instance and the exact same wrtVar String object as last time (true on
        // essentially every call in a tight evaluation loop, since callers typically pass the same
        // interned/reused String) -- a single reference comparison, no String.equals at all.
        boolean stale = state.cachedForInstance != this
                || (state.cachedWrtVar != wrtVar
                && (state.cachedWrtVar == null || !state.cachedWrtVar.equals(wrtVar)));
        if (stale) {
            refreshWrtSlots(state, wrtVar);
        }

        final double[][] valStack = state.valStack;
        final double[] scratch1 = state.scratch1;
        final double[] scratch2 = state.scratch2;
        final double[] scratch3 = state.scratch3;
        final double[] scratchArg = state.scratchArg;
        final double[] scratchU = state.scratchU;
        final double[] scratchV = state.scratchV;
        final boolean[] isWrtSlot = state.isWrtSlot;

        int sp = 0;
        final int n = opcodes.length;
        for (int i = 0; i < n; i++) {
            switch (opcodes[i]) {
                case OP_NUMBER: {
                    double[] jet = valStack[sp++];
                    jet[0] = constants[i];
                    for (int k = 1; k <= order; k++) {
                        jet[k] = 0.0;
                    }
                    break;
                }
                case OP_VARIABLE: {
                    double[] jet = valStack[sp++];
                    if (isWrtSlot[i]) {
                        jet[0] = evalPoint;
                        jet[1] = 1.0;
                    } else {
                        Token tok = rpnTokens[i];
                        jet[0] = tok.v != null ? tok.v.getValue() : constants[i];
                        jet[1] = 0.0;
                    }
                    for (int k = 2; k <= order; k++) {
                        jet[k] = 0.0;
                    }
                    break;
                }
                case OP_ADD: {
                    double[] b = valStack[--sp];
                    double[] a = valStack[--sp];
                    double[] res = valStack[sp];
                    for (int k = 0; k <= order; k++) {
                        res[k] = a[k] + b[k];
                    }
                    sp++;
                    break;
                }
                case OP_SUB: {
                    double[] b = valStack[--sp];
                    double[] a = valStack[--sp];
                    double[] res = valStack[sp];
                    for (int k = 0; k <= order; k++) {
                        res[k] = a[k] - b[k];
                    }
                    sp++;
                    break;
                }
                case OP_MUL: {
                    double[] b = valStack[--sp];
                    double[] a = valStack[--sp];
                    double[] res = valStack[sp];
                    // res aliases a's memory; b is never written to, so only a needs protecting.
                    System.arraycopy(a, 0, scratchU, 0, order + 1);
                    mul(scratchU, b, res, order);
                    sp++;
                    break;
                }
                case OP_DIV: {
                    double[] b = valStack[--sp];
                    double[] a = valStack[--sp];
                    double[] res = valStack[sp];
                    if (Math.abs(b[0]) < 1e-300) {
                        throw new ArithmeticException("Division by zero");
                    }
                    System.arraycopy(a, 0, scratchU, 0, order + 1);
                    recipJet(b, scratch1, order);
                    mul(scratchU, scratch1, res, order);
                    sp++;
                    break;
                }
                case OP_POW: {
                    double[] b = valStack[--sp];
                    double[] a = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(a, 0, scratchU, 0, order + 1);
                    powJet(scratchU, b, res, order, scratch1, scratch2, scratch3);
                    sp++;
                    break;
                }
                case OP_NEG: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    for (int k = 0; k <= order; k++) {
                        res[k] = -arg[k];
                    }
                    sp++;
                    break;
                }
                case OP_SIN: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    sinCosJet(scratchArg, res, scratch1, order);
                    sp++;
                    break;
                }
                case OP_COS: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    sinCosJet(scratchArg, scratch1, res, order);
                    sp++;
                    break;
                }
                case OP_TAN: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    tanJet(scratchArg, res, order, scratch1);
                    sp++;
                    break;
                }
                case OP_ASIN: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    if (Math.abs(scratchArg[0]) > 1.0) {
                        throw new ArithmeticException("asin domain");
                    }
                    res[0] = Math.asin(scratchArg[0]);
                    mul(scratchArg, scratchArg, scratch1, order);
                    scratch2[0] = 1.0 - scratch1[0];
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = -scratch1[k];
                    }
                    sqrtJet(scratch2, scratch3, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * res[l] * scratch3[k - l];
                        }
                        res[k] = (scratchArg[k] - s / k) / scratch3[0];
                    }
                    sp++;
                    break;
                }
                case OP_ACOS: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    if (Math.abs(scratchArg[0]) > 1.0) {
                        throw new ArithmeticException("acos domain");
                    }
                    res[0] = Math.acos(scratchArg[0]);
                    mul(scratchArg, scratchArg, scratch1, order);
                    scratch2[0] = 1.0 - scratch1[0];
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = -scratch1[k];
                    }
                    sqrtJet(scratch2, scratch3, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * res[l] * scratch3[k - l];
                        }
                        res[k] = (-scratchArg[k] - s / k) / scratch3[0];
                    }
                    sp++;
                    break;
                }
                case OP_ATAN: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    atanJet(scratchArg, res, order, scratch1);
                    sp++;
                    break;
                }
                case OP_SEC: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    sinCosJet(scratchArg, scratch1, scratch2, order);
                    recipJet(scratch2, res, order);
                    sp++;
                    break;
                }
                case OP_COSEC: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    sinCosJet(scratchArg, scratch1, scratch2, order);
                    recipJet(scratch1, res, order);
                    sp++;
                    break;
                }
                case OP_COT: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    tanJet(scratchArg, scratch1, order, scratch2);
                    recipJet(scratch1, res, order);
                    sp++;
                    break;
                }
                case OP_SINH: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    sinhCoshJet(scratchArg, res, scratch1, order);
                    sp++;
                    break;
                }
                case OP_COSH: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    sinhCoshJet(scratchArg, scratch1, res, order);
                    sp++;
                    break;
                }
                case OP_TANH: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    tanhJet(scratchArg, res, order, scratch1);
                    sp++;
                    break;
                }
                case OP_ASINH: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    res[0] = Math.log(scratchArg[0] + Math.sqrt(scratchArg[0] * scratchArg[0] + 1));
                    mul(scratchArg, scratchArg, scratch1, order);
                    scratch2[0] = scratch1[0] + 1.0;
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = scratch1[k];
                    }
                    sqrtJet(scratch2, scratch3, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * res[l] * scratch3[k - l];
                        }
                        res[k] = (scratchArg[k] - s / k) / scratch3[0];
                    }
                    sp++;
                    break;
                }
                case OP_ACOSH: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    if (scratchArg[0] < 1.0) {
                        throw new ArithmeticException("acosh domain");
                    }
                    res[0] = Math.log(scratchArg[0] + Math.sqrt(scratchArg[0] * scratchArg[0] - 1));
                    mul(scratchArg, scratchArg, scratch1, order);
                    scratch2[0] = scratch1[0] - 1.0;
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = scratch1[k];
                    }
                    sqrtJet(scratch2, scratch3, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * res[l] * scratch3[k - l];
                        }
                        res[k] = (scratchArg[k] - s / k) / scratch3[0];
                    }
                    sp++;
                    break;
                }
                case OP_ATANH: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    if (Math.abs(scratchArg[0]) >= 1.0) {
                        throw new ArithmeticException("atanh domain");
                    }
                    res[0] = 0.5 * Math.log((1 + scratchArg[0]) / (1 - scratchArg[0]));
                    mul(scratchArg, scratchArg, scratch1, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * res[l] * scratch1[k - l];
                        }
                        res[k] = (scratchArg[k] + s / k) / (1.0 - scratch1[0]);
                    }
                    sp++;
                    break;
                }
                case OP_SQRT: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    sqrtJet(scratchArg, res, order);
                    sp++;
                    break;
                }
                case OP_CBRT: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    cbrtJet(scratchArg, res, order, scratch1);
                    sp++;
                    break;
                }
                case OP_EXP: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    expJet(scratchArg, res, order);
                    sp++;
                    break;
                }
                case OP_LN: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    lnJet(scratchArg, res, order);
                    sp++;
                    break;
                }
                case OP_LG: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    if (scratchArg[0] <= 0) {
                        throw new ArithmeticException("log domain");
                    }
                    lnJet(scratchArg, res, order);
                    double ln10 = Math.log(10.0);
                    for (int k = 0; k <= order; k++) {
                        res[k] /= ln10;
                    }
                    sp++;
                    break;
                }
                case OP_ABS: {
                    double[] arg = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(arg, 0, scratchArg, 0, order + 1);
                    res[0] = Math.abs(scratchArg[0]);
                    for (int k = 1; k <= order; k++) {
                        res[k] = (scratchArg[0] > 0) ? scratchArg[k]
                                : (scratchArg[0] < 0) ? -scratchArg[k] : 0.0;
                    }
                    sp++;
                    break;
                }
                case OP_ATAN2: {
                    double[] v = valStack[--sp];
                    double[] u = valStack[--sp];
                    double[] res = valStack[sp];
                    System.arraycopy(u, 0, scratchU, 0, order + 1);
                    System.arraycopy(v, 0, scratchV, 0, order + 1);
                    res[0] = Math.atan2(scratchU[0], scratchV[0]);
                    mul(scratchU, scratchU, scratch1, order);
                    mul(scratchV, scratchV, scratch2, order);
                    add(scratch1, scratch2, scratch1, order);
                    if (scratch1[0] < 1e-300) {
                        throw new ArithmeticException("atan2 at origin");
                    }
                    for (int k = 1; k <= order; k++) {
                        double rhs = 0.0;
                        for (int j = 1; j <= k; j++) {
                            rhs += j * scratchU[j] * scratchV[k - j];
                        }
                        for (int j = 0; j < k; j++) {
                            rhs -= (k - j) * scratchU[j] * scratchV[k - j];
                        }
                        double lhsSum = 0.0;
                        for (int l = 1; l < k; l++) {
                            lhsSum += l * res[l] * scratch1[k - l];
                        }
                        res[k] = (rhs - lhsSum) / (k * scratch1[0]);
                    }
                    sp++;
                    break;
                }
                case OP_LOG_BASE: {
                    // log_v(u) = ln(u)/ln(v). Newly implemented -- the original fell through
                    // this case doing nothing, silently corrupting the evaluation stack.
                    double[] v = valStack[--sp];
                    double[] u = valStack[--sp];
                    double[] res = valStack[sp];
                    if (u[0] <= 0.0) {
                        throw new ArithmeticException("log domain");
                    }
                    if (v[0] <= 0.0 || v[0] == 1.0) {
                        throw new ArithmeticException("log base domain");
                    }
                    System.arraycopy(u, 0, scratchU, 0, order + 1);
                    System.arraycopy(v, 0, scratchV, 0, order + 1);
                    lnJet(scratchU, scratch1, order);
                    lnJet(scratchV, scratch2, order);
                    recipJet(scratch2, scratch3, order);
                    mul(scratch1, scratch3, res, order);
                    sp++;
                    break;
                }
                default:
                    throw new UnsupportedOperationException("Unrecognized opcode: " + opcodes[i]);
            }
        }

        if (sp != 1) {
            throw new IllegalStateException("Malformed RPN: expected 1 result, got " + sp);
        }

        return state;
    }

    /**
     * Rebuilds {@code state.isWrtSlot} for this instance/wrtVar combination.
     * {@code THREAD_LOCAL_STATE} is static, so a thread's cached state can be
     * reused across different {@link AutoDiffNEvaluator} instances -- without
     * the instance-identity check this would silently serve one expression's
     * variable-slot map to a different expression's opcode layout.
     */
    private void refreshWrtSlots(EvalState state, String wrtVar) {
        if (state.isWrtSlot == null || state.isWrtSlot.length < opcodes.length) {
            state.isWrtSlot = new boolean[opcodes.length];
        }
        for (int i = 0; i < opcodes.length; i++) {
            if (opcodes[i] == OP_VARIABLE) {
                Token tok = rpnTokens[i];
                state.isWrtSlot[i] = tok.name != null && tok.name.equals(wrtVar);
            }
        }
        state.cachedForInstance = this;
        state.cachedWrtVar = wrtVar;
    }

    // ===================================================================
    // Per-thread pre-allocated state (strictly GC-free after init)
    // ===================================================================
    private static final class EvalState {

        final double[][] valStack;
        final double[] scratch1, scratch2, scratch3, scratchArg, scratchU, scratchV;
        final int currentMaxStackSize;
        final int currentMaxOrder;

        Object cachedForInstance;
        String cachedWrtVar;
        boolean[] isWrtSlot;

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
    // Math helpers (unchanged from the pre-flattening version -- these are
    // the actual math, already exercised, and not touched by this rewrite)
    // ===================================================================
    private static long factorial(int n) {
        long f = 1;
        for (int i = 2; i <= n; i++) {
            f *= i;
        }
        return f;
    }

    private void add(double[] a, double[] b, double[] out, int ord) {
        for (int k = 0; k <= ord; k++) {
            out[k] = a[k] + b[k];
        }
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
            for (int i = 1; i <= k; i++) {
                s += b[i] * out[k - i];
            }
            out[k] = -out[0] * s;
        }
    }

    private void lnJet(double[] u, double[] out, int ord) {
        double u0 = u[0];
        if (u0 <= 0) {
            throw new ArithmeticException("ln domain");
        }
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

    private void intPowJet(double[] u, int n, double[] out, int ord, double[] scratch1) {
        if (n == 0) {
            out[0] = 1.0;
            for (int k = 1; k <= ord; k++) {
                out[k] = 0.0;
            }
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
                    if (u[0] == 0.0) {
                        throw new ArithmeticException("pow domain: zero base with negative exponent");
                    }
                    intPowJet(u, -n, scratch3, ord, scratch1);
                    recipJet(scratch3, out, ord);
                }
                return;
            }
        }
        if (u[0] <= 0.0) {
            throw new ArithmeticException("pow domain: non-real result");
        }
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
            for (int i = 0; i <= k - 1; i++) {
                pk_1 += out[i] * out[k - 1 - i];
            }
            pScratch[k - 1] = pk_1;
            double s = 0.0;
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
            double pk_1 = 0.0;
            for (int i = 0; i <= k - 1; i++) {
                pk_1 += out[i] * out[k - 1 - i];
            }
            pScratch[k - 1] = pk_1;
            double s = 0.0;
            for (int j = 1; j < k; j++) {
                s += j * u[j] * pScratch[k - j];
            }
            out[k] = u[k] * (1.0 - pScratch[0]) - s / k;
        }
    }

    private void sqrtJet(double[] u, double[] out, int ord) {
        if (u[0] < 0) {
            throw new ArithmeticException("sqrt domain");
        }
        out[0] = Math.sqrt(u[0]);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
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
            double pk_1 = 0.0;
            for (int i = 0; i <= k - 1; i++) {
                pk_1 += out[i] * out[k - 1 - i];
            }
            pScratch[k - 1] = pk_1;
            double s = 0.0;
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
            double s = 0.0;
            for (int l = 1; l < k; l++) {
                s += l * out[l] * pScratch[k - l];
            }
            out[k] = (u[k] - s / k) / (1.0 + pScratch[0]);
        }
    }

    // ===================================================================
    // Self-tests / benchmark
    // ===================================================================
    private static void test(String expr, String var, int order, double evalPoint) {
        MathExpression me = new MathExpression(expr);
        AutoDiffNEvaluator ad = new AutoDiffNEvaluator(me, 10);
        double[] out = ad.evaluateRPN(var, evalPoint, order);
        System.out.println(expr + ": derivatives up to order " + order + " at x = " + evalPoint);
        for (int i = 0; i < out.length; i++) {
            System.out.printf("Order %d: %.18f%n", i, out[i]);
        }
    }

    public static void main(String[] args) {
        test("x^3+3*x^2-5*x-8*atan2(2*x, 3)", "x", 1, 2);
        test("sin(x)", "x", 5, 2);
        test("sin(x)-cos(x)", "x", 5, 2);
        test("x^x", "x", 3, 2);
        test("x^2", "x", 2, 0);
        test("x^3", "x", 3, -1);
        test("log(x,2)", "x", 3, 8); // exercises the newly-implemented binary log

        String expr = "x^2*cos(x)-2*x*sin(x)-2*cos(x)";
        double evalPoint = 4;
        int orderOfDiff = 18;
        double[] resultOut = new double[orderOfDiff + 1];
        AutoDiffNEvaluator adne = new AutoDiffNEvaluator(new MathExpression(expr), 20);
        double N = 10_000_000;
        double t = System.nanoTime();
        for (int i = 0; i < N; i++) {
            adne.evaluateRPN("x", evalPoint, orderOfDiff, resultOut);
        }
        t = (System.nanoTime() - t) / N;
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < resultOut.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(resultOut[i]);
        }
        sb.append("]");
        System.out.println("res = " + sb + ", \n" + t + "ns");
    }
}
