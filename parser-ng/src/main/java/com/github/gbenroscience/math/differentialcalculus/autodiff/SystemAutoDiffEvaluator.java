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
package com.github.gbenroscience.math.differentialcalculus.autodiff;

import com.github.gbenroscience.math.differentialcalculus.Derivative;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.methods.Declarations;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author GBEMIRO Strictly GC-free (per-call), inherently thread-safe Forward
 * Mode Automatic Differentiation Evaluator.
 *
 * <h2>Frame-based variable resolution</h2>
 * Every VARIABLE token in the compiled RPN carries a {@code frameIndex} — the
 * same execution-frame slot convention used throughout
 * {@code DifferentialEquations}/{@code VectorODE} (tSlot, ySlotStart, etc.).
 * This evaluator reads every variable's value directly from a caller-supplied
 * {@code double[] frame} at that token's frameIndex — the exact same array a
 * compiled {@code dy_dt} MethodHandle would receive. There is no dependency on
 * a shared, mutable {@code Variable} binding: two threads (or two Jacobian
 * columns on the same thread) evaluating with different frames never interfere
 * with each other, and the result is fully determined by the frame passed in.
 *
 * <h2>Architectural notes carried over</h2>
 * <ul>
 * <li><b>1D Flattened Matrix Stack:</b> The 2D valStack is replaced by a single
 * contiguous {@code double[] flatStack} to maximize CPU L1 cache prefetching
 * and eliminate JVM pointer-chasing during array access.</li>
 * <li><b>Static Method Combinators:</b> All mathematical operations are
 * decoupled into pure static methods operating on array offsets, making them
 * direct targets for {@code MethodHandle} linking and JIT inlining.</li>
 * </ul>
 */
public class SystemAutoDiffEvaluator implements Cloneable {

    @Override
    protected SystemAutoDiffEvaluator clone() throws CloneNotSupportedException {
        return new SystemAutoDiffEvaluator(formatTokens(this.rpnTokens), opcodes.clone(), constants.clone(), maxOrder, maxStackSize);
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
    private static final int OP_SECH = 20;
    private static final int OP_COSECH = 21;
    private static final int OP_COTH = 22;
    private static final int OP_ASINH = 23;
    private static final int OP_ACOSH = 24;
    private static final int OP_ATANH = 25;
    private static final int OP_ASECH = 26;
    private static final int OP_ACOSECH = 27;
    private static final int OP_ACOTH = 28;
    private static final int OP_SQRT = 29;
    private static final int OP_CBRT = 30;
    private static final int OP_EXP = 31;
    private static final int OP_LN = 32;
    private static final int OP_LG = 33;
    private static final int OP_ABS = 34;
    private static final int OP_ATAN2 = 35;
    private static final int OP_LOG_BASE = 36;
    private static final int OP_FLOOR = 37;
    private static final int OP_CEIL = 38;

    /**
     * Sentinel wrtFrameIndex meaning "no variable matches" — every VARIABLE
     * token is treated as a constant w.r.t. the differentiation, since real
     * frame indices are always >= 0. Used for order-0-only (plain value)
     * evaluation and for differentiating w.r.t. a variable name absent from the
     * expression (whose derivative is, correctly, identically zero).
     */
    public static final int NO_WRT_VARIABLE = -1;

    private final Token[] rpnTokens;
    private final byte[] opcodes;
    private final double[] constants;
    private final int maxOrder;
    private final int stride;
    private final int maxStackSize;

    /**
     * name -> frameIndex for every distinct VARIABLE token in this expression,
     * built once at construction. Enables the String-keyed convenience
     * overloads without re-scanning the token tape on every call.
     */
    private final Map<String, Integer> nameToFrameIndex;

    public final MathExpression targetExpr;

    private static final ThreadLocal<FlatEvalState> THREAD_LOCAL_STATE = new ThreadLocal<>();

    public SystemAutoDiffEvaluator(MathExpression me) {
        this(me, Derivative.MAX_ORDER);
    }

    private SystemAutoDiffEvaluator(Token[] rpnTokens, byte[] opcodes, double[] constants, int maxOrder, int maxStackSize) {
        this.targetExpr = null;
        this.rpnTokens = rpnTokens;
        this.opcodes = opcodes;
        this.constants = constants;
        this.maxOrder = maxOrder;
        this.stride = maxOrder + 1;
        this.maxStackSize = maxStackSize;
        this.nameToFrameIndex = buildNameToFrameIndex(rpnTokens, opcodes);
    }

    public SystemAutoDiffEvaluator(MathExpression me, int maxOrder) {
        if (me == null || me.getCachedPostfix() == null || me.getCachedPostfix().length == 0) {
            throw new IllegalArgumentException("rpnTokens must not be null or empty");
        }
        if (maxOrder < 0) {
            throw new IllegalArgumentException("maxOrder >= 0 required");
        }
        this.targetExpr = me;
        this.rpnTokens = formatTokens(me.getCachedPostfix());
        this.maxOrder = maxOrder;
        this.stride = maxOrder + 1;
        this.maxStackSize = rpnTokens.length + 1;

        int n = rpnTokens.length;
        this.opcodes = new byte[n];
        this.constants = new double[n];
        for (int i = 0; i < n; i++) {
            Token t = rpnTokens[i];
            opcodes[i] = (byte) opcodeFor(t);
            constants[i] = t.value;
        }
        this.nameToFrameIndex = buildNameToFrameIndex(this.rpnTokens, this.opcodes);
    }

    private static Map<String, Integer> buildNameToFrameIndex(Token[] rpnTokens, byte[] opcodes) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < opcodes.length; i++) {
            if (opcodes[i] == OP_VARIABLE) {
                Token tok = rpnTokens[i];
                if (tok.name != null) {
                    map.putIfAbsent(tok.name, tok.frameIndex);
                }
            }
        }
        return map;
    }

    public int getMaxOrder() {
        return maxOrder;
    }

    /**
     * Frame index of the given variable name in this expression, or
     * NO_WRT_VARIABLE if the name does not appear in the expression at all
     * (whose derivative w.r.t. it is correctly zero everywhere).
     * @param varName
     * @return 
     */
    public int frameIndexOf(String varName) {
        Integer fi = nameToFrameIndex.get(varName);
        return fi == null ? NO_WRT_VARIABLE : fi;
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
                    case Declarations.SECH:
                        return OP_SECH;
                    case Declarations.COSECH:
                        return OP_COSECH;
                    case Declarations.COTH:
                        return OP_COTH;
                    case Declarations.ARC_SECH:
                        return OP_ASECH;
                    case Declarations.ARC_COSECH:
                        return OP_ACOSECH;
                    case Declarations.ARC_COTH:
                        return OP_ACOTH;
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
                    case Declarations.CEIL:
                        return OP_CEIL;
                    case Declarations.FLOOR:
                        return OP_FLOOR;
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
                        return OP_POW;
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

    // ------------------------------------------------------------------
    // Public API — frame-based (primary; matches solver's vars[] convention)
    // ------------------------------------------------------------------
    /**
     * Raw nth-order derivatives d^k f / dx_wrt^k at the point described by
     * frame, for k = 0..order. resultOut[k] is the true kth derivative (not the
     * Taylor coefficient — it has been multiplied by k!).
     *
     * @param frame execution frame holding every variable's current value,
     * indexed by Token.frameIndex — the same array layout a compiled dy_dt
     * MethodHandle uses
     * @param wrtFrameIndex frame index of the variable to differentiate
     * against; use NO_WRT_VARIABLE for "treat every variable as constant" (only
     * order 0 is meaningful in that case)
     * @param order
     * @param resultOut
     */
    public void evaluateDerivatives(double[] frame, int wrtFrameIndex, int order, double[] resultOut) {
        if (order > maxOrder) {
            throw new IllegalArgumentException("order > maxOrder");
        }
        if (resultOut == null || resultOut.length < order + 1) {
            throw new IllegalArgumentException("resultOut too small");
        }

        FlatEvalState state = computeJet(frame, wrtFrameIndex, order);
        System.arraycopy(state.flatStack, 0, resultOut, 0, order + 1);
        for (int k = 1; k <= order; k++) {
            resultOut[k] *= factorial(k);
        }
    }

    /**
     *
     * @param frame execution frame holding every variable's current value,
     * indexed by Token.frameIndex — the same array layout a compiled dy_dt
     * MethodHandle uses
     * @param wrtFrameIndex frame index of the variable to differentiate
     * against; use NO_WRT_VARIABLE for "treat every variable as constant" (only
     * order 0 is meaningful in that case)
     * @param order
     * @return
     */
    public double[] evaluateDerivatives(double[] frame, int wrtFrameIndex, int order) {
        double[] out = new double[order + 1];
        evaluateDerivatives(frame, wrtFrameIndex, order, out);
        return out;
    }

    /**
     * Taylor coefficients [f, f', f''/2!, ..., f^(order)/order!] at the point
     * described by frame — the natural form for Taylor-series integration and
     * for reading a single derivative (resultOut[1] === df/dx_wrt) without
     * paying the factorial multiply.
     *
     * @param frame execution frame holding every variable's current value,
     * indexed by Token.frameIndex — the same array layout a compiled dy_dt
     * MethodHandle uses
     * @param wrtFrameIndex frame index of the variable to differentiate
     * against; use NO_WRT_VARIABLE for "treat every variable as constant" (only
     * order 0 is meaningful in that case)
     * @param order
     * @param resultOut
     */
    public void taylorCoefficients(double[] frame, int wrtFrameIndex, int order, double[] resultOut) {
        if (order > maxOrder) {
            throw new IllegalArgumentException("order > maxOrder");
        }
        if (resultOut == null || resultOut.length < order + 1) {
            throw new IllegalArgumentException("resultOut too small");
        }

        FlatEvalState state = computeJet(frame, wrtFrameIndex, order);
        System.arraycopy(state.flatStack, 0, resultOut, 0, order + 1);
    }

    public double[] taylorCoefficients(double[] frame, int wrtFrameIndex, int order) {
        double[] out = new double[order + 1];
        taylorCoefficients(frame, wrtFrameIndex, order, out);
        return out;
    }

    // ------------------------------------------------------------------
    // Public API — name-based convenience (single-variable expressions, or
    // exploratory/test use; production solver code should prefer the
    // frame-based overloads above since they share the solver's frame layout
    // directly with no extra lookups on the hot path)
    // ------------------------------------------------------------------
    public double[] taylorCoefficients(String wrtVarName, double[] frame, int order) {
        return taylorCoefficients(frame, frameIndexOf(wrtVarName), order);
    }

    /**
     *
     * @param wrtVarName
     * @param frame execution frame holding every variable's current value,
     * indexed by Token.frameIndex — the same array layout a compiled dy_dt
     * MethodHandle uses
     * @param order
     * @return
     */
    public double[] evaluateDerivatives(String wrtVarName, double[] frame, int order) {
        return evaluateDerivatives(frame, frameIndexOf(wrtVarName), order);
    }

    // ------------------------------------------------------------------
    // Core interpreter
    // ------------------------------------------------------------------
    private FlatEvalState computeJet(double[] frame, int wrtFrameIndex, int order) {
        if (frame == null) {
            throw new IllegalArgumentException("frame must not be null");
        }

        FlatEvalState state = THREAD_LOCAL_STATE.get();
        if (state == null || state.currentMaxStackSize < maxStackSize || state.currentMaxOrder < maxOrder) {
            state = new FlatEvalState(maxStackSize, stride);
            THREAD_LOCAL_STATE.set(state);
        }

        final double[] flatStack = state.flatStack;
        final double[] scratch1 = state.scratch1;
        final double[] scratch2 = state.scratch2;
        final double[] scratch3 = state.scratch3;
        final double[] scratchArg = state.scratchArg;
        final double[] scratchU = state.scratchU;
        final double[] scratchV = state.scratchV;

        int sp = 0;
        final int n = opcodes.length;
        for (int i = 0; i < n; i++) {
            int currentOff = sp * stride;

            switch (opcodes[i]) {
                case OP_NUMBER: {
                    flatStack[currentOff] = constants[i];
                    for (int k = 1; k <= order; k++) {
                        flatStack[currentOff + k] = 0.0;
                    }
                    sp++;
                    break;
                }
                case OP_VARIABLE: {
                    Token tok = rpnTokens[i];
                    if (tok.frameIndex < 0 || tok.frameIndex >= frame.length) {
                        throw new IllegalArgumentException(
                                "Variable '" + tok.name + "' has frameIndex=" + tok.frameIndex
                                + " out of bounds for frame of length " + frame.length);
                    }
                    double val = frame[tok.frameIndex];
                    flatStack[currentOff] = val;
                    flatStack[currentOff + 1] = (tok.frameIndex == wrtFrameIndex) ? 1.0 : 0.0;
                    for (int k = 2; k <= order; k++) {
                        flatStack[currentOff + k] = 0.0;
                    }
                    sp++;
                    break;
                }
                case OP_ADD: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    add(flatStack, aOff, flatStack, bOff, flatStack, aOff, order);
                    break;
                }
                case OP_SUB: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    sub(flatStack, aOff, flatStack, bOff, flatStack, aOff, order);
                    break;
                }
                case OP_MUL: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, aOff, scratchU, 0, order + 1);
                    mul(scratchU, 0, flatStack, bOff, flatStack, aOff, order);
                    break;
                }
                case OP_DIV: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    if (Math.abs(flatStack[bOff]) < 1e-300) {
                        throw new ArithmeticException("Division by zero");
                    }
                    System.arraycopy(flatStack, aOff, scratchU, 0, order + 1);
                    recipJet(flatStack, bOff, scratch1, 0, order);
                    mul(scratchU, 0, scratch1, 0, flatStack, aOff, order);
                    break;
                }
                case OP_POW: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, aOff, scratchU, 0, order + 1);
                    powJet(scratchU, 0, flatStack, bOff, flatStack, aOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                    break;
                }
                case OP_NEG: {
                    int argOff = (sp - 1) * stride;
                    for (int k = 0; k <= order; k++) {
                        flatStack[argOff + k] = -flatStack[argOff + k];
                    }
                    break;
                }
                case OP_SIN: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    sinCosJet(scratchArg, 0, flatStack, argOff, scratch1, 0, order);
                    break;
                }
                case OP_COS: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    sinCosJet(scratchArg, 0, scratch1, 0, flatStack, argOff, order);
                    break;
                }
                case OP_TAN: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    tanJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                    break;
                }
                case OP_ASIN: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    if (Math.abs(scratchArg[0]) > 1.0) {
                        throw new ArithmeticException("asin domain");
                    }
                    flatStack[argOff] = Math.asin(scratchArg[0]);
                    mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                    scratch2[0] = 1.0 - scratch1[0];
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = -scratch1[k];
                    }
                    sqrtJet(scratch2, 0, scratch3, 0, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * flatStack[argOff + l] * scratch3[k - l];
                        }
                        flatStack[argOff + k] = (scratchArg[k] - s / k) / scratch3[0];
                    }
                    break;
                }
                case OP_ACOS: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    if (Math.abs(scratchArg[0]) > 1.0) {
                        throw new ArithmeticException("acos domain");
                    }
                    flatStack[argOff] = Math.acos(scratchArg[0]);
                    mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                    scratch2[0] = 1.0 - scratch1[0];
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = -scratch1[k];
                    }
                    sqrtJet(scratch2, 0, scratch3, 0, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * flatStack[argOff + l] * scratch3[k - l];
                        }
                        flatStack[argOff + k] = (-scratchArg[k] - s / k) / scratch3[0];
                    }
                    break;
                }
                case OP_ATAN: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    atanJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                    break;
                }
                case OP_SEC: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    sinCosJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                    recipJet(scratch2, 0, flatStack, argOff, order);
                    break;
                }
                case OP_COSEC: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    sinCosJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                    recipJet(scratch1, 0, flatStack, argOff, order);
                    break;
                }
                case OP_COT: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    tanJet(scratchArg, 0, scratch1, 0, order, scratch2, 0);
                    recipJet(scratch1, 0, flatStack, argOff, order);
                    break;
                }
                case OP_SINH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    sinhCoshJet(scratchArg, 0, flatStack, argOff, scratch1, 0, order);
                    break;
                }
                case OP_COSH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    sinhCoshJet(scratchArg, 0, scratch1, 0, flatStack, argOff, order);
                    break;
                }
                case OP_TANH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    tanhJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                    break;
                }
                case OP_ASINH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    flatStack[argOff] = Math.log(scratchArg[0] + Math.sqrt(scratchArg[0] * scratchArg[0] + 1));
                    mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                    scratch2[0] = scratch1[0] + 1.0;
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = scratch1[k];
                    }
                    sqrtJet(scratch2, 0, scratch3, 0, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * flatStack[argOff + l] * scratch3[k - l];
                        }
                        flatStack[argOff + k] = (scratchArg[k] - s / k) / scratch3[0];
                    }
                    break;
                }
                case OP_ACOSH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    if (scratchArg[0] < 1.0) {
                        throw new ArithmeticException("acosh domain");
                    }
                    flatStack[argOff] = Math.log(scratchArg[0] + Math.sqrt(scratchArg[0] * scratchArg[0] - 1));
                    mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                    scratch2[0] = scratch1[0] - 1.0;
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = scratch1[k];
                    }
                    sqrtJet(scratch2, 0, scratch3, 0, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * flatStack[argOff + l] * scratch3[k - l];
                        }
                        flatStack[argOff + k] = (scratchArg[k] - s / k) / scratch3[0];
                    }
                    break;
                }
                case OP_ATANH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    if (Math.abs(scratchArg[0]) >= 1.0) {
                        throw new ArithmeticException("atanh domain");
                    }
                    flatStack[argOff] = 0.5 * Math.log((1 + scratchArg[0]) / (1 - scratchArg[0]));
                    mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * flatStack[argOff + l] * scratch1[k - l];
                        }
                        flatStack[argOff + k] = (scratchArg[k] + s / k) / (1.0 - scratch1[0]);
                    }
                    break;
                }
                case OP_SECH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    sinhCoshJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                    recipJet(scratch2, 0, flatStack, argOff, order);
                    break;
                }
                case OP_COSECH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    sinhCoshJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                    recipJet(scratch1, 0, flatStack, argOff, order);
                    break;
                }
                case OP_COTH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    tanhJet(scratchArg, 0, scratch1, 0, order, scratch2, 0);
                    recipJet(scratch1, 0, flatStack, argOff, order);
                    break;
                }
                case OP_ASECH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    if (scratchArg[0] <= 0.0 || scratchArg[0] > 1.0) {
                        throw new ArithmeticException("asech domain");
                    }
                    flatStack[argOff] = Math.log((1.0 + Math.sqrt(1.0 - scratchArg[0] * scratchArg[0])) / scratchArg[0]);
                    mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                    scratch2[0] = 1.0 - scratch1[0];
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = -scratch1[k];
                    }
                    sqrtJet(scratch2, 0, scratch3, 0, order);
                    mul(scratchArg, 0, scratch3, 0, scratch1, 0, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * flatStack[argOff + l] * scratch1[k - l];
                        }
                        flatStack[argOff + k] = (-scratchArg[k] - s / k) / scratch1[0];
                    }
                    break;
                }
                case OP_ACOSECH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    if (scratchArg[0] == 0.0) {
                        throw new ArithmeticException("acosech domain");
                    }
                    flatStack[argOff] = Math.log(1.0 / scratchArg[0] + Math.sqrt(1.0 / (scratchArg[0] * scratchArg[0]) + 1.0));
                    mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                    scratch2[0] = scratch1[0] + 1.0;
                    for (int k = 1; k <= order; k++) {
                        scratch2[k] = scratch1[k];
                    }
                    sqrtJet(scratch2, 0, scratch3, 0, order);
                    scratch1[0] = Math.abs(scratchArg[0]);
                    for (int k = 1; k <= order; k++) {
                        scratch1[k] = (scratchArg[0] > 0) ? scratchArg[k] : (scratchArg[0] < 0) ? -scratchArg[k] : 0.0;
                    }
                    mul(scratch1, 0, scratch3, 0, scratch2, 0, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * flatStack[argOff + l] * scratch2[k - l];
                        }
                        flatStack[argOff + k] = (-scratchArg[k] - s / k) / scratch2[0];
                    }
                    break;
                }
                case OP_ACOTH: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    if (Math.abs(scratchArg[0]) <= 1.0) {
                        throw new ArithmeticException("acoth domain");
                    }
                    flatStack[argOff] = 0.5 * Math.log((scratchArg[0] + 1.0) / (scratchArg[0] - 1.0));
                    mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                    for (int k = 1; k <= order; k++) {
                        double s = 0.0;
                        for (int l = 1; l < k; l++) {
                            s += l * flatStack[argOff + l] * scratch1[k - l];
                        }
                        flatStack[argOff + k] = (scratchArg[k] + s / k) / (1.0 - scratch1[0]);
                    }
                    break;
                }
                case OP_SQRT: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    sqrtJet(scratchArg, 0, flatStack, argOff, order);
                    break;
                }
                case OP_CBRT: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    cbrtJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                    break;
                }
                case OP_EXP: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    expJet(scratchArg, 0, flatStack, argOff, order);
                    break;
                }
                case OP_LN: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    lnJet(scratchArg, 0, flatStack, argOff, order);
                    break;
                }
                case OP_LG: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    if (scratchArg[0] <= 0) {
                        throw new ArithmeticException("log domain");
                    }
                    lnJet(scratchArg, 0, flatStack, argOff, order);
                    double ln10 = Math.log(10.0);
                    for (int k = 0; k <= order; k++) {
                        flatStack[argOff + k] /= ln10;
                    }
                    break;
                }
                case OP_ABS: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    flatStack[argOff] = Math.abs(scratchArg[0]);
                    for (int k = 1; k <= order; k++) {
                        flatStack[argOff + k] = (scratchArg[0] > 0) ? scratchArg[k] : (scratchArg[0] < 0) ? -scratchArg[k] : 0.0;
                    }
                    break;
                }
                case OP_CEIL: {
                    // ceil is locally constant everywhere except at integer
                    // arguments (non-differentiable there); every derivative
                    // order >= 1 is exactly zero, NOT proportional to the
                    // argument's own derivative (that formula belongs to abs,
                    // which IS locally linear away from its kink).
                    int argOff = (sp - 1) * stride;
                    double argVal = flatStack[argOff];
                    flatStack[argOff] = Math.ceil(argVal);
                    for (int k = 1; k <= order; k++) {
                        flatStack[argOff + k] = 0.0;
                    }
                    break;
                }
                case OP_FLOOR: {
                    int argOff = (sp - 1) * stride;
                    double argVal = flatStack[argOff];
                    flatStack[argOff] = Math.floor(argVal);
                    for (int k = 1; k <= order; k++) {
                        flatStack[argOff + k] = 0.0;
                    }
                    break;
                }
                case OP_ATAN2: {
                    sp--;
                    int vOff = sp * stride;
                    int uOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, uOff, scratchU, 0, order + 1);
                    System.arraycopy(flatStack, vOff, scratchV, 0, order + 1);
                    flatStack[uOff] = Math.atan2(scratchU[0], scratchV[0]);
                    mul(scratchU, 0, scratchU, 0, scratch1, 0, order);
                    mul(scratchV, 0, scratchV, 0, scratch2, 0, order);
                    add(scratch1, 0, scratch2, 0, scratch1, 0, order);
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
                            lhsSum += l * flatStack[uOff + l] * scratch1[k - l];
                        }
                        flatStack[uOff + k] = (rhs - lhsSum) / (k * scratch1[0]);
                    }
                    break;
                }
                case OP_LOG_BASE: {
                    sp--;
                    int vOff = sp * stride;
                    int uOff = (sp - 1) * stride;
                    if (flatStack[uOff] <= 0.0) {
                        throw new ArithmeticException("log domain");
                    }
                    if (flatStack[vOff] <= 0.0 || flatStack[vOff] == 1.0) {
                        throw new ArithmeticException("log base domain");
                    }
                    System.arraycopy(flatStack, uOff, scratchU, 0, order + 1);
                    System.arraycopy(flatStack, vOff, scratchV, 0, order + 1);
                    lnJet(scratchU, 0, scratch1, 0, order);
                    lnJet(scratchV, 0, scratch2, 0, order);
                    recipJet(scratch2, 0, scratch3, 0, order);
                    mul(scratch1, 0, scratch3, 0, flatStack, uOff, order);
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

    // ===================================================================
    // Thread-local 1D Pre-allocated State
    // ===================================================================
    private static final class FlatEvalState {

        final double[] flatStack;
        final double[] scratch1, scratch2, scratch3, scratchArg, scratchU, scratchV;
        final int currentMaxStackSize;
        final int currentMaxOrder;

        FlatEvalState(int stackSize, int stride) {
            this.flatStack = new double[stackSize * stride];
            this.scratch1 = new double[stride];
            this.scratch2 = new double[stride];
            this.scratch3 = new double[stride];
            this.scratchArg = new double[stride];
            this.scratchU = new double[stride];
            this.scratchV = new double[stride];
            this.currentMaxStackSize = stackSize;
            this.currentMaxOrder = stride - 1;
        }
    }

    // ===================================================================
    // Static Offset Math Helpers (Ready for MethodHandle Integration)
    // ===================================================================
    /**
     * factorial(k) as a double: degrades gracefully toward +Infinity for very
     * large k instead of silently wrapping around like a `long` accumulator
     * would past 20! (9223372036854775807 overflows at 21!). Callers driving
     * evaluateDerivatives with large `order` should treat an Infinity/NaN
     * result as "order too high for double precision here", not as a bug.
     */
    private static double factorial(int n) {
        double f = 1.0;
        for (int i = 2; i <= n; i++) {
            f *= i;
        }
        return f;
    }

    public static void add(double[] a, int aOff, double[] b, int bOff, double[] out, int outOff, int ord) {
        for (int k = 0; k <= ord; k++) {
            out[outOff + k] = a[aOff + k] + b[bOff + k];
        }
    }

    public static void sub(double[] a, int aOff, double[] b, int bOff, double[] out, int outOff, int ord) {
        for (int k = 0; k <= ord; k++) {
            out[outOff + k] = a[aOff + k] - b[bOff + k];
        }
    }

    public static void mul(double[] a, int aOff, double[] b, int bOff, double[] out, int outOff, int ord) {
        for (int k = 0; k <= ord; k++) {
            double sum = 0.0;
            for (int i = 0; i <= k; i++) {
                sum += a[aOff + i] * b[bOff + k - i];
            }
            out[outOff + k] = sum;
        }
    }

    public static void recipJet(double[] b, int bOff, double[] out, int outOff, int ord) {
        out[outOff] = 1.0 / b[bOff];
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int i = 1; i <= k; i++) {
                s += b[bOff + i] * out[outOff + k - i];
            }
            out[outOff + k] = -out[outOff] * s;
        }
    }

    public static void lnJet(double[] u, int uOff, double[] out, int outOff, int ord) {
        double u0 = u[uOff];
        if (u0 <= 0) {
            throw new ArithmeticException("ln domain");
        }
        out[outOff] = Math.log(u0);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int j = 1; j < k; j++) {
                s += j * out[outOff + j] * u[uOff + k - j];
            }
            out[outOff + k] = (u[uOff + k] - s / k) / u0;
        }
    }

    public static void expJet(double[] w, int wOff, double[] out, int outOff, int ord) {
        out[outOff] = Math.exp(w[wOff]);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int j = 1; j <= k; j++) {
                s += j * w[wOff + j] * out[outOff + k - j];
            }
            out[outOff + k] = s / k;
        }
    }

    public static void intPowJet(double[] u, int uOff, int n, double[] out, int outOff, int ord, double[] scratch1, int s1Off) {
        if (n == 0) {
            out[outOff] = 1.0;
            for (int k = 1; k <= ord; k++) {
                out[outOff + k] = 0.0;
            }
            return;
        }
        System.arraycopy(u, uOff, out, outOff, ord + 1);
        for (int p = 2; p <= n; p++) {
            mul(out, outOff, u, uOff, scratch1, s1Off, ord);
            System.arraycopy(scratch1, s1Off, out, outOff, ord + 1);
        }
    }

    public static void powJet(double[] u, int uOff, double[] v, int vOff, double[] out, int outOff, int ord,
            double[] scratch1, int s1Off, double[] scratch2, int s2Off, double[] scratch3, int s3Off) {
        boolean vConstant = true;
        for (int k = 1; k <= ord; k++) {
            if (v[vOff + k] != 0.0) {
                vConstant = false;
                break;
            }
        }
        if (vConstant) {
            double r = v[vOff];
            if (!Double.isNaN(r) && !Double.isInfinite(r) && r == Math.floor(r)) {
                int n = (int) r;
                if (n >= 0) {
                    intPowJet(u, uOff, n, out, outOff, ord, scratch1, s1Off);
                } else {
                    if (u[uOff] == 0.0) {
                        throw new ArithmeticException("pow domain: zero base with negative exponent");
                    }
                    intPowJet(u, uOff, -n, scratch3, s3Off, ord, scratch1, s1Off);
                    recipJet(scratch3, s3Off, out, outOff, ord);
                }
                return;
            }
        }
        if (u[uOff] <= 0.0) {
            throw new ArithmeticException("pow domain: non-real result");
        }
        lnJet(u, uOff, scratch1, s1Off, ord);
        mul(v, vOff, scratch1, s1Off, scratch2, s2Off, ord);
        expJet(scratch2, s2Off, out, outOff, ord);
    }

    public static void sinCosJet(double[] u, int uOff, double[] sinOut, int sinOff, double[] cosOut, int cosOff, int ord) {
        sinOut[sinOff] = Math.sin(u[uOff]);
        cosOut[cosOff] = Math.cos(u[uOff]);
        for (int k = 1; k <= ord; k++) {
            double sSum = 0.0, cSum = 0.0;
            for (int j = 1; j <= k; j++) {
                sSum += j * u[uOff + j] * cosOut[cosOff + k - j];
                cSum += j * u[uOff + j] * sinOut[sinOff + k - j];
            }
            sinOut[sinOff + k] = sSum / k;
            cosOut[cosOff + k] = -cSum / k;
        }
    }

    public static void sinhCoshJet(double[] u, int uOff, double[] sinhOut, int sinhOff, double[] coshOut, int coshOff, int ord) {
        sinhOut[sinhOff] = Math.sinh(u[uOff]);
        coshOut[coshOff] = Math.cosh(u[uOff]);
        for (int k = 1; k <= ord; k++) {
            double sSum = 0.0, cSum = 0.0;
            for (int j = 1; j <= k; j++) {
                sSum += j * u[uOff + j] * coshOut[coshOff + k - j];
                cSum += j * u[uOff + j] * sinhOut[sinhOff + k - j];
            }
            sinhOut[sinhOff + k] = sSum / k;
            coshOut[coshOff + k] = cSum / k;
        }
    }

    public static void tanJet(double[] u, int uOff, double[] out, int outOff, int ord, double[] pScratch, int pOff) {
        out[outOff] = Math.tan(u[uOff]);
        pScratch[pOff] = out[outOff] * out[outOff];
        for (int k = 1; k <= ord; k++) {
            double pk_1 = 0.0;
            for (int i = 0; i <= k - 1; i++) {
                pk_1 += out[outOff + i] * out[outOff + k - 1 - i];
            }
            pScratch[pOff + k - 1] = pk_1;
            double s = 0.0;
            for (int j = 1; j < k; j++) {
                s += j * u[uOff + j] * pScratch[pOff + k - j];
            }
            out[outOff + k] = u[uOff + k] * (1.0 + pScratch[pOff]) + s / k;
        }
    }

    public static void tanhJet(double[] u, int uOff, double[] out, int outOff, int ord, double[] pScratch, int pOff) {
        out[outOff] = Math.tanh(u[uOff]);
        pScratch[pOff] = out[outOff] * out[outOff];
        for (int k = 1; k <= ord; k++) {
            double pk_1 = 0.0;
            for (int i = 0; i <= k - 1; i++) {
                pk_1 += out[outOff + i] * out[outOff + k - 1 - i];
            }
            pScratch[pOff + k - 1] = pk_1;
            double s = 0.0;
            for (int j = 1; j < k; j++) {
                s += j * u[uOff + j] * pScratch[pOff + k - j];
            }
            out[outOff + k] = u[uOff + k] * (1.0 - pScratch[pOff]) - s / k;
        }
    }

    public static void sqrtJet(double[] u, int uOff, double[] out, int outOff, int ord) {
        if (u[uOff] < 0) {
            throw new ArithmeticException("sqrt domain");
        }
        out[outOff] = Math.sqrt(u[uOff]);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int j = 1; j < k; j++) {
                s += j * out[outOff + j] * out[outOff + k - j];
            }
            out[outOff + k] = (u[uOff + k] - 2.0 * s / k) / (2.0 * out[outOff]);
        }
    }

    public static void cbrtJet(double[] u, int uOff, double[] out, int outOff, int ord, double[] pScratch, int pOff) {
        out[outOff] = Math.cbrt(u[uOff]);
        pScratch[pOff] = out[outOff] * out[outOff];
        for (int k = 1; k <= ord; k++) {
            double pk_1 = 0.0;
            for (int i = 0; i <= k - 1; i++) {
                pk_1 += out[outOff + i] * out[outOff + k - 1 - i];
            }
            pScratch[pOff + k - 1] = pk_1;
            double s = 0.0;
            for (int l = 1; l < k; l++) {
                s += l * out[outOff + l] * pScratch[pOff + k - l];
            }
            out[outOff + k] = (u[uOff + k] - 3.0 * s / k) / (3.0 * pScratch[pOff]);
        }
    }

    public static void atanJet(double[] u, int uOff, double[] out, int outOff, int ord, double[] pScratch, int pOff) {
        out[outOff] = Math.atan(u[uOff]);
        mul(u, uOff, u, uOff, pScratch, pOff, ord);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int l = 1; l < k; l++) {
                s += l * out[outOff + l] * pScratch[pOff + k - l];
            }
            out[outOff + k] = (u[uOff + k] - s / k) / (1.0 + pScratch[pOff]);
        }
    }
}
