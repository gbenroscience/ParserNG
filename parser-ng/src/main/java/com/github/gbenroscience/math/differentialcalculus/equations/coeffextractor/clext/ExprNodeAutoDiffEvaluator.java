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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

/**
 *
 * @author GBEMIRO
 */ 
import com.github.gbenroscience.math.differentialcalculus.autodiff.SystemAutoDiffEvaluator;

import java.util.HashMap;
import java.util.Map;

/**
 * The {@link ExprNode} counterpart to {@link SystemAutoDiffEvaluator} — real
 * nth-order forward-mode automatic differentiation, not the finite-difference
 * fallback {@code ExprNodeCompiler}'s own javadoc describes. This is the
 * "second compilation target building jet arithmetic over the same ExprNode
 * tree" that class flagged as separate work; this class is that work.
 *
 * <h2>Why a separate class instead of reusing SystemAutoDiffEvaluator directly</h2>
 * {@code SystemAutoDiffEvaluator} is wired to {@code MathExpression}'s
 * {@code Token[]} postfix stream — it needs a real {@code MathExpression} to
 * construct from. {@code ExprNode} trees built via {@link TokenTreeBuilder}
 * carry everything needed (real frame indices, resolved function opcodes)
 * without one. Rather than force a round-trip through a synthetic
 * {@code MathExpression}, this class flattens an {@code ExprNode} tree
 * straight into the same flat post-order representation
 * {@code SystemAutoDiffEvaluator} interprets, and reuses that class's actual
 * jet-arithmetic primitives ({@code add, sub, mul, recipJet, lnJet, expJet,
 * sqrtJet, cbrtJet, sinCosJet, sinhCoshJet, tanJet, tanhJet, atanJet, powJet,
 * intPowJet} — all {@code public static}) directly, rather than copying
 * them. Only {@code asin}/{@code acos} jets are re-derived here, since
 * {@code SystemAutoDiffEvaluator} has those inlined in its switch rather than
 * factored into reusable statics.
 *
 * <h2>Function coverage: shares FunctionOpcodes with ExprNodeCompiler</h2>
 * Every {@code ExprNode} function-call node already carries a resolved
 * {@link ExprNode#funcOpcode}, set once by {@link FunctionOpcodes} at
 * tree-build time. This evaluator switches on that exact same {@code int}
 * space, so it supports precisely the functions {@code ExprNodeCompiler}
 * does — including the deg/grad trig variants — with no separate resolution
 * table to keep in sync.
 * <ul>
 *   <li><b>asec/acsc/acot</b> — absent from {@code SystemAutoDiffEvaluator}'s
 *       own opcode set — are computed by composition: {@code asec(a) =
 *       acos(1/a)}, so the reciprocal jet of the argument is computed first,
 *       then run through the ordinary {@code acos} jet routine (and
 *       symmetrically for acsc/atan). The chain rule falls out of that
 *       composition automatically; no separate derivative formula needed.</li>
 *   <li><b>deg/grad variants</b> — scaling by a constant (DEG_TO_RAD,
 *       RAD_TO_DEG, ...) is linear, so it commutes with differentiation: a
 *       forward trig deg/grad function pre-scales the argument jet
 *       (uniformly, all orders) before the ordinary jet routine; an inverse
 *       trig deg/grad function post-scales the result jet the same way.</li>
 * </ul>
 *
 * <h2>Frame-only — matches TokenTreeBuilder's real trees</h2>
 * Like {@link ExprNodeCompiler#compileStandard(ExprNode)} /
 * {@link ExprNodeCompiler#compileTurbo(ExprNode)} (the no-tSlot/ySlotStart
 * overloads), this class requires every VARIABLE leaf to already carry a
 * real, registry-assigned {@code frameIndex} — thrown at construction
 * otherwise. This is the tree shape {@code TokenTreeBuilder} actually
 * produces, and it's also all a Jacobian column needs: which frame slot to
 * differentiate against.
 *
 * <h2>Performance / allocation</h2>
 * The tree is flattened to parallel primitive arrays exactly once, at
 * construction — not per evaluation. Each call to
 * {@link #evaluateDerivatives} / {@link #taylorCoefficients} reuses a
 * {@code ThreadLocal}, pre-sized scratch-buffer set (mirroring
 * {@code SystemAutoDiffEvaluator}'s {@code FlatEvalState}), so repeated
 * Jacobian-column evaluations at solver steady-state are allocation-free.
 */
public final class ExprNodeAutoDiffEvaluator implements Cloneable {

    /**
     * Sentinel wrtFrameIndex meaning "no variable matches" — every VARIABLE
     * leaf is treated as a constant w.r.t. the differentiation, since real
     * frame indices are always >= 0.
     */
    public static final int NO_WRT_VARIABLE = -1;

    // ------------------------------------------------------------------
    // Flat post-order node kinds (structural — function identity lives in
    // funcOpcodes[i], resolved once already by FunctionOpcodes)
    // ------------------------------------------------------------------
    private static final byte NK_NUMBER = 0;
    private static final byte NK_VARIABLE = 1;
    private static final byte NK_ADD = 2;
    private static final byte NK_SUB = 3;
    private static final byte NK_MUL = 4;
    private static final byte NK_DIV = 5;
    private static final byte NK_POW = 6;
    private static final byte NK_NEG = 7;
    private static final byte NK_FUNC1 = 8;
    private static final byte NK_FUNC2 = 9;

    private static final double DEG_TO_RAD = Math.PI / 180.0;
    private static final double RAD_TO_DEG = 180.0 / Math.PI;
    private static final double GRAD_TO_RAD = Math.PI / 200.0;
    private static final double RAD_TO_GRAD = 200.0 / Math.PI;
    private static final double LN10 = Math.log(10.0);

    private final byte[] nodeKinds;
    private final double[] constants;
    private final int[] frameIndices;
    private final int[] funcOpcodes;
    private final int maxOrder;
    private final int stride;
    private final int maxStackSize;

    /** name -> frameIndex for every distinct VARIABLE leaf, built once at construction. */
    private final Map<String, Integer> nameToFrameIndex;

    private static final ThreadLocal<AdEvalState> THREAD_LOCAL_STATE = new ThreadLocal<>();

    public ExprNodeAutoDiffEvaluator(ExprNode expression, int maxOrder) {
        if (expression == null) {
            throw new IllegalArgumentException("expression must not be null");
        }
        if (maxOrder < 0) {
            throw new IllegalArgumentException("maxOrder >= 0 required");
        }
        this.maxOrder = maxOrder;
        this.stride = maxOrder + 1;

        int n = countNodes(expression);
        this.nodeKinds = new byte[n];
        this.constants = new double[n];
        this.frameIndices = new int[n];
        this.funcOpcodes = new int[n];
        flatten(expression, new int[]{0});
        this.maxStackSize = n + 1;

        Map<String, Integer> names = new HashMap<>();
        collectVariableNames(expression, names);
        this.nameToFrameIndex = names;
    }

    private ExprNodeAutoDiffEvaluator(byte[] nodeKinds, double[] constants, int[] frameIndices, int[] funcOpcodes,
                                       int maxOrder, int maxStackSize, Map<String, Integer> nameToFrameIndex) {
        this.nodeKinds = nodeKinds;
        this.constants = constants;
        this.frameIndices = frameIndices;
        this.funcOpcodes = funcOpcodes;
        this.maxOrder = maxOrder;
        this.stride = maxOrder + 1;
        this.maxStackSize = maxStackSize;
        this.nameToFrameIndex = nameToFrameIndex;
    }

    @Override
    protected ExprNodeAutoDiffEvaluator clone() {
        return new ExprNodeAutoDiffEvaluator(nodeKinds.clone(), constants.clone(), frameIndices.clone(),
                funcOpcodes.clone(), maxOrder, maxStackSize, nameToFrameIndex);
    }

    public int getMaxOrder() {
        return maxOrder;
    }

    /**
     * Frame index of the given variable name in this expression, or
     * NO_WRT_VARIABLE if the name does not appear at all (whose derivative
     * w.r.t. it is correctly zero everywhere).
     */
    public int frameIndexOf(String varName) {
        Integer fi = nameToFrameIndex.get(varName);
        return fi == null ? NO_WRT_VARIABLE : fi;
    }

    // ------------------------------------------------------------------
    // ExprNode tree -> flat post-order compilation (once, at construction)
    // ------------------------------------------------------------------

    private static int countNodes(ExprNode node) {
        int count = 1;
        if (node.kind == ExprNode.Kind.OP) {
            for (ExprNode child : node.children) {
                count += countNodes(child);
            }
        }
        return count;
    }

    private void flatten(ExprNode node, int[] cursor) {
        if (node.kind == ExprNode.Kind.OP) {
            for (ExprNode child : node.children) {
                flatten(child, cursor);
            }
        }
        int i = cursor[0]++;
        switch (node.kind) {
            case NUMBER:
                nodeKinds[i] = NK_NUMBER;
                constants[i] = node.numberValue;
                return;
            case VARIABLE:
                if (node.frameIndex < 0) {
                    throw new IllegalArgumentException(
                            "Variable '" + node.variableName + "' has no real frameIndex — "
                            + "ExprNodeAutoDiffEvaluator requires every leaf to come from TokenTreeBuilder "
                            + "(or otherwise carry a real registry slot).");
                }
                nodeKinds[i] = NK_VARIABLE;
                frameIndices[i] = node.frameIndex;
                return;
            case OP:
                if (node.isFunctionCall()) {
                    if (node.funcOpcode == FunctionOpcodes.UNRESOLVED) {
                        throw new IllegalArgumentException(
                                "Unsupported function '" + node.funcName + "' with " + node.children.size()
                                + " argument(s) for automatic differentiation.");
                    }
                    nodeKinds[i] = node.children.size() == 1 ? NK_FUNC1 : NK_FUNC2;
                    funcOpcodes[i] = node.funcOpcode;
                    return;
                }
                nodeKinds[i] = node.children.size() == 1 ? NK_NEG : opNodeKind(node.opChar);
                return;
            default:
                throw new IllegalStateException("Unreachable");
        }
    }

    private static byte opNodeKind(char opChar) {
        switch (opChar) {
            case '+': return NK_ADD;
            case '-': return NK_SUB;
            case '*': return NK_MUL;
            case '/': return NK_DIV;
            case '^': return NK_POW;
            default:
                throw new IllegalArgumentException(
                        "Unsupported operator '" + opChar + "' for automatic differentiation.");
        }
    }

    private static void collectVariableNames(ExprNode node, Map<String, Integer> out) {
        if (node.kind == ExprNode.Kind.VARIABLE) {
            if (node.variableName != null) {
                out.putIfAbsent(node.variableName, node.frameIndex);
            }
            return;
        }
        if (node.kind == ExprNode.Kind.OP) {
            for (ExprNode child : node.children) {
                collectVariableNames(child, out);
            }
        }
    }

    // ------------------------------------------------------------------
    // Public API — raw derivatives and Taylor coefficients
    // ------------------------------------------------------------------

    /**
     * Raw nth-order derivatives d^k f / dx_wrt^k at the point described by
     * frame, for k = 0..order. resultOut[k] is the true kth derivative (not
     * the Taylor coefficient — it has been multiplied by k!).
     *
     * @param frame          execution frame holding every variable's current value,
     *                       indexed by frameIndex — the same array layout a compiled
     *                       dy_dt MethodHandle uses
     * @param wrtFrameIndex  frame index of the variable to differentiate against;
     *                       use NO_WRT_VARIABLE to treat every variable as constant
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
        AdEvalState state = computeJet(frame, wrtFrameIndex, order);
        System.arraycopy(state.flatStack, 0, resultOut, 0, order + 1);
        for (int k = 1; k <= order; k++) {
            resultOut[k] *= factorial(k);
        }
    }

    public double[] evaluateDerivatives(double[] frame, int wrtFrameIndex, int order) {
        double[] out = new double[order + 1];
        evaluateDerivatives(frame, wrtFrameIndex, order, out);
        return out;
    }

    /**
     * Taylor coefficients [f, f', f''/2!, ..., f^(order)/order!] at the point
     * described by frame — resultOut[1] === df/dx_wrt directly, no factorial
     * multiply needed; this is what a Jacobian entry actually wants.
     */
    public void taylorCoefficients(double[] frame, int wrtFrameIndex, int order, double[] resultOut) {
        if (order > maxOrder) {
            throw new IllegalArgumentException("order > maxOrder");
        }
        if (resultOut == null || resultOut.length < order + 1) {
            throw new IllegalArgumentException("resultOut too small");
        }
        AdEvalState state = computeJet(frame, wrtFrameIndex, order);
        System.arraycopy(state.flatStack, 0, resultOut, 0, order + 1);
    }

    public double[] taylorCoefficients(double[] frame, int wrtFrameIndex, int order) {
        double[] out = new double[order + 1];
        taylorCoefficients(frame, wrtFrameIndex, order, out);
        return out;
    }

    /** Convenience for a single Jacobian entry: df/d(frame[wrtFrameIndex]) at this frame. */
    public double firstDerivative(double[] frame, int wrtFrameIndex) {
        return taylorCoefficients(frame, wrtFrameIndex, 1)[1];
    }

    // ------------------------------------------------------------------
    // Public API — name-based convenience
    // ------------------------------------------------------------------

    public double[] taylorCoefficients(String wrtVarName, double[] frame, int order) {
        return taylorCoefficients(frame, frameIndexOf(wrtVarName), order);
    }

    public double[] evaluateDerivatives(String wrtVarName, double[] frame, int order) {
        return evaluateDerivatives(frame, frameIndexOf(wrtVarName), order);
    }

    // ------------------------------------------------------------------
    // Core interpreter — mirrors SystemAutoDiffEvaluator.computeJet's flat-
    // stack shape exactly, sourced from this class's flattened arrays
    // instead of a Token[]/opcode[] pair, and switching on FunctionOpcodes'
    // int space instead of SystemAutoDiffEvaluator's own OP_* constants.
    // ------------------------------------------------------------------

    private AdEvalState computeJet(double[] frame, int wrtFrameIndex, int order) {
        if (frame == null) {
            throw new IllegalArgumentException("frame must not be null");
        }

        AdEvalState state = THREAD_LOCAL_STATE.get();
        if (state == null || state.currentMaxStackSize < maxStackSize || state.currentMaxOrder < maxOrder) {
            state = new AdEvalState(maxStackSize, stride);
            THREAD_LOCAL_STATE.set(state);
        }

        final double[] flatStack = state.flatStack;
        final double[] scratch1 = state.scratch1;
        final double[] scratch2 = state.scratch2;
        final double[] scratch3 = state.scratch3;
        final double[] scratch4 = state.scratch4;
        final double[] scratchArg = state.scratchArg;
        final double[] scratchU = state.scratchU;
        final double[] scratchV = state.scratchV;

        int sp = 0;
        final int n = nodeKinds.length;
        for (int i = 0; i < n; i++) {
            switch (nodeKinds[i]) {
                case NK_NUMBER: {
                    int currentOff = sp * stride;
                    flatStack[currentOff] = constants[i];
                    for (int k = 1; k <= order; k++) {
                        flatStack[currentOff + k] = 0.0;
                    }
                    sp++;
                    break;
                }
                case NK_VARIABLE: {
                    int currentOff = sp * stride;
                    int fi = frameIndices[i];
                    if (fi < 0 || fi >= frame.length) {
                        throw new IllegalArgumentException(
                                "frameIndex " + fi + " out of bounds for frame of length " + frame.length);
                    }
                    flatStack[currentOff] = frame[fi];
                    flatStack[currentOff + 1] = (fi == wrtFrameIndex) ? 1.0 : 0.0;
                    for (int k = 2; k <= order; k++) {
                        flatStack[currentOff + k] = 0.0;
                    }
                    sp++;
                    break;
                }
                case NK_ADD: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    SystemAutoDiffEvaluator.add(flatStack, aOff, flatStack, bOff, flatStack, aOff, order);
                    break;
                }
                case NK_SUB: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    SystemAutoDiffEvaluator.sub(flatStack, aOff, flatStack, bOff, flatStack, aOff, order);
                    break;
                }
                case NK_MUL: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, aOff, scratchU, 0, order + 1);
                    SystemAutoDiffEvaluator.mul(scratchU, 0, flatStack, bOff, flatStack, aOff, order);
                    break;
                }
                case NK_DIV: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    if (Math.abs(flatStack[bOff]) < 1e-300) {
                        throw new ArithmeticException("Division by zero");
                    }
                    System.arraycopy(flatStack, aOff, scratchU, 0, order + 1);
                    SystemAutoDiffEvaluator.recipJet(flatStack, bOff, scratch1, 0, order);
                    SystemAutoDiffEvaluator.mul(scratchU, 0, scratch1, 0, flatStack, aOff, order);
                    break;
                }
                case NK_POW: {
                    sp--;
                    int bOff = sp * stride;
                    int aOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, aOff, scratchU, 0, order + 1);
                    SystemAutoDiffEvaluator.powJet(scratchU, 0, flatStack, bOff, flatStack, aOff, order,
                            scratch1, 0, scratch2, 0, scratch3, 0);
                    break;
                }
                case NK_NEG: {
                    int argOff = (sp - 1) * stride;
                    for (int k = 0; k <= order; k++) {
                        flatStack[argOff + k] = -flatStack[argOff + k];
                    }
                    break;
                }
                case NK_FUNC1: {
                    int argOff = (sp - 1) * stride;
                    System.arraycopy(flatStack, argOff, scratchArg, 0, order + 1);
                    evaluateOneArgJet(funcOpcodes[i], scratchArg, flatStack, argOff, order,
                            scratch1, scratch2, scratch3, scratch4);
                    break;
                }
                case NK_FUNC2: {
                    sp--;
                    int vOff = sp * stride;
                    int uOff = (sp - 1) * stride;
                    evaluateTwoArgJet(funcOpcodes[i], flatStack, uOff, vOff, order,
                            scratch1, scratch2, scratch3, scratchU, scratchV);
                    break;
                }
                default:
                    throw new IllegalStateException("Unreachable: node kind " + nodeKinds[i]);
            }
        }

        if (sp != 1) {
            throw new IllegalStateException("Malformed flattened tree: expected 1 result, got " + sp);
        }
        return state;
    }

    // ------------------------------------------------------------------
    // One-argument function jets — switch(int) on FunctionOpcodes' space,
    // sharing SystemAutoDiffEvaluator's public static jet primitives.
    // ------------------------------------------------------------------

    private static void evaluateOneArgJet(int opcode, double[] scratchArg, double[] flatStack, int argOff, int order,
                                           double[] scratch1, double[] scratch2, double[] scratch3, double[] scratch4) {
        switch (opcode) {
            // --- non-angular ---
            case FunctionOpcodes.SQRT:
                SystemAutoDiffEvaluator.sqrtJet(scratchArg, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.CBRT:
                SystemAutoDiffEvaluator.cbrtJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                return;
            case FunctionOpcodes.EXP:
                SystemAutoDiffEvaluator.expJet(scratchArg, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.LN:
                SystemAutoDiffEvaluator.lnJet(scratchArg, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.LOG10:
                SystemAutoDiffEvaluator.lnJet(scratchArg, 0, flatStack, argOff, order);
                scaleJetInPlace(flatStack, argOff, 1.0 / LN10, order);
                return;
            case FunctionOpcodes.ABS:
                flatStack[argOff] = Math.abs(scratchArg[0]);
                for (int k = 1; k <= order; k++) {
                    flatStack[argOff + k] = (scratchArg[0] > 0) ? scratchArg[k]
                            : (scratchArg[0] < 0) ? -scratchArg[k] : 0.0;
                }
                return;

            // --- hyperbolic (forward) ---
            case FunctionOpcodes.SINH:
                SystemAutoDiffEvaluator.sinhCoshJet(scratchArg, 0, flatStack, argOff, scratch1, 0, order);
                return;
            case FunctionOpcodes.COSH:
                SystemAutoDiffEvaluator.sinhCoshJet(scratchArg, 0, scratch1, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.TANH:
                SystemAutoDiffEvaluator.tanhJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                return;

            // --- hyperbolic (inverse) ---
            case FunctionOpcodes.ASINH: {
                flatStack[argOff] = Math.log(scratchArg[0] + Math.sqrt(scratchArg[0] * scratchArg[0] + 1));
                SystemAutoDiffEvaluator.mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                scratch2[0] = scratch1[0] + 1.0;
                for (int k = 1; k <= order; k++) {
                    scratch2[k] = scratch1[k];
                }
                SystemAutoDiffEvaluator.sqrtJet(scratch2, 0, scratch3, 0, order);
                for (int k = 1; k <= order; k++) {
                    double s = 0.0;
                    for (int l = 1; l < k; l++) {
                        s += l * flatStack[argOff + l] * scratch3[k - l];
                    }
                    flatStack[argOff + k] = (scratchArg[k] - s / k) / scratch3[0];
                }
                return;
            }
            case FunctionOpcodes.ACOSH: {
                if (scratchArg[0] < 1.0) {
                    throw new ArithmeticException("acosh domain");
                }
                flatStack[argOff] = Math.log(scratchArg[0] + Math.sqrt(scratchArg[0] * scratchArg[0] - 1));
                SystemAutoDiffEvaluator.mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                scratch2[0] = scratch1[0] - 1.0;
                for (int k = 1; k <= order; k++) {
                    scratch2[k] = scratch1[k];
                }
                SystemAutoDiffEvaluator.sqrtJet(scratch2, 0, scratch3, 0, order);
                for (int k = 1; k <= order; k++) {
                    double s = 0.0;
                    for (int l = 1; l < k; l++) {
                        s += l * flatStack[argOff + l] * scratch3[k - l];
                    }
                    flatStack[argOff + k] = (scratchArg[k] - s / k) / scratch3[0];
                }
                return;
            }
            case FunctionOpcodes.ATANH: {
                if (Math.abs(scratchArg[0]) >= 1.0) {
                    throw new ArithmeticException("atanh domain");
                }
                flatStack[argOff] = 0.5 * Math.log((1 + scratchArg[0]) / (1 - scratchArg[0]));
                SystemAutoDiffEvaluator.mul(scratchArg, 0, scratchArg, 0, scratch1, 0, order);
                for (int k = 1; k <= order; k++) {
                    double s = 0.0;
                    for (int l = 1; l < k; l++) {
                        s += l * flatStack[argOff + l] * scratch1[k - l];
                    }
                    flatStack[argOff + k] = (scratchArg[k] + s / k) / (1.0 - scratch1[0]);
                }
                return;
            }

            // --- standard trig ---
            case FunctionOpcodes.SIN:
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, flatStack, argOff, scratch1, 0, order);
                return;
            case FunctionOpcodes.SIN_DEG:
                scaleJetInPlace(scratchArg, 0, DEG_TO_RAD, order);
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, flatStack, argOff, scratch1, 0, order);
                return;
            case FunctionOpcodes.SIN_GRAD:
                scaleJetInPlace(scratchArg, 0, GRAD_TO_RAD, order);
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, flatStack, argOff, scratch1, 0, order);
                return;
            case FunctionOpcodes.COS:
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, scratch1, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.COS_DEG:
                scaleJetInPlace(scratchArg, 0, DEG_TO_RAD, order);
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, scratch1, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.COS_GRAD:
                scaleJetInPlace(scratchArg, 0, GRAD_TO_RAD, order);
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, scratch1, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.TAN:
                SystemAutoDiffEvaluator.tanJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                return;
            case FunctionOpcodes.TAN_DEG:
                scaleJetInPlace(scratchArg, 0, DEG_TO_RAD, order);
                SystemAutoDiffEvaluator.tanJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                return;
            case FunctionOpcodes.TAN_GRAD:
                scaleJetInPlace(scratchArg, 0, GRAD_TO_RAD, order);
                SystemAutoDiffEvaluator.tanJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                return;

            // --- inverse trig ---
            case FunctionOpcodes.ASIN:
                asinJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                return;
            case FunctionOpcodes.ASIN_DEG:
                asinJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_DEG, order);
                return;
            case FunctionOpcodes.ASIN_GRAD:
                asinJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_GRAD, order);
                return;
            case FunctionOpcodes.ACOS:
                acosJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                return;
            case FunctionOpcodes.ACOS_DEG:
                acosJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_DEG, order);
                return;
            case FunctionOpcodes.ACOS_GRAD:
                acosJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_GRAD, order);
                return;
            case FunctionOpcodes.ATAN:
                SystemAutoDiffEvaluator.atanJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                return;
            case FunctionOpcodes.ATAN_DEG:
                SystemAutoDiffEvaluator.atanJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_DEG, order);
                return;
            case FunctionOpcodes.ATAN_GRAD:
                SystemAutoDiffEvaluator.atanJet(scratchArg, 0, flatStack, argOff, order, scratch1, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_GRAD, order);
                return;

            // --- reciprocal trig ---
            case FunctionOpcodes.SEC:
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                SystemAutoDiffEvaluator.recipJet(scratch2, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.SEC_DEG:
                scaleJetInPlace(scratchArg, 0, DEG_TO_RAD, order);
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                SystemAutoDiffEvaluator.recipJet(scratch2, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.SEC_GRAD:
                scaleJetInPlace(scratchArg, 0, GRAD_TO_RAD, order);
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                SystemAutoDiffEvaluator.recipJet(scratch2, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.CSC:
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                SystemAutoDiffEvaluator.recipJet(scratch1, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.CSC_DEG:
                scaleJetInPlace(scratchArg, 0, DEG_TO_RAD, order);
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                SystemAutoDiffEvaluator.recipJet(scratch1, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.CSC_GRAD:
                scaleJetInPlace(scratchArg, 0, GRAD_TO_RAD, order);
                SystemAutoDiffEvaluator.sinCosJet(scratchArg, 0, scratch1, 0, scratch2, 0, order);
                SystemAutoDiffEvaluator.recipJet(scratch1, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.COT:
                SystemAutoDiffEvaluator.tanJet(scratchArg, 0, scratch1, 0, order, scratch2, 0);
                SystemAutoDiffEvaluator.recipJet(scratch1, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.COT_DEG:
                scaleJetInPlace(scratchArg, 0, DEG_TO_RAD, order);
                SystemAutoDiffEvaluator.tanJet(scratchArg, 0, scratch1, 0, order, scratch2, 0);
                SystemAutoDiffEvaluator.recipJet(scratch1, 0, flatStack, argOff, order);
                return;
            case FunctionOpcodes.COT_GRAD:
                scaleJetInPlace(scratchArg, 0, GRAD_TO_RAD, order);
                SystemAutoDiffEvaluator.tanJet(scratchArg, 0, scratch1, 0, order, scratch2, 0);
                SystemAutoDiffEvaluator.recipJet(scratch1, 0, flatStack, argOff, order);
                return;

            // --- inverse reciprocal trig: asec(a)=acos(1/a), acsc(a)=asin(1/a), acot(a)=atan(1/a) ---
            case FunctionOpcodes.ASEC:
                SystemAutoDiffEvaluator.recipJet(scratchArg, 0, scratch4, 0, order);
                acosJet(scratch4, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                return;
            case FunctionOpcodes.ASEC_DEG:
                SystemAutoDiffEvaluator.recipJet(scratchArg, 0, scratch4, 0, order);
                acosJet(scratch4, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_DEG, order);
                return;
            case FunctionOpcodes.ASEC_GRAD:
                SystemAutoDiffEvaluator.recipJet(scratchArg, 0, scratch4, 0, order);
                acosJet(scratch4, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_GRAD, order);
                return;
            case FunctionOpcodes.ACSC:
                SystemAutoDiffEvaluator.recipJet(scratchArg, 0, scratch4, 0, order);
                asinJet(scratch4, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                return;
            case FunctionOpcodes.ACSC_DEG:
                SystemAutoDiffEvaluator.recipJet(scratchArg, 0, scratch4, 0, order);
                asinJet(scratch4, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_DEG, order);
                return;
            case FunctionOpcodes.ACSC_GRAD:
                SystemAutoDiffEvaluator.recipJet(scratchArg, 0, scratch4, 0, order);
                asinJet(scratch4, 0, flatStack, argOff, order, scratch1, 0, scratch2, 0, scratch3, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_GRAD, order);
                return;
            case FunctionOpcodes.ACOT:
                SystemAutoDiffEvaluator.recipJet(scratchArg, 0, scratch4, 0, order);
                SystemAutoDiffEvaluator.atanJet(scratch4, 0, flatStack, argOff, order, scratch1, 0);
                return;
            case FunctionOpcodes.ACOT_DEG:
                SystemAutoDiffEvaluator.recipJet(scratchArg, 0, scratch4, 0, order);
                SystemAutoDiffEvaluator.atanJet(scratch4, 0, flatStack, argOff, order, scratch1, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_DEG, order);
                return;
            case FunctionOpcodes.ACOT_GRAD:
                SystemAutoDiffEvaluator.recipJet(scratchArg, 0, scratch4, 0, order);
                SystemAutoDiffEvaluator.atanJet(scratch4, 0, flatStack, argOff, order, scratch1, 0);
                scaleJetInPlace(flatStack, argOff, RAD_TO_GRAD, order);
                return;

            default:
                throw new IllegalStateException(
                        "Unreachable: unresolved/unsupported one-argument opcode " + opcode
                        + " — this should have been caught by the constructor's flatten() validation.");
        }
    }

    // ------------------------------------------------------------------
    // Two-argument function jets
    // ------------------------------------------------------------------

    private static void evaluateTwoArgJet(int opcode, double[] flatStack, int uOff, int vOff, int order,
                                           double[] scratch1, double[] scratch2, double[] scratch3,
                                           double[] scratchU, double[] scratchV) {
        switch (opcode) {
            case FunctionOpcodes.ATAN2: {
                System.arraycopy(flatStack, uOff, scratchU, 0, order + 1);
                System.arraycopy(flatStack, vOff, scratchV, 0, order + 1);
                flatStack[uOff] = Math.atan2(scratchU[0], scratchV[0]);
                SystemAutoDiffEvaluator.mul(scratchU, 0, scratchU, 0, scratch1, 0, order);
                SystemAutoDiffEvaluator.mul(scratchV, 0, scratchV, 0, scratch2, 0, order);
                SystemAutoDiffEvaluator.add(scratch1, 0, scratch2, 0, scratch1, 0, order);
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
                return;
            }
            case FunctionOpcodes.LOG_BASE: {
                if (flatStack[uOff] <= 0.0) {
                    throw new ArithmeticException("log domain");
                }
                if (flatStack[vOff] <= 0.0 || flatStack[vOff] == 1.0) {
                    throw new ArithmeticException("log base domain");
                }
                System.arraycopy(flatStack, uOff, scratchU, 0, order + 1);
                System.arraycopy(flatStack, vOff, scratchV, 0, order + 1);
                SystemAutoDiffEvaluator.lnJet(scratchU, 0, scratch1, 0, order);
                SystemAutoDiffEvaluator.lnJet(scratchV, 0, scratch2, 0, order);
                SystemAutoDiffEvaluator.recipJet(scratch2, 0, scratch3, 0, order);
                SystemAutoDiffEvaluator.mul(scratch1, 0, scratch3, 0, flatStack, uOff, order);
                return;
            }
            default:
                throw new IllegalStateException(
                        "Unreachable: unresolved/unsupported two-argument opcode " + opcode
                        + " — this should have been caught by the constructor's flatten() validation.");
        }
    }

    // ------------------------------------------------------------------
    // asin/acos jets — SystemAutoDiffEvaluator has these inlined in its own
    // switch rather than factored into reusable statics; re-derived here in
    // that same factored form so asec/acsc can compose onto them (see
    // ASEC/ACSC cases above, which feed a reciprocal jet through these).
    // ------------------------------------------------------------------

    private static void asinJet(double[] u, int uOff, double[] out, int outOff, int ord,
                                 double[] scratch1, int s1Off, double[] scratch2, int s2Off,
                                 double[] scratch3, int s3Off) {
        if (Math.abs(u[uOff]) > 1.0) {
            throw new ArithmeticException("asin domain");
        }
        out[outOff] = Math.asin(u[uOff]);
        SystemAutoDiffEvaluator.mul(u, uOff, u, uOff, scratch1, s1Off, ord);
        scratch2[s2Off] = 1.0 - scratch1[s1Off];
        for (int k = 1; k <= ord; k++) {
            scratch2[s2Off + k] = -scratch1[s1Off + k];
        }
        SystemAutoDiffEvaluator.sqrtJet(scratch2, s2Off, scratch3, s3Off, ord);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int l = 1; l < k; l++) {
                s += l * out[outOff + l] * scratch3[s3Off + k - l];
            }
            out[outOff + k] = (u[uOff + k] - s / k) / scratch3[s3Off];
        }
    }

    private static void acosJet(double[] u, int uOff, double[] out, int outOff, int ord,
                                 double[] scratch1, int s1Off, double[] scratch2, int s2Off,
                                 double[] scratch3, int s3Off) {
        if (Math.abs(u[uOff]) > 1.0) {
            throw new ArithmeticException("acos domain");
        }
        out[outOff] = Math.acos(u[uOff]);
        SystemAutoDiffEvaluator.mul(u, uOff, u, uOff, scratch1, s1Off, ord);
        scratch2[s2Off] = 1.0 - scratch1[s1Off];
        for (int k = 1; k <= ord; k++) {
            scratch2[s2Off + k] = -scratch1[s1Off + k];
        }
        SystemAutoDiffEvaluator.sqrtJet(scratch2, s2Off, scratch3, s3Off, ord);
        for (int k = 1; k <= ord; k++) {
            double s = 0.0;
            for (int l = 1; l < k; l++) {
                s += l * out[outOff + l] * scratch3[s3Off + k - l];
            }
            out[outOff + k] = (-u[uOff + k] - s / k) / scratch3[s3Off];
        }
    }

    /** Multiplies every jet coefficient (order 0..ord) by a constant, in place — used for deg/grad scaling. */
    private static void scaleJetInPlace(double[] jet, int off, double factor, int ord) {
        for (int k = 0; k <= ord; k++) {
            jet[off + k] *= factor;
        }
    }

    private static double factorial(int n) {
        double f = 1.0;
        for (int i = 2; i <= n; i++) {
            f *= i;
        }
        return f;
    }

    // ===================================================================
    // Thread-local pre-allocated scratch state
    // ===================================================================
    private static final class AdEvalState {

        final double[] flatStack;
        final double[] scratch1, scratch2, scratch3, scratch4, scratchArg, scratchU, scratchV;
        final int currentMaxStackSize;
        final int currentMaxOrder;

        AdEvalState(int stackSize, int stride) {
            this.flatStack = new double[stackSize * stride];
            this.scratch1 = new double[stride];
            this.scratch2 = new double[stride];
            this.scratch3 = new double[stride];
            this.scratch4 = new double[stride];
            this.scratchArg = new double[stride];
            this.scratchU = new double[stride];
            this.scratchV = new double[stride];
            this.currentMaxStackSize = stackSize;
            this.currentMaxOrder = stride - 1;
        }
    }
}