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
import com.github.gbenroscience.math.differentialcalculus.equations.standard.ODEFunction;
import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.EquationRuntime;
import com.github.gbenroscience.parser.MathExpression;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.Arrays;

/**
 * Compiles an {@link ExprNode} tree — typically
 * {@code CoefficientExtractor.Result.topDerivativeExpression}, or any single
 * coefficient — into something the solvers can actually call: an
 * {@link ODEFunction} for ParserNG Standard, or a {@link MethodHandle} for
 * Turbo. This is the piece that connects CoefficientExtractor's symbolic output
 * to VectorODE/HigherOrderODE/CompanionSystemHandles: those all expect a
 * callable matching the (double[] vars, double[] outDerivatives) convention,
 * and nothing before this class produced one from an ExprNode.
 *
 * <h2>Frame layout</h2>
 * The compiled callable reads t from {@code vars[tSlot]} and each state
 * component y[k] from {@code vars[ySlotStart + k]} — the exact same convention
 * every solver in this codebase already uses. The independent variable's
 * <em>name</em> (t, x, whatever the equation actually uses) is resolved once at
 * compile time; any other bare (non-indexed) identifier encountered while
 * compiling is a compile-time error, not a silent zero.
 *
 * <h2>Opcode-based dispatch — no string switch on the hot path</h2>
 * This mirrors the two-phase design {@code VectorTurboEvaluator} already uses
 * for its tiled Turbo pipeline: {@code compileToPrimitiveProgram} switches on
 * the function's {@code String} name exactly once, up front, producing an
 * {@code int} opcode array that {@code evaluateTile}'s hot loop then switches
 * on with no further string work. This class follows the identical shape, just
 * for the scalar (non-tiled) evaluator:
 * <ul>
 * <li>The one-time, string-comparing step is {@link FunctionOpcodes}, consulted
 * exactly once per node, inside {@link ExprNode#func} at tree-build time —
 * never during evaluation.</li>
 * <li>Every node it builds carries the resolved {@code int} opcode
 * ({@link ExprNode#funcOpcode}) from then on.</li>
 * <li>{@link #evaluate} / {@link #evaluateFrameOnly} — the methods actually
 * called once per solver step, potentially millions of times per solve — switch
 * on nothing but that {@code int}. There is no {@code String}, {@code .equals},
 * or {@code switch(String)} anywhere in this class's evaluation path.</li>
 * </ul>
 *
 * <h2>Supported functions</h2>
 * The full angular range {@code MethodSack} exposes — standard trig, inverse
 * trig, reciprocal trig, and inverse-reciprocal trig, each in
 * radians/degrees/gradians — plus the non-angular one-argument functions this
 * pipeline already needed ({@code sqrt, cbrt, exp, ln, lg, abs}) and the
 * two-argument functions ({@code atan2}, and {@code log} with an explicit
 * base). See {@link FunctionOpcodes} for the exact alias-to-opcode table (it
 * lists every ParserNG spelling — {@code sind}, {@code arcsin},
 * {@code sin-¹}, {@code asin_grad}, etc. — that resolves to each one).
 * <p>
 * {@code MethodSack} itself only exposes batch, in-place {@code void} mutators
 * over a {@code double[]} tile (and the trivial {@code if3} blend) — not usable
 * here, and not needed here: this class only ever evaluates one scalar
 * coefficient at a time. {@link #evaluateOneArgFunction} below reproduces each
 * {@code MethodSack} formula exactly (same
 * DEG_TO_RAD/RAD_TO_DEG/GRAD_TO_RAD/RAD_TO_GRAD constants, same order of
 * operations), so results agree bit-for-bit with what the Turbo tiled path
 * computes. {@code asinh}/{@code acosh}/{@code atanh} have no
 * {@code MethodSack} counterpart at all (only forward hyperbolics do), so those
 * three call straight into {@link Maths#asinh}, {@link Maths#acosh},
 * {@link Maths#atanh}.
 * <p>
 * Anything unresolved (an unsupported name, or a name used at an arity it
 * doesn't support) throws at compile time — in {@link #validate} or
 * {@link #requireFullyFrameIndexed} — naming the function, rather than failing
 * at solve time with a less specific error.
 *
 * <h2>Automatic differentiation lives in a sibling class</h2>
 * This class produces a plain numeric evaluator — no derivatives. Exact
 * derivatives over the same ExprNode tree (needed for an implicit-method
 * Jacobian) are {@link ExprNodeAutoDiffEvaluator}'s job, via real forward-mode
 * AD jet arithmetic, not finite differences — {@link ExprNodeAnalyticJacobian}
 * is what wires a set of those into a
 * {@code DifferentialEquations.JacobianStrategy}. Neither of those is folded
 * into this class: this one only ever produces the plain value-returning
 * callable VectorODE/HigherOrderODE call at every solver step; the Jacobian,
 * when one is wanted, is a separate, explicit opt-in via JacobianStrategy —
 * DifferentialEquations' finite-difference default remains what runs if no
 * JacobianStrategy is supplied at all, but that is no longer the only option
 * for an equation compiled through this pipeline.
 */
public final class ExprNodeCompiler {

    private ExprNodeCompiler() {
    }

    // ------------------------------------------------------------------
    // ParserNG Standard: ExprNode -> ODEFunction
    // ------------------------------------------------------------------
    /**
     * @param expression the expression to compile — must reference only the
     * independent variable (by name), state leaves y[k] with k in [0,
     * expectedStateCount), and constants
     * @param independentVariableName the one free-variable name this expression
     * may reference (e.g. "t" or "x"); any other bare identifier is a
     * compile-time error
     * @param tSlot frame index holding the independent variable
     * @param ySlotStart frame index where the state block starts
     */
    public static ODEFunction compileStandard(ExprNode expression, String independentVariableName,
            int tSlot, int ySlotStart) {
        validate(expression, independentVariableName);
        return (vars, out) -> out[0] = evaluate(expression, independentVariableName, tSlot, ySlotStart, vars);
    }

    /**
     * For a tree built via {@link TokenTreeBuilder}, where every leaf already
     * carries a real, registry-assigned frameIndex — no independent-variable
     * name or tSlot/ySlotStart bookkeeping needed at all. Throws if any leaf
     * turns out not to have one, rather than silently falling back to an
     * undefined position.
     *
     * @param expression
     * @return
     */
    public static ODEFunction compileStandard(ExprNode expression) {
        requireFullyFrameIndexed(expression);
        return (vars, out) -> out[0] = evaluateFrameOnly(expression, vars);
    }

    // ------------------------------------------------------------------
    // Turbo: ExprNode -> MethodHandle
    // ------------------------------------------------------------------
    private static final MethodType CALLABLE_TYPE
            = MethodType.methodType(void.class, double[].class, double[].class);

    public static MethodHandle compileTurbo(ExprNode expression, String independentVariableName,
            int tSlot, int ySlotStart) {
        validate(expression, independentVariableName);
        CompiledAdapter adapter = new CompiledAdapter(expression, independentVariableName, tSlot, ySlotStart);
        try {
            return MethodHandles.lookup().bind(adapter, "apply", CALLABLE_TYPE);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException("Failed to bind compiled-expression adapter", e);
        }
    }

    /**
     * Frame-complete counterpart to {@link #compileStandard(ExprNode)}, for the
     * Turbo/MethodHandle tier.
     *
     * @param expression
     * @return
     */
    public static MethodHandle compileTurbo(ExprNode expression) {
        requireFullyFrameIndexed(expression);
        FrameOnlyAdapter adapter = new FrameOnlyAdapter(expression);
        try {
            return MethodHandles.lookup().bind(adapter, "apply", CALLABLE_TYPE);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException("Failed to bind compiled-expression adapter", e);
        }
    }

    private static final class FrameOnlyAdapter {

        private final ExprNode expression;

        FrameOnlyAdapter(ExprNode expression) {
            this.expression = expression;
        }

        // Signature must exactly match (double[], double[])void for invokeExact/bind.
        void apply(double[] vars, double[] out) {
            out[0] = evaluateFrameOnly(expression, vars);
        }
    }

    private static final class CompiledAdapter {

        private final ExprNode expression;
        private final String independentVariableName;
        private final int tSlot;
        private final int ySlotStart;

        CompiledAdapter(ExprNode expression, String independentVariableName, int tSlot, int ySlotStart) {
            this.expression = expression;
            this.independentVariableName = independentVariableName;
            this.tSlot = tSlot;
            this.ySlotStart = ySlotStart;
        }

        // Signature must exactly match (double[], double[])void for invokeExact/bind.
        void apply(double[] vars, double[] out) {
            out[0] = evaluate(expression, independentVariableName, tSlot, ySlotStart, vars);
        }
    }

    // ------------------------------------------------------------------
    // Angular conversion constants — copied verbatim from MethodSack so
    // results agree bit-for-bit with the Turbo tiled path.
    // ------------------------------------------------------------------
    private static final double DEG_TO_RAD = Math.PI / 180.0;
    private static final double RAD_TO_DEG = 180.0 / Math.PI;
    private static final double GRAD_TO_RAD = Math.PI / 200.0;
    private static final double RAD_TO_GRAD = 200.0 / Math.PI;

    // ------------------------------------------------------------------
    // Shared: compile-time validation and evaluation core
    // ------------------------------------------------------------------
    /**
     * Walks the tree once up front confirming every VARIABLE leaf carries a
     * real frameIndex (>= 0) — used by the frame-complete overloads, where
     * there is no independentVariableName/tSlot/ySlotStart fallback to catch a
     * leaf that somehow wasn't bound to a real registry slot.
     */
    private static void requireFullyFrameIndexed(ExprNode node) {
        switch (node.kind) {
            case NUMBER:
                return;
            case VARIABLE:
                if (node.frameIndex < 0) {
                    throw new IllegalArgumentException(
                            "Variable '" + node.variableName + "' has no real frameIndex — the frame-complete "
                            + "compile methods require every leaf to come from TokenTreeBuilder (or otherwise "
                            + "carry a real registry slot); use the (expression, independentVariableName, "
                            + "tSlot, ySlotStart) overload instead for hand-built trees.");
                }
                return;
            case OP:
                if (node.isFunctionCall()) {
                    requireSupportedFunction(node);
                }
                for (ExprNode child : node.children) {
                    requireFullyFrameIndexed(child);
                }
                return;
            default:
                throw new IllegalStateException("Unreachable");
        }
    }

    private static double evaluateFrameOnly(ExprNode node, double[] vars) {
        switch (node.kind) {
            case NUMBER:
                return node.numberValue;
            case VARIABLE:
                return vars[node.frameIndex];
            case OP: {
                if (node.isFunctionCall()) {
                    double a = evaluateFrameOnly(node.children.get(0), vars);
                    if (node.children.size() == 1) {
                        return evaluateOneArgFunction(node.funcOpcode, a);
                    }
                    double b = evaluateFrameOnly(node.children.get(1), vars);
                    return evaluateTwoArgFunction(node.funcOpcode, a, b);
                }
                if (node.children.size() == 1) {
                    return -evaluateFrameOnly(node.children.get(0), vars);
                }
                double x = evaluateFrameOnly(node.children.get(0), vars);
                double y = evaluateFrameOnly(node.children.get(1), vars);
                return evaluateBinaryOp(node.opChar, x, y);
            }
            default:
                throw new IllegalStateException("Unreachable");
        }
    }

    // ------------------------------------------------------------------
    // One-argument function dispatch — switch(int) only, opcode already
    // resolved by FunctionOpcodes at ExprNode-construction time.
    // ------------------------------------------------------------------
    private static double evaluateOneArgFunction(int opcode, double a) {
        switch (opcode) {
            // non-angular
            case FunctionOpcodes.SQRT:
                return Math.sqrt(a);
            case FunctionOpcodes.CBRT:
                return Math.cbrt(a);
            case FunctionOpcodes.EXP:
                return Math.exp(a);
            case FunctionOpcodes.LN:
                return Math.log(a);
            case FunctionOpcodes.LOG10:
                return Math.log10(a);
            case FunctionOpcodes.ABS:
                return Math.abs(a);

            // hyperbolic (forward)
            case FunctionOpcodes.SINH:
                return Math.sinh(a);
            case FunctionOpcodes.COSH:
                return Math.cosh(a);
            case FunctionOpcodes.TANH:
                return Math.tanh(a);

            // hyperbolic (inverse) — no MethodSack counterpart, fall back to Maths
            case FunctionOpcodes.ASINH:
                return Maths.asinh(a);
            case FunctionOpcodes.ACOSH:
                return Maths.acosh(a);
            case FunctionOpcodes.ATANH:
                return Maths.atanh(a);

            // standard trig — MethodSack scales the angle INPUT before Math.sin/cos/tan
            case FunctionOpcodes.SIN:
                return Math.sin(a);
            case FunctionOpcodes.SIN_DEG:
                return Math.sin(a * DEG_TO_RAD);
            case FunctionOpcodes.SIN_GRAD:
                return Math.sin(a * GRAD_TO_RAD);
            case FunctionOpcodes.COS:
                return Math.cos(a);
            case FunctionOpcodes.COS_DEG:
                return Math.cos(a * DEG_TO_RAD);
            case FunctionOpcodes.COS_GRAD:
                return Math.cos(a * GRAD_TO_RAD);
            case FunctionOpcodes.TAN:
                return Math.tan(a);
            case FunctionOpcodes.TAN_DEG:
                return Math.tan(a * DEG_TO_RAD);
            case FunctionOpcodes.TAN_GRAD:
                return Math.tan(a * GRAD_TO_RAD);

            // inverse trig — MethodSack scales the RESULT (radians -> deg/grad) after Math.asin/acos/atan
            case FunctionOpcodes.ASIN:
                return Math.asin(a);
            case FunctionOpcodes.ASIN_DEG:
                return Math.asin(a) * RAD_TO_DEG;
            case FunctionOpcodes.ASIN_GRAD:
                return Math.asin(a) * RAD_TO_GRAD;
            case FunctionOpcodes.ACOS:
                return Math.acos(a);
            case FunctionOpcodes.ACOS_DEG:
                return Math.acos(a) * RAD_TO_DEG;
            case FunctionOpcodes.ACOS_GRAD:
                return Math.acos(a) * RAD_TO_GRAD;
            case FunctionOpcodes.ATAN:
                return Math.atan(a);
            case FunctionOpcodes.ATAN_DEG:
                return Math.atan(a) * RAD_TO_DEG;
            case FunctionOpcodes.ATAN_GRAD:
                return Math.atan(a) * RAD_TO_GRAD;

            // reciprocal trig — MethodSack scales the angle input, then reciprocates
            case FunctionOpcodes.SEC:
                return 1.0 / Math.cos(a);
            case FunctionOpcodes.SEC_DEG:
                return 1.0 / Math.cos(a * DEG_TO_RAD);
            case FunctionOpcodes.SEC_GRAD:
                return 1.0 / Math.cos(a * GRAD_TO_RAD);
            case FunctionOpcodes.CSC:
                return 1.0 / Math.sin(a);
            case FunctionOpcodes.CSC_DEG:
                return 1.0 / Math.sin(a * DEG_TO_RAD);
            case FunctionOpcodes.CSC_GRAD:
                return 1.0 / Math.sin(a * GRAD_TO_RAD);
            case FunctionOpcodes.COT:
                return 1.0 / Math.tan(a);
            case FunctionOpcodes.COT_DEG:
                return 1.0 / Math.tan(a * DEG_TO_RAD);
            case FunctionOpcodes.COT_GRAD:
                return 1.0 / Math.tan(a * GRAD_TO_RAD);

            // inverse reciprocal trig — MethodSack's own formula: acos/asin/atan(1/a), result scaled after
            case FunctionOpcodes.ASEC:
                return Math.acos(1.0 / a);
            case FunctionOpcodes.ASEC_DEG:
                return Math.acos(1.0 / a) * RAD_TO_DEG;
            case FunctionOpcodes.ASEC_GRAD:
                return Math.acos(1.0 / a) * RAD_TO_GRAD;
            case FunctionOpcodes.ACSC:
                return Math.asin(1.0 / a);
            case FunctionOpcodes.ACSC_DEG:
                return Math.asin(1.0 / a) * RAD_TO_DEG;
            case FunctionOpcodes.ACSC_GRAD:
                return Math.asin(1.0 / a) * RAD_TO_GRAD;
            case FunctionOpcodes.ACOT:
                return Math.atan(1.0 / a);
            case FunctionOpcodes.ACOT_DEG:
                return Math.atan(1.0 / a) * RAD_TO_DEG;
            case FunctionOpcodes.ACOT_GRAD:
                return Math.atan(1.0 / a) * RAD_TO_GRAD;

            default:
                throw new IllegalStateException(
                        "Unreachable: unresolved/unsupported one-argument opcode " + opcode
                        + " — this should have been caught by validate()/requireFullyFrameIndexed() at compile time.");
        }
    }

    private static double evaluateTwoArgFunction(int opcode, double a, double b) {
        switch (opcode) {
            case FunctionOpcodes.ATAN2:
                return Math.atan2(a, b);
            case FunctionOpcodes.LOG_BASE:
                return Math.log(a) / Math.log(b);
            default:
                throw new IllegalStateException(
                        "Unreachable: unresolved/unsupported two-argument opcode " + opcode
                        + " — this should have been caught by validate()/requireFullyFrameIndexed() at compile time.");
        }
    }

    private static double evaluateBinaryOp(char opChar, double x, double y) {
        switch (opChar) {
            case '+':
                return x + y;
            case '-':
                return x - y;
            case '*':
                return x * y;
            case '/':
                return x / y;
            case '^':
                return Math.pow(x, y);
            default:
                throw new IllegalStateException("Unreachable: unsupported opChar " + opChar);
        }
    }

    /**
     * Walks the tree once up front so an unbound variable or unsupported
     * function fails at compile time.
     */
    private static void validate(ExprNode node, String independentVariableName) {
        switch (node.kind) {
            case NUMBER:
                return;
            case VARIABLE:
                if (node.isStateVariable()) {
                    return;
                }
                if (node.frameIndex >= 0) {
                    return; // real, registry-assigned slot — no name check needed
                }
                if (!node.variableName.equals(independentVariableName)) {
                    throw new IllegalArgumentException(
                            "Unbound variable '" + node.variableName + "' — this expression may only reference "
                            + "the independent variable '" + independentVariableName + "', state leaves, "
                            + "and constants.");
                }
                return;
            case OP:
                if (node.isFunctionCall()) {
                    requireSupportedFunction(node);
                }
                for (ExprNode child : node.children) {
                    validate(child, independentVariableName);
                }
                return;
            default:
                throw new IllegalStateException("Unreachable");
        }
    }

    /**
     * The only place in this class that ever looks at a function node's name as
     * a String — and only on the failure path, to build an error message.
     * Whether the function is supported is decided purely by
     * {@code node.funcOpcode >= 0}, a value FunctionOpcodes already resolved
     * once at tree-build time; there is no string comparison here, just an int
     * check.
     */
    private static void requireSupportedFunction(ExprNode node) {
        if (node.funcOpcode == FunctionOpcodes.UNRESOLVED) {
            throw new IllegalArgumentException(
                    "Unsupported function '" + node.funcName + "' with " + node.children.size()
                    + " argument(s) — see FunctionOpcodes for the full list of supported names and arities.");
        }
    }

    private static double evaluate(ExprNode node, String independentVariableName,
            int tSlot, int ySlotStart, double[] vars) {
        switch (node.kind) {
            case NUMBER:
                return node.numberValue;
            case VARIABLE:
                if (node.frameIndex >= 0) {
                    return vars[node.frameIndex]; // real, registry-assigned slot — use it directly
                }
                return node.isStateVariable() ? vars[ySlotStart + node.stateIndex] : vars[tSlot];
            case OP: {
                if (node.isFunctionCall()) {
                    double a = evaluate(node.children.get(0), independentVariableName, tSlot, ySlotStart, vars);
                    if (node.children.size() == 1) {
                        return evaluateOneArgFunction(node.funcOpcode, a);
                    }
                    double b = evaluate(node.children.get(1), independentVariableName, tSlot, ySlotStart, vars);
                    return evaluateTwoArgFunction(node.funcOpcode, a, b);
                }
                if (node.children.size() == 1) {
                    return -evaluate(node.children.get(0), independentVariableName, tSlot, ySlotStart, vars);
                }
                double x = evaluate(node.children.get(0), independentVariableName, tSlot, ySlotStart, vars);
                double y = evaluate(node.children.get(1), independentVariableName, tSlot, ySlotStart, vars);
                return evaluateBinaryOp(node.opChar, x, y);
            }
            default:
                throw new IllegalStateException("Unreachable");
        }
    }

    public static void main(String[] args) {
        // Corrected per feedback: MathExpression only vets y[k]-style names
        // inside one of the four recognized call forms (diffeqn/diffeqnPath/
        // diffeqnHO/diffeqnPathHO) — it cannot parse the bare equation on its
        // own, which is why an earlier version of this demo crashed. The fix
        // is exactly what was specified: always wrap the equation in the call,
        // never parse it standalone. And since ParserNG runs end to end, this
        // demo no longer stops at "compiled but unsolved" — it goes through
        // DiffEqnCallRunner, which isolates the equation from the full call's
        // own postfix (one parse, one consistent registry — no second,
        // independent re-parse), pulls t0/y0/tEnd/h/method from the root call
        // token's getRawArgs(), and actually drives the solve.
        //
        // y0 is written as @(1,4)(1,0,0,0) — a 1x4 array literal, ParserNG's
        // real syntax for an array-valued argument — not the bare (1,0,0,0)
        // tuple an earlier version of this demo incorrectly used. ParserNG
        // resolves that literal during scanning into an anonXXX placeholder
        // (see DiffEqnCallRunner.resolveArrayArgument), which is what
        // getRawArgs() actually returns for this argument.
        String call = "diffeqnPathHO((3*t^2)*y[4]+(5*sin(t))*sin(y[3])+(5/t)*cos(y[2])-3*y[1]+3*t*y[0], 10, @(1,4)(1,0,0,0), 30, 0.01, euler)";

        Object o = EquationRuntime.solve(new MathExpression(call));
        if (o instanceof double[][]) {
            double[][] mat = (double[][]) o;
            for (double[] d : mat) {
                System.out.println("res ==> " + Arrays.toString(d));
            }
        } else if (o instanceof double[]) {
            System.out.println("res => " + Arrays.toString(((double[]) o)));
        } else {
            System.out.println("res -> " + ((o == null) ? "null" : o.toString()));
        }
    }
}
