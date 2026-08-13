package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard;

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.CoefficientExtractor;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.EquationCoefficientResolver;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.ResolvedEquation;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.ExprNodeAutoDiffEvaluator;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.CanonicalFrame;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.DiffEqnArgParser;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.DiffEqnCall;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.PostfixArgumentIsolator;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.refactor.ODESolverMethod;
import com.github.gbenroscience.math.differentialcalculus.equations.standard.DifferentialEquations;
import com.github.gbenroscience.math.differentialcalculus.equations.standard.HigherOrderODE;
import com.github.gbenroscience.math.differentialcalculus.equations.standard.ODEFunction;
import com.github.gbenroscience.math.differentialcalculus.equations.standard.TurboODE;
import com.github.gbenroscience.math.differentialcalculus.equations.standard.VectorODE;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;

import java.util.Arrays;

/**
 * ParserNG's single entry point for the diffeqn-family runtime: given the full
 * postfix of a diffeqn/diffeqnPath/diffeqnHO/diffeqnPathHO call, routes end to
 * end — argument parsing, equation isolation, coefficient extraction,
 * canonical-frame compilation, solver dispatch — and returns the result in
 * exactly the shape the calling convention promises.
 *
 * <h2>Calling convention (current)</h2>
 * The first argument to every one of the four entry points is the raw equation
 * itself, already rearranged to {@code LHS-RHS} with the {@code =} sign omitted
 * (implicitly {@code = 0}) — not a lambda. {@code y[0]} is the state itself,
 * {@code y[k]} its kth derivative, up through {@code y[n]} — the term this
 * class's coefficient resolver isolates and divides out.
 *
 * <h2>Canonical frame</h2>
 * Every solver call in this class uses the SAME canonical layout: slot 0 = t,
 * slots {@code 1..order} = {@code y[0..order-1]} — contiguous and ascending by
 * construction, regardless of what real, possibly-scattered slots ParserNG's
 * registry actually assigned to those names in the source text. See
 * {@link CanonicalFrame}.
 *
 * <h2>Analytic Jacobian for IMPLICIT_EULER</h2>
 * Built directly here rather than through {@code ExprNodeAnalyticJacobian}. For
 * the scalar case (order 1, {@code diffeqn}/{@code diffeqnPath}) it's a single
 * AD evaluator. For the higher-order case, the companion system's shift rows
 * ({@code dY_i/dt = Y_(i+1)}) have a trivially known derivative (1 for the one
 * matching column, 0 elsewhere) that needs no AD evaluator at all — routing
 * them through one anyway would require constructing an {@code ExprNode} leaf
 * for a state component that, in the sparse-equation case, may have no real
 * frame slot to build one from. Only the genuinely compiled top row goes
 * through {@code ExprNodeAutoDiffEvaluator}; the trivial rows are filled
 * directly.
 *
 * <h2>Known gap: genuine vector (non-HO) systems</h2>
 * A {@code diffeqn}/{@code diffeqnPath} call whose {@code y0} is a length-n
 * vector for a real system (n independent equations, not a higher-order
 * reduction) implies an equation producing n outputs — but the coefficient
 * extractor this class delegates to only ever isolates and divides ONE top
 * term. That case is not wired up: {@link #buildJacobianIfNeeded} returns null
 * for it (falls back to the solver's own finite-difference Jacobian) rather
 * than failing the call outright.
 *
 * <h2>Wired coefficient resolver</h2> {@link CoefficientExtractor} is the real
 * {@link EquationCoefficientResolver} implementation — term splitting,
 * linearity checking on the top-order term only, coefficient division, and the
 * canonical/real frame mapping are all live. {@link #solve(MathExpression)} is
 * the one-line entry point most callers actually want: given an already-parsed
 * diffeqn-family {@code
 * MathExpression}, it wires {@link CoefficientExtractor} in and returns the
 * result directly.
 */
public final class EquationRuntime {

    private final EquationCoefficientResolver coefficientResolver;

    public EquationRuntime(EquationCoefficientResolver coefficientResolver) {
        if (coefficientResolver == null) {
            throw new IllegalArgumentException("coefficientResolver must not be null");
        }
        this.coefficientResolver = coefficientResolver;
    }

    /**
     * One-call convenience: parses, isolates, extracts, compiles, and solves in
     * a single line, using the real {@link CoefficientExtractor}. Equivalent
     * to:
     * <pre>{@code
     * new EquationRuntime(CoefficientExtractor::resolve).execute(me.getCachedPostfix())
     * }</pre> This is the method to wire into {@code MathExpression}'s own {@code
     * solve()}/{@code solveGeneric()} — e.g. a one-line delegation:
     * {@code public Object solve() { return EquationRuntime.solve(this); } }
     * — so a ParserNG user only ever writes
     * {@code new MathExpression(call).solve()}.
     *
     * @param me an already-constructed {@code MathExpression} whose parsed
     * postfix is a diffeqn/diffeqnPath/diffeqnHO/diffeqnPathHO call
     */
    public static Object solve(MathExpression me) {
        return new EquationRuntime(CoefficientExtractor::resolve).execute(me.getCachedPostfix());
    }

    /**
     * Full pipeline for one call. Returns a {@code Double} for a scalar
     * {@code diffeqn} and for {@code diffeqnHO}'s y(tEnd), a {@code
     * double[]} for a vector {@code diffeqn}, or a {@code double[][]} for
     * either *_PATH variant — exactly the shape the calling convention promises
     * for that call kind.
     *
     * @param fullCallPostfix the WHOLE parsed postfix, with the diffeqn-family
     * call as its last token (the postfix root) — e.g. {@code
     *                        me.getCachedPostfix()} for a call parsed as the top-level statement
     */
    public Object execute(Token[] fullCallPostfix) {
        DiffEqnCall call = DiffEqnArgParser.parse(fullCallPostfix);
        Token[] equationPostfix = PostfixArgumentIsolator.isolateArgument(fullCallPostfix, 0);
        ResolvedEquation resolved = coefficientResolver.resolve(equationPostfix, call.y0.length);
        CanonicalFrame frame = new CanonicalFrame(resolved.canonicalToReal, resolved.realFrameSize);
        ODEFunction fn = new FrameRemapper(resolved.topDerivativeRealFrame, frame);

        int tSlot = 0;
        int ySlotStart = 1;
        int frameSize = 1 + call.y0.length;

        switch (call.kind) {
            case DIFFEQN:
                return executeDiffEqn(call, resolved, fn, frame, tSlot, ySlotStart, frameSize);
            case DIFFEQN_PATH:
                return executeDiffEqnPath(call, resolved, fn, frame, tSlot, ySlotStart, frameSize);
            case DIFFEQN_HO:
                return executeDiffEqnHO(call, resolved, fn, frame, tSlot, ySlotStart, frameSize);
            case DIFFEQN_PATH_HO:
                return executeDiffEqnPathHO(call, resolved, fn, frame, tSlot, ySlotStart, frameSize);
            default:
                throw new IllegalStateException("Unreachable");
        }
    }

    // ------------------------------------------------------------------
    // Kind-specific dispatch
    // ------------------------------------------------------------------
    private Object executeDiffEqn(DiffEqnCall call, ResolvedEquation resolved, ODEFunction fn, CanonicalFrame frame,
            int tSlot, int ySlotStart, int frameSize) {
        DifferentialEquations.JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, false);
        if (call.y0.length == 1) {
            return TurboODE.executeTurboODE(fn, tSlot, ySlotStart, frameSize,
                    call.t0, call.y0[0], call.tEnd, call.h, call.method, jac);
        }
        return VectorODE.executeVectorODE(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, jac);
    }

    private Object executeDiffEqnPath(DiffEqnCall call, ResolvedEquation resolved, ODEFunction fn, CanonicalFrame frame,
            int tSlot, int ySlotStart, int frameSize) {
        DifferentialEquations.JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, false);
        if (call.y0.length == 1) {
            return TurboODE.executeTurboODEPath(fn, tSlot, ySlotStart, frameSize,
                    call.t0, call.y0[0], call.tEnd, call.h, call.method, call.points, jac);
        }
        return VectorODE.executeVectorODEPath(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, call.points, jac);
    }

    private Object executeDiffEqnHO(DiffEqnCall call, ResolvedEquation resolved, ODEFunction fn, CanonicalFrame frame,
            int tSlot, int ySlotStart, int frameSize) {
        DifferentialEquations.JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, true);
        return HigherOrderODE.executeTurboODEHO(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, jac);
    }

    private Object executeDiffEqnPathHO(DiffEqnCall call, ResolvedEquation resolved, ODEFunction fn, CanonicalFrame frame,
            int tSlot, int ySlotStart, int frameSize) {
        DifferentialEquations.JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, true);
        return HigherOrderODE.executeTurboODEPathHO(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, call.points, jac);
    }

    // ------------------------------------------------------------------
    // Analytic Jacobian wiring
    // ------------------------------------------------------------------
    private DifferentialEquations.JacobianStrategy buildJacobianIfNeeded(
            DiffEqnCall call, ResolvedEquation resolved, CanonicalFrame frame, boolean higherOrder) {
        if (call.method != ODESolverMethod.IMPLICIT_EULER) {
            return null;
        }

        int order = call.y0.length;

        if (!higherOrder && order != 1) {
            // Genuine vector system — see class javadoc "Known gap". Falls back to the
            // solver's own finite-difference Jacobian rather than failing the call outright.
            return null;
        }

        ExprNodeAutoDiffEvaluator topRowEvaluator = new ExprNodeAutoDiffEvaluator(resolved.topDerivativeTree, 1);
        // canonical slot 1+k -> real frame slot of y[k]; NO_REAL_SLOT when y[k] never
        // appears in the equation text (sparse case) -> its true partial is exactly 0.
        int[] realStateSlots = new int[order];
        for (int k = 0; k < order; k++) {
            realStateSlots[k] = resolved.canonicalToReal[1 + k];
        }
        double[] scratch = new double[2];

        if (!higherOrder) {
            // order == 1 here (checked above): a single equation, single Jacobian entry.
            return (canonicalVars, outDfDy) -> {
                double[] realVars = frame.toReal(canonicalVars);
                if (realStateSlots[0] == CanonicalFrame.NO_REAL_SLOT) {
                    outDfDy[0][0] = 0.0;
                } else {
                    topRowEvaluator.taylorCoefficients(realVars, realStateSlots[0], 1, scratch);
                    outDfDy[0][0] = scratch[1];
                }
            };
        }

        return (canonicalVars, outDfDy) -> {
            double[] realVars = frame.toReal(canonicalVars);
            for (double[] row : outDfDy) {
                Arrays.fill(row, 0.0);
            }
            for (int i = 0; i < order - 1; i++) {
                outDfDy[i][i + 1] = 1.0; // companion shift rows: dY_i/dt = Y_(i+1)
            }
            for (int j = 0; j < order; j++) {
                int realSlot = realStateSlots[j];
                if (realSlot == CanonicalFrame.NO_REAL_SLOT) {
                    continue; // top row provably doesn't read Y[j] -> derivative is 0, already filled above
                }
                topRowEvaluator.taylorCoefficients(realVars, realSlot, 1, scratch);
                outDfDy[order - 1][j] = scratch[1];
            }
        };
    }

    // ------------------------------------------------------------------
    // Demo
    // ------------------------------------------------------------------
    /**
     * Wired to the real {@link CoefficientExtractor} — this now runs the
     * complete pipeline, not just up to the isolation step. Also fixes a bug
     * the extractor itself would otherwise have caught: the equation's highest
     * term is {@code y[4]}, so its order is 4 and {@code y0} needs 4 components
     * (the earlier {@code (1, 0, 0, 0, 0)} had 5, implying order 5 —
     * {@code y[5]} — which never appears in the equation at all).
     *
     * @param args
     */
    public static void main(String[] args) {

        String fullCall = "diffeqn((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.1, rk4)";
        MathExpression.EvalResult e = new MathExpression(fullCall).solveGeneric();

        System.out.println("res => " + e);

        System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

        fullCall = "diffeqnPath((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.1, rk4)";
        e = new MathExpression(fullCall).solveGeneric();

        System.out.println("res => " + e);

        System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

        fullCall = "diffeqn((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.1, rk4)";
        Object o = solve(new MathExpression(fullCall));
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

        System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
        fullCall = "diffeqn((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.00001, implicit_euler)";

        o = solve(new MathExpression(fullCall));
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

        System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
        fullCall = "diffeqn((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.01, rk45)";

        o = solve(new MathExpression(fullCall));
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

        System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
        fullCall = "diffeqnPathHO((3t^2)*y[4]+(5*sin(t))*y[3]+(5/t)*y[2]-3*y[1]+3*t*y[0], 1, @(1,4)(1, 0, 0, 0), 20, 0.0001, rk4)";

        o = solve(new MathExpression(fullCall));
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
