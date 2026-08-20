package com.github.gbenroscience.math.differentialcalculus.equations.standard;

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.ExprNodeAutoDiffEvaluator;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.CanonicalFrame;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.DiffEqnArgParser;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.DiffEqnCall;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.JacobianStrategy;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.PostfixArgumentIsolator;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.ODESolverMethod;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.PresentationStrategy;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.CoefficientExtractor;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.EquationCoefficientResolver;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.FrameRemapper;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.ResolvedEquation;
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.ParserResult;
import com.github.gbenroscience.util.FunctionManager;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * ParserNG's single entry point for the diffeqn-family runtime: given the full
 * postfix of a diffeqn/diffeqnPath/diffeqnHO/diffeqnPathHO call, routes end to
 * end — argument parsing, equation isolation, coefficient extraction,
 * canonical-frame compilation, solver dispatch — and returns the result in
 * exactly the shape the calling convention promises.
 *
 * <h2>Calling convention (current)</h2>
 * Argument 0 is either the raw equation itself, already rearranged to
 * {@code LHS-RHS} with the {@code =} sign omitted (implicitly {@code = 0}), or
 * — for an explicit system — {@code @(n)("eq1", ..., "eqN")}, an ARRAY-kind
 * literal of N such equations, one per state component. {@code
 * y[0]} is the state itself, {@code y[k]} its kth derivative; for a single
 * equation the divided-out term is {@code y[order]}; for a system, every
 * equation divides out {@code y[n]} where n == the system's component count (==
 * y0.length), constant across the whole system. See {@link
 * #executeSystem} for how the array form is compiled.
 *
 * <h2>Canonical frame</h2>
 * Every solver call in this class uses the SAME canonical layout: slot 0 = t,
 * slots {@code 1..order} = {@code y[0..order-1]} — contiguous and ascending by
 * construction, regardless of what real, possibly-scattered slots ParserNG's
 * registry actually assigned to those names in the source text. See
 * {@link CanonicalFrame}. For an explicit system, each equation is
 * independently parsed (its own fresh MathExpression, its own registry) and so
 * gets its own independent CanonicalFrame — there is no shared/merged frame
 * across a system's equations, and none is needed, since the canonical indices
 * (0 = t, 1+k = y[k]) are a fixed convention rather than anything
 * registry-derived.
 *
 * <h2>Analytic Jacobian for IMPLICIT_EULER / BDF2</h2>
 * Built directly here rather than through {@code ExprNodeAnalyticJacobian}
 * (whose shared-single-frame batching doesn't fit an explicit system, where
 * every equation has its own independent frame). For the scalar case (order 1)
 * it's a single AD evaluator. For the higher-order case, the companion system's
 * shift rows ({@code dY_i/dt = Y_(i+1)}) have a trivially known derivative (1
 * for the one matching column, 0 elsewhere) that needs no AD evaluator at all.
 * For an explicit system, {@link #buildSystemJacobianIfNeeded} builds the
 * Jacobian row-by-row, translating canonicalVars into each row's own real frame
 * before differentiating that row's own compiled tree.
 *
 * <h2>Wired coefficient resolver</h2> {@link CoefficientExtractor} is the real
 * {@link EquationCoefficientResolver} implementation — term splitting,
 * linearity checking on the top-order term only, coefficient division, and the
 * canonical/real frame mapping are all live, and is reused unchanged for every
 * equation in an explicit system (called with order == systemSize for each).
 * {@link #solve(MathExpression)} is the one-line entry point most callers
 * actually want: given an already-parsed diffeqn-family {@code
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
     * @return
     */
    public static MathExpression.EvalResult solve(MathExpression me) throws Throwable {
        return solve(me.getCachedPostfix(), me.getNextResult());
    }

    /**
     * One-call convenience, the Turbo-tier twin of {@link
     * EquationRuntime#solve(Token[]postfix)}. Equivalent to:
     * <pre>{@code
     * new EquationRuntime(CoefficientExtractor::resolve).execute(postfix)
     * }</pre>
     *
     * @param postfix
     * @param out
     * @return
     */
    public static MathExpression.EvalResult solve(MathExpression.Token[] postfix, MathExpression.EvalResult out) throws Throwable{
        try {
            Object o = new EquationRuntime(CoefficientExtractor::resolve).execute(postfix);
            if (o instanceof double[][]) {
                out.wrap(new Matrix((double[][]) o));
            } else if (o instanceof double[]) {
                out.wrap((double[]) o);
            } else {
                out.wrap((double) o);
            }
        } catch (Throwable ex) {
            Logger.getLogger(MathExpression.class.getName()).log(Level.SEVERE, null, ex);
            out.wrap(ParserResult.INVALID_FUNCTION);
            throw ex;
        }
        return out;
    }

    /**
     * Full pipeline for one call. Returns a {@code Double} for a scalar
     * {@code diffeqn} and for {@code diffeqnHO}'s y(tEnd), a {@code
     * double[]} for a vector {@code diffeqn} (or an explicit system's endpoint
     * state), or a {@code double[][]} for either *_PATH variant — exactly the
     * shape the calling convention promises for that call kind.
     *
     * @param fullCallPostfix the WHOLE parsed postfix, with the diffeqn-family
     * call as its last token (the postfix root) — e.g. {@code
     *                        me.getCachedPostfix()} for a call parsed as the top-level statement
     * @return
     */
    public Object execute(Token[] fullCallPostfix) {
        DiffEqnCall call = DiffEqnArgParser.parse(fullCallPostfix);

        if (call.equationArraySyntax) {
            return executeSystem(call);
        }

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
                return executeDiffEqnPathHO(call, resolved, fn, frame, tSlot, ySlotStart, frameSize, call.presentationStrategy);
            default:
                throw new IllegalStateException("Unreachable");
        }
    }

    // ------------------------------------------------------------------
    // Kind-specific dispatch (single equation / HO — unchanged)
    // ------------------------------------------------------------------
    private Object executeDiffEqn(DiffEqnCall call, ResolvedEquation resolved, ODEFunction fn, CanonicalFrame frame,
            int tSlot, int ySlotStart, int frameSize) {
        JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, false);
        if (call.y0.length == 1) {
            return TurboODE.executeTurboODE(fn, tSlot, ySlotStart, frameSize,
                    call.t0, call.y0[0], call.tEnd, call.h, call.method, jac);
        }
        return VectorODE.executeVectorODE(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, jac);
    }

    private Object executeDiffEqnPath(DiffEqnCall call, ResolvedEquation resolved, ODEFunction fn, CanonicalFrame frame,
            int tSlot, int ySlotStart, int frameSize) {
        JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, false);
        if (call.y0.length == 1) {
            return TurboODE.executeTurboODEPath(fn, tSlot, ySlotStart, frameSize,
                    call.t0, call.y0[0], call.tEnd, call.h, call.method, call.points, jac);
        }
        return VectorODE.executeVectorODEPath(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, call.points, jac);
    }

    private Object executeDiffEqnHO(DiffEqnCall call, ResolvedEquation resolved, ODEFunction fn, CanonicalFrame frame,
            int tSlot, int ySlotStart, int frameSize) {
        JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, true);
        return HigherOrderODE.executeTurboODEHO(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, jac);
    }

    private Object executeDiffEqnPathHO(DiffEqnCall call, ResolvedEquation resolved, ODEFunction fn, CanonicalFrame frame,
            int tSlot, int ySlotStart, int frameSize, PresentationStrategy ps) {
        JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, true);
        return HigherOrderODE.executeTurboODEPathHO(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, call.points, jac, ps);
    }

    // ------------------------------------------------------------------
    // Explicit system dispatch — diffeqn/diffeqnPath with @(n)("eq1",...) form.
    // Every equation resolves via the SAME coefficientResolver.resolve(...)
    // used for single equations, called with order = n (the system size) for
    // EVERY equation — each equation's own text divides out the symbol y[n]
    // (n constant across the whole system), independently parsed, so there's
    // no shared-frame problem to solve: each equation just gets its own
    // CanonicalFrame, same as the single-equation path always has.
    // ------------------------------------------------------------------
    private Object executeSystem(DiffEqnCall call) {
        int n = call.equationTexts.length;

        ResolvedEquation[] resolved = new ResolvedEquation[n];
        CanonicalFrame[] frames = new CanonicalFrame[n];
        ODEFunction[] perEquationFn = new ODEFunction[n];

// standard tier, executeSystem:
        for (int i = 0; i < n; i++) {
            // Wrapped in a synthetic diffeqn(...) call and isolated via
            // PostfixArgumentIsolator (not parsed bare) so "y[k]" gets the same
            // protected, symbolic-only parsing treatment the classic single-
            // equation form already relies on. A bare standalone MathExpression
            // has no diffeqn-argument context, so ParserNG tries to resolve "y"
            // as a real registered name and fails outright (empty/null postfix)
            // rather than treating y[k] as an opaque state-variable leaf.
            MathExpression synthetic = new MathExpression("diffeqn(" + call.equationTexts[i] + ", 0, 0, 1)");
            Token[] eqPostfix = PostfixArgumentIsolator.isolateArgument(synthetic.getCachedPostfix(), 0);
            resolved[i] = coefficientResolver.resolve(eqPostfix, n);
            frames[i] = new CanonicalFrame(resolved[i].canonicalToReal, resolved[i].realFrameSize);
            perEquationFn[i] = new FrameRemapper(resolved[i].topDerivativeRealFrame, frames[i]);
        }

        ODEFunction fn = SystemFunctionHandles.buildSystem(perEquationFn, n);

        int tSlot = 0;
        int ySlotStart = 1;
        int frameSize = 1 + n;

        JacobianStrategy jac = buildSystemJacobianIfNeeded(call, resolved, frames);

        switch (call.kind) {
            case DIFFEQN:
                return VectorODE.executeVectorODE(fn, tSlot, ySlotStart, frameSize,
                        call.t0, call.y0, call.tEnd, call.h, call.method, jac);
            case DIFFEQN_PATH:
                return VectorODE.executeVectorODEPath(fn, tSlot, ySlotStart, frameSize,
                        call.t0, call.y0, call.tEnd, call.h, call.method, call.points, jac);
            default:
                // DIFFEQN_HO / DIFFEQN_PATH_HO already rejected in DiffEqnArgParser.parse.
                throw new IllegalStateException("Unreachable");
        }
    }

    /**
     * Per-row Jacobian for an explicit system: each equation carries its own
     * independent CanonicalFrame (see executeSystem), so this translates
     * canonicalVars into row i's OWN real frame before differentiating row i's
     * own compiled tree — ExprNodeAnalyticJacobian's single-shared-frame
     * batching doesn't apply here.
     */
    private JacobianStrategy buildSystemJacobianIfNeeded(
            DiffEqnCall call, ResolvedEquation[] resolved, CanonicalFrame[] frames) {
        if (call.method != ODESolverMethod.IMPLICIT_EULER && call.method != ODESolverMethod.BDF2) {
            return null;
        }

        int n = resolved.length;
        ExprNodeAutoDiffEvaluator[] rowEvaluators = new ExprNodeAutoDiffEvaluator[n];
        for (int i = 0; i < n; i++) {
            rowEvaluators[i] = new ExprNodeAutoDiffEvaluator(resolved[i].topDerivativeTree, 1);
        }
        double[] scratch = new double[2];

        return (canonicalVars, outDfDy) -> {
            for (int row = 0; row < n; row++) {
                double[] realVarsForRow = frames[row].toReal(canonicalVars);
                for (int col = 0; col < n; col++) {
                    int realSlot = resolved[row].canonicalToReal[1 + col];
                    if (realSlot == CanonicalFrame.NO_REAL_SLOT) {
                        outDfDy[row][col] = 0.0;
                    } else {
                        rowEvaluators[row].taylorCoefficients(realVarsForRow, realSlot, 1, scratch);
                        outDfDy[row][col] = scratch[1];
                    }
                }
            }
        };
    }

    // ------------------------------------------------------------------
    // Analytic Jacobian wiring (single equation / HO — unchanged)
    // ------------------------------------------------------------------
    private JacobianStrategy buildJacobianIfNeeded(
            DiffEqnCall call, ResolvedEquation resolved, CanonicalFrame frame, boolean higherOrder) {
        if (call.method != ODESolverMethod.IMPLICIT_EULER) {
            return null;
        }

        int order = call.y0.length;

        if (!higherOrder && order != 1) {
            // Genuine vector system without the array syntax shouldn't reach here —
            // DiffEqnArgParser only produces equationArraySyntax==false for a
            // single-equation call, which always has order == 1 for this branch.
            return null;
        }

        ExprNodeAutoDiffEvaluator topRowEvaluator = new ExprNodeAutoDiffEvaluator(resolved.topDerivativeTree, 1);
        int[] realStateSlots = new int[order];
        for (int k = 0; k < order; k++) {
            realStateSlots[k] = resolved.canonicalToReal[1 + k];
        }
        double[] scratch = new double[2];

        if (!higherOrder) {
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
                    continue;
                }
                topRowEvaluator.taylorCoefficients(realVars, realSlot, 1, scratch);
                outDfDy[order - 1][j] = scratch[1];
            }
        };
    }

    // ------------------------------------------------------------------
    // Demo
    // ------------------------------------------------------------------
    public static void main(String[] args) throws Throwable {

        runODE("diffeqnPathHO(y[2]+3*y[1]-sin(x)*y[0]+3*x^2, 3, @(1,2)(1,0.5), 10, 0.001, rk4, trajectory)");
        runODE("diffeqn((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.01, rk4)");
        runODE("diffeqnPath((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.1, rk4)");
        runODE("diffeqn((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.01, implicit_euler)");
        runODE("diffeqn((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.01, rk45)");
        runODE("diffeqnPathHO((3t^2)*y[4]+(5*sin(t))*y[3]+(5/t)*y[2]-3*y[1]+3*t*y[0], 1, @(1,4)(1, 0, 0, 0), 20, 0.01, rk4)");
        runODE("diffeqnPathHO(3*x*sin(x)*y[3]+4*x*y[2]+3*ln(x)*y[1]+4*y[0], 1, @(1,3)(1, 0, 0), 3, 0.01, rk45)");
        runODE("diffeqnPathHO(3*x*sin(x)*y[3]+4*x*y[2]+3*ln(x)*y[1]+4*y[0], 1, @(1,3)(1, 0, 0), 3, 0.01, bdf2, state)");

        // NEW: explicit system, Lotka-Volterra — each equation divides out y[2]
        // (n = 2, the system's own component count), independently parsed:
        runODE("diffeqn(@(2)(\"y[2]-(0.6*y[0]-0.03*y[0]*y[1])\", \"y[2]-(-0.9*y[1]+0.02*y[0]*y[1])\"), 0, @(1,2)(30, 4), 20, 0.01, rk4)");
         
        runODE("diffeqnHO(y[2] + (9.81/2.5)*sin(y[0]), 0, @(1,2)(0.5, 0.5), 30, 0.0001, rk4)");

        MathExpression me = new MathExpression("A=diffeqnPathHO(3*x*sin(x)*y[3]+4*x*y[2]+3*ln(x)*y[1]+4*y[0], 1, @(1,3)(1, 0, 0), 3, 0.01, bdf2, state)");
        me.solve();
        FunctionManager.lookUp("A").getMatrix().print();

        MathExpression m = new MathExpression("b=diffeqn((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.01, rk45)");
        m.solve();
        System.out.println("b = " + m.getValue("b"));

        //runODE("diffeqn(@(4)(\"y[2]-y[1]\",\"y[2]-(-2*y[0]+y[2])\",\"y[2]-y[3]\",\"y[2]-(y[0]-2*y[2])\"), 0, @(1,4)(1,0,0,1), 10, 0.01, rk4)");
        runODE("diffeqn(@(4)(\"y[4]-y[1]\",\"y[4]-(-2*y[0]+y[2])\",\"y[4]-2*sin(t)*y[3]\",\"y[4]-(y[0]-2*y[2])\"), 0, @(1,4)(1,0,0,1), 10, 0.01, rk4)");
        
        
        runODE("diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.001, rk4)");
        runODE("diffeqnPathHO(y[2] + 0.5*y[1] + 1*sin(y[0]) - 1.2*cos((2/3)*t), 0, @(1,2)(0.2, 0), 100, 0.0001, rk45, 2000, state)");
    }

    public static void runODE(String in) throws Throwable {
        MathExpression.EvalResult ev = solve(new MathExpression(in));
        System.out.println(ev + "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    }
}
