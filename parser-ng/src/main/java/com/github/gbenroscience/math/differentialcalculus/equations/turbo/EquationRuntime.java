package com.github.gbenroscience.math.differentialcalculus.equations.turbo;

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.ExprNodeAutoDiffEvaluator;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.CanonicalFrame;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.DiffEqnArgParser;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.DiffEqnCall;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.JacobianStrategy;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.PostfixArgumentIsolator;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.ODESolverMethod;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.turbo.CoefficientExtractor;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.turbo.EquationCoefficientResolver;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.turbo.FrameRemapper;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.turbo.ResolvedEquation;
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.ParserResult;

import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author GBEMIRO Turbo-tier counterpart to {@link EquationRuntime} — identical
 * routing, argument parsing, equation isolation, and canonical-frame logic,
 * wired to the {@code MethodHandle}-based solvers ({@code
 * com.github.gbenroscience.math.differentialcalculus.equations.TurboODE/
 * VectorODE/HigherOrderODE}) instead of the {@code ODEFunction}-based Standard
 * tier.
 *
 * <h2>Explicit systems</h2>
 * Argument 0 may be {@code @(n)("eq1", ..., "eqN")} — an explicit system, one
 * equation per state component, each independently parsed and compiled. See the
 * Standard tier's {@code EquationRuntime#executeSystem} javadoc for the full
 * explanation (identical design, {@code MethodHandle} substituted for
 * {@code ODEFunction} throughout).
 *
 * <h2>Checked exceptions</h2>
 * Every Turbo solver entry point declares {@code throws Throwable} (a
 * {@code MethodHandle.invokeExact} constraint) — propagated here rather than
 * swallowed, so callers see the same signal Standard-tier callers get for free
 * from unchecked exceptions.
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
     * One-call convenience, the Turbo-tier twin of {@link
     * EquationRuntime#solve(MathExpression)}. Equivalent to:
     * <pre>{@code
     * new TurboEquationRuntime(TurboCoefficientExtractor::resolve).execute(me.getCachedPostfix())
     * }</pre>
     *
     * @param me
     * @return
     */
    public static MathExpression.EvalResult solve(MathExpression me) {
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
    public static MathExpression.EvalResult solve(MathExpression.Token[] postfix, MathExpression.EvalResult out) {
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
        }
        return out;
    }

    /**
     * Full pipeline for one call. Returns a {@code Double} for a scalar
     * {@code diffeqn} and for {@code diffeqnHO}'s y(tEnd), a {@code
     * double[]} for a vector {@code diffeqn} (or an explicit system's endpoint
     * state), or a {@code double[][]} for either *_PATH variant.
     *
     * @param fullCallPostfix
     * @return
     * @throws java.lang.Throwable
     */
    public Object execute(Token[] fullCallPostfix) throws Throwable {
        DiffEqnCall call = DiffEqnArgParser.parse(fullCallPostfix);

        if (call.equationArraySyntax) {
            return executeSystem(call);
        }

        Token[] equationPostfix = PostfixArgumentIsolator.isolateArgument(fullCallPostfix, 0);
        ResolvedEquation resolved = coefficientResolver.resolve(equationPostfix, call.y0.length);
        CanonicalFrame frame = new CanonicalFrame(resolved.canonicalToReal, resolved.realFrameSize);
        MethodHandle fn = FrameRemapper.wrap(resolved.topDerivativeRealFrame, frame);

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
    // Kind-specific dispatch (single equation / HO — unchanged)
    // ------------------------------------------------------------------
    private Object executeDiffEqn(DiffEqnCall call, ResolvedEquation resolved, MethodHandle fn,
            CanonicalFrame frame, int tSlot, int ySlotStart, int frameSize) throws Throwable {
        JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, false);
        if (call.y0.length == 1) {
            return TurboODE.executeTurboODE(fn, tSlot, ySlotStart, frameSize,
                    call.t0, call.y0[0], call.tEnd, call.h, call.method, jac);
        }
        return VectorODE.executeVectorODE(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, jac);
    }

    private Object executeDiffEqnPath(DiffEqnCall call, ResolvedEquation resolved, MethodHandle fn,
            CanonicalFrame frame, int tSlot, int ySlotStart, int frameSize) throws Throwable {
        JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, false);
        if (call.y0.length == 1) {
            return TurboODE.executeTurboODEPath(fn, tSlot, ySlotStart, frameSize,
                    call.t0, call.y0[0], call.tEnd, call.h, call.method, call.points, jac);
        }
        return VectorODE.executeVectorODEPath(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, call.points, jac);
    }

    private Object executeDiffEqnHO(DiffEqnCall call, ResolvedEquation resolved, MethodHandle fn,
            CanonicalFrame frame, int tSlot, int ySlotStart, int frameSize) throws Throwable {
        JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, true);
        return HigherOrderODE.executeTurboODEHO(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, jac);
    }

    private Object executeDiffEqnPathHO(DiffEqnCall call, ResolvedEquation resolved, MethodHandle fn,
            CanonicalFrame frame, int tSlot, int ySlotStart, int frameSize) throws Throwable {
        JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, true);
        return HigherOrderODE.executeTurboODEPathHO(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, call.points, jac, call.presentationStrategy);
    }

    // ------------------------------------------------------------------
    // Explicit system dispatch — see Standard-tier EquationRuntime#executeSystem
    // for the full design rationale (identical here, MethodHandle throughout).
    // ------------------------------------------------------------------
    private Object executeSystem(DiffEqnCall call) throws Throwable {
        int n = call.equationTexts.length;

        ResolvedEquation[] resolved = new ResolvedEquation[n];
        CanonicalFrame[] frames = new CanonicalFrame[n];
        MethodHandle[] perEquationFn = new MethodHandle[n];
        System.out.println("-----------------------------"+Arrays.toString(call.equationTexts));
// turbo tier, executeSystem — identical reasoning:
        for (int i = 0; i < n; i++) {
            MathExpression synthetic = new MathExpression("diffeqn(" + call.equationTexts[i] + ", 0, 0, 1)");
            Token[] eqPostfix = PostfixArgumentIsolator.isolateArgument(synthetic.getCachedPostfix(), 0);
            resolved[i] = coefficientResolver.resolve(eqPostfix, n);
            frames[i] = new CanonicalFrame(resolved[i].canonicalToReal, resolved[i].realFrameSize);
            perEquationFn[i] = FrameRemapper.wrap(resolved[i].topDerivativeRealFrame, frames[i]);
        }
        MethodHandle fn = SystemFunctionHandles.buildSystem(perEquationFn, n);

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
     * Per-row Jacobian for an explicit system — see the Standard tier's
     * equivalent method for the full rationale.
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
        String fullCall = "diffeqnHO((3t^2)*y[4]+(5*sin(t))*y[3]+(5/t)*y[2]-3*y[1]+3*t*y[0], 1, @(1,4)(1, 0, 0, 0), 20, 0.01, rk4)";

        MathExpression me = new MathExpression(fullCall);

        DiffEqnCall call = DiffEqnArgParser.parse(me.getCachedPostfix());
        System.out.println("Routed as: " + call.kind);
        System.out.println("t0=" + call.t0 + ", y0=" + Arrays.toString(call.y0)
                + ", tEnd=" + call.tEnd + ", h=" + call.h + ", method=" + call.method);

        Object result = solve(me);
        System.out.println("Result: " + result);

        // NEW: explicit system.
        MathExpression sys = new MathExpression(
                "diffeqn(@(2)(\"y[2]-(0.6*y[0]-0.03*y[0]*y[1])\", \"y[2]-(-0.9*y[1]+0.02*y[0]*y[1])\"), "
                + "0, @(1,2)(30, 4), 20, 0.01, rk4)");
        System.out.println("System result: " + solve(sys));
        
        
        
        MathExpression mz = new MathExpression("diffeqn(@(2)( \"y[2]+y[0]\", \"y[2]-y[0]\" ),0, @(1,2)(1,0), 10, 0.01, rk4)");
        System.out.println("------------"+mz.solve());
        
        MathExpression mzz = new MathExpression("diffeqnPathHO(y[2] + 9.81*sin(y[0]), 0, @(1,2)(0.5, 0), 10, 0.001, rk4, 500, trajectory)");
         System.out.println("SIMPLE-PENDULUM------------"+mzz.solve());
        
        
    }
}
/*
------------103043.08662101434
------------165600.66356249657
------------227482.4757648183
*/