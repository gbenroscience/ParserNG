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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.turbo;
 
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.ExprNodeAutoDiffEvaluator;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.CanonicalFrame;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.DiffEqnArgParser;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.DiffEqnCall;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.PostfixArgumentIsolator;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.ODESolverMethod;
import com.github.gbenroscience.math.differentialcalculus.equations.turbo.DifferentialEquations;
import com.github.gbenroscience.math.differentialcalculus.equations.turbo.HigherOrderODE;
import com.github.gbenroscience.math.differentialcalculus.equations.turbo.TurboODE;
import com.github.gbenroscience.math.differentialcalculus.equations.turbo.VectorODE;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;

import java.lang.invoke.MethodHandle;
import java.util.Arrays;
/**
 *
 * @author GBEMIRO 
 * Turbo-tier counterpart to {@link EquationRuntime} — identical routing,
 * argument parsing, equation isolation, and canonical-frame logic, wired to
 * the {@code MethodHandle}-based solvers ({@code
 * com.github.gbenroscience.math.differentialcalculus.equations.TurboODE/
 * VectorODE/HigherOrderODE}) instead of the {@code ODEFunction}-based
 * Standard tier.
 *
 * <h2>What's shared vs. what's duplicated</h2>
 * {@link DiffEqnArgParser}, {@link DiffEqnCall}, {@link
 * PostfixArgumentIsolator}, {@link CanonicalFrame}, and {@link
 * EquationDivider} (the actual symbolic term-splitting/linearity-checking
 * core) are all reused verbatim from the Standard-tier pipeline — none of
 * that logic touches {@code ODEFunction} or {@code MethodHandle} at all.
 * What's genuinely tier-specific and duplicated here: the final compile
 * step ({@link TurboCoefficientExtractor} calls {@code
 * ExprNodeCompiler.compileTurbo} instead of {@code compileStandard}), the
 * frame-remapping wrapper ({@link FrameRemapper} instead of
 * {@link FrameRemappingODEFunction}), and this class's dispatch — because
 * {@code TurboODE}/{@code VectorODE}/{@code HigherOrderODE} live in a
 * separate package with a separate (structurally identical, but distinct)
 * {@code DifferentialEquations.JacobianStrategy} type, so the ~15-line
 * Jacobian-building lambda can't be shared as a single Java object across
 * both functional-interface types even though the logic is identical.
 *
 * <h2>Checked exceptions</h2>
 * Every Turbo solver entry point declares {@code throws Throwable} (a
 * {@code MethodHandle.invokeExact} constraint) — propagated here rather
 * than swallowed, so callers see the same signal Standard-tier callers get
 * for free from unchecked exceptions.
 *
 * <h2>Known gap</h2>
 * Same as {@link EquationRuntime}: a genuine vector (non-HO) system falls
 * back to the solver's default finite-difference Jacobian rather than
 * failing outright, since the coefficient extractor only isolates one top
 * term per call.
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
     */
    public static Object solve(MathExpression me) throws Throwable {
        return new EquationRuntime(CoefficientExtractor::resolve).execute(me.getCachedPostfix());
    }

    /**
     * Full pipeline for one call. Returns a {@code Double} for a scalar
     * {@code diffeqn} and for {@code diffeqnHO}'s y(tEnd), a {@code
     * double[]} for a vector {@code diffeqn}, or a {@code double[][]} for
     * either *_PATH variant.
     */
    public Object execute(Token[] fullCallPostfix) throws Throwable {
        DiffEqnCall call = DiffEqnArgParser.parse(fullCallPostfix);
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
    // Kind-specific dispatch
    // ------------------------------------------------------------------

    private Object executeDiffEqn(DiffEqnCall call, ResolvedEquation resolved, MethodHandle fn,
                                   CanonicalFrame frame, int tSlot, int ySlotStart, int frameSize) throws Throwable {
        DifferentialEquations.JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, false);
        if (call.y0.length == 1) {
            return TurboODE.executeTurboODE(fn, tSlot, ySlotStart, frameSize,
                    call.t0, call.y0[0], call.tEnd, call.h, call.method, jac);
        }
        return VectorODE.executeVectorODE(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, jac);
    }

    private Object executeDiffEqnPath(DiffEqnCall call, ResolvedEquation resolved, MethodHandle fn,
                                       CanonicalFrame frame, int tSlot, int ySlotStart, int frameSize) throws Throwable {
        DifferentialEquations.JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, false);
        if (call.y0.length == 1) {
            return TurboODE.executeTurboODEPath(fn, tSlot, ySlotStart, frameSize,
                    call.t0, call.y0[0], call.tEnd, call.h, call.method, call.points, jac);
        }
        return VectorODE.executeVectorODEPath(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, call.points, jac);
    }

    private Object executeDiffEqnHO(DiffEqnCall call, ResolvedEquation resolved, MethodHandle fn,
                                     CanonicalFrame frame, int tSlot, int ySlotStart, int frameSize) throws Throwable {
        DifferentialEquations.JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, true);
        return HigherOrderODE.executeTurboODEHO(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, jac);
    }

    private Object executeDiffEqnPathHO(DiffEqnCall call, ResolvedEquation resolved, MethodHandle fn,
                                         CanonicalFrame frame, int tSlot, int ySlotStart, int frameSize) throws Throwable {
        DifferentialEquations.JacobianStrategy jac = buildJacobianIfNeeded(call, resolved, frame, true);
        return HigherOrderODE.executeTurboODEPathHO(fn, tSlot, ySlotStart, frameSize,
                call.t0, call.y0, call.tEnd, call.h, call.method, call.points, jac);
    }

    // ------------------------------------------------------------------
    // Analytic Jacobian wiring -- identical logic to EquationRuntime's, just
    // targeting the Turbo tier's own (structurally identical, distinct type)
    // DifferentialEquations.JacobianStrategy, and reading resolved.topDerivativeTree
    // (an ExprNode -- representation-agnostic, works unchanged for either tier)
    // through the same ExprNodeAutoDiffEvaluator used by the Standard tier.
    // ------------------------------------------------------------------

    private DifferentialEquations.JacobianStrategy buildJacobianIfNeeded(
            DiffEqnCall call, ResolvedEquation resolved, CanonicalFrame frame, boolean higherOrder) {
        if (call.method != ODESolverMethod.IMPLICIT_EULER) {
            return null;
        }

        int order = call.y0.length;

        if (!higherOrder && order != 1) {
            // Genuine vector system -- see class javadoc "Known gap". Falls back to
            // the solver's own finite-difference Jacobian rather than failing the call.
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
    }
}