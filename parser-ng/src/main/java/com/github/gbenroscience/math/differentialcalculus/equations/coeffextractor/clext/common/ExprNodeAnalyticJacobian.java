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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common;

/**
 * Connects {@link ExprNodeAutoDiffEvaluator} to
 * {@code DifferentialEquations.JacobianStrategy} — the piece that was
 * missing before this class: {@code ExprNodeAutoDiffEvaluator} computes
 * exact derivatives correctly, but nothing in the solver-facing path
 * (VectorODE, HigherOrderODE, LinearHODifferentialEquations) had any way to
 * actually call it. Without this connector, the finite-difference default
 * in {@code DifferentialEquations.finiteDifferenceStrategy} was the only
 * thing that could serve as a JacobianStrategy for an equation compiled
 * through this ExprNode-based pipeline, regardless of how correct the AD
 * evaluator sitting next to it was.
 *
 * This is the direct counterpart of {@code AnalyticJacobian} (which plays
 * the same role for the older {@code MathExpression}/{@code
 * SystemAutoDiffEvaluator} pipeline) — same shape, same contract, just
 * sourced from {@code ExprNodeAutoDiffEvaluator} instances instead of
 * {@code SystemAutoDiffEvaluator} ones.
 *
 * <h2>Usage</h2>
 * Build one {@link ExprNodeAutoDiffEvaluator} per companion-system RHS
 * component (order 1, since a Jacobian only ever needs first derivatives),
 * and the frame indices of the state block's components (ySlotStart through
 * ySlotStart+order-1, exactly as CompanionSystemHandles already lays them
 * out). {@link #compute} then matches the signature
 * {@code DifferentialEquations.JacobianStrategy#computeDfDy} expects, so a
 * method reference — {@code jacobian::compute} — is a complete
 * JacobianStrategy, passable straight into {@code VectorODE.executeVectorODE}
 * or {@code HigherOrderODE.executeTurboODEHO} for the IMPLICIT_EULER method,
 * with no adapter glue needed.
 *
 * <h2>Threading</h2>
 * Not thread-safe: the scratch buffer is reused across calls to avoid
 * per-Newton-iteration allocation, the same contract {@code AnalyticJacobian}
 * and {@code CompanionSystemHandles}'s per-equation adapter already use. One
 * instance backs exactly one sequential solve; each concurrent solve builds
 * its own instance from the same (immutable, safely shareable)
 * {@code ExprNodeAutoDiffEvaluator} array.
 */
public final class ExprNodeAnalyticJacobian {

    private final ExprNodeAutoDiffEvaluator[] componentEvaluators;
    private final int[] stateFrameIndices;
    private final double[] scratch; // order-0 and order-1 coefficients, reused per call

    /**
     * 
     * @param componentEvaluators
     * @param stateFrameIndices 
     */
    public ExprNodeAnalyticJacobian(ExprNodeAutoDiffEvaluator[] componentEvaluators, int[] stateFrameIndices) {
        if (componentEvaluators == null || componentEvaluators.length == 0) {
            throw new IllegalArgumentException("componentEvaluators must not be empty");
        }
        if (stateFrameIndices == null || stateFrameIndices.length != componentEvaluators.length) {
            throw new IllegalArgumentException(
                    "stateFrameIndices.length must equal componentEvaluators.length (systemSize), got "
                    + (stateFrameIndices == null ? "null" : stateFrameIndices.length)
                    + " vs " + componentEvaluators.length);
        }
        for (int i = 0; i < componentEvaluators.length; i++) {
            if (componentEvaluators[i] == null) {
                throw new IllegalArgumentException("componentEvaluators[" + i + "] must not be null");
            }
            if (componentEvaluators[i].getMaxOrder() < 1) {
                throw new IllegalArgumentException(
                        "componentEvaluators[" + i + "] was built with maxOrder < 1 — a Jacobian needs at "
                        + "least first-order derivatives; construct it with new ExprNodeAutoDiffEvaluator("
                        + "expression, 1) or higher.");
            }
        }
        this.componentEvaluators = componentEvaluators;
        this.stateFrameIndices = stateFrameIndices;
        this.scratch = new double[2];
    }

    public int systemSize() {
        return componentEvaluators.length;
    }

    /**
     * Fills outDfDy[i][j] = d f_i / d y_j (exact, via forward-mode AD — no
     * finite differences anywhere in this call), evaluated at the given
     * frame. outDfDy must already be allocated as systemSize x systemSize.
     * Matches DifferentialEquations.JacobianStrategy#computeDfDy's
     * signature exactly.
     * @param frame
     * @param outDfDy
     */
    public void compute(double[] frame, double[][] outDfDy) {
        int n = componentEvaluators.length;
        for (int i = 0; i < n; i++) {
            ExprNodeAutoDiffEvaluator fi = componentEvaluators[i];
            double[] row = outDfDy[i];
            for (int j = 0; j < n; j++) {
                fi.taylorCoefficients(frame, stateFrameIndices[j], 1, scratch);
                row[j] = scratch[1];
            }
        }
    }

    /**
     * Evaluates f_i(frame) for every i in one pass (order 0 only), reusing
     * the same compiled evaluators compute() uses — a convenience for
     * callers that want the residual and the Jacobian from one consistent
     * source rather than re-deriving f_i(frame) some other way.
     * @param frame
     * @param outF
     */
    public void evaluateResiduals(double[] frame, double[] outF) {
        int n = componentEvaluators.length;
        for (int i = 0; i < n; i++) {
            componentEvaluators[i].taylorCoefficients(frame, ExprNodeAutoDiffEvaluator.NO_WRT_VARIABLE, 0, scratch);
            outF[i] = scratch[0];
        }
    }
}