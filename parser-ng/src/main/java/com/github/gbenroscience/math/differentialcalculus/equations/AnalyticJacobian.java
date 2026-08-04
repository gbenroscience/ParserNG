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
package com.github.gbenroscience.math.differentialcalculus.equations;

import com.github.gbenroscience.math.differentialcalculus.autodiff.SystemAutoDiffEvaluator;

/**
 * Computes an exact n x n Jacobian d f_i / d y_j for a system of ODE RHS
 * expressions, using forward-mode AD ({@link SystemAutoDiffEvaluator}) instead
 * of finite differences — the analytic replacement for the central-difference
 * Jacobian in DifferentialEquations.stepImplicitEulerCore.
 *
 * componentEvaluators[i] must be the AD evaluator compiled from the i-th RHS
 * expression f_i(t, y_1..y_n), and must share the same frame layout (the same
 * tSlot / ySlotStart assignment) as the solver's vars array — the frame passed
 * to compute() IS that vars array, unmodified.
 *
 * <h2>Threading</h2>
 * Not thread-safe: the scratch buffer is reused across calls to avoid
 * per-Newton-iteration allocation. One AnalyticJacobian instance is meant to
 * back exactly one sequential solve, matching the same contract as
 * {@code CompanionSystemHandles}'s per-equation adapter — do not share a single
 * instance across concurrently running solves (e.g. ensemble lanes). Each
 * concurrent solve should build its own AnalyticJacobian from the same
 * (immutable, safely shareable) SystemAutoDiffEvaluator array.
 */
public final class AnalyticJacobian {

    private final SystemAutoDiffEvaluator[] componentEvaluators;
    private final int[] stateFrameIndices;
    private final double[] scratch; // order-0 and order-1 coefficients, reused per call

    public AnalyticJacobian(SystemAutoDiffEvaluator[] componentEvaluators, int[] stateFrameIndices) {
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
        }
        this.componentEvaluators = componentEvaluators;
        this.stateFrameIndices = stateFrameIndices;
        this.scratch = new double[2];
    }

    public int systemSize() {
        return componentEvaluators.length;
    }

    /**
     * Fills outJacobian[i][j] = d f_i / d y_j, evaluated at the given frame.
     * outJacobian must already be allocated as systemSize x systemSize.
     */
    public void compute(double[] frame, double[][] outJacobian) {
        int n = componentEvaluators.length;
        for (int i = 0; i < n; i++) {
            SystemAutoDiffEvaluator fi = componentEvaluators[i];
            double[] row = outJacobian[i];
            for (int j = 0; j < n; j++) {
                fi.taylorCoefficients(frame, stateFrameIndices[j], 1, scratch);
                row[j] = scratch[1];
            }
        }
    }

    /**
     * Evaluates f_i(frame) for every i in one pass (order 0 only), reusing the
     * same compiled evaluators used by compute() — a convenience for callers
     * that want the residual and the Jacobian from one consistent source.
     */
    public void evaluateResiduals(double[] frame, double[] outF) {
        int n = componentEvaluators.length;
        for (int i = 0; i < n; i++) {
            componentEvaluators[i].taylorCoefficients(frame, SystemAutoDiffEvaluator.NO_WRT_VARIABLE, 0, scratch);
            outF[i] = scratch[0];
        }
    }
}
