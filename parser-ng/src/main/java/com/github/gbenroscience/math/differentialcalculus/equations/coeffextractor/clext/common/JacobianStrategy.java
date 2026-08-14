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
 *
 * @author GBEMIRO Supplies the raw df/dy Jacobian (NOT the Newton "I - h*df/dy"
 * matrix — the solver applies that transform itself) for the implicit solver.
 * The default implementation used when none is supplied is central- difference
 * finite differences; passing an
 * {@code com.github.gbenroscience.math.differentialcalculus.autodiff.AnalyticJacobian}-backed
 * strategy replaces that with an exact forward-mode-AD Jacobian.
 */
@FunctionalInterface
public interface JacobianStrategy {

    /**
     * @param vars the current frame — vars[ySlotStart..ySlotStart+systemSize)
     * holds the Newton iterate to differentiate at, vars[tSlot] holds the
     * corresponding evaluation time
     * @param outDfDy systemSize x systemSize; fill outDfDy[row][col] = d f_row
     * / d y_col
     *
     * @throws Throwable
     */
    void computeDfDy(double[] vars, double[][] outDfDy);
}
