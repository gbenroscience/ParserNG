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
package com.github.gbenroscience.math.differentialcalculus.pde;

/**
 * Thrown when an explicit time integrator is asked to use a time step that
 * violates its estimated stability limit (CFL-type restriction), or when the
 * numerical solution is detected to be diverging (non-finite values or
 * runaway growth).
 */
public class UnstableTimeStepException extends PDEException {

    private static final long serialVersionUID = 1L;

    private final double dt;
    private final double maxStableDt;

    public UnstableTimeStepException(String message, double dt, double maxStableDt) {
        super(message);
        this.dt = dt;
        this.maxStableDt = maxStableDt;
    }

    /** @return the offending time step (NaN if not applicable). */
    public double getDt() {
        return dt;
    }

    /** @return the estimated largest stable time step (NaN if unknown). */
    public double getMaxStableDt() {
        return maxStableDt;
    }
}
