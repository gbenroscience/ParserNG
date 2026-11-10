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

/** Snapshot of the health of a finite-volume run; see {@link FiniteVolumeSolver#diagnostics()}. */
public final class FVDiagnostics {

    double time;
    long steps;
    double dt;
    double cfl;
    double[] totals;
    double[] initialTotals;
    double[] conservationDefect;
    double[] boundaryFlux;
    double[] min;
    double[] max;
    double[] minPrimitive;
    long reconstructionFallbacks;

    public double getTime() {
        return time;
    }

    public long getSteps() {
        return steps;
    }

    public double getDt() {
        return dt;
    }

    /** @return the Courant number of the last step. */
    public double getCfl() {
        return cfl;
    }

    /** @return volume integral of each conserved component now. */
    public double[] getTotals() {
        return totals.clone();
    }

    /** @return the same at t = 0. */
    public double[] getInitialTotals() {
        return initialTotals.clone();
    }

    /**
     * @return {@code total(t) - total(0) + (time integral of the net outward boundary flux) - (time integral of the sources)}
     * per component. For a conservative scheme this is zero up to round-off for ANY boundary condition: it is the
     * discrete conservation error.
     */
    public double[] getConservationDefect() {
        return conservationDefect.clone();
    }

    /** @return time integral of the net outward flux through the domain boundary, per component. */
    public double[] getBoundaryFluxIntegral() {
        return boundaryFlux.clone();
    }

    /** @return minimum over cells of each conserved component. */
    public double[] getMin() {
        return min.clone();
    }

    /** @return maximum over cells of each conserved component. */
    public double[] getMax() {
        return max.clone();
    }

    /** @return minimum over cells of each primitive variable (equals {@link #getMin()} for laws without primitives); for Euler: density, velocities, pressure. */
    public double[] getMinPrimitive() {
        return minPrimitive.clone();
    }

    /** @return faces where the limited reconstruction was unphysical and the first-order states were used instead (cumulative). */
    public long getReconstructionFallbacks() {
        return reconstructionFallbacks;
    }

    @Override
    public String toString() {
        return String.format("FVDiagnostics{t=%.5g, steps=%d, dt=%.3g, cfl=%.3g, totals=%s, conservationDefect=%s, fallbacks=%d}",
                time, steps, dt, cfl, java.util.Arrays.toString(totals), java.util.Arrays.toString(conservationDefect), reconstructionFallbacks);
    }
}
