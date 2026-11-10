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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Diagnostics of a solve: what was run, whether it converged / stayed stable,
 * and how accurate it is believed to be.
 */
public final class SolverDiagnostics {

    String method = "";
    PDEType type;
    boolean converged = true;
    long steps;
    long rejectedSteps;
    int iterations;
    long newtonIterations;
    long krylovIterations;
    double residualNorm = Double.NaN;
    double initialResidualNorm = Double.NaN;
    double updateNorm = Double.NaN;
    double errorEstimate = Double.NaN;
    double contractionFactor = Double.NaN;
    double spectralRadius = Double.NaN;
    double maxStableDt = Double.NaN;
    double lastDt = Double.NaN;
    double minDt = Double.NaN;
    double maxDt = Double.NaN;
    double finalTime = Double.NaN;
    double omega = Double.NaN;
    int spatialOrder;
    int threads = 1;
    long wallNanos;
    final List<String> warnings = new ArrayList<String>();

    /** @return textual name of the numerical method actually used. */
    public String getMethod() {
        return method;
    }

    /** @return classification of the equation(s). */
    public PDEType getType() {
        return type;
    }

    /** @return false if an iterative solver stopped without reaching its tolerance. */
    public boolean isConverged() {
        return converged;
    }

    /** @return accepted time steps (time-dependent problems). */
    public long getSteps() {
        return steps;
    }

    /** @return time steps rejected by the adaptive controller. */
    public long getRejectedSteps() {
        return rejectedSteps;
    }

    /** @return iterations of the elliptic solver. */
    public int getIterations() {
        return iterations;
    }

    /** @return total Newton iterations (implicit steps / Newton-Krylov). */
    public long getNewtonIterations() {
        return newtonIterations;
    }

    /** @return total Krylov (BiCGStab) iterations. */
    public long getKrylovIterations() {
        return krylovIterations;
    }

    /** @return max-norm of the final PDE residual (elliptic) or last nonlinear residual (implicit). */
    public double getResidualNorm() {
        return residualNorm;
    }

    /** @return max-norm of the residual of the starting guess (elliptic). */
    public double getInitialResidualNorm() {
        return initialResidualNorm;
    }

    /** @return max-norm of the last iterate update (elliptic). */
    public double getUpdateNorm() {
        return updateNorm;
    }

    /**
     * @return estimated remaining iteration error {@code rho/(1-rho)*|du|}
     * (elliptic) or the last embedded error estimate (adaptive RK).
     */
    public double getErrorEstimate() {
        return errorEstimate;
    }

    /** @return geometric-mean per-iteration residual reduction (elliptic). */
    public double getContractionFactor() {
        return contractionFactor;
    }

    /** @return estimated spectral radius of the semi-discrete Jacobian (explicit methods). */
    public double getSpectralRadius() {
        return spectralRadius;
    }

    /** @return estimated largest stable explicit time step (NaN if not estimated). */
    public double getMaxStableDt() {
        return maxStableDt;
    }

    public double getLastDt() {
        return lastDt;
    }

    public double getMinDt() {
        return minDt;
    }

    public double getMaxDt() {
        return maxDt;
    }

    /** @return time reached. */
    public double getFinalTime() {
        return finalTime;
    }

    /** @return relaxation factor used by SOR-type methods. */
    public double getOmega() {
        return omega;
    }

    public int getSpatialOrder() {
        return spatialOrder;
    }

    public int getThreads() {
        return threads;
    }

    /** @return wall-clock time spent inside solve calls, in nanoseconds. */
    public long getWallNanos() {
        return wallNanos;
    }

    /** @return non-fatal warnings (e.g. stability warnings, initial/boundary incompatibility). */
    public List<String> getWarnings() {
        return Collections.unmodifiableList(warnings);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("SolverDiagnostics{method=").append(method).append(", type=").append(type)
                .append(", converged=").append(converged);
        if (steps > 0) {
            sb.append(", steps=").append(steps).append(", t=").append(finalTime);
            if (rejectedSteps > 0) {
                sb.append(", rejected=").append(rejectedSteps);
            }
        }
        if (iterations > 0) {
            sb.append(", iterations=").append(iterations).append(", residual=").append(residualNorm);
        }
        if (newtonIterations > 0) {
            sb.append(", newton=").append(newtonIterations).append(", krylov=").append(krylovIterations);
        }
        if (!Double.isNaN(spectralRadius)) {
            sb.append(", spectralRadius=").append(spectralRadius).append(", maxStableDt=").append(maxStableDt);
        }
        if (!warnings.isEmpty()) {
            sb.append(", warnings=").append(warnings);
        }
        return sb.append('}').toString();
    }
}
