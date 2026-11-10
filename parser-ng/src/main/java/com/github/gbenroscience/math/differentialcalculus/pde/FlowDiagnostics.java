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
 * A snapshot of the quantities needed to judge whether a flow simulation is
 * numerically and physically well behaved. Obtained from
 * {@link IncompressibleFlow#diagnostics()}.
 */
public final class FlowDiagnostics {

    double time;
    long steps;
    double dt;
    double divergenceMax;
    double divergenceL2;
    double massFluxError;
    double divergenceIntegral;
    double pressureResidual;
    int pressureCycles;
    double velocityResidual;
    double kineticEnergy;
    double maxVorticity;
    double enstrophy;
    double maxSpeed;
    double reynolds;
    double cellReynolds;
    double cfl;
    double diffusionNumber;
    boolean pressureConverged;

    /** @return simulation time. */
    public double getTime() {
        return time;
    }

    /** @return number of steps taken. */
    public long getSteps() {
        return steps;
    }

    /** @return the last time step. */
    public double getDt() {
        return dt;
    }

    /** @return max-norm of the discrete divergence of the velocity (cell by cell); ~ the Poisson tolerance after a step. */
    public double getDivergenceMax() {
        return divergenceMax;
    }

    /** @return RMS norm of the discrete divergence. */
    public double getDivergenceL2() {
        return divergenceL2;
    }

    /**
     * @return net volume flux through the boundary (outward positive). For an incompressible flow with
     * consistent boundary data this is zero: it is the mass-conservation error of the boundary conditions.
     */
    public double getMassFluxError() {
        return massFluxError;
    }

    /** @return volume integral of the divergence; equals the net boundary flux by the discrete divergence theorem. */
    public double getDivergenceIntegral() {
        return divergenceIntegral;
    }

    /** @return max-norm residual of the last pressure-Poisson solve. */
    public double getPressureResidual() {
        return pressureResidual;
    }

    /** @return multigrid cycles used by the last pressure solve. */
    public int getPressureCycles() {
        return pressureCycles;
    }

    /** @return false if the last pressure solve did not reach its tolerance. */
    public boolean isPressureConverged() {
        return pressureConverged;
    }

    /** @return max |u^{n+1} - u^n| / dt over all faces in the last step: tends to zero at steady state. */
    public double getVelocityResidual() {
        return velocityResidual;
    }

    /** @return {@code 1/2 * integral |u|^2 dV}. */
    public double getKineticEnergy() {
        return kineticEnergy;
    }

    /** @return max vorticity magnitude (interior cell centres). */
    public double getMaxVorticity() {
        return maxVorticity;
    }

    /** @return {@code 1/2 * integral |vorticity|^2 dV}. */
    public double getEnstrophy() {
        return enstrophy;
    }

    /** @return max speed |u| at cell centres. */
    public double getMaxSpeed() {
        return maxSpeed;
    }

    /** @return {@code U_ref * L_ref / nu}. */
    public double getReynolds() {
        return reynolds;
    }

    /** @return {@code max|u| * min(h) / nu}: above 2 central differencing of convection is oscillation-prone. */
    public double getCellReynolds() {
        return cellReynolds;
    }

    /** @return {@code dt * sum_a max|u_a| / h_a}: the advective Courant number of the last step. */
    public double getCfl() {
        return cfl;
    }

    /** @return {@code dt * nu * sum_a 1/h_a^2}: the diffusion number (explicit stability needs roughly &lt; 1/4). */
    public double getDiffusionNumber() {
        return diffusionNumber;
    }

    @Override
    public String toString() {
        return String.format("FlowDiagnostics{t=%.5g, steps=%d, dt=%.3g, div(max)=%.2e, div(L2)=%.2e, massFlux=%.2e, "
                + "pressureRes=%.2e (%d cycles), du/dt=%.2e, KE=%.6g, maxVort=%.5g, Re=%.5g, CFL=%.3g, diffNo=%.3g}",
                time, steps, dt, divergenceMax, divergenceL2, massFluxError, pressureResidual, pressureCycles,
                velocityResidual, kineticEnergy, maxVorticity, reynolds, cfl, diffusionNumber);
    }
}
