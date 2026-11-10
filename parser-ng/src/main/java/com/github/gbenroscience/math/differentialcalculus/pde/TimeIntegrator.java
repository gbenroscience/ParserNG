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
 * Time integration methods for time-dependent PDEs (method of lines).
 */
public enum TimeIntegrator {
    /** Chosen from the PDE class: RK4 for hyperbolic, Crank-Nicolson for parabolic problems. */
    AUTO(false, false),
    /** Forward Euler. First order, explicit, conditionally stable. */
    EULER(true, false),
    /** Heun's method (explicit trapezoid, RK2). Second order, explicit. */
    HEUN(true, false),
    /** Classical Runge-Kutta. Fourth order, explicit. */
    RK4(true, false),
    /** Dormand-Prince 5(4) with embedded error control (adaptive step). Explicit. */
    DORMAND_PRINCE(true, true),
    /** Backward Euler. First order, implicit, L-stable (Newton-Krylov). */
    BACKWARD_EULER(false, false),
    /** Crank-Nicolson (implicit trapezoid). Second order, A-stable (Newton-Krylov). */
    CRANK_NICOLSON(false, false),
    /** Velocity-Verlet / leapfrog for second-order-in-time equations. Explicit, symplectic. */
    VELOCITY_VERLET(true, false);

    private final boolean explicit;
    private final boolean adaptive;

    TimeIntegrator(boolean explicit, boolean adaptive) {
        this.explicit = explicit;
        this.adaptive = adaptive;
    }

    public boolean isExplicit() {
        return explicit;
    }

    public boolean isAdaptive() {
        return adaptive;
    }

    /** Parses a user-supplied name, tolerant of case, '-' and '_' (e.g. "rk4", "crank-nicolson", "rk45"). */
    public static TimeIntegrator parse(String name) {
        if (name == null) {
            return AUTO;
        }
        String n = name.trim().toLowerCase().replace("-", "").replace("_", "");
        switch (n) {
            case "":
            case "auto":
                return AUTO;
            case "euler":
            case "explicit":
            case "forwardeuler":
                return EULER;
            case "heun":
            case "rk2":
                return HEUN;
            case "rk4":
                return RK4;
            case "rk45":
            case "dormandprince":
            case "dp45":
            case "adaptive":
                return DORMAND_PRINCE;
            case "implicit":
            case "backwardeuler":
            case "impliciteuler":
            case "be":
                return BACKWARD_EULER;
            case "cranknicolson":
            case "cn":
                return CRANK_NICOLSON;
            case "verlet":
            case "velocityverlet":
            case "leapfrog":
                return VELOCITY_VERLET;
            default:
                throw new PDEConfigurationException("Unknown time integrator '" + name
                        + "'. Use one of: auto, euler, heun, rk4, rk45, backward_euler, crank_nicolson, verlet");
        }
    }
}
