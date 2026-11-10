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

/** Approximate Riemann solvers / numerical fluxes for the finite-volume solver. */
public enum NumericalFlux {
    /** Local Lax-Friedrichs: central flux plus dissipation scaled by the largest local wave speed. Robust, diffusive. */
    LLF,
    /** Rusanov: in this implementation the same local-speed flux as {@link #LLF} (the two names are used interchangeably in the literature). */
    RUSANOV,
    /** Harten-Lax-van Leer two-wave solver with Davis speed estimates. Sharper than LLF; smears contacts. */
    HLL,
    /** HLL with the contact wave restored (laws that provide it, e.g. Euler). Resolves contacts and shear. */
    HLLC;

    /** Parses a name (case, '-' and '_' insensitive). */
    public static NumericalFlux parse(String s) {
        String n = s.trim().toLowerCase().replace("-", "").replace("_", "");
        switch (n) {
            case "llf":
            case "locallaxfriedrichs":
                return LLF;
            case "rusanov":
                return RUSANOV;
            case "hll":
                return HLL;
            case "hllc":
                return HLLC;
            default:
                throw new PDEConfigurationException("Unknown numerical flux '" + s + "'; use llf, rusanov, hll or hllc");
        }
    }
}
