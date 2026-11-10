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
 * Slope limiters for MUSCL reconstruction. Given the backward and forward
 * differences {@code a} and {@code b} of a cell, {@link #slope} returns the
 * limited slope {@code s}; the face values are {@code w -/+ s/2}. All of them
 * are total-variation-diminishing (with an SSP time integrator under a CFL
 * restriction); they differ in how sharp they keep smooth extrema and contacts.
 */
public enum Limiter {
    /** First order: no reconstruction (piecewise-constant cells). */
    NONE,
    /** The most diffusive and most robust second-order limiter. */
    MINMOD,
    /** Monotonised central: the least diffusive of the three standard limiters, steeper on contacts. */
    MC,
    /** Smooth harmonic-mean limiter. */
    VAN_LEER;

    /** @return the limited slope for backward difference {@code a} and forward difference {@code b}. */
    public double slope(double a, double b) {
        if (a * b <= 0.0) {
            return 0.0;
        }
        switch (this) {
            case MINMOD:
                return Math.abs(a) < Math.abs(b) ? a : b;
            case MC: {
                double c = 0.5 * (a + b);
                double m = Math.min(2.0 * Math.abs(a), Math.min(2.0 * Math.abs(b), Math.abs(c)));
                return a > 0 ? m : -m;
            }
            case VAN_LEER:
                return 2.0 * a * b / (a + b);
            default:
                return 0.0;
        }
    }

    /** Parses a name (case, '-' and '_' insensitive). */
    public static Limiter parse(String s) {
        String n = s.trim().toLowerCase().replace("-", "").replace("_", "");
        switch (n) {
            case "none":
            case "firstorder":
                return NONE;
            case "minmod":
                return MINMOD;
            case "mc":
                return MC;
            case "vanleer":
                return VAN_LEER;
            default:
                throw new PDEConfigurationException("Unknown limiter '" + s + "'; use none, minmod, mc or van_leer");
        }
    }
}
