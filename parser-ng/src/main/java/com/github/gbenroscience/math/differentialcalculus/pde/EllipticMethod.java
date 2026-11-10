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
 * Iterative solvers for steady-state (elliptic) problems.
 */
public enum EllipticMethod {
    /** Red-black SOR when the stencil allows it, otherwise lexicographic SOR. */
    AUTO,
    /** Jacobi iteration (point-Newton for non-linear problems). Slow but trivially parallel. */
    JACOBI,
    /** Lexicographic Gauss-Seidel (SOR with omega = 1). */
    GAUSS_SEIDEL,
    /** Lexicographic successive over-relaxation. */
    SOR,
    /** Red-black ordered SOR (parallel-friendly; needs a nearest-neighbour stencil without mixed derivatives). */
    RED_BLACK_SOR,
    /** Newton iteration with a matrix-free BiCGStab linear solver. Handles systems and nonlinear problems. */
    NEWTON_KRYLOV;

    /** Parses a user-supplied name, tolerant of case, '-' and '_'. */
    public static EllipticMethod parse(String name) {
        if (name == null) {
            return AUTO;
        }
        String n = name.trim().toLowerCase().replace("-", "").replace("_", "");
        switch (n) {
            case "":
            case "auto":
                return AUTO;
            case "jacobi":
                return JACOBI;
            case "gaussseidel":
            case "gs":
                return GAUSS_SEIDEL;
            case "sor":
                return SOR;
            case "redblack":
            case "redblacksor":
            case "rbsor":
                return RED_BLACK_SOR;
            case "newton":
            case "newtonkrylov":
            case "krylov":
                return NEWTON_KRYLOV;
            default:
                throw new PDEConfigurationException("Unknown elliptic method '" + name
                        + "'. Use one of: auto, jacobi, gauss_seidel, sor, red_black_sor, newton_krylov");
        }
    }
}
