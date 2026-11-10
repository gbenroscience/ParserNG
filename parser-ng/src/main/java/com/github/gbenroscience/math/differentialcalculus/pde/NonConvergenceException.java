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
 * Thrown when an iterative solver (elliptic iteration, Newton or Krylov
 * iteration of an implicit step) fails to reach its tolerance and the
 * configuration asks for failure rather than a best-effort result.
 */
public class NonConvergenceException extends PDEException {

    private static final long serialVersionUID = 1L;

    private final SolverDiagnostics diagnostics;

    public NonConvergenceException(String message, SolverDiagnostics diagnostics) {
        super(message);
        this.diagnostics = diagnostics;
    }

    /** @return diagnostics captured when the iteration gave up (may be null). */
    public SolverDiagnostics getDiagnostics() {
        return diagnostics;
    }
}
