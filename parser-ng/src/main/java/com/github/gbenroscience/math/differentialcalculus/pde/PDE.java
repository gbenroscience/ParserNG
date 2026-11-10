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
 * Entry point of the PDE engine, mirroring the style of ParserNG's
 * {@code diffeqn} family: equations are written as ParserNG expressions, with
 * derivatives spelled {@code diff(u,x,2)}, and are compiled once.
 * <pre>{@code
 * // u_t = alpha u_xx on [0,1], u(0,t)=u(1,t)=0, u(x,0)=sin(pi x)
 * PDEResult r = PDE.problem("diff(u,t) = alpha*diff(u,x,2)")
 *         .param("alpha", 0.1)
 *         .domain("x", 0, 1, 101)
 *         .initial("u", "sin(pi*x)")
 *         .dirichlet("u", "0")
 *         .solve(0.5, 1e-3, "rk4");
 *
 * // Laplace: u_xx + u_yy = 0, u = 1 on y = 1 and 0 elsewhere
 * PDEResult s = PDE.problem("diff(u,x,2) + diff(u,y,2) = 0")
 *         .domain("x", 0, 1, 65).domain("y", 0, 1, 65)
 *         .dirichlet("u", "0")
 *         .dirichlet(Face.Y_MAX, "u", "1")
 *         .solveSteady("sor");
 * }</pre>
 * Systems: pass several equations, or ParserNG's array syntax
 * {@code @(2)("diff(u,t)=...","diff(v,t)=...")}.
 */
public final class PDE {

    private PDE() {
    }

    /** Starts a problem definition from one or more equations. */
    public static PDEProblem.Builder problem(String... equations) {
        return PDEProblem.builder(equations);
    }

    /** Creates a reusable solver (for real-time stepping). */
    public static PDESolver solver(PDEProblem problem, PDEOptions options) {
        return new PDESolver(problem, options);
    }

    /** Solves a time-dependent problem from {@code options.t0} to {@code tEnd}. */
    public static PDEResult solve(PDEProblem problem, PDEOptions options, double tEnd) {
        PDESolver s = new PDESolver(problem, options);
        try {
            if (!s.isTimeDependent()) {
                throw new PDEConfigurationException("The problem has no time derivative; use solveSteady()");
            }
            s.initialize();
            s.advanceTo(tEnd);
            return s.result();
        } finally {
            s.close();
        }
    }

    /** Solves a steady-state (elliptic) problem. */
    public static PDEResult solveSteady(PDEProblem problem, PDEOptions options) {
        PDESolver s = new PDESolver(problem, options);
        try {
            if (s.isTimeDependent()) {
                throw new PDEConfigurationException("The problem contains time derivatives; use solve(tEnd, dt, method)");
            }
            s.initialize();
            return s.solveSteady();
        } finally {
            s.close();
        }
    }
}
