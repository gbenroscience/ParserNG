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
 * Worked examples of the PDE engine. Each method states the mathematical
 * problem, then solves it in a few lines of Java, and is executed (and its
 * answer checked) by {@code PDEExamplesTest}, so the documentation cannot
 * drift away from the code.
 */
public final class PDEExamples {

    private PDEExamples() {
    }

    /**
     * <b>Heat equation.</b>
     * <pre>
     *   u_t = alpha u_xx          0 &lt; x &lt; 1,  t &gt; 0
     *   u(x,0) = sin(pi x)        initial condition
     *   u(0,t) = u(1,t) = 0       boundary conditions
     * </pre>
     * Exact solution {@code exp(-pi^2 alpha t) sin(pi x)}.
     */
    public static PDEResult heatEquation() {
        return PDE.problem("diff(u,t) = alpha*diff(u,x,2)")
                .param("alpha", 0.1)
                .domain("x", 0, 1, 101)
                .initial("u", "sin(pi*x)")
                .dirichlet("u", "0")
                .solve(0.5, 1e-3, "crank_nicolson");
    }

    /**
     * <b>Vibrating membrane</b> (square drum, clamped edge, plucked in the middle).
     * <pre>
     *   u_tt = c^2 (u_xx + u_yy)         on the unit square
     *   u(x,y,0) = exp(-80 ((x-1/2)^2 + (y-1/2)^2)),  u_t(x,y,0) = 0
     *   u = 0 on the boundary
     * </pre>
     * Velocity-Verlet is symplectic, so the discrete energy stays bounded.
     */
    public static PDEResult vibratingMembrane() {
        return PDE.problem("diff(u,t,2) = c^2*(diff(u,x,2) + diff(u,y,2))")
                .param("c", 1.0)
                .domain("x", 0, 1, 81).domain("y", 0, 1, 81)
                .initial("u", "exp(-80*((x-0.5)^2 + (y-0.5)^2))")
                .dirichlet("u", "0")
                .solve(1.0, 0.004, "verlet");
    }

    /**
     * <b>Laplace's equation</b> (steady heat in a plate, top edge held hot).
     * <pre>
     *   u_xx + u_yy = 0 on the unit square,  u = sin(pi x) on y = 1,  u = 0 elsewhere
     * </pre>
     * Exact solution {@code sin(pi x) sinh(pi y) / sinh(pi)}.
     */
    public static PDEResult laplaceEquation() {
        return PDE.problem("diff(u,x,2) + diff(u,y,2) = 0")
                .domain("x", 0, 1, 65).domain("y", 0, 1, 65)
                .dirichlet("u", "0")
                .dirichlet(Face.Y_MAX, "u", "sin(pi*x)")
                .solveSteady("sor");
    }

    /**
     * <b>Wave equation</b> (vibrating string with fixed ends, released with an initial velocity).
     * <pre>
     *   u_tt = c^2 u_xx,  u(x,0) = 0,  u_t(x,0) = 2 pi sin(2 pi x),  u(0,t) = u(1,t) = 0
     * </pre>
     * Exact solution {@code sin(2 pi t) sin(2 pi x)} for {@code c = 1}.
     */
    public static PDEResult waveEquation() {
        return PDE.problem("diff(u,t,2) = c^2*diff(u,x,2)")
                .param("c", 1.0)
                .domain("x", 0, 1, 201)
                .initial("u", "0")
                .initialVelocity("u", "2*pi*sin(2*pi*x)")
                .dirichlet("u", "0")
                .options(new PDEOptions().spatialOrder(4).integrator("rk4").dt(1e-3))
                .solve(1.25);
    }

    /**
     * <b>Coupled system</b> (two diffusing, mutually rotating species).
     * <pre>
     *   u_t = D u_xx - v        v_t = D v_xx + u        u(x,0) = sin(pi x), v(x,0) = 0, u = v = 0 at the ends
     * </pre>
     * Exact solution {@code u + i v = exp((-D pi^2 + i) t) sin(pi x)}.
     */
    public static PDEResult coupledSystem() {
        return PDE.problem("diff(u,t) = D*diff(u,x,2) - v", "diff(v,t) = D*diff(v,x,2) + u")
                .param("D", 0.1)
                .domain("x", 0, 1, 81)
                .initial("u", "sin(pi*x)").initial("v", "0")
                .dirichlet("*", "0")
                .solve(1.0, 1e-3, "rk4");
    }

    /**
     * <b>Gray-Scott reaction-diffusion</b> (a non-linear coupled system on a periodic square).
     * <pre>
     *   u_t = Du lap(u) - u v^2 + F (1-u)      v_t = Dv lap(v) + u v^2 - (F+k) v
     * </pre>
     */
    public static PDEResult grayScott() {
        return PDE.problem("diff(u,t) = Du*(diff(u,x,2) + diff(u,y,2)) - u*v^2 + F*(1-u)",
                "diff(v,t) = Dv*(diff(v,x,2) + diff(v,y,2)) + u*v^2 - (F+k)*v")
                .param("Du", 2e-5).param("Dv", 1e-5).param("F", 0.04).param("k", 0.06)
                .domain("x", 0, 2.5, 64).domain("y", 0, 2.5, 64)
                .initial("u", "1 - 0.5*exp(-100*((x-1.25)^2 + (y-1.25)^2))")
                .initial("v", "0.25*exp(-100*((x-1.25)^2 + (y-1.25)^2))")
                .periodic("x").periodic("y")
                .solve(20.0, 0.1, "rk4");
    }

    /**
     * <b>Real-time simulation loop</b>: configure once, then advance repeatedly with no allocation
     * and read the live solution storage (no copy) after every step.
     *
     * @return the largest |u| seen after the final step
     */
    public static double realTimeLoop(int frames) {
        try (PDESolver solver = PDE.problem("diff(u,t,2) = diff(u,x,2) + diff(u,y,2)")
                .domain("x", 0, 1, 65).domain("y", 0, 1, 65)
                .initial("u", "exp(-100*((x-0.3)^2 + (y-0.6)^2))")
                .dirichlet("u", "0")
                .options(new PDEOptions().integrator("verlet"))
                .solver()) {
            solver.initialize();
            double dt = 0.5 * solver.maxStableTimeStep();       // half the stability limit
            double[] u = solver.state("u");                     // live storage
            Grid g = solver.grid();
            double peak = 0;
            for (int frame = 0; frame < frames; frame++) {
                solver.step(dt);
                peak = 0;
                for (int j = 0; j < g.n(1); j++) {
                    for (int i = 0; i < g.n(0); i++) {
                        peak = Math.max(peak, Math.abs(u[g.index(i, j, 0)]));   // hand to the renderer here
                    }
                }
            }
            return peak;
        }
    }

    public static void main(String[] args) {
        System.out.println("heat equation        : " + heatEquation().getDiagnostics());
        System.out.println("vibrating membrane   : " + vibratingMembrane().getDiagnostics());
        System.out.println("laplace              : " + laplaceEquation().getDiagnostics());
        System.out.println("wave equation        : " + waveEquation().getDiagnostics());
        System.out.println("coupled system       : " + coupledSystem().getDiagnostics());
        System.out.println("gray-scott           : " + grayScott().getDiagnostics());
        System.out.println("real-time peak |u|   : " + realTimeLoop(200));
    }
}
