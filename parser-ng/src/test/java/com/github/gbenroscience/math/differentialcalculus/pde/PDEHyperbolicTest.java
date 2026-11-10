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

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Wave, advection and hyperbolic-system problems against analytical solutions.
 */
public class PDEHyperbolicTest {

    private static final double PI = Math.PI;

    @ParameterizedTest
    @CsvSource({
        "rk4,            1e-3, 2e-4",
        "verlet,         1e-3, 5e-4",
        "rk45,           1e-3, 2e-4",
        "crank_nicolson, 1e-3, 5e-3"
    })
    void wave1d_standingWave(String method, double dt, double tol) {
        PDEResult r = PDE.problem("diff(u,t,2) = c^2*diff(u,x,2)").param("c", 1.0).domain("x", 0, 1, 101)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(1.0, dt, method);
        double err = r.maxError("u", (x, y, z, t) -> Math.cos(PI * t) * Math.sin(PI * x));
        assertTrue(err < tol, method + " err " + err);
        assertEquals(PDEType.HYPERBOLIC, r.getDiagnostics().getType());
    }

    @Test
    void wave1d_initialVelocity() {
        PDEResult r = PDE.problem("diff(u,t,2) = diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "0").initialVelocity("u", "pi*sin(pi*x)").dirichlet("u", "0").solve(1.0, 1e-3, "rk4");
        double err = r.maxError("u", (x, y, z, t) -> Math.sin(PI * t) * Math.sin(PI * x));
        assertTrue(err < 2e-4, "err " + err);
    }

    @Test
    void wave1d_timeDependentBoundaryData() {
        // u = sin(t + x) solves u_tt = u_xx; boundary values are driven in time
        PDEResult r = PDE.problem("diff(u,t,2) = diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "sin(x)").initialVelocity("u", "cos(x)")
                .dirichlet(Face.X_MIN, "u", "sin(t)").dirichlet(Face.X_MAX, "u", "sin(t+1)")
                .solve(2.0, 1e-3, "rk4");
        double err = r.maxError("u", (x, y, z, t) -> Math.sin(t + x));
        assertTrue(err < 5e-4, "err " + err);
    }

    @Test
    void wave1d_neumann() {
        PDEResult r = PDE.problem("diff(u,t,2) = diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "cos(pi*x)").neumann("u", "0").solve(1.0, 1e-3, "rk4");
        double err = r.maxError("u", (x, y, z, t) -> Math.cos(PI * t) * Math.cos(PI * x));
        assertTrue(err < 3e-4, "err " + err);
    }

    @Test
    void wave1d_dampedWaveUsesTheVelocityVariable() {
        double g = 0.5;
        double w = Math.sqrt(PI * PI - g * g);
        PDEResult r = PDE.problem("diff(u,t,2) + 2*g*diff(u,t) = diff(u,x,2)").param("g", g).domain("x", 0, 1, 101)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(1.0, 1e-3, "rk4");
        double err = r.maxError("u", (x, y, z, t) -> Math.exp(-g * t) * (Math.cos(w * t) + g / w * Math.sin(w * t)) * Math.sin(PI * x));
        assertTrue(err < 3e-4, "err " + err);
    }

    @Test
    void membrane2d_vibratingSquare() {
        PDEResult r = PDE.problem("diff(u,t,2) = c^2*(diff(u,x,2) + diff(u,y,2))").param("c", 1.0)
                .domain("x", 0, 1, 41).domain("y", 0, 1, 41)
                .initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0").solve(0.5, 0.005, "verlet");
        double err = r.maxError("u", (x, y, z, t) -> Math.cos(PI * Math.sqrt(2) * t) * Math.sin(PI * x) * Math.sin(PI * y));
        assertTrue(err < 1e-3, "err " + err);
    }

    @Test
    void membrane2d_higherModeWithRk4() {
        PDEResult r = PDE.problem("diff(u,t,2) = diff(u,x,2) + diff(u,y,2)")
                .domain("x", 0, 1, 51).domain("y", 0, 1, 51)
                .initial("u", "sin(2*pi*x)*sin(pi*y)").dirichlet("u", "0").solve(0.3, 0.004, "rk4");
        double w = PI * Math.sqrt(5);
        double err = r.maxError("u", (x, y, z, t) -> Math.cos(w * t) * Math.sin(2 * PI * x) * Math.sin(PI * y));
        assertTrue(err < 2e-3, "err " + err);
    }

    @Test
    void advection_periodicFourthOrderCentral() {
        PDEOptions o = new PDEOptions().spatialOrder(4).integrator("rk4").dt(0.002);
        PDEResult r = PDE.problem("diff(u,t) + c*diff(u,x) = 0").param("c", 1.0).domain("x", 0, 1, 101)
                .initial("u", "sin(2*pi*x)").periodic("x").options(o).solve(1.0);
        double err = r.maxError("u", (x, y, z, t) -> Math.sin(2 * PI * (x - t)));
        assertTrue(err < 1e-5, "err " + err);
        assertEquals(PDEType.HYPERBOLIC, r.getDiagnostics().getType());
    }

    @Test
    void advection_upwindSchemesConvergeAtTheirFormalOrder() {
        for (int order : new int[]{1, 2}) {
            double[] e = new double[3];
            int[] ns = {41, 81, 161};
            for (int i = 0; i < 3; i++) {
                PDEOptions o = new PDEOptions().spatialOrder(order).firstDerivativeScheme(DerivativeScheme.BACKWARD)
                        .integrator("rk4").dt(0.2 / (ns[i] - 1));
                e[i] = PDE.problem("diff(u,t) = -diff(u,x)").domain("x", 0, 1, ns[i])
                        .initial("u", "sin(2*pi*x)").periodic("x").options(o).solve(0.5)
                        .maxError("u", (x, y, z, t) -> Math.sin(2 * PI * (x - t)));
            }
            double o1 = PDETestSupport.order(e[0], e[1], 2);
            double o2 = PDETestSupport.order(e[1], e[2], 2);
            assertTrue(o1 > order - 0.3 && o2 > order - 0.3, "upwind order " + order + " observed " + o1 + ", " + o2);
        }
    }

    @Test
    void advection_inflowDirichletBoundary() {
        // u_t + u_x = 0, inflow u(0,t) = sin(2 pi (-t)), right boundary left to a one-sided (outflow) condition
        PDEOptions o = new PDEOptions().spatialOrder(2).firstDerivativeScheme(DerivativeScheme.BACKWARD).integrator("rk4").dt(0.002);
        PDEResult r = PDE.problem("diff(u,t) = -diff(u,x)").domain("x", 0, 1, 201)
                .initial("u", "sin(2*pi*x)")
                .dirichlet(Face.X_MIN, "u", "sin(-2*pi*t)").neumann(Face.X_MAX, "u", "0")
                .options(o).solve(0.5);
        // away from the outflow boundary layer the transported profile is accurate
        double err = 0;
        for (int i = 0; i <= 150; i++) {
            double x = i / 200.0;
            err = Math.max(err, Math.abs(r.value("u", i) - Math.sin(2 * PI * (x - 0.5))));
        }
        assertTrue(err < 2e-3, "err " + err);
    }

    @Test
    void hyperbolicSystem_acousticsWithMixedBoundaryTypes() {
        // p_t = -u_x, u_t = -p_x; walls: u = 0 (Dirichlet), p_x = 0 (Neumann).  p = cos(pi x) cos(pi t), u = sin(pi x) sin(pi t)
        PDEResult r = PDE.problem("diff(p,t) = -diff(u,x)", "diff(u,t) = -diff(p,x)")
                .domain("x", 0, 1, 101).initial("p", "cos(pi*x)").initial("u", "0")
                .neumann("p", "0").dirichlet("u", "0").solve(1.0, 1e-3, "rk4");
        assertEquals(java.util.Arrays.asList("p", "u"), r.getUnknowns());
        double ep = r.maxError("p", (x, y, z, t) -> Math.cos(PI * x) * Math.cos(PI * t));
        double eu = r.maxError("u", (x, y, z, t) -> Math.sin(PI * x) * Math.sin(PI * t));
        assertTrue(ep < 1e-3 && eu < 1e-3, "errors " + ep + ", " + eu);
        // the same system written in implicit (non-solved) form
        PDEResult r2 = PDE.problem("diff(p,t) + diff(u,x) = 0", "diff(u,t) + diff(p,x) = 0")
                .domain("x", 0, 1, 101).initial("p", "cos(pi*x)").initial("u", "0")
                .neumann("p", "0").dirichlet("u", "0").solve(1.0, 1e-3, "rk4");
        assertEquals(0.0, r2.maxError("p", (x, y, z, t) -> r.valueAt("p", x, 0, 0)), 1e-9);
    }

    @Test
    void coupledParabolicSystem_arraySyntax() {
        double a = 0.1 * PI * PI;
        PDEResult r = PDE.problem("@(2)(\"diff(u,t) = 0.1*diff(u,x,2) - v\",\"diff(v,t) = 0.1*diff(v,x,2) + u\")")
                .domain("x", 0, 1, 81).initial("u", "sin(pi*x)").initial("v", "0").dirichlet("*", "0")
                .solve(1.0, 1e-3, "rk4");
        assertTrue(r.maxError("u", (x, y, z, t) -> Math.exp(-a * t) * Math.cos(t) * Math.sin(PI * x)) < 1e-4);
        assertTrue(r.maxError("v", (x, y, z, t) -> Math.exp(-a * t) * Math.sin(t) * Math.sin(PI * x)) < 1e-4);
    }

    @Test
    void hyperbolicProblemsDefaultToAnExplicitHighOrderIntegrator() {
        try (PDESolver s = PDE.problem("diff(u,t,2) = diff(u,x,2)").domain("x", 0, 1, 51)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solver()) {
            assertEquals(TimeIntegrator.RK4, s.integrator());
        }
        try (PDESolver s = PDETestSupport.heat1d(0.1, 51).solver()) {
            assertEquals(TimeIntegrator.CRANK_NICOLSON, s.integrator());
        }
    }
}
