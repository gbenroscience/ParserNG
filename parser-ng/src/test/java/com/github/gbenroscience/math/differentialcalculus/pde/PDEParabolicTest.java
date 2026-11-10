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
 * Parabolic problems (heat, diffusion, reaction-diffusion) against analytical
 * and manufactured solutions.
 */
public class PDEParabolicTest {

    private static final double PI = Math.PI;

    @ParameterizedTest
    @CsvSource({
        "rk4,       5e-4, 1e-4",
        "heun,      2e-4, 1e-4",
        "euler,     2e-4, 2e-4",
        "rk45,      5e-4, 1e-4",
        "crank_nicolson, 1e-3, 1e-4",
        "backward_euler, 2.5e-4, 1e-3"
    })
    void heat1d_everyIntegratorMatchesAnalyticSolution(String method, double dt, double tol) {
        PDEResult r = PDETestSupport.heat1d(0.1, 101).solve(0.5, dt, method);
        double err = r.maxError("u", PDETestSupport.heat1dExact(0.1));
        assertTrue(err < tol, method + " error " + err);
        assertTrue(r.isConverged());
        assertEquals(PDEType.PARABOLIC, r.getDiagnostics().getType());
        assertEquals(0.5, r.getTime(), 1e-12);
    }

    @Test
    void heat1d_temporalOrderOfCrankNicolsonIsTwo() {
        // 4th-order space on a fine grid pushes the spatial error far below the temporal one
        double[] e = new double[3];
        double[] dts = {4e-3, 2e-3, 1e-3};
        for (int i = 0; i < 3; i++) {
            PDEOptions o = new PDEOptions().spatialOrder(4).integrator("crank_nicolson").dt(dts[i]).krylovTolerance(1e-12).newtonTolerance(1e-13);
            e[i] = PDETestSupport.heat1d(0.1, 161).options(o).solve(0.2)
                    .maxError("u", PDETestSupport.heat1dExact(0.1));
        }
        double o1 = PDETestSupport.order(e[0], e[1], 2);
        double o2 = PDETestSupport.order(e[1], e[2], 2);
        assertTrue(o1 > 1.8 && o1 < 2.3, "order " + o1);
        assertTrue(o2 > 1.8 && o2 < 2.3, "order " + o2);
    }

    @Test
    void heat1d_temporalOrderOfBackwardEulerIsOne() {
        double[] e = new double[2];
        double[] dts = {4e-3, 2e-3};
        for (int i = 0; i < 2; i++) {
            PDEOptions o = new PDEOptions().spatialOrder(4).integrator("backward_euler").dt(dts[i]).krylovTolerance(1e-12).newtonTolerance(1e-13);
            e[i] = PDETestSupport.heat1d(0.1, 161).options(o).solve(0.2)
                    .maxError("u", PDETestSupport.heat1dExact(0.1));
        }
        double o = PDETestSupport.order(e[0], e[1], 2);
        assertTrue(o > 0.9 && o < 1.2, "order " + o);
    }

    @Test
    void heat1d_spatialOrderIsTwoAndFourForTheRespectiveStencils() {
        int[] ns = {11, 21, 41};
        for (int order : new int[]{2, 4}) {
            double[] e = new double[3];
            for (int i = 0; i < 3; i++) {
                PDEOptions o = new PDEOptions().spatialOrder(order).integrator("crank_nicolson").dt(2e-4)
                        .krylovTolerance(1e-13).newtonTolerance(1e-13);
                e[i] = PDETestSupport.heat1d(0.1, ns[i]).options(o).solve(0.1)
                        .maxError("u", PDETestSupport.heat1dExact(0.1));
            }
            double o1 = PDETestSupport.order(e[0], e[1], 2);
            double o2 = PDETestSupport.order(e[1], e[2], 2);
            double want = order;
            assertTrue(o1 > want - 0.35 && o2 > want - 0.35, "formal order " + order + " observed " + o1 + ", " + o2);
        }
    }

    @Test
    void heat2d_matchesAnalyticSolution() {
        PDEResult r = PDE.problem("diff(u,t) = a*(diff(u,x,2) + diff(u,y,2))").param("a", 0.05)
                .domain("x", 0, 1, 41).domain("y", 0, 1, 41)
                .initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0")
                .solve(0.2, 0.005, "crank_nicolson");
        double err = r.maxError("u", (x, y, z, t) -> Math.exp(-2 * PI * PI * 0.05 * t) * Math.sin(PI * x) * Math.sin(PI * y));
        assertTrue(err < 2e-4, "err " + err);
    }

    @Test
    void heat3d_matchesAnalyticSolution() {
        PDEResult r = PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2) + diff(u,z,2))")
                .domain("x", 0, 1, 17).domain("y", 0, 1, 17).domain("z", 0, 1, 17)
                .initial("u", "sin(pi*x)*sin(pi*y)*sin(pi*z)").dirichlet("u", "0")
                .solve(0.1, 0.002, "crank_nicolson");
        double err = r.maxError("u", (x, y, z, t) -> Math.exp(-3 * PI * PI * 0.1 * t)
                * Math.sin(PI * x) * Math.sin(PI * y) * Math.sin(PI * z));
        assertTrue(err < 2e-3, "err " + err);
    }

    @Test
    void heat1d_neumannBoundaries() {
        PDEResult r = PDE.problem("diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "cos(pi*x)").neumann("u", "0").solve(0.1, 1e-3, "crank_nicolson");
        double err = r.maxError("u", (x, y, z, t) -> Math.exp(-PI * PI * t) * Math.cos(PI * x));
        assertTrue(err < 1e-4, "err " + err);
    }

    @Test
    void heat1d_robinBoundary() {
        // u(0)=0 and u + du/dn = 0 at x=1: eigenfunctions sin(kx) with tan(k) = -k
        double k = 2.0287578381104342;
        PDEResult r = PDE.problem("diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 201)
                .initial("u", "sin(" + k + "*x)")
                .dirichlet(Face.X_MIN, "u", "0")
                .robin(Face.X_MAX, "u", "1", "1", "0")
                .solve(0.2, 5e-4, "crank_nicolson");
        double err = r.maxError("u", (x, y, z, t) -> Math.exp(-k * k * t) * Math.sin(k * x));
        assertTrue(err < 2e-4, "err " + err);
    }

    @Test
    void heat1d_periodic() {
        PDEResult r = PDE.problem("diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "cos(2*pi*x)").periodic("x").solve(0.05, 5e-4, "crank_nicolson");
        double err = r.maxError("u", (x, y, z, t) -> Math.exp(-4 * PI * PI * t) * Math.cos(2 * PI * x));
        assertTrue(err < 2e-4, "err " + err);
        assertEquals(r.value("u", 0), r.value("u", 100), 1e-12, "periodic end nodes coincide");
    }

    @Test
    void heat2d_neumannAndDirichletMixed() {
        // exact u = exp(-pi^2 t/ 4 ... ) use separable cos(pi x) (Neumann in x) * sin(pi y) (Dirichlet in y)
        PDEResult r = PDE.problem("diff(u,t) = diff(u,x,2) + diff(u,y,2)")
                .domain("x", 0, 1, 41).domain("y", 0, 1, 41)
                .initial("u", "cos(pi*x)*sin(pi*y)")
                .neumann(Face.X_MIN, "u", "0").neumann(Face.X_MAX, "u", "0")
                .dirichlet(Face.Y_MIN, "u", "0").dirichlet(Face.Y_MAX, "u", "0")
                .solve(0.05, 1e-3, "crank_nicolson");
        double err = r.maxError("u", (x, y, z, t) -> Math.exp(-2 * PI * PI * t) * Math.cos(PI * x) * Math.sin(PI * y));
        assertTrue(err < 5e-4, "err " + err);
    }

    @Test
    void timeDependentDirichletData_exactForLinearSolution() {
        // u_t = u_xx + x  with  u = t*x :  u(0,t)=0, u(1,t)=t
        PDEResult r = PDE.problem("diff(u,t) = diff(u,x,2) + x").domain("x", 0, 1, 51)
                .initial("u", "0").dirichlet(Face.X_MIN, "u", "0").dirichlet(Face.X_MAX, "u", "t")
                .solve(1.0, 2e-4, "rk4");
        double err = r.maxError("u", (x, y, z, t) -> t * x);
        assertTrue(err < 1e-10, "err " + err);
    }

    @ParameterizedTest
    @CsvSource({"crank_nicolson, 1e-2", "rk4, 1e-2", "rk45, 1e-2"})
    void nonlinearManufacturedSolution(String method, double tol) {
        // u = e^{-t} sin(pi x);  u_t = u_xx + u^2 + f
        String eq = "diff(u,t) = diff(u,x,2) + u^2 + (-exp(-t)*sin(pi*x) + pi^2*exp(-t)*sin(pi*x) - exp(-2*t)*sin(pi*x)^2)";
        PDEResult r = PDE.problem(eq).domain("x", 0, 1, 81).initial("u", "sin(pi*x)").dirichlet("u", "0")
                .solve(0.3, 1e-4, method);
        double err = r.maxError("u", (x, y, z, t) -> Math.exp(-t) * Math.sin(PI * x));
        assertTrue(err < tol * 0.1, method + " err " + err);
        assertTrue(r.isConverged());
    }

    @Test
    void nonlinearImplicitAndExplicitIntegratorsAgree() {
        String eq = "diff(u,t) = 0.01*diff(u,x,2) + u*(1-u)";
        PDEResult a = PDE.problem(eq).domain("x", 0, 1, 81).initial("u", "0.5+0.4*sin(pi*x)").neumann("u", "0")
                .solve(2.0, 0.002, "crank_nicolson");
        PDEResult b = PDE.problem(eq).domain("x", 0, 1, 81).initial("u", "0.5+0.4*sin(pi*x)").neumann("u", "0")
                .solve(2.0, 0.002, "rk4");
        double d = 0;
        for (int i = 0; i < 81; i++) {
            d = Math.max(d, Math.abs(a.value("u", i) - b.value("u", i)));
        }
        assertTrue(d < 1e-5, "CN vs RK4 " + d);
        assertTrue(a.getDiagnostics().getNewtonIterations() >= 1000, "Newton must have run");
    }

    @Test
    void mixedDerivativeTerm() {
        // u = e^{-t} sin(pi x) sin(pi y);  u_t = u_xx + u_yy + 0.5 u_xy + f
        String f = "(-exp(-t)*sin(pi*x)*sin(pi*y) + 2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y) - 0.5*pi^2*exp(-t)*cos(pi*x)*cos(pi*y))";
        PDEResult r = PDE.problem("diff(u,t) = diff(u,x,2) + diff(u,y,2) + 0.5*diff(u,x,y) + " + f)
                .domain("x", 0, 1, 41).domain("y", 0, 1, 41)
                .initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0").solve(0.2, 2e-3, "crank_nicolson");
        double err = r.maxError("u", (x, y, z, t) -> Math.exp(-t) * Math.sin(PI * x) * Math.sin(PI * y));
        assertTrue(err < 2e-3, "err " + err);
        // nested spelling must be identical
        PDEResult r2 = PDE.problem("diff(u,t) = diff(u,x,2) + diff(u,y,2) + 0.5*diff(diff(u,x),y) + " + f)
                .domain("x", 0, 1, 41).domain("y", 0, 1, 41)
                .initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0").solve(0.2, 2e-3, "crank_nicolson");
        assertArrayEquals(r.values("u"), r2.values("u"), 0.0);
    }

    @Test
    void coefficientOnTheTimeDerivativeIsHandled() {
        // 2 u_t = u_xx  is  u_t = 0.5 u_xx
        PDEResult a = PDE.problem("2*diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(0.5, 1e-4, "rk4");
        double err = a.maxError("u", PDETestSupport.heat1dExact(0.5));
        assertTrue(err < 1e-4, "err " + err);
        PDEResult b = PDE.problem("diff(u,t) - 0.5*diff(u,x,2) = 0").domain("x", 0, 1, 101)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(0.5, 1e-4, "rk4");
        assertTrue(b.maxError("u", PDETestSupport.heat1dExact(0.5)) < 1e-4);
        PDEResult c = PDE.problem("2*diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(0.5, 5e-4, "crank_nicolson");
        assertTrue(c.maxError("u", PDETestSupport.heat1dExact(0.5)) < 1e-4, "general form + implicit (finite-difference Jacobian)");
    }

    @Test
    void adaptiveIntegratorHonoursItsTolerance() {
        PDEOptions loose = new PDEOptions().integrator("rk45").rtol(1e-4).atol(1e-6).spatialOrder(4);
        PDEOptions tight = new PDEOptions().integrator("rk45").rtol(1e-9).atol(1e-12).spatialOrder(4);
        PDEResult a = PDETestSupport.heat1d(0.1, 61).options(loose).solve(0.2);
        PDEResult b = PDETestSupport.heat1d(0.1, 61).options(tight).solve(0.2);
        double ea = a.maxError("u", PDETestSupport.heat1dExact(0.1));
        double eb = b.maxError("u", PDETestSupport.heat1dExact(0.1));
        assertTrue(eb < ea, "tight tolerance must be more accurate: " + eb + " vs " + ea);
        assertTrue(b.getDiagnostics().getSteps() > a.getDiagnostics().getSteps());
        assertFalse(Double.isNaN(b.getDiagnostics().getErrorEstimate()));
        assertTrue(a.getDiagnostics().getMinDt() < a.getDiagnostics().getMaxDt() * 1.0000001);
    }
}
