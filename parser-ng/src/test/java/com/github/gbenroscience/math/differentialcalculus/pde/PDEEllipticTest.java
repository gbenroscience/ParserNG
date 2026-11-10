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
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Elliptic problems (Laplace, Poisson, variable-coefficient, nonlinear and
 * coupled) and the behaviour of the iterative solvers.
 */
public class PDEEllipticTest {

    private static final double PI = Math.PI;

    private static PDEProblem.Builder poisson2d(int n) {
        return PDE.problem("diff(u,x,2) + diff(u,y,2) = -2*pi^2*sin(pi*x)*sin(pi*y)")
                .domain("x", 0, 1, n).domain("y", 0, 1, n).dirichlet("u", "0");
    }

    private static final PDEResult.Exact POISSON = (x, y, z, t) -> Math.sin(PI * x) * Math.sin(PI * y);

    @Test
    void laplace2d_separableSolution() {
        PDEResult r = PDE.problem("diff(u,x,2) + diff(u,y,2) = 0").domain("x", 0, 1, 33).domain("y", 0, 1, 33)
                .dirichlet("u", "0").dirichlet(Face.Y_MAX, "u", "sin(pi*x)").solveSteady("sor");
        double err = r.maxError("u", (x, y, z, t) -> Math.sin(PI * x) * Math.sinh(PI * y) / Math.sinh(PI));
        assertTrue(err < 5e-4, "err " + err);
        assertTrue(r.isConverged());
        assertEquals(PDEType.ELLIPTIC, r.getDiagnostics().getType());
        assertTrue(r.getDiagnostics().getResidualNorm() < 1e-6);
        assertTrue(r.getDiagnostics().getIterations() > 10);
        // maximum principle
        for (double v : r.values("u")) {
            assertTrue(v >= -1e-9 && v <= 1 + 1e-9);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"jacobi", "gauss_seidel", "sor", "red_black_sor", "newton_krylov", "auto"})
    void poisson2d_everyMethodReachesTheSameDiscreteSolution(String method) {
        PDEResult r = poisson2d(17).solveSteady(method);
        assertTrue(r.isConverged(), method);
        assertEquals(3.2e-3, r.maxError("u", POISSON), 3e-4, method + " discretisation error");
    }

    @Test
    void poisson2d_sorBeatsGaussSeidelBeatsJacobi() {
        int it[] = new int[3];
        String[] m = {"jacobi", "gauss_seidel", "sor"};
        for (int i = 0; i < 3; i++) {
            it[i] = poisson2d(17).solveSteady(m[i]).getDiagnostics().getIterations();
        }
        assertTrue(it[0] > it[1] && it[1] > 3 * it[2], "iterations " + java.util.Arrays.toString(it));
        // Gauss-Seidel needs about half the sweeps of Jacobi
        assertEquals(2.0, it[0] / (double) it[1], 0.2);
    }

    @Test
    void poisson2d_secondOrderConvergence() {
        double[] e = new double[3];
        int[] ns = {9, 17, 33};
        for (int i = 0; i < 3; i++) {
            e[i] = poisson2d(ns[i]).solveSteady("newton_krylov").maxError("u", POISSON);
        }
        double o1 = PDETestSupport.order(e[0], e[1], 2);
        double o2 = PDETestSupport.order(e[1], e[2], 2);
        assertTrue(o1 > 1.9 && o1 < 2.3 && o2 > 1.9 && o2 < 2.3, "orders " + o1 + ", " + o2);
    }

    @Test
    void poisson2d_fourthOrderConvergence() {
        double[] e = new double[3];
        int[] ns = {9, 17, 33};
        for (int i = 0; i < 3; i++) {
            PDEOptions o = new PDEOptions().spatialOrder(4).ellipticMethod("newton_krylov").krylovTolerance(1e-12).newtonTolerance(1e-13);
            e[i] = poisson2d(ns[i]).options(o).solveSteady().maxError("u", POISSON);
        }
        double o2 = PDETestSupport.order(e[1], e[2], 2);
        assertTrue(o2 > 3.5, "orders " + PDETestSupport.order(e[0], e[1], 2) + ", " + o2);
    }

    @Test
    void poisson3d() {
        PDEResult r = PDE.problem("diff(u,x,2)+diff(u,y,2)+diff(u,z,2) = -3*pi^2*sin(pi*x)*sin(pi*y)*sin(pi*z)")
                .domain("x", 0, 1, 17).domain("y", 0, 1, 17).domain("z", 0, 1, 17).dirichlet("u", "0").solveSteady();
        double err = r.maxError("u", (x, y, z, t) -> Math.sin(PI * x) * Math.sin(PI * y) * Math.sin(PI * z));
        assertTrue(err < 1e-2, "err " + err);
        assertTrue(r.isConverged());
    }

    @Test
    void neumannAtTheUpperFace_exactForQuadraticSolutions() {
        // u'' = -2, u(0) = 0, du/dn(1) = u'(1) = 1  ->  u = 3x - x^2 (reproduced exactly by the one-sided formula)
        PDEResult r = PDE.problem("diff(u,x,2) = -2").domain("x", 0, 1, 21)
                .dirichlet(Face.X_MIN, "u", "0").neumann(Face.X_MAX, "u", "1").solveSteady("gauss_seidel");
        assertTrue(r.maxError("u", (x, y, z, t) -> 3 * x - x * x) < 1e-7);
    }

    @Test
    void robinBoundary_exactForLinearSolutions() {
        // u'' = 0, u(0) = 1, u + du/dn = 3 at x = 1  ->  u = 1 + x
        PDEResult r = PDE.problem("diff(u,x,2) = 0").domain("x", 0, 1, 21)
                .dirichlet(Face.X_MIN, "u", "1").robin(Face.X_MAX, "u", "1", "1", "3").solveSteady("sor");
        assertTrue(r.maxError("u", (x, y, z, t) -> 1 + x) < 1e-7);
    }

    @Test
    void mixedNeumannDirichlet2d_exactForLinearSolutions() {
        PDEResult r = PDE.problem("diff(u,x,2) + diff(u,y,2) = 0").domain("x", 0, 1, 15).domain("y", 0, 1, 15)
                .dirichlet(Face.X_MIN, "u", "3*y").dirichlet(Face.X_MAX, "u", "2 + 3*y")
                .dirichlet(Face.Y_MIN, "u", "2*x").neumann(Face.Y_MAX, "u", "3")
                .solveSteady("newton_krylov");
        assertTrue(r.maxError("u", (x, y, z, t) -> 2 * x + 3 * y) < 1e-6);
    }

    @Test
    void periodicInOneDirection() {
        PDEResult r = PDE.problem("diff(u,x,2) + diff(u,y,2) = -5*pi^2*sin(2*pi*x)*sin(pi*y)")
                .domain("x", 0, 1, 41).domain("y", 0, 1, 21).periodic("x").dirichlet("u", "0").solveSteady("newton_krylov");
        double err = r.maxError("u", (x, y, z, t) -> Math.sin(2 * PI * x) * Math.sin(PI * y));
        assertTrue(err < 5e-3, "err " + err);
    }

    @Test
    void variableCoefficientOperator() {
        // (1+x) u_xx + u_yy = f with u = sin(pi x) sin(pi y)
        PDEResult a = PDE.problem("(1+x)*diff(u,x,2) + diff(u,y,2) = -(1+x)*pi^2*sin(pi*x)*sin(pi*y) - pi^2*sin(pi*x)*sin(pi*y)")
                .domain("x", 0, 1, 33).domain("y", 0, 1, 33).dirichlet("u", "0").solveSteady("sor");
        assertTrue(a.maxError("u", POISSON) < 2e-3);
        PDEResult b = PDE.problem("(1+x)*diff(u,x,2) + diff(u,y,2) = -(1+x)*pi^2*sin(pi*x)*sin(pi*y) - pi^2*sin(pi*x)*sin(pi*y)")
                .domain("x", 0, 1, 33).domain("y", 0, 1, 33).dirichlet("u", "0").solveSteady("newton_krylov");
        assertEquals(0.0, maxDiff(a.values("u"), b.values("u")), 1e-6);
    }

    @Test
    void nonlinearEllipticProblem() {
        String eq = "diff(u,x,2) + u^2 = -pi^2*sin(pi*x) + sin(pi*x)^2";
        PDEResult nk = PDE.problem(eq).domain("x", 0, 1, 101).dirichlet("u", "0").solveSteady("newton_krylov");
        assertTrue(nk.maxError("u", (x, y, z, t) -> Math.sin(PI * x)) < 2e-4);
        assertTrue(nk.getDiagnostics().getNewtonIterations() >= 2, "a nonlinear problem needs several Newton steps");
        PDEResult gs = PDE.problem(eq).domain("x", 0, 1, 101).dirichlet("u", "0").solveSteady("gauss_seidel");
        assertTrue(gs.isConverged());
        assertEquals(0.0, maxDiff(nk.values("u"), gs.values("u")), 1e-7);
    }

    @Test
    void coupledEllipticSystem() {
        // u'' + v = f1, v'' - u = f2 with u = sin(pi x), v = x(1-x)
        PDEResult r = PDE.problem("diff(u,x,2) + v = -pi^2*sin(pi*x) + x*(1-x)", "diff(v,x,2) - u = -2 - sin(pi*x)")
                .domain("x", 0, 1, 101).dirichlet("*", "0").solveSteady();
        assertEquals("NEWTON_KRYLOV", r.getDiagnostics().getMethod().split(" ")[0], "systems default to Newton-Krylov");
        assertTrue(r.maxError("u", (x, y, z, t) -> Math.sin(PI * x)) < 3e-4);
        assertTrue(r.maxError("v", (x, y, z, t) -> x * (1 - x)) < 5e-5, "v inherits the O(h^2) error of u through the coupling");
    }

    @Test
    void newtonKrylovConvergesFromAColdStartOnLargerGrids() {
        // regression: the Newton tolerance must scale with the current iterate, not with the (zero) initial guess
        PDEResult r = poisson2d(129).solveSteady("newton_krylov");
        assertTrue(r.isConverged(), r.getDiagnostics().toString());
        assertTrue(r.getDiagnostics().getNewtonIterations() <= 3);
        assertTrue(r.maxError("u", POISSON) < 3e-4);
    }

    @Test
    void nonConvergenceIsDetectedAndReported() {
        PDEOptions o = new PDEOptions().ellipticMethod("jacobi").maxIterations(5);
        PDEResult r = poisson2d(17).options(o).solveSteady();
        assertFalse(r.isConverged());
        assertFalse(r.getDiagnostics().getWarnings().isEmpty());
        assertEquals(5, r.getDiagnostics().getIterations());
        assertTrue(r.getDiagnostics().getResidualNorm() > 1e-3);
    }

    @Test
    void nonConvergenceCanRaiseAnException() {
        PDEOptions o = new PDEOptions().ellipticMethod("jacobi").maxIterations(5).failOnNonConvergence(true);
        NonConvergenceException ex = assertThrows(NonConvergenceException.class, () -> poisson2d(17).options(o).solveSteady());
        assertNotNull(ex.getDiagnostics());
    }

    @Test
    void unsolvableNeumannProblemDoesNotPretendToConverge() {
        // u'' = 1 with homogeneous Neumann data on both ends has no solution (compatibility condition violated)
        PDEOptions o = new PDEOptions().ellipticMethod("gauss_seidel").maxIterations(3000);
        PDEResult r = PDE.problem("diff(u,x,2) = 1").domain("x", 0, 1, 21).neumann("u", "0").options(o).solveSteady();
        assertFalse(r.isConverged());
    }

    @Test
    void diagnosticsExposeResidualsAndErrorEstimates() {
        PDEResult r = poisson2d(17).solveSteady("sor");
        SolverDiagnostics d = r.getDiagnostics();
        assertTrue(d.isConverged());
        assertTrue(d.getResidualNorm() < d.getInitialResidualNorm() * 1e-8);
        assertTrue(d.getContractionFactor() > 0 && d.getContractionFactor() < 1);
        assertFalse(Double.isNaN(d.getErrorEstimate()));
        assertTrue(d.getOmega() > 1 && d.getOmega() < 2);
    }

    @Test
    void redBlackOrderingIsRejectedWhenTheStencilForbidsIt() {
        PDEConfigurationException ex = assertThrows(PDEConfigurationException.class,
                () -> PDE.problem("diff(u,x,2) + diff(u,y,2) + diff(u,x,y) = 0").domain("x", 0, 1, 9).domain("y", 0, 1, 9)
                        .dirichlet("u", "0").solveSteady("red_black_sor"));
        assertTrue(ex.getMessage().contains("RED_BLACK_SOR"));
    }

    @Test
    void parallelSolveIsBitIdenticalToSerial() {
        PDEResult a = poisson2d(33).options(new PDEOptions().ellipticMethod("red_black_sor").threads(1)).solveSteady();
        PDEResult b = poisson2d(33).options(new PDEOptions().ellipticMethod("red_black_sor").threads(3)).solveSteady();
        assertArrayEquals(a.values("u"), b.values("u"), 0.0);
        assertEquals(a.getDiagnostics().getIterations(), b.getDiagnostics().getIterations());
    }

    private static double maxDiff(double[] a, double[] b) {
        double m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}
