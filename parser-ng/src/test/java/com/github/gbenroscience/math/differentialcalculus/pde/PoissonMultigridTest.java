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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.github.gbenroscience.math.differentialcalculus.pde.PoissonMultigrid.BC;
import com.github.gbenroscience.math.differentialcalculus.pde.PoissonMultigrid.Cycle;
import com.github.gbenroscience.math.differentialcalculus.pde.PoissonMultigrid.Smoother;
import com.github.gbenroscience.math.differentialcalculus.pde.PoissonMultigrid.Stats;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Geometric multigrid: convergence rates, accuracy, boundary conditions, smoothers, 3-D and allocation. */
public class PoissonMultigridTest {

    private static final double PI = Math.PI;

    private static BC[][] all(BC b) {
        return new BC[][]{{b, b}, {b, b}, {b, b}};
    }

    private static PoissonMultigrid sine2d(int n, Smoother s, Cycle c) {
        double h = 1.0 / n;
        PoissonMultigrid mg = new PoissonMultigrid(2, new int[]{n, n}, new double[]{h, h}, all(BC.DIRICHLET))
                .smoother(s).cycle(c).tolerance(1e-10, 1e-14, 60);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                mg.rhs()[mg.index(i, j, 0)] = -2 * PI * PI * Math.sin(PI * (i + .5) * h) * Math.sin(PI * (j + .5) * h);
            }
        }
        return mg;
    }

    private static double sine2dError(PoissonMultigrid mg, int n) {
        double h = 1.0 / n;
        double e = 0;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                e = Math.max(e, Math.abs(mg.phi()[mg.index(i, j, 0)] - Math.sin(PI * (i + .5) * h) * Math.sin(PI * (j + .5) * h)));
            }
        }
        return e;
    }

    @Test
    void convergenceFactorIsIndependentOfTheGridSize() {
        double[] rho = new double[4];
        int[] ns = {16, 32, 64, 128};
        for (int q = 0; q < ns.length; q++) {
            PoissonMultigrid mg = sine2d(ns[q], Smoother.RED_BLACK_GS, Cycle.V);
            Stats s = mg.solve();
            assertTrue(s.converged, "n=" + ns[q]);
            rho[q] = s.convergenceFactor;
            assertTrue(s.convergenceFactor < 0.15, "n=" + ns[q] + " factor " + s.convergenceFactor);
            assertTrue(s.cycles <= 14, "cycles " + s.cycles);
        }
        assertTrue(rho[3] < 2.0 * rho[0] + 0.03, "factor must not grow with the grid: " + java.util.Arrays.toString(rho));
    }

    @Test
    void solutionIsSecondOrderAccurate() {
        double[] e = new double[3];
        int[] ns = {16, 32, 64};
        for (int q = 0; q < 3; q++) {
            PoissonMultigrid mg = sine2d(ns[q], Smoother.RED_BLACK_GS, Cycle.V);
            mg.solve();
            e[q] = sine2dError(mg, ns[q]);
        }
        assertEquals(2.0, PDETestSupport.order(e[0], e[1], 2), 0.15);
        assertEquals(2.0, PDETestSupport.order(e[1], e[2], 2), 0.1);
    }

    @ParameterizedTest
    @EnumSource(Smoother.class)
    void everySmootherConverges(Smoother s) {
        PoissonMultigrid mg = sine2d(64, s, Cycle.V);
        Stats st = mg.solve();
        assertTrue(st.converged, s + " " + st.residual);
        assertTrue(sine2dError(mg, 64) < 1e-3);
    }

    @Test
    void wCyclesConvergeFasterPerCycleThanVCycles() {
        double v = sine2d(64, Smoother.RED_BLACK_GS, Cycle.V).solve().convergenceFactor;
        double w = sine2d(64, Smoother.RED_BLACK_GS, Cycle.W).solve().convergenceFactor;
        assertTrue(w < v, "W " + w + " vs V " + v);
    }

    @Test
    void tenVCyclesReduceTheResidualByManyOrdersOfMagnitude() {
        PoissonMultigrid mg = sine2d(64, Smoother.RED_BLACK_GS, Cycle.V).tolerance(0, 0, 10);
        Stats st = mg.solve();
        assertEquals(10, st.cycles);
        assertTrue(st.residual < 1e-7 * st.initialResidual, "reduction " + st.initialResidual / st.residual);
        assertTrue(mg.levels() >= 5, "levels " + mg.levels());
    }

    @Test
    void periodicNeumannSingularProblemIn3D() {
        int n = 32;
        double h = 1.0 / n;
        BC[][] bc = {{BC.PERIODIC, BC.PERIODIC}, {BC.NEUMANN, BC.NEUMANN}, {BC.PERIODIC, BC.PERIODIC}};
        PoissonMultigrid mg = new PoissonMultigrid(3, new int[]{n, n, n}, new double[]{h, h, h}, bc);
        assertTrue(mg.isSingular());
        double lam = (2 * PI) * (2 * PI) * 2 + PI * PI;
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    mg.rhs()[mg.index(i, j, k)] = -lam * Math.sin(2 * PI * (i + .5) * h) * Math.cos(PI * (j + .5) * h) * Math.cos(2 * PI * (k + .5) * h);
                }
            }
        }
        Stats s = mg.solve();
        assertTrue(s.converged);
        double err = 0;
        double mean = 0;
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    double ex = Math.sin(2 * PI * (i + .5) * h) * Math.cos(PI * (j + .5) * h) * Math.cos(2 * PI * (k + .5) * h);
                    double v = mg.phi()[mg.index(i, j, k)];
                    err = Math.max(err, Math.abs(v - ex));
                    mean += v;
                }
            }
        }
        assertTrue(err < 5e-3, "err " + err);
        assertEquals(0.0, mean / (n * n * n), 1e-12, "singular problems are normalised to zero mean");
        assertTrue(mg.levels() >= 4);
    }

    @Test
    void mixedNeumannDirichletConditions() {
        // phi = cos(pi x) sinh-free: phi_xx = -pi^2 cos(pi x) with Neumann in x, and phi = 0 (Dirichlet) at both y faces for sin(pi y)
        int n = 64;
        double h = 1.0 / n;
        BC[][] bc = {{BC.NEUMANN, BC.NEUMANN}, {BC.DIRICHLET, BC.DIRICHLET}, {BC.NEUMANN, BC.NEUMANN}};
        PoissonMultigrid mg = new PoissonMultigrid(2, new int[]{n, n}, new double[]{h, h}, bc);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                mg.rhs()[mg.index(i, j, 0)] = -2 * PI * PI * Math.cos(PI * (i + .5) * h) * Math.sin(PI * (j + .5) * h);
            }
        }
        assertTrue(mg.solve().converged);
        double err = 0;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                err = Math.max(err, Math.abs(mg.phi()[mg.index(i, j, 0)] - Math.cos(PI * (i + .5) * h) * Math.sin(PI * (j + .5) * h)));
            }
        }
        assertTrue(err < 1e-3, "err " + err);
    }

    @Test
    void anisotropicSpacingStillConverges() {
        int nx = 64;
        int ny = 32;
        double hx = 2.0 / nx;
        double hy = 1.0 / ny;
        PoissonMultigrid mg = new PoissonMultigrid(2, new int[]{nx, ny}, new double[]{hx, hy}, all(BC.DIRICHLET));
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                mg.rhs()[mg.index(i, j, 0)] = -(PI * PI / 4 + PI * PI) * Math.sin(PI * (i + .5) * hx / 2) * Math.sin(PI * (j + .5) * hy);
            }
        }
        Stats s = mg.solve();
        assertTrue(s.converged);
        double err = 0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                err = Math.max(err, Math.abs(mg.phi()[mg.index(i, j, 0)] - Math.sin(PI * (i + .5) * hx / 2) * Math.sin(PI * (j + .5) * hy)));
            }
        }
        assertTrue(err < 2e-3, "err " + err);
    }

    @Test
    void invalidConfigurationsAreRejected() {
        BC[][] bad = {{BC.PERIODIC, BC.NEUMANN}, {BC.NEUMANN, BC.NEUMANN}, {BC.NEUMANN, BC.NEUMANN}};
        assertThrows(PDEConfigurationException.class, () -> new PoissonMultigrid(2, new int[]{16, 16}, new double[]{1, 1}, bad));
        assertThrows(PDEConfigurationException.class, () -> new PoissonMultigrid(1, new int[]{16, 16}, new double[]{1, 1}, all(BC.NEUMANN)));
    }

    @Test
    void gridsThatDoNotHalveStillSolveWithFewerLevels() {
        int n = 36;      // 36 -> 18 -> 9 : three levels
        PoissonMultigrid mg = sine2d(n, Smoother.RED_BLACK_GS, Cycle.V);
        assertEquals(3, mg.levels());
        assertTrue(mg.solve().converged);
        assertTrue(sine2dError(mg, n) < 2e-3);
    }

    @Test
    void solveAllocatesNothing() {
        long id = Thread.currentThread().getId();
        assumeTrue(allocated(id) >= 0, "allocation counters unavailable");
        PoissonMultigrid mg = sine2d(64, Smoother.RED_BLACK_GS, Cycle.V);
        for (int i = 0; i < 300; i++) {
            java.util.Arrays.fill(mg.phi(), 0.0);
            mg.solve();
        }
        long a0 = allocated(id);
        for (int i = 0; i < 50; i++) {
            java.util.Arrays.fill(mg.phi(), 0.0);
            mg.solve();
        }
        long a1 = allocated(id);
        for (int i = 0; i < 250; i++) {
            java.util.Arrays.fill(mg.phi(), 0.0);
            mg.solve();
        }
        long a2 = allocated(id);
        double perSolve = ((a2 - a1) - (a1 - a0) * 5.0) / 200.0;   // slope between 50 and 250 solves
        assertTrue(perSolve < 16.0, perSolve + " bytes per solve");
    }

    private static long allocated(long id) {
        try {
            Method m = Class.forName("com.sun.management.ThreadMXBean").getMethod("getThreadAllocatedBytes", long.class);
            return (Long) m.invoke(ManagementFactory.getThreadMXBean(), id);
        } catch (Throwable t) {
            return -1;
        }
    }
}
