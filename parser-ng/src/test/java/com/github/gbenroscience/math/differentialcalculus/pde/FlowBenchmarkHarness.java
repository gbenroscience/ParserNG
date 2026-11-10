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

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import java.util.Locale;

/**
 * Benchmark harness for the incompressible flow solver and the multigrid
 * (time per step and per pressure solve, throughput, cycle counts, allocation
 * and GC). Run: {@code java -Xmx2g -cp ... ...pde.FlowBenchmarkHarness}.
 * Single-threaded; numbers are machine dependent.
 */
public final class FlowBenchmarkHarness {

    private FlowBenchmarkHarness() {
    }

    private static long allocated() {
        try {
            Method m = Class.forName("com.sun.management.ThreadMXBean").getMethod("getThreadAllocatedBytes", long.class);
            return (Long) m.invoke(ManagementFactory.getThreadMXBean(), Thread.currentThread().getId());
        } catch (Throwable t) {
            return -1;
        }
    }

    private static long gcCount() {
        long c = 0;
        for (GarbageCollectorMXBean g : ManagementFactory.getGarbageCollectorMXBeans()) {
            c += Math.max(0, g.getCollectionCount());
        }
        return c;
    }

    private static void flow(String name, IncompressibleFlow f, double dt, int warm, int steps) {
        f.initialize();
        for (int i = 0; i < warm; i++) {
            f.step(dt);
        }
        long g0 = gcCount();
        long a0 = allocated();
        for (int i = 0; i < steps / 5; i++) {
            f.step(dt);
        }
        long a1 = allocated();
        long t0 = System.nanoTime();
        for (int i = 0; i < steps; i++) {
            f.step(dt);
        }
        long t1 = System.nanoTime();
        long a2 = allocated();
        long g1 = gcCount();
        // bytes(n) = c + s*n for a run of n steps (c = cost of the measurement itself): the slope s is the per-step allocation
        double nShort = steps / 5.0;
        double nLong = steps;
        double alloc = ((a2 - a1) - (a1 - a0)) / (nLong - nShort);
        double perStep = (t1 - t0) / (double) steps;
        double cells = 1;
        for (int a = 0; a < f.dimension(); a++) {
            cells *= f.cells(a);
        }
        FlowDiagnostics d = f.diagnostics();
        System.out.println(String.format(Locale.ROOT, "%-34s %9.0f cells  %8.3f ms/step  %7.2f Mcell-steps/s  poisson %2d cycles  alloc %5.1f B/step  GC %d  div %.1e",
                name, cells, perStep / 1e6, cells / perStep * 1e3, d.getPressureCycles(), Math.max(0, alloc), g1 - g0, d.getDivergenceMax()));
    }

    public static void main(String[] args) {
        System.out.println("Flow / multigrid benchmark  JVM " + System.getProperty("java.version") + ", cores=" + Runtime.getRuntime().availableProcessors());
        System.out.println("\n== 2-D lid-driven cavity, Re=1000 (explicit AB2 + projection, multigrid pressure solve)");
        for (int n : new int[]{64, 128, 256}) {
            IncompressibleFlow f = IncompressibleFlow.builder().cells(n, n).reynolds(1000)
                    .boundary(Face.Y_MAX, FlowBoundary.movingWall("1", "0")).build();
            flow("cavity " + n + "^2", f, 0.2 / n, 200, 200);
        }
        System.out.println("\n== 2-D Taylor-Green (periodic)");
        for (int n : new int[]{64, 256}) {
            IncompressibleFlow f = IncompressibleFlow.builder().cells(n, n).size(2 * Math.PI, 2 * Math.PI).viscosity(0.01)
                    .periodic("x").periodic("y").initialVelocity("-cos(x)*sin(y)", "sin(x)*cos(y)").build();
            flow("taylor-green " + n + "^2", f, 0.2 * 2 * Math.PI / n, 100, 100);
        }
        System.out.println("\n== 3-D lid-driven cavity, Re=400");
        for (int n : new int[]{24, 32, 48}) {
            IncompressibleFlow f = IncompressibleFlow.builder().cells(n, n, n).reynolds(400)
                    .boundary(Face.Z_MAX, FlowBoundary.movingWall("1", "0", "0")).build();
            flow("cavity3d " + n + "^3", f, 0.2 / n, 50, 40);
        }
        System.out.println("\n== Pressure-Poisson multigrid alone (2-D, Dirichlet, V(2,2) red-black GS)");
        for (int n : new int[]{64, 128, 256, 512, 1024}) {
            double h = 1.0 / n;
            PoissonMultigrid.BC d = PoissonMultigrid.BC.DIRICHLET;
            PoissonMultigrid mg = new PoissonMultigrid(2, new int[]{n, n}, new double[]{h, h}, new PoissonMultigrid.BC[][]{{d, d}, {d, d}, {d, d}})
                    .tolerance(1e-10, 1e-14, 60);
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    mg.rhs()[mg.index(i, j, 0)] = -2 * Math.PI * Math.PI * Math.sin(Math.PI * (i + .5) * h) * Math.sin(Math.PI * (j + .5) * h);
                }
            }
            mg.solve();
            long t0 = System.nanoTime();
            int reps = Math.max(2, 4_000_000 / (n * n));
            PoissonMultigrid.Stats st = null;
            for (int r = 0; r < reps; r++) {
                java.util.Arrays.fill(mg.phi(), 0.0);
                st = mg.solve();
            }
            double ms = (System.nanoTime() - t0) / 1e6 / reps;
            System.out.println(String.format(Locale.ROOT, "multigrid %5d^2  levels=%d  %2d cycles  factor %.3f  %9.2f ms/solve  %7.1f ns per cell per cycle",
                    n, mg.levels(), st.cycles, st.convergenceFactor, ms, ms * 1e6 / ((double) n * n * st.cycles)));
        }
    }
}
