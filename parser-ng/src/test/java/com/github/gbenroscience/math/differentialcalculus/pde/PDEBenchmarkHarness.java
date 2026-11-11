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

import com.github.gbenroscience.util.AllocationMeter;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import java.util.Locale;

/**
 * Dependency-free benchmark harness for the PDE engine (the JMH equivalents
 * live in the {@code parser-ng-bench} module). Run with
 * <pre>java -Xmx3g -cp ... com.github.gbenroscience.math.differentialcalculus.pde.PDEBenchmarkHarness [quick|full]</pre>
 * For each workload it reports time per step, grid-point updates per second,
 * time per million updates, bytes allocated per step (stepping thread plus
 * every engine worker), GC activity during the measured window, and, where a
 * hand-written equivalent exists, the slowdown relative to it.
 * <p>
 * This is a measurement tool, not a test: numbers depend on the machine.
 */
public final class PDEBenchmarkHarness {

    private PDEBenchmarkHarness() {
    }

    // ------------------------------------------------------------ measurement plumbing
    private static long allocated(long id) {
        return AllocationMeter.allocatedBytes(id);
    }

    private static long[] engineThreads() {
        java.lang.management.ThreadMXBean b = ManagementFactory.getThreadMXBean();
        java.util.List<Long> ids = new java.util.ArrayList<Long>();
        ids.add(Thread.currentThread().getId());
        for (java.lang.management.ThreadInfo ti : b.getThreadInfo(b.getAllThreadIds())) {
            if (ti != null && ti.getThreadName().startsWith("pde-worker-")) {
                ids.add(ti.getThreadId());
            }
        }
        long[] r = new long[ids.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = ids.get(i);
        }
        return r;
    }

    private static long alloc(long[] ids) {
        long s = 0;
        for (long id : ids) {
            s += allocated(id);
        }
        return s;
    }

    private static long gcCount() {
        long c = 0;
        for (GarbageCollectorMXBean g : ManagementFactory.getGarbageCollectorMXBeans()) {
            c += Math.max(0, g.getCollectionCount());
        }
        return c;
    }

    private static long gcMillis() {
        long c = 0;
        for (GarbageCollectorMXBean g : ManagementFactory.getGarbageCollectorMXBeans()) {
            c += Math.max(0, g.getCollectionTime());
        }
        return c;
    }

    /** Result of one measured run. */
    static final class Stat {

        double nsPerStep;
        double mupdPerSec;
        double allocPerStep;
        long gcs;
        long gcMs;
        long heapMb;
    }

    /** Warms up, then times roughly {@code seconds} of stepping. */
    static Stat measure(PDESolver s, double dt, double seconds, int warmSteps) {
        s.initialize();
        for (int i = 0; i < warmSteps; i++) {
            s.step(dt);
        }
        // calibrate
        long t0 = System.nanoTime();
        int probe = 0;
        while (System.nanoTime() - t0 < 2e8) {
            s.step(dt);
            probe++;
        }
        double perStep = (System.nanoTime() - t0) / (double) probe;
        int steps = (int) Math.max(5, seconds * 1e9 / perStep);
        long[] ids = engineThreads();
        long gc0 = gcCount();
        long gm0 = gcMillis();
        long a0 = alloc(ids);
        long b0 = alloc(ids);
        for (int i = 0; i < steps / 5; i++) {
            s.step(dt);
        }
        long a1 = alloc(ids);
        long tStart = System.nanoTime();
        for (int i = 0; i < steps; i++) {
            s.step(dt);
        }
        long tEnd = System.nanoTime();
        long a2 = alloc(ids);
        Stat st = new Stat();
        st.nsPerStep = (tEnd - tStart) / (double) steps;
        double nodes = s.grid().nodeCount();
        st.mupdPerSec = nodes / st.nsPerStep * 1e3;
        st.allocPerStep = ((a2 - a1) - (a1 - a0)) / (double) (steps - steps / 5);
        st.gcs = gcCount() - gc0;
        st.gcMs = gcMillis() - gm0;
        Runtime rt = Runtime.getRuntime();
        st.heapMb = (rt.totalMemory() - rt.freeMemory()) >> 20;
        return st;
    }

    private static void row(String name, PDESolver s, Stat st, double baselineNsPerStep) {
        double nodes = s.grid().nodeCount();
        String base = Double.isNaN(baselineNsPerStep) ? "     -" : String.format(Locale.ROOT, "%5.1fx", st.nsPerStep / baselineNsPerStep);
        System.out.println(String.format(Locale.ROOT,
                "%-34s %10.0f %11.3f %9.1f %9.1f %7.2f %4d/%-4d %6d %7s",
                name, nodes, st.nsPerStep / 1e6, st.mupdPerSec, 1e3 / st.mupdPerSec,
                Math.abs(st.allocPerStep) < 64 ? 0.0 : st.allocPerStep, st.gcs, st.gcMs, st.heapMb, base));
    }

    private static void header(String title) {
        System.out.println();
        System.out.println("== " + title);
        System.out.println(String.format("%-34s %10s %11s %9s %9s %7s %9s %6s %7s", "workload", "nodes", "ms/step",
                "Mupd/s", "ms/Mupd", "B/step", "GC#/ms", "heapMB", "vs hand"));
    }

    // ------------------------------------------------------------ hand-written baselines
    /** Forward-Euler heat equation, 5-point stencil, flat array: the cost floor of the algorithm. */
    private static double baselineHeat2d(int n, int steps) {
        double h = 1.0 / (n - 1);
        double c = 0.1 * 1e-6 / (h * h);
        double[] u = new double[n * n];
        double[] v = new double[n * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                u[j * n + i] = Math.sin(Math.PI * i * h) * Math.sin(Math.PI * j * h);
            }
        }
        double best = Double.MAX_VALUE;
        for (int rep = 0; rep < 3; rep++) {
            long t0 = System.nanoTime();
            for (int st = 0; st < steps; st++) {
                for (int j = 1; j < n - 1; j++) {
                    int row = j * n;
                    for (int i = 1; i < n - 1; i++) {
                        int k = row + i;
                        v[k] = u[k] + c * (u[k - 1] + u[k + 1] + u[k - n] + u[k + n] - 4 * u[k]);
                    }
                }
                double[] t = u;
                u = v;
                v = t;
            }
            best = Math.min(best, (System.nanoTime() - t0) / (double) steps);
        }
        return best;
    }

    /** Velocity-Verlet wave equation, 5-point stencil. */
    private static double baselineWave2d(int n, int steps) {
        double h = 1.0 / (n - 1);
        double dt = 1e-5;
        double c = 1.0 / (h * h);
        double[] u = new double[n * n];
        double[] vel = new double[n * n];
        double[] acc = new double[n * n];
        double[] acc2 = new double[n * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                u[j * n + i] = Math.sin(Math.PI * i * h) * Math.sin(Math.PI * j * h);
            }
        }
        double best = Double.MAX_VALUE;
        for (int rep = 0; rep < 3; rep++) {
            long t0 = System.nanoTime();
            for (int st = 0; st < steps; st++) {
                for (int k = 0; k < n * n; k++) {
                    vel[k] += 0.5 * dt * acc[k];
                    u[k] += dt * vel[k];
                }
                for (int j = 1; j < n - 1; j++) {
                    int row = j * n;
                    for (int i = 1; i < n - 1; i++) {
                        int k = row + i;
                        acc2[k] = c * (u[k - 1] + u[k + 1] + u[k - n] + u[k + n] - 4 * u[k]);
                    }
                }
                for (int k = 0; k < n * n; k++) {
                    vel[k] += 0.5 * dt * acc2[k];
                }
                double[] t = acc;
                acc = acc2;
                acc2 = t;
            }
            best = Math.min(best, (System.nanoTime() - t0) / (double) steps);
        }
        return best;
    }

    /** Forward-Euler 3D diffusion, 7-point stencil. */
    private static double baselineHeat3d(int nx, int ny, int nz, int steps) {
        double h = 1.0 / (nx - 1);
        double c = 0.1 * 1e-6 / (h * h);
        double[] u = new double[nx * ny * nz];
        double[] v = new double[nx * ny * nz];
        int sy = nx;
        int sz = nx * ny;
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    u[k * sz + j * sy + i] = Math.sin(Math.PI * i * h);
                }
            }
        }
        double best = Double.MAX_VALUE;
        for (int rep = 0; rep < 2; rep++) {
            long t0 = System.nanoTime();
            for (int st = 0; st < steps; st++) {
                for (int k = 1; k < nz - 1; k++) {
                    for (int j = 1; j < ny - 1; j++) {
                        int row = k * sz + j * sy;
                        for (int i = 1; i < nx - 1; i++) {
                            int q = row + i;
                            v[q] = u[q] + c * (u[q - 1] + u[q + 1] + u[q - sy] + u[q + sy] + u[q - sz] + u[q + sz] - 6 * u[q]);
                        }
                    }
                }
                double[] t = u;
                u = v;
                v = t;
            }
            best = Math.min(best, (System.nanoTime() - t0) / (double) steps);
        }
        return best;
    }

    // ------------------------------------------------------------ workloads
    private static PDEOptions opt(String m, int threads) {
        return new PDEOptions().integrator(m).threads(threads).stabilityPolicy(StabilityPolicy.OFF).blowupCheckSteps(0);
    }

    private static PDESolver heat1d(int n, String m, int th) {
        return PDE.problem("diff(u,t) = 0.1*diff(u,x,2)").domain("x", 0, 1, n)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").options(opt(m, th)).solver();
    }

    private static PDESolver heat2d(int n, String m, int th) {
        return PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2))").domain("x", 0, 1, n).domain("y", 0, 1, n)
                .initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0").options(opt(m, th)).solver();
    }

    private static PDESolver wave2d(int n, String m, int th) {
        return PDE.problem("diff(u,t,2) = diff(u,x,2) + diff(u,y,2)").domain("x", 0, 1, n).domain("y", 0, 1, n)
                .initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0").options(opt(m, th)).solver();
    }

    private static PDESolver diffusion3d(int nx, int ny, int nz, String m, int th) {
        return PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2) + diff(u,z,2))")
                .domain("x", 0, 1, nx).domain("y", 0, 1, ny).domain("z", 0, 1, nz)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").options(opt(m, th)).solver();
    }

    private static PDESolver reaction2d(int n, String m, int th) {
        return PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2)) + u*(1-u)*(u-0.3)").domain("x", 0, 1, n).domain("y", 0, 1, n)
                .initial("u", "0.5+0.4*sin(pi*x)*sin(pi*y)").neumann("u", "0").options(opt(m, th)).solver();
    }

    private static PDESolver grayScott(int n, String m, int th) {
        return PDE.problem("diff(u,t) = 2e-5*(diff(u,x,2) + diff(u,y,2)) - u*v^2 + 0.04*(1-u)",
                "diff(v,t) = 1e-5*(diff(v,x,2) + diff(v,y,2)) + u*v^2 - 0.099*v")
                .domain("x", 0, 2.5, n).domain("y", 0, 2.5, n)
                .initial("u", "1 - 0.5*exp(-100*((x-1.25)^2+(y-1.25)^2))").initial("v", "0.25*exp(-100*((x-1.25)^2+(y-1.25)^2))")
                .periodic("x").periodic("y").options(opt(m, th)).solver();
    }

    public static void main(String[] args) {
        boolean full = args.length > 0 && args[0].equals("full");
        double secs = full ? 1.5 : 0.6;
        System.out.println("PDE engine benchmark harness  (" + (full ? "full" : "quick") + ")");
        System.out.println("JVM " + System.getProperty("java.version") + ", cores=" + Runtime.getRuntime().availableProcessors()
                + ", maxHeap=" + (Runtime.getRuntime().maxMemory() >> 20) + " MB");

        header("1D heat equation, forward Euler (constant-coefficient linear: fused kernel)");
        for (int n : new int[]{128, 4096, 1 << 20}) {
            PDESolver s = heat1d(n, "euler", 1);
            row("heat1d euler n=" + n, s, measure(s, 1e-12, secs, 200), Double.NaN);
            s.close();
        }

        header("2D heat equation, forward Euler vs hand-written Java");
        for (int n : new int[]{128, 512, 1024}) {
            PDESolver s = heat2d(n, "euler", 1);
            Stat st = measure(s, 1e-12, secs, 100);
            row("heat2d euler " + n + "^2", s, st, baselineHeat2d(n, Math.max(5, (int) (5e7 / ((double) n * n)))));
            s.close();
        }
        for (int n : new int[]{128, 512}) {
            PDESolver s = heat2d(n, "rk4", 1);
            row("heat2d rk4 " + n + "^2", s, measure(s, 1e-12, secs, 50), Double.NaN);
            s.close();
        }

        header("2D wave equation, velocity-Verlet vs hand-written Java");
        for (int n : new int[]{128, 512, 1024}) {
            PDESolver s = wave2d(n, "verlet", 1);
            Stat st = measure(s, 1e-6, secs, 100);
            row("wave2d verlet " + n + "^2", s, st, baselineWave2d(n, Math.max(5, (int) (5e7 / ((double) n * n)))));
            s.close();
        }

        header("3D diffusion, forward Euler vs hand-written Java");
        {
            PDESolver s = diffusion3d(64, 64, 64, "euler", 1);
            row("diffusion3d euler 64^3", s, measure(s, 1e-12, secs, 20), baselineHeat3d(64, 64, 64, 40));
            s.close();
            s = diffusion3d(128, 128, 128, "euler", 1);
            row("diffusion3d euler 128^3", s, measure(s, 1e-12, secs, 5), baselineHeat3d(128, 128, 128, 10));
            s.close();
            if (full) {
                s = diffusion3d(256, 256, 128, "euler", 1);
                row("diffusion3d euler 256x256x128", s, measure(s, 1e-12, secs, 3), baselineHeat3d(256, 256, 128, 4));
                s.close();
            }
        }

        header("Non-linear / coupled (general compiled-expression kernel)");
        for (int n : new int[]{128, 512}) {
            PDESolver s = reaction2d(n, "rk4", 1);
            row("reaction-diffusion rk4 " + n + "^2", s, measure(s, 1e-12, secs, 30), Double.NaN);
            s.close();
        }
        for (int n : new int[]{128, 512}) {
            PDESolver s = grayScott(n, "rk4", 1);
            row("gray-scott (2 PDEs) rk4 " + n + "^2", s, measure(s, 1e-12, secs, 30), Double.NaN);
            s.close();
        }

        header("Implicit (Newton-Krylov, exact AD Jacobian)");
        {
            PDESolver s = heat2d(129, "crank_nicolson", 1);
            row("heat2d crank-nicolson 129^2", s, measure(s, 1e-4, secs, 20), Double.NaN);
            System.out.println("    diagnostics: " + s.diagnostics());
            s.close();
            s = reaction2d(129, "crank_nicolson", 1);
            row("reaction2d crank-nicolson 129^2", s, measure(s, 1e-4, secs, 20), Double.NaN);
            s.close();
        }

        header("Steady state (elliptic): iterations per second and time to converge");
        for (int n : new int[]{129, 257}) {
            PDEProblem p = PDE.problem("diff(u,x,2) + diff(u,y,2) = -2*pi^2*sin(pi*x)*sin(pi*y)")
                    .domain("x", 0, 1, n).domain("y", 0, 1, n).dirichlet("u", "0").build();
            for (String meth : new String[]{"red_black_sor", "sor", "newton_krylov"}) {
                PDEOptions o = new PDEOptions().ellipticMethod(meth).threads(1);
                long t0 = System.nanoTime();
                PDEResult r = PDE.solveSteady(p, o);
                double ms = (System.nanoTime() - t0) / 1e6;
                System.out.println(String.format(Locale.ROOT, "poisson %3d^2 %-14s %8.1f ms  iterations=%-6d krylov=%-6d residual=%.2e converged=%s",
                        n, meth, ms, r.getDiagnostics().getIterations(), r.getDiagnostics().getKrylovIterations(),
                        r.getDiagnostics().getResidualNorm(), r.isConverged()));
            }
        }

        header("Threads (this machine has " + Runtime.getRuntime().availableProcessors() + " core(s); speedups need real cores)");
        for (int th : new int[]{1, 2, 4}) {
            PDESolver s = heat2d(1024, "euler", th);
            row("heat2d euler 1024^2 threads=" + th, s, measure(s, 1e-12, secs, 20), Double.NaN);
            s.close();
        }
        for (int th : new int[]{1, 2, 4}) {
            PDESolver s = reaction2d(512, "rk4", th);
            row("reaction2d rk4 512^2 threads=" + th, s, measure(s, 1e-12, secs, 20), Double.NaN);
            s.close();
        }
    }
}
