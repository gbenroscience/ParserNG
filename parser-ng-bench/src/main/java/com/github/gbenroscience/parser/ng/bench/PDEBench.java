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
package com.github.gbenroscience.parser.ng.bench;
  

import com.github.gbenroscience.math.differentialcalculus.pde.PDE;
import com.github.gbenroscience.math.differentialcalculus.pde.PDEOptions;
import com.github.gbenroscience.math.differentialcalculus.pde.PDEProblem;
import com.github.gbenroscience.math.differentialcalculus.pde.PDESolver;
import com.github.gbenroscience.math.differentialcalculus.pde.StabilityPolicy;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * JMH benchmarks of the PDE engine: time per time step and, with
 * {@code -prof gc}, the allocation rate (the engine's contract is
 * {@code gc.alloc.rate.norm ~ 0 B/op} for steady-state stepping).
 * <p>
 * Run from the shaded jar of this module, for example
 * <pre>
 * java -jar target/benchmarks.jar PDEBench -prof gc
 * java -jar target/benchmarks.jar "PDEBench.heat2d.*" -p n=512,1024 -p threads=1,4 -prof gc
 * </pre>
 * Every benchmark advances ONE time step per invocation, so the reported
 * time is "per step" and grid points per second is {@code n*n / time}.
 * The hand-written {@code hand*} benchmarks run the same algorithm on flat
 * arrays and give the cost floor against which the engine's overhead is read.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 10, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 2, jvmArgs = {"-Xms3g", "-Xmx3g", "-XX:+UseG1GC"})
public class PDEBench {

    private static PDEOptions opts(String integrator, int threads) {
        return new PDEOptions().integrator(integrator).threads(threads)
                .stabilityPolicy(StabilityPolicy.OFF).blowupCheckSteps(0);
    }

    // ---------------------------------------------------------------- engine
    @State(Scope.Thread)
    public static class Grid2d {

        @Param({"128", "512", "1024"})
        public int n;
        @Param({"1", "4"})
        public int threads;
        PDESolver heat;
        PDESolver wave;
        PDESolver reaction;
        PDESolver grayScott;
        double dtHeat;
        double dtWave;

        @Setup(Level.Trial)
        public void setup() {
            double h = 1.0 / (n - 1);
            dtHeat = 0.5 * h * h;     // stable for alpha = 0.1 (limit h^2 / (4 alpha))
            dtWave = 0.3 * h;         // c dt / h = 0.3 (Verlet limit 1/sqrt(2))
            heat = PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2))").domain("x", 0, 1, n).domain("y", 0, 1, n)
                    .initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0").options(opts("euler", threads)).solver();
            wave = PDE.problem("diff(u,t,2) = diff(u,x,2) + diff(u,y,2)").domain("x", 0, 1, n).domain("y", 0, 1, n)
                    .initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0").options(opts("verlet", threads)).solver();
            reaction = PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2)) + u*(1-u)*(u-0.3)").domain("x", 0, 1, n).domain("y", 0, 1, n)
                    .initial("u", "0.5+0.4*sin(pi*x)*sin(pi*y)").neumann("u", "0").options(opts("rk4", threads)).solver();
            grayScott = PDE.problem("diff(u,t) = 2e-5*(diff(u,x,2) + diff(u,y,2)) - u*v^2 + 0.04*(1-u)",
                    "diff(v,t) = 1e-5*(diff(v,x,2) + diff(v,y,2)) + u*v^2 - 0.099*v")
                    .domain("x", 0, 2.5, n).domain("y", 0, 2.5, n)
                    .initial("u", "1 - 0.5*exp(-100*((x-1.25)^2+(y-1.25)^2))").initial("v", "0.25*exp(-100*((x-1.25)^2+(y-1.25)^2))")
                    .periodic("x").periodic("y").options(opts("rk4", threads)).solver();
            heat.initialize();
            wave.initialize();
            reaction.initialize();
            grayScott.initialize();
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            heat.close();
            wave.close();
            reaction.close();
            grayScott.close();
        }
    }

    @Benchmark
    public void heat2dEuler(Grid2d s) {
        s.heat.step(s.dtHeat);
    }

    @Benchmark
    public void wave2dVerlet(Grid2d s) {
        s.wave.step(s.dtWave);
    }

    @Benchmark
    public void reactionDiffusion2dRk4(Grid2d s) {
        s.reaction.step(1e-5);
    }

    @Benchmark
    public void grayScott2dRk4(Grid2d s) {
        s.grayScott.step(1e-3);
    }

    @State(Scope.Thread)
    public static class Heat1d {

        @Param({"128", "65536", "1048576"})
        public int n;
        PDESolver solver;
        double dt;

        @Setup(Level.Trial)
        public void setup() {
            double h = 1.0 / (n - 1);
            dt = 0.5 * h * h / 0.4;
            solver = PDE.problem("diff(u,t) = 0.1*diff(u,x,2)").domain("x", 0, 1, n).initial("u", "sin(pi*x)")
                    .dirichlet("u", "0").options(opts("euler", 1)).solver();
            solver.initialize();
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            solver.close();
        }
    }

    @Benchmark
    public void heat1dEuler(Heat1d s) {
        s.solver.step(s.dt);
    }

    @State(Scope.Thread)
    public static class Diffusion3d {

        @Param({"64", "128"})
        public int n;
        @Param({"1", "4"})
        public int threads;
        PDESolver solver;
        double dt;

        @Setup(Level.Trial)
        public void setup() {
            double h = 1.0 / (n - 1);
            dt = 0.5 * h * h / 0.6;
            solver = PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2) + diff(u,z,2))")
                    .domain("x", 0, 1, n).domain("y", 0, 1, n).domain("z", 0, 1, n)
                    .initial("u", "sin(pi*x)*sin(pi*y)*sin(pi*z)").dirichlet("u", "0").options(opts("euler", threads)).solver();
            solver.initialize();
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            solver.close();
        }
    }

    @Benchmark
    public void diffusion3dEuler(Diffusion3d s) {
        s.solver.step(s.dt);
    }

    /** Implicit step: Newton-Krylov with the exact (automatic-differentiation) Jacobian. */
    @State(Scope.Thread)
    public static class Implicit2d {

        @Param({"65", "129"})
        public int n;
        PDESolver solver;

        @Setup(Level.Trial)
        public void setup() {
            solver = PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2)) + u*(1-u)").domain("x", 0, 1, n).domain("y", 0, 1, n)
                    .initial("u", "0.5+0.4*sin(pi*x)*sin(pi*y)").dirichlet("u", "0.5").options(opts("crank_nicolson", 1)).solver();
            solver.initialize();
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            solver.close();
        }
    }

    @Benchmark
    public void crankNicolson2d(Implicit2d s) {
        s.solver.step(1e-3);
    }

    /** Whole elliptic solves (time per converged solve). */
    @State(Scope.Thread)
    public static class Poisson {

        @Param({"65", "129"})
        public int n;
        @Param({"red_black_sor", "sor", "newton_krylov"})
        public String method;
        PDEProblem problem;

        @Setup(Level.Trial)
        public void setup() {
            problem = PDE.problem("diff(u,x,2) + diff(u,y,2) = -2*pi^2*sin(pi*x)*sin(pi*y)")
                    .domain("x", 0, 1, n).domain("y", 0, 1, n).dirichlet("u", "0").build();
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.SingleShotTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public Object poissonSolve(Poisson s) {
        return PDE.solveSteady(s.problem, new PDEOptions().ellipticMethod(s.method).threads(1));
    }

    // ---------------------------------------------------------------- hand-written baselines
    @State(Scope.Thread)
    public static class Hand2d {

        @Param({"128", "512", "1024"})
        public int n;
        double[] u;
        double[] v;
        double c;
        double[] vel;
        double[] acc;
        double[] acc2;
        double dtw;
        double cw;

        @Setup(Level.Trial)
        public void setup() {
            double h = 1.0 / (n - 1);
            c = 0.1 * (0.5 * h * h) / (h * h);
            u = new double[n * n];
            v = new double[n * n];
            vel = new double[n * n];
            acc = new double[n * n];
            acc2 = new double[n * n];
            dtw = 0.3 * h;
            cw = 1.0 / (h * h);
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    u[j * n + i] = Math.sin(Math.PI * i * h) * Math.sin(Math.PI * j * h);
                }
            }
        }
    }

    @Benchmark
    public double[] handHeat2dEuler(Hand2d s) {
        final int n = s.n;
        final double[] u = s.u;
        final double[] v = s.v;
        final double c = s.c;
        for (int j = 1; j < n - 1; j++) {
            int row = j * n;
            for (int i = 1; i < n - 1; i++) {
                int k = row + i;
                v[k] = u[k] + c * (u[k - 1] + u[k + 1] + u[k - n] + u[k + n] - 4 * u[k]);
            }
        }
        double[] t = s.u;
        s.u = s.v;
        s.v = t;
        return s.u;
    }

    @Benchmark
    public double[] handWave2dVerlet(Hand2d s) {
        final int n = s.n;
        final double dt = s.dtw;
        final double[] u = s.u;
        final double[] vel = s.vel;
        final double[] acc = s.acc;
        final double[] acc2 = s.acc2;
        for (int k = 0; k < n * n; k++) {
            vel[k] += 0.5 * dt * acc[k];
            u[k] += dt * vel[k];
        }
        for (int j = 1; j < n - 1; j++) {
            int row = j * n;
            for (int i = 1; i < n - 1; i++) {
                int k = row + i;
                acc2[k] = s.cw * (u[k - 1] + u[k + 1] + u[k - n] + u[k + n] - 4 * u[k]);
            }
        }
        for (int k = 0; k < n * n; k++) {
            vel[k] += 0.5 * dt * acc2[k];
        }
        s.acc = acc2;
        s.acc2 = acc;
        return u;
    }
}
