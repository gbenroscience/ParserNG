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
 * Benchmark harness for the finite-volume solver: time per step, cell updates
 * per second (one update = one SSP-RK3 step of a cell, i.e. three flux
 * evaluations per face), allocation and GC. Single-threaded; machine dependent.
 */
public final class FVBenchmarkHarness {

    private FVBenchmarkHarness() {
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

    private static void run(String name, FiniteVolumeSolver s, int warm, int steps) {
        double dt = 0.5 * s.suggestTimeStep();
        for (int i = 0; i < warm; i++) {
            s.step(dt);
        }
        long g0 = gcCount();
        long a0 = allocated();
        for (int i = 0; i < steps / 5; i++) {
            s.step(dt);
        }
        long a1 = allocated();
        long t0 = System.nanoTime();
        for (int i = 0; i < steps; i++) {
            s.step(dt);
        }
        long t1 = System.nanoTime();
        long a2 = allocated();
        long g1 = gcCount();
        double cells = 1;
        for (int a = 0; a < s.law().dimension(); a++) {
            cells *= s.cells(a);
        }
        double nsStep = (t1 - t0) / (double) steps;
        double alloc = ((a2 - a1) - (a1 - a0)) / (steps - steps / 5.0);
        System.out.println(String.format(Locale.ROOT, "%-40s %10.0f cells %9.3f ms/step %8.2f Mcell-updates/s %7.1f ns/cell/step  alloc %5.1f B/step  GC %d",
                name, cells, nsStep / 1e6, cells / nsStep * 1e3, nsStep / cells, Math.max(0, alloc), g1 - g0));
    }

    public static void main(String[] args) {
        System.out.println("Finite-volume benchmark  JVM " + System.getProperty("java.version") + ", cores=" + Runtime.getRuntime().availableProcessors());
        System.out.println("\n== 1-D Euler (Sod), SSP-RK3, MUSCL");
        for (NumericalFlux f : new NumericalFlux[]{NumericalFlux.LLF, NumericalFlux.HLL, NumericalFlux.HLLC}) {
            for (int n : new int[]{1000, 100000, 1000000}) {
                FiniteVolumeSolver s = FiniteVolumeSolver.builder().law(new EulerLaw(1, 1.4)).cells(n).limiter(Limiter.MC).numericalFlux(f)
                        .initialPrimitive("if(x<0.5,1,0.125)", "0", "if(x<0.5,1,0.1)").build().initialize();
                run("euler1d " + f + " n=" + n, s, 20, n >= 1000000 ? 20 : 100);
            }
        }
        System.out.println("\n== 2-D Euler (isentropic-vortex-like smooth flow), HLLC, MC");
        for (int n : new int[]{128, 256, 512}) {
            FiniteVolumeSolver s = FiniteVolumeSolver.builder().law(new EulerLaw(2, 1.4)).cells(n, n).periodic("x").periodic("y").limiter(Limiter.MC)
                    .numericalFlux(NumericalFlux.HLLC).initialPrimitive("1+0.2*sin(2*pi*x)", "0.5", "0.3", "1+0.1*cos(2*pi*y)").build().initialize();
            run("euler2d HLLC " + n + "^2", s, 10, 30);
        }
        System.out.println("\n== 3-D Euler, HLLC, MC");
        for (int n : new int[]{32, 64}) {
            FiniteVolumeSolver s = FiniteVolumeSolver.builder().law(new EulerLaw(3, 1.4)).cells(n, n, n).periodic("x").periodic("y").periodic("z").limiter(Limiter.MC)
                    .numericalFlux(NumericalFlux.HLLC).initialPrimitive("1+0.2*sin(2*pi*x)", "0.5", "0.3", "0.1", "1+0.1*cos(2*pi*y)").build().initialize();
            run("euler3d HLLC " + n + "^3", s, 5, 10);
        }
        System.out.println("\n== expression law (Burgers from diff(u,t)+diff(u^2/2,x)=0, exact AD wave speeds)");
        for (int n : new int[]{1000, 100000}) {
            FiniteVolumeSolver s = new com.github.gbenroscience.parser.MathExpression("diff(u,t) + diff(u^2/2,x) = 0").asFiniteVolume()
                    .cells(n).periodic("x").limiter(Limiter.MC).numericalFlux(NumericalFlux.HLL).initial("1.5+sin(2*pi*x)").build().initialize();
            run("burgers expression HLL n=" + n, s, 50, 100);
        }
        System.out.println("\n== first-order (no reconstruction) vs MUSCL limiters, euler2d 256^2 HLLC");
        for (Limiter l : new Limiter[]{Limiter.NONE, Limiter.MINMOD, Limiter.MC, Limiter.VAN_LEER}) {
            FiniteVolumeSolver s = FiniteVolumeSolver.builder().law(new EulerLaw(2, 1.4)).cells(256, 256).periodic("x").periodic("y").limiter(l)
                    .numericalFlux(NumericalFlux.HLLC).initialPrimitive("1+0.2*sin(2*pi*x)", "0.5", "0.3", "1+0.1*cos(2*pi*y)").build().initialize();
            run("euler2d 256^2 " + l, s, 10, 30);
        }
    }
}
