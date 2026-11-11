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
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.github.gbenroscience.math.differentialcalculus.equations.turbo.DifferentialEquations;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadInfo;
import java.lang.management.ThreadMXBean;
import java.lang.reflect.Method;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Verifies the performance contract of the engine: steady-state stepping
 * allocates nothing (measured with the JVM's per-thread allocation counter on
 * the stepping thread AND on every engine worker thread), parallel execution
 * does not change results, and the solver can be advanced repeatedly without
 * re-parsing. Timings are NOT asserted (they are machine dependent); the
 * benchmark harness prints them.
 */
public class PDEPerformanceContractTest {

    /** @return bytes allocated so far by the given thread, or -1 if the JVM cannot tell. */
    private static long allocated(long threadId) {
        return AllocationMeter.allocatedBytes(threadId);
    }

    private static long[] engineThreadIds() {
        ThreadMXBean bean = ManagementFactory.getThreadMXBean();
        java.util.List<Long> ids = new java.util.ArrayList<Long>();
        ids.add(Thread.currentThread().getId());
        for (ThreadInfo ti : bean.getThreadInfo(bean.getAllThreadIds())) {
            if (ti != null && ti.getThreadName().startsWith("pde-worker-")) {
                ids.add(ti.getThreadId());
            }
        }
        long[] out = new long[ids.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ids.get(i);
        }
        return out;
    }

    private static long allocatedBy(long[] ids) {
        long sum = 0;
        for (long id : ids) {
            sum += allocated(id);
        }
        return sum;
    }

    /**
     * Bytes allocated per step, measured as the SLOPE between a short and a long
     * run, so that the (constant) cost of the measurement itself cancels.
     */
    private static void assertAllocationFree(PDESolver s, double dt, int warm, int measured) {
        assumeTrue(allocated(Thread.currentThread().getId()) >= 0, "JVM does not expose per-thread allocation counters");
        s.initialize();
        for (int i = 0; i < warm; i++) {
            s.step(dt);
        }
        long[] ids = engineThreadIds();
        // The slope of two independent runs; a genuine per-step allocation shows up in both, a one-off event that
        // happens to fall inside one window (a GC-triggered TLAB retirement, JIT deoptimisation...) in only one.
        double perStep = Double.POSITIVE_INFINITY;
        for (int attempt = 0; attempt < 2; attempt++) {
            long a0 = allocatedBy(ids);
            for (int i = 0; i < measured / 4; i++) {
                s.step(dt);
            }
            long a1 = allocatedBy(ids);
            for (int i = 0; i < measured; i++) {
                s.step(dt);
            }
            long a2 = allocatedBy(ids);
            perStep = Math.min(perStep, ((a2 - a1) - (a1 - a0)) / (double) (measured - measured / 4));
        }
        assertTrue(perStep < 4.0, s.integrator() + ": " + perStep + " bytes allocated per step");
    }

    @ParameterizedTest
    @ValueSource(strings = {"euler", "heun", "rk4", "rk45", "crank_nicolson", "backward_euler"})
    void parabolicStepping_isAllocationFree(String method) {
        try (PDESolver s = PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2)) + u*(1-u)")
                .domain("x", 0, 1, 49).domain("y", 0, 1, 49).initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0")
                .options(new PDEOptions().integrator(method).threads(1)).solver()) {
            assertAllocationFree(s, 1e-4, 1500, 400);
        }
    }

    @Test
    void waveStepping_isAllocationFree() {
        for (String method : new String[]{"verlet", "rk4"}) {
            try (PDESolver s = PDE.problem("diff(u,t,2) = diff(u,x,2) + diff(u,y,2)")
                    .domain("x", 0, 1, 49).domain("y", 0, 1, 49).initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0")
                    .options(new PDEOptions().integrator(method).threads(1)).solver()) {
                assertAllocationFree(s, 1e-3, 1500, 400);
            }
        }
    }

    @Test
    void neumannPeriodicAndTimeDependentBoundaries_areAllocationFree() {
        try (PDESolver s = PDE.problem("diff(u,t) = diff(u,x,2) + diff(u,y,2)")
                .domain("x", 0, 1, 41).domain("y", 0, 1, 41).initial("u", "cos(pi*x)")
                .periodic("x").dirichlet(Face.Y_MIN, "u", "sin(t)").neumann(Face.Y_MAX, "u", "cos(t)*x")
                .options(new PDEOptions().integrator("rk4").threads(1).dt(1e-4)).solver()) {
            assertAllocationFree(s, 1e-4, 1500, 400);
        }
    }

    @Test
    void threeDimensionalStepping_isAllocationFree() {
        try (PDESolver s = PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2) + diff(u,z,2))")
                .domain("x", 0, 1, 17).domain("y", 0, 1, 17).domain("z", 0, 1, 17)
                .initial("u", "sin(pi*x)*sin(pi*y)*sin(pi*z)").dirichlet("u", "0")
                .options(new PDEOptions().integrator("rk4").threads(1)).solver()) {
            assertAllocationFree(s, 1e-3, 2000, 400);
        }
    }

    @Test
    void multiThreadedStepping_isAllocationFreeOnEveryThread() {
        try (PDESolver s = PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2)) + u*(1-u)")
                .domain("x", 0, 1, 65).domain("y", 0, 1, 65).initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0")
                .options(new PDEOptions().integrator("rk4").threads(3)).solver()) {
            assertEquals(3, s.threads());
            assertAllocationFree(s, 1e-4, 1500, 400);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"euler", "rk4", "rk45", "crank_nicolson"})
    void parallelExecutionIsBitIdenticalToSerial(String method) {
        double[] ref = null;
        for (int threads : new int[]{1, 2, 3, 4}) {
            PDEResult r = PDE.problem("diff(u,t) = 0.1*(diff(u,x,2) + diff(u,y,2)) + sin(u)")
                    .domain("x", 0, 1, 33).domain("y", 0, 1, 29).initial("u", "sin(pi*x)*sin(pi*y)").dirichlet("u", "0")
                    .options(new PDEOptions().integrator(method).threads(threads).dt(5e-5)).solve(0.01);
            if (ref == null) {
                ref = r.values("u");
            } else {
                assertArrayEquals(ref, r.values("u"), 0.0, method + " with " + threads + " threads");
            }
        }
    }

    @Test
    void automaticThreadingStaysSerialForSmallProblems() {
        try (PDESolver s = PDETestSupport.heat1d(0.1, 101).solver()) {
            assertEquals(1, s.threads());
        }
    }

    @Test
    void realTimeLoop_advancesRepeatedlyAndExposesLiveStorage() {
        try (PDESolver s = PDETestSupport.heat1d(0.1, 101).options(new PDEOptions().integrator("rk4")).solver()) {
            s.initialize();
            double[] live = s.state("u");
            int mid = s.grid().index(50, 0, 0);
            double previous = live[mid];
            for (int i = 0; i < 200; i++) {
                s.step(2e-4);
                assertSame(live, s.state("u"), "storage must be reused, not reallocated");
                assertTrue(live[mid] < previous, "diffusion decays the mode monotonically");
                previous = live[mid];
            }
            assertEquals(0.04, s.time(), 1e-12);
            double exact = Math.exp(-Math.PI * Math.PI * 0.1 * 0.04);
            assertEquals(exact, live[mid], 1e-4);
            // restart
            s.initialize();
            assertEquals(1.0, live[mid], 1e-12);
            assertEquals(0.0, s.time(), 0.0);
        }
    }

    @Test
    void copyStateAndInterpolationAreConsistent() {
        PDEResult r = PDETestSupport.heat1d(0.1, 101).solve(0.1, 1e-3, "crank_nicolson");
        try (PDESolver s = PDETestSupport.heat1d(0.1, 101).options(new PDEOptions().integrator("crank_nicolson").dt(1e-3)).solver()) {
            s.initialize();
            s.advanceTo(0.1);
            double[] out = new double[101];
            s.copyState("u", out);
            assertArrayEquals(r.values("u"), out, 1e-12);
        }
        assertEquals(r.value("u", 25), r.valueAt("u", 0.25, 0, 0), 1e-12);
        double mid = 0.5 * (r.value("u", 25) + r.value("u", 26));
        assertEquals(mid, r.valueAt("u", 0.255, 0, 0), 1e-12);
    }

    @Test
    void existingOdeIntegratorsCanDriveTheSemiDiscreteSystem() throws Throwable {
        try (PDESolver s = PDE.problem("diff(u,t) = 0.1*diff(u,x,2)").domain("x", 0, 1, 51)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").options(new PDEOptions().integrator("rk4")).solver()) {
            s.initialize();
            MethodOfLinesSystem ode = s.toOdeSystem();
            assertEquals(49, ode.size());
            double[] y0 = ode.initialState();
            double[] y1 = DifferentialEquations.stepRK4(ode.handle(), ode.tSlot(), ode.ySlotStart(), ode.size(),
                    ode.frameSize(), 0.0, y0, 0.2, 400);
            ode.load(0.2, y1);
            double[] u = s.state("u");
            double worst = 0;
            for (int i = 0; i <= 50; i++) {
                double x = i / 50.0;
                worst = Math.max(worst, Math.abs(u[s.grid().index(i, 0, 0)] - Math.exp(-Math.PI * Math.PI * 0.1 * 0.2) * Math.sin(Math.PI * x)));
            }
            assertTrue(worst < 3e-4, "RK4 from the existing ODE engine, err " + worst);
            // adaptive RK45 from the existing engine
            double[] y2 = DifferentialEquations.stepRK45Adaptive(ode.handle(), ode.tSlot(), ode.ySlotStart(), ode.size(),
                    ode.frameSize(), 0.0, y0, 0.2, 1e-4);
            for (int i = 0; i < y1.length; i++) {
                assertEquals(y1[i], y2[i], 1e-5);
            }
        }
    }

    @Test
    void odeBridgeHandlesBoundaryConstraintsAndSecondOrderSystems() throws Throwable {
        // wave with Neumann ends: boundary nodes are algebraic and recomputed inside every evaluation
        try (PDESolver s = PDE.problem("diff(u,t,2) = diff(u,x,2)").domain("x", 0, 1, 61)
                .initial("u", "cos(pi*x)").neumann("u", "0").options(new PDEOptions().integrator("rk4")).solver()) {
            s.initialize();
            MethodOfLinesSystem ode = s.toOdeSystem();
            assertEquals(2 * 59, ode.size(), "59 free nodes x (u, u_t)");
            double[] y = DifferentialEquations.stepRK4(ode.handle(), ode.tSlot(), ode.ySlotStart(), ode.size(),
                    ode.frameSize(), 0.0, ode.initialState(), 0.5, 1000);
            ode.load(0.5, y);
            double u0 = s.state("u")[s.grid().index(0, 0, 0)];
            assertEquals(Math.cos(Math.PI * 0.5) * 1.0, u0, 2e-3, "boundary node follows from the Neumann relation");
        }
    }
}
