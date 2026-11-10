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

/**
 * Jacobi preconditioning of the matrix-free BiCGStab solves: it must never change the answer, must help where the
 * Jacobian diagonal varies, and must be harmless where it cannot help.
 */
public class PDEPreconditionerTest {

    private static final String RHS = "-(sin(pi*x)*sin(pi*y)+0.5*sin(3*pi*x)*sin(2*pi*y)+exp(-50*((x-0.3)^2+(y-0.6)^2)))";

    private static PDEResult solve(String lhs, boolean pc, int n) {
        PDEOptions o = new PDEOptions().ellipticMethod("newton_krylov").jacobiPreconditioner(pc).krylovMaxIterations(20000)
                .krylovTolerance(1e-11).newtonTolerance(1e-12).threads(1);
        return PDE.problem(lhs + " = " + RHS).domain("x", 0, 1, n).domain("y", 0, 1, n).dirichlet("u", "0").options(o).solveSteady();
    }

    private static double maxDiff(PDEResult a, PDEResult b) {
        double m = 0;
        for (int i = 0; i < a.values("u").length; i++) {
            m = Math.max(m, Math.abs(a.values("u")[i] - b.values("u")[i]));
        }
        return m;
    }

    @Test
    void variableCoefficientProblemsNeedMarkedlyFewerKrylovIterations() {
        String op = "(1+20*x*y)*diff(u,x,2)+(1+20*x*y)*diff(u,y,2)";
        PDEResult off = solve(op, false, 49);
        PDEResult on = solve(op, true, 49);
        assertTrue(off.isConverged() && on.isConverged());
        assertTrue(on.getDiagnostics().getKrylovIterations() * 1.8 < off.getDiagnostics().getKrylovIterations(),
                "with " + on.getDiagnostics().getKrylovIterations() + " vs without " + off.getDiagnostics().getKrylovIterations());
        assertEquals(0.0, maxDiff(on, off), 1e-8, "the preconditioner must not change the solution");
    }

    @Test
    void constantCoefficientProblemsAreUnchanged() {
        PDEResult off = solve("diff(u,x,2)+diff(u,y,2)", false, 33);
        PDEResult on = solve("diff(u,x,2)+diff(u,y,2)", true, 33);
        assertEquals(0.0, maxDiff(on, off), 1e-8);
        assertTrue(on.getDiagnostics().getKrylovIterations() <= off.getDiagnostics().getKrylovIterations() + 2);
    }

    @Test
    void nonlinearProblemsStillConvergeToTheSameSolution() {
        PDEResult off = solve("diff(u,x,2)+diff(u,y,2)+u^2", false, 33);
        PDEResult on = solve("diff(u,x,2)+diff(u,y,2)+u^2", true, 33);
        assertTrue(on.isConverged() && off.isConverged());
        assertEquals(0.0, maxDiff(on, off), 1e-8);
    }

    @Test
    void implicitTimeStepsBenefitAndKeepTheSameSolution() {
        PDEResult[] r = new PDEResult[2];
        for (int k = 0; k < 2; k++) {
            PDEOptions o = new PDEOptions().integrator("crank_nicolson").dt(0.05).jacobiPreconditioner(k == 1)
                    .krylovTolerance(1e-11).newtonTolerance(1e-12).threads(1);
            r[k] = PDE.problem("diff(u,t) = (1+4*x)*(diff(u,x,2)+diff(u,y,2)) + u*(1-u)").domain("x", 0, 1, 33).domain("y", 0, 1, 33)
                    .initial("u", "0.5+0.4*sin(pi*x)*sin(pi*y)").dirichlet("u", "0.5").options(o).solve(0.5);
        }
        assertEquals(0.0, maxDiff(r[1], r[0]), 1e-7);
        assertTrue(r[1].getDiagnostics().getKrylovIterations() < r[0].getDiagnostics().getKrylovIterations());
    }
}
