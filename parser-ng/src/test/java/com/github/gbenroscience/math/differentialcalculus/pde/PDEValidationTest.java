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
 * Invalid configurations, unstable time steps and the stability policies.
 * The engine must refuse (with a message that says why) rather than return
 * plausible-looking nonsense.
 */
public class PDEValidationTest {

    private static PDEProblem.Builder base() {
        return PDE.problem("diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 21).initial("u", "sin(pi*x)").dirichlet("u", "0");
    }

    private static String message(Runnable r) {
        PDEException ex = assertThrows(PDEException.class, r::run);
        return ex.getMessage();
    }

    // ------------------------------------------------------------ configuration errors
    @Test
    void missingBoundaryConditionIsReported() {
        String m = message(() -> PDE.problem("diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 21).initial("u", "0")
                .dirichlet(Face.X_MIN, "u", "0").solver());
        assertTrue(m.contains("Missing boundary condition") && m.contains("X_MAX"), m);
    }

    @Test
    void missingInitialConditionIsReported() {
        String m = message(() -> PDE.problem("diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 21).dirichlet("u", "0").solver());
        assertTrue(m.contains("initial condition") && m.contains("'u'"), m);
    }

    @Test
    void unknownSymbolIsReportedWithAHint() {
        String m = message(() -> PDE.problem("diff(u,t) = kappa*diff(u,x,2)").domain("x", 0, 1, 21).initial("u", "0")
                .dirichlet("u", "0").solver());
        assertTrue(m.contains("kappa") && m.contains("param"), m);
        String m2 = message(() -> base().initial("u", "sin(q*x)").solver());
        assertTrue(m2.contains("q"), m2);
        String m3 = message(() -> PDE.problem("diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 21).initial("u", "0")
                .dirichlet("u", "bogus*t").solver());
        assertTrue(m3.contains("bogus"), m3);
    }

    @Test
    void malformedExpressionsAreRejected() {
        assertThrows(PDEException.class, () -> PDE.problem("diff(u,t) = diff(u,x,2) +* 3").domain("x", 0, 1, 21)
                .initial("u", "0").dirichlet("u", "0").solver());
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t) = diff(u,x,2").domain("x", 0, 1, 21)
                .initial("u", "0").dirichlet("u", "0").solver());
    }

    @Test
    void domainAndEquationStructureIsValidated() {
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t) = diff(u,x,2)").build(), "no domain");
        assertThrows(PDEConfigurationException.class, () -> PDE.problem().domain("x", 0, 1, 11).build(), "no equation");
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t)=0").domain("x", 1, 1, 11));
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t)=0").domain("x", 0, 1, 2));
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t)=0").domain("t", 0, 1, 11).build());
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t)=0").domain("x", 0, 1, 11).domain("x", 0, 1, 11).build());
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t)=0").domain("x", 0, 1, 11).domain("y", 0, 1, 11)
                .domain("z", 0, 1, 11).domain("w", 0, 1, 11));
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t)=0").domain("x", 0, 1, 11).periodic("q"));
    }

    @Test
    void sparseGridsAreRejectedForWideStencils() {
        String m = message(() -> base().domain("y", 0, 1, 4).options(new PDEOptions().spatialOrder(6)).solver());
        assertTrue(m.contains("points"), m);
    }

    @Test
    void unsupportedEquationFormsAreExplained() {
        // no time derivative in an equation of a time-dependent system
        assertTrue(message(() -> PDE.problem("diff(u,t) = diff(u,x,2)", "diff(v,x,2) = u").domain("x", 0, 1, 11)
                .initial("u", "0").dirichlet("*", "0").solver()).contains("algebraic"));
        // mixed space-time derivative
        assertTrue(message(() -> PDE.problem("diff(u,t) = diff(u,x,t)").domain("x", 0, 1, 11).initial("u", "0")
                .dirichlet("u", "0").solver()).contains("space-time"));
        // time derivative on both sides
        assertTrue(message(() -> PDE.problem("diff(u,t) = diff(u,t) + diff(u,x,2)").domain("x", 0, 1, 11).initial("u", "0")
                .dirichlet("u", "0").solver()).toLowerCase().contains("time derivative"));
        // nonlinear in the time derivative
        assertTrue(message(() -> PDE.problem("diff(u,t)^2 = diff(u,x,2)").domain("x", 0, 1, 11).initial("u", "1")
                .dirichlet("u", "0").solver()).contains("not linear"));
        // unknown / equation count mismatch
        assertTrue(message(() -> PDE.problem("diff(u,t) = diff(v,x,2)").domain("x", 0, 1, 11).initial("u", "0")
                .dirichlet("*", "0").solver()).contains("square"));
        // order above two
        assertTrue(message(() -> PDE.problem("diff(u,t,3) = diff(u,x,2)").domain("x", 0, 1, 11).initial("u", "0")
                .dirichlet("*", "0").solver()).contains("order 2"));
        // undeclared independent variable inside diff
        assertTrue(message(() -> PDE.problem("diff(u,t) = diff(u,q,2)").domain("x", 0, 1, 11).initial("u", "0")
                .dirichlet("u", "0").solver()).contains("independent variable"));
        // unknown named like a parameter
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t) = diff(u,x,2)").param("u", 1)
                .domain("x", 0, 1, 11).initial("u", "0").dirichlet("u", "0").solver());
    }

    @Test
    void periodicConditionsMustBePairedAndConsistent() {
        assertTrue(message(() -> base().boundary(Face.X_MIN, "u", BoundaryCondition.periodic()).solver()).contains("BOTH faces"));
    }

    @Test
    void methodSelectionIsValidated() {
        assertTrue(message(() -> PDE.problem("diff(u,t,2) = diff(u,x,2) - diff(u,t)").domain("x", 0, 1, 11).initial("u", "0")
                .dirichlet("u", "0").options(new PDEOptions().integrator("verlet")).solver()).contains("u_t"));
        assertTrue(message(() -> base().options(new PDEOptions().integrator("verlet")).solver()).contains("second order"));
        assertThrows(PDEConfigurationException.class, () -> TimeIntegrator.parse("bogus"));
        assertThrows(PDEConfigurationException.class, () -> EllipticMethod.parse("bogus"));
        assertThrows(PDEConfigurationException.class, () -> new PDEOptions().spatialOrder(9));
        assertTrue(message(() -> PDE.problem("diff(u,x,2) = 0", "diff(v,x,2) = 0").domain("x", 0, 1, 11).dirichlet("*", "0")
                .options(new PDEOptions().ellipticMethod("sor")).solver()).contains("NEWTON_KRYLOV"));
        assertTrue(message(() -> PDE.problem("diff(u,x,2) = 0").domain("x", 0, 1, 11).dirichlet("u", "0")
                .options(new PDEOptions().ellipticMethod("sor").omega(2.5)).solver()).contains("omega"));
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,x,2) = 0").domain("x", 0, 1, 11).dirichlet("u", "0")
                .options(new PDEOptions().spatialOrder(3)).solver());
    }

    @Test
    void steadyAndTransientEntryPointsAreNotInterchangeable() {
        assertThrows(PDEConfigurationException.class, () -> base().solveSteady());
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,x,2) = 0").domain("x", 0, 1, 11).dirichlet("u", "0").solve(1.0, 0.1, "rk4"));
        try (PDESolver s = base().solver()) {
            assertThrows(PDEException.class, () -> s.step(1e-4), "must initialize first");
            s.initialize();
            assertThrows(PDEConfigurationException.class, () -> s.step(-1.0));
            assertThrows(PDEConfigurationException.class, () -> s.step(Double.NaN));
        }
        PDESolver closed = base().solver();
        closed.close();
        assertThrows(PDEException.class, closed::initialize);
    }

    @Test
    void initialDataIncompatibleWithBoundariesIsFlagged() {
        PDEResult r = PDE.problem("diff(u,t) = diff(u,x,2)").domain("x", 0, 1, 21).initial("u", "1").dirichlet("u", "0")
                .solve(0.001, 1e-4, "crank_nicolson");
        assertFalse(r.getDiagnostics().getWarnings().isEmpty());
        assertTrue(r.getDiagnostics().getWarnings().get(0).contains("boundary"));
        assertEquals(0.0, r.value("u", 0), 0.0, "boundary condition takes precedence");
    }

    // ------------------------------------------------------------ stability
    @Test
    void explicitEulerBeyondTheDiffusionLimitIsRefusedWithAUsefulEstimate() {
        // limit is h^2/(2 alpha) = 1/(100^2 * 0.2) = 5e-4
        UnstableTimeStepException ex = assertThrows(UnstableTimeStepException.class,
                () -> PDETestSupport.heat1d(0.1, 101).solve(0.1, 1e-3, "euler"));
        assertEquals(5e-4, ex.getMaxStableDt(), 5e-5);
        assertEquals(1e-3, ex.getDt(), 0.0);
        assertTrue(ex.getMessage().contains("CRANK_NICOLSON"), ex.getMessage());
    }

    @Test
    void explicitRk4BeyondTheCflLimitIsRefusedOnWaves() {
        // rho = 2c/h = 200 -> dt_max ~ 2.78/200 = 0.0139
        assertThrows(UnstableTimeStepException.class, () -> PDE.problem("diff(u,t,2) = diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(1.0, 0.03, "rk4"));
        // inside the limit it runs
        PDEResult ok = PDE.problem("diff(u,t,2) = diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(1.0, 0.01, "rk4");
        assertTrue(ok.maxError("u", (x, y, z, t) -> Math.cos(Math.PI * t) * Math.sin(Math.PI * x)) < 1e-2);
    }

    @Test
    void forwardEulerIsRefusedOnOscillatoryProblemsAtAnyReasonableStep() {
        PDEOptions o = new PDEOptions().integrator("euler").dt(0.5 * 0.01);
        assertThrows(UnstableTimeStepException.class, () -> PDE.problem("diff(u,t) + diff(u,x) = 0").domain("x", 0, 1, 101)
                .initial("u", "sin(2*pi*x)").periodic("x").options(o).solve(1.0));
    }

    @Test
    void substepPolicyKeepsTheSolutionStableAndAccurate() {
        PDEOptions o = new PDEOptions().integrator("rk4").dt(2e-3).stabilityPolicy(StabilityPolicy.SUBSTEP);
        PDEResult r = PDETestSupport.heat1d(0.1, 101).options(o).solve(0.5);
        assertTrue(r.maxError("u", PDETestSupport.heat1dExact(0.1)) < 1e-4);
        assertTrue(r.getDiagnostics().getSteps() > 250, "step must have been split: " + r.getDiagnostics().getSteps());
    }

    @Test
    void warnPolicyRecordsTheViolationButRuns() {
        PDEOptions o = new PDEOptions().integrator("rk4").dt(1e-3).stabilityPolicy(StabilityPolicy.WARN).blowupFactor(1e30);
        PDEResult r = PDETestSupport.heat1d(0.1, 101).options(o).solve(0.002);
        assertFalse(r.getDiagnostics().getWarnings().isEmpty());
    }

    @Test
    void divergenceGuardCatchesInstabilityWhenTheEstimateIsSwitchedOff() {
        PDEOptions o = new PDEOptions().integrator("euler").dt(2e-3).stabilityPolicy(StabilityPolicy.OFF);
        UnstableTimeStepException ex = assertThrows(UnstableTimeStepException.class,
                () -> PDETestSupport.heat1d(0.1, 101).options(o).solve(1.0));
        assertTrue(ex.getMessage().contains("diverging") || ex.getMessage().contains("non-finite"), ex.getMessage());
    }

    @Test
    void implicitMethodsAreStableFarBeyondTheExplicitLimit() {
        PDEResult r = PDETestSupport.heat1d(0.1, 101).solve(0.5, 0.05, "backward_euler");
        assertTrue(r.isConverged());
        for (double v : r.values("u")) {
            assertTrue(Math.abs(v) <= 1.0);
        }
        assertTrue(r.maxError("u", PDETestSupport.heat1dExact(0.1)) < 0.05);
    }

    @Test
    void cflBasedDefaultTimeStepProducesAStableSolution() {
        try (PDESolver s = PDE.problem("diff(u,t) = 0.1*diff(u,x,2)").domain("x", 0, 1, 101).initial("u", "sin(pi*x)")
                .dirichlet("u", "0").options(new PDEOptions().integrator("rk4")).solver()) {
            s.initialize();
            assertEquals(4000, s.spectralRadius(), 100);
            assertTrue(s.suggestedTimeStep() < s.maxStableTimeStep());
            s.advanceTo(0.2);
            PDEResult r = s.result();
            assertTrue(r.maxError("u", PDETestSupport.heat1dExact(0.1)) < 1e-4);
        }
    }
}
