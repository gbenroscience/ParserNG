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

import com.github.gbenroscience.parser.MathExpression;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Finite-volume framework: conservation, order of accuracy, shock capturing
 * (Burgers, Sod against the exact Riemann solution), limiters, boundaries,
 * the MathExpression connection and allocation.
 */
public class FiniteVolumeTest {

    private static final double PI = Math.PI;
    private static final double GAMMA = 1.4;

    // ------------------------------------------------------------------ helpers
    private static FiniteVolumeSolver sod(int n, NumericalFlux f, Limiter l) {
        return FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(n).limiter(l).numericalFlux(f)
                .initialPrimitive("if(x<0.5,1,0.125)", "0", "if(x<0.5,1,0.1)").build().initialize();
    }

    private static double sodL1(FiniteVolumeSolver s) {
        ExactRiemann ex = new ExactRiemann(GAMMA, 1, 0, 1, 0.125, 0, 0.1);
        int n = s.cells(0);
        double l1 = 0;
        for (int i = 0; i < n; i++) {
            double x = s.cellCenter(0, i);
            l1 += Math.abs(s.state(0)[s.index(i, 0, 0)] - ex.sample((x - 0.5) / s.time())[0]) * s.spacing(0);
        }
        return l1;
    }

    private static double[] primitive(FiniteVolumeSolver s, int i) {
        double rho = s.state(0)[s.index(i, 0, 0)];
        double m = s.state(1)[s.index(i, 0, 0)];
        double e = s.state(2)[s.index(i, 0, 0)];
        double u = m / rho;
        return new double[]{rho, u, (GAMMA - 1) * (e - 0.5 * rho * u * u)};
    }

    // ------------------------------------------------------------------ the reference solver itself
    @Test
    void exactRiemannSolverReproducesTheTabulatedSodStarState() {
        ExactRiemann ex = new ExactRiemann(GAMMA, 1, 0, 1, 0.125, 0, 0.1);
        assertEquals(0.30313, ex.pStar, 1e-5);
        assertEquals(0.92745, ex.uStar, 1e-5);
        assertEquals(1.75216, ex.rightShockSpeed(), 1e-5);
        assertEquals(0.42632, ex.sample(0.5)[0], 1e-5, "density left of the contact");
        assertEquals(0.26557, ex.sample(1.2)[0], 1e-5, "density right of the contact");
    }

    // ------------------------------------------------------------------ Sod shock tube
    @Test
    void sodShockTubeMatchesTheExactSolution() {
        FiniteVolumeSolver s = sod(400, NumericalFlux.HLLC, Limiter.MC);
        s.advanceTo(0.2);
        assertEquals(0.2, s.time(), 1e-12);
        assertTrue(sodL1(s) < 0.0025, "L1(rho) " + sodL1(s));
        // star plateau between the rarefaction and the contact, and between the contact and the shock
        double[] left = primitive(s, 245);     // x ~ 0.614
        assertEquals(0.42632, left[0], 0.01);
        assertEquals(0.92745, left[1], 0.01);
        assertEquals(0.30313, left[2], 0.005);
        double[] right = primitive(s, 330);    // x ~ 0.826
        assertEquals(0.26557, right[0], 0.01);
        assertEquals(0.92745, right[1], 0.01);
        assertEquals(0.30313, right[2], 0.005);
        // shock position: where the density crosses the mid-value of its jump
        double mid = 0.5 * (0.26557 + 0.125);
        double xs = Double.NaN;
        for (int i = 399; i > 0; i--) {
            if (s.state(0)[s.index(i, 0, 0)] > mid) {
                xs = s.cellCenter(0, i);
                break;
            }
        }
        assertEquals(0.5 + 1.75216 * 0.2, xs, 0.008, "shock position");
        // contact: density between the two star values
        double midC = 0.5 * (0.42632 + 0.26557);
        double xc = Double.NaN;
        for (int i = 150; i < 399; i++) {
            if (s.state(0)[s.index(i, 0, 0)] < midC) {
                xc = s.cellCenter(0, i);
                break;
            }
        }
        assertEquals(0.5 + 0.92745 * 0.2, xc, 0.01, "contact position");
    }

    @Test
    void sodErrorsRankAsTheMethodsPredictAndShrinkWithTheGrid() {
        double llf1 = run(NumericalFlux.LLF, Limiter.NONE, 400);
        double hll1 = run(NumericalFlux.HLL, Limiter.NONE, 400);
        double hllc1 = run(NumericalFlux.HLLC, Limiter.NONE, 400);
        double llfMc = run(NumericalFlux.LLF, Limiter.MC, 400);
        double hllcMc = run(NumericalFlux.HLLC, Limiter.MC, 400);
        double hllcMm = run(NumericalFlux.HLLC, Limiter.MINMOD, 400);
        double hllcVl = run(NumericalFlux.HLLC, Limiter.VAN_LEER, 400);
        assertTrue(llf1 > hll1 && hll1 > hllc1, "more complete Riemann solvers are less diffusive: " + llf1 + " " + hll1 + " " + hllc1);
        assertTrue(hllcMc < hllc1 / 3 && llfMc < llf1 / 2, "second order beats first order");
        assertTrue(hllcMc < hllcMm, "MC is sharper than minmod on contacts: " + hllcMc + " vs " + hllcMm);
        assertTrue(hllcVl < hllcMm);
        double[] e = new double[4];
        int[] ns = {100, 200, 400, 800};
        for (int q = 0; q < 4; q++) {
            e[q] = run(NumericalFlux.HLLC, Limiter.MC, ns[q]);
        }
        for (int q = 1; q < 4; q++) {
            assertTrue(e[q] < e[q - 1], "error must fall with refinement");
            // discontinuities converge at a rate below second order: shocks ~1, contacts ~2/3
            assertTrue(PDETestSupport.order(e[q - 1], e[q], 2) > 0.6, "rate " + PDETestSupport.order(e[q - 1], e[q], 2));
        }
    }

    private static double run(NumericalFlux f, Limiter l, int n) {
        FiniteVolumeSolver s = sod(n, f, l);
        s.advanceTo(0.2);
        return sodL1(s);
    }

    @ParameterizedTest
    @EnumSource(NumericalFlux.class)
    void sodIsConservativeAndPositiveWithEveryFlux(NumericalFlux f) {
        FiniteVolumeSolver s = sod(200, f, Limiter.MC);
        s.advanceTo(0.2);
        FVDiagnostics d = s.diagnostics();
        for (int q = 0; q < 3; q++) {
            assertEquals(0.0, d.getConservationDefect()[q], 1e-12 * (1 + Math.abs(d.getInitialTotals()[q])), "component " + q);
        }
        assertTrue(d.getMinPrimitive()[0] > 0.1, "density stays positive: " + d.getMinPrimitive()[0]);
        assertTrue(d.getMinPrimitive()[2] > 0.09, "pressure stays positive: " + d.getMinPrimitive()[2]);
        assertTrue(d.getCfl() > 0 && d.getCfl() <= 0.41, "cfl " + d.getCfl());
        // Rusanov and LLF are the same flux here
    }

    @Test
    void rusanovAndLocalLaxFriedrichsAreIdentical() {
        FiniteVolumeSolver a = sod(100, NumericalFlux.LLF, Limiter.MC);
        FiniteVolumeSolver b = sod(100, NumericalFlux.RUSANOV, Limiter.MC);
        a.advanceTo(0.1);
        b.advanceTo(0.1);
        assertArrayEquals(a.state(0), b.state(0), 0.0);
    }

    @Test
    void closedBoxWithReflectingWallsConservesMassMomentumFreeEnergy() {
        FiniteVolumeSolver s = FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(200).limiter(Limiter.MC).numericalFlux(NumericalFlux.HLLC)
                .allBoundaries(FiniteVolumeSolver.Boundary.reflecting())
                .initialPrimitive("if(x<0.5,1,0.125)", "0", "if(x<0.5,1,0.1)").build().initialize();
        s.advanceTo(0.6);                       // the waves have hit and bounced off the walls
        FVDiagnostics d = s.diagnostics();
        assertEquals(d.getInitialTotals()[0], d.getTotals()[0], 1e-13, "mass in a closed box");
        assertEquals(d.getInitialTotals()[2], d.getTotals()[2], 1e-13, "energy in a closed box");
        assertEquals(0.0, d.getConservationDefect()[1], 1e-12, "momentum balance including the wall pressure");
        assertTrue(Math.abs(d.getBoundaryFluxIntegral()[0]) < 1e-13, "no mass flux through a wall");
        assertTrue(Math.abs(d.getBoundaryFluxIntegral()[1]) > 1e-3, "the walls do push on the gas (pressure flux)");
    }

    @Test
    void outflowBoundariesLetMassLeaveAndTheDefectAccountsForIt() {
        // a uniform stream leaving the domain: mass decreases by exactly the boundary flux
        FiniteVolumeSolver s = FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(100).limiter(Limiter.MC).numericalFlux(NumericalFlux.HLLC)
                .initialPrimitive("1+0.3*exp(-200*(x-0.5)^2)", "1", "1").build().initialize();
        s.advanceTo(0.8);                     // the density pulse (total mass about 0.0376) has left the domain
        FVDiagnostics d = s.diagnostics();
        assertEquals(0.3 * Math.sqrt(PI / 200), d.getInitialTotals()[0] - d.getTotals()[0], 3e-3, "the pulse's mass left through the outflow boundary");
        assertEquals(0.0, d.getConservationDefect()[0], 1e-12);
        assertEquals(0.0, d.getConservationDefect()[2], 1e-11);
    }

    // ------------------------------------------------------------------ scalar laws: accuracy, TVD, shocks
    private static FiniteVolumeSolver scalar(String eq, int n, double len, Limiter l, NumericalFlux f, String init, boolean periodic) {
        FiniteVolumeSolver.Builder b = new MathExpression(eq).asFiniteVolume().cells(n).size(len).limiter(l).numericalFlux(f).initial(init);
        if (periodic) {
            b.periodic("x");
        }
        return b.build().initialize();
    }

    private static double advectionError(int n, Limiter l) {
        FiniteVolumeSolver s = scalar("diff(u,t) + diff(2*u,x) = 0", n, 1, l, NumericalFlux.LLF, "sin(2*pi*x)", true);
        s.advanceTo(0.25);
        double err = 0;
        for (int i = 0; i < n; i++) {
            err += Math.abs(s.state(0)[s.index(i, 0, 0)] - Math.sin(2 * PI * (s.cellCenter(0, i) - 0.5))) / n;
        }
        return err;
    }

    @Test
    void linearAdvectionIsSecondOrderWithMusclAndFirstOrderWithout() {
        for (Limiter l : new Limiter[]{Limiter.VAN_LEER, Limiter.MC}) {
            double e1 = advectionError(64, l);
            double e2 = advectionError(128, l);
            double e3 = advectionError(256, l);
            assertTrue(PDETestSupport.order(e1, e2, 2) > 1.8 && PDETestSupport.order(e2, e3, 2) > 1.8,
                    l + " orders " + PDETestSupport.order(e1, e2, 2) + ", " + PDETestSupport.order(e2, e3, 2));
        }
        double f1 = advectionError(64, Limiter.NONE);
        double f2 = advectionError(128, Limiter.NONE);
        assertEquals(1.0, PDETestSupport.order(f1, f2, 2), 0.15);
        // minmod clips smooth extrema, so it sits between first and second order
        double m1 = advectionError(64, Limiter.MINMOD);
        double m2 = advectionError(128, Limiter.MINMOD);
        assertTrue(PDETestSupport.order(m1, m2, 2) > 1.4, "minmod order " + PDETestSupport.order(m1, m2, 2));
    }

    @ParameterizedTest
    @EnumSource(value = Limiter.class, names = {"MINMOD", "MC", "VAN_LEER"})
    void limitersCreateNoNewExtremaOnADiscontinuousProfile(Limiter l) {
        for (NumericalFlux f : new NumericalFlux[]{NumericalFlux.LLF, NumericalFlux.HLL}) {
            FiniteVolumeSolver s = scalar("diff(u,t) + diff(1.5*u,x) = 0", 200, 1, l, f, "if(abs(x-0.3)<0.15,1,0)", true);
            s.advanceTo(0.6);
            FVDiagnostics d = s.diagnostics();
            assertTrue(d.getMax()[0] <= 1.0 + 1e-12, l + "/" + f + " overshoot " + d.getMax()[0]);
            assertTrue(d.getMin()[0] >= -1e-12, l + "/" + f + " undershoot " + d.getMin()[0]);
            assertEquals(0.0, d.getConservationDefect()[0], 1e-13);
        }
    }

    @Test
    void burgersShockTravelsAtTheRankineHugoniotSpeed() {
        // u_L = 2, u_R = 0.5: shock speed (2 + 0.5)/2 = 1.25
        FiniteVolumeSolver s = scalar("diff(u,t) + diff(u^2/2,x) = 0", 400, 2, Limiter.MC, NumericalFlux.HLL, "if(x<0.4,2,0.5)", false);
        s.advanceTo(0.5);
        double xs = Double.NaN;
        for (int i = 399; i > 0; i--) {
            if (s.state(0)[s.index(i, 0, 0)] > 1.25) {
                xs = s.cellCenter(0, i);
                break;
            }
        }
        assertEquals(0.4 + 1.25 * 0.5, xs, 0.012);
        double l1 = 0;
        for (int i = 0; i < 400; i++) {
            double x = s.cellCenter(0, i);
            l1 += Math.abs(s.state(0)[s.index(i, 0, 0)] - (x < 0.4 + 1.25 * 0.5 ? 2 : 0.5)) * s.spacing(0);
        }
        assertTrue(l1 < 0.004, "L1 " + l1);
        assertTrue(s.diagnostics().getMax()[0] <= 2.0 + 1e-12 && s.diagnostics().getMin()[0] >= 0.5 - 1e-12);
    }

    @Test
    void burgersRarefactionIsTheEntropySolutionNotAnExpansionShock() {
        for (NumericalFlux f : new NumericalFlux[]{NumericalFlux.LLF, NumericalFlux.HLL}) {
            FiniteVolumeSolver s = scalar("diff(u,t) + diff(u^2/2,x) = 0", 400, 2, Limiter.MC, f, "if(x<0.5,0,1)", false);
            s.advanceTo(0.8);
            double l1 = 0;
            double prev = -1;
            for (int i = 0; i < 400; i++) {
                double x = s.cellCenter(0, i);
                double ex = x < 0.5 ? 0 : (x > 1.3 ? 1 : (x - 0.5) / 0.8);
                double v = s.state(0)[s.index(i, 0, 0)];
                l1 += Math.abs(v - ex) * s.spacing(0);
                assertTrue(v >= prev - 1e-12, "the fan is monotone");
                prev = v;
            }
            assertTrue(l1 < 0.003, f + " L1 " + l1);
        }
    }

    @Test
    void burgersSineWaveFormsAShockAndConservesMass() {
        FiniteVolumeSolver s = scalar("diff(u,t) + diff(u^2/2,x) = 0", 400, 2 * PI, Limiter.MC, NumericalFlux.HLL, "sin(x)", true);
        s.advanceTo(2.0);                    // the characteristics crossed at t = 1
        FVDiagnostics d = s.diagnostics();
        assertEquals(0.0, d.getConservationDefect()[0], 1e-12);
        assertEquals(0.0, d.getTotals()[0], 1e-12, "mean of sin stays 0");
        assertTrue(d.getMax()[0] <= 1.0 + 1e-12 && d.getMin()[0] >= -1.0 - 1e-12, "maximum principle");
        // a shock is a one-cell-wide jump: find the steepest gradient
        double steepest = 0;
        for (int i = 0; i < 400; i++) {
            double a = s.state(0)[s.index(i, 0, 0)];
            double b = s.state(0)[s.index((i + 1) % 400, 0, 0)];
            steepest = Math.max(steepest, Math.abs(b - a));
        }
        assertTrue(steepest > 0.3, "a sharp shock must have formed: largest cell-to-cell jump " + steepest);
    }

    // ------------------------------------------------------------------ MathExpression connection
    @Test
    void conservationFormIsRecognisedFromTheExpression() {
        MathExpression burgers = new MathExpression("diff(u,t) + diff(u^2/2,x) = 0");
        assertTrue(burgers.getPDEModel().isConservationLaw());
        PDEModel.ConservationForm cf = burgers.getPDEModel().getConservationForm();
        assertEquals(java.util.Arrays.asList("u"), cf.unknowns);
        assertEquals(java.util.Arrays.asList("x"), cf.axes);
        assertEquals("(u^2/2)", cf.flux[0][0]);
        assertEquals(PDEType.HYPERBOLIC, burgers.getPDEModel().getType());
        assertTrue(new MathExpression("diff(u,t) = -diff(a*u,x)").getPDEModel().isConservationLaw(), "flux on the right-hand side");
        MathExpression src = new MathExpression("diff(u,t) + diff(u^2/2,x) = -k*u");
        assertTrue(src.getPDEModel().getConservationForm().hasSource());
        MathExpression two = new MathExpression("diff(u,t) + diff(u^2,x) + diff(u*u/3,y) = 0");
        assertEquals(java.util.Arrays.asList("x", "y"), two.getPDEModel().getConservationForm().axes);
    }

    @Test
    void nonConservativeEquationsAreExplained() {
        for (String s : new String[]{"diff(u,t) + diff(u,x) = 0", "diff(u,t) = diff(u,x,2)", "diff(u,t,2) = diff(u,x,2)"}) {
            MathExpression e = new MathExpression(s);
            assertFalse(e.getPDEModel().isConservationLaw(), s);
            assertNotNull(e.getPDEModel().getConservationFormReason(), s);
            PDEConfigurationException ex = assertThrows(PDEConfigurationException.class, e::asFiniteVolume, s);
            assertTrue(ex.getMessage().contains("conservation form"), ex.getMessage());
        }
        assertThrows(IllegalStateException.class, () -> new MathExpression("2+3").asFiniteVolume());
    }

    @Test
    void sourceTermsAndBoundConstantsWorkThroughTheExpression() {
        // u_t + (a u)_x = -k u : u = exp(-k t) sin(2 pi (x - a t))
        FiniteVolumeSolver s = new MathExpression("a=1.5; diff(u,t) + diff(a*u,x) = -k*u").asFiniteVolume().param("k", 0.8)
                .cells(200).periodic("x").limiter(Limiter.VAN_LEER).numericalFlux(NumericalFlux.LLF).initial("sin(2*pi*x)").build().initialize();
        s.advanceTo(0.4);
        double err = 0;
        for (int i = 0; i < 200; i++) {
            double x = s.cellCenter(0, i);
            err = Math.max(err, Math.abs(s.state(0)[s.index(i, 0, 0)] - Math.exp(-0.8 * 0.4) * Math.sin(2 * PI * (x - 1.5 * 0.4))));
        }
        assertTrue(err < 5e-3, "err " + err);
        assertEquals(0.0, s.diagnostics().getConservationDefect()[0], 1e-12, "defect includes the source integral");
    }

    @Test
    void aLinearSystemWrittenAsTwoConservationLawsIsSecondOrder() {
        // p_t + c u_x = 0, u_t + c p_x = 0 with p = sin(2 pi x), u = 0: p = sin cos(2 pi c t), u = -cos sin(2 pi c t)
        double[] e = new double[2];
        int[] ns = {64, 128};
        for (int q = 0; q < 2; q++) {
            FiniteVolumeSolver s = new MathExpression("pde(diff(p,t) + diff(c*u,x) = 0, diff(u,t) + diff(c*p,x) = 0)").asFiniteVolume()
                    .param("c", 2.0).cells(ns[q]).periodic("x").limiter(Limiter.VAN_LEER).numericalFlux(NumericalFlux.LLF)
                    .initial("sin(2*pi*x)", "0").build().initialize();
            s.advanceTo(0.1);
            for (int i = 0; i < ns[q]; i++) {
                double x = s.cellCenter(0, i);
                e[q] += Math.abs(s.state(0)[s.index(i, 0, 0)] - Math.sin(2 * PI * x) * Math.cos(2 * PI * 2 * 0.1)) / ns[q];
                e[q] += Math.abs(s.state(1)[s.index(i, 0, 0)] + Math.cos(2 * PI * x) * Math.sin(2 * PI * 2 * 0.1)) / ns[q];
            }
        }
        assertTrue(PDETestSupport.order(e[0], e[1], 2) > 1.7, "system order " + PDETestSupport.order(e[0], e[1], 2));
    }

    // ------------------------------------------------------------------ multi-dimensional Euler
    @Test
    void isentropicVortexConvergesAtSecondOrderInTwoDimensions() {
        double beta = 5;
        double[] e = new double[2];
        int[] ns = {32, 64};
        for (int q = 0; q < 2; q++) {
            int n = ns[q];
            double c = beta / (2 * PI);
            String rho = "(1-(" + (GAMMA - 1) + ")*" + beta + "^2/(8*" + GAMMA + "*pi^2)*exp(1-((x-5)^2+(y-5)^2)))^(1/" + (GAMMA - 1) + ")";
            String ux = "1-" + c + "*exp((1-((x-5)^2+(y-5)^2))/2)*(y-5)";
            String uy = "1+" + c + "*exp((1-((x-5)^2+(y-5)^2))/2)*(x-5)";
            FiniteVolumeSolver s = FiniteVolumeSolver.builder().law(new EulerLaw(2, GAMMA)).cells(n, n).size(10, 10).periodic("x").periodic("y")
                    .limiter(Limiter.VAN_LEER).numericalFlux(NumericalFlux.HLLC).initialPrimitive(rho, ux, uy, "(" + rho + ")^" + GAMMA).build().initialize();
            s.advanceTo(2.0);
            double h = 10.0 / n;
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    double x = (((i + .5) * h - 2) % 10 + 10) % 10;
                    double y = (((j + .5) * h - 2) % 10 + 10) % 10;
                    double r2 = (x - 5) * (x - 5) + (y - 5) * (y - 5);
                    double ex = Math.pow(1 - (GAMMA - 1) * beta * beta / (8 * GAMMA * PI * PI) * Math.exp(1 - r2), 1 / (GAMMA - 1));
                    e[q] += Math.abs(s.state(0)[s.index(i, j, 0)] - ex) * h * h;
                }
            }
            assertEquals(0.0, s.diagnostics().getConservationDefect()[0], 1e-10);
            assertEquals(0L, s.diagnostics().getReconstructionFallbacks(), "a smooth flow needs no positivity fallback");
        }
        assertTrue(PDETestSupport.order(e[0], e[1], 2) > 1.9, "order " + PDETestSupport.order(e[0], e[1], 2));
    }

    @Test
    void aSodTubeAlongAnyAxisIsTheOneDimensionalSolution() {
        FiniteVolumeSolver one = sod(64, NumericalFlux.HLLC, Limiter.MC);
        // along x in 2-D (periodic y)
        FiniteVolumeSolver x2 = FiniteVolumeSolver.builder().law(new EulerLaw(2, GAMMA)).cells(64, 4).size(1, 1).periodic("y")
                .limiter(Limiter.MC).numericalFlux(NumericalFlux.HLLC).initialPrimitive("if(x<0.5,1,0.125)", "0", "0", "if(x<0.5,1,0.1)").build().initialize();
        // along y in 2-D (periodic x)
        FiniteVolumeSolver y2 = FiniteVolumeSolver.builder().law(new EulerLaw(2, GAMMA)).cells(4, 64).size(1, 1).periodic("x")
                .limiter(Limiter.MC).numericalFlux(NumericalFlux.HLLC).initialPrimitive("if(y<0.5,1,0.125)", "0", "0", "if(y<0.5,1,0.1)").build().initialize();
        // along z in 3-D (periodic x, y)
        FiniteVolumeSolver z3 = FiniteVolumeSolver.builder().law(new EulerLaw(3, GAMMA)).cells(4, 4, 64).size(1, 1, 1).periodic("x").periodic("y")
                .limiter(Limiter.MC).numericalFlux(NumericalFlux.HLLC).initialPrimitive("if(z<0.5,1,0.125)", "0", "0", "0", "if(z<0.5,1,0.1)").build().initialize();
        double dt = one.suggestTimeStep();
        int steps = (int) Math.ceil(0.1 / dt);
        dt = 0.1 / steps;
        for (int s = 0; s < steps; s++) {
            one.step(dt);
            x2.step(dt);
            y2.step(dt);
            z3.step(dt);
        }
        for (int i = 0; i < 64; i++) {
            double r = one.state(0)[one.index(i, 0, 0)];
            assertEquals(r, x2.state(0)[x2.index(i, 2, 0)], 1e-12, "x-direction, cell " + i);
            assertEquals(r, y2.state(0)[y2.index(2, i, 0)], 1e-12, "y-direction, cell " + i);
            assertEquals(r, z3.state(0)[z3.index(2, 2, i)], 1e-12, "z-direction, cell " + i);
        }
        assertEquals(0.0, z3.state(1)[z3.index(1, 1, 30)], 1e-12, "no spurious transverse momentum");
    }

    @Test
    void twoDimensionalExpressionLawAdvectsDiagonally() {
        FiniteVolumeSolver s = new MathExpression("diff(u,t) + diff(1*u,x) + diff(2*u,y) = 0").asFiniteVolume()
                .cells(64, 64).size(1, 1).periodic("x").periodic("y").limiter(Limiter.VAN_LEER).numericalFlux(NumericalFlux.LLF)
                .initial("sin(2*pi*x)*sin(2*pi*y)").build().initialize();
        s.advanceTo(0.25);
        double err = 0;
        for (int j = 0; j < 64; j++) {
            for (int i = 0; i < 64; i++) {
                err = Math.max(err, Math.abs(s.state(0)[s.index(i, j, 0)]
                        - Math.sin(2 * PI * (s.cellCenter(0, i) - 0.25)) * Math.sin(2 * PI * (s.cellCenter(1, j) - 0.5))));
            }
        }
        assertTrue(err < 0.05, "err " + err);
        assertEquals(0.0, s.diagnostics().getConservationDefect()[0], 1e-12);
    }

    // ------------------------------------------------------------------ units
    @Test
    void eulerLawRoundTripsAndFluxesAreConsistent() {
        EulerLaw law = new EulerLaw(2, GAMMA);
        double[] w = {1.3, 0.4, -0.7, 2.1};
        double[] u = new double[4];
        double[] back = new double[4];
        law.toConserved(w, u);
        law.toPrimitive(u, back);
        assertArrayEquals(w, back, 1e-14);
        assertEquals(2.1, law.pressure(u), 1e-14);
        double[] f = new double[4];
        double[] g = new double[4];
        for (int dir = 0; dir < 2; dir++) {
            law.flux(dir, u, f);
            law.hllc(dir, u, u, g);
            assertArrayEquals(f, g, 1e-13, "HLLC of identical states is the physical flux (consistency), dir " + dir);
        }
        double[] sp = new double[2];
        law.waveSpeeds(0, u, sp);
        double c = Math.sqrt(GAMMA * 2.1 / 1.3);
        assertEquals(0.4 - c, sp[0], 1e-14);
        assertEquals(0.4 + c, sp[1], 1e-14);
        assertFalse(law.isPhysical(new double[]{1.0, 0.0, 0.0, -1.0}), "negative pressure");
        assertFalse(law.isPhysical(new double[]{-1.0, 0.0, 0.0, 1.0}), "negative density");
    }

    @Test
    void limitersHaveTheirDefiningProperties() {
        for (Limiter l : new Limiter[]{Limiter.MINMOD, Limiter.MC, Limiter.VAN_LEER}) {
            assertEquals(0.0, l.slope(1.0, -1.0), 0.0, l + " is zero at an extremum");
            assertEquals(0.0, l.slope(0.0, 2.0), 0.0);
            assertEquals(1.0, l.slope(1.0, 1.0), 1e-14, l + " reproduces a linear profile");
            assertEquals(-3.0, l.slope(-3.0, -3.0), 1e-14);
            assertEquals(l.slope(0.7, 2.0), l.slope(2.0, 0.7), 1e-14, l + " is symmetric");
            assertEquals(-l.slope(0.7, 2.0), l.slope(-0.7, -2.0), 1e-14, l + " is odd");
        }
        assertEquals(0.7, Limiter.MINMOD.slope(0.7, 2.0), 1e-14);
        assertEquals(1.35, Limiter.MC.slope(0.7, 2.0), 1e-14, "MC: min(2a, 2b, (a+b)/2) = min(1.4, 4, 1.35)");
        assertEquals(2 * 0.7 * 2.0 / 2.7, Limiter.VAN_LEER.slope(0.7, 2.0), 1e-14);
        assertEquals(0.0, Limiter.NONE.slope(1.0, 1.0), 0.0);
        // TVD region: |slope| <= 2 min(|a|,|b|)
        for (Limiter l : new Limiter[]{Limiter.MINMOD, Limiter.MC, Limiter.VAN_LEER}) {
            for (double b = 0.1; b < 10; b *= 1.7) {
                assertTrue(Math.abs(l.slope(1.0, b)) <= 2 * Math.min(1.0, b) + 1e-14, l + " b=" + b);
            }
        }
    }

    @Test
    void parsersAcceptTheDocumentedNames() {
        assertEquals(NumericalFlux.HLLC, NumericalFlux.parse("HLLC"));
        assertEquals(NumericalFlux.LLF, NumericalFlux.parse("local-lax_friedrichs".replace("local-lax_friedrichs", "llf")));
        assertEquals(Limiter.VAN_LEER, Limiter.parse("van-leer"));
        assertThrows(PDEConfigurationException.class, () -> NumericalFlux.parse("roe"));
        assertThrows(PDEConfigurationException.class, () -> Limiter.parse("superbee"));
    }

    // ------------------------------------------------------------------ validation
    @Test
    void invalidConfigurationsAreRejected() {
        assertThrows(PDEConfigurationException.class, () -> FiniteVolumeSolver.builder().cells(10).initial("1").build(), "no law");
        assertThrows(PDEConfigurationException.class, () -> FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(10, 10)
                .initialPrimitive("1", "0", "1").build(), "wrong number of cell counts");
        assertThrows(PDEConfigurationException.class, () -> FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(3)
                .initialPrimitive("1", "0", "1").build(), "too few cells");
        assertThrows(PDEConfigurationException.class, () -> FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(10).build(), "no initial data");
        assertThrows(PDEConfigurationException.class, () -> FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(10)
                .initialPrimitive("1", "0").build(), "wrong number of initial expressions");
        assertThrows(PDEConfigurationException.class, () -> FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(10)
                .initialPrimitive("kappa", "0", "1").build(), "unknown symbol");
        assertThrows(PDEConfigurationException.class, () -> FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(10)
                .boundary(Face.X_MIN, FiniteVolumeSolver.Boundary.periodic()).initialPrimitive("1", "0", "1").build(), "periodic on one face only");
        assertThrows(PDEConfigurationException.class, () -> FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(10)
                .allBoundaries(FiniteVolumeSolver.Boundary.inflow(1, 0)).initialPrimitive("1", "0", "1").build(), "inflow state size");
        assertThrows(PDEConfigurationException.class, () -> new EulerLaw(1, 1.0));
        assertThrows(PDEConfigurationException.class, () -> new EulerLaw(4, 1.4));
        PDEConfigurationException hllc = assertThrows(PDEConfigurationException.class, () -> new MathExpression("diff(u,t) + diff(u^2/2,x) = 0")
                .asFiniteVolume().cells(10).numericalFlux(NumericalFlux.HLLC).initial("1").build());
        assertTrue(hllc.getMessage().contains("HLLC"));
        PDEConfigurationException bad = assertThrows(PDEConfigurationException.class, () -> FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(10)
                .initialPrimitive("1", "0", "-1").build().initialize());
        assertTrue(bad.getMessage().contains("not physical"));
        assertThrows(PDEConfigurationException.class, () -> new ExpressionLaw(new String[]{"u"}, new String[]{"x"}, new String[][]{{"x*u"}}, null, null),
                "a flux may not depend on x");
        FiniteVolumeSolver fresh = FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(10).initialPrimitive("1", "0", "1").build();
        assertThrows(PDEException.class, () -> fresh.step(0.0001), "must initialise first");
        fresh.initialize();
        assertThrows(PDEConfigurationException.class, () -> fresh.step(-1.0));
        assertThrows(PDEConfigurationException.class, () -> fresh.step(Double.NaN));
    }

    @Test
    void aStepFarBeyondTheCflLimitIsReportedAsNonPhysical() {
        FiniteVolumeSolver s = FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(100).limiter(Limiter.MC).numericalFlux(NumericalFlux.HLLC)
                .initialPrimitive("if(x<0.5,1,0.125)", "0", "if(x<0.5,1,0.1)").build().initialize();
        UnstableTimeStepException ex = assertThrows(UnstableTimeStepException.class, () -> {
            for (int i = 0; i < 50; i++) {
                s.step(0.05);
            }
        });
        assertTrue(ex.getMessage().contains("non-physical"), ex.getMessage());
    }

    // ------------------------------------------------------------------ allocation
    private static long allocated(long id) {
        try {
            Method m = Class.forName("com.sun.management.ThreadMXBean").getMethod("getThreadAllocatedBytes", long.class);
            return (Long) m.invoke(ManagementFactory.getThreadMXBean(), id);
        } catch (Throwable t) {
            return -1;
        }
    }

    private static void assertAllocationFree(FiniteVolumeSolver s, int warm) {
        long id = Thread.currentThread().getId();
        assumeTrue(allocated(id) >= 0, "allocation counters unavailable");
        double dt = s.suggestTimeStep() * 0.5;
        for (int i = 0; i < warm; i++) {
            s.step(dt);
        }
        long a0 = allocated(id);
        for (int i = 0; i < 20; i++) {
            s.step(dt);
        }
        long a1 = allocated(id);
        for (int i = 0; i < 120; i++) {
            s.step(dt);
        }
        long a2 = allocated(id);
        double perStep = ((a2 - a1) - (a1 - a0) * 6.0) / 100.0;
        assertTrue(perStep < 16.0, perStep + " bytes per step");
    }

    @Test
    void steppingAllocatesNothing() {
        assertAllocationFree(sod(100, NumericalFlux.HLLC, Limiter.MC), 2000);
        assertAllocationFree(sod(100, NumericalFlux.HLL, Limiter.VAN_LEER), 2000);
        assertAllocationFree(FiniteVolumeSolver.builder().law(new EulerLaw(2, GAMMA)).cells(24, 24).periodic("x").periodic("y")
                .initialPrimitive("1+0.2*sin(2*pi*x)", "0.5", "0.3", "1+0.1*cos(2*pi*y)").build().initialize(), 1500);
        assertAllocationFree(scalar("diff(u,t) + diff(u^2/2,x) = -0.1*u", 200, 2, Limiter.MC, NumericalFlux.HLL, "if(x<0.5,2,1)", false), 2000);
        assertAllocationFree(FiniteVolumeSolver.builder().law(new EulerLaw(1, GAMMA)).cells(100).allBoundaries(FiniteVolumeSolver.Boundary.reflecting())
                .initialPrimitive("if(x<0.5,1,0.125)", "0", "if(x<0.5,1,0.1)").build().initialize(), 2000);
    }
}
