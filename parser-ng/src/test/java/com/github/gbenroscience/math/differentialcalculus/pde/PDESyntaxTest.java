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
 * Front-end syntax: scientific notation, equation arrays, derivative
 * spellings, and graceful degradation when the automatic differentiator
 * cannot handle an expression.
 */
public class PDESyntaxTest {

    @Test
    void scientificNotationIsNotMistakenForEulersNumber() {
        assertEquals("2E-5", CompiledExpr.normalizeScientific("2e-5"));
        assertEquals("3.5E+2*x", CompiledExpr.normalizeScientific("3.5e+2*x"));
        assertEquals("1E3+1", CompiledExpr.normalizeScientific("1e3+1"));
        assertEquals(".5E2", CompiledExpr.normalizeScientific(".5e2"));
        assertEquals("2*e-5", CompiledExpr.normalizeScientific("2*e-5"), "Euler's number untouched");
        assertEquals("x2e3", CompiledExpr.normalizeScientific("x2e3"), "inside an identifier");
        assertEquals("a1e5", CompiledExpr.normalizeScientific("a1e5"));
        assertEquals("exp(-2E-3*t)", CompiledExpr.normalizeScientific("exp(-2e-3*t)"));
        assertEquals("2e", CompiledExpr.normalizeScientific("2e"));
        assertEquals("3E2", CompiledExpr.normalizeScientific("3E2"));
    }

    @Test
    void scientificNotationInEquationsAndConditions() {
        PDEResult a = PDE.problem("diff(u,t) = 1e-1*diff(u,x,2)").domain("x", 0, 1, 101)
                .initial("u", "1e0*sin(pi*x)").dirichlet(Face.X_MIN, "u", "0e0").dirichlet(Face.X_MAX, "u", "1e-30")
                .solve(0.5, 1e-4, "rk4");
        assertTrue(a.maxError("u", PDETestSupport.heat1dExact(0.1)) < 1e-4);
    }

    @Test
    void symbolsAreIsolatedFromParserNgsGlobalVariableRegistry() {
        // ParserNG keeps variables in a process-wide registry; names defined elsewhere in the application must not
        // leak into (and silently change) a PDE problem
        new com.github.gbenroscience.parser.MathExpression("kappa=99").solve();
        new com.github.gbenroscience.parser.MathExpression("zeta=7").solve();
        PDEResult r = PDE.problem("diff(zeta,t) = kappa*diff(zeta,x,2)").param("kappa", 0.1).domain("x", 0, 1, 101)
                .initial("zeta", "sin(pi*x)").dirichlet("zeta", "0").solve(0.5, 1e-4, "rk4");
        assertTrue(r.maxError("zeta", PDETestSupport.heat1dExact(0.1)) < 1e-4, "the declared parameter must win over the global one");
    }

    @Test
    void symbolIsolationRenamesVariablesButNotFunctionsOrConstants() {
        java.util.Map<String, String> m = new java.util.LinkedHashMap<String, String>();
        String out = CompiledExpr.isolateSymbols("2x + sin(y*pi) - exp(-3E-2*t) + e + alpha(1)", m);
        assertEquals("2*pdeQ0 + sin(pdeQ1*pi) - exp(-3E-2*pdeQ2) + e + alpha(1)", out);
        assertEquals(java.util.Arrays.asList("x", "y", "t"), new java.util.ArrayList<String>(m.keySet()));
    }

    @Test
    void equationArraySyntaxMatchesVarargs() {
        PDEResult a = PDE.problem("@(2)(\"diff(u,t) = 0.1*diff(u,x,2) - v\",\"diff(v,t) = 0.1*diff(v,x,2) + u\")")
                .domain("x", 0, 1, 41).initial("u", "sin(pi*x)").initial("v", "0").dirichlet("*", "0").solve(0.2, 1e-3, "rk4");
        PDEResult b = PDE.problem("diff(u,t) = 0.1*diff(u,x,2) - v", "diff(v,t) = 0.1*diff(v,x,2) + u")
                .domain("x", 0, 1, 41).initial("u", "sin(pi*x)").initial("v", "0").dirichlet("*", "0").solve(0.2, 1e-3, "rk4");
        assertArrayEquals(a.values("u"), b.values("u"), 0.0);
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("@(3)(\"diff(u,t)=0\",\"diff(v,t)=0\")"));
    }

    @Test
    void explicitUnknownsFixTheFieldOrder() {
        PDEResult r = PDE.problem("diff(u,t) = 0.1*diff(u,x,2) - v", "diff(v,t) = 0.1*diff(v,x,2) + u").unknowns("v", "u")
                .domain("x", 0, 1, 21).initial("u", "sin(pi*x)").initial("v", "0").dirichlet("*", "0").solve(0.01, 1e-3, "rk4");
        assertEquals(java.util.Arrays.asList("v", "u"), r.getUnknowns());
    }

    @Test
    void diffTargetsOutsideTheDeclaredUnknownsAreRejected() {
        assertThrows(PDEConfigurationException.class, () -> PDE.problem("diff(u,t) = diff(w,x,2)").unknowns("u")
                .domain("x", 0, 1, 11).initial("u", "0").dirichlet("u", "0").solver());
    }

    @Test
    void expressionsInsideDiffAreLeftToParserNg() {
        // diff of an ordinary expression is ParserNG's own symbolic differentiation, untouched by the PDE front end
        PDEResult r = PDE.problem("diff(u,t) = diff(u,x,2) + 0*x").domain("x", 0, 1, 21).initial("u", "sin(pi*x)")
                .dirichlet("u", "0").solve(0.01, 1e-4, "rk4");
        assertTrue(r.isConverged());
    }

    @Test
    void axisNamesAreFree() {
        PDEResult r = PDE.problem("diff(T,t) = 0.1*(diff(T,r,2) + diff(T,s,2))").domain("r", 0, 1, 31).domain("s", 0, 1, 31)
                .initial("T", "sin(pi*r)*sin(pi*s)").dirichlet("T", "0").solve(0.1, 2e-3, "crank_nicolson");
        double err = r.maxError("T", (x, y, z, t) -> Math.exp(-2 * Math.PI * Math.PI * 0.1 * t) * Math.sin(Math.PI * x) * Math.sin(Math.PI * y));
        assertTrue(err < 1e-3, "err " + err);
    }

    @Test
    void expressionsTheAutoDifferentiatorCannotHandleFallBackToFiniteDifferences() {
        // comparisons inside if() are outside the automatic differentiator's operator set
        PDEResult r = PDE.problem("diff(u,t) = 0.1*diff(u,x,2) + if(u>0.5,0.1,0)").domain("x", 0, 1, 41)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(0.2, 1e-2, "crank_nicolson");
        assertTrue(r.isConverged());
        assertTrue(r.getDiagnostics().getMethod().contains("finite-difference Jacobian"), r.getDiagnostics().getMethod());
        // the same structure but differentiable: exact Jacobian
        PDEResult q = PDE.problem("diff(u,t) = 0.1*diff(u,x,2) + 0.1*u*(1-u)").domain("x", 0, 1, 41)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(0.2, 1e-2, "crank_nicolson");
        assertTrue(q.getDiagnostics().getMethod().contains("exact AD Jacobian"), q.getDiagnostics().getMethod());
    }

    @Test
    void exactAndFiniteDifferenceJacobiansGiveTheSameSolution() {
        String smooth = "diff(u,t) = 0.1*diff(u,x,2) + 0.3*u*(1-u)";
        PDEResult exact = PDE.problem(smooth).domain("x", 0, 1, 41).initial("u", "0.5*sin(pi*x)").dirichlet("u", "0")
                .options(new PDEOptions().integrator("crank_nicolson").dt(1e-2).newtonTolerance(1e-13).krylovTolerance(1e-12)).solve(0.5);
        // abs() is differentiable almost everywhere but the automatic differentiator does not support it with this form: use if()
        String nonSmoothButEquivalent = "diff(u,t) = 0.1*diff(u,x,2) + 0.3*u*(1-u) + if(u>1e9,1,0)";
        PDEResult fd = PDE.problem(nonSmoothButEquivalent).domain("x", 0, 1, 41).initial("u", "0.5*sin(pi*x)").dirichlet("u", "0")
                .options(new PDEOptions().integrator("crank_nicolson").dt(1e-2).newtonTolerance(1e-13).krylovTolerance(1e-12)).solve(0.5);
        assertTrue(fd.getDiagnostics().getMethod().contains("finite-difference"));
        double d = 0;
        for (int i = 0; i < 41; i++) {
            d = Math.max(d, Math.abs(exact.value("u", i) - fd.value("u", i)));
        }
        assertTrue(d < 1e-8, "difference " + d);
    }

    @Test
    void constantCoefficientProblemsUseTheFusedKernelAndAgreeWithTheGenericOne() {
        // 0*x*u makes the equation non-affine for the detector's random test only if the coefficient varies; use a
        // variable coefficient that is numerically 1 everywhere but not recognisable as constant
        PDEResult fused = PDE.problem("diff(u,t) = 0.1*diff(u,x,2)").domain("x", 0, 1, 101).initial("u", "sin(pi*x)")
                .dirichlet("u", "0").solve(0.2, 2e-4, "rk4");
        PDEResult generic = PDE.problem("diff(u,t) = 0.1*(1+0*x)*diff(u,x,2) + sin(0*u)").domain("x", 0, 1, 101).initial("u", "sin(pi*x)")
                .dirichlet("u", "0").solve(0.2, 2e-4, "rk4");
        double d = 0;
        for (int i = 0; i < 101; i++) {
            d = Math.max(d, Math.abs(fused.value("u", i) - generic.value("u", i)));
        }
        assertTrue(d < 1e-12, "fused vs generic kernel " + d);
    }
}
