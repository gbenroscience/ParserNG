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
package com.github.gbenroscience.parser;

import static org.junit.jupiter.api.Assertions.*;

import com.github.gbenroscience.math.differentialcalculus.pde.PDEModel;
import com.github.gbenroscience.math.differentialcalculus.pde.PDEResult;
import com.github.gbenroscience.math.differentialcalculus.pde.PDEType;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.parser.methods.Method;
import java.util.Arrays;
import java.util.InputMismatchException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Milestone 1: MathExpression is the universal entry point for PDE
 * expressions. Covers recognition, the {@code pde} command, the
 * MathExpressionTreeDepth pass-through, the typed {@code asPDE()} access
 * path, and regression of scalar and ODE expressions.
 */
public class PdeExpressionTest {

    private static final double PI = Math.PI;

    private static MathExpressionTreeDepth.Result depth(String s) {
        return new MathExpressionTreeDepth(s, MathExpression.createNewVariableRegistry()).calculate();
    }

    private static String rejection(String s) {
        InputMismatchException ex = assertThrows(InputMismatchException.class, () -> new MathExpression(s), s);
        return ex.getMessage();
    }

    // ------------------------------------------------------------------ 1. scalar regression
    @ParameterizedTest
    @ValueSource(strings = {"2+3*4", "sqrt(x*x + y*y)", "x==3", "1+2==3", "x<=3", "x!=3", "sin(2)^2+cos(2)^2", "max(1,2,3)"})
    void scalarExpressionsAreStillScalar(String s) {
        MathExpression e = new MathExpression(s);
        assertEquals(ExpressionKind.SCALAR, e.getExpressionKind(), s);
        assertFalse(e.isPDE());
        assertNull(e.getPDEModel());
        assertTrue(e.isScannedAndOptimized());
    }

    @Test
    void scalarEvaluationIsUnchanged() {
        assertEquals("14.0", new MathExpression("2+3*4").solve());
        assertEquals("true", new MathExpression("1+2==3").solve());
        assertEquals("false", new MathExpression("1+2==4").solve());
        assertEquals("9.0", new MathExpression("a=3; a^2").solve());
    }

    @Test
    void asPdeOnAScalarExpressionExplainsItself() {
        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> new MathExpression("2+3").asPDE());
        assertTrue(ex.getMessage().contains("not a PDE"), ex.getMessage());
    }

    @Test
    void functionDefinitionsAndAssignmentsAreNotReclassified() {
        new MathExpression("pf(x)=x^3");
        assertEquals(ExpressionKind.SCALAR, new MathExpression("pf(x)=sin(x)").getExpressionKind());
        assertEquals(ExpressionKind.SCALAR, new MathExpression("pq=7").getExpressionKind());
        // ParserNG's own differentiation of a defined function keeps its meaning
        MathExpression d = new MathExpression("diff(pf,1)");
        assertFalse(d.isPDE());
        assertEquals(ExpressionKind.SCALAR, d.getExpressionKind());
    }

    @Test
    void statementClassificationSeparatesPdesFromEverythingElse() {
        PDEModel.StatementKind eq = PDEModel.StatementKind.PDE_EQUATION;
        PDEModel.StatementKind cmd = PDEModel.StatementKind.PDE_COMMAND;
        PDEModel.StatementKind no = PDEModel.StatementKind.NOT_PDE;
        assertEquals(eq, PDEModel.classifyStatement("diff(u,t) = alpha*diff(u,x,2)"));
        assertEquals(eq, PDEModel.classifyStatement("(diff(u,t) = alpha*diff(u,x,2))"));
        assertEquals(eq, PDEModel.classifyStatement("alpha*diff(u,x,2) = diff(u,t)"));
        assertEquals(eq, PDEModel.classifyStatement("diff(u,x,2)+diff(u,y,2) = 0"));
        assertEquals(eq, PDEModel.classifyStatement("diff(diff(u,x),y) = f1*u"));
        assertEquals(cmd, PDEModel.classifyStatement("pde(diff(u,t)=diff(u,x,2))"));
        assertEquals(cmd, PDEModel.classifyStatement("1+pde(diff(u,t)=diff(u,x,2))"), "misplaced command goes to the validator");
        // never PDEs:
        assertEquals(no, PDEModel.classifyStatement("x==3"));
        assertEquals(no, PDEModel.classifyStatement("x<=3"));
        assertEquals(no, PDEModel.classifyStatement("2+3"));
        assertEquals(no, PDEModel.classifyStatement("sin(x)"));
        assertEquals(no, PDEModel.classifyStatement("y = diff(v,x)"), "assignment to a plain name stays an assignment");
        assertEquals(no, PDEModel.classifyStatement("h(x) = diff(x^2,x)"), "symbolic differentiation inside a definition");
        assertEquals(no, PDEModel.classifyStatement("diff(F,2)"), "ParserNG differentiation of a function handle");
        assertEquals(no, PDEModel.classifyStatement("pde = 5"), "a variable that happens to be called pde");
        assertEquals(no, PDEModel.classifyStatement("alpha=0.1"));
    }

    // ------------------------------------------------------------------ 2. ODE regression
    @Test
    void odeExpressionsStillWorkAndAreClassified() {
        MathExpression ode = new MathExpression("diffeqn(y[1]-1, 0, 0, 1, 0.01, rk4)");
        assertEquals(ExpressionKind.ODE, ode.getExpressionKind());
        assertFalse(ode.isPDE());
        assertEquals(1.0, Double.parseDouble(ode.solve()), 1e-9);

        MathExpression sys = new MathExpression(
                "diffeqn(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.01, rk4)");
        assertEquals(ExpressionKind.ODE_SYSTEM, sys.getExpressionKind());
        assertFalse(sys.isPDE());
        assertNotNull(sys.solve());
    }

    @Test
    void odeRootOnlyRuleIsUntouched() {
        assertThrows(InputMismatchException.class, () -> depth("2+diffeqn(y[1]-1, 0, 0, 1, 0.01, rk4)"));
        assertThrows(InputMismatchException.class, () -> depth("sin(diffeqn(y[1]-1, 0, 0, 1, 0.01, rk4))"));
        depth("diffeqn(y[1]-1, 0, 0, 1, 0.01, rk4)");
        depth("((diffeqn(y[1]-1, 0, 0, 1, 0.01, rk4)))");
        assertThrows(InputMismatchException.class, () -> depth("y[3]"), "derivative index notation is still ODE-only");
    }

    // ------------------------------------------------------------------ 3. PDE recognition
    @Test
    void bareEquationIsRecognisedAsAPde() {
        MathExpression e = new MathExpression("diff(u,t) = alpha*diff(u,x,2)");
        assertEquals(ExpressionKind.PDE, e.getExpressionKind());
        assertTrue(e.isPDE());
        assertTrue(e.isScannedAndOptimized());
        PDEModel m = e.getPDEModel();
        assertEquals(Arrays.asList("diff(u,t) = alpha*diff(u,x,2)"), m.getEquations());
        assertEquals(Arrays.asList("u"), m.getUnknowns());
        assertEquals(Arrays.asList("t", "x"), m.getIndependentVariables());
        assertEquals(PDEType.PARABOLIC, m.getType());
        assertFalse(m.isSystem());
    }

    @Test
    void theExplicitPdeCommandAndTheBareFormAreEquivalent() {
        MathExpression a = new MathExpression("diff(u,t) = alpha*diff(u,x,2)");
        MathExpression b = new MathExpression("pde(diff(u,t) = alpha*diff(u,x,2))");
        MathExpression c = new MathExpression("((pde(diff(u,t) = alpha*diff(u,x,2))))");
        assertEquals(a.getPDEModel().getEquations(), b.getPDEModel().getEquations());
        assertEquals(a.getPDEModel().getEquations(), c.getPDEModel().getEquations());
        assertEquals(a.getExpression(), b.getExpression(), "both normalise to the canonical command form");
    }

    @Test
    void pdeSystemsInEveryNotation() {
        String e1 = "diff(u,t) = 0.1*diff(u,x,2) - v";
        String e2 = "diff(v,t) = 0.1*diff(v,x,2) + u";
        MathExpression semi = new MathExpression(e1 + "; " + e2);
        MathExpression args = new MathExpression("pde(" + e1 + ", " + e2 + ")");
        MathExpression array = new MathExpression("pde(@(2)(\"" + e1 + "\",\"" + e2 + "\"))");
        for (MathExpression m : new MathExpression[]{semi, args, array}) {
            assertEquals(ExpressionKind.PDE_SYSTEM, m.getExpressionKind());
            assertEquals(Arrays.asList(e1, e2), m.getPDEModel().getEquations());
            assertEquals(Arrays.asList("u", "v"), m.getPDEModel().getUnknowns());
            assertTrue(m.getPDEModel().isSystem());
        }
    }

    @Test
    void ellipticHyperbolicAndMixedDerivativeFormsAreRecognised() {
        assertEquals(PDEType.ELLIPTIC, new MathExpression("diff(u,x,2)+diff(u,y,2) = -2*pi^2*sin(pi*x)*sin(pi*y)").getPDEModel().getType());
        assertEquals(PDEType.HYPERBOLIC, new MathExpression("diff(u,t,2) = c^2*diff(u,x,2)").getPDEModel().getType());
        assertEquals(PDEType.HYPERBOLIC, new MathExpression("diff(u,t) + c*diff(u,x) = 0").getPDEModel().getType());
        MathExpression mixed = new MathExpression("diff(u,t) = diff(u,x,y) + diff(diff(u,x),y)");
        assertEquals(Arrays.asList("t", "x", "y"), mixed.getPDEModel().getIndependentVariables());
    }

    @Test
    void ordinaryFunctionsConstantsAndScientificNotationMixWithDerivatives() {
        MathExpression e = new MathExpression("diff(u,t) = 1e-3*diff(u,x,2) + sin(pi*x)*exp(-t) + max(u,0)");
        assertEquals(ExpressionKind.PDE, e.getExpressionKind());
        assertEquals(Arrays.asList("u"), e.getPDEModel().getUnknowns());
    }

    @Test
    void constantsAssignedNextToThePdeBecomeProblemParameters() {
        MathExpression e = new MathExpression("alpha=0.1; diff(u,t) = alpha*diff(u,x,2)");
        assertEquals(ExpressionKind.PDE, e.getExpressionKind());
        assertEquals(0.1, e.getPDEModel().getBoundParameters().get("alpha"), 0.0);
        // and they are used by the numerical problem without being declared again
        PDEResult r = e.asPDE().domain("x", 0, 1, 101).initial("u", "sin(pi*x)").dirichlet("u", "0")
                .solve(0.5, 1e-3, "crank_nicolson");
        assertTrue(r.maxError("u", (x, y, z, t) -> Math.exp(-PI * PI * 0.1 * t) * Math.sin(PI * x)) < 1e-4);
    }

    // ------------------------------------------------------------------ 4. TreeDepth pass-through
    @Test
    void validPdeFormsPassThroughTheTreeDepthValidator() {
        for (String s : new String[]{
            "pde(diff(u,t) = alpha*diff(u,x,2))",
            "pde(diff(u,x,2) + diff(u,y,2) = 0)",
            "pde(diff(u,t) = diff(u,x,y))",
            "pde(diff(u,t,2) = c^2*(diff(u,x,2)+diff(u,y,2)))",
            "pde(diff(u,t) = diff(u,x,2) - v, diff(v,t) = diff(v,x,2) + u)",
            "pde(@(2)(\"diff(u,t) = diff(u,x,2) - v\",\"diff(v,t) = diff(v,x,2) + u\"))",
            "pde(diff(u,t) = k(x)*diff(u,x,2) + sin(pi*x)*exp(-t))",
            "((pde(diff(u,t) = diff(u,x,2))))",
            "pde(  diff( u , t )  =  alpha * diff( u , x , 2 )  )"}) {
            assertDoesNotThrow(() -> depth(s), s);
        }
    }

    @Test
    void invalidPdeSyntaxIsRejectedWithMeaningfulDiagnostics() {
        assertTrue(rejection("pde()").contains("at least one equation"));
        assertTrue(rejection("pde(diff(u,t)=)").contains("empty side"));
        assertTrue(rejection("pde(3=4)").contains("no derivative of an unknown"));
        assertTrue(rejection("diff(u,t) = diff(u,x,2) = 3").contains("more than one '='"));
        assertTrue(rejection("diff(u,t,0) = diff(u,x,2)").contains("order must be between 1 and 8"));
        assertTrue(rejection("diff(u,2x) = 0").contains("no derivative of an unknown"));
        assertTrue(rejection("pde(diff(u,t)=sin(").contains("Missing closing"));
        assertTrue(rejection("1+pde(diff(u,t)=diff(u,x,2))").contains("root expression"));
        assertTrue(rejection("sin(pde(diff(u,t)=diff(u,x,2)))").contains("root expression"));
        assertTrue(rejection("pde(diff(u,t)=diff(u,x,2)) + 1").contains("root expression"));
        assertTrue(rejection("pde(diff(u,t)=diff(u,x,2) * )").contains("pde()"), "a broken right-hand side is reported against the equation");
        assertTrue(rejection("pde(@(2)(\"diff(u,t)=diff(u,x,2)\"))").contains("pde()"), "array count must match");
    }

    @Test
    void anExplicitPdeCommandIsImmuneToFunctionsDefinedElsewhere() {
        // the application has a ParserNG function called pq1; inside pde(...) the intent is unambiguous, so diff(pq1,t)
        // is still the derivative of the unknown pq1, not ParserNG's differentiation of the function
        new MathExpression("pq1(x)=x^2");
        MathExpression e = new MathExpression("pde(diff(pq1,t) = diff(pq1,x,2))");
        assertEquals(ExpressionKind.PDE, e.getExpressionKind());
        assertEquals(Arrays.asList("pq1"), e.getPDEModel().getUnknowns());
        // a bare equation whose left side starts with diff( is also unambiguous
        assertEquals(ExpressionKind.PDE, new MathExpression("diff(pq1,t) = diff(pq1,x,2)").getExpressionKind());
    }

    @Test
    void aPdeCannotBeMixedWithAnOrdinaryExpression() {
        String m = rejection("2+3; diff(u,t) = diff(u,x,2)");
        assertTrue(m.contains("cannot be combined"), m);
    }

    // ------------------------------------------------------------------ 5. the command itself
    @Test
    void theCommandIsRegisteredLikeTheOtherInbuiltCommands() {
        assertEquals("pde", Declarations.PDE);
        assertTrue(Declarations.isInbuiltMethodName("pde"));
        assertEquals(Declarations.ARITY_N, Declarations.getInbuiltMethodArity("pde"));
        assertTrue(Declarations.acceptsInbuiltMethodArgCount("pde", 1));
        assertTrue(Declarations.acceptsInbuiltMethodArgCount("pde", 3));
        assertTrue(Method.isSupportedPdeMethod("pde"));
        assertFalse(Method.isSupportedPdeMethod("diffeqn"));
        assertFalse(Method.isSupportedDiffEqnMethod("pde"));
        assertTrue(Method.isMultiArgsMethod("pde"));
        assertEquals(TYPE.STRING.toString(), Declarations.returnTypeDef("pde"));
    }

    @Test
    void aPdeExpressionIsAModelNotANumber() throws Throwable {
        MathExpression e = new MathExpression("diff(u,t) = alpha*diff(u,x,2)");
        assertEquals("pde(diff(u,t) = alpha*diff(u,x,2))", e.solve(), "solving reports the canonical model");
        assertThrows(IllegalStateException.class, e::compileTurbo);
    }

    @Test
    void aPdeExpressionSurvivesCloning() throws Exception {
        MathExpression e = new MathExpression("alpha=0.2; diff(u,t) = alpha*diff(u,x,2)");
        MathExpression c = e.clone();
        assertTrue(c.isPDE());
        assertEquals(e.getPDEModel().getEquations(), c.getPDEModel().getEquations());
        assertEquals(0.2, c.getPDEModel().getBoundParameters().get("alpha"), 0.0);
    }

    // ------------------------------------------------------------------ 6. end to end through MathExpression
    @Test
    void heatEquationEndToEndFromAMathExpression() {
        MathExpression equation = new MathExpression("diff(u,t) = alpha*diff(u,x,2)");
        PDEResult result = equation.asPDE()
                .param("alpha", 0.1)
                .domain("x", 0, 1, 101)
                .initial("u", "sin(pi*x)")
                .dirichlet("u", "0")
                .solve(0.5, 1e-3, "crank_nicolson");
        assertTrue(result.maxError("u", (x, y, z, t) -> Math.exp(-PI * PI * 0.1 * t) * Math.sin(PI * x)) < 1e-4);
    }

    @Test
    void theSameExpressionCanBeSolvedWithDifferentNumericalChoices() {
        MathExpression equation = new MathExpression("diff(u,t) = alpha*diff(u,x,2)");
        double[] errors = new double[3];
        String[] methods = {"rk4", "crank_nicolson", "rk45"};
        double[] dts = {5e-4, 1e-3, 1e-3};
        for (int i = 0; i < 3; i++) {
            PDEResult r = equation.asPDE().param("alpha", 0.1).domain("x", 0, 1, 101)
                    .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(0.5, dts[i], methods[i]);
            errors[i] = r.maxError("u", (x, y, z, t) -> Math.exp(-PI * PI * 0.1 * t) * Math.sin(PI * x));
            assertTrue(errors[i] < 1e-4, methods[i] + " " + errors[i]);
        }
        // and with different grids: the expression is not tied to one of them
        PDEResult fine = equation.asPDE().param("alpha", 0.1).domain("x", 0, 1, 201)
                .initial("u", "sin(pi*x)").dirichlet("u", "0").solve(0.5, 1e-3, "crank_nicolson");
        assertEquals(201, fine.getGrid().n(0));
    }

    @Test
    void ellipticAndSystemExpressionsEndToEnd() {
        PDEResult poisson = new MathExpression("diff(u,x,2)+diff(u,y,2) = -2*pi^2*sin(pi*x)*sin(pi*y)").asPDE()
                .domain("x", 0, 1, 33).domain("y", 0, 1, 33).dirichlet("u", "0").solveSteady();
        assertTrue(poisson.maxError("u", (x, y, z, t) -> Math.sin(PI * x) * Math.sin(PI * y)) < 1e-3);

        PDEResult system = new MathExpression("pde(@(2)(\"diff(u,t) = 0.1*diff(u,x,2) - v\",\"diff(v,t) = 0.1*diff(v,x,2) + u\"))").asPDE()
                .domain("x", 0, 1, 81).initial("u", "sin(pi*x)").initial("v", "0").dirichlet("*", "0").solve(1.0, 1e-3, "rk4");
        double a = 0.1 * PI * PI;
        assertTrue(system.maxError("v", (x, y, z, t) -> Math.exp(-a * t) * Math.sin(t) * Math.sin(PI * x)) < 1e-4);
    }

    @Test
    void realTimeSolverFromAMathExpression() {
        try (com.github.gbenroscience.math.differentialcalculus.pde.PDESolver solver
                = new MathExpression("diff(u,t,2) = diff(u,x,2)").asPDE()
                        .domain("x", 0, 1, 51).initial("u", "sin(pi*x)").dirichlet("u", "0").solver()) {
            solver.initialize();
            double[] u = solver.state("u");
            for (int i = 0; i < 100; i++) {
                solver.step(1e-3);
            }
            assertEquals(Math.cos(PI * 0.1) * Math.sin(PI * 0.5), u[solver.grid().index(25, 0, 0)], 2e-3);
        }
    }
}
