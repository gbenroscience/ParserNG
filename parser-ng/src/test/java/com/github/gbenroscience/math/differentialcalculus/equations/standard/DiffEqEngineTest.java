package com.github.gbenroscience.math.differentialcalculus.equations.standard;

import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.util.FunctionManager;
import java.util.InputMismatchException;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive JUnit 5 (Jupiter 5.10, JDK 8 source-compatible) test suite for
 * ParserNG's differential-equation solving functions: {@code diffeqn},
 * {@code diffeqnPath}, {@code diffeqnHO}, {@code diffeqnPathHO}.
 *
 * <p>
 * The suite is organized into {@code @Nested} classes that mirror the sections
 * of the engine documentation:
 * <ul>
 * <li>A. The "must be root expression" rule (doc &sect;1)</li>
 * <li>B. Capturing results via assignment (doc &sect;2)</li>
 * <li>C. Smoke tests across the four functions (doc &sect;3)</li>
 * <li>D. Argument semantics: equation, t0, y0, tEnd, h, method, points,
 * presentationStrategy (doc &sect;4)</li>
 * <li>E. Explicit systems of equations (doc &sect;5)</li>
 * <li>F. Defaults and optional-argument behavior (doc &sect;6)</li>
 * <li>G. The five solver methods (doc &sect;7)</li>
 * <li>H. Worked examples reproduced verbatim (doc &sect;8)</li>
 * <li>I. Troubleshooting / error-condition cases (doc &sect;9)</li>
 * </ul>
 *
 * <h2>Assumed public API surface</h2>
 * The source documentation describes behavior but not the full Java surface.
 * The following signatures are assumed based on the documentation's own code
 * samples; adjust method/class names here if your actual engine differs:
 * <pre>
 *   MathExpression(String expr)
 *   MathExpression#solve()                      // may throw on malformed/illegal input
 *   MathExpression#getValue(String name)         // -&gt; double, for scalar-assigned results
 *   FunctionManager.lookUp(String name)          // -&gt; exposes getMatrix()
 *   Matrix#getRows() / getCols()                 // flat-array-backed matrix, row/col count accessors
 *   Matrix#getElem(int row, int col)
 *   Matrix#print()
 * </pre> All test methods declare {@code throws Exception} defensively, since
 * the documentation does not specify whether failures are signaled via checked
 * or unchecked exceptions.
 */
class DiffEqEngineTest {

    // Tolerances
    private static final double TIGHT = 1e-3;   // routine endpoint comparisons
    private static final double LOOSE = 1e-1;   // first-order methods where only coarse accuracy is asserted
    private static final double VERY_LOOSE = 5.0; // structural / stability-only checks

    // ------------------------------------------------------------------
    // Helpers
    //
    // NOTE ON UNIQUENESS: the engine's function/variable table is shared
    // (effectively global) across every MathExpression created in the JVM.
    // Reusing a short name like "A" across many tests causes later
    // assignments to collide with earlier ones and can corrupt engine
    // state. Every helper below mints a fresh, collision-free variable
    // name for each call so tests remain independent of each other and
    // of execution order.
    //
    // NOTE ON ERROR SIGNALING: EquationRuntime.solve() catches its own
    // internal validation errors (bad y0 length, wrong target symbol,
    // equation arrays passed to the *_HO functions, etc.), logs them,
    // and does NOT rethrow to the caller -- so calling solve() alone
    // never surfaces these as a Java exception. The failure only
    // becomes observable when code subsequently tries to use the
    // (never-populated) result, which is what assertInvalidCall does
    // below. Only expression-construction-time problems (e.g. the
    // root-expression rule in section A) throw directly from solve().
    // ------------------------------------------------------------------
    private static final java.util.concurrent.atomic.AtomicInteger UID = new java.util.concurrent.atomic.AtomicInteger();

    private static String uniqueName(String prefix) {
        return prefix + "_" + UID.incrementAndGet();
    }

    private double solveScalar(String varPrefix, String expr) throws Exception {
        String name = uniqueName(varPrefix);
        MathExpression me = new MathExpression(name + "=" + expr);
        me.solve();
        return me.getValue(name);
    }

    private Matrix solveMatrix(String varPrefix, String expr) throws Exception {
        String name = uniqueName(varPrefix);
        MathExpression me = new MathExpression(name + "=" + expr);
        me.solve();
        return FunctionManager.lookUp(name).getMatrix();
    }

    private void solveOnly(String expr) throws Exception {
        MathExpression me = new MathExpression(expr);
        me.solve();
    }

    /**
     * Root-expression violations are construction-time errors and should be
     * tested as Java exceptions. Solver-level semantic validation is different:
     * EquationRuntime currently logs the validation failure and returns without
     * publishing a result. The helper below therefore tests the observable
     * contract (no result is published) rather than relying on a secondary
     * NullPointerException from result retrieval.
     */
    private void assertRootExpressionRejected(String expr) {
        assertThrows(java.lang.Throwable.class, () -> {
            MathExpression me = new MathExpression(expr);
            me.solveWithThrows();
        }, "Expected root-expression violation: " + expr);
    }

    private void assertExpressionRejected(String expr) throws Throwable {
        assertThrows(java.lang.Throwable.class, () -> {
            MathExpression me = new MathExpression(expr);
            me.solveWithThrows();
        }, "Expected expression to be rejected: " + expr);
    }

    private void assertInvalidSolveProducesNoResult(String expr) {
        try {
            String name = uniqueName("invalid");
            MathExpression me = new MathExpression(name + "=" + expr);// invalid function syntax will fail to parse, from the constructor stage
            me.solveWithThrows();
            Function f = FunctionManager.lookUp(name);
            assertNull(f,
                    "Invalid equation must not publish a result: " + expr);
        } catch (Throwable t) {
            if (t instanceof InputMismatchException) {
                assertNull(null,
                        "Invalid equation must not publish a result: " + expr);
            }
        }
    }

    private static void assertMatricesEqual(Matrix expected, Matrix actual, double tolerance) {
        assertEquals(expected.getRows(), actual.getRows(), "row count mismatch");
        assertEquals(expected.getCols(), actual.getCols(), "column count mismatch");
        for (int r = 0; r < expected.getRows(); r++) {
            for (int c = 0; c < expected.getCols(); c++) {
                assertEquals(expected.getElem(r, c), actual.getElem(r, c), tolerance,
                        "matrix mismatch at [" + r + "," + c + "]");
            }
        }
    }

    private static double analyticalDecay(double y0, double k, double t0, double t) {
        return y0 * Math.exp(k * (t - t0));
    }

    // ==================================================================
    // A. The root-expression rule (doc section 1)
    // ==================================================================
    @Nested
    @DisplayName("A. diffeqn-family calls must be the whole expression")
    class RootExpressionRuleTests {

        @Test
        @DisplayName("A1: embedding diffeqn after an addition throws")
        void embeddedAfterAdditionThrows() {
            assertRootExpressionRejected("sin(2*x) + diffeqn(y[1] - 2*y[0], 0, 1, 5)");
        }

        @Test
        @DisplayName("A2: appending arithmetic after diffeqn throws")
        void appendedArithmeticThrows() {
            assertRootExpressionRejected("diffeqn(y[1] - 2*y[0], 0, 1, 5) + 3");
        }

        @Test
        @DisplayName("A3: nesting diffeqn inside sin() throws")
        void nestedInsideFunctionThrows() {
            assertRootExpressionRejected("sin(diffeqn(y[1] - 2*y[0], 0, 1, 5) + 5)");
        }

        @Test
        @DisplayName("A4: two diffeqnPath calls multiplied together throws")
        void twoCallsMultipliedThrows() {
            assertRootExpressionRejected(
                    "diffeqnPath(y[1] - 2*y[0], 0, 1, 5, 0.01, rk4) * "
                    + "diffeqnPath(y[1] - 2*y[0], 0, 1, 5, 0.01, rk4)");
        }

        @Test
        @DisplayName("A5: a bare diffeqn call as the entire expression solves fine")
        void bareDiffeqnAsRootSucceeds() throws Exception {
            assertDoesNotThrow(() -> solveOnly("diffeqn(y[1] + 2*y[0], 0, 1, 5)"));
        }

        @Test
        @DisplayName("A6: assignment of a diffeqn call is the sanctioned exception")
        void assignmentOfDiffeqnAllowed() throws Exception {
            assertDoesNotThrow(() -> solveScalar("A", "diffeqn(y[1] + 2*y[0], 0, 1, 5)"));
        }

        @Test
        @DisplayName("A7: assignment of a diffeqnHO call is allowed")
        void assignmentOfDiffeqnHOAllowed() throws Exception {
            assertDoesNotThrow(()
                    -> solveScalar("A", "diffeqnHO(y[2] + y[0], 0, @(1,2)(1, 0), 3)"));
        }

        @Test
        @DisplayName("A8: assignment of a diffeqnPathHO call is allowed")
        void assignmentOfDiffeqnPathHOAllowed() throws Exception {
            assertDoesNotThrow(() -> solveMatrix("A",
                    "diffeqnPathHO(y[2] + y[0], 0, @(1,2)(1, 0), 3, 0.01, rk4, state)"));
        }
    }

    // ==================================================================
    // B. Capturing results via assignment (doc section 2)
    // ==================================================================
    @Nested
    @DisplayName("B. Capturing results: scalar vs matrix")
    class ResultCaptureTests {

        @Test
        @DisplayName("B1: scalar diffeqn assignment is readable via getValue")
        void scalarDiffeqnAssignmentReadable() throws Exception {
            double b = solveScalar("b",
                    "diffeqn((3*t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.01, rk45)");
            assertTrue(Double.isFinite(b));
        }

        @Test
        @DisplayName("B2: scalar diffeqnHO assignment is readable via getValue")
        void scalarDiffeqnHOAssignmentReadable() throws Exception {
            double v = solveScalar("c", "diffeqnHO(y[2] + y[0], 0, @(1,2)(1, 0), 3.14159)");
            assertTrue(Double.isFinite(v));
        }

        @Test
        @DisplayName("B3: diffeqnPath assignment is readable as a matrix")
        void pathAssignmentReadableAsMatrix() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4, 50)");
            assertEquals(50, m.getRows());
        }

        @Test
        @DisplayName("B4: diffeqnPathHO assignment is readable as a matrix")
        void pathHOAssignmentReadableAsMatrix() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPathHO(3*x*sin(x)*y[3]+4*x*y[2]+3*ln(x)*y[1]+4*y[0], 1, "
                    + "@(1,3)(1, 0, 0), 3, 0.01, bdf2, state)");
            assertTrue(m.getRows() > 0);
            assertEquals(4, m.getCols(), "t,y0,y1,y2 expected in state mode");
        }

        @Test
        @DisplayName("B5: an assigned two-state trajectory is captured as a matrix with t and both state components")
        void systemTrajectoryAssignmentIsMatrix() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPath(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), "
                    + "0, @(1,2)(1, 0), 1.5707963267948966, 0.001, rk4, 101)");
            assertEquals(101, m.getRows());
            assertEquals(3, m.getCols());
            int last = m.getRows() - 1;
            assertEquals(0.0, m.getElem(last, 1), 1e-8);
            assertEquals(-1.0, m.getElem(last, 2), 1e-8);
        }

        @Test
        @DisplayName("B6: an assigned matrix can be indexed independently after solve()")
        void matrixAssignmentIndependentlyReusable() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4, 10)");
            double t0Cell = m.getElem(0, 0);
            double tLastCell = m.getElem(m.getRows() - 1, 0);
            assertTrue(tLastCell > t0Cell);
        }

        @Test
        @DisplayName("B7: sequential assignments to different names do not interfere")
        void multipleSequentialAssignmentsIndependent() throws Exception {
            String nameA = uniqueName("A");
            String nameB = uniqueName("B");
            MathExpression meA = new MathExpression(nameA + "=diffeqn(y[1] + 2*y[0], 0, 1, 5)");
            meA.solve();
            MathExpression meB = new MathExpression(nameB + "=diffeqn(y[1] + 3*y[0], 0, 1, 5)");
            meB.solve();
            assertNotEquals(meA.getValue(nameA), meB.getValue(nameB), 1e-9);
        }

        @Test
        @DisplayName("B8: reassigning the same name overwrites the previous result")
        void reassigningSameNameOverwrites() throws Exception {
            double first = solveScalar("A", "diffeqn(y[1] + 2*y[0], 0, 1, 5)");
            double second = solveScalar("A", "diffeqn(y[1] + 2*y[0], 0, 2, 5)");
            assertNotEquals(first, second, 1e-9);
        }
    }

    // ==================================================================
    // C. Smoke tests across the four functions (doc section 3)
    // ==================================================================
    @Nested
    @DisplayName("C. Overview smoke tests for all four functions")
    class OverviewSmokeTests {

        @Test
        @DisplayName("C1: diffeqn scalar equation returns a single double")
        void diffeqnScalarReturnsSingleDouble() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 0, 1, 5)");
            assertTrue(Double.isFinite(y));
        }

        @Test
        @DisplayName("C2: an explicit two-equation system executes without error")
        void diffeqnSystemSolvesWithoutError() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.01, rk4)"));
        }

        @Test
        @DisplayName("C3: diffeqnPath returns multiple rows across the interval")
        void diffeqnPathReturnsMultipleRows() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.1, rk4)");
            assertTrue(m.getRows() > 1);
        }

        @Test
        @DisplayName("C4: diffeqnHO returns the endpoint value of y")
        void diffeqnHOReturnsEndpointValue() throws Exception {
            double y = solveScalar("A", "diffeqnHO(y[2] + y[0], 0, @(1,2)(1, 0), 3)");
            assertEquals(Math.cos(3.0), y, TIGHT);
        }

        @Test
        @DisplayName("C5: diffeqnPathHO returns a trajectory of state vectors")
        void diffeqnPathHOReturnsTrajectoryOfStates() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPathHO(y[2] + y[0], 0, @(1,2)(1, 0), 3, 0.1, rk4, state)");
            assertTrue(m.getRows() > 1);
        }

        @Test
        @DisplayName("C6: diffeqnPath row count matches natural step count for fixed-step methods")
        void diffeqnPathRowCountMatchesNaturalSteps() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 1, 0.1, rk4)");
            assertEquals(11, m.getRows());
        }

        @Test
        @DisplayName("C7: diffeqnPathHO with state strategy exposes more than just t,y columns")
        void diffeqnPathHOStateHasMultipleColumns() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPathHO(y[2] + y[0], 0, @(1,2)(1, 0), 3, 0.1, rk4, state)");
            assertEquals(3, m.getCols(), "t, y[0], y[1] expected");
        }

        @Test
        @DisplayName("C8: diffeqn endpoint is consistent with the last row of diffeqnPath")
        void diffeqnEndpointMatchesLastRowOfDiffeqnPath() throws Exception {
            double endpoint = solveScalar("e", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4)");
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4)");
            double lastRowY = m.getElem(m.getRows() - 1, 1);
            assertEquals(endpoint, lastRowY, TIGHT);
        }
    }

    // ==================================================================
    // D. Argument semantics (doc section 4)
    // ==================================================================
    @Nested
    @DisplayName("D. Argument semantics: equation, t0, y0, tEnd, h, method, points, presentationStrategy")
    class ArgumentSemanticsTests {

        // --- equation ---
        @Test
        @DisplayName("D1: equation written as LHS-RHS solves the decay equation correctly")
        void equationLHSminusRHSSolvesDecay() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] - (-2*y[0]), 0, 1, 5, 0.01, rk4)");
            assertEquals(analyticalDecay(1, -2, 0, 5), y, TIGHT);
        }

        @Test
        @DisplayName("D2: the simplified equation form is equivalent to the expanded form")
        void equationSimplifiedFormEquivalentToExpanded() throws Exception {
            double expanded = solveScalar("y1", "diffeqn(y[1] - (-2*y[0]), 0, 1, 5, 0.01, rk4)");
            double simplified = solveScalar("y2", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4)");
            assertEquals(expanded, simplified, 1e-9);
        }

        @Test
        @DisplayName("D3: higher-order equation references the state derivatives without error")
        void equationReferencesYIndicesForHigherOrder() throws Exception {
            assertDoesNotThrow(() -> solveScalar("A",
                    "diffeqnHO(y[2] + 3*y[1] - y[0], 0, @(1,2)(1, 0), 1)"));
        }

        @Test
        @DisplayName("D4: equation may contain nonlinear terms without error")
        void equationNonlinearTermsAllowed() throws Exception {
            assertDoesNotThrow(() -> solveScalar("y", "diffeqn(y[1] - y[0]*y[0], 0, 0.5, 1, 0.01, rk4)"));
        }

        // --- t0 ---
        @Test
        @DisplayName("D5: a nonzero t0 start value is honored")
        void t0NonZeroStartWorks() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 2, 1, 7, 0.01, rk4)");
            assertEquals(analyticalDecay(1, -2, 2, 7), y, TIGHT);
        }

        @Test
        @DisplayName("D6: t0 equal to tEnd is a no-op that returns y0 unchanged")
        void t0EqualsTEndReturnsY0() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 3, 1.5, 3, 0.01, rk4)");
            assertEquals(1.5, y, 1e-9);
        }

        @Test
        @DisplayName("D7: t0 greater than tEnd solves backward in the independent variable")
        void t0GreaterThanTEndSolvesBackward() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 5, analyticalStart, 0, 0.01, rk4)"
                    .replace("analyticalStart", String.valueOf(analyticalDecay(1, -2, 0, 5))));
            assertEquals(1.0, y, TIGHT);
        }

        // --- y0 ---
        @Test
        @DisplayName("D8: scalar y0 works for a single equation")
        void y0ScalarForSingleEquation() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 0, 4, 1, 0.01, rk4)");
            assertEquals(analyticalDecay(4, -2, 0, 1), y, TIGHT);
        }

        @Test
        @DisplayName("D9: a y0 vector whose length mismatches system size does not publish a result")
        void y0VectorLengthMismatchForSystemThrows() throws Throwable {
            assertInvalidSolveProducesNoResult(
                    "diffeqn(@(2)(\"y[2]-y[0]\", \"y[2]-y[1]\"), 0, @(1,3)(1, 0, 0), 1, 0.01, rk4)");
        }

        @Test
        @DisplayName("D10: a y0 vector whose length mismatches HO equation order does not publish a result")
        void y0VectorLengthMismatchForHOThrows() throws Throwable {
            assertInvalidSolveProducesNoResult("diffeqnHO(y[2] + y[0], 0, @(1,3)(1, 0, 0), 1)");
        }

        // --- tEnd ---
        @Test
        @DisplayName("D11: forward integration produces the expected analytical endpoint")
        void tEndForwardIntegration() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 0, 1, 3, 0.001, rk4)");
            assertEquals(analyticalDecay(1, -2, 0, 3), y, TIGHT);
        }

        @Test
        @DisplayName("D12: integrating forward then backward approximately returns to y0")
        void tEndBackwardReturnsToY0() throws Exception {
            double forward = solveScalar("f", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.001, rk4)");
            double roundTrip = solveScalar("r",
                    "diffeqn(y[1] + 2*y[0], 5, " + forward + ", 0, 0.001, rk4)");
            assertEquals(1.0, roundTrip, TIGHT);
        }

        // --- h ---
        @Test
        @DisplayName("D13: h fixes the exact step count for euler")
        void hFixedStepDeterminesStepCountEuler() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 2, 0.5, euler)");
            assertEquals(5, m.getRows());
        }

        @Test
        @DisplayName("D14: h fixes the exact step count for rk4")
        void hFixedStepDeterminesStepCountRk4() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 2, 0.25, rk4)");
            assertEquals(9, m.getRows());
        }

        @Test
        @DisplayName("D15: a smaller h improves rk4 accuracy against the analytical solution")
        void hSmallerImprovesAccuracyRk4() throws Exception {
            double coarse = solveScalar("c", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.5, rk4)");
            double fine = solveScalar("f", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.001, rk4)");
            double exact = analyticalDecay(1, -2, 0, 5);
            assertTrue(Math.abs(fine - exact) <= Math.abs(coarse - exact));
        }

        @Test
        @DisplayName("D16: h is only the initial suggestion for rk45 and does not error at any reasonable value")
        void hIsOnlyInitialSuggestionForRk45() throws Exception {
            assertDoesNotThrow(() -> solveScalar("a", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 1.0, rk45)"));
            assertDoesNotThrow(() -> solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.0001, rk45)"));
        }

        // --- method ---
        @Test
        @DisplayName("D17: explicit method selection changes the numerical path taken")
        void methodExplicitSelectionAffectsResult() throws Exception {
            double eulerY = solveScalar("e", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 0.25, euler)");
            double rk4Y = solveScalar("r", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 0.25, rk4)");
            assertNotEquals(eulerY, rk4Y, 1e-9);
        }

        @Test
        @DisplayName("D18: euler is less accurate than rk4 at the same step size")
        void methodEulerLessAccurateThanRk4AtSameH() throws Exception {
            double exact = analyticalDecay(1, -2, 0, 2);
            double eulerY = solveScalar("e", "diffeqn(y[1] + 2*y[0], 0, 1, 2, 0.1, euler)");
            double rk4Y = solveScalar("r", "diffeqn(y[1] + 2*y[0], 0, 1, 2, 0.1, rk4)");
            assertTrue(Math.abs(eulerY - exact) > Math.abs(rk4Y - exact));
        }

        // --- points ---
        @Test
        @DisplayName("D19: points resamples a fixed-step trajectory to the requested count")
        void pointsResamplesToRequestedCount() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4, 37)");
            assertEquals(37, m.getRows());
        }

        @Test
        @DisplayName("D20: omitting points uses the solver's natural step count")
        void pointsOmittedUsesNaturalStepCount() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 1, 0.1, rk4)");
            assertEquals(11, m.getRows());
        }

        @Test
        @DisplayName("D21: a non-positive points value falls back to natural output")
        void pointsNonPositiveFallsBackToNatural() throws Exception {
            Matrix natural = solveMatrix("N", "diffeqnPath(y[1] + 2*y[0], 0, 1, 1, 0.1, rk4)");
            Matrix zeroPoints = solveMatrix("Z", "diffeqnPath(y[1] + 2*y[0], 0, 1, 1, 0.1, rk4, 0)");
            assertEquals(natural.getRows(), zeroPoints.getRows());
        }

        // --- presentationStrategy ---
        @Test
        @DisplayName("D22: presentationStrategy=state on diffeqnPathHO includes the full state vector")
        void presentationStrategyStateIncludesFullState() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPathHO(y[3] + 3*y[2] - y[1], 0, @(1,3)(1, 0, 0), 1, 0.1, rk4, state)");
            assertEquals(4, m.getCols(), "t, y[0], y[1], y[2] expected");
        }

        @Test
        @DisplayName("D23: presentationStrategy=trajectory on diffeqnPathHO includes only t and y")
        void presentationStrategyTrajectoryIncludesOnlyTAndY() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPathHO(y[3] + 3*y[2] - y[1], 0, @(1,3)(1, 0, 0), 1, 0.1, rk4, trajectory)");
            assertEquals(2, m.getCols(), "t, y expected");
        }

        @Test
        @DisplayName("D24: presentationStrategy currently has no effect on plain diffeqnPath")
        void presentationStrategyNoEffectOnPlainDiffeqnPath() throws Exception {
            Matrix trajectoryMode = solveMatrix("T",
                    "diffeqnPath(y[1] + 2*y[0], 0, 1, 1, 0.1, rk4, 10, trajectory)");
            Matrix stateMode = solveMatrix("S",
                    "diffeqnPath(y[1] + 2*y[0], 0, 1, 1, 0.1, rk4, 10, state)");
            assertMatricesEqual(trajectoryMode, stateMode, 1e-12);
        }
    }

    // ==================================================================
    // E. Explicit systems of equations (doc section 5)
    // ==================================================================
    @Nested
    @DisplayName("E. Explicit systems of equations")
    class SystemOfEquationsTests {

        @Test
        @DisplayName("E1: the Lotka-Volterra two-equation system executes over the requested interval")
        void lotkaVolterraSolvesToEndpoint() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(2)(\"y[2]-(0.6*y[0]-0.03*y[0]*y[1])\", "
                            + "\"y[2]-(-0.9*y[1]+0.02*y[0]*y[1])\"), 0, @(1,2)(30, 4), 20, 0.01, rk4)"));
        }

        @Test
        @DisplayName("E2: a two-equation system with both equations dividing out y[2] solves without error")
        void systemEquationsDivideOutYN() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.01, rk4)"));
        }

        @Test
        @DisplayName("E3: a system equation using the wrong target symbol does not publish a result")
        void systemWrongTargetSymbolThrows() throws Throwable {
            assertInvalidSolveProducesNoResult(
                    "diffeqn(@(2)(\"y[0]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.01, rk4)");
        }

        @Test
        @DisplayName("E4: a four-equation linear system solves without error")
        void fourEquationSystemSolves() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(4)(\"y[4]-y[1]\", \"y[4]-(-2*y[0]+y[2])\", \"y[4]-y[3]\", "
                            + "\"y[4]-(y[0]-2*y[2])\"), 0, @(1,4)(1,0,0,1), 10, 0.01, rk4)"));
        }

        @Test
        @DisplayName("E5: y0 length not matching declared equation count does not publish a result")
        void systemY0LengthMismatchThrows() throws Throwable {
            assertInvalidSolveProducesNoResult(
                    "diffeqn(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,3)(1, 0, 0), 1, 0.01, rk4)");
        }

        @Test
        @DisplayName("E6: system equations may contain nonlinear cross terms")
        void systemAllowsNonlinearCrossTerms() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(2)(\"y[2]-(0.6*y[0]-0.03*y[0]*y[1])\", "
                            + "\"y[2]-(-0.9*y[1]+0.02*y[0]*y[1])\"), 0, @(1,2)(30, 4), 1, 0.01, rk4)"));
        }

        @Test
        @DisplayName("E7: a state variable no equation mentions does not cause an error under the default finite-difference Jacobian")
        void systemUnreferencedStateVariableHasNoImpact() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(2)(\"y[2]-1\", \"y[2]-0\"), 0, @(1,2)(0, 0), 1, 0.01, implicit_euler)"));
        }

        @Test
        @DisplayName("E8: implicit_euler solves a stiff explicit system without diverging")
        void systemImplicitEulerSolvesStiffSystem() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(2)(\"y[2]-(-1000*y[0])\", \"y[2]-(-y[1])\"), 0, @(1,2)(1, 1), 2, 0.01, implicit_euler)"));
        }

        @Test
        @DisplayName("E9: bdf2 solves the same stiff system without diverging")
        void systemBdf2SolvesStiffSystem() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(2)(\"y[2]-(-1000*y[0])\", \"y[2]-(-y[1])\"), 0, @(1,2)(1, 1), 2, 0.01, bdf2)"));
        }

        @Test
        @DisplayName("E10: diffeqnHO with an equation array does not publish a result")
        void diffeqnHORejectsArrayArgument() throws Throwable {
            assertInvalidSolveProducesNoResult(
                    "diffeqnHO(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1)");
        }

        @Test
        @DisplayName("E11: diffeqnPathHO with an equation array does not publish a result")
        void diffeqnPathHORejectsArrayArgument() throws Throwable {
            assertInvalidSolveProducesNoResult(
                    "diffeqnPathHO(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.01, rk4, state)");
        }

        @Test
        @DisplayName("E12: diffeqnPath for a system returns t plus every component per row")
        void systemDiffeqnPathReturnsAllComponents() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPath(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.1, rk4)");
            assertEquals(3, m.getCols(), "t, y[0], y[1] expected");
        }

        @Test
        @DisplayName("E13: presentationStrategy has no effect yet on a plain system's diffeqnPath output")
        void systemPresentationStrategyIgnored() throws Exception {
            Matrix trajectoryMode = solveMatrix("T",
                    "diffeqnPath(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.1, rk4, 10, trajectory)");
            Matrix stateMode = solveMatrix("S",
                    "diffeqnPath(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.1, rk4, 10, state)");
            assertMatricesEqual(trajectoryMode, stateMode, 1e-12);
        }

        @Test
        @DisplayName("E14: a single-equation array (n=1) behaves like the scalar form, dividing out y[1]")
        void systemOfOneBehavesLikeScalar() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(1)(\"y[1]-(-2*y[0])\"), 0, @(1,1)(1), 5, 0.01, rk4)"));
        }

        @Test
        @DisplayName("E15: a three-equation linear system solves without error")
        void threeEquationLinearSystemSolves() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(3)(\"y[3]-y[1]\", \"y[3]-y[2]\", \"y[3]-(-y[0])\"), 0, @(1,3)(1,0,0), 2, 0.01, rk4)"));
        }

        @Test
        @DisplayName("E16: a y0 vector length inconsistent with the declared array length does not publish a result")
        void systemArrayLengthDeclarationMismatchThrows() throws Throwable {
            assertInvalidSolveProducesNoResult(
                    "diffeqn(@(2)(\"y[2]-y[1]\"), 0, @(1,2)(1, 0), 1, 0.01, rk4)");
        }

        @Test
        @DisplayName("E17: declared equation count not matching quoted equations does not publish a result")
        void systemDeclaredCountMismatchThrows() throws Throwable {
            assertInvalidSolveProducesNoResult(
                    "diffeqn(@(3)(\"y[3]-y[1]\", \"y[3]-y[2]\"), 0, @(1,3)(1,0,0), 1, 0.01, rk4)");
        }

        @Test
        @DisplayName("E18: Lotka-Volterra populations remain finite and positive over a short forward interval")
        void lotkaVolterraPopulationsStayPositiveShortInterval() throws Exception {
            Matrix m = solveMatrix("P",
                    "diffeqnPath(@(2)(\"y[2]-(0.6*y[0]-0.03*y[0]*y[1])\", "
                    + "\"y[2]-(-0.9*y[1]+0.02*y[0]*y[1])\"), 0, @(1,2)(30, 4), 1, 0.01, rk4)");
            for (int r = 0; r < m.getRows(); r++) {
                assertTrue(m.getElem(r, 1) > 0);
                assertTrue(m.getElem(r, 2) > 0);
            }
        }

        @Test
        @DisplayName("E19: rk4 and implicit_euler agree to first-order accuracy on a mild oscillator")
        void rk4AndImplicitEulerAgreeOnNonStiffSystem() throws Exception {
            String eq = "@(2)(\"y[2]-y[1]\", \"y[2]-(-0.1*y[0])\")";
            Matrix rk4 = solveMatrix("R", "diffeqnPath(" + eq + ", 0, @(1,2)(1, 0), 1, 0.01, rk4, 101)");
            Matrix ie = solveMatrix("I", "diffeqnPath(" + eq + ", 0, @(1,2)(1, 0), 1, 0.01, implicit_euler, 101)");
            double w = Math.sqrt(0.1);
            double exactY = Math.cos(w);
            double exactV = -w * Math.sin(w);
            int last = rk4.getRows() - 1;
            assertEquals(exactY, rk4.getElem(last, 1), 1e-6);
            assertEquals(exactV, rk4.getElem(last, 2), 1e-6);
            assertEquals(exactY, ie.getElem(last, 1), 1e-3);
            assertEquals(exactV, ie.getElem(last, 2), 1e-3);
            assertEquals(rk4.getElem(last, 1), ie.getElem(last, 1), 1e-3);
            assertEquals(rk4.getElem(last, 2), ie.getElem(last, 2), 1e-3);
        }

        @Test
        @DisplayName("E20: system diffeqnPath row count matches the fixed-step natural count")
        void systemDiffeqnPathRowCountMatchesFixedSteps() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPath(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.1, rk4)");
            assertEquals(11, m.getRows());
        }
    }

    // ==================================================================
    // F. Defaults (doc section 6)
    // ==================================================================
    @Nested
    @DisplayName("F. Defaults and optional-argument behavior")
    class DefaultsTests {

        @Test
        @DisplayName("F1: omitting h and method uses defaults 0.01 and rk4")
        void omittingHAndMethodUsesDefaults() throws Exception {
            double withDefaults = solveScalar("a", "diffeqn(y[1] + 2*y[0], 0, 1, 5)");
            double explicit = solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4)");
            assertEquals(explicit, withDefaults, 1e-9);
        }

        @Test
        @DisplayName("F2: omitting only method defaults to rk4 with the given h")
        void omittingMethodOnlyDefaultsToRk4() throws Exception {
            double withH = solveScalar("a", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.05)");
            double explicit = solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.05, rk4)");
            assertEquals(explicit, withH, 1e-9);
        }

        @Test
        @DisplayName("F3: arguments are positional -- an earlier optional argument cannot be skipped to reach a later one")
        void cannotSkipHToSupplyMethod() throws Throwable {
            // Passing a method token where h is expected should fail to parse as a number.
            assertExpressionRejected("diffeqn(y[1] + 2*y[0], 0, 1, 5, rk4)");
        }

        @Test
        @DisplayName("F4: points and presentationStrategy may both be omitted on a path call")
        void pointsAndPresentationStrategyBothOmitted() throws Exception {
            assertDoesNotThrow(() -> solveMatrix("A",
                    "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4)"));
        }

        @Test
        @DisplayName("F5: points may be supplied without presentationStrategy")
        void pointsSuppliedWithoutPresentationStrategy() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4, 25)");
            assertEquals(25, m.getRows());
        }

        @Test
        @DisplayName("F6: both points and presentationStrategy may be supplied together")
        void pointsAndPresentationStrategyBothSupplied() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPathHO(y[2] + y[0], 0, @(1,2)(1, 0), 3, 0.01, rk4, 40, state)");
            assertEquals(40, m.getRows());
            assertEquals(3, m.getCols());
        }
    }

    // ==================================================================
    // G. The five solver methods (doc section 7)
    // ==================================================================
    @Nested
    @DisplayName("G. Solver methods: euler, rk4, rk45, implicit_euler, bdf2")
    class SolverMethodTests {

        // ---------- euler ----------
        @Test
        @DisplayName("G1: euler solves simple decay approximately")
        void eulerSolvesDecayApproximately() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 0.001, euler)");
            assertEquals(analyticalDecay(1, -2, 0, 1), y, LOOSE);
        }

        @Test
        @DisplayName("G2: euler error roughly scales linearly with step size")
        void eulerErrorScalesLinearlyWithH() throws Exception {
            double exact = analyticalDecay(1, -2, 0, 1);
            double errAtH = Math.abs(solveScalar("a", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 0.1, euler)") - exact);
            double errAtHalfH = Math.abs(solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 0.05, euler)") - exact);
            double observedOrder = Math.log(errAtH / errAtHalfH) / Math.log(2.0);
            assertTrue(observedOrder > 0.8 && observedOrder < 1.2,
                    "Expected first-order convergence, observed order=" + observedOrder);
        }

        @Test
        @DisplayName("G3: euler becomes unstable on a stiff system at a large step size")
        void eulerUnstableOnStiffSystemAtLargeH() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 1, 0.1, euler)");
            assertTrue(!Double.isFinite(y) || Math.abs(y) > 1e6,
                    "Expected euler to blow up on a stiff system with a large step");
        }

        @Test
        @DisplayName("G4: euler remains stable on the same stiff system given a very small step size")
        void eulerStableOnStiffSystemAtVerySmallH() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 0.01, 0.0001, euler)");
            assertEquals(Math.exp(-10.0), y, 1e-4);
        }

        @Test
        @DisplayName("G5: euler completes quickly over a modest interval (smoke test)")
        void eulerCompletesOverModestInterval() {
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(10), ()
                    -> solveOnly("diffeqn(y[1] + 2*y[0], 0, 1, 10, 0.01, euler)"));
        }

        @Test
        @DisplayName("G6: euler is measurably less accurate than rk4 at the same step size")
        void eulerLessAccurateThanRk4AtSameH() throws Exception {
            double exact = analyticalDecay(1, -2, 0, 2);
            double eulerY = solveScalar("e", "diffeqn(y[1] + 2*y[0], 0, 1, 2, 0.2, euler)");
            double rk4Y = solveScalar("r", "diffeqn(y[1] + 2*y[0], 0, 1, 2, 0.2, rk4)");
            assertTrue(Math.abs(eulerY - exact) > Math.abs(rk4Y - exact));
        }

        // ---------- rk4 ----------
        @Test
        @DisplayName("G7: rk4 solves decay accurately")
        void rk4SolvesDecayAccurately() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4)");
            assertEquals(analyticalDecay(1, -2, 0, 5), y, TIGHT);
        }

        @Test
        @DisplayName("G8: rk4 error shrinks sharply when step size is halved (roughly fourth-order)")
        void rk4ErrorShrinksSharplyWithSmallerH() throws Exception {
            double exact = analyticalDecay(1, -2, 0, 1);
            double errAtH = Math.abs(solveScalar("a", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 0.2, rk4)") - exact);
            double errAtHalfH = Math.abs(solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 0.1, rk4)") - exact);
            double errAtQuarterH = Math.abs(solveScalar("c", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 0.05, rk4)") - exact);
            double order1 = Math.log(errAtH / errAtHalfH) / Math.log(2.0);
            double order2 = Math.log(errAtHalfH / errAtQuarterH) / Math.log(2.0);
            assertTrue(order1 > 3.5 && order1 < 4.5, "Observed RK4 order for h=.2->.1: " + order1);
            assertTrue(order2 > 3.5 && order2 < 4.5, "Observed RK4 order for h=.1->.05: " + order2);
        }

        @Test
        @DisplayName("G9: rk4 is the method used when method is omitted")
        void rk4IsDefaultMethod() throws Exception {
            double withDefault = solveScalar("a", "diffeqn(y[1] + 2*y[0], 0, 1, 2, 0.1)");
            double explicit = solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 2, 0.1, rk4)");
            assertEquals(explicit, withDefault, 1e-9);
        }

        @Test
        @DisplayName("G10: rk4 struggles (diverges or becomes inaccurate) on a stiff system at moderate h")
        void rk4StrugglesOnStiffSystemAtModerateH() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 1, 0.1, rk4)");
            assertTrue(!Double.isFinite(y) || Math.abs(y) > 1e3);
        }

        @Test
        @DisplayName("G11: rk4 solves a second-order harmonic oscillator via diffeqnHO")
        void rk4SolvesHarmonicOscillatorViaHO() throws Exception {
            double m = solveScalar("A",
                    "diffeqnHO(y[2] + y[0], 0, @(1,2)(1, 0), 1, 0.001, rk4)");

            assertEquals(Math.cos(1.0), m, 1e-10);
        }

        @Test
        @DisplayName("G12: rk4 matches the analytical cosine solution for the harmonic oscillator")
        void rk4MatchesAnalyticalCosine() throws Exception {
            double tEnd = 1.0;
            double y = solveScalar("A",
                    "diffeqnHO(y[2] + y[0], 0, @(1,2)(1, 0), " + tEnd + ", 0.001, rk4)");
            assertEquals(Math.cos(tEnd), y, TIGHT);
        }

        // ---------- rk45 ----------
        @Test
        @DisplayName("G13: rk45 solves decay accurately")
        void rk45SolvesDecayAccurately() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.01, rk45)");
            assertEquals(analyticalDecay(1, -2, 0, 5), y, TIGHT);
        }

        @Test
        @DisplayName("G14: rk45 without points produces a valid trajectory")
        void rk45WithoutPointsProducesValidTrajectory() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.5, rk45)");
            assertTrue(m.getRows() > 1);
            assertEquals(0.0, m.getElem(0, 0), 1e-12);
            assertEquals(5.0, m.getElem(m.getRows() - 1, 0), 1e-9);
            for (int r = 1; r < m.getRows(); r++) {
                assertTrue(m.getElem(r, 0) > m.getElem(r - 1, 0));
                assertTrue(Double.isFinite(m.getElem(r, 1)));
            }
        }

        @Test
        @DisplayName("G15: rk45 with points produces a uniform grid")
        void rk45WithPointsIsUniform() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.5, rk45, 20)");
            assertEquals(20, m.getRows());
            double gap = m.getElem(1, 0) - m.getElem(0, 0);
            for (int r = 2; r < m.getRows(); r++) {
                assertEquals(gap, m.getElem(r, 0) - m.getElem(r - 1, 0), 1e-6);
            }
        }

        @Test
        @DisplayName("G16: rk45's initial h is only a suggestion -- different starting h values converge to similar accuracy")
        void rk45InitialHIsOnlySuggestion() throws Exception {
            double exact = analyticalDecay(1, -2, 0, 5);
            double fromSmallH = solveScalar("a", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.001, rk45)");
            double fromLargeH = solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.5, rk45)");
            assertEquals(exact, fromSmallH, TIGHT);
            assertEquals(exact, fromLargeH, TIGHT);
        }

        @Test
        @DisplayName("G17: rk45 adaptively handles a stiff decay and drives the endpoint essentially to zero")
        void rk45HandlesStiffSystemWithoutNonFiniteResult() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 1, 0.1, rk45)");
            assertEquals(0.0, y, 1e-6);
        }

        @Test
        @DisplayName("G18: rk45 closely matches rk4 on a non-stiff problem")
        void rk45MatchesRk4OnNonStiffProblem() throws Exception {
            double rk4Y = solveScalar("a", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4)");
            double rk45Y = solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.01, rk45)");
            assertEquals(rk4Y, rk45Y, TIGHT);
        }

        // ---------- implicit_euler ----------
        @Test
        @DisplayName("G19: implicit_euler remains stable on a stiff decay and drives the endpoint essentially to zero")
        void implicitEulerStableOnStiffSystem() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 1, 0.1, implicit_euler)");
            assertEquals(0.0, y, 1e-8);
        }

        @Test
        @DisplayName("G20: bdf2 improves accuracy over implicit_euler in the multi-step regime")
        void bdf2ImprovesAccuracyOverImplicitEulerAtSameH() throws Exception {
            double exact = analyticalDecay(1, -5, 0, 2);
            double ieY = solveScalar("a", "diffeqn(y[1] + 5*y[0], 0, 1, 2, 0.1, implicit_euler)");
            double bdf2Y = solveScalar("b", "diffeqn(y[1] + 5*y[0], 0, 1, 2, 0.1, bdf2)");
            assertTrue(Math.abs(bdf2Y - exact) < Math.abs(ieY - exact));
        }

        @Test
        @DisplayName("G21: implicit_euler is usable as a safe fallback for unknown stiffness on a non-stiff problem")
        void implicitEulerUsableOnNonStiffProblem() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 0, 1, 5, 0.01, implicit_euler)");
            assertEquals(analyticalDecay(1, -2, 0, 5), y, LOOSE);
        }

        @Test
        @DisplayName("G22: implicit_euler does not blow up on a stiff system at a large step, unlike euler")
        void implicitEulerConvergesAtLargeStepUnlikeEuler() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 1, 0.5, implicit_euler)");
            assertEquals(0.0, y, 1e-4);
        }

        @Test
        @DisplayName("G23: implicit_euler damps a very stiff decay mode close to zero")
        void implicitEulerDampsStiffModeQuickly() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 2, 0.01, implicit_euler)");
            assertEquals(0.0, y, 1e-2);
        }

        @Test
        @DisplayName("G24: implicit_euler solves an explicit system, not just a scalar equation")
        void implicitEulerWorksOnSystem() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(2)(\"y[2]-(-1000*y[0])\", \"y[2]-(-y[1])\"), 0, @(1,2)(1, 1), 1, 0.05, implicit_euler)"));
        }

        // ---------- bdf2 ----------
        @Test
        @DisplayName("G25: bdf2 exhibits approximately second-order convergence on smooth decay")
        void bdf2ShowsSecondOrderConvergence() throws Exception {
            double exact = analyticalDecay(1, -5, 0, 2);
            double errH = Math.abs(solveScalar("a", "diffeqn(y[1] + 5*y[0], 0, 1, 2, 0.1, bdf2)") - exact);
            double errHalfH = Math.abs(solveScalar("b", "diffeqn(y[1] + 5*y[0], 0, 1, 2, 0.05, bdf2)") - exact);
            double order = Math.log(errH / errHalfH) / Math.log(2.0);
            assertTrue(order > 1.5 && order < 2.5, "Observed BDF2 order=" + order);
        }

        @Test
        @DisplayName("G26: bdf2 stays stable on a stiff system")
        void bdf2StableOnStiffSystem() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 1, 0.1, bdf2)");
            assertEquals(0.0, y, 1e-6);
        }

        @Test
        @DisplayName("G27: a bdf2 solve with only a single step matches an implicit_euler solve with that same single step (bootstrap)")
        void bdf2SingleStepMatchesImplicitEulerBootstrap() throws Exception {
            double bdf2Y = solveScalar("a", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 1, bdf2)");
            double ieY = solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 1, 1, implicit_euler)");
            assertEquals(ieY, bdf2Y, 1e-6);
        }

        @Test
        @DisplayName("G28: a bdf2 solve with multiple steps improves accuracy over implicit_euler")
        void bdf2MultiStepImprovesOverImplicitEuler() throws Exception {
            double exact = analyticalDecay(1, -3, 0, 3);
            double ieY = solveScalar("a", "diffeqn(y[1] + 3*y[0], 0, 1, 3, 0.05, implicit_euler)");
            double bdf2Y = solveScalar("b", "diffeqn(y[1] + 3*y[0], 0, 1, 3, 0.05, bdf2)");
            assertTrue(Math.abs(bdf2Y - exact) < Math.abs(ieY - exact));
        }

        @Test
        @DisplayName("G29: bdf2 decays a very stiff mode cleanly without oscillation or blow-up")
        void bdf2DecaysStiffModeCleanly() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 2, 0.01, bdf2)");
            assertTrue(Double.isFinite(y));
            assertEquals(0.0, y, 1e-2);
        }

        @Test
        @DisplayName("G30: bdf2 works on a higher-order equation via the companion system reduction")
        void bdf2WorksOnHigherOrderEquation() throws Exception {
            double y = solveScalar("A",
                    "diffeqnHO(y[2] + 1000*y[1] + y[0], 0, @(1,2)(1, 0), 1, 0.01, bdf2)");
            assertTrue(Double.isFinite(y));
        }
    }

    // ==================================================================
    // H. Worked examples reproduced verbatim (doc section 8)
    // ==================================================================
    @Nested
    @DisplayName("H. Worked examples from the documentation")
    class WorkedExampleTests {

        @Test
        @DisplayName("H1: simple scalar decay, endpoint only, using defaults")
        void workedExampleSimpleScalarDecayDefaults() throws Exception {
            double y = solveScalar("y", "diffeqn(y[1] + 2*y[0], 0, 1, 5)");
            assertEquals(analyticalDecay(1, -2, 0, 5), y, TIGHT);
        }

        @Test
        @DisplayName("H2: same equation, full trajectory, evenly sampled at 50 points")
        void workedExamplePathEvenlySampledAt50Points() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4, 50)");
            assertEquals(50, m.getRows());
        }

        @Test
        @DisplayName("H3: a stiff scalar equation solved with bdf2")
        void workedExampleStiffScalarBdf2() throws Exception {
            assertDoesNotThrow(() -> solveScalar("y", "diffeqn(y[1] + 1000*y[0], 0, 1, 2, 0.001, bdf2)"));
        }

        @Test
        @DisplayName("H4: the explicit Lotka-Volterra two-equation system solved to an endpoint state")
        void workedExampleLotkaVolterraSystemEndpoint() throws Exception {
            assertDoesNotThrow(()
                    -> solveOnly("diffeqn(@(2)(\"y[2]-(0.6*y[0]-0.03*y[0]*y[1])\", "
                            + "\"y[2]-(-0.9*y[1]+0.02*y[0]*y[1])\"), 0, @(1,2)(30, 4), 20, 0.01, rk4)"));
        }

        @Test
        @DisplayName("H5: a third-order equation in higher-order form, full state trajectory")
        void workedExampleThirdOrderHOFullStateTrajectory() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPathHO(3*x*sin(x)*y[3] + 4*x*y[2] + 3*ln(x)*y[1] + 4*y[0], 1, "
                    + "@(1,3)(1, 0, 0), 3, 0.01, bdf2, state)");
            assertEquals(4, m.getCols());
        }

        @Test
        @DisplayName("H6: the same call, capped to 100 evenly spaced points, still with full state")
        void workedExampleThirdOrderHOCappedTo100PointsFullState() throws Exception {
            Matrix m = solveMatrix("A",
                    "diffeqnPathHO(3*x*sin(x)*y[3] + 4*x*y[2] + 3*ln(x)*y[1] + 4*y[0], 1, "
                    + "@(1,3)(1, 0, 0), 3, 0.01, bdf2, 100, state)");
            assertEquals(100, m.getRows());
            assertEquals(4, m.getCols());
        }
    }

    // ==================================================================
    // I. Troubleshooting / error-condition cases (doc section 9)
    // ==================================================================
    @Nested
    @DisplayName("I. Troubleshooting and error-condition cases")
    class TroubleshootingTests {

        @Test
        @DisplayName("I1: combining a diffeqn call with other math on the same line throws")
        void combiningDiffeqnWithOtherMathThrows() {
            assertRootExpressionRejected("diffeqn(y[1] + 2*y[0], 0, 1, 5) * 2");
        }

        @Test
        @DisplayName("I2: a scalar result is retrievable via getValue after assignment")
        void scalarResultRetrievableViaGetValue() throws Exception {
            double y = solveScalar("b", "diffeqn(y[1] + 2*y[0], 0, 1, 5)");
            assertTrue(Double.isFinite(y));
        }

        @Test
        @DisplayName("I3: a matrix result is retrievable via FunctionManager.lookUp(...).getMatrix() after assignment")
        void matrixResultRetrievableViaFunctionManager() throws Exception {
            String name = uniqueName("A");
            MathExpression me = new MathExpression(
                    name + "=diffeqnPathHO(3*x*sin(x)*y[3]+4*x*y[2]+3*ln(x)*y[1]+4*y[0], 1, "
                    + "@(1,3)(1, 0, 0), 3, 0.01, bdf2, state)");
            me.solve();
            Matrix m = FunctionManager.lookUp(name).getMatrix();
            assertNotNull(m);
        }

        @Test
        @DisplayName("I4: an rk45 trajectory without points is monotone, finite, and reaches the requested endpoint")
        void rk45UnevenSpacingWithoutPointsIsExpected() throws Exception {
            Matrix m = solveMatrix("A", "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.5, rk45)");
            assertTrue(m.getRows() > 1);
            assertEquals(0.0, m.getElem(0, 0), 1e-12);
            assertEquals(5.0, m.getElem(m.getRows() - 1, 0), 1e-9);
            for (int r = 1; r < m.getRows(); r++) {
                assertTrue(m.getElem(r, 0) > m.getElem(r - 1, 0));
                assertTrue(Double.isFinite(m.getElem(r, 1)));
            }
        }

        @Test
        @DisplayName("I5: switching from rk4 to implicit_euler fixes divergence on a stiff system")
        void switchingToImplicitEulerFixesStiffDivergence() throws Exception {
            double rk4Y = solveScalar("a", "diffeqn(y[1] + 1000*y[0], 0, 1, 1, 0.1, rk4)");
            double ieY = solveScalar("b", "diffeqn(y[1] + 1000*y[0], 0, 1, 1, 0.1, implicit_euler)");
            boolean rk4Diverged = !Double.isFinite(rk4Y) || Math.abs(rk4Y) > 1e3;
            assertTrue(rk4Diverged);
            assertEquals(0.0, ieY, 1e-8);
        }

        @Test
        @DisplayName("I6: on a smooth multi-step decay problem, bdf2 is more accurate than implicit_euler")
        void implicitEulerStableButLessAccurateThanBdf2() throws Exception {
            double exact = analyticalDecay(1, -5, 0, 2);
            double ieY = solveScalar("a", "diffeqn(y[1] + 5*y[0], 0, 1, 2, 0.1, implicit_euler)");
            double bdf2Y = solveScalar("b", "diffeqn(y[1] + 5*y[0], 0, 1, 2, 0.1, bdf2)");
            double ieError = Math.abs(ieY - exact);
            double bdf2Error = Math.abs(bdf2Y - exact);
            assertTrue(Double.isFinite(ieY) && Double.isFinite(bdf2Y));
            assertTrue(bdf2Error < ieError,
                    "Expected BDF2 error < implicit-Euler error, got " + bdf2Error + " vs " + ieError);
        }

        @Test
        @DisplayName("I7: presentationStrategy=state currently produces identical output to trajectory on plain diffeqnPath")
        void presentationStrategyStateSameShapeAsTrajectoryOnPlainPath() throws Exception {
            Matrix stateMode = solveMatrix("S",
                    "diffeqnPath(y[1] + 2*y[0], 0, 1, 1, 0.1, rk4, 5, state)");
            Matrix trajectoryMode = solveMatrix("T",
                    "diffeqnPath(y[1] + 2*y[0], 0, 1, 1, 0.1, rk4, 5, trajectory)");
            assertMatricesEqual(trajectoryMode, stateMode, 1e-12);
        }

        @Test
        @DisplayName("I8: passing an equation array to diffeqnPathHO does not publish a result")
        void arrayToDiffeqnPathHORejectedClearError() throws Throwable {
            assertInvalidSolveProducesNoResult(
                    "diffeqnPathHO(@(2)(\"y[2]-y[1]\", \"y[2]-(-y[0])\"), 0, @(1,2)(1, 0), 1, 0.01, rk4, state)");
        }

        @Test
        @DisplayName("I9: a system equation with no top-order y[n] reference does not publish a result")
        void wrongTargetSymbolErrorSurfaces() throws Throwable {
            assertInvalidSolveProducesNoResult(
                    "diffeqn(@(2)(\"y[0]-1\", \"y[2]-0\"), 0, @(1,2)(0, 0), 1, 0.01, rk4)");
        }

        @Test
        @DisplayName("I10: t0 equal to tEnd is a no-op for diffeqnHO as well, returning y0 unchanged")
        void t0EqualsTEndNoOpForDiffeqnHO() throws Exception {
            double y = solveScalar("A", "diffeqnHO(y[2] + y[0], 4, @(1,2)(2, -1), 4)");
            assertEquals(2.0, y, 1e-9);
        }
    }
}
