package com.github.gbenroscience.math.differentialcalculus.equations.standard;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.util.FunctionManager;
import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer.OrderAnnotation;
import org.junit.jupiter.api.Order;

/**
 * Battery tests for ParserNG's differential-equation frontend and numerical
 * integration layer.
 *
 * Target: JUnit Jupiter 5.10.3
 *
 * Coverage: - diffeqn - diffeqnPath - diffeqnHO - diffeqnPathHO - scalar
 * equations - genuine coupled first-order systems - trajectory/state
 * presentation strategies - Euler - RK4 - RK45 - implicit Euler - BDF2
 *
 * The tests deliberately favor equations with known analytical behavior,
 * invariant relationships, or clean convergence characteristics.
 */
@TestMethodOrder(OrderAnnotation.class)
public class DiffEngineTest {

    private static final double ABS_LOOSE = 1e-3;
    private static final double ABS_MEDIUM = 1e-5;
    private static final double ABS_TIGHT = 1e-8;

    /*
     * Numeric extraction is used only for vector-valued EvalResult output.
     * Scalar endpoints use MathExpression#getValue directly.
     */
    private static final Pattern NUMBER = Pattern.compile(
            "[-+]?(?:(?:\\d+(?:\\.\\d*)?)|(?:\\.\\d+))(?:[Ee][-+]?\\d+)?"
    );

    private static int testCounter;

    @BeforeEach
    void resetTestCounter() {
        testCounter++;
    }

    // =========================================================================
    // Helpers
    // =========================================================================
    /**
     * Executes a scalar assignment and returns the resulting scalar value.
     */
    private static double scalar(String expression) {
        String name = "__ode_scalar_test_result";
        MathExpression me = new MathExpression(name + "=" + expression);
        me.solve();
        return me.getValue(name);
    }

    /**
     * Executes an arbitrary ParserNG expression and returns its EvalResult.
     *
     * This assumes the public MathExpression#solve() method returns EvalResult,
     * consistent with the EvalResult usage already present in the ParserNG
     * command-line harness.
     */
    private static MathExpression.EvalResult eval(String expression) {
        MathExpression me = new MathExpression(expression);
        return me.solveGeneric();
    }

    /**
     * Parses the numeric values represented by EvalResult.toString().
     *
     * This is mainly for vector-valued diffeqn(system, ...) results.
     */
    private static double[] numbersFromResult(MathExpression.EvalResult result) {
        String text = String.valueOf(result);
        Matcher matcher = NUMBER.matcher(text);

        List<Double> values = new ArrayList<>();

        while (matcher.find()) {
            values.add(Double.parseDouble(matcher.group()));
        }

        double[] out = new double[values.size()];

        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }

        return out;
    }

    private static void assertNear(double expected, double actual, double tolerance) {
        assertEquals(
                expected,
                actual,
                tolerance,
                () -> "expected=" + expected + ", actual=" + actual
        );
    }

    private static void assertVectorNear(
            double[] expected,
            double[] actual,
            double tolerance) {

        assertEquals(
                expected.length,
                actual.length,
                "Unexpected vector length"
        );

        for (int i = 0; i < expected.length; i++) {
            assertNear(expected[i], actual[i], tolerance);
        }
    }

    /**
     * Executes a path expression assigned to a matrix variable and retrieves
     * the matrix object through ParserNG's public FunctionManager API.
     */
    private static Object pathMatrix(String expression) {
        String name = "__ode_path_test_result";

        MathExpression me = new MathExpression(name + "=" + expression);
        me.solve();

        Object function = FunctionManager.lookUp(name);

        assertNotNull(function, "Path assignment was not registered");

        try {
            Method getMatrix = function.getClass().getMethod("getMatrix");
            Object matrix = getMatrix.invoke(function);

            assertNotNull(matrix, "Path returned a null matrix");

            return matrix;
        } catch (ReflectiveOperationException ex) {
            throw new AssertionError(
                    "Unable to retrieve ParserNG matrix through getMatrix()",
                    ex
            );
        }
    }

    /**
     * Extract the numerical contents of a ParserNG matrix without assuming the
     * concrete Matrix implementation's accessor naming.
     *
     * Supported possibilities: - getArray() - getData() - getValues() -
     * toArray() - public/private double[][] field named data/array/values -
     * getRows()/getCols() + get(row,col)
     */
    private static double[][] matrixData(Object matrix) {

        String[] arrayMethods = {
            "getArray",
            "getData",
            "getValues",
            "toArray"
        };

        for (String methodName : arrayMethods) {
            try {
                Method method = matrix.getClass().getMethod(methodName);

                Object value = method.invoke(matrix);

                if (value instanceof double[][]) {
                    return (double[][]) value;
                }
            } catch (ReflectiveOperationException ignored) {
                // Try the next representation.
            }
        }

        String[] fieldNames = {
            "data",
            "array",
            "values"
        };

        for (String fieldName : fieldNames) {
            try {
                Field field = matrix.getClass().getDeclaredField(fieldName);
                field.setAccessible(true);

                Object value = field.get(matrix);

                if (value instanceof double[][]) {
                    return (double[][]) value;
                }
            } catch (ReflectiveOperationException ignored) {
                // Try the next representation.
            }
        }

        Integer rows = invokeInt(matrix, "getRows");
        Integer cols = invokeInt(matrix, "getCols");

        if (rows == null) {
            rows = invokeInt(matrix, "rows");
        }

        if (cols == null) {
            cols = invokeInt(matrix, "cols");
        }

        if (rows != null && cols != null) {

            Method getter = null;

            try {
                getter = matrix.getClass().getMethod(
                        "get",
                        int.class,
                        int.class
                );
            } catch (NoSuchMethodException ignored) {
                // Search declared method below.
            }

            if (getter == null) {
                try {
                    getter = matrix.getClass().getDeclaredMethod(
                            "get",
                            int.class,
                            int.class
                    );
                    getter.setAccessible(true);
                } catch (ReflectiveOperationException ignored) {
                    // Give up with a useful message.
                }
            }

            if (getter != null) {
                double[][] result = new double[rows][cols];

                try {
                    for (int r = 0; r < rows; r++) {
                        for (int c = 0; c < cols; c++) {
                            Object value = getter.invoke(matrix, r, c);

                            if (!(value instanceof Number)) {
                                throw new AssertionError(
                                        "Matrix get(row,col) did not return Number"
                                );
                            }

                            result[r][c] = ((Number) value).doubleValue();
                        }
                    }

                    return result;

                } catch (ReflectiveOperationException ex) {
                    throw new AssertionError(
                            "Unable to extract matrix values",
                            ex
                    );
                }
            }
        }

        throw new AssertionError(
                "Could not discover a numerical matrix representation for "
                + matrix.getClass().getName()
        );
    }

    private static Integer invokeInt(Object target, String methodName) {
        try {
            Method method = target.getClass().getMethod(methodName);

            Object value = method.invoke(target);

            if (value instanceof Number) {
                return ((Number) value).intValue();
            }

        } catch (ReflectiveOperationException ignored) {
            // Not available.
        }

        return null;
    }

    private static void assertPathShape(
            double[][] data,
            int expectedRows,
            int expectedColumns) {

        assertEquals(expectedRows, data.length, "Unexpected trajectory row count");

        for (double[] row : data) {
            assertEquals(
                    expectedColumns,
                    row.length,
                    "Unexpected trajectory column count"
            );
        }
    }

    // =========================================================================
    // diffeqn — scalar equations
    // =========================================================================
    @Test
    @Order(1)
    void diffeqnRK4_constantDerivative() {
        // Tests the simplest non-trivial IVP: y' = 1, y(0)=0 => y(1)=1.
        double y = scalar("diffeqn(y[1]-1, 0, 0, 1, 0.01, rk4)");
        assertNear(1.0, y, ABS_TIGHT);
    }

    @Test
    @Order(2)
    void diffeqnEuler_constantDerivative() {
        // Euler should be exact for a constant derivative.
        double y = scalar("diffeqn(y[1]-1, 0, 0, 1, 0.01, euler)");
        assertNear(1.0, y, ABS_TIGHT);
    }

    @Test
    @Order(3)
    void diffeqnImplicitEuler_constantDerivative() {
        // Implicit Euler is also exact for a constant derivative.
        double y = scalar(
                "diffeqn(y[1]-1, 0, 0, 1, 0.01, implicit_euler)"
        );
        assertNear(1.0, y, ABS_TIGHT);
    }

    @Test
    @Order(4)
    void diffeqnBDF2_constantDerivative() {
        // BDF2 should preserve an affine solution exactly up to floating point.
        double y = scalar("diffeqn(y[1]-1, 0, 0, 1, 0.01, bdf2)");
        assertNear(1.0, y, ABS_TIGHT);
    }

    @Test
    @Order(5)
    void diffeqnRK45_constantDerivative() {
        // Adaptive RK45 should also reproduce an affine solution essentially exactly.
        double y = scalar("diffeqn(y[1]-1, 0, 0, 1, 0.01, rk45)");
        assertNear(1.0, y, ABS_TIGHT);
    }

    @Test
    @Order(6)
    void diffeqnRK4_linearDecay() {
        // Classic exact test: y'=-2y, y(0)=1 => y(5)=e^-10.
        double expected = Math.exp(-10.0);

        double actual = scalar(
                "diffeqn(y[1]+2*y[0], 0, 1, 5, 0.01, rk4)"
        );

        assertNear(expected, actual, 1e-10);
    }

    @Test
    @Order(7)
    void diffeqnEuler_linearDecay() {
        // Explicit Euler's discrete solution is (1-2h)^N; tests first-order behavior.
        double h = 0.01;
        int n = 100;
        double expected = Math.pow(1.0 - 2.0 * h, n);

        double actual = scalar(
                "diffeqn(y[1]+2*y[0], 0, 1, 1, 0.01, euler)"
        );

        assertNear(expected, actual, 1e-12);
    }

    @Test
    @Order(8)
    void diffeqnImplicitEuler_linearDecay() {
        // Backward Euler has the closed-form discrete factor 1/(1+2h).
        double h = 0.01;
        int n = 100;
        double expected = Math.pow(1.0 / (1.0 + 2.0 * h), n);

        double actual = scalar(
                "diffeqn(y[1]+2*y[0], 0, 1, 1, 0.01, implicit_euler)"
        );

        assertNear(expected, actual, 1e-10);
    }

    @Test
    @Order(9)
    void diffeqnBDF2_linearDecay() {
        // Tests the BDF2 nonlinear/linear solve path on a scalar linear equation.
        double actual = scalar(
                "diffeqn(y[1]+2*y[0], 0, 1, 1, 0.01, bdf2)"
        );

        assertNear(Math.exp(-2.0), actual, ABS_MEDIUM);
    }

    @Test
    @Order(10)
    void diffeqnRK45_linearDecay() {
        // Tests adaptive Dormand-Prince against a known exponential solution.
        double actual = scalar(
                "diffeqn(y[1]+2*y[0], 0, 1, 5, 0.01, rk45)"
        );

        assertNear(Math.exp(-10.0), actual, 1e-8);
    }

    @Test
    @Order(11)
    void diffeqnRK4_integratingT() {
        // y'=t, y(0)=0 => y=t^2/2. RK4 integrates this polynomial exactly.
        double actual = scalar(
                "diffeqn(y[1]-t, 0, 0, 2, 0.01, rk4)"
        );

        assertNear(2.0, actual, 1e-11);
    }

    @Test
    @Order(12)
    void diffeqnRK4_trigonometricForcing() {
        // y'=cos(t), y(0)=0 => y=sin(t). Tests transcendental forcing.
        double actual = scalar(
                "diffeqn(y[1]-cos(t), 0, 0, 1, 0.005, rk4)"
        );

        assertNear(Math.sin(1.0), actual, 1e-8);
    }

    @Test
    @Order(13)
    void diffeqnRK4_timeVaryingCoefficient() {
        // y'=t*y with y(0)=1 => exp(t^2/2). Tests state-dependent variable coefficient.
        double actual = scalar(
                "diffeqn(y[1]-t*y[0], 0, 1, 1, 0.005, rk4)"
        );

        assertNear(Math.exp(0.5), actual, 1e-7);
    }

    @Test
    @Order(14)
    void diffeqnBackwardIntegration() {
        // Tests reverse-time integration: y'=1 with y(1)=0 must give y(0)=-1.
        double actual = scalar(
                "diffeqn(y[1]-1, 1, 0, 0, 0.01, rk4)"
        );

        assertNear(-1.0, actual, ABS_TIGHT);
    }

    @Test
    @Order(15)
    void diffeqnDefaultMethodAndStep() {
        // Tests the documented defaults: h=0.01 and method=rk4.
        double explicit = scalar(
                "diffeqn(y[1]+2*y[0], 0, 1, 1)"
        );

        double explicitNamed = scalar(
                "diffeqn(y[1]+2*y[0], 0, 1, 1, 0.01, rk4)"
        );

        assertNear(explicitNamed, explicit, 1e-12);
    }

    // =========================================================================
    // diffeqn — genuine first-order systems
    // =========================================================================
    @Test
    @Order(16)
    void diffeqnSystemRK4_constantVectorField() {
        // Two-state system: y0'=1, y1'=2. Tests @(2)(...) and y[2] derivative placeholder.
        MathExpression.EvalResult result = eval(
                "diffeqn(@(2)("
                + "\"y[2]-1\","
                + "\"y[2]-2\""
                + "), 0, @(1,2)(0,0), 1, 0.01, rk4)"
        );

        assertVectorNear(
                new double[]{1.0, 2.0},
                numbersFromResult(result),
                ABS_TIGHT
        );
    }

    @Test
    @Order(17)
    void diffeqnSystemEuler_constantVectorField() {
        // Same system with Euler; exact because the derivative field is constant.
        MathExpression.EvalResult result = eval(
                "diffeqn(@(2)("
                + "\"y[2]-1\","
                + "\"y[2]-2\""
                + "), 0, @(1,2)(0,0), 1, 0.01, euler)"
        );

        assertVectorNear(
                new double[]{1.0, 2.0},
                numbersFromResult(result),
                ABS_TIGHT
        );
    }

    @Test
    @Order(18)
    void diffeqnSystemImplicitEuler_constantVectorField() {
        // Tests the implicit system pathway without nonlinear coupling.
        MathExpression.EvalResult result = eval(
                "diffeqn(@(2)("
                + "\"y[2]-1\","
                + "\"y[2]-2\""
                + "), 0, @(1,2)(0,0), 1, 0.01, implicit_euler)"
        );

        assertVectorNear(
                new double[]{1.0, 2.0},
                numbersFromResult(result),
                ABS_TIGHT
        );
    }

    @Test
    @Order(19)
    void diffeqnSystemBDF2_constantVectorField() {
        // BDF2 should preserve affine state trajectories accurately.
        MathExpression.EvalResult result = eval(
                "diffeqn(@(2)("
                + "\"y[2]-1\","
                + "\"y[2]-2\""
                + "), 0, @(1,2)(0,0), 1, 0.01, bdf2)"
        );

        assertVectorNear(
                new double[]{1.0, 2.0},
                numbersFromResult(result),
                ABS_TIGHT
        );
    }

    @Test
    @Order(20)
    void diffeqnSystemRK45_constantVectorField() {
        // Adaptive vector integration of a trivial constant field.
        MathExpression.EvalResult result = eval(
                "diffeqn(@(2)("
                + "\"y[2]-1\","
                + "\"y[2]-2\""
                + "), 0, @(1,2)(0,0), 1, 0.01, rk45)"
        );

        assertVectorNear(
                new double[]{1.0, 2.0},
                numbersFromResult(result),
                1e-8
        );
    }

    @Test
    @Order(21)
    void diffeqnSystemChain() {
        // Tests a coupled chain: y0'=y1, y1'=0. Exact state at t=1 is [2,2].
        MathExpression.EvalResult result = eval(
                "diffeqn(@(2)("
                + "\"y[2]-y[1]\","
                + "\"y[2]\""
                + "), 0, @(1,2)(0,2), 1, 0.01, rk4)"
        );

        assertVectorNear(
                new double[]{2.0, 2.0},
                numbersFromResult(result),
                1e-8
        );
    }

    @Test
    @Order(22)
    void diffeqnSystemHarmonicOscillatorRK4() {
        // First-order form of y''=-y: y0'=y1, y1'=-y0. At pi/2 => [0,-1].
        MathExpression.EvalResult result = eval(
                "diffeqn(@(2)("
                + "\"y[2]-y[1]\","
                + "\"y[2]-(-y[0])\""
                + "), 0, @(1,2)(1,0), 1.5707963267948966, 0.01, rk4)"
        );

        assertVectorNear(
                new double[]{0.0, -1.0},
                numbersFromResult(result),
                2e-4
        );
    }

    @Test
    @Order(23)
    void diffeqnSystemDecoupledLinearRK4() {
        // Tests independent state equations sharing one solver invocation.
        MathExpression.EvalResult result = eval(
                "diffeqn(@(2)("
                + "\"y[2]-(-y[0])\","
                + "\"y[2]-(-2*y[1])\""
                + "), 0, @(1,2)(1,1), 1, 0.01, rk4)"
        );

        assertVectorNear(
                new double[]{Math.exp(-1.0), Math.exp(-2.0)},
                numbersFromResult(result),
                1e-6
        );
    }

    @Test
    @Order(24)
    void diffeqnLotkaVolterraSystem() {
        // Tests genuinely nonlinear cross-coupling using the documented two-state syntax.
        MathExpression.EvalResult result = eval(
                "diffeqn(@(2)("
                + "\"y[2]-(0.6*y[0]-0.03*y[0]*y[1])\","
                + "\"y[2]-(-0.9*y[1]+0.02*y[0]*y[1])\""
                + "), 0, @(1,2)(30,4), 2, 0.01, rk4)"
        );

        double[] actual = numbersFromResult(result);

        assertEquals(2, actual.length);
        assertTrue(Double.isFinite(actual[0]));
        assertTrue(Double.isFinite(actual[1]));
    }

    @Test
    @Order(25)
    void diffeqnFourStateCoupledSystem() {
        // Tests a four-state coupled system and the y[4] derivative placeholder convention.
        MathExpression.EvalResult result = eval(
                "diffeqn(@(4)("
                + "\"y[4]-y[1]\","
                + "\"y[4]-(-2*y[0]+y[2])\","
                + "\"y[4]-2*sin(t)*y[3]\","
                + "\"y[4]-(y[0]-2*y[2])\""
                + "), 0, @(1,4)(1,0,0,1), 1, 0.01, rk4)"
        );

        double[] actual = numbersFromResult(result);

        assertEquals(4, actual.length);

        for (double value : actual) {
            assertTrue(Double.isFinite(value));
        }
    }

    // =========================================================================
    // diffeqnPath — scalar and system trajectories
    // =========================================================================
    @Test
    @Order(26)
    void diffeqnPathScalarTrajectoryShape() {
        // trajectory presentation should produce [t, y] rows.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(y[1]-1, 0, 0, 1, 0.1, rk4, 11, trajectory)"
                )
        );

        assertPathShape(data, 11, 2);
        assertNear(0.0, data[0][0], 1e-12);
        assertNear(1.0, data[10][0], 1e-12);
        assertNear(0.0, data[0][1], 1e-12);
        assertNear(1.0, data[10][1], 1e-12);
    }

    @Test
    @Order(27)
    void diffeqnPathScalarStateShape() {
        // For a scalar first-order equation, state and trajectory both contain t,y.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(y[1]-1, 0, 0, 1, 0.1, rk4, 11, state)"
                )
        );

        assertPathShape(data, 11, 2);
    }

    @Test
    @Order(28)
    void diffeqnPathRK4ExponentialTrajectory() {
        // Tests scalar trajectory values against y=e^-2t.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(y[1]+2*y[0], 0, 1, 1, 0.01, rk4, 11, trajectory)"
                )
        );

        assertNear(Math.exp(-2.0), data[10][1], 1e-6);
    }

    @Test
    @Order(29)
    void diffeqnPathEulerTrajectory() {
        // Tests that Euler records every accepted fixed step.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(y[1]-1, 0, 0, 1, 0.05, euler, 21, trajectory)"
                )
        );

        assertPathShape(data, 21, 2);
        assertNear(1.0, data[20][1], 1e-12);
    }

    @Test
    @Order(30)
    void diffeqnPathImplicitEulerTrajectory() {
        // Tests path recording through an implicit method.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(y[1]+2*y[0], 0, 1, 1, 0.01, implicit_euler, 21, trajectory)"
                )
        );

        assertPathShape(data, 21, 2);
        assertTrue(Double.isFinite(data[20][1]));
    }

    @Test
    @Order(31)
    void diffeqnPathBDF2Trajectory() {
        // Tests the multistep history path through the public Path API.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(y[1]+2*y[0], 0, 1, 1, 0.01, bdf2, 21, trajectory)"
                )
        );

        assertPathShape(data, 21, 2);
        assertNear(Math.exp(-2.0), data[20][1], 1e-3);
    }

    @Test
    @Order(32)
    void diffeqnPathRK45TrajectoryResampling() {
        // RK45 produces irregular accepted steps internally; points=21 forces a uniform output grid.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(y[1]+2*y[0], 0, 1, 1, 0.01, rk45, 21, trajectory)"
                )
        );

        assertPathShape(data, 21, 2);

        for (int i = 1; i < data.length; i++) {
            assertTrue(data[i][0] > data[i - 1][0]);
        }

        assertNear(Math.exp(-2.0), data[20][1], 1e-6);
    }

    @Test
    @Order(33)
    void diffeqnPathSystemTrajectoryPresentation() {
        // For an explicit system, trajectory output contains t followed by
        // the complete state vector: [t, y[0], y[1], ...].
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(@(2)("
                        + "\"y[2]-1\","
                        + "\"y[2]-2\""
                        + "), 0, @(1,2)(0,0), 1, 0.1, rk4, 11, trajectory)"
                )
        );

        assertPathShape(data, 11, 3);

        assertNear(0.0, data[0][0], 1e-12);
        assertNear(0.0, data[0][1], 1e-12);
        assertNear(0.0, data[0][2], 1e-12);

        assertNear(1.0, data[10][1], 1e-12);
        assertNear(2.0, data[10][2], 1e-12);
    }

    @Test
    @Order(34)
    void diffeqnPathSystemStatePresentation() {
        // state mode for a two-state system should expose t,y[0],y[1].
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(@(2)("
                        + "\"y[2]-1\","
                        + "\"y[2]-2\""
                        + "), 0, @(1,2)(0,0), 1, 0.1, rk4, 11, state)"
                )
        );

        assertPathShape(data, 11, 3);
        assertNear(1.0, data[10][1], 1e-12);
        assertNear(2.0, data[10][2], 1e-12);
    }

    @Test
    @Order(35)
    void diffeqnPathSystemRK45StatePresentation() {
        // Combines adaptive stepping, system solving, point resampling, and state presentation.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPath(@(2)("
                        + "\"y[2]-y[1]\","
                        + "\"y[2]\""
                        + "), 0, @(1,2)(0,2), 1, 0.01, rk45, 21, state)"
                )
        );

        assertPathShape(data, 21, 3);
        assertNear(2.0, data[20][1], 1e-5);
        assertNear(2.0, data[20][2], 1e-8);
    }

    // =========================================================================
    // diffeqnHO — higher-order scalar equations
    // =========================================================================
    @Test
    @Order(36)
    void diffeqnHORK4SecondOrderAffine() {
        // y''=0, y(0)=1, y'(0)=2 => y(1)=3. Companion-system correctness.
        double y = scalar(
                "diffeqnHO(y[2], 0, @(1,2)(1,2), 1, 0.01, rk4)"
        );

        assertNear(3.0, y, 1e-10);
    }

    @Test
    @Order(37)
    void diffeqnHOEulerSecondOrderAffine() {
        // Euler on the companion system should preserve this affine solution exactly.
        double y = scalar(
                "diffeqnHO(y[2], 0, @(1,2)(1,2), 1, 0.01, euler)"
        );

        assertNear(3.0, y, 1e-10);
    }

    @Test
    @Order(38)
    void diffeqnHOImplicitEulerSecondOrderAffine() {
        // Tests implicit Euler through HO->first-order reduction.
        double y = scalar(
                "diffeqnHO(y[2], 0, @(1,2)(1,2), 1, 0.01, implicit_euler)"
        );

        assertNear(3.0, y, 1e-10);
    }

    @Test
    @Order(39)
    void diffeqnHOBDF2SecondOrderAffine() {
        // Tests BDF2's bootstrap followed by multistep evolution of an HO equation.
        double y = scalar(
                "diffeqnHO(y[2], 0, @(1,2)(1,2), 1, 0.01, bdf2)"
        );

        assertNear(3.0, y, 1e-8);
    }

    @Test
    @Order(40)
    void diffeqnHORK45SecondOrderAffine() {
        // Adaptive RK45 must preserve the exact affine companion solution.
        double y = scalar(
                "diffeqnHO(y[2], 0, @(1,2)(1,2), 1, 0.01, rk45)"
        );

        assertNear(3.0, y, 1e-8);
    }

    @Test
    @Order(41)
    void diffeqnHOSecondOrderConstantAccelerationRK4() {
        // y''=2, y(0)=1, y'(0)=0 => y(1)=2.
        double y = scalar(
                "diffeqnHO(y[2]-2, 0, @(1,2)(1,0), 1, 0.01, rk4)"
        );

        assertNear(2.0, y, 1e-10);
    }

    @Test
    @Order(42)
    void diffeqnHOSecondOrderConstantAccelerationEuler() {
        // Euler companion integration of a polynomial solution.
        double y = scalar(
                "diffeqnHO(y[2]-2, 0, @(1,2)(1,0), 1, 0.01, euler)"
        );

        assertNear(1.99, y, 1e-10);
    }

    @Test
    @Order(43)
    void diffeqnHOSecondOrderConstantAccelerationImplicitEuler() {
        // y''=2, y(0)=1, y'(0)=0 => exact y(1)=2.
        // Backward Euler on the companion system gives the discrete result 2.01
        // for h=0.01 and 100 steps.
        double y = scalar(
                "diffeqnHO(y[2]-2, 0, @(1,2)(1,0), 1, 0.01, implicit_euler)"
        );

        assertNear(2.01, y, 1e-12);
    }

    @Test
    @Order(44)
    void diffeqnHOSecondOrderConstantAccelerationBDF2() {
        // BDF2 should closely reproduce the quadratic exact solution.
        double y = scalar(
                "diffeqnHO(y[2]-2, 0, @(1,2)(1,0), 1, 0.01, bdf2)"
        );

        assertNear(2.0, y, 3e-4);
    }

    @Test
    @Order(45)
    void diffeqnHOSecondOrderConstantAccelerationRK45() {
        // Adaptive RK45 on a polynomial forcing case.
        double y = scalar(
                "diffeqnHO(y[2]-2, 0, @(1,2)(1,0), 1, 0.01, rk45)"
        );

        assertNear(2.0, y, 1e-8);
    }

    @Test
    @Order(46)
    void diffeqnHOSimpleHarmonicOscillatorRK4() {
        // y''+y=0, y(0)=1,y'(0)=0 => y(pi/2)=0.
        double y = scalar(
                "diffeqnHO(y[2]+y[0], 0, @(1,2)(1,0), "
                + "1.5707963267948966, 0.01, rk4)"
        );

        assertNear(0.0, y, 1e-6);
    }

    @Test
    @Order(47)
    void diffeqnHOSimpleHarmonicOscillatorRK45() {
        // Adaptive version of the harmonic oscillator; checks high-order HO reduction.
        double y = scalar(
                "diffeqnHO(y[2]+y[0], 0, @(1,2)(1,0), "
                + "1.5707963267948966, 0.01, rk45)"
        );

        assertNear(0.0, y, 1e-7);
    }

    @Test
    @Order(48)
    void diffeqnHOThirdOrderPolynomial() {
        // y'''=6, y(0)=1, y'(0)=2, y''(0)=3
        // Exact solution: y(t)=1+2t+(3/2)t²+t³, hence y(1)=5.5.
        double y = scalar(
                "diffeqnHO(y[3]-6, 0, @(1,3)(1,2,3), 1, 0.01, rk4)"
        );

        assertNear(5.5, y, 1e-8);
    }

    @Test
    @Order(49)
    void diffeqnHOFourthOrderZeroDerivative() {
        // y''''=0 with [1,2,3,4] gives y(t)=1+2t+1.5t^2+(2/3)t^3; y(1)=5.166666...
        double y = scalar(
                "diffeqnHO(y[4], 0, @(1,4)(1,2,3,4), 1, 0.01, rk4)"
        );

        assertNear(5.166666666666667, y, 1e-8);
    }

    @Test
    @Order(50)
    void diffeqnHOComplexCoefficientEquation() {
        // Exercises nonlinear-looking syntax, sin(x), and higher-order state references.
        double y = scalar(
                "diffeqnHO(y[2]+3*y[1]-sin(x)*y[0]+3*x^2, "
                + "3, @(1,2)(1,0.5), 4, 0.01, rk4)"
        );

        assertTrue(Double.isFinite(y));
    }

    // =========================================================================
    // diffeqnPathHO — higher-order trajectories
    // =========================================================================
    @Test
    @Order(51)
    void diffeqnPathHOThirdOrderStatePresentation() {
        // state mode must expose t,y[0],y[1],y[2] for a third-order equation.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[3], 0, @(1,3)(1,2,3), "
                        + "1, 0.1, rk4, 11, state)"
                )
        );

        assertPathShape(data, 11, 4);
        assertNear(0.0, data[0][0], 1e-12);
        assertNear(1.0, data[0][1], 1e-12);
        assertNear(2.0, data[0][2], 1e-12);
        assertNear(3.0, data[0][3], 1e-12);
    }

    @Test
    @Order(52)
    void diffeqnPathHOThirdOrderTrajectoryPresentation() {
        // y''' = 0 with y(0)=1, y'(0)=2, y''(0)=3.
        // Exact solution: y(t) = 1 + 2t + 1.5t^2, hence y(1) = 4.5.
        // trajectory mode must collapse the HO state to t,y[0].
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[3], 0, @(1,3)(1,2,3), "
                        + "1, 0.1, rk4, 11, trajectory)"
                )
        );
 

        assertPathShape(data, 11, 2);
        assertNear(1.0, data[0][1], 1e-12);
        assertNear(4.5, data[10][1], 1e-8);
    }

    @Test
    @Order(53)
    void diffeqnPathHOFourthOrderStatePresentation() {
        // Exercises four-state companion history and full state presentation.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[4], 0, @(1,4)(1,2,3,4), "
                        + "1, 0.1, rk4, 11, state)"
                )
        );

        assertPathShape(data, 11, 5);
        assertNear(1.0, data[0][1], 1e-12);
        assertNear(2.0, data[0][2], 1e-12);
        assertNear(3.0, data[0][3], 1e-12);
        assertNear(4.0, data[0][4], 1e-12);
    }

    @Test
    @Order(54)
    void diffeqnPathHOThirdOrderEuler() {
        // Tests HO trajectory recording with the lowest-order explicit method.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[3]-6, 0, @(1,3)(1,2,3), "
                        + "1, 0.1, euler, 11, state)"
                )
        );

        assertPathShape(data, 11, 4);
        assertTrue(Double.isFinite(data[10][1]));
    }

    @Test
    @Order(55)
    void diffeqnPathHOThirdOrderImplicitEuler() {
        // Tests Newton-based implicit integration through the HO presentation.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[3]-6, 0, @(1,3)(1,2,3), "
                        + "1, 0.1, implicit_euler, 11, state)"
                )
        );

        assertPathShape(data, 11, 4);
        assertTrue(Double.isFinite(data[10][1]));
    }

    @Test
    @Order(56)
    void diffeqnPathHOThirdOrderBDF2() {
        // Tests multistep history with a third-order companion system.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[3]-6, 0, @(1,3)(1,2,3), "
                        + "1, 0.1, bdf2, 11, state)"
                )
        );

        assertPathShape(data, 11, 4);
        assertTrue(Double.isFinite(data[10][1]));
    }

    @Test
    @Order(57)
    void diffeqnPathHOThirdOrderRK45() {
        // Tests adaptive RK45 with full higher-order state output.
        //
        // y''' = 6
        // y(0) = 1, y'(0) = 2, y''(0) = 3
        //
        // Exact solution:
        // y(t) = 1 + 2t + (3/2)t² + t³
        // Therefore y(1) = 5.5.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[3]-6, 0, @(1,3)(1,2,3), "
                        + "1, 0.01, rk45, 21, state)"
                )
        );

        assertPathShape(data, 21, 4);
        assertNear(5.5, data[20][1], 1e-6);
    }

    @Test
    @Order(58)
    void diffeqnPathHOHarmonicOscillatorTrajectory() {
        // Tests trajectory presentation for a second-order oscillatory equation.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[2]+y[0], 0, @(1,2)(1,0), "
                        + "1.5707963267948966, 0.01, rk4, 17, trajectory)"
                )
        );

        assertPathShape(data, 17, 2);
        assertNear(0.0, data[16][1], 1e-5);
    }

    @Test
    @Order(59)
    void diffeqnPathHOHarmonicOscillatorState() {
        // Same oscillator, but verifies that velocity/state component is retained.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[2]+y[0], 0, @(1,2)(1,0), "
                        + "1.5707963267948966, 0.01, rk4, 17, state)"
                )
        );

        assertPathShape(data, 17, 3);
        assertNear(0.0, data[16][1], 1e-5);
        assertNear(-1.0, data[16][2], 1e-5);
    }

    @Test
    @Order(60)
    void diffeqnPathHODocumentedThirdOrderBDF2Example() {
        // Regression test based on the documented third-order ParserNG example.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO("
                        + "3*x*sin(x)*y[3]+4*x*y[2]+3*ln(x)*y[1]+4*y[0], "
                        + "1, @(1,3)(1,0,0), 3, 0.01, bdf2, state)"
                )
        );

        assertPathShape(data, 201, 4);
        assertNear(1.0, data[0][1], 1e-12);
        assertNear(0.3505756956, data[data.length - 1][1], 1e-4);
    }

    @Test
    @Order(61)
    void diffeqnPathHOWithRequestedPoints() {
        // Tests resampling of a fixed-step higher-order trajectory to a smaller requested point count.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[2]-2, 0, @(1,2)(1,0), "
                        + "1, 0.01, rk4, 13, state)"
                )
        );

        assertPathShape(data, 13, 3);

        for (int i = 1; i < data.length; i++) {
            assertTrue(data[i][0] > data[i - 1][0]);
        }
    }

    @Test
    @Order(62)
    void diffeqnPathHOAdaptiveResampling() {
        // Tests that RK45's irregular accepted steps can be presented as a uniform grid.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[2]+y[0], 0, @(1,2)(1,0), "
                        + "2, 0.05, rk45, 21, state)"
                )
        );

        assertPathShape(data, 21, 3);

        for (int i = 1; i < data.length; i++) {
            assertTrue(data[i][0] > data[i - 1][0]);
        }
    }

    @Test
    @Order(63)
    void diffeqnPathHOBackwardIntegration() {
        // Tests reverse-time HO trajectory generation.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[2], 1, @(1,2)(0,1), "
                        + "0, 0.1, rk4, 11, state)"
                )
        );

        assertPathShape(data, 11, 3);
        assertNear(1.0, data[0][0], 1e-12);
        assertNear(0.0, data[10][0], 1e-12);
    }

    @Test
    @Order(64)
    void diffeqnPathHOScalarStateCarriesDerivatives() {
        // Verifies that state mode exposes derivative components,
        // not merely a plotting trajectory.
        //
        // y'' = 2
        // y(0) = 1
        // y'(0) = 0
        //
        // Exact solution:
        // y(t)  = 1 + t²
        // y'(t) = 2t
        //
        // At t=1: y=2, y'=2.
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[2]-2, 0, @(1,2)(1,0), "
                        + "1, 0.1, rk4, 11, state)"
                )
        );

        assertPathShape(data, 11, 3);
        assertNear(1.0, data[0][1], 1e-12);
        assertNear(0.0, data[0][2], 1e-12);

        assertNear(2.0, data[10][1], 1e-12);
        assertNear(2.0, data[10][2], 1e-12);
    }

    @Test
    @Order(65)
    void diffeqnPathHOTrajectoryDropsDerivativeColumns() {
        // trajectory presentation must reduce a third-order state history to [t,y[0]].
        double[][] data = matrixData(
                pathMatrix(
                        "diffeqnPathHO(y[3]-6, 0, @(1,3)(1,2,3), "
                        + "1, 0.1, rk4, 11, trajectory)"
                )
        );

        assertPathShape(data, 11, 2);
    }

    // =========================================================================
    // Parser/API semantics and edge cases
    // =========================================================================
    @Test
    @Order(66)
    void diffeqnAssignmentIsLegal() {
        // Differential-equation commands are root commands, but assignment is the sanctioned exception.
        MathExpression me = new MathExpression(
                "saved=diffeqn(y[1]-1, 0, 0, 1, 0.01, rk4)"
        );

        assertDoesNotThrow(() -> me.solve());
        assertNear(1.0, me.getValue("saved"), 1e-10);
    }

    @Test
    @Order(67)
    void diffeqnCannotBeEmbeddedInsideArithmetic() {
        // Tests the root-command restriction enforced by the differential-equation frontend.
        assertThrows(
                RuntimeException.class,
                () -> new MathExpression(
                        "diffeqn(y[1]-1, 0, 0, 1, 0.01, rk4)+1"
                ).solve()
        );
    }

    @Test
    @Order(68)
    void diffeqnCannotBeNestedInsideAnotherFunction() {
        // A solve command is not an ordinary scalar function and cannot be nested.
        assertThrows(
                RuntimeException.class,
                () -> new MathExpression(
                        "sin(diffeqn(y[1]-1, 0, 0, 1, 0.01, rk4))"
                ).solve()
        );
    }


}
