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
package com.github.gbenroscience.parser.turbo;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.TurboEvaluatorFactory;
import com.github.gbenroscience.util.FunctionManager;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class MatrixTurboEvaluatorTest {

    private final Random rand = new Random();
    private final double[] emptyFrame = new double[0];

    @BeforeEach
    void setup() {
        // Clear previous functions to ensure test isolation
        FunctionManager.clear();
    }

    @ParameterizedTest
    @ValueSource(ints = {3, 5, 10})
    @DisplayName("Turbo Matrix Method Validation")
    void testMatrixMethods(int size) throws Throwable {
        testLinearSolver(size);
        testCofactorAndAdjoint(size);
        testMatrixDivision(size);
        testEigenvalues(size);
    }

    @ParameterizedTest
    @ValueSource(ints = {3, 5, 10, 50}) // Test with multiple matrix sizes
    public void testLinearSolver(int n) throws Throwable {
        double[] data = generateRandomArray(n * (n + 1));
        // Create function and embed the matrix
        Matrix m = new Matrix(data, n, n + 1);
        m.setName("M");
        Function f = new Function(m);

        MathExpression expr = new MathExpression("linear_sys(M)");

        FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(expr)
                .compile();

        Matrix res = turbo.applyMatrix(emptyFrame);

        assertNotNull(res);
        assertEquals(n, res.getRows());
        assertEquals(1, res.getCols());
    }

    @ParameterizedTest
    @ValueSource(ints = {3, 5, 10, 50}) // Test with multiple matrix sizes
    public void testCofactorAndAdjoint(int n) throws Throwable {
        Matrix m = new Matrix(generateRandomArray(n * n), n, n);
        m.setName("A");
        FunctionManager.add(new Function(m));

        // Adjoint test
        MathExpression adjExpr = new MathExpression("adjoint(A)");
        FastCompositeExpression turboAdj = TurboEvaluatorFactory.getCompiler(adjExpr)
                .compile();

        assertEquals(n, turboAdj.applyMatrix(emptyFrame).getRows());

        // Cofactors test
        MathExpression cofExpr = new MathExpression("cofactor(A)");
        FastCompositeExpression turboCof = TurboEvaluatorFactory.getCompiler(cofExpr)
                .compile();

        assertEquals(n, turboCof.applyMatrix(emptyFrame).getRows());
    }

    @ParameterizedTest
    @ValueSource(ints = {3, 5, 10, 50}) // Test with multiple matrix sizes
    public void testMatrixDivision(int n) throws Throwable {
        Matrix a = new Matrix(generateRandomArray(n * n), n, n);
        a.setName("A");
        Matrix b = new Matrix(generateRandomArray(n * n), n, n);
        b.setName("B");
        FunctionManager.add(new Function(a));
        FunctionManager.add(new Function(b));

        MathExpression divExpr = new MathExpression("A / B");
        FastCompositeExpression turboDiv = TurboEvaluatorFactory.getCompiler(divExpr)
                .compile();

        assertEquals(n, turboDiv.applyMatrix(emptyFrame).getRows());
    }

    @ParameterizedTest
    @ValueSource(ints = {3, 5, 10, 50}) // Test with multiple matrix sizes
    public void testEigenvalues(int n) throws Throwable {
        Matrix e = new Matrix(generateRandomArray(n * n), n, n);
        e.setName("R");
        FunctionManager.add(new Function(e));

        MathExpression eigenExpr = new MathExpression("eigvalues(R)");
        FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(eigenExpr)
                .compile();

        Matrix res = turbo.applyMatrix(emptyFrame);
        assertEquals(n, res.getRows());
        assertEquals(2, res.getCols());
    }

    public double[] generateRandomArray(int size) {
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) {
            arr[i] = 1.0 + (rand.nextDouble() * 10.0);
        }
        return arr;
    }

    //  @Test
    public void testEigenval() {

        MathExpression me = new MathExpression("R=@(5,5)(3.6960389979858523  ,10.656660507703922  ,8.250361808124694  ,1.2864528025782198  ,9.735431283674686,"
                + "5.585459956012235  ,7.5122356839343745  ,6.063066728284797  ,8.559695263800457  ,3.7673226536438857,"
                + "8.701609027359616  ,7.689979890725766  ,3.9690824306208285  ,3.664071779088659  ,5.514556971406468,"
                + "1.7165078288826077  ,8.089363716212478  ,8.651319052236992  ,4.1374739688508955  ,1.234189853093153,"
                + "1.2567034692845471  ,2.0793007773147147  ,7.254190558589741  ,7.715028903257325  ,2.8009022598677604 );eigvalues(R)");
 

        FastCompositeExpression turbo = null;
        try {
            MathExpression eigenExpr = new MathExpression("eigvalues(R)");
            turbo = TurboEvaluatorFactory.getCompiler(eigenExpr)
                    .compile();
            Matrix res = turbo.applyMatrix(emptyFrame);
            System.out.println("eigValues-----------------------\n " + res);
            assertEquals(5, res.getRows());
            assertEquals(2, res.getCols());
        } catch (Throwable ex) {
            Logger.getLogger(MatrixTurboEvaluatorTest.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
}
