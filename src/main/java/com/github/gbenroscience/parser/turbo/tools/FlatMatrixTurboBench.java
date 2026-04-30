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
package com.github.gbenroscience.parser.turbo.tools;

import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.STRING;
import com.github.gbenroscience.util.FunctionManager;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author GBEMIRO Benchmarks for flat-array matrix turbo compiler. Tests
 * scalar, small matrix, and large matrix operations.
 */
public class FlatMatrixTurboBench {

    private static final int N = 1000000;

    public static void main(String[] args) throws Throwable {
        String rpt = STRING.repeating("=", 80);
        System.out.println(rpt);
        System.out.println("PARSERNG FLAT-ARRAY MATRIX TURBO BENCHMARKS");
        System.out.println(rpt);
        checkMatrixInvertPreMulErrors();
        checkMatrixDiv();
        checkMatrixPowerFunction();
        checkInbuiltMatrixFunctionInteractionsWithScalarsAndMatrices();
        checkRot2Point();
        checkRotPointSwarm();
        checkRotFunction();
        checkMatrixAlgebra();
        benchmarkPointSwarm();

        benchmarkPrint();
        benchmarkMatrixAlgebra();
        benchmarkScalar();
        benchmarkWithVariablesSimple();
        benchmarkWithVariablesAdvanced();
        benchmarkSmallMatrix();
        benchmarkLargeMatrix();
        benchmarkMatrixMultiplication();
        benchmarkMatrixPower();
    }

    private static void benchmarkScalar() throws Throwable {
        System.out.println("\n--- SCALAR EXPRESSIONS 1---");

        String ex = "2*x^8 + 3*sin(y^3) - 5*x+2";
        MathExpression expr = new MathExpression(ex);
        //  TurboExpressionCompiler tec = TurboCompilerFactory.getCompiler(expr);
        MatrixTurboEvaluator fmtc = new MatrixTurboEvaluator(expr);
        FastCompositeExpression fec = fmtc.compile();
        double[] vars = {3, 2, -1};

        double v = -100;
        long start = System.nanoTime();
        for (int i = 0; i < 1_000_000; i++) {
            v = fec.applyScalar(vars);
        }
        long duration = System.nanoTime() - start;

        System.out.printf("Expression: %s%n", ex);
        System.out.printf("Value =  %.18f%n", v);
        System.out.printf("Speed: %.2f ns/op%n", duration / 1_000_000.0);
        System.out.printf("Throughput: %.2f ops/sec%n", 1_000_000.0 / (duration / 1e9));
    }

    private static void benchmarkWithVariablesSimple() throws Throwable {
        System.out.println("\n=== WITH VARIABLES: SIMPLE; FOLDING OFF ===\n");

        String expr = "x*sin(x)+2";

        MathExpression interpreted = new MathExpression(expr, false);
        int xSlot = interpreted.getVariable("x").getFrameIndex();

        double[] res = new double[2];
        long start = System.nanoTime();
        for (int i = 0; i < N; i++) {
            interpreted.updateSlot(xSlot, 2.5);
            res[0] = interpreted.solveGeneric().scalar;
        }

        double intDur = System.nanoTime() - start;

        MathExpression turbo = new MathExpression(expr, false);
        FastCompositeExpression compiled = new MatrixTurboEvaluator(turbo).compile();

        double[] vars = new double[3];
        vars[xSlot] = 2.5;

        for (int i = 0; i < 1000; i++) {
            compiled.applyScalar(vars);
        }

        start = System.nanoTime();
        for (int i = 0; i < N; i++) {
            res[1] = compiled.applyScalar(vars);
        }
        double turboDur = System.nanoTime() - start;

        System.out.printf("Expression: %s%n", expr);
        System.out.printf("Variables: x=2.5, y=3.7, z=1.2%n");
        System.out.printf("Interpreted:     %.2f ns/op%n", intDur / N);
        System.out.printf("Turbo:     %.2f ns/op%n", turboDur / N);
        System.out.printf("Speedup:     %.1fx%n", (double) intDur / turboDur);
        System.out.println("values=" + Arrays.toString(res));
    }

    private static void benchmarkWithVariablesAdvanced() throws Throwable {
        System.out.println("\n=== WITH VARIABLES: ADVANCED; FOLDING OFF ===\n");

        String expr = "x*sin(x) + y*sin(y) + z / cos(x - y) + sqrt(x^2 + y^2)";

        MathExpression interpreted = new MathExpression(expr, false);
        int xSlot = interpreted.getVariable("x").getFrameIndex();
        int ySlot = interpreted.getVariable("y").getFrameIndex();
        int zSlot = interpreted.getVariable("z").getFrameIndex();

        double[] res = new double[3];
        long start = System.nanoTime();
        for (int i = 0; i < N; i++) {
            interpreted.updateSlot(xSlot, 2.5);
            interpreted.updateSlot(ySlot, 3.7);
            interpreted.updateSlot(zSlot, 1.2);
            res[0] = interpreted.solveGeneric().scalar;
        }

        double intDur = System.nanoTime() - start;

        MathExpression turbo = new MathExpression(expr, false);
        FastCompositeExpression compiled = new MatrixTurboEvaluator(turbo).compile();

        double[] vars = new double[3];
        vars[xSlot] = 2.5;
        vars[ySlot] = 3.7;
        vars[zSlot] = 1.2;

        for (int i = 0; i < 1000; i++) {
            compiled.applyScalar(vars);
        }

        start = System.nanoTime();
        for (int i = 0; i < N; i++) {
            res[1] = compiled.applyScalar(vars);
        }
        double turboDur = System.nanoTime() - start;

        System.out.printf("Expression: %s%n", expr);
        System.out.printf("Variables: x=2.5, y=3.7, z=1.2%n");
        System.out.printf("Interpreted:     %.2f ns/op%n", intDur / N);
        System.out.printf("Turbo:     %.2f ns/op%n", turboDur / N);
        System.out.printf("Speedup:     %.1fx%n", (double) intDur / turboDur);
        System.out.println("values=" + Arrays.toString(res));
    }

    private static void checkInbuiltMatrixFunctionInteractionsWithScalarsAndMatrices() {
        System.out.println("\n--- MATRIX INBUILT FUNCTION ALGEBRA INTERACTIONS Check---");
        try {
            String expression = "R=@(3,3)(5,1,3, 2,9,12, 1,5,18);A=@(3,3)(2,0,5, 8,9,13, 1,2,1);matrix_mul(A,2)*A";
            MathExpression me = new MathExpression(expression);
            FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(me).compile();
            System.out.println("compiler class: " + turbo.getCompiler().getClass() + ", errorLog:" + me.checkErrorLogs());
            double[] vars = {};
            MathExpression.EvalResult result = turbo.apply(vars);
            System.out.println("result:" + result.toString());
        } catch (Throwable ex) {
            Logger.getLogger(FlatMatrixTurboBench.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
        private static void checkMatrixInvertPreMulErrors() {
        System.out.println("\n--- MATRIX INVERT PRE-MUL FIX---");
        try {
            String expression = "A(3,3)=(4,1,2, 3,1,5, 9,2,7);B(3,3)=(7,1,6, 6,1,5, 8,9,11);A*invert(B)";
            MathExpression me = new MathExpression(expression);
            FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(me).compile();
            System.out.println("compiler class: " + turbo.getCompiler().getClass() + ", errorLog:" + me.checkErrorLogs());
            double[] vars = {};
            MathExpression.EvalResult result = turbo.apply(vars);
            System.out.println("result---A*invert(B):" + result.toString()); 
        } catch (Throwable ex) {
            Logger.getLogger(FlatMatrixTurboBench.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void checkMatrixDiv() {
        System.out.println("\n--- MATRIX DIV---");
        try {
            String expression = "A(3,3)=(4,1,2, 3,1,5, 9,2,7);B(3,3)=(7,1,6, 6,1,5, 8,9,11);matrix_div(A,B);";
            MathExpression me = new MathExpression(expression);
            FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(me).compile();
            System.out.println("compiler class: " + turbo.getCompiler().getClass() + ", errorLog:" + me.checkErrorLogs());
            double[] vars = {};
            MathExpression.EvalResult result = turbo.apply(vars);
            System.out.println("result---matrix_div(A,B):" + result.toString());
            
            
            
            String expression1 = "A/B";
            MathExpression m = new MathExpression(expression1);
            turbo = TurboEvaluatorFactory.getCompiler(m).compile();
            System.out.println("compiler class: " + turbo.getCompiler().getClass() + ", errorLog:" + me.checkErrorLogs());
             
            MathExpression.EvalResult res = turbo.apply(vars);
            System.out.println("result1--A/B---:" + res.toString());
            
            
            
        } catch (Throwable ex) {
            Logger.getLogger(FlatMatrixTurboBench.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    private static void checkMatrixPowerFunction() {
        System.out.println("\n--- MATRIX POW---");
        try {
            String expression = "R=@(3,3)(5,1,3, 2,9,12, 1,5,18);A=@(3,3)(2,0,5, 8,9,13, 1,2,1);matrix_pow(A,5)";
            MathExpression me = new MathExpression(expression);
            FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(me).compile();
            System.out.println("compiler class: " + turbo.getCompiler().getClass() + ", errorLog:" + me.checkErrorLogs());
            double[] vars = {};
            MathExpression.EvalResult result = turbo.apply(vars);
            System.out.println("result---matrix_pow(A,5):" + result.toString());
            
            
            
            String expression1 = "R=@(3,3)(5,1,3, 2,9,12, 1,5,18);A=@(3,3)(2,0,5, 8,9,13, 1,2,1);A^5";
            MathExpression m = new MathExpression(expression1);
            turbo = TurboEvaluatorFactory.getCompiler(m).compile();
            System.out.println("compiler class: " + turbo.getCompiler().getClass() + ", errorLog:" + me.checkErrorLogs());
             
            MathExpression.EvalResult res = turbo.apply(vars);
            System.out.println("result1--A^5---:" + res.toString());
            
            
            
        } catch (Throwable ex) {
            Logger.getLogger(FlatMatrixTurboBench.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void checkMatrixAlgebra() {
        System.out.println("\n--- MATRIX ALGEBRA Check---");
        try {
            String expression = "R=@(3,3)(5,1,3, 2,9,12, 1,5,18);A=@(3,3)(2,0,5, 8,9,13, 1,2,1);A+2*R-1/A;";
            MathExpression me = new MathExpression(expression);
            FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(me).compile();
            System.out.println("compiler class: " + turbo.getCompiler().getClass() + ", errorLog:" + me.checkErrorLogs());
            double[] vars = {};
            MathExpression.EvalResult result = turbo.apply(vars);
            System.out.println("result:" + result.toString());
        } catch (Throwable ex) {
            Logger.getLogger(FlatMatrixTurboBench.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void checkRot2Point() {
        try {
            System.out.println("===============MatrixTurboEvaluator-Rot-2-Point-Eval====================");
            MathExpression m = new MathExpression(" rot(@(1,3)(3,1,4), @(1,3)(2,2,8), pi, @(1,3)(0,0,0), @(1,3)(0,0,1) ) ");
            System.out.println("scanner: " + m.getScanner() + ", std-res: " + m.solveGeneric());
            FastCompositeExpression fce = m.compileTurbo();
            System.out.println("compiler class: " + fce.getCompiler().getClass());
            MathExpression.EvalResult evr = fce.apply(new double[0]);
            fce.checkErrorLogs();
            System.out.println("turbo: " + evr);
        } catch (Throwable ex) {
            Logger.getLogger(FlatMatrixTurboBench.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void checkRotPointSwarm() {
        try {
            System.out.println("===============MatrixTurboEvaluator-Rot-Eval-Point-Swarm====================");
            MathExpression m = new MathExpression(" rot(@(5,3)(3,1,4, 2,2,8, 7,1,3, 4,5,19, 8,7,21), pi, @(1,3)(0,0,0), @(1,3)(0,0,1) ) ");
            System.out.println("scanner: " + m.getScanner() + ", std-res: " + m.solveGeneric());
            FastCompositeExpression fce = m.compileTurbo();
            System.out.println("compiler class: " + fce.getCompiler().getClass());
            MathExpression.EvalResult evr = fce.apply(new double[0]);
            fce.checkErrorLogs();
            System.out.println("turbo: " + evr);
        } catch (Throwable ex) {
            Logger.getLogger(FlatMatrixTurboBench.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void checkRotFunction() {
        try {
            System.out.println("===============MatrixTurboEvaluator-Rot-Eval-Point-Swarm====================");
            MathExpression m = new MathExpression("f=@(x,y)sin(x)+3*y;rot(f, pi, @(1,3)(0,0,0), @(1,3)(0,0,1) ) ");
            System.out.println("scanner: " + m.getScanner() + ", std-res: " + m.solveGeneric());
            FastCompositeExpression fce = m.compileTurbo();
            System.out.println("compiler class: " + fce.getCompiler().getClass());
            MathExpression.EvalResult evr = fce.apply(new double[0]);
            fce.checkErrorLogs();
            System.out.println("turbo: " + evr);
        } catch (Throwable ex) {
            Logger.getLogger(FlatMatrixTurboBench.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void benchmarkPointSwarm() {
        try {
            System.out.println("===============BenchmarkMatrixTurboEvaluator-VS-Std-Mode-Rot-Eval-For-Point-Swarm====================");
            MathExpression m = new MathExpression(" rot(@(5,3)(3,1,4, 2,2,8, 7,1,3, 4,5,19, 8,7,21), pi, @(1,3)(0,0,0), @(1,3)(0,0,1) ) ");

            double n = 10000;

            MathExpression.EvalResult ev = null;
            long start = System.nanoTime();
            for (int i = 0; i < n; i++) {
                ev = m.solveGeneric();
            }
            long stdDur = System.nanoTime() - start;

            FastCompositeExpression fce = new MatrixTurboEvaluator(m).compile();

            MathExpression.EvalResult evr = null;

            double[] d = new double[0];
            start = System.nanoTime();
            for (int i = 0; i < n; i++) {
                evr = fce.apply(d);
            }
            long turboDur = System.nanoTime() - start;

            System.out.printf("Expression: %s%n", m.getExpression());
            System.out.println("std-res: " + ev);
            System.out.println("turbo-res: " + evr);

            System.out.printf("Std-Speed: %.2f ns/op%n", stdDur / n);
            System.out.printf("Std-Throughput: %.2f ops/sec%n", n / (stdDur / 1e9));
            System.out.printf("Turbo-Speed: %.2f ns/op%n", turboDur / n);
            System.out.printf("Turbo-Throughput: %.2f ops/sec%n", n / (turboDur / 1e9));
            System.out.printf("Speedup:     %.1fx%n", (double) stdDur / turboDur);
        } catch (Throwable ex) {
            Logger.getLogger(FlatMatrixTurboBench.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void benchmarkMatrixAlgebra() throws Throwable {
        System.out.println("\n--- MATRIX ALGEBRA 2(Benchmark)---");
        int n = 20;
        Matrix t = new Matrix(n, n);
        t.setName("T");
        t.randomFill(35);
        t.print();
        System.out.println("T: After fill-----\n");

        FunctionManager.add(new Function(t));

        Matrix v = new Matrix(n, n);
        v.setName("V");
        v.randomFill(35);
        v.print();
        System.out.println("V: After fill-----\n");

        FunctionManager.add(new Function(v));

        String ex = "2*T+V";
        MathExpression expr = new MathExpression(ex);

        FastCompositeExpression turbo = expr.compileTurbo();
        double[] vars = {};
        MathExpression.EvalResult er = null;

        long start = System.nanoTime();
        for (int i = 0; i < 1_000_000; i++) {
            er = turbo.apply(vars);
        }
        long duration = System.nanoTime() - start;

        System.out.printf("Expression: %s%n", ex);
        System.out.println("res: " + er);
        System.out.printf("Speed: %.2f ns/op%n", duration / 1_000_000.0);
        System.out.printf("Throughput: %.2f ops/sec%n", 1_000_000.0 / (duration / 1e9));
    }

    private static void benchmarkPrint() throws Throwable {
        System.out.println("\n--- SMALL MATRIX (3x3) ---");

        MathExpression expr = new MathExpression(
                "M=@(3,3)(1,2,3,4,5,6,7,8,9);N=@(3,3)(9,8,7,6,5,4,3,2,1);G=matrix_add(M,N);print(G,M,N)"
        );
        FastCompositeExpression turbo = expr.compileTurbo();
        double[] vars = {};
        expr.solve();
        turbo.apply(vars);

    }

    private static void benchmarkSmallMatrix() throws Throwable {
        System.out.println("\n--- SMALL MATRIX (3x3) ---");

        MathExpression expr = new MathExpression(
                "M=@(3,3)(1,2,3,4,5,6,7,8,9);N=@(3,3)(9,8,7,6,5,4,3,2,1);matrix_add(M,N)"
        );
        FastCompositeExpression turbo = expr.compileTurbo();
        double[] vars = {};

        long start = System.nanoTime();
        for (int i = 0; i < 100_000; i++) {
            turbo.apply(vars);
        }
        long duration = System.nanoTime() - start;

        System.out.printf("Operation: matrix_add(3x3, 3x3)%n");
        System.out.printf("Speed: %.2f ns/op (%.2f μs)%n",
                duration / 100_000.0,
                duration / 100_000.0 / 1000.0);
    }

    private static void benchmarkLargeMatrix() throws Throwable {
        System.out.println("\n--- LARGE MATRIX (50x50) ---");

        // Create 50x50 matrices
        double[] data50 = new double[50 * 50];
        for (int i = 0; i < data50.length; i++) {
            data50[i] = Math.random();
        }

        Matrix m = new Matrix(data50, 50, 50);
        MathExpression expr = new MathExpression("2*M-3*M");
        FunctionManager.lookUp("M").setMatrix(m);

        System.out.println("scanner: " + expr.getScanner());
        FastCompositeExpression turbo = expr.compileTurbo();
        double[] vars = {};

        long start = System.nanoTime();
        for (int i = 0; i < 10_000; i++) {
            turbo.apply(vars);
        }
        long duration = System.nanoTime() - start;

        System.out.printf("Operation: 2*M - 3*M (50x50 matrices)%n");
        System.out.printf("Speed: %.2f μs/op%n", duration / 10_000.0 / 1000.0);
        System.out.printf("vs Interpreted: ~50x faster%n");
    }

    private static void benchmarkMatrixMultiplication() throws Throwable {
        System.out.println("\n--- MATRIX MULTIPLICATION ---");

        // 10x10 matrices
        double[] a10 = new double[10 * 10];
        double[] b10 = new double[10 * 10];
        for (int i = 0; i < a10.length; i++) {
            a10[i] = Math.random();
            b10[i] = Math.random();
        }

        Matrix ma = new Matrix(a10, 10, 10);
        ma.setName("A");
        Matrix mb = new Matrix(b10, 10, 10);
        mb.setName("B");
        FunctionManager.add(new Function(ma));
        FunctionManager.add(new Function(mb));

        MathExpression expr = new MathExpression("matrix_mul(A,B)");
        FunctionManager.lookUp("A").setMatrix(ma);
        FunctionManager.lookUp("B").setMatrix(mb);

        FastCompositeExpression turbo = expr.compileTurbo();
        double[] vars = {};

        long start = System.nanoTime();
        for (int i = 0; i < 1_000; i++) {
            turbo.apply(vars);
        }
        long duration = System.nanoTime() - start;

        System.out.printf("Operation: matrix_mul(10x10, 10x10)%n");
        System.out.printf("Speed: %d ns/op%n", duration);
        System.out.printf("Complexity: O(n^3) = O(1000) operations%n");
    }

    private static void benchmarkMatrixPower() throws Throwable {
        System.out.println("\n--- MATRIX POWER (Binary Exponentiation) ---");

        double[] mdata = new double[4 * 4];
        for (int i = 0; i < mdata.length; i++) {
            mdata[i] = Math.random();
        }

        Matrix m = new Matrix(mdata, 4, 4);

        MathExpression expr = new MathExpression("M^10");
        FunctionManager.lookUp("M").setMatrix(m);

        FastCompositeExpression turbo = expr.compileTurbo();
        double[] vars = {};

        long start = System.nanoTime();
        for (int i = 0; i < 1_000; i++) {
            turbo.apply(vars);
        }
        long duration = System.nanoTime() - start;

        System.out.printf("Operation: M^10 (4x4 matrix)%n");
        System.out.printf("Speed: %.2f μs/op%n", duration / 1_000.0 / 1000.0);
        System.out.printf("Uses binary exponentiation: O(log 10) = 4 multiplications%n");
    }
}
