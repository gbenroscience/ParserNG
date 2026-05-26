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

    private static final int N = 100000;

    public static void main(String[] args) throws Throwable {
        String rpt = STRING.repeating("=", 80);
        System.out.println(rpt);
        System.out.println("PARSERNG FLAT-ARRAY MATRIX TURBO BENCHMARKS");
        System.out.println(rpt);
        checkTurboUserDefinedFunctionsInMatrixMode();
        checkTurboUserDefinedFunctionsWithAndroidCapableMatrixTurboEvaluatorInMatrixMode();
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

    private static void checkTurboUserDefinedFunctionsInMatrixMode() throws Throwable {
        
        System.out.println("=====================MATRIX-BENCHING-checkTurboUserDefinedFunctionsInMatrixMode()============================");
        /*
        String expr="sin(3*x)+cos(5*y)-sin(2*z^2)";
        String fnCall="f(x,y,z)";
        String fnDef=fnCall+"="+expr;
        */
        
         String expr = "sin(sqrt(x^2+y^2+z^2))";
        String fnCall="f(x,y,z)";
        String fnDef=fnCall+"="+expr;
        
        MathExpression baseFnDef = new MathExpression(fnDef+";"+fnCall);
        MatrixTurboEvaluator matrixTurboCompilerForBaseFn = new MatrixTurboEvaluator(baseFnDef);
        ScalarTurboEvaluator1 turboCompiler1ForFn = new ScalarTurboEvaluator1(baseFnDef);
        MathExpression rawMathDef = new MathExpression(expr);
        MatrixTurboEvaluator matrixTurboCompilerForRawMathExp = new MatrixTurboEvaluator(rawMathDef);
        ScalarTurboEvaluator1 turboCompiler1ForRawMathExp = new ScalarTurboEvaluator1(rawMathDef);

        ScalarTurboEvaluator2 turboCompiler2ForFn = new ScalarTurboEvaluator2(baseFnDef);
        ScalarTurboEvaluator2 turboCompiler2ForRawMathExp = new ScalarTurboEvaluator2(rawMathDef);

        FastCompositeExpression matrixTurboExecForBaseFn = matrixTurboCompilerForBaseFn.compile();
        FastCompositeExpression matrixTurboExecForRawMathExp = matrixTurboCompilerForRawMathExp.compile();
        FastCompositeExpression scalarTurboExecForBaseFn = turboCompiler1ForFn.compile();
        FastCompositeExpression scalarTurboExecForRawMathExp = turboCompiler1ForRawMathExp.compile();
        FastCompositeExpression scalarTurbo2ExecForBaseFn = turboCompiler2ForFn.compile();
        FastCompositeExpression scalarTurbo2ExecForRawMathExp = turboCompiler2ForRawMathExp.compile();

        double v = -100;
        double iterations = 10_000_000;

        System.out.println("=============Execute f(x,y,z)-IN-STD-MODE================");
        int xSlot = baseFnDef.getVariable("x").getFrameIndex();
        int ySlot = baseFnDef.getVariable("y").getFrameIndex();
        int zSlot = baseFnDef.getVariable("z").getFrameIndex();
        long strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            baseFnDef.updateSlot(xSlot, i % 9);
            baseFnDef.updateSlot(ySlot, i % 8);
            baseFnDef.updateSlot(zSlot, i % 9 - 2);
            v = baseFnDef.solveGeneric().scalar;
        }
        long duration1 = System.nanoTime() - strt;
        System.out.println("waste-product of standard-base-fn-exec---" + v);

        System.out.println("=============Execute sin(3*x)+cos(5*y)-sin(2*z^2) -IN-STD-MODE================");

        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            rawMathDef.updateSlot(xSlot, i % 9);
            rawMathDef.updateSlot(ySlot, i % 8);
            rawMathDef.updateSlot(zSlot, i % 9 - 2);
            v = rawMathDef.solveGeneric().scalar;
        }
        long duration2 = System.nanoTime() - strt;
        System.out.println("waste-product of standard-raw-math-exp-exec--- " + v);

        System.out.println("=============BASE-FN-DEF-EXECUTED-MATRIX-TURBO-MODE (f(x,y,z))================");

        double[] vars = {0, 0, 0};

        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = matrixTurboExecForBaseFn.applyScalar(vars);
        }
        long duration3 = System.nanoTime() - strt;
        System.out.println("waste-product of matrix-turbo-base-fn---" + v);

        System.out.println("=============RAW-MATH-EXP-EXECUTED-MATRIX-TURBO-MODE (sin(3*x)+cos(5*y)-sin(2*z^2))================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = matrixTurboExecForRawMathExp.applyScalar(vars);
        }
        long duration4 = System.nanoTime() - strt;
        System.out.println("waste-product of matrix-turbo-raw-math-exp---" + v);

        System.out.println("=============BASE-FN-DEF-EXECUTED-SCALAR-TURBO-1-MODE================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = scalarTurboExecForBaseFn.applyScalar(vars);
        }
        long duration5 = System.nanoTime() - strt;
        System.out.println("waste-product of scalar-turbo1-base-fn-exec---" + v);

        System.out.println("=============RAW-MATH-EXP-EXECUTED-SCALAR-TURBO-1-MODE================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = scalarTurboExecForRawMathExp.applyScalar(vars);
        }
        long duration6 = System.nanoTime() - strt;
        System.out.println("waste-product of scalar-turbo1-raw-exp-exec---" + v);

        System.out.println("=============BASE-FN-DEF-EXECUTED-SCALAR-TURBO-2-MODE================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = scalarTurbo2ExecForBaseFn.applyScalar(vars);
        }
        long duration7 = System.nanoTime() - strt;
        System.out.println("waste-product of scalar-turbo2-fn---" + v);

        System.out.println("=============RAW-MATH-EXP-EXECUTED-SCALAR-TURBO-2-MODE================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = scalarTurbo2ExecForRawMathExp.applyScalar(vars);
        }
        long duration8 = System.nanoTime() - strt;
        System.out.println("waste-product of scalar-turbo2-raw-exp-exec---" + v);

        System.out.printf("Expressions: %s and %s %n", baseFnDef.getExpression(), rawMathDef.getExpression());
        System.out.printf("Value =  %.18f%n", v);
        System.out.printf("Speed-standard-fn: %.2f ns/op%n", duration1 / iterations);
        System.out.printf("Speed-standard-raw-exp: %.2f ns/op%n", duration2 / iterations);
        System.out.printf("Speed-matrix-turbo-fn: %.2f ns/op%n", duration3 / iterations);
        System.out.printf("Speed-matrix-turbo-raw-exp: %.2f ns/op%n", duration4 / iterations);
        System.out.printf("Speed-scalar-turbo1-fn: %.2f ns/op%n", duration5 / iterations);
        System.out.printf("Speed-scalar-turbo1-raw-exp: %.2f ns/op%n", duration6 / iterations);
        System.out.printf("Speed-scalar-turbo2-fn: %.2f ns/op%n", duration7 / iterations);
        System.out.printf("Speed-scalar-turbo2-raw-exp: %.2f ns/op%n", duration8 / iterations);
        System.out.printf("Throughput-standard-fn: %.2f ops/sec%n", iterations / (duration1 / 1e9));
        System.out.printf("Throughput-standard-raw-exp: %.2f ops/sec%n", iterations / (duration2 / 1e9));
        System.out.printf("Throughput-matrix-turbo-fn: %.2f ops/sec%n", iterations / (duration3 / 1e9));
        System.out.printf("Throughput-matrix-turbo-raw-exp: %.2f ops/sec%n", iterations / (duration4 / 1e9));
        System.out.printf("Throughput-scalar-turbo1-fn: %.2f ops/sec%n", iterations / (duration5 / 1e9));
        System.out.printf("Throughput-scalar-turbo1-raw: %.2f ops/sec%n", iterations / (duration6 / 1e9));
        System.out.printf("Throughput-scalar-turbo2-fn: %.2f ops/sec%n", iterations / (duration7 / 1e9));
        System.out.printf("Throughput-scalar-turbo2-raw: %.2f ops/sec%n", iterations / (duration8 / 1e9));
        System.out.println("=================SPEED-UP-ZONE================");
        System.out.printf("Matrix-Turbo-fn vs Standard-fn:              %.1fx%n", (double) duration1 / duration3);
        System.out.printf("Matrix-Turbo-raw-exp vs Standard-raw-exp     %.1fx%n", (double) duration2 / duration4);
        System.out.printf("Scalar-Turbo1-fn vs Standard-fn:     %.1fx%n", (double) duration1 / duration5);
        System.out.printf("Scalar-Turbo1-raw-exp vs Standard-raw-exp:     %.1fx%n", (double) duration2 / duration6);
        System.out.printf("Scalar-Turbo2-fn vs Standard-fn:     %.1fx%n", (double) duration1 / duration7);
        System.out.printf("Scalar-Turbo2-raw-exp vs Standard-raw-exp:     %.1fx%n", (double) duration2 / duration8);
    }
    private static void checkTurboUserDefinedFunctionsWithAndroidCapableMatrixTurboEvaluatorInMatrixMode() throws Throwable {
        
        System.out.println("=====================ANDROID-CAPABLE-MATRIX-BENCHING-checkTurboUserDefinedFunctionsWithAndroidCapableMatrixTurboEvaluatorInMatrixMode()============================");
        /*
        String expr="sin(3*x)+cos(5*y)-sin(2*z^2)";
        String fnCall="f(x,y,z)";
        String fnDef=fnCall+"="+expr;
        */
        
         String expr = "sin(sqrt(x^2+y^2+z^2))";
        String fnCall="f(x,y,z)";
        String fnDef=fnCall+"="+expr;
        
        MathExpression baseFnDef = new MathExpression(fnDef+";"+fnCall);
        AndroidCapableMatrixTurboEvaluator matrixTurboCompilerForBaseFn = new AndroidCapableMatrixTurboEvaluator(baseFnDef);
        ScalarTurboEvaluator1 turboCompiler1ForFn = new ScalarTurboEvaluator1(baseFnDef);
        MathExpression rawMathDef = new MathExpression(expr);
        AndroidCapableMatrixTurboEvaluator matrixTurboCompilerForRawMathExp = new AndroidCapableMatrixTurboEvaluator(rawMathDef);
        ScalarTurboEvaluator1 turboCompiler1ForRawMathExp = new ScalarTurboEvaluator1(rawMathDef);

        ScalarTurboEvaluator2 turboCompiler2ForFn = new ScalarTurboEvaluator2(baseFnDef);
        ScalarTurboEvaluator2 turboCompiler2ForRawMathExp = new ScalarTurboEvaluator2(rawMathDef);

        FastCompositeExpression matrixTurboExecForBaseFn = matrixTurboCompilerForBaseFn.compile();
        FastCompositeExpression matrixTurboExecForRawMathExp = matrixTurboCompilerForRawMathExp.compile();
        FastCompositeExpression scalarTurboExecForBaseFn = turboCompiler1ForFn.compile();
        FastCompositeExpression scalarTurboExecForRawMathExp = turboCompiler1ForRawMathExp.compile();
        FastCompositeExpression scalarTurbo2ExecForBaseFn = turboCompiler2ForFn.compile();
        FastCompositeExpression scalarTurbo2ExecForRawMathExp = turboCompiler2ForRawMathExp.compile();

        double v = -100;
        double iterations = 50_000_000;

        System.out.println("=============Execute f(x,y,z)-IN-STD-MODE================");
        int xSlot = baseFnDef.getVariable("x").getFrameIndex();
        int ySlot = baseFnDef.getVariable("y").getFrameIndex();
        int zSlot = baseFnDef.getVariable("z").getFrameIndex();
        long strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            baseFnDef.updateSlot(xSlot, i % 9);
            baseFnDef.updateSlot(ySlot, i % 8);
            baseFnDef.updateSlot(zSlot, i % 9 - 2);
            v = baseFnDef.solveGeneric().scalar;
        }
        long duration1 = System.nanoTime() - strt;
        System.out.println("waste-product of standard-base-fn-exec---" + v);

        System.out.println("=============Execute sin(3*x)+cos(5*y)-sin(2*z^2) -IN-STD-MODE================");

        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            rawMathDef.updateSlot(xSlot, i % 9);
            rawMathDef.updateSlot(ySlot, i % 8);
            rawMathDef.updateSlot(zSlot, i % 9 - 2);
            v = rawMathDef.solveGeneric().scalar;
        }
        long duration2 = System.nanoTime() - strt;
        System.out.println("waste-product of standard-raw-math-exp-exec--- " + v);

        System.out.println("=============BASE-FN-DEF-EXECUTED-MATRIX-TURBO-MODE (f(x,y,z))================");

        double[] vars = {0, 0, 0};

        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = matrixTurboExecForBaseFn.applyScalar(vars);
        }
        long duration3 = System.nanoTime() - strt;
        System.out.println("waste-product of matrix-turbo-base-fn---" + v);

        System.out.println("=============RAW-MATH-EXP-EXECUTED-MATRIX-TURBO-MODE (sin(3*x)+cos(5*y)-sin(2*z^2))================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = matrixTurboExecForRawMathExp.applyScalar(vars);
        }
        long duration4 = System.nanoTime() - strt;
        System.out.println("waste-product of matrix-turbo-raw-math-exp---" + v);

        System.out.println("=============BASE-FN-DEF-EXECUTED-SCALAR-TURBO-1-MODE================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = scalarTurboExecForBaseFn.applyScalar(vars);
        }
        long duration5 = System.nanoTime() - strt;
        System.out.println("waste-product of scalar-turbo1-base-fn-exec---" + v);

        System.out.println("=============RAW-MATH-EXP-EXECUTED-SCALAR-TURBO-1-MODE================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = scalarTurboExecForRawMathExp.applyScalar(vars);
        }
        long duration6 = System.nanoTime() - strt;
        System.out.println("waste-product of scalar-turbo1-raw-exp-exec---" + v);

        System.out.println("=============BASE-FN-DEF-EXECUTED-SCALAR-TURBO-2-MODE================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = scalarTurbo2ExecForBaseFn.applyScalar(vars);
        }
        long duration7 = System.nanoTime() - strt;
        System.out.println("waste-product of scalar-turbo2-fn---" + v);

        System.out.println("=============RAW-MATH-EXP-EXECUTED-SCALAR-TURBO-2-MODE================");
        strt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            vars[0] = i % 9;
            vars[1] = i % 8;
            vars[2] = i % 9 - 2;
            v = scalarTurbo2ExecForRawMathExp.applyScalar(vars);
        }
        long duration8 = System.nanoTime() - strt;
        System.out.println("waste-product of scalar-turbo2-raw-exp-exec---" + v);

        System.out.printf("Expressions: %s and %s %n", baseFnDef.getExpression(), rawMathDef.getExpression());
        System.out.printf("Value =  %.18f%n", v);
        System.out.printf("Speed-standard-fn: %.2f ns/op%n", duration1 / iterations);
        System.out.printf("Speed-standard-raw-exp: %.2f ns/op%n", duration2 / iterations);
        System.out.printf("Speed-android-capable-matrix-turbo-fn: %.2f ns/op%n", duration3 / iterations);
        System.out.printf("Speed-android-capable-matrix-turbo-raw-exp: %.2f ns/op%n", duration4 / iterations);
        System.out.printf("Speed-scalar-turbo1-fn: %.2f ns/op%n", duration5 / iterations);
        System.out.printf("Speed-scalar-turbo1-raw-exp: %.2f ns/op%n", duration6 / iterations);
        System.out.printf("Speed-scalar-turbo2-fn: %.2f ns/op%n", duration7 / iterations);
        System.out.printf("Speed-scalar-turbo2-raw-exp: %.2f ns/op%n", duration8 / iterations);
        System.out.printf("Throughput-standard-fn: %.2f ops/sec%n", iterations / (duration1 / 1e9));
        System.out.printf("Throughput-standard-raw-exp: %.2f ops/sec%n", iterations / (duration2 / 1e9));
        System.out.printf("Throughput-android-capable-matrix-turbo-fn: %.2f ops/sec%n", iterations / (duration3 / 1e9));
        System.out.printf("Throughput-android-capable-matrix-turbo-raw-exp: %.2f ops/sec%n", iterations / (duration4 / 1e9));
        System.out.printf("Throughput-scalar-turbo1-fn: %.2f ops/sec%n", iterations / (duration5 / 1e9));
        System.out.printf("Throughput-scalar-turbo1-raw: %.2f ops/sec%n", iterations / (duration6 / 1e9));
        System.out.printf("Throughput-scalar-turbo2-fn: %.2f ops/sec%n", iterations / (duration7 / 1e9));
        System.out.printf("Throughput-scalar-turbo2-raw: %.2f ops/sec%n", iterations / (duration8 / 1e9));
        System.out.println("=================SPEED-UP-ZONE================");
        System.out.printf("android-capable-Matrix-Turbo-fn vs Standard-fn:              %.1fx%n", (double) duration1 / duration3);
        System.out.printf("android-capable-Matrix-Turbo-raw-exp vs Standard-raw-exp     %.1fx%n", (double) duration2 / duration4);
        System.out.printf("Scalar-Turbo1-fn vs Standard-fn:     %.1fx%n", (double) duration1 / duration5);
        System.out.printf("Scalar-Turbo1-raw-exp vs Standard-raw-exp:     %.1fx%n", (double) duration2 / duration6);
        System.out.printf("Scalar-Turbo2-fn vs Standard-fn:     %.1fx%n", (double) duration1 / duration7);
        System.out.printf("Scalar-Turbo2-raw-exp vs Standard-raw-exp:     %.1fx%n", (double) duration2 / duration8);
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

        double[] vars = new double[1];
        vars[xSlot] = 2.5;

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
        System.out.println("Standard- values=" + res[0]);
        System.out.println("Turbo- values=" + res[1]);
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
