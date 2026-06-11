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

import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator2;
import com.github.gbenroscience.parser.turbo.tools.TurboEvaluatorFactory;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.VariableManager;
import java.util.Arrays;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 *
 * @author GBEMIRO
 */
/**
 *
 * @author GBEMIRO Benchmarks for ScalarTurboCompiler vs Interpreted evaluation.
 * Tests basic arithmetic, trig functions, and complex expressions.
 */
public class ScalarTurboEvaluatorTest {

    public final int N = 1000000;

    @Test
    public void testPrinting() throws Throwable {
        String expr = "F=@(x,y,z)3*x+y-z^2";
        Function f = FunctionManager.add(expr);

        String ex = "z=3;x=pi;A=@(2,2)(4,2,-1,9);print(A,F,x,y,z)";
        System.out.printf("Expression: %s%n", ex);
        // Warm up JIT
        MathExpression interpreted = new MathExpression(ex, false);
        MathExpression.EvalResult ev = interpreted.solveGeneric();
        System.out.println("scanner: " + interpreted.getScanner());
        // Compile to turbo
        FastCompositeExpression compiled = interpreted.compileTurbo();
        System.out.println(compiled.getCompiler().getClass());
        // Warm up turbo JIT
        double[] vars = new double[3];
        MathExpression.EvalResult evr = compiled.apply(vars);
        Assertions.assertTrue(evr.textRes.contains("A") && evr.textRes.contains("F"));

    }

    @Test
    public void testIntegralCalculus() throws Throwable {
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "intg(@(x)(sin(x)+cos(x)), 1,2)";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);
        MathExpression.EvalResult ev = interpreted.solveGeneric();

        // Compile to turbo
        FastCompositeExpression compiled = interpreted.compileTurbo();
        // Warm up turbo JIT
        double[] vars = new double[0];
        MathExpression.EvalResult evr = compiled.apply(vars);

        Assertions.assertEquals(ev.scalar, evr.scalar);

    }

    @Test
    public void testComplexIntegralCalculus() throws Throwable {
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "intg(@(x)(1/(x*sin(x)+3*x*cos(x))), 0.5, 1.8)";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);
        MathExpression.EvalResult ev = interpreted.solveGeneric();

        // Compile to turbo
        FastCompositeExpression compiled = interpreted.compileTurbo();
        // Warm up turbo JIT
        double[] vars = new double[0];
        MathExpression.EvalResult evr = compiled.apply(vars);

        Assertions.assertEquals(ev.scalar, evr.scalar);
    }

    @Test
    public void testPointRotation() throws Throwable {
        System.out.println("-----------POINT ROTOR-------------");
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "rot(@(1,3)(0,2,0),pi,@(1,3)(0,0,0),@(1,3)(1,0,0))";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);
        MathExpression.EvalResult ev = interpreted.solveGeneric();

        // Compile to turbo
        FastCompositeExpression compiled = new ScalarTurboEvaluator1(interpreted).compile();//interpreted.compileTurbo();
        // Warm up turbo JIT
        double[] vars = new double[0];
        MathExpression.EvalResult evr = compiled.apply(vars);
        System.out.println("std: " + ev);
        System.out.println("tur: " + evr);

        Assertions.assertEquals(ev.scalar, evr.scalar);
    }

    @Test
    public void testLineRotation() throws Throwable {
        System.out.println("-----------LINE ROTOR-------------");
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "rot(@(1,3)(0,0,0),@(1,3)(0,10,0),pi/2,@(1,3)(0,0,0),@(1,3)(1,0,0))";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);
        MathExpression.EvalResult ev = interpreted.solveGeneric();
        System.out.println("std: " + ev);

        ScalarTurboEvaluator2 sc = new ScalarTurboEvaluator2(interpreted);
        // Compile to turbo
        FastCompositeExpression compiled = sc.compile();//interpreted.compileTurbo();
        // Warm up turbo JIT
        double[] vars = new double[0];
        MathExpression.EvalResult evr = compiled.apply(vars);
        System.out.println("tur: " + evr);

        Assertions.assertEquals(ev.scalar, evr.scalar);
    }

    @Test
    public void testFunctionRotation() throws Throwable {
        System.out.println("-----------FUNCTION ROTOR-------------");
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "rot(@(x,y,z)(x+y+z),pi,@(1,3)(0,0,0),@(1,3)(1,1,1))";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);
        System.out.println("sc_int:" + interpreted.getScanner());
        MathExpression.EvalResult ev = interpreted.solveGeneric();
        System.out.println("scanner: " + interpreted.getScanner());
        // Compile to turbo
        FastCompositeExpression compiled = new ScalarTurboEvaluator1(interpreted).compile();//interpreted.compileTurbo();
        // Warm up turbo JIT
        double[] vars = new double[0];
        MathExpression.EvalResult evr = compiled.apply(vars);
        System.out.println("std: " + ev);
        System.out.println("tur: " + evr);

        Assertions.assertEquals(ev.scalar, evr.scalar);
    }

    @Test
    public void testDiffCalculus() throws Throwable {
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "diff(@(x)(sin(x)+cos(x)), 2,1)";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);
        MathExpression.EvalResult ev = interpreted.solveGeneric();

        System.out.println("scanner--: " + interpreted.getScanner());

        // Compile to turbo
        FastCompositeExpression compiled = interpreted.compileTurbo();
        // Warm up turbo JIT
        double[] vars = new double[0];
        MathExpression.EvalResult evr = compiled.apply(vars);
        System.out.println("std: " + ev);
        System.out.println("tur: " + evr);

        Assertions.assertEquals(ev.scalar, evr.scalar);
    }

    @Test
    public void testDiffCalculusGradientFunction() throws Throwable {
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "diff(@(x)(sin(x)+cos(x)),1)";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);
        MathExpression.EvalResult ev = interpreted.solveGeneric();
        System.out.println("std: " + ev);

        // Compile to turbo
        FastCompositeExpression compiled = interpreted.compileTurbo();
        // Warm up turbo JIT
        double[] vars = new double[0];
        MathExpression.EvalResult evr = compiled.apply(vars);
        System.out.println("tur: " + evr);

        Assertions.assertTrue(ev.toString().startsWith("anon") && evr.toString().startsWith("anon"));
    }

    @Test
    public void testDiffCalculusGradientFunctionWithReturnHandle() throws Throwable {
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "diff(@(x)(sin(x)+cos(x)),v,1)";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);
        MathExpression.EvalResult ev = interpreted.solveGeneric();
        System.out.println("std: " + ev);

        System.out.println("scanner: "+interpreted.getScanner());
        System.out.println("postfix: "+Arrays.deepToString(interpreted.getCachedPostfix()));
        // Compile to turbo
        FastCompositeExpression compiled = new ScalarTurboEvaluator1(interpreted).compile();// interpreted.compileTurbo();
        // Warm up turbo JIT
        double[] vars = new double[0];
        MathExpression.EvalResult evr = compiled.apply(vars);
        System.out.println("tur: " + evr);

        Assertions.assertTrue(ev.toString().equals(evr.toString()));
    }

    @Test
    public void testBasicArithmetic() throws Throwable {
        String expr = "2 + 3 * 4 - 5 / 2 + 1 ^ 3";

        MathExpression interpreted = new MathExpression(expr, false);

        // Compile to turbo
        MathExpression turbo = new MathExpression(expr, false);
        FastCompositeExpression compiled = turbo.compileTurbo();

        double v = interpreted.solveGeneric().scalar;
        double[] vars = new double[0];
        double v1 = compiled.applyScalar(vars);
        Assertions.assertEquals(v, v1);

    }

    @Test
    public void testSum() throws Throwable {
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "listsum(12,1,23,5,13,2,20,30,40,1,1,1,2)";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);

        FastCompositeExpression compiled = interpreted.compileTurbo();

        double v = interpreted.solveGeneric().scalar;
        double[] vars = new double[0];
        double v1 = compiled.applyScalar(vars);
        Assertions.assertEquals(v, v1);
    }

    @Test
    public void testSort() throws Throwable {
        //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "sort(12,1,23,5,13,2,20,30,40,1,1,1,2)";

        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);

        FastCompositeExpression compiled = interpreted.compileTurbo();

        double[] v = interpreted.solveGeneric().vector;
        double[] vars = new double[0];
        double[] v1 = compiled.apply(vars).vector;

        Assertions.assertArrayEquals(v, v1);

    }

    @Test
    public void testTrigonometric() throws Throwable {
        String expr = "sin(3.14159/2) + cos(1.5708) * tan(0.785398)";

        MathExpression interpreted = new MathExpression(expr, false);

        double v = interpreted.solveGeneric().scalar;
        TurboExpressionEvaluator tee = TurboEvaluatorFactory.getCompiler(interpreted);
        FastCompositeExpression fce = tee.compile();
        double[] vars = new double[0];

        double v1 = fce.applyScalar(vars);

        Assertions.assertEquals(v, v1);
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    public void testComplexExpression(boolean withFolding) throws Throwable {
        System.out.println("\n=== COMPLEX EXPRESSION " + (withFolding ? "WITH FOLDING" : "WITHOUT FOLDING") + " ===\n");

        String expr = "sqrt(16) + 2^3 * sin(0) + 3! - cos(-5) + ln(2.718281828)";

        MathExpression interpreted = new MathExpression(expr, false);

        double v = interpreted.solveGeneric().scalar;
        TurboExpressionEvaluator tee = TurboEvaluatorFactory.getCompiler(interpreted);
        FastCompositeExpression fce = tee.compile();
        double[] vars = new double[0];

        double v1 = fce.applyScalar(vars);

        Assertions.assertEquals(v, v1);
    }

    @Test
    public void testWithVariablesSimple() throws Throwable {

        String expr = "x*sin(x)+2";

        MathExpression interpreted = new MathExpression(expr, false);
        int xSlot = interpreted.getVariable("x").getFrameIndex();

        double[] vars = new double[3];
        vars[xSlot] = 2.5;

        double v = interpreted.solveGeneric(2.5).scalar;
        TurboExpressionEvaluator tee = TurboEvaluatorFactory.getCompiler(interpreted);
        FastCompositeExpression fce = tee.compile();

        double v1 = fce.applyScalar(vars);

        Assertions.assertEquals(v, v1);
    }

    @Test
    public void testWithVariablesAdvanced() throws Throwable {

        String expr = "x=0;y=0;z=0;x*sin(x) + y*sin(y) + z / cos(x - y) + sqrt(x^2 + y^2)";
        MathExpression interpreted = new MathExpression(expr, false);

        int xSlot = interpreted.getVariable("x").getFrameIndex();
        int ySlot = interpreted.getVariable("y").getFrameIndex();
        int zSlot = interpreted.getVariable("z").getFrameIndex();

        double[] vars = new double[3];
        vars[xSlot] = 2.5;
        vars[ySlot] = 3.7;
        vars[zSlot] = 1.2;
        interpreted.updateSlot(xSlot, 2.5);
        interpreted.updateSlot(ySlot, 3.7);
        interpreted.updateSlot(zSlot, 1.2);
        double v = interpreted.solveGeneric().scalar;
        TurboExpressionEvaluator tee = TurboEvaluatorFactory.getCompiler(interpreted);
        FastCompositeExpression fce = tee.compile();
        double v1 = fce.applyScalar(vars);
        Assertions.assertEquals(v, v1);
    }

    @Test
    public void testConstantFolding() throws Throwable {

        String expr = "2^10 + 3^5 - 4! + sqrt(256)";
        MathExpression interpreted = new MathExpression(expr, false);

        double v = interpreted.solveGeneric().scalar;
        interpreted.setWillFoldConstants(true); // Enable optimization
        TurboExpressionEvaluator tee = TurboEvaluatorFactory.getCompiler(interpreted);
        FastCompositeExpression fce = tee.compile();
        double[] vars = new double[0];

        double v1 = fce.applyScalar(vars);

        Assertions.assertEquals(v, v1);
    }

    @Test
    public void testQuadratic() throws Throwable {

        String expr = "quadratic(@(x)3*x^2-4*x-18)";

        MathExpression interpreted = new MathExpression(expr, false);

        double[] vars = new double[0];

        double[] v = interpreted.solveGeneric().vector;
        TurboExpressionEvaluator tee = TurboEvaluatorFactory.getCompiler(interpreted);
        FastCompositeExpression fce = tee.compile();
        double[] v1 = fce.apply(vars).vector;
        Assertions.assertTrue(Arrays.toString(v).equals(Arrays.toString(v1)));
    }

    @Test
    public void testTartaglia() throws Throwable {

        String expr = "t_root(@(x)3*x^3-4*x-18)";

        MathExpression interpreted = new MathExpression(expr, false);

        double[] vars = new double[0];

        double[] v = interpreted.solveGeneric().vector;
        TurboExpressionEvaluator tee = TurboEvaluatorFactory.getCompiler(interpreted);
        FastCompositeExpression fce = tee.compile();
        double[] v1 = fce.applyVector(vars);

        Assertions.assertTrue(Arrays.toString(v).equals(Arrays.toString(v1)));
    }

    @Test
    public void testGeneralRoot() throws Throwable {//-1.8719243686213027618871370090528, -3.2052577019546360952204703423861

        String expr = "root(@(x)3*x^3-4*x-18,2)";
        MathExpression interpreted = new MathExpression(expr, false);

        double[] vars = new double[0];

        double v = interpreted.solveGeneric().scalar;
        TurboExpressionEvaluator tee = TurboEvaluatorFactory.getCompiler(interpreted);
        FastCompositeExpression fce = tee.compile();
        double v1 = fce.applyScalar(vars);

    }

}
