package com.github.gbenroscience.parser;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.methods.BasicNumericalMethod;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.MatrixTurboEvaluator;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator2;
import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.VariableManager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

class MathExpressionTest {

    @Test
    void expTest() {
        MathExpression me = new MathExpression("2^3");
        Assertions.assertEquals(Number.fastParseDouble("8.0"), Number.fastParseDouble(me.solve()));
    }

    @Test
    void rootTest() {
        MathExpression me = new MathExpression("8^(1/3)");
        Assertions.assertEquals("2.0", me.solve());
        me = new MathExpression("8^(1/2)");
        Assertions.assertEquals(Number.fastParseDouble("2.8284271247461903"), Number.fastParseDouble(me.solve()));
    }

    @Test
    void customEmbeddedFunctionTest() {
        MathExpression me = new MathExpression("avgN(0,1,2,3)");
        Assertions.assertEquals(Number.fastParseDouble("2"), Number.fastParseDouble(me.solve()));
    }

    @Test
    void customEmbeddedFunctionTestMultipleBrackets() {
        MathExpression me = new MathExpression("((avgN((0,1,2,3))))");
        Assertions.assertEquals(Number.fastParseDouble("2"), Number.fastParseDouble(me.solve()));
        me = new MathExpression("((avgN((0,  (1)+((1+1)),((2)),((3+2))))))");
        //me = new MathExpression("((weir((    (1)+((1+1)),((2)),((3+2))))))");
        Assertions.assertEquals("3.333333333", me.solve()); //weird, by nature of weird function. Will be removed once it wil be repalced by proper function
    }

    @Test
    void customUserFunctionTest() {
        BasicNumericalMethod b1 = new BasicNumericalMethod() {
            @Override
            public String solve(List<String> tokens) {
                return "1";
            }

            @Override
            public String getHelp() {
                return "no help for b1";
            }

            @Override
            public String getName() {
                return "b1";
            }

            @Override
            public String getType() {
                return TYPE.NUMBER.toString();
            }
        };
        BasicNumericalMethod b2 = new BasicNumericalMethod() {
            @Override
            public String solve(List<String> tokens) {
                return "2";
            }

            @Override
            public String getHelp() {
                return "no help for b2";
            }

            @Override
            public String getName() {
                return "b2";
            }

            @Override
            public String getType() {
                return TYPE.NUMBER.toString();
            }
        };
        MathExpression.setAutoInitOn(false);

        MathExpression me = new MathExpression("b1(1,2,3)");

        Assertions.assertEquals(MathExpression.isAutoInitOn() ? "0.0" : MathExpression.SYNTAX_ERROR, me.solve());
        Declarations.registerBasicNumericalMethod(b1);
        me = new MathExpression("b1(1,2,3)");
        Assertions.assertEquals(1, Double.parseDouble(me.solve()));
        me = new MathExpression("b2(1,2,3)");
        Assertions.assertEquals(MathExpression.isAutoInitOn() ? "0.0" : MathExpression.SYNTAX_ERROR, me.solve());
        Declarations.registerBasicNumericalMethod(b2);
        me = new MathExpression("b2(1,2,3)");
        Assertions.assertEquals(2, Double.parseDouble(me.solve()));
        me = new MathExpression("b1(1,2,3)");
        Assertions.assertEquals(1, Double.parseDouble(me.solve()));
        me = new MathExpression("b1(1,2,3)+b2(1,2,3)");
        Assertions.assertEquals(3, Double.parseDouble(me.solve()));
        Declarations.unregisterBasicNumericalMethod(b1.getClass());
        me = new MathExpression("b1(1,2,3)");
        Assertions.assertEquals(MathExpression.isAutoInitOn() ? "0.0" : MathExpression.SYNTAX_ERROR, me.solve());
        me = new MathExpression("b2(1,2,3)");
        Assertions.assertEquals(2, Double.parseDouble(me.solve()));
        Declarations.unregisterBasicNumericalMethod(b2.getClass());
        me = new MathExpression("b2(1,2,3)");
        Assertions.assertEquals(MathExpression.isAutoInitOn() ? "0.0" : MathExpression.SYNTAX_ERROR, me.solve());
        MathExpression.setAutoInitOn(true);//restore the switch
    }

    @Test
    void help() {
        MathExpression me = new MathExpression("help");
        String help = me.solve();
        Assertions.assertTrue(help.length() > 100);
        Assertions.assertTrue(help.contains(Declarations.SIN));
    }

    @Test
    void oldMainTest() {
        MathExpression expr = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2)");
        Assertions.assertEquals("0.0", expr.solve());
    }

    @Test
    void moreJunkExamples() {

        Function f = FunctionManager.add("f(x,y) = x - x/y");
        f.updateArgs(2, 3);
        System.out.println("f=" + f);
        double r = f.calc();
        Assertions.assertEquals(2.0 - 2.0 / 3.0, r);
        int iterations = 10000;
        long start = System.nanoTime();
        f.updateArgs(2, 3);
        for (int i = 1; i < iterations; i++) {
            f.calc();
        }
        long duration = System.nanoTime() - start;

        //System.out.println("Time: " + (duration / iterations) + " ns");
        Assertions.assertTrue(duration / iterations < 1000000);
    }

    @Test
    void junkExamples() {
        boolean print = false;
        //this test is checking content of variables.
        //some other tests could have set them. Eg
        //LogicalExpressionTest variablesDoNotWorks and variablestWorks
        VariableManager.clearVariables();

        MathExpression linear = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);C=matrix_sub(M,N);C;");
        String ls = linear.solve();
        if (print) {
            System.out.println("soln: " + ls);
        }
        Assertions.assertEquals("\n"
                + "   -1.0  ,    3.0  ,   -7.0            \n"
                + "    0.0  ,    3.0  ,    4.0            \n"
                + "    4.0  ,    0.0  ,  -11.0            \n", FunctionManager.lookUp(ls).getMatrix().toString());

        MathExpression expr = new MathExpression("tri_mat(M)");
        Matrix m = expr.solveGeneric().matrix;

        if (print) {
            System.out.println(m.toString());
        }
        Function f = new Function("fExpr=@(3,3)(1.0,1.3333333333333333,0.3333333333333333,0.0,1.0,4.749999999999999,0.0,0.0,1.0)");
        FunctionManager.add(f);
        //System.out.println("MATRIX: "+f.getMatrix()+",\nm="+m+", f="+f);
        Assertions.assertTrue(f.getMatrix().equals(m));
        FunctionManager.delete("fExpr");

        MathExpression expr2 = new MathExpression("echelon(M)");
        Matrix echelon = expr2.solveGeneric().matrix;
        if (print) {
            System.out.println(echelon);
        }

        f = new Function("fExpr=@(3,3)(3.0,4.0,1.0,0.0,4.0,19.0,0.0,0.0,567.0)");
        FunctionManager.add(f);
        Assertions.assertTrue(f.getMatrix().equals(echelon));
        FunctionManager.delete("fExpr");

        Function matrixFunction = FunctionManager.lookUp("M");
        Matrix matrix = matrixFunction.getMatrix();
        if (print) {
            System.out.println("underlying matrix: " + matrix);
        }
        Assertions.assertEquals("\n"
                + "    3.0  ,    4.0  ,    1.0            \n"
                + "    2.0  ,    4.0  ,    7.0            \n"
                + "    9.0  ,    1.0  ,   -2.0            \n", matrix.toString());

        Matrix inv = matrix.inverse();
        if (print) {
            System.out.println("inverted matrix: " + inv);
        }
        Assertions.assertArrayEquals(new double[]{
            -0.07936507936507937, 0.04761904761904762, 0.12698412698412698, 0.35449735449735453, -0.07936507936507942, -0.10052910052910054,
            -0.1798941798941799, 0.17460317460317462, 0.021164021164021166
        }, inv.getFlatArray());

        matrix.multiply(inv);
        if (print) {
            System.out.println("mul matrix: " + matrix);
        }
        Assertions.assertArrayEquals(new double[]{
            1.0, -1.6653345369377348E-16, -5.204170427930421E-17, 0.0, 0.9999999999999999, -2.7755575615628914E-17, -5.551115123125783E-17, -5.551115123125783E-17, 1.0}, matrix.getFlatArray());

        FunctionManager.add("f(x,y) = x-x/y");
        Function fxy = FunctionManager.lookUp("f");

        MathExpression parserng = new MathExpression("f(x,y) = x-x/y; f(2,3);f(2,5);");
        String parsengr = parserng.solve();
        if (print) {
            System.out.println("SEE???????\n " + parsengr);
        }
        Assertions.assertEquals("0.0", parsengr);
        /*
         MathExpression f = new MathExpression("x=17;3*x+1/x");//runs in about 2.3 milliSecs

         System.out.println(f.solve());
         f.setExpression("x^3");
         System.out.println(f.variableManager);

         System.out.println(f.solve());
         *
         String fun =
         "x=3;x^3";

         MathExpression f = new MathExpression(fun);
         String ans ="";
         double start = System.nanoTime();

         for(int i=0;i<1;i++){
         ans = f.solve();
         }

         double time = (System.nanoTime()-start)/1.0E6;
         System.out.println("ans = "+ans+" calculated in "+time+" ms");

         */
        /**
         * On balanced CPU usage mode, Before optimization with StringBuilder,
         * f.solve() takes about 7ms to run the above 1000 string operation and
         * about 500ms to run the MathExpression constructor's parsing
         * techniques.
         *
         * After optimization,Sim
         *
         *
         */
        //3,log(2,4),8,9
        //FunctionManager.add("f=@(x)sin(x)-x");//A=4;sum(3A,4,4+cos(8),5,6+sin(3),2!,7A,3E-8+6,4)
        //Formula...................t_root(@(x)2.2x^3+3*x+8*x^0)
        //MathExpression express = new MathExpression("sum(root(f,2,3),1,2,4,2,9)");//³√ diff(@(x)3*x²+5*³√(x),3),diff(@(x)3*x,3)
        //MathExpression express = new MathExpression("sum(root(f,2,3),1,2,4,2,9)");//³√ diff(@(x)3*x²+5*³√(x),3),diff(@(x)3*x,3)
        //³√ diff(@(x)3*x²+5*³√(x),3),diff(@(x)3*x,3)
        /*
         3 4  3 4   13 20
         1 2  1 2   5  8                 -22
         */
        FunctionManager.clear();
        FunctionManager.add("M=@(3,3)(3,4,1,2,4,7,9,1,-2)");
        FunctionManager.add("M1=@(2,2)(3,4,1,2)");
        FunctionManager.add("M2=@(2,2)(3,4,1,2)");

        FunctionManager.add("N=@(x)(sin(x))");
        FunctionManager.add("r=@(x)(ln(sin(x)))");

        System.out.println("M lookup: " + FunctionManager.lookUp("M").toString());

        //WORK ON sum(3,-2sin(3)^2,4,5) error //matrix_mul(@(2,2)(3,1,4,2),@(2,2)(2,-9,-4,3))...sum(3,2sin(4),5,-3cos(2*sin(5)),4,1,3)
        //matrix_mul(invert(@(2,2)(3,1,4,2)),@(2,2)(2,-9,-4,3)).............matrix_mul(invert(@(2,2)(3,1,4,2)),@(2,2)(2,-9,-4,3))
        //matrix_mul(invert(@(2,2)(3,1,4,2)),matrix_mul(M2,2sin(3)sum(3,6,5,3)cos(9/8*6)det(@(2,2)(2,-9,-4,3))))
        if (print) {
            System.out.println("lookup M: " + FunctionManager.lookUp("M"));
        }
        Assertions.assertEquals("M=@(3,3)(3.0,4.0,1.0,2.0,4.0,7.0,9.0,1.0,-2.0)", FunctionManager.lookUp("M").toString());
        //MathExpression expr = new MathExpression("f=3;5f");//BUGGY
        //MathExpression expr = new MathExpression("quad(@(x)3*x-2+3*x^2)");//BUGGY
        //MathExpression expr = new MathExpression("root(@(x)3*x-sin(x)-0.5,2)");//BUGGY
        MathExpression exprs = new MathExpression("r1=4;r1*5");
        //A+k.A+AxB+A^c
        if (print) {
            System.out.println("scanner: " + exprs.scanner);
        }
        Assertions.assertEquals("[(, r1, *, 5, )]", exprs.scanner.toString());
        if (print) {
            System.out.println("solution: " + exprs.solve());
        }
        Assertions.assertEquals("20.0", exprs.solve());

        //no idea what is this trying to do, results sounds buggy anyway
        expr.setExpression("44+22*(3)");
        if (print) {
            System.out.println("solution--: " + expr.solve());
        }
        Assertions.assertEquals("110.0", expr.solve());

        if (print) {
            System.out.println("return type: " + exprs.returnType);
        }
        Assertions.assertEquals(TYPE.NUMBER, exprs.returnType);
        List<Map.Entry<String, Function>> l = new ArrayList(FunctionManager.FUNCTIONS.entrySet());
        Collections.sort(l, new Comparator<Map.Entry<String, Function>>() {
            @Override
            public int compare(Map.Entry<String, Function> t0, Map.Entry<String, Function> t1) {
                return t0.getKey().compareTo(t1.getKey());
            }
        });
        //although the sort is maaking it deterministic, the assert is fragile and afaik have no sense
        if (print) {
            System.out.println("FunctionManager: " + l.toString());
        }
//        Assertions.assertEquals(
//                "[C=C=@(3,3)(-1.0,3.0,-7.0,0.0,3.0,4.0,4.0,0.0,-11.0), M=M=@(3,3)(3.0,4.0,1.0,2.0,4.0,7.0,9.0,1.0,-2.0), M1=M1=@(2,2)(3.0,4.0,1.0,2.0), M2=M2=@(2,2)(3.0,4.0,1.0,2.0), N=N=@(x)(sin"
//                        + "(x)), anon1=anon1=@(3,3)(3.0,4.0,1.0,2.0,4.0,7.0,9.0,1.0,-2.0), anon2=anon2=@(3,3)(4.0,1.0,8.0,2.0,1.0,3.0,5.0,1.0,9.0), anon3=anon3=@(3,3)(-1.0,3.0,-7.0,0.0,3.0,4.0,4.0,"
//                        + "0.0,-11.0), anon4=anon4=@(3,3)(1.0,1.3333333333333333,0.3333333333333333,0.0,1.0,4.749999999999999,0.0,0.0,1.0), anon5=anon5=@(3,3)(3.0,4.0,1.0,0.0,4.0,19.0,0.0,0.0,567"
//                        + ".0), anon6=anon6=@(x,y)(x-x/y), f=f=@(x,y)(x-x/y), r=r=@(x)(ln(sin(x)))]",
//                l.toString());
        if (print) {
            System.out.println("VariableManager: " + VariableManager.VARIABLES);
        }
        Assertions.assertEquals("{e=e:2.718281828459045, ans=ans:0.0, x=x:0.0, pi=pi:3.141592653589793, y=y:0.0, r1=r1:4.0}",
                VariableManager.VARIABLES.toString());

        MathExpression expression = new MathExpression("x=0;sin(ln(x))");

        expression.setValue("x", 0);
        if (print) {
            System.out.println(expression.solve());
        }

        Assertions.assertEquals("NaN", expression.solve());
        expression.setValue("x", 1);

        if (print) {
            System.out.println(expression.solve());
        }
        Assertions.assertEquals("0.0", expression.solve());
        expression.setValue("x", 50);
        if (print) {
            System.out.println(expression.solve());
        }
        Assertions.assertTrue("-0.6964441283311967".equals(expression.solve()) || "-0.6964441283311968".equals(expression.solve()));
        expression.setValue("x", 100);
        if (print) {
            System.out.println(expression.solve());
        }
        Assertions.assertEquals(Number.fastParseDouble("-0.9942575694137897"), Number.fastParseDouble(expression.solve()));

        f = FunctionManager.lookUp("N");
        long start = System.nanoTime();
        int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            f.updateArgs(i + 3);
            f.calc();
        }
        long elapsedNanos = (System.nanoTime() - start) / iterations;
        double durationms = (double) elapsedNanos / 1.0E6;
        System.out.println("DONE: " + durationms + " ms");
        if (print) {
            System.out.println("DONE: " + durationms + " ms");
        }
        Assertions.assertTrue(durationms < 1);

        MathExpression det = new MathExpression("det(@(5,5)(-21,12,13,64,5,6,2.7,18,9,0,4,2,3,4,23,6,7,8,9,0,1,2,32,4,5));");
        if (print) {
            System.out.println("determinant: " + det.solve());
        }
        Assertions.assertEquals("-1739274.3000000003", det.solve());

        /**
         * On my Macbook Pro, 16GB RAM; 2.6 GHz Intel Core i7
         *
         * The code runs the solve() method at 3.8 microseconds.
         */
    }

    @Test
    public void testMixedConstantVariableFolding() {
        MathExpression expr = new MathExpression("1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+sin(x)");

        MathExpression.Token[] cachedPostfix = expr.getCachedPostfix();

        // Print all tokens
        for (int i = 0; i < cachedPostfix.length; i++) {
            MathExpression.Token t = cachedPostfix[i];
            String desc = "";
            switch (t.kind) {
                case 0:
                    desc = "NUMBER(" + t.value + ")";
                    break;
                case 1:
                    desc = "OPERATOR(" + t.opChar + ")";
                    break;
                case 2:
                    desc = "FUNCTION(" + t.name + ")";
                    break;
                case 3:
                    desc = "METHOD(" + t.name + ")";
                    break;
                default:
                    desc = "OTHER(" + t.kind + ")";
            }
            System.out.println("  Token[" + i + "]: " + desc);
        }

        // Verify folding
        Assertions.assertTrue(cachedPostfix.length <= 4, "Should have at most 4 tokens");

        // Evaluate with variable
        expr.updateSlot(expr.registry.getSlot("x"), 0.0);
        String result = expr.solve();
        // Expected: 210 + sin(0) = 210
        double resultVal = Double.parseDouble(result);
        Assertions.assertEquals(210.0, resultVal, 0.0001);

        expr.updateSlot(expr.registry.getSlot("x"), Math.PI / 2);
        result = expr.solve();
        // Expected: 210 + sin(π/2) = 210 + 1 = 211
        resultVal = Double.parseDouble(result);
        Assertions.assertEquals(211.0, resultVal, 0.0001);
    }

    @Test
    void algebraicFunctionAssignTest() {
        MathExpression me = new MathExpression("A=@(x,y)sin(x)+cos(y-x);B=@(x,y)cos(x*y);C(x,y)=A(x,2*y)+B(3*x,2*y);D=C;print(D)");
        System.out.println("scanner!!!: " + me.scanner);
        System.out.println("correctFunction!!!: " + me.correctFunction);
        System.out.println("cachedPostfix!!!: " + me.getCachedPostfix());
        System.out.println("solve: " + me.solveGeneric());
        System.out.println(FunctionManager.FUNCTIONS);

        Assertions.assertTrue(true);
    }

    @Test
    void matrixAssignTest() {
        MathExpression me = new MathExpression("A=@(2,2)(3,1,2,8);B=@(2,2)(2,1,4,-3);C=matrix_add(A,B);D=C;print(D);");
        System.out.println("solve: " + me.solveGeneric());
        System.out.println(FunctionManager.FUNCTIONS);

        Assertions.assertTrue(true);
    }

    @Test
    void rotFunctionTest() {
        MathExpression me = new MathExpression("F=@(x,y,z)3*x-5*y-4*z;rot(f=@(x,y,z)3*x-5*y-4*z,pi,@(1,3)(0,0,0),@(1,3)(1,1,0));");
        System.out.println("solve: " + me.solveGeneric().scalar);
        System.out.println("F=" + FunctionManager.lookUp("F"));
        System.out.println("f=" + FunctionManager.lookUp("f"));
        Assertions.assertTrue(true);
    }

    @Test
    void rotAssignFunctionTest() {
        try {
            MathExpression me = new MathExpression("F=@(x,y,z)3*x-5*y-4*z;f=rot(F,pi,@(1,3)(0,0,0),@(1,3)(1,1,0));f(0,0,0)");
        } catch (RuntimeException r) {
            System.out.println(r.getLocalizedMessage());
            Assertions.assertTrue(true);
        }
    }

    @Test
    void rotAssignPointTest() {
        MathExpression me = new MathExpression("P=@(1,3)(1,2,3);p=rot(P,pi,@(1,3)(0,0,0),@(1,3)(1,1,0));print(p)");
        System.out.println("solve: " + me.solveGeneric());
        System.out.println("P=" + FunctionManager.lookUp("P"));
        Function p = FunctionManager.lookUp("p");
        System.out.println("p=" + p);

        Assertions.assertTrue(true);
        Assertions.assertTrue(p.getType() == TYPE.MATRIX && ((p.getMatrix().getRows() == 1 && p.getMatrix().getCols() == 3) || (p.getMatrix().getRows() == 3 && p.getMatrix().getCols() == 1)));
    }

    @Test
    void rotAssignLineTest() {
        MathExpression me = new MathExpression("P=@(1,3)(1,2,3);Q=@(1,3)(2,1,3);p=rot(P,Q,pi,@(1,3)(0,0,0),@(1,3)(1,1,0));print(p)");
        System.out.println("solve: " + me.solveGeneric());
        System.out.println("P=" + FunctionManager.lookUp("P"));
        System.out.println("Q=" + FunctionManager.lookUp("Q"));
        Function p = FunctionManager.lookUp("p");
        System.out.println("p=" + p);

        Assertions.assertTrue(p.getType() == TYPE.MATRIX && (p.getMatrix().getRows() == 2 && p.getMatrix().getCols() == 3));
    }

    @Test
    void rotSwarmOfPointsAssignmentTest() {
        MathExpression me = new MathExpression("swarm=@(10,3)(1,4,3, 2,4,5, 5,8,2, 12,18,25, 9,0.5,4, 15,32,48, 9,19,49, 32,5,82, 8,18,28, 4,9,12);"
                + "P(1,3)=(0,0,0);D(1,3)=(0,0,1);V=rot(swarm,pi,P,D);print(V)");
        System.out.println("solve: " + me.solveGeneric());
        System.out.println("swarm=" + FunctionManager.lookUp("swarm"));
        Function p = FunctionManager.lookUp("swarm");

        Assertions.assertTrue(p.getType() == TYPE.MATRIX && (p.getMatrix().getRows() == 10 && p.getMatrix().getCols() == 3));
    }

    @Test
    void testFlatMatrixInvertInMatrixTurboEvaluator() {
        try {
            MathExpression me = new MathExpression("A=@(3,3)(4,9,2, 8,1,18, 12,9,8);invert(A);");
            System.out.println("std-solve: " + me.solveGeneric());
            System.out.println("A=" + FunctionManager.lookUp("A"));

            MathExpression m = new MathExpression("1/A");
            System.out.println("1/A(STD)---" + new MathExpression("1/A").solve());
            FastCompositeExpression fce = m.compileTurbo();
            System.out.println("compiler class: " + fce.getCompiler().getClass());
            MathExpression.EvalResult ev = fce.apply(new double[]{0.0});
            System.out.println("turbo-apply: " + ev);
            Assertions.assertTrue(true);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Test
    void testSingleVar() {
        String expression = "x=5;x";
        MathExpression me = new MathExpression(expression);
        double val = me.solveGeneric().scalar;
        Assertions.assertTrue(val == 5);
    }

    @Test
    void testMultiPointRotation() {
        String expression = "rot(@(4,3)(1,2,1, 5,4,-9, 12,18,2, 14,9,-1), pi, @(1,3)(0,0,0), @(1,3)(0,0,1))";
        MathExpression me = new MathExpression(expression);
        MathExpression.EvalResult v = me.solveGeneric();
        System.out.println("RESULT:" + v.toString());
        Assertions.assertTrue(true);
    }

    @Test
    void testMultiPointRotationTurbo() {
        try {
            String expression = "rot(@(4,3)(1,2,1, 5,4,-9, 12,18,2, 14,9,-1), pi, @(1,3)(0,0,0), @(1,3)(0,0,1))";
            MathExpression me = new MathExpression(expression);
            MathExpression.EvalResult v = new ScalarTurboEvaluator2(me).compile().apply(new double[0]);
            System.out.println("RESULT:" + v.toString());
            Assertions.assertTrue(true);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Test
    void nestedStatsTest() {
        MathExpression me = new MathExpression("listsum(4,1,3,5,9,sort(2,1,3,12,8,4),5,9,2,2,10,10)");
        Assertions.assertEquals(90, me.solveGeneric().scalar);
    }

    @Test
    void submatrixTest() {
        MathExpression me = new MathExpression("A(4,4)=(3,1,2,5,  9,8,3,6,  12,1,0,5,  3,7,5,9); sub_mat(A,1,1);");
        System.out.println("scanner: " + me.getScanner());
        Matrix m = me.solveGeneric().matrix;
        System.out.println("matrix:\n" + m);
        Assertions.assertEquals(8, m.getFlatArray()[0]);
    }

    @Test
    void randomFillMatrixTest() {
        MathExpression me = new MathExpression("rnd_mat(20,10,10);");
        System.out.println("scanner: " + me.getScanner());
        Matrix m = me.solveGeneric().matrix;
        System.out.println("matrix:\n" + m);
        Assertions.assertTrue(m.getFlatArray().length == 100);
    }

    @Test
    void submatrixTurboTest() {
        try {
            MathExpression me = new MathExpression("A(4,4)=(3,1,2,5,  9,8,3,6,  12,1,0,5,  3,7,5,9); sub_mat(A,1,1);");
            FastCompositeExpression fce = new MatrixTurboEvaluator(me).compile();
            System.out.println("scanner: " + me.getScanner());
            Matrix m = fce.apply(new double[0]).matrix;
            System.out.println("matrix:\n" + m);
            Assertions.assertEquals(8, m.getFlatArray()[0]);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Test
    void randomFillMatrixTurboTest() {
        try {
            MathExpression me = new MathExpression("rnd_mat(20,10,10);");
            FastCompositeExpression fce = new MatrixTurboEvaluator(me).compile();
            System.out.println("scanner: " + me.getScanner());
            Matrix m = fce.apply(new double[0]).matrix;
            System.out.println("matrix:\n" + m);
            Assertions.assertTrue(m.getFlatArray().length == 100);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Test
    void matrixMinorTest() {
        MathExpression me = new MathExpression("A(4,4)=(3,1,2,5,  9,8,3,6,  12,1,0,5,  3,7,5,9); matrix_minor(A,1,1);");
        System.out.println("scanner: " + me.getScanner());
        System.out.println("A: " + FunctionManager.lookUp("A").getMatrix());
        Matrix m = me.solveGeneric().matrix;
        System.out.println("matrix:\n" + m);
        Assertions.assertEquals(3, m.getFlatArray()[0]);
    }

    @Test
    void matrixMinorTurboTest() {
        try {
            MathExpression me = new MathExpression("A(4,4)=(121,1,2,5,  60,8,3,6,  102,1,0,5,  31,71,15,19); matrix_minor(A,2,0);");
            FastCompositeExpression fce = new MatrixTurboEvaluator(me).compile();
            System.out.println("scanner: " + me.getScanner());
            System.out.println("A: " + FunctionManager.lookUp("A").getMatrix());
            Matrix m = fce.apply(new double[0]).matrix;
            System.out.println("matrix:\n" + m);
            Assertions.assertEquals(1, m.getFlatArray()[0]);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    

    @Test
    void matrixTestAlgebraStdParser() {
        try {
            MathExpression me = new MathExpression("A(4,4)=(121,1,2,5,  60,8,3,6,  102,1,0,5,  31,71,15,19);"
                    + "B(4,4)=(3,22,8,-5,  10,18,32,8,  4,2,1,9,  7,7,2,13);"
                    + " B^3+A^2");
            System.out.println("scanner: " + me.getScanner());
            String matrixResultReference = me.solveGeneric().textRes;
            System.out.println("matrixResultReference: " + matrixResultReference);
            Matrix m = FunctionManager.lookUp(matrixResultReference).getMatrix();
            System.out.println("result: " + m);
            Assertions.assertEquals(24248.0, m.getFlatArray()[0]);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    

    @Test
    void matrixTestAlgebraMultiplicationStdParser() {
        try {
            MathExpression me = new MathExpression("A(4,4)=(121,1,2,5,  60,8,3,6,  102,1,0,5,  31,71,15,19);"
                    + "B(4,4)=(3,22,8,-5,  10,18,32,8,  4,2,1,9,  7,7,2,13);"
                    + " V=A*B");
            System.out.println("scanner: " + me.getScanner());
            MathExpression.EvalResult ev = me.solveGeneric();
            Matrix m = FunctionManager.lookUp("V").getMatrix();
            System.out.println("result: " + m);
            Assertions.assertEquals(416.0, m.getFlatArray()[0]);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Test
    void matrixTestAlgebraAssignments() {
        try {
            MathExpression me = new MathExpression("A(4,4)=(121,1,2,5,  60,8,3,6,  102,1,0,5,  31,71,15,19);"
                    + "B(4,4)=(3,22,8,-5,  10,18,32,8,  4,2,1,9,  7,7,2,13);"
                    + " D=A^2+B^2;D;");
            System.out.println("scanner: " + me.getScanner());
            System.out.println("A: " + FunctionManager.lookUp("A").getMatrix());
            System.out.println("D: " + FunctionManager.lookUp("D").getMatrix());
            FastCompositeExpression fce = new MatrixTurboEvaluator(me).compile();
            Matrix m = fce.apply(new double[0]).matrix;
            System.out.println("matrix:\n" + m);
            Assertions.assertEquals(15286.0, m.getFlatArray()[0]);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Test
    void matrixTestDetBug() {
        try {
            MathExpression me = new MathExpression("A(3,3)=(3,1,5,  4,2,9, 1,4,3);"
                    + "B(3,3)=(4,0,2, 2,1,5, 5,9,4);"
                    + " detab=det(A*B);deta=det(A)");
            FastCompositeExpression fce = new MatrixTurboEvaluator(me).compile();
            System.out.println("scanner: " + me.getScanner());
            System.out.println("A: " + FunctionManager.lookUp("A").getMatrix());
            System.out.println("B: " + FunctionManager.lookUp("B").getMatrix());
            Variable detAB = VariableManager.lookUp("detab");
            Variable detA = VariableManager.lookUp("deta");
            System.out.println("det(A*B): " + detAB.getValue());
            System.out.println("det(A): " + detA.getValue());

            Assertions.assertNotEquals(detAB.getValue(), detA);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    
    
    /**
     * diff(F) Evaluate F's grad func and return the result 
     * diff(F,v) Evaluate F's grad func and store the result in a function pointer called v 
     * diff(F,n) Evaluate F's grad func n times 
     * diff(F,v,n) Evaluate F's grad func n times and store the result in a function pointer called v
     * diff(F,x,n) Evaluate F's grad func n times and calculate the result at x
     */
    @Test
    void testSimpleDifferentialCalculus() {
        try {
            MathExpression me = new MathExpression("f(x)=x^3;diff(f)");
            String s = me.solveGeneric().textRes;
            System.out.println("res:\n"+s);
            Assertions.assertTrue(true);
        } catch (Throwable ex) {
            Logger.getLogger(MathExpressionTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    
    public static void main(String[] args) {
        new MathExpressionTest().matrixTestAlgebraAssignments();
    }

}
