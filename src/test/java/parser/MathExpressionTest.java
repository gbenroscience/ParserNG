package parser;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import math.matrix.expressParser.Matrix;
import parser.methods.BasicNumericalMethod;
import parser.methods.Declarations;
import util.FunctionManager;
import util.VariableManager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

class MathExpressionTest {

    @Test
    void countExpression() {
        MathExpression me = new MathExpression("count(1,1,2,3,3)");
        Assertions.assertEquals("5", me.solve());
    }

    @Test
    void gsumExpression() {
        MathExpression me = new MathExpression("gsum(1,1,2,3)");
        Assertions.assertEquals("6", me.solve());
    }

    @Test
    void geomExpression() {
        MathExpression me = new MathExpression("geom(2,8,4)");
        Assertions.assertEquals("4.000000000", me.solve());
    }

    @Test
    void geomExpressionMultipleBrackets() {
        MathExpression me = new MathExpression("geom((2+((2-2)),(8+8)-(((8))),4))");
        Assertions.assertEquals("4.000000000", me.solve());
        me = new MathExpression("(geom(((2,8,4))))");
        Assertions.assertEquals("4.000000000", me.solve());
        me = new MathExpression("geom((2,8,4))+geom(((2,8,4)))");
        Assertions.assertEquals("8.0", me.solve());
        me = new MathExpression("(geom((2+((2-2)),(8+8)-(((8))),4))+geom(((2,8,4))))");
        Assertions.assertEquals("8.0", me.solve());
        me = new MathExpression("((((geom((2,8,4))+geom(((2,8,4)))))))");
        Assertions.assertEquals("8.0", me.solve());
    }

    @Test
    void expTest() {
        MathExpression me = new MathExpression("2^3");
        Assertions.assertEquals("8.0", me.solve());
    }

    @Test
    void rootTest() {
        MathExpression me = new MathExpression("8^(1/3)");
        Assertions.assertEquals("2.0", me.solve());
        me = new MathExpression("8^(1/2)");
        Assertions.assertEquals("2.8284271247461903", me.solve());
    }

    @Test
    void customEmbeddedFunctionTest() {
        MathExpression me = new MathExpression("avgN(0,1,2,3)");
        Assertions.assertEquals("2", me.solve());
    }

    @Test
    void customEmbeddedFunctionTestMultipleBrackets() {
        MathExpression me = new MathExpression("((avgN((0,1,2,3))))");
        Assertions.assertEquals("2", me.solve());
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
        MathExpression me = new MathExpression("b1(1,2,3)");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
        Declarations.registerBasicNumericalMethod(b1);
        me = new MathExpression("b1(1,2,3)");
        Assertions.assertEquals("1", me.solve());
        me = new MathExpression("b2(1,2,3)");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
        Declarations.registerBasicNumericalMethod(b2);
        me = new MathExpression("b2(1,2,3)");
        Assertions.assertEquals("2", me.solve());
        me = new MathExpression("b1(1,2,3)");
        Assertions.assertEquals("1", me.solve());
        me = new MathExpression("b1(1,2,3)+b2(1,2,3)");
        Assertions.assertEquals("3.0", me.solve());
        Declarations.unregisterBasicNumericalMethod(b1.getClass());
        me = new MathExpression("b1(1,2,3)");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
        me = new MathExpression("b2(1,2,3)");
        Assertions.assertEquals("2", me.solve());
        Declarations.unregisterBasicNumericalMethod(b2.getClass());
        me = new MathExpression("b2(1,2,3)");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
    }

    @Test
    void help() {
        MathExpression me = new MathExpression("help");
        String help = me.solve();
        Assertions.assertTrue(help.length() > 100);
        Assertions.assertTrue(help.contains(Declarations.COS));
    }

    @Test
    void oldMainTest() {
        MathExpression expr = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2)");
        Assertions.assertEquals("0.0", expr.solve());
    }

    @Test
    void moreJunkExamples() {
//        System.out.println(0xFF000000);
//        System.out.println(0xFF888888);
//        System.out.println(0xFFFFFFFF);
        Function f = FunctionManager.add("f(x,y) = x - x/y");
        double r = f.calc(2, 3);
        Assertions.assertEquals((double) 2 - ((double) 2 / (double) 3), r);
        int iterations = 10000;
        long start = System.nanoTime();
        for (int i = 1; i < iterations; i++) {
            f.calc(2, 3);
        }
        long duration = System.nanoTime() - start;

        //System.out.println("Time: " + (duration / iterations) + " ns");
        Assertions.assertTrue(duration / iterations < 1000000);
    }

    @Test
    void junkExamples() {
        boolean print = false;
        MathExpression linear = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);C=matrix_sub(M,N);C;");
        String ls = linear.solve();
        if (print) System.out.println("soln: " + ls);
        Assertions.assertEquals("\n"
                + "   -1.0  ,    3.0  ,   -7.0            \n"
                + "    0.0  ,    3.0  ,    4.0            \n"
                + "    4.0  ,    0.0  ,  -11.0            \n", ls);

        MathExpression expr = new MathExpression("tri_mat(M)");
        String tm = expr.solve();
        if (print) System.out.println(tm);
        Assertions.assertEquals("\n"
                + "    1.0  ,1.3333333333333333  ,0.3333333333333333            \n"
                + "    0.0  ,    1.0  ,4.749999999999999            \n"
                + "    0.0  ,    0.0  ,    1.0            \n", tm);

        MathExpression expr2 = new MathExpression("echelon(M)");
        String echelon = expr2.solve();
        if (print) System.out.println(echelon);
        Assertions.assertEquals("\n"
                + "    3.0  ,    4.0  ,    1.0            \n"
                + "    0.0  ,    4.0  ,   19.0            \n"
                + "    0.0  ,    0.0  ,  567.0            \n", echelon);

        Function matrixFunction = FunctionManager.lookUp("M");
        Matrix matrix = matrixFunction.getMatrix();
        if (print) System.out.println("underlying matrix: " + matrix);
        Assertions.assertEquals("\n"
                + "    3.0  ,    4.0  ,    1.0            \n"
                + "    2.0  ,    4.0  ,    7.0            \n"
                + "    9.0  ,    1.0  ,   -2.0            \n", matrix.toString());

        Matrix inv = matrix.inverse();
        if (print) System.out.println("inverted matrix: " + inv);
        Assertions.assertEquals("\n"
                + "-0.07936507936507936  ,0.04761904761904753  ,0.12698412698412698            \n"
                + "0.3544973544973545  ,-0.0793650793650793  ,-0.10052910052910052            \n"
                + "-0.1798941798941799  ,0.1746031746031746  ,0.021164021164021166            \n", inv.toString());

        matrix.multiply(inv);
        if (print) System.out.println("mul matrix: " + matrix);
        Assertions.assertEquals("\n"
                + "0.9999999999999999  ,-2.7755575615628914E-17  ,3.469446951953614E-18            \n"
                + "-2.220446049250313E-16  ,    1.0  ,2.7755575615628914E-17            \n"
                + "1.1102230246251565E-16  ,-6.661338147750939E-16  ,    1.0            \n", matrix.toString());

        FunctionManager.add("f(x,y) = x-x/y");
        Function fxy = FunctionManager.lookUp("f");

        MathExpression parserng = new MathExpression("f(x,y) = x-x/y; f(2,3);f(2,5);");
        String parsengr = parserng.solve();
        if (print) System.out.println("SEE???????\n " + parsengr);
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
        FunctionManager.add("M=@(3,3)(3,4,1,2,4,7,9,1,-2)");
        FunctionManager.add("M1=@(2,2)(3,4,1,2)");
        FunctionManager.add("M2=@(2,2)(3,4,1,2)");

        FunctionManager.add("N=@(x)(sin(x))");
        FunctionManager.add("r=@(x)(ln(sin(x)))");

        //WORK ON sum(3,-2sin(3)^2,4,5) error //matrix_mul(@(2,2)(3,1,4,2),@(2,2)(2,-9,-4,3))...sum(3,2sin(4),5,-3cos(2*sin(5)),4,1,3)
        //matrix_mul(invert(@(2,2)(3,1,4,2)),@(2,2)(2,-9,-4,3)).............matrix_mul(invert(@(2,2)(3,1,4,2)),@(2,2)(2,-9,-4,3))
        //matrix_mul(invert(@(2,2)(3,1,4,2)),matrix_mul(M2,2sin(3)sum(3,6,5,3)cos(9/8*6)det(@(2,2)(2,-9,-4,3))))
        if (print) System.out.println("lookup M: " + FunctionManager.lookUp("M"));
        Assertions.assertEquals("M=@(3,3)(3.0,4.0,1.0,2.0,4.0,7.0,9.0,1.0,-2.0)", FunctionManager.lookUp("M").toString());
        //MathExpression expr = new MathExpression("f=3;5f");//BUGGY
        //MathExpression expr = new MathExpression("quad(@(x)3*x-2+3*x^2)");//BUGGY
        //MathExpression expr = new MathExpression("root(@(x)3*x-sin(x)-0.5,2)");//BUGGY
        MathExpression exprs = new MathExpression("r1=4;r1*5");

        //A+k.A+AxB+A^c
        if (print) System.out.println("scanner: " + exprs.scanner);
        Assertions.assertEquals("[(, r1, *, 5, )]", exprs.scanner.toString());
        if (print) System.out.println("solution: " + exprs.solve());
        Assertions.assertEquals("20.0", exprs.solve());

        //ni idea what is this trying to do, resuts sounds buggy anyway
        expr.setExpression("44+22*(3)");
        if (print) System.out.println("solution--: " + expr.solve());
        Assertions.assertEquals("44, +, 22, *, 3", expr.solve());

        if (print) System.out.println("return type: " + exprs.returnType);
        Assertions.assertEquals(TYPE.NUMBER, exprs.returnType);
        List<Map.Entry<String, Function>> l = new ArrayList(FunctionManager.FUNCTIONS.entrySet());
        Collections.sort(l, new Comparator<Map.Entry<String, Function>>() {
            @Override
            public int compare(Map.Entry<String, Function> t0, Map.Entry<String, Function> t1) {
                return t0.getKey().compareTo(t1.getKey());
            }
        });
        //although the sort is maaking it deterministic, the assert is fragile and afaik have no sense
        if (print) System.out.println("FunctionManager: " + l.toString());
//        Assertions.assertEquals(
//                "[C=C=@(3,3)(-1.0,3.0,-7.0,0.0,3.0,4.0,4.0,0.0,-11.0), M=M=@(3,3)(3.0,4.0,1.0,2.0,4.0,7.0,9.0,1.0,-2.0), M1=M1=@(2,2)(3.0,4.0,1.0,2.0), M2=M2=@(2,2)(3.0,4.0,1.0,2.0), N=N=@(x)(sin"
//                        + "(x)), anon1=anon1=@(3,3)(3.0,4.0,1.0,2.0,4.0,7.0,9.0,1.0,-2.0), anon2=anon2=@(3,3)(4.0,1.0,8.0,2.0,1.0,3.0,5.0,1.0,9.0), anon3=anon3=@(3,3)(-1.0,3.0,-7.0,0.0,3.0,4.0,4.0,"
//                        + "0.0,-11.0), anon4=anon4=@(3,3)(1.0,1.3333333333333333,0.3333333333333333,0.0,1.0,4.749999999999999,0.0,0.0,1.0), anon5=anon5=@(3,3)(3.0,4.0,1.0,0.0,4.0,19.0,0.0,0.0,567"
//                        + ".0), anon6=anon6=@(x,y)(x-x/y), f=f=@(x,y)(x-x/y), r=r=@(x)(ln(sin(x)))]",
//                l.toString());
        if (print) System.out.println("VariableManager: " + VariableManager.VARIABLES);
        Assertions.assertEquals("{e=e:2.718281828459045, ans=ans:0.0, x=x:0.0, pi=pi:3.1415926535897932, y=y:0.0, r1=r1:4}", VariableManager.VARIABLES.toString());

        MathExpression expression = new MathExpression("x=0;sin(ln(x))");

        expression.setValue("x", 0 + "");
        if (print) System.out.println(expression.solve());
        Assertions.assertEquals("SYNTAX ERROR", expression.solve());
        expression.setValue("x", 1 + "");
        if (print) System.out.println(expression.solve());
        Assertions.assertEquals("0.0", expression.solve());
        expression.setValue("x", 50 + "");
        if (print) System.out.println(expression.solve());
        Assertions.assertTrue("-0.6964441283311967".equals(expression.solve()) || "-0.6964441283311968".equals(expression.solve()));
        expression.setValue("x", 100 + "");
        if (print) System.out.println(expression.solve());
        Assertions.assertEquals("-0.9942575694137897", expression.solve());

        Function f = FunctionManager.lookUp("N");
        long start = System.nanoTime();
        int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            f.calc(i + 3);
        }
        long elapsedNanos = (System.nanoTime() - start) / iterations;
        double durationms = (double) elapsedNanos / 1.0E6;
        if (print) System.out.println("DONE: " + durationms + " ms");
        Assertions.assertTrue(durationms < 1);

        MathExpression det = new MathExpression("det(@(5,5)(-21,12,13,64,5,6,2.7,18,9,0,4,2,3,4,23,6,7,8,9,0,1,2,32,4,5));");
        if (print) System.out.println("determinant: " + det.solve());
        Assertions.assertEquals("-1739274.3000000003", det.solve());

        /**
         * On my Macbook Pro, 16GB RAM; 2.6 GHz Intel Core i7
         *
         * The code runs the solve() method at 3.8 microseconds.
         */
    }

}