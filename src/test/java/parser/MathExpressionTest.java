package parser;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import parser.methods.BasicNumericalMethod;
import parser.methods.Declarations;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

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
        MathExpression me = new MathExpression("weirdFce(1,2,3)");
        Assertions.assertEquals("0.1", me.solve());
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
        Assertions.assertTrue(help.contains(Declarations.SIN));
    }

}