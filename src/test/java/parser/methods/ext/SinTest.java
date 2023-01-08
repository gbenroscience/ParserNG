package parser.methods.ext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import parser.MathExpression;

public class SinTest {

    @Test
    void sinAcceptsOnePlainArgument() {
        MathExpression me;
        me = new MathExpression("sin(5)");
        Assertions.assertEquals("-0.9589242746631385", me.solve());
    }

    @Test
    void sinAcceptsOnePlainExpression() {
        MathExpression me;
        me = new MathExpression("sin(5+5)");
        Assertions.assertEquals("-0.5440211108893698", me.solve());
    }


    @Test
    void sinDontAcceptsTwoPlainArguments() {
        MathExpression me;
        me = new MathExpression("sin(5,6)");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
    }

    @Test
    void sinDontAcceptsTwoPlainExpressions() {
        MathExpression me;
        me = new MathExpression("sin(5+5, 4+4)");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
    }

    @Test
    void sinAcceptsStatFunction() {
        MathExpression me;
        me = new MathExpression("sin(sum(5))");
        Assertions.assertEquals("-0.9589242746631385", me.solve());
    }

    @Test
    void sinAcceptsNumFunction() {
        MathExpression me;
        me = new MathExpression("sin(sin(5))");
        Assertions.assertEquals("-0.8185741444617193", me.solve());
    }

    @Test
    void sinDontAcceptsTwoStatFunction() {
        MathExpression me;
        me = new MathExpression("sin(sum(5),sum(6))");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
    }

    @Test
    void sinDontAcceptsTwoNumFunction() {
        MathExpression me;
        me = new MathExpression("sin(sin(5),sin(5))");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
    }

    @Test
    void sinDontAcceptsStatAndNumFunction() {
        MathExpression me;
        me = new MathExpression("sin(sin(5),sum(5))");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
        me = new MathExpression("sin(sum(5),sin(5))");
        Assertions.assertEquals("SYNTAX ERROR", me.solve());
    }
}