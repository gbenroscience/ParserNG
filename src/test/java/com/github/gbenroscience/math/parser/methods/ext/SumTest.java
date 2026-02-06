package parser.methods.ext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import parser.MathExpression;

public class SumTest {

    @Test
    void sumAcceptsOnePlainArgument() {
        MathExpression me;
        me = new MathExpression("sum(5)");
        Assertions.assertEquals("5", me.solve());
    }

    @Test
    void sumAcceptsOnePlainExpression() {
        MathExpression me;
        me = new MathExpression("sum(5+5)");
        Assertions.assertEquals("10.0", me.solve());
    }


    @Test
    void sumAcceptsTwoPlainArguments() {
        MathExpression me;
        me = new MathExpression("sum(5,6)");
        Assertions.assertEquals("11", me.solve());
    }

    @Test
    void sumAcceptsTwoPlainExpressions() {
        MathExpression me;
        me = new MathExpression("sum(5+5, 4+4)");
        Assertions.assertEquals("18.0", me.solve());
    }

    @Test
    void sumAcceptsStatFunction() {
        MathExpression me;
        me = new MathExpression("sum(sum(5))");
        Assertions.assertEquals("5", me.solve());
    }

    @Test
    void sumAcceptsNumFunction() {
        MathExpression me;
        me = new MathExpression("sum(sin(5))");
        Assertions.assertEquals("-0.9589242746631385", me.solve());
    }

    @Test
    void sumAcceptsTwoStatFunction() {
        MathExpression me;
        me = new MathExpression("sum(sum(5),sum(6))");
        Assertions.assertEquals("11", me.solve());
    }

    @Test
    void sumAcceptsTwoNumFunction() {
        MathExpression me;
        me = new MathExpression("sum(sin(5),sin(5))");
        Assertions.assertEquals("-1.9178485493262770", me.solve());
    }

    @Test
    void sumAcceptsStatAndNumFunction() {
        MathExpression me;
        me = new MathExpression("sum(sum(5),sin(5))");
        Assertions.assertEquals("4.0410757253368615", me.solve());
        me = new MathExpression("sum(sin(5),sum(5))");
        Assertions.assertEquals("4.0410757253368615", me.solve());
    }
}