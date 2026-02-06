package com.github.gbenroscience.math.parser.methods;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import com.github.gbenroscience.parser.MathExpression;

public class ProdTest {

    @Test
    void prodAcceptsOnePlainArgument() {
        MathExpression me;
        me = new MathExpression("prod(5)");
        Assertions.assertEquals("5.0", me.solve());
    }

    @Test
    void prodAcceptsOnePlainExpression() {
        MathExpression me;
        me = new MathExpression("prod(5+5)");
        Assertions.assertEquals("10.0", me.solve());
    }


    @Test
    void prodAcceptsTwoPlainArguments() {
        MathExpression me;
        me = new MathExpression("prod(5,6)");
        Assertions.assertEquals("30.0", me.solve());
    }

    @Test
    void prodAcceptsTwoPlainExpressions() {
        MathExpression me;
        me = new MathExpression("prod(5+5, 4+4)");
        Assertions.assertEquals("80.0", me.solve());
    }

    @Test
    void prodAcceptsStatFunction() {
        MathExpression me;
        me = new MathExpression("prod(prod(5))");
        Assertions.assertEquals("5.0", me.solve());
    }

    @Test
    void prodAcceptsNumFunction() {
        MathExpression me;
        me = new MathExpression("prod(sin(5))");
        Assertions.assertEquals("-0.9589242746631385", me.solve());
    }

    @Test
    void prodAcceptsTwoStatFunction() {
        MathExpression me;
        me = new MathExpression("prod(prod(5),prod(6))");
        Assertions.assertEquals("30.0", me.solve());
    }

    @Test
    void prodAcceptsTwoNumFunction() {
        MathExpression me;
        me = new MathExpression("prod(sin(5),sin(5))");
        Assertions.assertEquals("0.9195357645382262", me.solve());
    }

    @Test
    void prodAcceptsStatAndNumFunction() {
        MathExpression me;
        me = new MathExpression("prod(prod(5),sin(5))");
        Assertions.assertEquals("-4.794621373315692", me.solve());
        me = new MathExpression("prod(sin(5),prod(5))");
        Assertions.assertEquals("-4.794621373315692", me.solve());
    }
}