package com.github.gbenroscience.parser.methods.ext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import com.github.gbenroscience.parser.MathExpression;
import java.util.InputMismatchException;

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
        try {
            me = new MathExpression("sin(5,6)");
            Assertions.assertTrue(false, "sin cannot have 2 args(numbers) except when overriden as such");
        } catch (InputMismatchException inp) {
            Assertions.assertTrue(true, "sin cannot have 2 args(numbers) except when overriden as such");
        }
    }

    @Test
    void sinDontAcceptsTwoPlainExpressions() {
        MathExpression me;
        try {
            me = new MathExpression("sin(5+5, 4+4)");
            Assertions.assertTrue(false, "sin cannot have 2 args(expressions) except when overriden as such");
        } catch (InputMismatchException inp) {
            Assertions.assertTrue(true, "sin cannot have 2 args(expressions) except when overriden as such");
        }
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
        try {
            me = new MathExpression("sin(sum(5),sum(6))");
            Assertions.assertTrue(false, "sin cannot have 2 args(stats functions) except when overriden as such");
        } catch (InputMismatchException inp) {
            Assertions.assertTrue(true, "sin cannot have 2 args(stats functions) except when overriden as such");
        }
    }

    @Test
    void sinDontAcceptsTwoNumFunction() {
        MathExpression me;
        try {
            me = new MathExpression("sin(sin(5),sin(5))");
            Assertions.assertTrue(false, "sin cannot have 2 args(number functions) except when overriden as such");
        } catch (InputMismatchException inp) {
            Assertions.assertTrue(true, "sin cannot have 2 args(number functions) except when overriden as such");
        }
    }

    @Test
    void sinDontAcceptsStatAndNumFunction() {
        MathExpression me;
        int i=0;
        try {
            me = new MathExpression("sin(sin(5),sum(5))");
            Assertions.assertTrue(false, "sin cannot have 2 args(number and stat) except when overriden as such");
            i++;
            me = new MathExpression("sin(sum(5),sin(5))");
            Assertions.assertTrue(false, "sin cannot have 2 args(stat and number) except when overriden as such");
        } catch (InputMismatchException inp) {
            Assertions.assertTrue(true, "sin cannot have 2 args("+(i==0 ? "number and stat" : "stat and number" )+") except when overriden as such");
        }
    }
}
