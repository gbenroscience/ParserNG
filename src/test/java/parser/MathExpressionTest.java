package parser;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

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

}