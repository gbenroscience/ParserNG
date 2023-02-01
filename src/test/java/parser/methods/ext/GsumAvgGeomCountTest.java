package parser.methods.ext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import parser.MathExpression;

public class GsumAvgGeomCountTest {

    @Test
    void issue25() {
        MathExpression me;
        me = new MathExpression("avg(1+13)");
        Assertions.assertEquals("14.0", me.solve());
        me = new MathExpression("count(1+13)");
        Assertions.assertEquals("1", me.solve());
        me = new MathExpression("geom(1+13)");
        Assertions.assertEquals("14.0", me.solve());
        me = new MathExpression("gsum(1+13)");
        Assertions.assertEquals("14.0", me.solve());
    }

    @Test
    void avg() {
        MathExpression me = new MathExpression("avg(1,3)");
        Assertions.assertEquals("2", me.solve());
        me = new MathExpression("avg(1,3,1)");
        Assertions.assertEquals("1.666666667", me.solve());
    }

    @Test
    void geom() {
        MathExpression me = new MathExpression("geom(1,2,4)");
        Assertions.assertEquals("2.000000000", me.solve());
        me = new MathExpression("geom(1,3,1)");
        Assertions.assertEquals("1.442249570", me.solve());
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
    void countExpression() {
        MathExpression me = new MathExpression("count(1,1,2,3,3)");
        Assertions.assertEquals("5", me.solve());
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

}
