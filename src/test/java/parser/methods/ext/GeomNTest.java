package parser.methods.ext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import parser.MathExpression;

public class GeomNTest {

    @Test
    void zeroZero() {
        MathExpression me = new MathExpression("geomN(0,0)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("geomN(1,0)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("geomN(2,0)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("geomN(3,0)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("geomN(0,1)");
        Assertions.assertEquals("1", me.solve());
        me = new MathExpression("geomN(0,1,2)");
        Assertions.assertEquals("1.414213562", me.solve());
        me = new MathExpression("geomN(0,1,2,3)");
        Assertions.assertEquals("1.817120593", me.solve());
    }

    @Test
    void decimalsInCountWorksAsHould() {
        MathExpression me;
        me = new MathExpression("geomN(0.4, 1, 2, 3, 4, 15)");
        Assertions.assertEquals("3.245342223", me.solve());
        me = new MathExpression("geomN(0.6, 1, 2, 3, 4, 15)");
        Assertions.assertEquals("2.884499141", me.solve());
    }

    @Test
    void limitWorks() {
        MathExpression me = new MathExpression("geomN(1,2,1,4,1000,8)");
        Assertions.assertEquals("4.000000000", me.solve());
        me = new MathExpression("geomN(2,10,1,1000,10)");
        Assertions.assertEquals("10.00000000", me.solve());
        me = new MathExpression("geomN(2,2,1,1000,8,4)");
        Assertions.assertEquals("4.000000000", me.solve());
        me = new MathExpression("geomN(3,4,1,1000,8,2)");
        Assertions.assertEquals("4.000000000", me.solve());
        me = new MathExpression("geomN(3,45,1,1000,70,55,0)");
        Assertions.assertEquals("49.74937185", me.solve());
        me = new MathExpression("geomN(3,45,1,1000,70,55,0,0)");
        Assertions.assertEquals("13.52669614", me.solve());
        me = new MathExpression("geomN(3,45,1,1000,70,55,0,0,0)");
        Assertions.assertEquals("6.708203932", me.solve());
    }

    @Test
    void testEmpty() {
        MathExpression me;
        me = new MathExpression("geomN(1+1)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("geomN(1)");
        Assertions.assertEquals("0", me.solve());
    }

    @Test
    void testExpres() {
        MathExpression me;
        me = new MathExpression("geomN(0+1, 0+1)");
        Assertions.assertEquals("1.0", me.solve());
        me = new MathExpression("geomN(geomN(0,1), geomN(0,1))");
        Assertions.assertEquals("1", me.solve());
    }
}
