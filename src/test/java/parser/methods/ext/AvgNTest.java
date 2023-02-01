package parser.methods.ext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import parser.MathExpression;

public class AvgNTest {

    @Test
    void zeroZero() {
        MathExpression me = new MathExpression("avgN(0,0)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("avgN(1,0)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("avgN(2,0)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("avgN(3,0)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("avgN(0,1)");
        Assertions.assertEquals("1", me.solve());
        me = new MathExpression("avgN(0,1,2)");
        Assertions.assertEquals("1.5", me.solve());
        me = new MathExpression("avgN(0,1,2,3)");
        Assertions.assertEquals("2", me.solve());
    }

    @Test
    void decimalsInCountWorksAsHould() {
        MathExpression me;
        me = new MathExpression("avgN(0.4, 1, 2, 3, 4, 15)");
        Assertions.assertEquals("5", me.solve());
        me = new MathExpression("avgN(0.6, 1, 2, 3, 4, 15)");
        Assertions.assertEquals("3", me.solve());
    }

    @Test
    void limitWorks() {
        MathExpression me = new MathExpression("avgN(1,45,1,55,1000)");
        Assertions.assertEquals("50", me.solve());
        me = new MathExpression("avgN(2,45,1,1000,55)");
        Assertions.assertEquals("50", me.solve());
        me = new MathExpression("avgN(2,45,1,1000,80,55)");
        // the avgN default size DEFAULT_AVGN_MISIZE is two, thus 45,1,1000,80,55 -> 1,45,55,80,1000-> 45,55,70 => 180/3
        Assertions.assertEquals("60", me.solve());
        me = new MathExpression("avgN(3,45,1,1000,80,55)");
        Assertions.assertEquals("60", me.solve());
        me = new MathExpression("avgN(3,45,1,1000,80,55,0)");
        Assertions.assertEquals("50", me.solve());
        me = new MathExpression("avgN(3,45,1,1000,80,55,0,0)");
        // the avgN default size DEFAULT_AVGN_MISIZE is two, thus 45,1,1000,80,55,0,0 -> 0,0,1,45,55,80,1000-> 1,45,55=> 101/3
        Assertions.assertEquals("33.66666667", me.solve());
        me = new MathExpression("avgN(3,45,1,1000,80,55,0,0,0)");
        Assertions.assertEquals("23", me.solve());
    }

    @Test
    void testEmpty() {
        MathExpression me;
        me = new MathExpression("avgN(1+1)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("avgN(1)");
        Assertions.assertEquals("0", me.solve());
    }

    @Test
    void testExpres() {
        MathExpression me;
        me = new MathExpression("avgN(0+1, 0+1)");
        Assertions.assertEquals("1.0", me.solve());
        me = new MathExpression("avgN(avgN(0,1), avgN(0,1))");
        Assertions.assertEquals("1", me.solve());
    }
}
