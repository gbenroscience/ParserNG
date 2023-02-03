package parser.methods.ext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import parser.MathExpression;

public class RoundingTests {

    @Test
    void roundingDigits() {
        MathExpression me;
        me = new MathExpression("roundDigitsN(1, 125.35)");
        Assertions.assertEquals("1E+2", me.solve());
        me = new MathExpression("roundDigitsN(2, 125.35)");
        Assertions.assertEquals("1.3E+2", me.solve());
        me = new MathExpression("roundDigitsN(2, -125.35)");
        Assertions.assertEquals("-1.3E+2", me.solve());
        me = new MathExpression("roundDigitsN(2, -124.35)");
        Assertions.assertEquals("-1.2E+2", me.solve());
        me = new MathExpression("roundDigitsN(3, 125.35)");
        Assertions.assertEquals("125", me.solve());
        me = new MathExpression("roundDigitsN(4, 125.35)");
        Assertions.assertEquals("125.4", me.solve());
        me = new MathExpression("roundDigitsN(5, 125.35)");
        Assertions.assertEquals("125.35", me.solve());
        me = new MathExpression("roundDigitsN(6, 125.35)");
        Assertions.assertEquals("125.35", me.solve());
    }

    @Test
    void roundingAsExpected() {
        MathExpression me;
        for (String s : new String[]{"", "-", "--"}) {
            String r = "";
            if (s.equals("-")) {
                r = "-";
            }
            me = new MathExpression("round(" + s + "1)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "2)");
            Assertions.assertEquals(r + "2", me.solve());

            me = new MathExpression("round(" + s + "1.1)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "1.9)");
            Assertions.assertEquals(r + "2", me.solve());
            me = new MathExpression("round(" + s + "1.4)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "1.5)");
            Assertions.assertEquals(r + "2", me.solve());

            me = new MathExpression("round(" + s + "1.44)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "1.45)");
            Assertions.assertEquals(r + "1", me.solve());

            me = new MathExpression("round(" + s + "1.444)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "1.445)");
            Assertions.assertEquals(r + "1", me.solve());

            me = new MathExpression("round(" + s + "12.99999)");
            Assertions.assertEquals(r + "13", me.solve());
        }
    }

    @Test
    void roundNingAsExpected() {
        MathExpression me;
        for (String s : new String[]{"", "-", "--"}) {
            String r = "";
            if (s.equals("-")) {
                r = "-";
            }
            me = new MathExpression("roundN(1," + s + "1)");
            Assertions.assertEquals(r + "1.0", me.solve());
            me = new MathExpression("roundN(2," + s + "2)");
            Assertions.assertEquals(r + "2.00", me.solve());

            me = new MathExpression("roundN(2," + s + "1.1)");
            Assertions.assertEquals(r + "1.10", me.solve());
            me = new MathExpression("roundN(2," + s + "1.9)");
            Assertions.assertEquals(r + "1.90", me.solve());
            me = new MathExpression("roundN(5," + s + "1.4)");
            Assertions.assertEquals(r + "1.40000", me.solve());
            me = new MathExpression("roundN(6, " + s + "1.5)");
            Assertions.assertEquals(r + "1.500000", me.solve());

            me = new MathExpression("roundN(1," + s + "1.44)");
            Assertions.assertEquals(r + "1.4", me.solve());
            me = new MathExpression("roundN(1," + s + "1.45)");
            Assertions.assertEquals(r + "1.5", me.solve());

            me = new MathExpression("roundN(1," + s + "1.444)");
            Assertions.assertEquals(r + "1.4", me.solve());
            me = new MathExpression("roundN(1," + s + "1.445)");
            Assertions.assertEquals(r + "1.4", me.solve());

            me = new MathExpression("roundN(2," + s + "1.444)");
            Assertions.assertEquals(r + "1.44", me.solve());
            me = new MathExpression("roundN(2," + s + "1.445)");
            Assertions.assertEquals(r + "1.45", me.solve());

            me = new MathExpression("roundN(2, " + s + "12.99999)");
            Assertions.assertEquals(r + "13.00", me.solve());
        }

    }

    @Test
    void roundXingAsExpected() {
        MathExpression me;
        for (String s : new String[]{"", "-", "--"}) {
            String r = "";
            if (s.equals("-")) {
                r = "-";
            }
            me = new MathExpression("roundX(1," + s + "1)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("roundX(2," + s + "2)");
            Assertions.assertEquals(r + "2", me.solve());

            me = new MathExpression("roundX(2," + s + "1.1)");
            Assertions.assertEquals(r + "1.1", me.solve());
            me = new MathExpression("roundX(2," + s + "1.9)");
            Assertions.assertEquals(r + "1.9", me.solve());
            me = new MathExpression("roundX(5," + s + "1.4)");
            Assertions.assertEquals(r + "1.4", me.solve());
            me = new MathExpression("roundX(6, " + s + "1.5)");
            Assertions.assertEquals(r + "1.5", me.solve());

            me = new MathExpression("roundX(1," + s + "1.44)");
            Assertions.assertEquals(r + "1.4", me.solve());
            me = new MathExpression("roundX(1," + s + "1.45)");
            Assertions.assertEquals(r + "1.5", me.solve());

            me = new MathExpression("roundX(1," + s + "1.444)");
            Assertions.assertEquals(r + "1.4", me.solve());
            me = new MathExpression("roundX(1," + s + "1.445)");
            Assertions.assertEquals(r + "1.5", me.solve());//!!!!

            me = new MathExpression("roundX(2," + s + "1.444)");
            Assertions.assertEquals(r + "1.44", me.solve());
            me = new MathExpression("roundX(2," + s + "1.445)");
            Assertions.assertEquals(r + "1.45", me.solve());

            me = new MathExpression("roundX(2, " + s + "12.99999)");
            Assertions.assertEquals(r + "13.0000", me.solve());
        }
    }

    @Test
    void aroundZero() {
        MathExpression me;
        for (String s : new String[]{"", "-", "--"}) {
            String r = "";
            if (s.equals("-")) {
                r = "-";
            }
            me = new MathExpression("round(" + s + "0.9)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "0.5)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "0.4)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("round(" + s + "0.1)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("round(" + s + "0)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("round(" + s + "0.0)");
            Assertions.assertEquals( "0", me.solve());

            me = new MathExpression("round(" + s + "0.94)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "0.95)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "0.54)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "0.55)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("round(" + s + "0.44)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("round(" + s + "0.45)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("round(" + s + "0.14)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("round(" + s + "0.15)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("round(" + s + "0)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("round(" + s + "0.04)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("round(" + s + "0.05)");
            Assertions.assertEquals( "0", me.solve());
        }
    }


    @Test
    void xaroundZero() {
        MathExpression me;
        for (String s : new String[]{"", "-", "--"}) {
            String r = "";
            if (s.equals("-")) {
                r = "-";
            }
            me = new MathExpression("roundX(0," + s + "0.9)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("roundX(0," + s + "0.5)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("roundX(0," + s + "0.4)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("roundX(0," + s + "0.1)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("roundX(0," + s + "0)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("roundX(0," + s + "0.0)");
            Assertions.assertEquals( "0.0", me.solve());

            me = new MathExpression("roundX(0," + s + "0.94)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("roundX(0," + s + "0.95)");
            Assertions.assertEquals(r + "1.0", me.solve());
            me = new MathExpression("roundX(0," + s + "0.54)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("roundX(0," + s + "0.55)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("roundX(0," + s + "0.44)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("roundX(0," + s + "0.45)");
            Assertions.assertEquals( r+ "1", me.solve());
            me = new MathExpression("roundX(0," + s + "0.14)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("roundX(0," + s + "0.15)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("roundX(0," + s + "0)");
            Assertions.assertEquals( "0", me.solve());
            me = new MathExpression("roundX(0," + s + "0.04)");
            Assertions.assertEquals( "0.0", me.solve());
            me = new MathExpression("roundX(0," + s + "0.05)");
            Assertions.assertEquals( "0", me.solve());
        }
    }

    @Test
    void roundAsExpectedExpr() {
        MathExpression me;
        me = new MathExpression("round(3+2)");
        Assertions.assertEquals("5", me.solve());
        me = new MathExpression("round(3/2)");
        Assertions.assertEquals("2", me.solve());
    }

    @Test
    void roundAsExpectedFunctions() {
        MathExpression me;
        me = new MathExpression("round(sin(5))");
        Assertions.assertEquals("-1", me.solve());
        me = new MathExpression("roundN(2,sin(5))");
        Assertions.assertEquals("-0.96", me.solve());
        me = new MathExpression("round(abs(-1.5))");
        Assertions.assertEquals("2", me.solve());

    }


}
