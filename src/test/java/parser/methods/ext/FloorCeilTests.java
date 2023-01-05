package parser.methods.ext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import parser.MathExpression;

public class FloorCeilTests {

    @Test
    void flooringDigits() {
        MathExpression me;
        me = new MathExpression("floorDigitsN(1, 125.35)");
        Assertions.assertEquals("1E+2", me.solve());
        me = new MathExpression("floorDigitsN(2, 125.35)");
        Assertions.assertEquals("1.2E+2", me.solve());
        me = new MathExpression("floorDigitsN(2, -125.35)");
        Assertions.assertEquals("-1.3E+2", me.solve());
        me = new MathExpression("floorDigitsN(2, -124.35)");
        Assertions.assertEquals("-1.3E+2", me.solve());
        me = new MathExpression("floorDigitsN(3, 125.35)");
        Assertions.assertEquals("125", me.solve());
        me = new MathExpression("floorDigitsN(4, 125.35)");
        Assertions.assertEquals("125.3", me.solve());
        me = new MathExpression("floorDigitsN(5, 125.35)");
        Assertions.assertEquals("125.35", me.solve());
        me = new MathExpression("floorDigitsN(6, 125.35)");
        Assertions.assertEquals("125.35", me.solve());
    }

    @Test
    void flooringAsExpected() {
        MathExpression me;
        for (String s : new String[]{"", "-", "--"}) {
            String r = "";
            if (s.equals("-")) {
                r = "-";
            }
            me = new MathExpression("floor(" + s + "1)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("floor(" + s + "2)");
            Assertions.assertEquals(r + "2", me.solve());

            me = new MathExpression("floor(" + s + "1.1)");
            if (s.equals("-")) {
                Assertions.assertEquals("-2", me.solve());
            } else {
                Assertions.assertEquals("1", me.solve());
            }
            me = new MathExpression("floor(" + s + "1.9)");
            if (s.equals("-")) {
                Assertions.assertEquals("-2", me.solve());
            } else {
                Assertions.assertEquals("1", me.solve());
            }
            me = new MathExpression("floor(" + s + "1.4)");
            if (s.equals("-")) {
                Assertions.assertEquals("-2", me.solve());
            } else {
                Assertions.assertEquals("1", me.solve());
            }
            me = new MathExpression("floor(" + s + "1.5)");
            if (s.equals("-")) {
                Assertions.assertEquals("-2", me.solve());
            } else {
                Assertions.assertEquals("1", me.solve());
            }

            me = new MathExpression("floor(" + s + "1.44)");
            if (s.equals("-")) {
                Assertions.assertEquals("-2", me.solve());
            } else {
                Assertions.assertEquals("1", me.solve());
            }
            me = new MathExpression("floor(" + s + "1.45)");
            if (s.equals("-")) {
                Assertions.assertEquals("-2", me.solve());
            } else {
                Assertions.assertEquals("1", me.solve());
            }

            me = new MathExpression("floor(" + s + "1.444)");
            if (s.equals("-")) {
                Assertions.assertEquals("-2", me.solve());
            } else {
                Assertions.assertEquals("1", me.solve());
            }
            me = new MathExpression("floor(" + s + "1.445)");
            if (s.equals("-")) {
                Assertions.assertEquals("-2", me.solve());
            } else {
                Assertions.assertEquals("1", me.solve());
            }

            me = new MathExpression("floor(" + s + "12.99999)");
            if (s.equals("-")) {
                Assertions.assertEquals("-13", me.solve());
            } else {
                Assertions.assertEquals("12", me.solve());
            }
        }
    }

    @Test
    void floorNingAsExpected() {
        MathExpression me;
        for (String s : new String[]{"", "-", "--"}) {
            String r = "";
            if (s.equals("-")) {
                r = "-";
            }
            me = new MathExpression("floorN(1," + s + "1)");
            Assertions.assertEquals(r + "1.0", me.solve());
            me = new MathExpression("floorN(2," + s + "2)");
            Assertions.assertEquals(r + "2.00", me.solve());

            me = new MathExpression("floorN(2," + s + "1.1)");
            Assertions.assertEquals(r + "1.10", me.solve());
            me = new MathExpression("floorN(2," + s + "1.9)");
            Assertions.assertEquals(r + "1.90", me.solve());
            me = new MathExpression("floorN(5," + s + "1.4)");
            Assertions.assertEquals(r + "1.40000", me.solve());
            me = new MathExpression("floorN(6, " + s + "1.5)");
            Assertions.assertEquals(r + "1.500000", me.solve());

            me = new MathExpression("floorN(1," + s + "1.44)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1.5", me.solve());
            } else {
                Assertions.assertEquals("1.4", me.solve());
            }
            me = new MathExpression("floorN(1," + s + "1.45)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1.5", me.solve());
            } else {
                Assertions.assertEquals("1.4", me.solve());
            }

            me = new MathExpression("floorN(1," + s + "1.444)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1.5", me.solve());
            } else {
                Assertions.assertEquals("1.4", me.solve());
            }
            me = new MathExpression("floorN(1," + s + "1.445)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1.5", me.solve());
            } else {
                Assertions.assertEquals("1.4", me.solve());
            }

            me = new MathExpression("floorN(2," + s + "1.444)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1.45", me.solve());
            } else {
                Assertions.assertEquals("1.44", me.solve());
            }
            me = new MathExpression("floorN(2," + s + "1.445)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1.45", me.solve());
            } else {
                Assertions.assertEquals("1.44", me.solve());
            }

            me = new MathExpression("floorN(2, " + s + "12.99999)");
            if (s.equals("-")) {
                Assertions.assertEquals("-13.00", me.solve());
            } else {
                Assertions.assertEquals("12.99", me.solve());
            }
        }

    }


    @Test
    void ceilingDigits() {
        MathExpression me;
        me = new MathExpression("ceilDigitsN(1, 125.35)");
        Assertions.assertEquals("2E+2", me.solve());
        me = new MathExpression("ceilDigitsN(2, 125.35)");
        Assertions.assertEquals("1.3E+2", me.solve());
        me = new MathExpression("ceilDigitsN(2, -125.35)");
        Assertions.assertEquals("-1.2E+2", me.solve());
        me = new MathExpression("ceilDigitsN(2, -124.35)");
        Assertions.assertEquals("-1.2E+2", me.solve());
        me = new MathExpression("ceilDigitsN(3, 125.35)");
        Assertions.assertEquals("126", me.solve());
        me = new MathExpression("ceilDigitsN(4, 125.35)");
        Assertions.assertEquals("125.4", me.solve());
        me = new MathExpression("ceilDigitsN(5, 125.35)");
        Assertions.assertEquals("125.35", me.solve());
        me = new MathExpression("ceilDigitsN(6, 125.35)");
        Assertions.assertEquals("125.35", me.solve());
    }

    @Test
    void ceilingAsExpected() {
        MathExpression me;
        for (String s : new String[]{"", "-", "--"}) {
            String r = "";
            if (s.equals("-")) {
                r = "-";
            }
            me = new MathExpression("ceil(" + s + "1)");
            Assertions.assertEquals(r + "1", me.solve());
            me = new MathExpression("ceil(" + s + "2)");
            Assertions.assertEquals(r + "2", me.solve());

            me = new MathExpression("ceil(" + s + "1.1)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1", me.solve());
            } else {
                Assertions.assertEquals("2", me.solve());
            }
            me = new MathExpression("ceil(" + s + "1.9)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1", me.solve());
            } else {
                Assertions.assertEquals("2", me.solve());
            }
            me = new MathExpression("ceil(" + s + "1.4)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1", me.solve());
            } else {
                Assertions.assertEquals("2", me.solve());
            }
            me = new MathExpression("ceil(" + s + "1.5)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1", me.solve());
            } else {
                Assertions.assertEquals("2", me.solve());
            }

            me = new MathExpression("ceil(" + s + "1.44)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1", me.solve());
            } else {
                Assertions.assertEquals("2", me.solve());
            }
            me = new MathExpression("ceil(" + s + "1.45)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1", me.solve());
            } else {

            }

            me = new MathExpression("ceil(" + s + "1.444)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1", me.solve());
            } else {
                Assertions.assertEquals("2", me.solve());
            }
            me = new MathExpression("ceil(" + s + "1.445)");
            if (s.equals("-")) {
                Assertions.assertEquals("-1", me.solve());
            } else {
                Assertions.assertEquals("2", me.solve());
            }
            me = new MathExpression("ceil(" + s + "12.99999)");
            if (s.equals("-")) {
                Assertions.assertEquals("-12", me.solve());
            } else {
                Assertions.assertEquals("13", me.solve());
            }
        }
    }

    @Test
    void ceilNingAsExpected() {
        MathExpression me;
        for (String s : new String[]{"", "-", "--"}) {
            String r = "";
            if (s.equals("-")) {
                r = "-";
            }
            me = new MathExpression("ceilN(1," + s + "1)");
            Assertions.assertEquals(r + "1.0", me.solve());
            me = new MathExpression("ceilN(2," + s + "2)");
            Assertions.assertEquals(r + "2.00", me.solve());

            me = new MathExpression("ceilN(2," + s + "1.1)");
            Assertions.assertEquals(r + "1.10", me.solve());
            me = new MathExpression("ceilN(2," + s + "1.9)");
            Assertions.assertEquals(r + "1.90", me.solve());
            me = new MathExpression("ceilN(5," + s + "1.4)");
            Assertions.assertEquals(r + "1.40000", me.solve());
            me = new MathExpression("ceilN(6, " + s + "1.5)");
            Assertions.assertEquals(r + "1.500000", me.solve());

            me = new MathExpression("ceilN(1," + s + "1.44)");
            if (s.equals("-")) {
                Assertions.assertEquals(r + "1.4", me.solve());
            } else {
                Assertions.assertEquals(r + "1.5", me.solve());
            }
            me = new MathExpression("ceilN(1," + s + "1.45)");
            if (s.equals("-")) {
                Assertions.assertEquals(r + "1.4", me.solve());
            } else {
                Assertions.assertEquals(r + "1.5", me.solve());
            }

            me = new MathExpression("ceilN(1," + s + "1.444)");
            if (s.equals("-")) {
                Assertions.assertEquals(r + "1.4", me.solve());
            } else {
                Assertions.assertEquals(r + "1.5", me.solve());
            }
            me = new MathExpression("ceilN(1," + s + "1.445)");
            if (s.equals("-")) {
                Assertions.assertEquals(r + "1.4", me.solve());
            } else {
                Assertions.assertEquals(r + "1.5", me.solve());
            }
            me = new MathExpression("ceilN(2," + s + "1.444)");
            if (s.equals("-")) {
                Assertions.assertEquals(r + "1.44", me.solve());
            } else {
                Assertions.assertEquals(r + "1.45", me.solve());
            }
            me = new MathExpression("ceilN(2," + s + "1.445)");
            if (s.equals("-")) {
                Assertions.assertEquals(r + "1.44", me.solve());
            } else {
                Assertions.assertEquals(r + "1.45", me.solve());
            }

            me = new MathExpression("ceilN(2, " + s + "12.99999)");
            if (s.equals("-")) {
                Assertions.assertEquals("-12.99", me.solve());
            } else {
                Assertions.assertEquals("13.00", me.solve());
            }
        }

    }

    @Test
    void fracPreReq() {
        MathExpression me;
        me = new MathExpression("3/2");
        Assertions.assertEquals("1.5", me.solve());
        me = new MathExpression("(3/2)");
        Assertions.assertEquals("1.5", me.solve());
        me = new MathExpression("sin(5)");
        Assertions.assertEquals("-0.9589242746631385", me.solve());
        me = new MathExpression("sin(2+3)");
        Assertions.assertEquals("-0.9589242746631385", me.solve());
    }

    @Test
    void fracPreReqNonStat() {
        MathExpression me;
        me = new MathExpression("sin(3/2)");
        Assertions.assertEquals("0.9974949866040544", me.solve());
    }

    @Test
    void fracPreReqStat() {
        MathExpression me;
        me = new MathExpression("sum(3/2)");
        Assertions.assertEquals("1.5", me.solve());
    }

    @Test
    void ceilingAsExpectedExpr() {
        MathExpression me;
        me = new MathExpression("ceil(3/2)");
        Assertions.assertEquals("2", me.solve());
    }

    @Test
    void flooringAsExpectedFraExpr() {
        MathExpression me;
        me = new MathExpression("floor(3/2)");
        Assertions.assertEquals("1", me.solve());
    }



}
