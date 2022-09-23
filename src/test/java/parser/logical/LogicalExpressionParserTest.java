package parser.logical;

import org.junit.jupiter.api.Assertions;

import java.util.List;

class LogicalExpressionParserTest {

    private static PrintingExpressionLogger log = new PrintingExpressionLogger();

    @org.junit.jupiter.api.Test
    void splitTest1() {
        List<String> s;
        s = new LogicalExpressionParser("not important now", log).split("1+2+3 < 5");
        Assertions.assertEquals(1, s.size());
        Assertions.assertEquals("1+2+3 < 5", s.get(0));
        s = new LogicalExpressionParser("not important now", log).split("1+2+3<5 & 2+5>7");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3<5", s.get(0));
        Assertions.assertEquals("&", s.get(1));
        Assertions.assertEquals("2+5>7", s.get(2));
        s = new LogicalExpressionParser("not important now", log).split("1+2+3<5    &&&&2+5>7");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3<5", s.get(0));
        Assertions.assertEquals("&", s.get(1));
        Assertions.assertEquals("2+5>7", s.get(2));
        s = new LogicalExpressionParser("not important now", log).split("1+2+3<5 | 2+5>7");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3<5", s.get(0));
        Assertions.assertEquals("|", s.get(1));
        Assertions.assertEquals("2+5>7", s.get(2));
        s = new LogicalExpressionParser("not important now", log).split("1+2+3<5||||    2+5>7");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3<5", s.get(0));
        Assertions.assertEquals("|", s.get(1));
        Assertions.assertEquals("2+5>7", s.get(2));
    }

    @org.junit.jupiter.api.Test
    void splitTest2() {
        List<String> s;
        s = new LogicalExpressionParser("not important now", log).split("1+2+3 < 5");
        Assertions.assertEquals(1, s.size());
        Assertions.assertEquals("1+2+3 < 5", s.get(0));
        s = new LogicalExpressionParser("not important now", log).split("1+2+3<5 & 2+5>7 | 5<6");
        Assertions.assertEquals(5, s.size());
        Assertions.assertEquals("1+2+3<5", s.get(0));
        Assertions.assertEquals("&", s.get(1));
        Assertions.assertEquals("2+5>7", s.get(2));
        Assertions.assertEquals("|", s.get(3));
        Assertions.assertEquals("5<6", s.get(4));
        s = new LogicalExpressionParser("not important now", log).split("1+2+3<5 & 2+5>7 & 5<6");
        Assertions.assertEquals(5, s.size());
        Assertions.assertEquals("1+2+3<5", s.get(0));
        Assertions.assertEquals("&", s.get(1));
        Assertions.assertEquals("2+5>7", s.get(2));
        Assertions.assertEquals("&", s.get(3));
        Assertions.assertEquals("5<6", s.get(4));
        s = new LogicalExpressionParser("not important now", log).split("1+2+3<5 | 2+5>7 & 5<6");
        Assertions.assertEquals(5, s.size());
        Assertions.assertEquals("1+2+3<5", s.get(0));
        Assertions.assertEquals("|", s.get(1));
        Assertions.assertEquals("2+5>7", s.get(2));
        Assertions.assertEquals("&", s.get(3));
        Assertions.assertEquals("5<6", s.get(4));
        s = new LogicalExpressionParser("not important now", log).split("1+2+3<5 | 2+5>7 | 5<6");
        Assertions.assertEquals(5, s.size());
        Assertions.assertEquals("1+2+3<5", s.get(0));
        Assertions.assertEquals("|", s.get(1));
        Assertions.assertEquals("2+5>7", s.get(2));
        Assertions.assertEquals("|", s.get(3));
        Assertions.assertEquals("5<6", s.get(4));
        s = new LogicalExpressionParser("not important now", log).split("1+2+3<5    &&&&2+5>7 & 5<7 & 1+2+3<5||||    2+5>7");
        Assertions.assertEquals(9, s.size());
        Assertions.assertEquals("1+2+3<5", s.get(0));
        Assertions.assertEquals("&", s.get(1));
        Assertions.assertEquals("2+5>7", s.get(2));
        Assertions.assertEquals("&", s.get(3));
        Assertions.assertEquals("5<7", s.get(4));
        Assertions.assertEquals("&", s.get(5));
        Assertions.assertEquals("1+2+3<5", s.get(6));
        Assertions.assertEquals("|", s.get(7));
        Assertions.assertEquals("2+5>7", s.get(8));
    }

    @org.junit.jupiter.api.Test
    void evalTest() {
        LogicalExpressionParser comp;
        comp = new LogicalExpressionParser("1+2+3 >= 7", log);
        Assertions.assertFalse(comp.evaluate());
        comp = new LogicalExpressionParser("1+2+3 >= 5", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new LogicalExpressionParser("1+2+3 >= 7 | 1+2+3 >= 5", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new LogicalExpressionParser("6 >= 7 & 6 >= 5", log);
        Assertions.assertFalse(comp.evaluate());
    }

    @org.junit.jupiter.api.Test
    void andTable() {
        Assertions.assertTrue(new LogicalExpressionParser("true and true", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("false and true", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("false and false", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("true and false", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("true & true", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("false && true", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("false &&& false", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("true &&&& false", log).evaluate());
    }

    @org.junit.jupiter.api.Test
    void orTable() {
        Assertions.assertTrue(new LogicalExpressionParser("true or true", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("false or true", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("false or false", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("true or false", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("true | true", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("false || true", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("false ||| false", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("true |||| false", log).evaluate());
    }

    @org.junit.jupiter.api.Test
    void xorTable() {
        Assertions.assertFalse(new LogicalExpressionParser("true xor true", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("false xor true", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("false xor false", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("true xor false", log).evaluate());
    }

    @org.junit.jupiter.api.Test
    void eqTable() {
        Assertions.assertTrue(new LogicalExpressionParser("true eq true", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("false eq true", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("false eq false", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("true eq false", log).evaluate());
    }

    @org.junit.jupiter.api.Test
    void implTable() {
        Assertions.assertTrue(new LogicalExpressionParser("true imp true", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("false imp true", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("false imp false", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("true imp false", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("true impl true", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("false impl true", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("false impl false", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("true impl false", log).evaluate());
    }


    @org.junit.jupiter.api.Test
    void eqTableWithEq() {
        Assertions.assertTrue(new LogicalExpressionParser("0 == 0  eq 0 == 0", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("0 != 0 eq 0 == 0", log).evaluate());
        Assertions.assertTrue(new LogicalExpressionParser("0 != 0  eq 0 != 0 ", log).evaluate());
        Assertions.assertFalse(new LogicalExpressionParser("0 == 0  eq 0 != 0 ", log).evaluate());
/**
 * For a hwile, there were eq/neq and -eq/-neq operators
 * But those colidate with logical eq and thus were removed
 * workarounds to make avaiblable both were sucide
 */
        int ex = 0;
        try {
            new LogicalExpressionParser("0 -eq 0 eq 0 -eq 0", log).evaluate();
        } catch (Exception e) {
            ex++;
        }
        try {
            new LogicalExpressionParser("0 -neq 0 eq 0 -eq 0", log).evaluate();
        } catch (Exception e) {
            ex++;
        }
        try {
            new LogicalExpressionParser("0 neq 0  eq 0 neq 0 ", log).evaluate();
        } catch (Exception e) {
            ex++;
        }
        try {
            new LogicalExpressionParser("0 eq 0  eq 0 neq 0 ", log).evaluate();
        } catch (Exception e) {
            ex++;
        }
        Assertions.assertEquals(4, ex, "four exceptions expected");
    }

}
