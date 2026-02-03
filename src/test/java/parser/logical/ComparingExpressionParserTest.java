package parser.logical;

import org.junit.jupiter.api.Assertions;

import java.util.List;

class ComparingExpressionParserTest {

    private static PrintingExpressionLogger log = new PrintingExpressionLogger();

    @org.junit.jupiter.api.Test
    void splitTest1() {
        List<String> s;
        ComparingExpressionParser comp;
        s = new ComparingExpressionParser("not important now", log).split("1+2+3 < 5");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3", s.get(0));
        Assertions.assertEquals("<", s.get(1));
        Assertions.assertEquals("5", s.get(2));
        s = new ComparingExpressionParser("not important now", log).split("1+2+3 > 5");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3", s.get(0));
        Assertions.assertEquals(">", s.get(1));
        Assertions.assertEquals("5", s.get(2));
        s = new ComparingExpressionParser("not important now", log).split("1+2+3 == 5");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3", s.get(0));
        Assertions.assertEquals("==", s.get(1));
        Assertions.assertEquals("5", s.get(2));
        s = new ComparingExpressionParser("not important now", log).split("1+2+3 != 5");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3", s.get(0));
        Assertions.assertEquals("!=", s.get(1));
        Assertions.assertEquals("5", s.get(2));
        s = new ComparingExpressionParser("not important now", log).split("1+2+3 >= 5");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3", s.get(0));
        Assertions.assertEquals(">=", s.get(1));
        Assertions.assertEquals("5", s.get(2));
        s = new ComparingExpressionParser("not important now", log).split("1+2+3 <= 5");
        Assertions.assertEquals(3, s.size());
        Assertions.assertEquals("1+2+3", s.get(0));
        Assertions.assertEquals("<=", s.get(1));
        Assertions.assertEquals("5", s.get(2));

    }

    @org.junit.jupiter.api.Test
    void evalTest() {
        ComparingExpressionParser comp;
        comp = new ComparingExpressionParser("1+2+3 < 2*2", log);
        Assertions.assertFalse(comp.evaluate());
        comp = new ComparingExpressionParser("1+2+3 > 2*2", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ComparingExpressionParser("1+2 <= 3*2", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ComparingExpressionParser("1+2+3 <= 3*2", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ComparingExpressionParser("30 <= 3*2", log);
        Assertions.assertFalse(comp.evaluate());
        comp = new ComparingExpressionParser("1+2+3 == 3*2", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ComparingExpressionParser("1+2+3 == 3*2+1", log);
        Assertions.assertFalse(comp.evaluate());
        comp = new ComparingExpressionParser("1+2+3 != 3*2+1", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ComparingExpressionParser("10 >= 6", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ComparingExpressionParser("1+2+3 >= 6", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ComparingExpressionParser("1+2+3 >= 7", log);
        Assertions.assertFalse(comp.evaluate());

        comp = new ComparingExpressionParser("  true", log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ComparingExpressionParser("false  ", log);
        Assertions.assertFalse(comp.evaluate());
    }

}