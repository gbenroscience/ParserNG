package com.github.gbenroscience.parser.expanding;

import com.github.gbenroscience.parser.expanding.ExpandingExpressionParser;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import com.github.gbenroscience.parser.logical.PrintingExpressionLogger;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Nit for REVERT
 * The poins in the chart, are ordered as
 * Ln Ln-1 ...L4 L3 L2 L1 L0
 * Unluckily lists and arrays are creating structures indexed as 0,1,2,3,...n-1,n
 * Which is surprisingly hard to keep rotating in mind
 *
 * So each input array is comming in declared as 0,1,2...N and then it is reverted before passing in,
 * so it behaves naturally  - we see it as real N .. 1,0  and parser is reading it as declared before revert, thuis 0,1..N
 *
 */
class ExpandingExpressionParserTest {

    private static PrintingExpressionLogger log = new PrintingExpressionLogger();

    @org.junit.jupiter.api.Test
    void expandAll() {
        String s = "avg(..L1)*1.1 <  L0 | L1*1.3 <  L0 ";
        ExpandingExpressionParser comp = new ExpandingExpressionParser(s, revert(Arrays.asList("60", "20", "80", "70")), log);
        String r = comp.getExpanded();
        Assertions.assertEquals("avg(60,20,80)*1.1 <  70 | 80*1.3 <  70 ", r);
        Assertions.assertTrue(comp.evaluate());
    }

    @org.junit.jupiter.api.Test
    void expandAMN() {
        String s = "avg(..L1)*1.1-MN <  L0 | L1*1.3 + MN<  L0 ";
        ExpandingExpressionParser comp = new ExpandingExpressionParser(s, revert(Arrays.asList("60", "20", "80", "70")), log);
        String r = comp.getExpanded();
        Assertions.assertEquals("avg(60,20,80)*1.1-4 <  70 | 80*1.3 + 4<  70 ", r);
        Assertions.assertTrue(comp.evaluate());

    }

    @org.junit.jupiter.api.Test
    void expandLLsTest() {
        String s;
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLL("L1..L2");
        Assertions.assertEquals("2,1", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLL("L2..L1");
        Assertions.assertEquals("1,2", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLL("L0..L0");
        Assertions.assertEquals("3", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLL("L1..L125");
        Assertions.assertEquals("2,1", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLL("L123..L125");
        Assertions.assertEquals("1", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLL("L5 L0..L3  L1   L3..L0  L2 L2..L2 L1..L1");
        Assertions.assertEquals("L5 3,2,1  L1   1,2,3  L2 1 2", s);
    }


    @org.junit.jupiter.api.Test
    void expandLdTest() {
        String s;
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLd("..L5");
        Assertions.assertEquals("1", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLd("..L2");
        Assertions.assertEquals("1", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLd("..L1");
        Assertions.assertEquals("1,2", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLd("..L0");
        Assertions.assertEquals("1,2,3", s);
    }

    @org.junit.jupiter.api.Test
    void expandLuTest() {
        String s;
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLu("L5..");
        Assertions.assertEquals("1,2,3", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLu("L2..");
        Assertions.assertEquals("1,2,3", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLu("L1..");
        Assertions.assertEquals("2,3", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandLu("L0..");
        Assertions.assertEquals("3", s);
    }

    @org.junit.jupiter.api.Test
    void expandLTest() {
        String s;
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandL("L5");
        Assertions.assertEquals("1", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandL("L0");
        Assertions.assertEquals("3", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandL("L1");
        Assertions.assertEquals("2", s);
        s = new ExpandingExpressionParser("not important now", revert(Arrays.asList("1", "2", "3")), log).expandL("L2");
        Assertions.assertEquals("1", s);
    }

    @org.junit.jupiter.api.Test
    void testEval() {
        ExpandingExpressionParser comp;
        comp = new ExpandingExpressionParser("max(L0..) == 3", revert(Arrays.asList("1", "2", "3")), log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ExpandingExpressionParser("min(L0..L3)<= 1", revert(Arrays.asList("1", "2", "3")), log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ExpandingExpressionParser("avg(..L1) <  L2/2+1", revert(Arrays.asList("1", "2", "3")), log);
        Assertions.assertFalse(comp.evaluate());
        comp = new ExpandingExpressionParser("sum(..L1) ==  L2 | false ", revert(Arrays.asList("1", "2", "3")), log);
        Assertions.assertFalse(comp.evaluate());
        comp = new ExpandingExpressionParser("sum(..L1) ==  L2 & false ", revert(Arrays.asList("1", "2", "3")), log);
        Assertions.assertFalse(comp.evaluate());
    }

    static List<String> revert(List<String> l){
        Collections.reverse(l);
        return l;
    }
    @org.junit.jupiter.api.Test
    /**
     * The difference here is that we use real conditions
     * and the revert, so L.. and ..L make now sense
     * the revert is here (and in main imp in GenericChartPublisher),
     * as the chart draw points as Lx...L2...L1..L0, not naturay,
     * as the ExpandingExpressionParser expects)
     */
    void testRealLive() {
        ExpandingExpressionParser comp;
        comp = new ExpandingExpressionParser("L1 < L0", revert(Arrays.asList("628", "453", "545")), log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ExpandingExpressionParser("avg(..L1) < L0", revert(Arrays.asList("5", "628", "453", "545")), log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ExpandingExpressionParser("max(..L1) < L0 || 500 < L0",  revert(Arrays.asList("5", "628", "453", "545")), log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ExpandingExpressionParser("500 < L0 && 600 > L0",  revert(Arrays.asList("5", "628", "453", "545")), log);
        Assertions.assertTrue(comp.evaluate());
        comp = new ExpandingExpressionParser("avg(..L1)*1.1 <  L0 | L1*1.3 <  L0 ", revert(Arrays.asList("60", "20", "45", "70")), log);
        Assertions.assertTrue(comp.evaluate());
    }

    @Test
    void testRealLiveCurl() {
        ExpandingExpressionParser comp;
        comp = new ExpandingExpressionParser("avg(..L{MN/2}) < avg(L{MN/2}..)", revert(Arrays.asList("2", "4", "6")), log);
        Assertions.assertEquals("true", comp.solve());
        comp = new ExpandingExpressionParser("L{1+L{1+0}}+L{-2+MN/0.8}", revert(Arrays.asList("22", "0", "66")), log);
        Assertions.assertEquals("0.0", comp.solve());
        comp = new ExpandingExpressionParser("L{{1}}", revert(Arrays.asList("2", "4", "6")), log);
        Assertions.assertEquals("4", comp.solve());
        comp = new ExpandingExpressionParser("L{{MN/2}}", revert(Arrays.asList("2", "4", "6")), log);
        Assertions.assertEquals("4", comp.solve());
    }

    @Test
    void testRealLiveWithBrackets() {
        ExpandingExpressionParser comp;
        comp = new ExpandingExpressionParser(" [[ avg(..L1)*1.1 <  L0 ] || [L1*1.3 <  L0 ]] || [ avgN(count(..L0)/4, ..L1)*1.1<L0 ] ", revert(Arrays.asList("60", "20", "45", "70")), log);
        Assertions.assertTrue(comp.evaluate());

    }
}