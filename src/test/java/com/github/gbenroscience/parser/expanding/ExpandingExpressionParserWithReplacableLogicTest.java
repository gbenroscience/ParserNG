package com.github.gbenroscience.parser.expanding;

import com.github.gbenroscience.parser.expanding.ExpandingExpressionParser;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;


import java.util.Arrays;

import com.github.gbenroscience.parser.ExpandingExpression;
import com.github.gbenroscience.parser.logical.LogicalExpressionFactoryTest;
import com.github.gbenroscience.parser.logical.PrintingExpressionLogger;
import static com.github.gbenroscience.parser.expanding.ExpandingExpressionParserTest.revert;

class ExpandingExpressionParserWithReplacableLogicTest {

    private static PrintingExpressionLogger log = new PrintingExpressionLogger();

    @Test
    void weirdRegressions() {
        Assertions.assertTrue(new ExpandingExpressionParser("1+1 < ((2+0)*1) impl  false  eq  false", revert(Arrays.asList("1", "2", "3")), log).evaluate());
        Assertions.assertFalse(new ExpandingExpressionParser("L0<L1 and L2<L3", revert(Arrays.asList("0", "1", "-1", "2")), log).evaluate());
        Assertions.assertTrue(new ExpandingExpressionParser("L0<L1 and L2<L3", (Arrays.asList("0", "1", "-1", "2")), log).evaluate());
    }

    @Test
    void replacableLogicalExpressionWithExpansion() {
        //correct
        Assertions.assertTrue(new ExpandingExpressionParser("L0<L1 and L2<L3", (Arrays.asList("0", "1", "-1", "2")), log).evaluate());
        Assertions.assertFalse(new ExpandingExpressionParser("L0>L1 and L2>L3", (Arrays.asList("0", "1", "-1", "2")), log).evaluate());
        //custom
        Exception ex = null;
        try {
            Assertions.assertTrue(new ExpandingExpressionParser("L0>L1 and L2>L3", (Arrays.asList("0", "1", "-1", "2")), log, new LogicalExpressionFactoryTest.HogwartsMemberFactory()).evaluate());
        } catch (Exception ex1) {
            ex = ex1;
        }
        Assertions.assertTrue(new ExpandingExpressionParser("harry(L0..L2) and ron(L0..L2)", (Arrays.asList("0", "1", "-1", "2")), log, new LogicalExpressionFactoryTest.HogwartsMemberFactory()).evaluate());
        Assertions.assertFalse(new ExpandingExpressionParser("tom(L0..L2) and ron(L0..L2)", (Arrays.asList("0", "1", "-1", "2")), log, new LogicalExpressionFactoryTest.HogwartsMemberFactory()).evaluate());
        Assertions.assertNotNull(ex);
        //not affected
        Assertions.assertTrue(new ExpandingExpressionParser("true and true", (Arrays.asList("0", "1", "-1", "2")), log).evaluate());
        Assertions.assertFalse(new ExpandingExpressionParser("false and true", (Arrays.asList("0", "1", "-1", "2")), log).evaluate());
    }

    @Test
    void replacableLogicalExpressionBracketsWithExpansion() {
        //correct
        Assertions.assertEquals("true", new ExpandingExpression("[ L0<L1 and L2<L3 ] or [ L0<L1 and L2<L3 ] ", (Arrays.asList("0", "1", "-1", "2")), log).solve());
        Assertions.assertEquals("false", new ExpandingExpression("[ L0>L1 and L2<2L3] or [L0>L1 and L2<L3]", (Arrays.asList("0", "1", "-1", "2")), log).solve());
        String h1 = new ExpandingExpression("help", (Arrays.asList()), log).solve();
        //custom
        Exception ex = null;
        try {
            Assertions.assertEquals("true", new ExpandingExpression("[ L0>L1 and L2<2L3] or [L0>L1 and L2<L3]", (Arrays.asList("0", "1", "-1", "2")),
                    log, new LogicalExpressionFactoryTest.HogwartsMemberFactory()).solve());
        } catch (Exception ex1) {
            ex = ex1;
        }
        Assertions.assertNotNull(ex);
        String h2 = new ExpandingExpression("help", (Arrays.asList()),
                log, new LogicalExpressionFactoryTest.HogwartsMemberFactory()).solve();
        Assertions.assertEquals("true", new ExpandingExpression("[ harry(L3..L1) and ron(L3..L1)] or [ron(L3..L1) and tom(L3..L1)]", (Arrays.asList("0", "1", "-1", "2")),
                log, new LogicalExpressionFactoryTest.HogwartsMemberFactory()).solve());
        //not affected
        Assertions.assertEquals("true", new ExpandingExpression("[ L0<L1 and L2<L3 ] or [ L0<L1 and L2<L3 ] ", (Arrays.asList("0", "1", "-1", "2")), log).solve());
        Assertions.assertEquals("false", new ExpandingExpression("[ L0>L1 and L2<2L3] or [L0>L1 and L2<L3]", (Arrays.asList("0", "1", "-1", "2")), log).solve());
        String h3 = new ExpandingExpression("help", (Arrays.asList()), log).solve();
        Assertions.assertEquals(h3, h1);
        Assertions.assertEquals(h3, h2); //the embedded factory have currenlty no eefect here
    }

}