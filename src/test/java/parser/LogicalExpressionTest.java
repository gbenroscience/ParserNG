package parser;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import parser.logical.PrintingExpressionLogger;

class LogicalExpressionTest {

    private static PrintingExpressionLogger log = new PrintingExpressionLogger();

    @Test
    void solveNonsense() {
        LogicalExpression expr = new LogicalExpression("1+1 < (2+0)*1 impl [ [5 == 6 || 33<(22-20)*2 ] xor [ [  5-3 < 2 or 7*(5+2)<=5 ] and 1+1 == 2]] eq [ true && false ] ", log);
        Assertions.assertEquals("true",expr.solve());
        LogicalExpression exprPartx = new LogicalExpression("1+1 < (2+0)*1 impl   false", log);
        Assertions.assertEquals("true",exprPartx.solve());
        LogicalExpression exprFixedPart1 = new LogicalExpression("[1+1 < (2+0)*1 impl  false ]  eq  false", log);
        Assertions.assertEquals("false",exprFixedPart1.solve());
        LogicalExpression exprFixedPart2 = new LogicalExpression("1+1 < (2+0)*1 impl  [ false   eq  false]", log);
        Assertions.assertEquals("true",exprFixedPart2.solve());
    }

    @Test
    void solve() {
        Assertions.assertEquals("true", new LogicalExpression(" 1<2  && 1< 3", log).solve());
        Assertions.assertEquals("false", new LogicalExpression(" 1<2  && 1<=0", log).solve());
        Assertions.assertEquals("true", new LogicalExpression(" 1<2  || 1<=0", log).solve());

        Assertions.assertEquals("true", new LogicalExpression(" [  true  || false ] && [  true  && true ] " , log).solve());
        Assertions.assertEquals("false", new LogicalExpression(" [  true  && false ] && [  true  && true ] " , log).solve());
        Assertions.assertEquals("true", new LogicalExpression(" [  true  && false ] || [  true  && true ] " , log).solve());

        Assertions.assertEquals("true", new LogicalExpression(" [  true  || false ] &&  true " , log).solve());
        Assertions.assertEquals("true", new LogicalExpression("   true  || [ false &&  true ]" , log).solve());
        Assertions.assertEquals("false", new LogicalExpression(" [  false  || true ] &&  false " , log).solve());
        Assertions.assertEquals("false", new LogicalExpression("   false  || [ true &&  false ]" , log).solve());

        Assertions.assertEquals("false", new LogicalExpression(" [  false  impl true ] impl  false " , log).solve());
        Assertions.assertEquals("true" , new LogicalExpression("    false  impl [ true impl  false ]" , log).solve());
    }

    @Test
    void not() {
        LogicalExpression expr = new LogicalExpression("[true]", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("![true]", log);
        Assertions.assertEquals("false", expr.solve());
        expr = new LogicalExpression("[false]", log);
        Assertions.assertEquals("false", expr.solve());
        expr = new LogicalExpression("![false]", log);
        Assertions.assertEquals("true", expr.solve());
    }

    @Test
    void notWithSpaces() {
        LogicalExpression expr = new LogicalExpression("[ true]  ", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression(" ! [ true ]", log);
        Assertions.assertEquals("false", expr.solve());
        expr = new LogicalExpression("[false]", log);
        Assertions.assertEquals("false", expr.solve());
        expr = new LogicalExpression(" !   [false]", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("[False]", log);
        Assertions.assertEquals("false", expr.solve());
        expr = new LogicalExpression("TRUE", log);
        Assertions.assertEquals("true", expr.solve());
    }

    @Test
    void notMore() {
        LogicalExpression expr;
        expr = new LogicalExpression("![true] || ![false] ", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("![ true && false ]", log);
        Assertions.assertEquals("true", expr.solve());

        expr = new LogicalExpression("![![true] || ![false] ]", log);
        Assertions.assertEquals("false",expr.solve());
        expr = new LogicalExpression("![![ true && false ]]", log);
        Assertions.assertEquals("false", expr.solve());

        expr = new LogicalExpression("![true] || ![false]  eq  ![ true && false ]", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("![![true] || ![false]]  eq  ![![ true && false ]]", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("![![![true] || ![false]]  eq  ![![ true && false ]]]", log);
        Assertions.assertEquals("false", expr.solve());
    }

    @Test
    void notMoreWithSpaces() {
        LogicalExpression expr;
        expr = new LogicalExpression("![true] || !    [false] ", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("  !   [ true && false ]", log);
        Assertions.assertEquals("true", expr.solve());

        expr = new LogicalExpression("! [ ! [true] ||  ! [false] ]", log);
        Assertions.assertEquals("false", expr.solve());
        expr = new LogicalExpression("!   [![ true && false ]]", log);
        Assertions.assertEquals("false", expr.solve());

        expr = new LogicalExpression("   ![true] ||   ! [false]  eq  ! [ true && false ]", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("!   [ ! [true] || ![false]]  eq  !    [![ true && false ]]", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("![ ! [! [true] || ![false]]  eq  ![  ! [ true && false ]]]", log);
        Assertions.assertEquals("false", expr.solve());
    }

    @Test
    void variablesWorks() {
        LogicalExpression expr;
        expr = new LogicalExpression("q=3;q<q+1", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("[q<q+1 || [q=3;q<5]]", log);
        Assertions.assertEquals("true", expr.solve());
        expr = new LogicalExpression("[q=3;q<1] || [q<q+1 || [q<5]]", log);
        Assertions.assertEquals("true", expr.solve());
    }

    @Test
    void variablesDoNotWorks(){
        try {
            LogicalExpression expr;
            expr = new LogicalExpression("[s=3;s<s+1 || [s<5]]", log);
            String s = expr.solve();
            Assertions.assertEquals("Character s is neither a decimal digit number, decimal point, nor \"e\" notation exponential mark.", s);
        }catch(NumberFormatException ex){
            ex.printStackTrace();
            return;
        }
        Assertions.assertTrue(false, "should have thrown");
    }
}
