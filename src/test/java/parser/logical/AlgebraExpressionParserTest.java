package parser.logical;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

class AlgebraExpressionParserTest {

    private static PrintingExpressionLogger log = new PrintingExpressionLogger();

    @Test
    void evaluateTest() {
        Assertions.assertEquals(0, new AlgebraExpressionParser("0", log).evaluate().compareTo(new BigDecimal(0)));
        Assertions.assertEquals(0, new AlgebraExpressionParser("2+3", log).evaluate().compareTo(new BigDecimal(5)));
        Assertions.assertEquals(0, new AlgebraExpressionParser("2+3*4", log).evaluate().compareTo(new BigDecimal(14)));
        Assertions.assertEquals(0, new AlgebraExpressionParser("(2+3)*4", log).evaluate().compareTo(new BigDecimal(20)));
        Assertions.assertEquals(0, new AlgebraExpressionParser("cos(pi)", log).evaluate().compareTo(new BigDecimal(-1)));
        Assertions.assertEquals(0, new AlgebraExpressionParser("sum(1,2,3,4)", log).evaluate().compareTo(new BigDecimal(10)));
        Assertions.assertEquals(0, new AlgebraExpressionParser("med(0,5,4)", log).evaluate().compareTo(new BigDecimal(4)));
        Assertions.assertEquals(0, new AlgebraExpressionParser("avg(0,5,4)", log).evaluate().compareTo(new BigDecimal(3)));
    }
}