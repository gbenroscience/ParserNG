package parser.methods;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import parser.MathExpression;
import parser.TYPE;
import parser.methods.ext.Utils;

import java.math.BigDecimal;
import java.util.List;

class DeclarationsTest {

    @Test
    void hardcodedFunctionCanBeOverwritten() {
        MathExpression me;
        me = new MathExpression("prod(1+1,2+2)");
        Assertions.assertEquals("8.0", me.solve());
        me = new MathExpression("prod(1+1)");
        Assertions.assertEquals("2.0", me.solve());
        BasicNumericalMethod override = new BasicNumericalMethod() {
            @Override
            public String solve(List<String> tokens) {
                List<BigDecimal> numericalTokens = Utils.tokensToNumbers(tokens);
                BigDecimal result = new BigDecimal("1");
                for(BigDecimal numericalToken: numericalTokens){
                    result = result.multiply(numericalToken);
                }
                return result.toString();
            }

            @Override
            public String getHelp() {
                return "fake prod";
            }

            @Override
            public String getName() {
                return "prod";
            }

            @Override
            public String getType() {
                return TYPE.NUMBER.toString();
            }
        };
        int before = Declarations.getInbuiltMethods().length;
        Declarations.registerBasicNumericalMethod(override);
        int during = Declarations.getInbuiltMethods().length;
        try {
            me = new MathExpression("prod(1+1)");
            Assertions.assertEquals("2.0", me.solve());
        } finally {
            Declarations.unregisterBasicNumericalMethod(override.getClass());
        }
        int after = Declarations.getInbuiltMethods().length;
        me = new MathExpression("prod(1+1)");
        Assertions.assertEquals("2.0", me.solve());
        Assertions.assertEquals(before, after);
        Assertions.assertEquals(before+1, during);
    }

}