package com.github.gbenroscience.math.parser.methods;

import com.github.gbenroscience.parser.methods.BasicNumericalMethod;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.logic.DRG_MODE;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.methods.ext.Utils;

import java.math.BigDecimal;
import java.util.List;

class DeclarationsTest {
    @Test
    void fromJunk() {
        MathExpression me = new MathExpression("x=10;sin(ln(x))");
        Assertions.assertTrue("0.743980336957493".equals(me.solve()) || "0.7439803369574931".equals(me.solve()));//different on jdk11+ and 8-
    }

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

    @Test
    void changeDegRadGradViaProperty() {
        try {
            MathExpression me;
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "DEG");
            me = new MathExpression("sin(1)");
            Assertions.assertEquals("0.01745240643728351", me.solve());
            me = new MathExpression("sin(1)");
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "RAD");
            me = new MathExpression("sin(1)");
            Assertions.assertEquals("0.8414709848078965", me.solve());
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "GRAD");
            me = new MathExpression("sin(1)");
            Assertions.assertEquals("0.015707317311820675", me.solve());
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "GRAD");
            me = new MathExpression("sin(1)");
            Assertions.assertEquals("0.015707317311820675", me.solve());
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "RAD");
            Assertions.assertEquals("0.015707317311820675", me.solve()); //do not affect existing instances
            me = new MathExpression("sin(1)"); //is affecting all future nstance
            Assertions.assertEquals("0.8414709848078965", me.solve());
        } finally {
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "RAD");
        }
   }

    @Test
    void changeDegRadGradViaSetter() {
        try {
            MathExpression meR = new MathExpression("sin(1)");
            meR.setDRG(DRG_MODE.RAD);
            Assertions.assertEquals("0.8414709848078965", meR.solve());
            MathExpression meG = new MathExpression("sin(1)");
            meG.setDRG(DRG_MODE.GRAD);
            Assertions.assertEquals("0.015707317311820675", meG.solve());
            MathExpression meD = new MathExpression("sin(1)");
            meD.setDRG(DRG_MODE.DEG);
            Assertions.assertEquals("0.01745240643728351", meD.solve());
            Assertions.assertEquals("0.8414709848078965", meR.solve());
            Assertions.assertEquals("0.015707317311820675", meG.solve());
            Assertions.assertEquals("0.01745240643728351", meD.solve());
        } finally {
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "RAD");
        }
    }

}
