package parser.methods.ext;

import logic.DRG_MODE;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import parser.MathExpression;

public class SinTest {

    @Test
    void sinAcceptsOnePlainArgument() {
        MathExpression me;
        me = new MathExpression("sin(5)");
        Assertions.assertEquals("-0.9589242746631385", me.solve());
    }

    @Test
    void sinAcceptsOnePlainExpression() {
        MathExpression me;
        me = new MathExpression("sin(5+5)");
        Assertions.assertEquals("-0.5440211108893698", me.solve());
    }


    @Test
    void sinDontAcceptsTwoPlainArguments() {
        Assertions.assertThrows(RuntimeException.class,
                () -> {
                    MathExpression me = new MathExpression("sin(5,6)");
                    Assertions.assertEquals("A SYNTAX ERROR OCCURRED", me.solve());
                });
    }

    @Test
    void sinDontAcceptsTwoPlainExpressions() {
        Assertions.assertThrows(RuntimeException.class,
                () -> {
                    MathExpression me = new MathExpression("sin(5+5, 4+4)");
                    Assertions.assertEquals("A SYNTAX ERROR OCCURRED", me.solve());
                });
    }

    @Test
    void sinAcceptsStatFunction() {
        MathExpression me;
        me = new MathExpression("sin(sum(5))");
        Assertions.assertEquals("-0.9589242746631385", me.solve());
    }

    @Test
    void sinAcceptsNumFunction() {
        MathExpression me;
        me = new MathExpression("sin(sin(5))");
        Assertions.assertEquals("-0.8185741444617193", me.solve());
    }

    @Test
    void sinDontAcceptsTwoStatFunction() {
        Assertions.assertThrows(RuntimeException.class,
                () -> {
                    MathExpression me = new MathExpression("sin(sum(5),sum(6))");
                    Assertions.assertEquals("A SYNTAX ERROR OCCURRED", me.solve());
                });
    }

    @Test
    void sinDontAcceptsTwoNumFunction() {
        Assertions.assertThrows(RuntimeException.class,
                () -> {
                    MathExpression me = new MathExpression("sin(sin(5),sin(5))");
                    Assertions.assertEquals("A SYNTAX ERROR OCCURRED", me.solve());
                });
    }

    @Test()
    void sinDontAcceptsStatAndNumFunction() {

        Assertions.assertThrows(RuntimeException.class,
                () -> {
                    MathExpression me = new MathExpression("sin(sin(5),sum(5))");
                    Assertions.assertEquals("A SYNTAX ERROR OCCURRED", me.solve());
                });
        Assertions.assertThrows(RuntimeException.class,
                () -> {
                    MathExpression me = new MathExpression("sin(sum(5),sin(5))");
                    Assertions.assertEquals("A SYNTAX ERROR OCCURRED", me.solve());
                });
    }

    @Test
    void fromJunk() {
        MathExpression me = new MathExpression("x=10;sin(ln(x))");
        Assertions.assertTrue("0.743980336957493".equals(me.solve()) || "0.7439803369574931".equals(me.solve()));//different on jdk11+ and 8-
    }

    @Test
    void changeDegRadGradViaProperty() {
        try {
            MathExpression me;
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "DEG");
            me= new MathExpression("sin(1)");
            Assertions.assertEquals("0.01745240643728351", me.solve());
            me= new MathExpression("sin(1)");
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "RAD");
            me= new MathExpression("sin(1)");
            Assertions.assertEquals("0.8414709848078965", me.solve());
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "GRAD");
            me= new MathExpression("sin(1)");
            Assertions.assertEquals("0.015707317311820675", me.solve());
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "GRAD");
            me= new MathExpression("sin(1)");
            Assertions.assertEquals("0.015707317311820675", me.solve());
            System.setProperty(DRG_MODE.DEG_MODE_VARIABLE, "RAD");
            Assertions.assertEquals("0.015707317311820675", me.solve()); //do not affect existing instances
            me= new MathExpression("sin(1)"); //is affecting all future nstance
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
