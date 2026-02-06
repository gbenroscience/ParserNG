package com.github.gbenroscience.math.parser.methods.ext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import com.github.gbenroscience.parser.MathExpression;

import static org.junit.jupiter.api.Assertions.*;

class LengthsTest {

    @Test
    void lengthsTest() {
        MathExpression me;
        me = new MathExpression("lengthFractional(20)");
        Assertions.assertEquals("0", me.solve());
        me = new MathExpression("length(1)");
        Assertions.assertEquals("1", me.solve());
        me = new MathExpression("length(-1.35)");
        Assertions.assertEquals("3", me.solve());
        me = new MathExpression("lengthFractional(-1.35)");
        Assertions.assertEquals("2", me.solve());
        me = new MathExpression("lengthDecimal(-1.35)");
        Assertions.assertEquals("1", me.solve());
        me = new MathExpression("length(-1.35)");
        Assertions.assertEquals("3", me.solve());
        me = new MathExpression("length(1E5)");
        Assertions.assertEquals("6", me.solve());
        me = new MathExpression("length(-1E-5)");
        Assertions.assertEquals("5", me.solve());
        me = new MathExpression("lengthFractional(1E-5)");
        Assertions.assertEquals("5", me.solve());
        me = new MathExpression("lengthDecimal(-1E-5)");
        Assertions.assertEquals("0", me.solve());

    }
}