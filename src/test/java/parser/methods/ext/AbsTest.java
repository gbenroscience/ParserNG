package parser.methods.ext;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import parser.MathExpression;

class AbsTest {

    @Test
    void absWorks() {
        MathExpression me;
        me = new MathExpression("abs(1)");
        Assertions.assertEquals("1", me.solve());
        me = new MathExpression("abs(1.1259)");
        Assertions.assertEquals("1.1259", me.solve());
        me = new MathExpression("abs(-1)");
        Assertions.assertEquals("1.0", me.solve());
        me = new MathExpression("abs(-1.1259)");
        Assertions.assertEquals("1.1259", me.solve());
    }
}