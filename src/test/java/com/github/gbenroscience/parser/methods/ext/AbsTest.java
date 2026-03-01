package com.github.gbenroscience.parser.methods.ext;
 

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import com.github.gbenroscience.parser.MathExpression;

class AbsTest {

    @Test
    void absWorks() {
        MathExpression me;
        me = new MathExpression("abs(1)");
        Assertions.assertEquals(com.github.gbenroscience.parser.Number.fastParseDouble("1"), com.github.gbenroscience.parser.Number.fastParseDouble(me.solve()));
        me = new MathExpression("abs(1.1259)");
        Assertions.assertEquals(com.github.gbenroscience.parser.Number.fastParseDouble("1.1259"), com.github.gbenroscience.parser.Number.fastParseDouble(me.solve()));
        me = new MathExpression("abs(-1)");
        Assertions.assertEquals(com.github.gbenroscience.parser.Number.fastParseDouble("1.0"), com.github.gbenroscience.parser.Number.fastParseDouble(me.solve()));
        me = new MathExpression("abs(-1.1259)");
        Assertions.assertEquals(com.github.gbenroscience.parser.Number.fastParseDouble("1.1259"), com.github.gbenroscience.parser.Number.fastParseDouble(me.solve()));
    }
}