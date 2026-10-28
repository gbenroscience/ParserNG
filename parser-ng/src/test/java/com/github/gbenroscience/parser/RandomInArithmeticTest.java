/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.parser;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@code rnd()} and {@code rnd(n)} return one number, so they must work as an
 * operand of ordinary arithmetic. They were listed with sort, mode and the
 * other list-returning methods, so the verifier rejected {@code rnd()+1},
 * {@code 2*rnd()} and the like as "a list used as a number".
 */
public class RandomInArithmeticTest {

    private static double eval(String expr) {
        return Double.parseDouble(new MathExpression(expr).solve());
    }

    @Test
    @DisplayName("rnd() and rnd(n) work on either side of an operator, in brackets and as an argument")
    public void rndInArithmetic() {
        for (int i = 0; i < 200; i++) {
            double a = eval("rnd()+1");
            assertTrue(a >= 1.0 && a < 2.0, "rnd()+1 = " + a);
            double b = eval("1+rnd()");
            assertTrue(b >= 1.0 && b < 2.0, "1+rnd() = " + b);
            double c = eval("2*rnd()");
            assertTrue(c >= 0.0 && c < 2.0, "2*rnd() = " + c);
            double d = eval("(rnd()+1)");
            assertTrue(d >= 1.0 && d < 2.0, "(rnd()+1) = " + d);
            double e = eval("rnd(5)+1");
            assertTrue(e >= 1.0 && e < 6.0, "rnd(5)+1 = " + e);
        }
        assertEquals(0.0, eval("rnd()*0"), 0.0);
        assertEquals(1.0, eval("sin(rnd()*0)+1"), 0.0);
        assertEquals(5.0, eval("sum(rnd()*0,2,3)"), 0.0, "as an argument of a statistics method");
    }

    @Test
    @DisplayName("Compiled expressions agree")
    public void rndInCompiledArithmetic() throws Throwable {
        double v = new MathExpression("rnd()*0+1+rnd(1)*0").compileTurbo().applyScalar(new double[0]);
        assertEquals(1.0, v, 0.0);
    }

    @Test
    @DisplayName("Genuine lists are still not allowed as numbers")
    public void listsStillRejected() {
        assertEquals("[1.0, 2.0, 3.0]", new MathExpression("sort(3,1,2)").solve());
        assertEquals("SYNTAX ERROR", new MathExpression("sort(3,1,2)+1").solve());
        assertEquals("SYNTAX ERROR", new MathExpression("1+sort(3,1,2)").solve());
    }
}