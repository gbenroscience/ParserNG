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

import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.FunctionScope;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Zero-argument user functions, e.g. {@code check()=5}, must be inlined by the
 * scanner like any other algebraic function.
 * <p>
 * They used to fail because the bracket cleaner deleted every empty "()" (its
 * "redundant nested brackets" rule matched an empty pair), so the inliner only
 * ever saw a bare name. Global names are prefixed {@code zaf} so the tests do
 * not collide with anything else in a shared JVM.
 */
public class ZeroArgFunctionTest {

    private static double eval(String expr) {
        return Double.parseDouble(new MathExpression(expr).solve());
    }

    private static double eval(String expr, FunctionScope scope) {
        return Double.parseDouble(new MathExpression(expr, scope).solve());
    }

    @Test
    @DisplayName("The bracket cleaner keeps the empty brackets after a name, and still folds real redundant ones")
    public void bracketCleanerKeepsEmptyBrackets() {
        List<String> a = new ArrayList<>(Arrays.asList("zafa", "(", ")"));
        MathScanner.removeExcessBrackets(a);
        assertEquals(Arrays.asList("zafa", "(", ")"), a);

        List<String> b = new ArrayList<>(Arrays.asList("(", "2", "*", "zafa", "(", ")", ")"));
        MathScanner.removeExcessBrackets(b);
        assertEquals(Arrays.asList("(", "2", "*", "zafa", "(", ")", ")"), b);

        // the rule the bug lived in must still do its real job
        List<String> c = new ArrayList<>(Arrays.asList("(", "(", "x", "+", "1", ")", ")"));
        MathScanner.removeExcessBrackets(c);
        assertEquals(Arrays.asList("(", "x", "+", "1", ")"), c);
    }

    @Test
    @DisplayName("A zero-argument function is inlined in every position")
    public void zeroArgFunctionsInline() {
        FunctionManager.add("zafcheck()=5");
        FunctionManager.add("zafadapt()=2+3*4");
        assertEquals(5.0, eval("zafcheck()"), 0.0);
        assertEquals(6.0, eval("zafcheck()+1"), 0.0);
        assertEquals(10.0, eval("2*zafcheck()"), 0.0);
        assertEquals(14.0, eval("zafadapt()"), 0.0);
        assertEquals(28.0, eval("zafadapt()*2"), 0.0, "the body keeps its own precedence: (2+3*4)*2");
        assertEquals(19.0, eval("zafcheck()+zafadapt()"), 0.0);
        assertEquals(25.0, eval("zafcheck()^2"), 0.0);
        assertEquals(-5.0, eval("-zafcheck()"), 0.0);
        assertEquals(Math.sin(5.0), eval("sin(zafcheck())"), 1e-15, "as an argument of a built-in");
    }

    @Test
    @DisplayName("A zero-argument function inside another function's body")
    public void zeroArgInsideBodies() {
        FunctionManager.add("zafbase()=5");
        FunctionManager.add("zafk(x)=x*zafbase()");
        assertEquals(15.0, eval("zafk(3)"), 0.0, "constant argument");
        assertEquals(15.0, eval("x=3;zafk(x)"), 0.0, "variable argument");
        FunctionManager.add("zafz()=zafbase()+1");
        assertEquals(6.0, eval("zafz()"), 0.0, "zero-arg calling zero-arg");
    }

    @Test
    @DisplayName("A zero-argument function reads variables live, not as a parse-time snapshot")
    public void zeroArgReadsLiveVariables() throws Throwable {
        FunctionManager.add("zaflive()=2*zafv");
        MathExpression me = new MathExpression("zafv=3;zaflive()+1");
        assertEquals(7.0, Double.parseDouble(me.solve()), 0.0);
    }

    @Test
    @DisplayName("Compiled (turbo) expressions inline zero-argument functions too")
    public void zeroArgInTurbo() throws Throwable {
        FunctionManager.add("zafturbo()=7");
        MathExpression me = new MathExpression("zafturbo()*2+1");
        assertEquals(15.0, me.compileTurbo().applyScalar(new double[0]), 0.0);
    }

    @Test
    @DisplayName("Calling a function with arguments as if it had none is reported, with the right count")
    public void wrongArityIsReported() {
        FunctionManager.add("zafone(x)=x+1");
        try {
            new MathExpression("zafone()");
            fail("zafone needs one argument, so zafone() must be rejected");
        } catch (RuntimeException e) {
            assertTrue(e.getMessage().contains("requires `1` parameters, but 0"), e.getMessage());
        }
    }

    @Test
    @DisplayName("Zero-argument functions are scoped like any other function")
    public void zeroArgFunctionsAreScoped() {
        FunctionManager.add("zafsc()=1");
        FunctionScope mine = FunctionScope.create();
        mine.add("zafsc()=100");
        assertEquals(100.0, eval("zafsc()", mine), 0.0, "the scope's own definition wins");
        assertEquals(1.0, eval("zafsc()"), 0.0, "the global one is untouched");
        assertEquals(200.0, eval("zafsc()*2", mine), 0.0);

        // lexical: a global function calling the zero-arg function keeps the global one
        FunctionManager.add("zafuse(x)=x+zafsc()");
        assertEquals(11.0, eval("zafuse(10)"), 0.0);
        assertEquals(11.0, eval("zafuse(10)", mine), 0.0, "a global caller is not hijacked by the scope's zafsc");
    }

    @Test
    @DisplayName("A user function and a built-in may share a name when their argument counts differ")
    public void userFunctionAndBuiltInShareAName() throws Throwable {
        // cos() is the user's (no arguments); cos(2) inside its own body, and cos(0) anywhere, are the built-in.
        FunctionScope mine = FunctionScope.create();
        mine.add("cos()=(4+cos(2))");
        double user = 4 + Math.cos(2);
        assertEquals(user, eval("cos()", mine), 1e-15, "the user's zero-argument cos()");
        assertEquals(user + 1, eval("cos()+1", mine), 1e-15);
        assertEquals(1.0, eval("cos(0)", mine), 0.0, "the built-in cos(x) still works");
        assertEquals(user + 1, eval("cos()+cos(0)", mine), 1e-15, "both in one expression");
        assertEquals(1.0, eval("x=0;cos(x)", mine), 0.0, "built-in with a variable argument");
        assertEquals(user + 1, new MathExpression("cos()+cos(0)", mine).compileTurbo().applyScalar(new double[0]), 1e-15, "compiled");
        mine.add("cos()=(4+cos(2))");   // redefining it must not trip over the earlier definition
        assertEquals(user, eval("cos()", mine), 1e-15);

        // the global registry was never touched
        assertEquals(1.0, eval("cos(0)"), 0.0);
        assertNull(FunctionManager.lookUp("cos"));
    }

    @Test
    @DisplayName("Same-arity user functions still override a built-in, and a bad count with no built-in is still an error")
    public void overrideAndErrorRulesUnchanged() {
        FunctionScope mine = FunctionScope.create();
        mine.add("sin(x)=x*2");
        assertEquals(6.0, eval("sin(3)", mine), 0.0, "same argument count: the user's function wins");
        assertTrue(Math.abs(eval("sin(3)") - Math.sin(3)) < 1e-15, "outside the scope the built-in is untouched");
        mine.add("zafonly(x)=x+1");
        try {
            new MathExpression("zafonly()", mine);
            fail("no built-in named zafonly, so zafonly() is an arity error");
        } catch (RuntimeException e) {
            assertTrue(e.getMessage().contains("requires `1` parameters, but 0"), e.getMessage());
        }
    }
}