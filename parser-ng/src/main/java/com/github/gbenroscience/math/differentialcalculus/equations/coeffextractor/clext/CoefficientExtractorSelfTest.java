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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;
 

import com.github.gbenroscience.math.Maths;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * @author GBEMIRO
 * Self-verifying regression harness for Lexer / ExprParser / ArgumentIsolator
 * / CoefficientExtractor. Every success case numerically evaluates the
 * extracted coefficients against independently computed expected values, not
 * just "did it throw or not". Run via main(); throws AssertionError with a
 * descriptive message on any mismatch.
 *
 * No compiler was available to run this in the environment it was written
 * in — every case was also traced by hand, but please actually compile and
 * run this before trusting CoefficientExtractor.
 */
public final class CoefficientExtractorSelfTest {

    private static int checks = 0;

    public static void main(String[] args) {
        testWorkedExampleFromFullScannedCall();
        testWorkedExampleFromRawStringMatchesScannedForm();
        testWorkedExampleFromBareEquationString();
        testForcingTermExtraction();
        testSparseMiddleOrderFillsWithZero();

        testTopDerivativeWrappedIsRejected();
        testDoubleNestedWrapIsRejected();
        testCrossOrderCouplingIsRejected();
        testSameOrderTwiceIsRejected();
        testDivisorStateIsRejected();
        testMultiArgFunctionWrapIsRejected();
        testNoStateReferenceAtAllIsRejected();
        testHardExampleMixedDerivativeTerms();

        System.out.println("All " + checks + " CoefficientExtractor checks passed.");
    }

    // ------------------------------------------------------------------
    // 1. The exact worked example, as the FULL scanned diffeqn(...) call —
    //    including the redundant outer grouping paren and the trailing
    //    t0/y0/method arguments that must be ignored.
    // ------------------------------------------------------------------
    private static void testWorkedExampleFromFullScannedCall() {
        List<String> scan = Arrays.asList(
                "(", "diffeqn", "(",
                "(", "(", "3", "*", "x", "^", "2", ")", "*", "y", "[", "4", "]",
                "+", "(", "5", "*", "sin", "(", "x", ")", ")", "*", "sin", "(", "y", "[", "3", "]", ")",
                "+", "(", "5", "/", "x", ")", "*", "ln", "(", "y", "[", "2", "]", ")",
                "-", "3", "*", "y", "[", "1", "]",
                "+", "3", "*", "x", "*", "y", "[", "0", "]",
                ")",
                ",", "1", ",", "0", ",", "anon1",
                ")", ")");

        CoefficientExtractor.Result r = CoefficientExtractor.extract(scan);
        assertWorkedExample(r);
        checks++;
    }

    // ------------------------------------------------------------------
    // 2. Same equation as a raw string, wrapped in a full diffeqn(...) call
    //    text — must agree exactly with the scanned-token-list result.
    // ------------------------------------------------------------------
    private static void testWorkedExampleFromRawStringMatchesScannedForm() {
        String raw = "diffeqn((3*x^2)*y[4]+(5*sin(x))*sin(y[3])+(5/x)*ln(y[2])-3*y[1]+3*x*y[0], 1, 0, anon1)";
        CoefficientExtractor.Result r = CoefficientExtractor.extract(raw);
        assertWorkedExample(r);
        checks++;
    }

    // ------------------------------------------------------------------
    // 3. Same equation as a bare string with no wrapping call at all.
    // ------------------------------------------------------------------
    private static void testWorkedExampleFromBareEquationString() {
        String raw = "(3*x^2)*y[4]+(5*sin(x))*sin(y[3])+(5/x)*ln(y[2])-3*y[1]+3*x*y[0]";
        CoefficientExtractor.Result r = CoefficientExtractor.extract(raw);
        assertWorkedExample(r);
        checks++;
    }

    private static void assertWorkedExample(CoefficientExtractor.Result r) {
        assertEquals("topOrder", 4, r.topOrder);
        assertEquals("term count", 5, r.terms.size());
        assertNull("no forcing term in this equation", r.forcingOrNull);

        double xv = 2.0; // sample point for the free variable, called "x" here
        assertClose("order 4 coefficient = 3*x^2", evalFree(r.coefficients[0], xv), 3 * xv * xv);
        assertNullString("order 4 must be unwrapped (linear)", r.wrappingFunctionNames[0]);

        assertClose("order 3 coefficient = 5*sin(x)", evalFree(r.coefficients[1], xv), 5 * Math.sin(xv));
        assertEqualsString("order 3 wrap = sin", "sin", r.wrappingFunctionNames[1]);

        assertClose("order 2 coefficient = 5/x", evalFree(r.coefficients[2], xv), 5 / xv);
        assertEqualsString("order 2 wrap = ln", "ln", r.wrappingFunctionNames[2]);

        assertClose("order 1 coefficient = -3", evalFree(r.coefficients[3], xv), -3.0);
        assertNullString("order 1 must be unwrapped", r.wrappingFunctionNames[3]);

        assertClose("order 0 coefficient = 3*x", evalFree(r.coefficients[4], xv), 3 * xv);
        assertNullString("order 0 must be unwrapped", r.wrappingFunctionNames[4]);
        System.out.println(r.toString());
    }

    // ------------------------------------------------------------------
    // 4. Forcing term extraction, with sign: y[1] + y[0] - 5*t = 0  =>  forcing = 5*t
    // ------------------------------------------------------------------
    private static void testForcingTermExtraction() {
        CoefficientExtractor.Result r = CoefficientExtractor.extract("y[1] + y[0] - 5*t");
        assertEquals("topOrder", 1, r.topOrder);

        double tv = 3.0;
        assertClose("forcing = 5*t", evalFree(r.forcingOrNull, tv), 5 * tv);
        assertClose("order 1 coefficient = 1", evalFree(r.coefficients[0], tv), 1.0);
        assertClose("order 0 coefficient = 1", evalFree(r.coefficients[1], tv), 1.0);
        checks++;
    }

    // ------------------------------------------------------------------
    // 5. Sparse: y[3] + y[0] = 0 (no y[2] or y[1] term at all) -> both must fill in as 0.
    // ------------------------------------------------------------------
    private static void testSparseMiddleOrderFillsWithZero() {
        CoefficientExtractor.Result r = CoefficientExtractor.extract("y[3] + y[0]");
        assertEquals("topOrder", 3, r.topOrder);
        assertEquals("term count (dense, always topOrder+1)", 4, r.terms.size());

        assertClose("order 3 = 1", evalFree(r.coefficients[0], 0), 1.0);
        assertClose("order 2 (absent) = 0", evalFree(r.coefficients[1], 0), 0.0);
        assertClose("order 1 (absent) = 0", evalFree(r.coefficients[2], 0), 0.0);
        assertClose("order 0 = 1", evalFree(r.coefficients[3], 0), 1.0);
        if (!r.terms.get(1).absent || !r.terms.get(2).absent) {
            throw new AssertionError("orders 2 and 1 should be flagged absent");
        }
        checks++;
    }

    // ------------------------------------------------------------------
    // 6. Top derivative wrapped in a function must be rejected — the one
    //    hard rule the user stated explicitly.
    // ------------------------------------------------------------------
    private static void testTopDerivativeWrappedIsRejected() {
        assertThrows("sin(y[2]) as the top term must be rejected: top derivative must stay linear",
                () -> CoefficientExtractor.extract("sin(y[2]) + y[1]"));
        checks++;
    }

    // ------------------------------------------------------------------
    // 7. Double-nested function wrap must be rejected — only one wrap layer supported.
    // ------------------------------------------------------------------
    private static void testDoubleNestedWrapIsRejected() {
        assertThrows("sin(cos(y[1])) must be rejected: more than one wrapping function",
                () -> CoefficientExtractor.extract("y[2] + sin(cos(y[1]))"));
        checks++;
    }

    // ------------------------------------------------------------------
    // 8. Two different orders coupled in one term must be rejected.
    // ------------------------------------------------------------------
    private static void testCrossOrderCouplingIsRejected() {
        assertThrows("y[2]*y[1] must be rejected: couples two distinct orders in one term",
                () -> CoefficientExtractor.extract("y[3] + y[2]*y[1]"));
        checks++;
    }

    // ------------------------------------------------------------------
    // 9. The same order appearing in two separate additive terms must be rejected.
    // ------------------------------------------------------------------
    private static void testSameOrderTwiceIsRejected() {
        assertThrows("y[1] appearing in two separate terms must be rejected",
                () -> CoefficientExtractor.extract("y[2] + sin(y[1]) + y[1]"));
        checks++;
    }

    // ------------------------------------------------------------------
    // 10. A state variable used as a divisor must be rejected.
    // ------------------------------------------------------------------
    private static void testDivisorStateIsRejected() {
        assertThrows("1/y[1] must be rejected: state variable as a divisor",
                () -> CoefficientExtractor.extract("y[2] + 1/y[1]"));
        checks++;
    }

    // ------------------------------------------------------------------
    // 11. A state variable as one of several arguments to a multi-arg
    //     function must be rejected (only single-argument wraps are supported).
    // ------------------------------------------------------------------
    private static void testMultiArgFunctionWrapIsRejected() {
        assertThrows("atan2(y[1], t) must be rejected: multi-argument function wrap unsupported",
                () -> CoefficientExtractor.extract("y[2] + atan2(y[1], t)"));
        checks++;
    }

    // ------------------------------------------------------------------
    // 12. No state reference at all.
    // ------------------------------------------------------------------
    private static void testNoStateReferenceAtAllIsRejected() {
        assertThrows("an equation with no y[...] reference at all must be rejected",
                () -> CoefficientExtractor.extract("3 + t - 5"));
        checks++;
    }
 // ------------------------------------------------------------------
    // 13. Same equation as a raw string, wrapped in a full diffeqn(...) call
    //    text — must agree exactly with the scanned-token-list result.
    // ------------------------------------------------------------------
    private static void testHardExampleMixedDerivativeTerms() {
        String raw = "diffeqn((3*x^2)*y[4]+(5*sin(x))*sin(y[3])+(5/x)*ln(y[2])-3*y[1]+3*x*y[0], 1, 0, anon1)";
        CoefficientExtractor.Result r = CoefficientExtractor.extract(raw);
        assertWorkedExample(r);
        checks++;
    }
    // ------------------------------------------------------------------
    // Small numeric evaluator: any non-state variable (whatever it's named —
    // "t", "x", etc.) takes the single supplied free-variable value.
    // ------------------------------------------------------------------

    private static double evalFree(ExprNode node, double freeVarValue) {
        switch (node.kind) {
            case NUMBER:
                return node.numberValue;
            case VARIABLE:
                return freeVarValue; // these tests only ever have one free variable
            case OP:
                List<ExprNode> c = node.children;
                if (node.funcName != null) {
                    double a = evalFree(c.get(0), freeVarValue);
                    switch (node.funcName) {
                        case "sin": return Math.sin(a);
                        case "cos": return Math.cos(a);
                        case "tan": return Math.tan(a);
                        case "asin": return Math.asin(a);
                        case "acos": return Math.acos(a);
                        case "atan": return Math.atan(a);
                        case "sinh": return Math.sinh(a);
                        case "cosh": return Math.cosh(a);
                        case "tanh": return Math.tanh(a);
                        case "asinh": return Maths.asinh(a);
                        case "acosh": return Maths.acosh(a);
                        case "atanh": return Maths.atanh(a);
                        case "ln": return Math.log(a);
                        case "exp": return Math.exp(a);
                        case "aln": return Math.exp(a);
                        case "sqrt": return Math.sqrt(a);
                        case "cbrt": return Math.cbrt(a);
                        default: throw new IllegalStateException("Test evaluator does not support: " + node.funcName);
                    }
                }
                if (c.size() == 1) {
                    return -evalFree(c.get(0), freeVarValue);
                }
                double x = evalFree(c.get(0), freeVarValue);
                double y = evalFree(c.get(1), freeVarValue);
                switch (node.opChar) {
                    case '+': return x + y;
                    case '-': return x - y;
                    case '*': return x * y;
                    case '/': return x / y;
                    case '^': return Math.pow(x, y);
                    case '%': return x % y;
                    default: throw new IllegalStateException("Test evaluator does not support opChar: " + node.opChar);
                }
            default:
                throw new IllegalStateException("Unreachable");
        }
    }

    // ------------------------------------------------------------------
    // Assertions
    // ------------------------------------------------------------------

    private static void assertEquals(String label, int expected, int actual) {
        if (expected != actual) {
            throw new AssertionError(label + ": expected " + expected + " but got " + actual);
        }
    }

    private static void assertEqualsString(String label, String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError(label + ": expected '" + expected + "' but got '" + actual + "'");
        }
    }

    private static void assertNullString(String label, String actual) {
        if (actual != null) {
            throw new AssertionError(label + ": expected null but got '" + actual + "'");
        }
    }

    private static void assertClose(String label, double actual, double expected) {
        double diff = Math.abs(actual - expected);
        double scale = Math.max(1.0, Math.abs(expected));
        if (diff / scale > 1e-9) {
            throw new AssertionError(label + ": expected " + expected + " but got " + actual);
        }
    }

    private static void assertNull(String label, Object value) {
        if (value != null) {
            throw new AssertionError(label + ": expected null but got " + ExprAlgebra.describe((ExprNode) value));
        }
    }

    private static void assertThrows(String label, Supplier<Object> action) {
        try {
            action.get();
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError(label + ": expected IllegalArgumentException but nothing was thrown");
    }
}