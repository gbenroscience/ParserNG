package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.old;

import java.util.List;
import java.util.function.Supplier;

/**
 * Self-verifying regression harness for ExprNode / ExprAlgebra /
 * LinearFormExtractor / TopDerivativeExtractor, covering 16 equations across
 * both extractors: linear equations of varying order and sign structure,
 * equations that are linear only in the top-order term (nonlinear elsewhere),
 * and equations that must be rejected by one or both extractors, each with a
 * specific expected failure reason.
 *
 * Every success case is checked by numerically evaluating the extracted
 * coefficient/forcing/remainder ExprNodes at sample points via a small
 * evaluator built for this test, and comparing against independently
 * computed expected values — not just "did it throw or not". Run via
 * main(); throws AssertionError with a descriptive message on any mismatch.
 *
 * No compiler was available to run this in the environment it was written
 * in — every equation was also traced by hand, but please actually compile
 * and run this before trusting either extractor.
 */
public final class LinearFormExtractorSelfTest {

    private static int checks = 0;

    public static void main(String[] args) {
        // ---- Linear equations: LinearFormExtractor should succeed on all of these ----
        testFirstOrderWithForcing();
        testDifferentStateVariableName();
        testThirdOrderPoleEquationHomogeneous();
        testFourthOrderWithForcing();
        testConstantCoefficientSecondOrder();
        testSparseMissingMiddleCoefficient();
        testMixedSignsConstantMovesToForcing();
        testNegativeForcingOnRHS();
        testCoefficientContainsDivisionStateNotDivisor();

        // ---- Nonlinear-but-top-linear: TopDerivativeExtractor succeeds, LinearFormExtractor must reject ----
        testPendulumTopLinearOnly();
        testNonlinearRemainderProductOfLowerStates();

        // ---- Equations where even TopDerivativeExtractor must reject ----
        testTopTermSquaredRejected();
        testTopTermRepeatedInSameTermRejected();
        testTopTermRepeatedInSeparateTermsRejected();
        testTopTermAsDivisorRejected();

        // ---- Degenerate cases ----
        testNoStateReferenceAtAllRejectedByBoth();
        testOrderZeroOnlyRejectedByTopDerivativeExtractor();

        System.out.println("All " + checks + " LinearFormExtractor/TopDerivativeExtractor checks passed.");
    }

    // ------------------------------------------------------------------
    // 1. First order, variable coefficient, forcing term: t*y[1] + 2*y[0] = sin(t)
    // ------------------------------------------------------------------
    private static void testFirstOrderWithForcing() {
        ExprNode lhs = plus(mul(t(), y(1)), mul(num(2), y(0)));
        ExprNode rhs = sin(t());

        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        assertEquals("order", 1, r.order);
        assertEquals("coefficient count", 2, r.coefficients.size());

        for (double tv : new double[]{0.5, 2.0, 5.0}) {
            assertClose("A(t)=t coefficient at t=" + tv, evalCoeff(r.coefficients.get(0), tv), tv);
            assertClose("B(t)=2 coefficient at t=" + tv, evalCoeff(r.coefficients.get(1), tv), 2.0);
            assertClose("forcing sin(t) at t=" + tv, evalCoeff(r.forcingOrNull, tv), Math.sin(tv));
        }
        checks++;
    }

    // ------------------------------------------------------------------
    // 1b. Same shape, different dependent-variable name: u used instead of y.
    //     u[1] + 2*u[0] = cos(t)  (u, not y)
    // ------------------------------------------------------------------
    private static void testDifferentStateVariableName() {
        ExprNode lhs = plus(state("u", 1), mul(num(2), state("u", 0)));
        ExprNode rhs = cos(t());

        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        assertEquals("order", 1, r.order);

        double tv = 1.0;
        assertClose("A(t)=1 coefficient", evalCoeff(r.coefficients.get(0), tv), 1.0);
        assertClose("B(t)=2 coefficient", evalCoeff(r.coefficients.get(1), tv), 2.0);
        assertClose("forcing cos(t)", evalCoeff(r.forcingOrNull, tv), Math.cos(tv));
        checks++;
    }

    // ------------------------------------------------------------------
    // 2. Third order pole equation (homogeneous): 3*t*sin(t)*y[3] + 4*t*y[2] + 3*ln(t)*y[1] + 4*y[0] = 0
    // ------------------------------------------------------------------
    private static void testThirdOrderPoleEquationHomogeneous() {
        ExprNode termA = mul(mul(num(3), t()), mul(sin(t()), y(3)));
        ExprNode termB = mul(mul(num(4), t()), y(2));
        ExprNode termC = mul(mul(num(3), ln(t())), y(1));
        ExprNode termD = mul(num(4), y(0));
        ExprNode lhs = plus(termA, plus(termB, plus(termC, termD)));
        ExprNode rhs = num(0);

        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        assertEquals("order", 3, r.order);
        assertEquals("coefficient count", 4, r.coefficients.size());
        assertNull("homogeneous -> no forcing term", r.forcingOrNull);

        for (double tv : new double[]{1.0, 2.0, 2.9}) {
            assertClose("A1(t)=3t*sin(t) at t=" + tv, evalCoeff(r.coefficients.get(0), tv), 3 * tv * Math.sin(tv));
            assertClose("A2(t)=4t at t=" + tv, evalCoeff(r.coefficients.get(1), tv), 4 * tv);
            assertClose("A3(t)=3ln(t) at t=" + tv, evalCoeff(r.coefficients.get(2), tv), 3 * Math.log(tv));
            assertClose("A4(t)=4 at t=" + tv, evalCoeff(r.coefficients.get(3), tv), 4.0);
        }
        checks++;
    }

    // ------------------------------------------------------------------
    // 3. Fourth order with forcing: t*y[4] + 2*y[3] + t*y[2] + 3*y[1] + y[0] = cos(t)
    // ------------------------------------------------------------------
    private static void testFourthOrderWithForcing() {
        ExprNode lhs = plus(mul(t(), y(4)),
                plus(mul(num(2), y(3)),
                        plus(mul(t(), y(2)),
                                plus(mul(num(3), y(1)), y(0)))));
        ExprNode rhs = cos(t());

        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        assertEquals("order", 4, r.order);
        assertEquals("coefficient count", 5, r.coefficients.size());

        double tv = 2.5;
        assertClose("A1(t)=t", evalCoeff(r.coefficients.get(0), tv), tv);
        assertClose("A2(t)=2", evalCoeff(r.coefficients.get(1), tv), 2.0);
        assertClose("A3(t)=t", evalCoeff(r.coefficients.get(2), tv), tv);
        assertClose("A4(t)=3", evalCoeff(r.coefficients.get(3), tv), 3.0);
        assertClose("A5(t)=1", evalCoeff(r.coefficients.get(4), tv), 1.0);
        assertClose("forcing cos(t)", evalCoeff(r.forcingOrNull, tv), Math.cos(tv));
        checks++;
    }

    // ------------------------------------------------------------------
    // 4. Constant-coefficient second order: y[2] + 3*y[1] + 2*y[0] = 0
    // ------------------------------------------------------------------
    private static void testConstantCoefficientSecondOrder() {
        ExprNode lhs = plus(y(2), plus(mul(num(3), y(1)), mul(num(2), y(0))));
        ExprNode rhs = num(0);

        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        assertEquals("order", 2, r.order);
        assertClose("A1=1", evalCoeff(r.coefficients.get(0), 0), 1.0);
        assertClose("A2=3", evalCoeff(r.coefficients.get(1), 0), 3.0);
        assertClose("A3=2", evalCoeff(r.coefficients.get(2), 0), 2.0);
        assertNull("homogeneous", r.forcingOrNull);
        checks++;
    }

    // ------------------------------------------------------------------
    // 5. Sparse: missing middle terms entirely -> their coefficients must resolve to 0.
    //    y[3] + y[0] = 0   (no y[2] or y[1] term at all)
    // ------------------------------------------------------------------
    private static void testSparseMissingMiddleCoefficient() {
        ExprNode lhs = plus(y(3), y(0));
        ExprNode rhs = num(0);

        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        assertEquals("order", 3, r.order);
        assertEquals("coefficient count", 4, r.coefficients.size());
        assertClose("A1 (y[3])=1", evalCoeff(r.coefficients.get(0), 0), 1.0);
        assertClose("A2 (y[2], absent)=0", evalCoeff(r.coefficients.get(1), 0), 0.0);
        assertClose("A3 (y[1], absent)=0", evalCoeff(r.coefficients.get(2), 0), 0.0);
        assertClose("A4 (y[0])=1", evalCoeff(r.coefficients.get(3), 0), 1.0);
        checks++;
    }

    // ------------------------------------------------------------------
    // 6. Mixed signs, subtraction, and a bare constant term that must move to forcing:
    //    y[2] - t*y[1] + 5*y[0] - t^2 = 0   =>   coefficient of y[1] is -t, forcing is +t^2
    // ------------------------------------------------------------------
    private static void testMixedSignsConstantMovesToForcing() {
        ExprNode lhs = minus(plus(minus(y(2), mul(t(), y(1))), mul(num(5), y(0))), pow(t(), num(2)));
        ExprNode rhs = num(0);

        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        assertEquals("order", 2, r.order);

        double tv = 3.0;
        assertClose("A1 (y[2])=1", evalCoeff(r.coefficients.get(0), tv), 1.0);
        assertClose("A2 (y[1])=-t", evalCoeff(r.coefficients.get(1), tv), -tv);
        assertClose("A3 (y[0])=5", evalCoeff(r.coefficients.get(2), tv), 5.0);
        assertClose("forcing = t^2 (moved from -t^2 on LHS)", evalCoeff(r.forcingOrNull, tv), tv * tv);
        checks++;
    }

    // ------------------------------------------------------------------
    // 7. Negative forcing on the RHS: y[1] + y[0] = -3*t
    // ------------------------------------------------------------------
    private static void testNegativeForcingOnRHS() {
        ExprNode lhs = plus(y(1), y(0));
        ExprNode rhs = neg(mul(num(3), t()));

        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        double tv = 4.0;
        assertClose("forcing = -3t", evalCoeff(r.forcingOrNull, tv), -3 * tv);
        checks++;
    }

    // ------------------------------------------------------------------
    // 8. Coefficient itself contains division; the state variable is NOT the divisor.
    //    exp(t)*y[2] + (1/t)*y[1] + y[0] = 0
    // ------------------------------------------------------------------
    private static void testCoefficientContainsDivisionStateNotDivisor() {
        ExprNode lhs = plus(mul(exp(t()), y(2)),
                plus(mul(div(num(1), t()), y(1)), y(0)));
        ExprNode rhs = num(0);

        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        double tv = 2.0;
        assertClose("A1 = exp(t)", evalCoeff(r.coefficients.get(0), tv), Math.exp(tv));
        assertClose("A2 = 1/t", evalCoeff(r.coefficients.get(1), tv), 1.0 / tv);
        assertClose("A3 = 1", evalCoeff(r.coefficients.get(2), tv), 1.0);
        checks++;
    }

    // ------------------------------------------------------------------
    // 9. Pendulum: y[2] + sin(y[0]) = 0
    //    TopDerivativeExtractor must succeed (top term y[2] is linear, remainder is nonlinear).
    //    LinearFormExtractor must reject (sin(y[0]) is not a plain multiplicative factor).
    // ------------------------------------------------------------------
    private static void testPendulumTopLinearOnly() {
        ExprNode lhs = plus(y(2), sin(y(0)));
        ExprNode rhs = num(0);

        TopDerivativeExtractor.Result r = TopDerivativeExtractor.extract(lhs, rhs);
        assertEquals("order", 2, r.order);
        assertClose("leading coefficient = 1", evalCoeff(r.leadingCoefficient, 0), 1.0);

        for (double y0 : new double[]{0.3, 1.0, 2.5}) {
            double expectedRemainder = Math.sin(y0);
            assertClose("remainder = sin(y[0]) at y0=" + y0,
                    evalWithState(r.remainder, 0, new double[]{y0}), expectedRemainder);
        }

        assertThrows("LinearFormExtractor must reject sin(y[0]) as nonlinear",
                () -> LinearFormExtractor.extract(lhs, rhs));
        checks++;
    }

    // ------------------------------------------------------------------
    // 10. Nonlinear remainder — product of two different lower-order states:
    //     y[3] + y[2]*y[1] + y[0] = 0
    //     TopDerivativeExtractor succeeds; LinearFormExtractor rejects the y[2]*y[1] term
    //     (two distinct state leaves in one additive term).
    // ------------------------------------------------------------------
    private static void testNonlinearRemainderProductOfLowerStates() {
        ExprNode lhs = plus(y(3), plus(mul(y(2), y(1)), y(0)));
        ExprNode rhs = num(0);

        TopDerivativeExtractor.Result r = TopDerivativeExtractor.extract(lhs, rhs);
        assertEquals("order", 3, r.order);
        assertClose("leading coefficient = 1", evalCoeff(r.leadingCoefficient, 0), 1.0);

        double[] yv = {0.5, 2.0, 0.0}; // y[0]=0.5, y[1]=2.0, y[2]=0.0
        assertClose("remainder = y[2]*y[1] + y[0]",
                evalWithState(r.remainder, 0, yv), yv[2] * yv[1] + yv[0]);

        assertThrows("LinearFormExtractor must reject y[2]*y[1] as coupling two states in one term",
                () -> LinearFormExtractor.extract(lhs, rhs));
        checks++;
    }

    // ------------------------------------------------------------------
    // 11. Top term itself squared: y[2]^2 + y[1] = 0 -> even TopDerivativeExtractor must reject.
    // ------------------------------------------------------------------
    private static void testTopTermSquaredRejected() {
        ExprNode lhs = plus(pow(y(2), num(2)), y(1));
        ExprNode rhs = num(0);

        assertThrows("y[2]^2 must be rejected: top derivative not a plain factor",
                () -> TopDerivativeExtractor.extract(lhs, rhs));
        checks++;
    }

    // ------------------------------------------------------------------
    // 12. Top term repeated within the SAME additive term: y[2]*y[2] + y[1] = 0
    // ------------------------------------------------------------------
    private static void testTopTermRepeatedInSameTermRejected() {
        ExprNode lhs = plus(mul(y(2), y(2)), y(1));
        ExprNode rhs = num(0);

        assertThrows("y[2]*y[2] must be rejected: top derivative appears twice",
                () -> TopDerivativeExtractor.extract(lhs, rhs));
        checks++;
    }

    // ------------------------------------------------------------------
    // 13. Top term repeated across two SEPARATE additive terms: y[2] + y[2] + y[1] = 0
    // ------------------------------------------------------------------
    private static void testTopTermRepeatedInSeparateTermsRejected() {
        ExprNode lhs = plus(plus(y(2), y(2)), y(1));
        ExprNode rhs = num(0);

        assertThrows("y[2] appearing in two separate terms must be rejected",
                () -> TopDerivativeExtractor.extract(lhs, rhs));
        checks++;
    }

    // ------------------------------------------------------------------
    // 14. Top term used as a divisor: 1/y[2] + y[1] = 0
    // ------------------------------------------------------------------
    private static void testTopTermAsDivisorRejected() {
        ExprNode lhs = plus(div(num(1), y(2)), y(1));
        ExprNode rhs = num(0);

        assertThrows("y[2] as a divisor must be rejected",
                () -> TopDerivativeExtractor.extract(lhs, rhs));
        checks++;
    }

    // ------------------------------------------------------------------
    // 15. No state reference at all: 3 = 5 -> both extractors must reject.
    // ------------------------------------------------------------------
    private static void testNoStateReferenceAtAllRejectedByBoth() {
        ExprNode lhs = num(3);
        ExprNode rhs = num(5);

        assertThrows("LinearFormExtractor must reject an equation with no state reference",
                () -> LinearFormExtractor.extract(lhs, rhs));
        assertThrows("TopDerivativeExtractor must reject an equation with no state reference",
                () -> TopDerivativeExtractor.extract(lhs, rhs));
        checks++;
    }

    // ------------------------------------------------------------------
    // 16. Order-0 only (y[0] present, no derivative at all): 3*y[0] = 5
    //     TopDerivativeExtractor rejects with a specific "no derivative" message.
    //     LinearFormExtractor currently does NOT have this guard and will return
    //     order=0 — flagged here as documented current behavior, not a silent
    //     success: downstream CompanionSystemHandles.buildCompanion(order=0)
    //     already throws "order must be positive", so this fails one layer later
    //     with a less specific message. Worth adding the same guard to
    //     LinearFormExtractor for a clearer error at the right layer.
    // ------------------------------------------------------------------
    private static void testOrderZeroOnlyRejectedByTopDerivativeExtractor() {
        ExprNode lhs = mul(num(3), y(0));
        ExprNode rhs = num(5);

        assertThrows("TopDerivativeExtractor must reject an order-0 (no derivative) equation",
                () -> TopDerivativeExtractor.extract(lhs, rhs));

        // Documented current gap: LinearFormExtractor does not yet guard this case.
        LinearFormExtractor.Result r = LinearFormExtractor.extract(lhs, rhs);
        assertEquals("KNOWN GAP: LinearFormExtractor currently accepts order=0 "
                + "(no equivalent guard to TopDerivativeExtractor's) — add one", 0, r.order);
        checks++;
    }

    // ------------------------------------------------------------------
    // Small ExprNode builder DSL, for readable test equations
    // ------------------------------------------------------------------

    private static ExprNode t() {
        return ExprNode.variable("t");
    }

    private static ExprNode num(double v) {
        return ExprNode.number(v);
    }

    private static ExprNode y(int index) {
        return ExprNode.stateVariable("y", index);
    }

    private static ExprNode state(String name, int index) {
        return ExprNode.stateVariable(name, index);
    }

    private static ExprNode plus(ExprNode a, ExprNode b) {
        return ExprNode.op('+', List.of(a, b));
    }

    private static ExprNode minus(ExprNode a, ExprNode b) {
        return ExprNode.op('-', List.of(a, b));
    }

    private static ExprNode neg(ExprNode a) {
        return ExprNode.op('-', List.of(a));
    }

    private static ExprNode mul(ExprNode a, ExprNode b) {
        return ExprNode.op('*', List.of(a, b));
    }

    private static ExprNode div(ExprNode a, ExprNode b) {
        return ExprNode.op('/', List.of(a, b));
    }

    private static ExprNode pow(ExprNode a, ExprNode b) {
        return ExprNode.op('^', List.of(a, b));
    }

    private static ExprNode sin(ExprNode a) {
        return ExprNode.func("sin", List.of(a));
    }

    private static ExprNode cos(ExprNode a) {
        return ExprNode.func("cos", List.of(a));
    }

    private static ExprNode ln(ExprNode a) {
        return ExprNode.func("ln", List.of(a));
    }

    private static ExprNode exp(ExprNode a) {
        return ExprNode.func("exp", List.of(a));
    }

    // ------------------------------------------------------------------
    // Small numeric evaluator, t (and optionally state values) in, double out.
    // Only supports the operators/functions used by this test's equations.
    // ------------------------------------------------------------------

    private static double evalCoeff(ExprNode node, double tValue) {
        return evalWithState(node, tValue, new double[0]);
    }

    private static double evalWithState(ExprNode node, double tValue, double[] yValues) {
        switch (node.kind) {
            case NUMBER:
                return node.numberValue;
            case VARIABLE:
                if (node.isStateVariable()) {
                    return yValues[node.stateIndex];
                }
                return tValue; // the only non-state variable these tests use is "t"
            case OP:
                List<ExprNode> c = node.children;
                if (node.funcName != null) {
                    double a = evalWithState(c.get(0), tValue, yValues);
                    switch (node.funcName) {
                        case "sin": return Math.sin(a);
                        case "cos": return Math.cos(a);
                        case "ln": return Math.log(a);
                        case "exp": return Math.exp(a);
                        default: throw new IllegalStateException("Test evaluator does not support: " + node.funcName);
                    }
                }
                if (c.size() == 1) { // unary minus
                    return -evalWithState(c.get(0), tValue, yValues);
                }
                double x = evalWithState(c.get(0), tValue, yValues);
                double yv = evalWithState(c.get(1), tValue, yValues);
                switch (node.opChar) {
                    case '+': return x + yv;
                    case '-': return x - yv;
                    case '*': return x * yv;
                    case '/': return x / yv;
                    case '^': return Math.pow(x, yv);
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
            return; // expected path
        }
        throw new AssertionError(label + ": expected IllegalArgumentException but nothing was thrown");
    }
}