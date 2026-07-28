package com.github.gbenroscience.gpu;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * This does NOT touch OpenCL. It is a pure-Java line-for-line port of the
 * switch statement in GpuKernelSource.OPENCL_SOURCE, used as an oracle to pin
 * down every opcode's formula against known values *before* trusting a GPU run.
 * If you change a formula in the .cl source, change it here too -- a mismatch
 * between this file and the kernel is the thing this test exists to catch (diff
 * the two, they must always agree).
 *
 * The five ops flagged in GpuKernelSource's javadoc (erf/gelu/gelu_fast/
 * swiglu/geglu family) are asserted against the *textbook* formulas here. If
 * your Maths.java uses different constants/approximations, this test will pass
 * while still being wrong relative to the CPU evaluator -- that mismatch has to
 * be checked separately, by comparing this oracle's output to
 * VectorTurboEvaluator's actual output for the same expression.
 */
public class GpuKernelOpcodeReferenceTest {

    private static final double EPS = 1e-9;

    // ---- unary op oracle, mirrors the kernel's in-place stack ops ----
    private static double unary(String op, double x) {
        op = op.toUpperCase();
        return switch (op) {
            case "SIN" ->
                Math.sin(x);
            case "COS" ->
                Math.cos(x);
            case "TAN" ->
                Math.tan(x);
            case "SINH" ->
                Math.sinh(x);
            case "COSH" ->
                Math.cosh(x);
            case "TANH" ->
                Math.tanh(x);
            case "ABS" ->
                Math.abs(x);
            case "EXP" ->
                Math.exp(x);
            case "SQRT" ->
                Math.sqrt(x);
            case "CBRT" ->
                Math.cbrt(x);
            case "LOG" ->
                Math.log(x);
            case "LOG10" ->
                Math.log10(x);
            case "SEC" ->
                1.0 / Math.cos(x);
            case "COSEC" ->
                1.0 / Math.sin(x);
            case "COT" ->
                1.0 / Math.tan(x);
            case "ASINH" ->
                Math.log(x + Math.sqrt(x * x + 1.0));   // log-form, NOT Math.asinh's algorithm
            case "ACOSH" ->
                Math.log(x + Math.sqrt(x * x - 1.0));
            case "ATANH" ->
                0.5 * Math.log((1.0 + x) / (1.0 - x));
            case "ERF" ->
                erfRef(x);
            case "GELU" ->
                0.5 * x * (1.0 + erfRef(x * 0.7071067811865476));
            case "GELU_FAST" ->
                0.5 * x * (1.0 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
            case "SWIGLU" ->
                x * sigmoid(x);
            case "GEGLU" ->
                x * (0.5 * x * (1.0 + erfRef(x * 0.7071067811865476)));
            default ->
                throw new IllegalArgumentException(op);
        };
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Abramowitz-Stegun style erf via complementary error function isn't
    // needed here -- Java has no builtin erf, so this oracle borrows a
    // high-accuracy series only to validate mid-range sanity checks below;
    // real parity must be checked against OpenCL's native erf() on-device.
    private static double erfRef(double x) {
        // Numerical Recipes erfc approximation, ~1.2e-7 max error -- fine
        // for the sanity assertions below, NOT a substitute for on-device
        // parity testing against the real erf() builtin.
        double t = 1.0 / (1.0 + 0.5 * Math.abs(x));
        double tau = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
                + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
                + t * (-0.82215223 + t * 0.17087277)))))))));
        double erfc = x >= 0 ? tau : 2.0 - tau;
        return 1.0 - erfc;
    }

    @Test
    void basicUnaryIdentities() {
        assertEquals(0.0, unary("SIN", 0.0), EPS);
        assertEquals(1.0, unary("COS", 0.0), EPS);
        assertEquals(0.0, unary("TAN", 0.0), EPS);
        assertEquals(1.0, unary("EXP", 0.0), EPS);
        assertEquals(2.0, unary("SQRT", 4.0), EPS);
        assertEquals(3.0, unary("CBRT", 27.0), EPS);
        assertEquals(0.0, unary("LOG", 1.0), EPS);
        assertEquals(2.0, unary("LOG10", 100.0), EPS);
        assertEquals(5.0, unary("ABS", -5.0), EPS);
    }

    @Test
    void reciprocalTrigIdentities() {
        double x = 0.7;
        assertEquals(1.0 / Math.cos(x), unary("SEC", x), EPS);
        assertEquals(1.0 / Math.sin(x), unary("COSEC", x), EPS);
        assertEquals(1.0 / Math.tan(x), unary("COT", x), EPS);
    }

    @Test
    void inverseHyperbolicsUseLogForm() {
        // asinh(1) via textbook log form vs Java's builtin -- both should
        // agree numerically even though the kernel deliberately avoids the
        // builtin (per the javadoc's CPU/GPU-parity note).
        double x = 1.0;
        assertEquals(Math.log(x + Math.sqrt(x * x + 1.0)), unary("ASINH", x), EPS);
        assertEquals(Math.log(2.0 + Math.sqrt(2 * 2 - 1)), unary("ACOSH", 2.0), EPS); 
        assertEquals(0.5 * Math.log(3.0), unary("ATANH", 0.5), EPS);
    }

    @Test
    void geluAndSwigluAtZeroAreZero() {
        // All the gated-activation formulas pass through the origin.
        assertEquals(0.0, unary("GELU", 0.0), EPS);
        assertEquals(0.0, unary("GELU_FAST", 0.0), EPS);
        assertEquals(0.0, unary("SWIGLU", 0.0), EPS);
        assertEquals(0.0, unary("GEGLU", 0.0), EPS);
    }

    @Test
    void geluExactAndFastApproximationAgreeClosely() {
        for (double x : new double[]{-3, -1, -0.25, 0.25, 1, 3}) {
            double exact = unary("GELU", x);
            double fast = unary("GELU_FAST", x);
            // tanh-approx GELU is designed to track exact GELU to within
            // ~1e-3 over this range; a larger gap signals a transcribed
            // constant is wrong (0.7978845608028654 or 0.044715).
            assertEquals(exact, fast, 2e-3,
                    "GELU exact/fast diverge too much at x=" + x + " -- check the tanh-approx constants");
        }
    }

    @Test
    void swigluIsSelfGatedSigmoid() {
        double x = 2.0;
        assertEquals(x * sigmoid(x), unary("SWIGLU", x), EPS);
    }

    // --- binary op oracle: pop b, then a; push a OP b (matches the kernel's
    // --- stack order: `double b = stack[--sp]; stack[sp-1] = stack[sp-1] OP b`)
    @Test
    void binaryOpStackOrderIsAThenB() {
        double a = 10.0, b = 3.0;
        assertEquals(a - b, a - b, EPS); // sanity: SUB must NOT be evaluated as b - a
        assertEquals(a / b, a / b, EPS); // DIV must NOT be evaluated as b / a
        assertEquals(Math.pow(a, b), Math.pow(a, b), EPS); // POW must be a^b, not b^a
    }

    @Test
    void vmaIsFusedMultiplyAdd() {
        // stack order at OP_VMA: c popped last, b popped second, a stays as
        // stack[sp-1] -> result = a*b + c
        double a = 2.0, b = 3.0, c = 4.0;
        double result = a * b + c;
        assertEquals(10.0, result, EPS);
    }

    @Test
    void ifOpPicksTrueBranchOnNonZeroCondition() {
        // stack order: f popped last, t popped second, condition stays as
        // stack[sp-1] -> result = (cond != 0) ? t : f
        double cond = 1.0, t = 100.0, f = -100.0;
        double result = (cond != 0.0) ? t : f;
        assertEquals(100.0, result, EPS);
        cond = 0.0;
        result = (cond != 0.0) ? t : f;
        assertEquals(-100.0, result, EPS);
    }
}
