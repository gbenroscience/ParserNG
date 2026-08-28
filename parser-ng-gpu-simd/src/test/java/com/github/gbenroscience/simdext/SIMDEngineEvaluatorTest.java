package com.github.gbenroscience.simdext;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression.BLOCK_SIZE;
import com.github.gbenroscience.simdext.turbo.tools.junk.SIMDEngineEvaluator;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link SIMDEngineEvaluator}'s bulk-evaluation storage paths:
 * the original {@code double[]} / {@code double[][]} / {@code MemorySegment}
 * (packed doubles) / {@code MemorySegment[]} (packed doubles) paths, and the
 * float support added alongside them — {@code float[]}, {@code float[][]},
 * {@code MemorySegment} (packed floats), and {@code MemorySegment[]} (packed
 * floats).
 *
 * <p>Variables are assumed to be resolved to stack slots in the order they
 * first appear in the expression source (the conventional behavior for this
 * kind of expression compiler); adjust variable naming/ordering in these
 * tests if your {@code MathExpression} front-end resolves slots differently.
 *
 * <p>Tolerances: {@code SIN}/{@code COS}/{@code EXP}/{@code SQRT}/etc. go
 * through the JDK Vector API's lanewise transcendental intrinsics and are
 * compared with a tight epsilon (looser for {@code float}, tighter for
 * {@code double}, reflecting each type's native precision).
 * {@code erf}/{@code gelu} go through the {@code VectorizedCodyMath}-backed
 * piecewise rational approximation and are compared with a looser epsilon.
 * <p>All {@code MemorySegment} tests use {@link Arena#ofShared()} rather than
 * {@code Arena.ofConfined()}. The {@code *_Parallel} tests dispatch work to
 * the evaluator's background worker-thread pool, and a confined arena's
 * segments can only be accessed from the single thread that created them —
 * any access from a worker thread throws {@code WrongThreadException}. Using
 * a shared arena everywhere (not just in the parallel tests) keeps the
 * pattern uniform and avoids that trap by construction.
 */
class SIMDEngineEvaluatorTest {

    private static final float TIGHT_EPS = 1e-4f;
    private static final float LOOSE_EPS = 5e-4f;
    private static final double D_TIGHT_EPS = 1e-9;
    private static final double D_LOOSE_EPS = 1e-6;

    private static SIMDEngineEvaluator.SIMDVectorCompositeExpression compile(String expr) throws Throwable {
        return SIMDEngineEvaluator.getEvaluator(expr);
    }

    private static float[] fillLinear(int n, float start, float step) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = start + i * step;
        }
        return a;
    }

    private static double[] fillLinearD(int n, double start, double step) {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) {
            a[i] = start + i * step;
        }
        return a;
    }

    // =====================================================================
    // 1. float[] (flat single-array, multi-variable) path
    // =====================================================================

    @Test
    void testFloatArray_Add() throws Throwable {
        try (var eval = compile("x + y")) {
            int n = 64;
            float[] flat = new float[2 * n];
            float[] x = fillLinear(n, 1.0f, 0.5f);
            float[] y = fillLinear(n, 2.0f, 0.25f);
            System.arraycopy(x, 0, flat, 0, n);
            System.arraycopy(y, 0, flat, n, n);

            float[] out = new float[n];
            eval.applyBulk(flat, out);

            for (int i = 0; i < n; i++) {
                assertEquals(x[i] + y[i], out[i], TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testFloatArray_MulDiv() throws Throwable {
        try (var eval = compile("(x * y) / (x + 1)")) {
            int n = 100;
            float[] x = fillLinear(n, 1.0f, 0.3f);
            float[] y = fillLinear(n, 5.0f, 0.1f);
            float[] flat = new float[2 * n];
            System.arraycopy(x, 0, flat, 0, n);
            System.arraycopy(y, 0, flat, n, n);

            float[] out = new float[n];
            eval.applyBulk(flat, out);

            for (int i = 0; i < n; i++) {
                float expected = (x[i] * y[i]) / (x[i] + 1.0f);
                assertEquals(expected, out[i], TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testFloatArray_Sin() throws Throwable {
        try (var eval = compile("sin(x)")) {
            int n = 50;
            float[] x = fillLinear(n, -3.0f, 0.13f);

            float[] out = new float[n];
            eval.applyBulk(x, out);

            for (int i = 0; i < n; i++) {
                assertEquals((float) Math.sin(x[i]), out[i], TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testFloatArray_Pow() throws Throwable {
        try (var eval = compile("x^3")) {
            int n = 40;
            float[] x = fillLinear(n, -2.0f, 0.2f);

            float[] out = new float[n];
            eval.applyBulk(x, out);

            for (int i = 0; i < n; i++) {
                assertEquals((float) Math.pow(x[i], 3), out[i], TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testFloatArray_Gelu() throws Throwable {
        try (var eval = compile("gelu(x)")) {
            int n = 30;
            float[] x = fillLinear(n, -4.0f, 0.3f);

            float[] out = new float[n];
            eval.applyBulk(x, out);

            for (int i = 0; i < n; i++) {
                double xi = x[i];
                double refGelu = xi * 0.5 * (1.0 + erfRef(xi / Math.sqrt(2.0)));
                assertEquals((float) refGelu, out[i], LOOSE_EPS, "index " + i);
            }
        }
    }

    @Test
    void testFloatArray_Comparison() throws Throwable {
        try (var eval = compile("x > y")) {
            int n = 20;
            float[] x = fillLinear(n, -2.0f, 0.4f);
            float[] y = fillLinear(n, 0.0f, 0.0f);
            float[] flat = new float[2 * n];
            System.arraycopy(x, 0, flat, 0, n);
            System.arraycopy(y, 0, flat, n, n);

            float[] out = new float[n];
            eval.applyBulk(flat, out);

            for (int i = 0; i < n; i++) {
                float expected = (x[i] > y[i]) ? 1.0f : 0.0f;
                assertEquals(expected, out[i], "index " + i);
            }
        }
    }

    // =====================================================================
    // 2. float[][] (one array per variable) path
    // =====================================================================

    @Test
    void testFloat2D_Add() throws Throwable {
        try (var eval = compile("x + y")) {
            int n = 70;
            float[][] vars = {fillLinear(n, 1.0f, 0.1f), fillLinear(n, -1.0f, 0.2f)};

            float[] out = new float[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                assertEquals(vars[0][i] + vars[1][i], out[i], TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testFloat2D_Trig() throws Throwable {
        try (var eval = compile("cos(x) + sin(x)")) {
            int n = 45;
            float[][] vars = {fillLinear(n, -5.0f, 0.25f)};

            float[] out = new float[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                float x = vars[0][i];
                float expected = (float) (Math.cos(x) + Math.sin(x));
                assertEquals(expected, out[i], TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testFloat2D_MultiBlock() throws Throwable {
        // Deliberately larger than BLOCK_SIZE so the internal block-looping
        // logic (multiple BLOCK_SIZE-sized chunks) is exercised.
        try (var eval = compile("x * 2 + 1")) {
            int n = BLOCK_SIZE * 3 + 17;
            float[][] vars = {fillLinear(n, 0.0f, 0.001f)};

            float[] out = new float[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                float expected = vars[0][i] * 2.0f + 1.0f;
                assertEquals(expected, out[i], TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testFloat2D_Sqrt() throws Throwable {
        try (var eval = compile("sqrt(x)")) {
            int n = 33;
            float[][] vars = {fillLinear(n, 0.0f, 1.0f)};

            float[] out = new float[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                assertEquals((float) Math.sqrt(vars[0][i]), out[i], TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testFloat2D_IfElse() throws Throwable {
        try (var eval = compile("if(x > 0, x, -x)")) {
            int n = 60;
            float[][] vars = {fillLinear(n, -30.0f, 1.0f)};

            float[] out = new float[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                float expected = Math.abs(vars[0][i]);
                assertEquals(expected, out[i], TIGHT_EPS, "index " + i);
            }
        }
    }

    // =====================================================================
    // 3. MemorySegment (single concatenated segment, packed floats) path
    // =====================================================================

    @Test
    void testMemSegFloat_Add() throws Throwable {
        try (var eval = compile("x + y"); Arena arena = Arena.ofShared()) {
            int n = 80;
            float[] x = fillLinear(n, 3.0f, 0.05f);
            float[] y = fillLinear(n, -1.0f, 0.02f);

            MemorySegment in = arena.allocate((long) 2 * n * Float.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_FLOAT, 0L, n);
            MemorySegment.copy(y, 0, in, ValueLayout.JAVA_FLOAT, (long) n * Float.BYTES, n);

            MemorySegment out = arena.allocate((long) n * Float.BYTES);
            eval.applyBulkFloat(in, out);

            for (int i = 0; i < n; i++) {
                float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(x[i] + y[i], actual, TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegFloat_Exp() throws Throwable {
        try (var eval = compile("exp(x)"); Arena arena = Arena.ofShared()) {
            int n = 25;
            float[] x = fillLinear(n, -3.0f, 0.25f);

            MemorySegment in = arena.allocate((long) n * Float.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_FLOAT, 0L, n);

            MemorySegment out = arena.allocate((long) n * Float.BYTES);
            eval.applyBulkFloat(in, out);

            for (int i = 0; i < n; i++) {
                float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals((float) Math.exp(x[i]), actual, TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegFloat_Parallel() throws Throwable {
        try (var eval = compile("x * y"); Arena arena = Arena.ofShared()) {
            // Large enough to clear PARALLEL_OPS_THRESHOLD and actually
            // exercise the worker-pool dispatch path (falls back to
            // single-threaded automatically if NUM_WORKERS <= 0).
            int n = 200_000;
            float[] x = fillLinear(n, 0.5f, 0.0001f);
            float[] y = fillLinear(n, -0.5f, 0.0002f);

            MemorySegment in = arena.allocate((long) 2 * n * Float.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_FLOAT, 0L, n);
            MemorySegment.copy(y, 0, in, ValueLayout.JAVA_FLOAT, (long) n * Float.BYTES, n);

            MemorySegment out = arena.allocate((long) n * Float.BYTES);
            eval.applyBulkParallelFloat(in, out);

            // Spot-check a sample rather than every element to keep the test fast.
            for (int i = 0; i < n; i += 997) {
                float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(x[i] * y[i], actual, TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegFloat_Erf() throws Throwable {
        try (var eval = compile("erf(x)"); Arena arena = Arena.ofShared()) {
            int n = 30;
            float[] x = fillLinear(n, -3.0f, 0.2f);

            MemorySegment in = arena.allocate((long) n * Float.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_FLOAT, 0L, n);

            MemorySegment out = arena.allocate((long) n * Float.BYTES);
            eval.applyBulkFloat(in, out);

            for (int i = 0; i < n; i++) {
                float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals((float) erfRef(x[i]), actual, LOOSE_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegFloat_MultiBlock() throws Throwable {
        try (var eval = compile("x - 1"); Arena arena = Arena.ofShared()) {
            int n = BLOCK_SIZE * 2 + 5;
            float[] x = fillLinear(n, 0.0f, 0.01f);

            MemorySegment in = arena.allocate((long) n * Float.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_FLOAT, 0L, n);

            MemorySegment out = arena.allocate((long) n * Float.BYTES);
            eval.applyBulkFloat(in, out);

            for (int i = 0; i < n; i++) {
                float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(x[i] - 1.0f, actual, TIGHT_EPS, "index " + i);
            }
        }
    }

    // =====================================================================
    // 4. MemorySegment[] (one segment per variable, zero-copy) path
    // =====================================================================

    @Test
    void testMemSegArrayFloat_Add() throws Throwable {
        try (var eval = compile("x + y"); Arena arena = Arena.ofShared()) {
            int n = 90;
            float[] x = fillLinear(n, 1.0f, 0.1f);
            float[] y = fillLinear(n, 2.0f, 0.2f);

            MemorySegment segX = arena.allocate((long) n * Float.BYTES);
            MemorySegment segY = arena.allocate((long) n * Float.BYTES);
            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_FLOAT, 0L, n);
            MemorySegment.copy(y, 0, segY, ValueLayout.JAVA_FLOAT, 0L, n);

            MemorySegment out = arena.allocate((long) n * Float.BYTES);
            eval.applyBulkFloat(new MemorySegment[]{segX, segY}, out);

            for (int i = 0; i < n; i++) {
                float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(x[i] + y[i], actual, TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegArrayFloat_Mul() throws Throwable {
        try (var eval = compile("x * y - x"); Arena arena = Arena.ofShared()) {
            int n = 55;
            float[] x = fillLinear(n, 2.0f, 0.05f);
            float[] y = fillLinear(n, 3.0f, 0.03f);

            MemorySegment segX = arena.allocate((long) n * Float.BYTES);
            MemorySegment segY = arena.allocate((long) n * Float.BYTES);
            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_FLOAT, 0L, n);
            MemorySegment.copy(y, 0, segY, ValueLayout.JAVA_FLOAT, 0L, n);

            MemorySegment out = arena.allocate((long) n * Float.BYTES);
            eval.applyBulkFloat(new MemorySegment[]{segX, segY}, out);

            for (int i = 0; i < n; i++) {
                float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float expected = x[i] * y[i] - x[i];
                assertEquals(expected, actual, TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegArrayFloat_Parallel() throws Throwable {
        try (var eval = compile("x + y * 2"); Arena arena = Arena.ofShared()) {
            int n = 200_000;
            float[] x = fillLinear(n, 0.1f, 0.00005f);
            float[] y = fillLinear(n, -0.1f, 0.00003f);

            MemorySegment segX = arena.allocate((long) n * Float.BYTES);
            MemorySegment segY = arena.allocate((long) n * Float.BYTES);
            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_FLOAT, 0L, n);
            MemorySegment.copy(y, 0, segY, ValueLayout.JAVA_FLOAT, 0L, n);

            MemorySegment out = arena.allocate((long) n * Float.BYTES);
            eval.applyBulkParallelFloat(new MemorySegment[]{segX, segY}, out);

            for (int i = 0; i < n; i += 997) {
                float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float expected = x[i] + y[i] * 2.0f;
                assertEquals(expected, actual, TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegArrayFloat_BareVariable() throws Throwable {
        // Edge case: the whole expression is just "x", so OP_LOAD's
        // segment-backed push is never consumed by any op that would
        // otherwise force materialization — exercises the explicit
        // materializeFloat() fallback in applyBulkInternalFloat(MemorySegment[]...).
        try (var eval = compile("x"); Arena arena = Arena.ofShared()) {
            int n = 48;
            float[] x = fillLinear(n, -10.0f, 0.4f);

            MemorySegment segX = arena.allocate((long) n * Float.BYTES);
            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_FLOAT, 0L, n);

            MemorySegment out = arena.allocate((long) n * Float.BYTES);
            eval.applyBulkFloat(new MemorySegment[]{segX}, out);

            for (int i = 0; i < n; i++) {
                float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(x[i], actual, TIGHT_EPS, "index " + i);
            }
        }
    }

    // =====================================================================
    // 5. double[] (flat, multi-variable) path
    // =====================================================================

    @Test
    void testDoubleArray_Add() throws Throwable {
        try (var eval = compile("x + y")) {
            int n = 64;
            double[] x = fillLinearD(n, 1.0, 0.5);
            double[] y = fillLinearD(n, 2.0, 0.25);
            double[] flat = new double[2 * n];
            System.arraycopy(x, 0, flat, 0, n);
            System.arraycopy(y, 0, flat, n, n);

            double[] out = new double[n];
            eval.applyBulk(flat, out);

            for (int i = 0; i < n; i++) {
                assertEquals(x[i] + y[i], out[i], D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testDoubleArray_MulDiv() throws Throwable {
        try (var eval = compile("(x * y) / (x + 1)")) {
            int n = 100;
            double[] x = fillLinearD(n, 1.0, 0.3);
            double[] y = fillLinearD(n, 5.0, 0.1);
            double[] flat = new double[2 * n];
            System.arraycopy(x, 0, flat, 0, n);
            System.arraycopy(y, 0, flat, n, n);

            double[] out = new double[n];
            eval.applyBulk(flat, out);

            for (int i = 0; i < n; i++) {
                double expected = (x[i] * y[i]) / (x[i] + 1.0);
                assertEquals(expected, out[i], D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testDoubleArray_Sin() throws Throwable {
        try (var eval = compile("sin(x)")) {
            int n = 50;
            double[] x = fillLinearD(n, -3.0, 0.13);

            double[] out = new double[n];
            eval.applyBulk(x, out);

            for (int i = 0; i < n; i++) {
                assertEquals(Math.sin(x[i]), out[i], D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testDoubleArray_Pow() throws Throwable {
        try (var eval = compile("x^3")) {
            int n = 40;
            double[] x = fillLinearD(n, -2.0, 0.2);

            double[] out = new double[n];
            eval.applyBulk(x, out);

            for (int i = 0; i < n; i++) {
                assertEquals(Math.pow(x[i], 3), out[i], D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testDoubleArray_Gelu() throws Throwable {
        try (var eval = compile("gelu(x)")) {
            int n = 30;
            double[] x = fillLinearD(n, -4.0, 0.3);

            double[] out = new double[n];
            eval.applyBulk(x, out);

            for (int i = 0; i < n; i++) {
                double xi = x[i];
                double refGelu = xi * 0.5 * (1.0 + erfRef(xi / Math.sqrt(2.0)));
                assertEquals(refGelu, out[i], D_LOOSE_EPS, "index " + i);
            }
        }
    }

    @Test
    void testDoubleArray_Comparison() throws Throwable {
        try (var eval = compile("x > y")) {
            int n = 20;
            double[] x = fillLinearD(n, -2.0, 0.4);
            double[] y = fillLinearD(n, 0.0, 0.0);
            double[] flat = new double[2 * n];
            System.arraycopy(x, 0, flat, 0, n);
            System.arraycopy(y, 0, flat, n, n);

            double[] out = new double[n];
            eval.applyBulk(flat, out);

            for (int i = 0; i < n; i++) {
                double expected = (x[i] > y[i]) ? 1.0 : 0.0;
                assertEquals(expected, out[i], "index " + i);
            }
        }
    }

    // =====================================================================
    // 6. double[][] (one array per variable) path
    // =====================================================================

    @Test
    void testDouble2D_Add() throws Throwable {
        try (var eval = compile("x + y")) {
            int n = 70;
            double[][] vars = {fillLinearD(n, 1.0, 0.1), fillLinearD(n, -1.0, 0.2)};

            double[] out = new double[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                assertEquals(vars[0][i] + vars[1][i], out[i], D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testDouble2D_Trig() throws Throwable {
        try (var eval = compile("cos(x) + sin(x)")) {
            int n = 45;
            double[][] vars = {fillLinearD(n, -5.0, 0.25)};

            double[] out = new double[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                double x = vars[0][i];
                double expected = Math.cos(x) + Math.sin(x);
                assertEquals(expected, out[i], D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testDouble2D_MultiBlock() throws Throwable {
        // Deliberately larger than BLOCK_SIZE so the internal block-looping
        // logic (multiple BLOCK_SIZE-sized chunks) is exercised.
        try (var eval = compile("x * 2 + 1")) {
            int n = BLOCK_SIZE * 3 + 17;
            double[][] vars = {fillLinearD(n, 0.0, 0.001)};

            double[] out = new double[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                double expected = vars[0][i] * 2.0 + 1.0;
                assertEquals(expected, out[i], D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testDouble2D_Sqrt() throws Throwable {
        try (var eval = compile("sqrt(x)")) {
            int n = 33;
            double[][] vars = {fillLinearD(n, 0.0, 1.0)};

            double[] out = new double[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                assertEquals(Math.sqrt(vars[0][i]), out[i], D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testDouble2D_IfElse() throws Throwable {
        try (var eval = compile("if(x > 0, x, -x)")) {
            int n = 60;
            double[][] vars = {fillLinearD(n, -30.0, 1.0)};

            double[] out = new double[n];
            eval.applyBulk(vars, out);

            for (int i = 0; i < n; i++) {
                double expected = Math.abs(vars[0][i]);
                assertEquals(expected, out[i], D_TIGHT_EPS, "index " + i);
            }
        }
    }

    // =====================================================================
    // 7. MemorySegment (single concatenated segment, packed doubles) path
    // =====================================================================

    @Test
    void testMemSegDouble_Add() throws Throwable {
        try (var eval = compile("x + y"); Arena arena = Arena.ofShared()) {
            int n = 80;
            double[] x = fillLinearD(n, 3.0, 0.05);
            double[] y = fillLinearD(n, -1.0, 0.02);

            MemorySegment in = arena.allocate((long) 2 * n * Double.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_DOUBLE, 0L, n);
            MemorySegment.copy(y, 0, in, ValueLayout.JAVA_DOUBLE, (long) n * Double.BYTES, n);

            MemorySegment out = arena.allocate((long) n * Double.BYTES);
            eval.applyBulk(in, out);

            for (int i = 0; i < n; i++) {
                double actual = out.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                assertEquals(x[i] + y[i], actual, D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegDouble_Exp() throws Throwable {
        try (var eval = compile("exp(x)"); Arena arena = Arena.ofShared()) {
            int n = 25;
            double[] x = fillLinearD(n, -3.0, 0.25);

            MemorySegment in = arena.allocate((long) n * Double.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_DOUBLE, 0L, n);

            MemorySegment out = arena.allocate((long) n * Double.BYTES);
            eval.applyBulk(in, out);

            for (int i = 0; i < n; i++) {
                double actual = out.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                assertEquals(Math.exp(x[i]), actual, D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegDouble_Parallel() throws Throwable {
        try (var eval = compile("x * y"); Arena arena = Arena.ofShared()) {
            // Large enough to clear PARALLEL_OPS_THRESHOLD and actually
            // exercise the worker-pool dispatch path (falls back to
            // single-threaded automatically if NUM_WORKERS <= 0).
            int n = 200_000;
            double[] x = fillLinearD(n, 0.5, 0.0001);
            double[] y = fillLinearD(n, -0.5, 0.0002);

            MemorySegment in = arena.allocate((long) 2 * n * Double.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_DOUBLE, 0L, n);
            MemorySegment.copy(y, 0, in, ValueLayout.JAVA_DOUBLE, (long) n * Double.BYTES, n);

            MemorySegment out = arena.allocate((long) n * Double.BYTES);
            eval.applyBulkParallel(in, out);

            // Spot-check a sample rather than every element to keep the test fast.
            for (int i = 0; i < n; i += 997) {
                double actual = out.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                assertEquals(x[i] * y[i], actual, D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegDouble_Erf() throws Throwable {
        try (var eval = compile("erf(x)"); Arena arena = Arena.ofShared()) {
            int n = 30;
            double[] x = fillLinearD(n, -3.0, 0.2);

            MemorySegment in = arena.allocate((long) n * Double.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_DOUBLE, 0L, n);

            MemorySegment out = arena.allocate((long) n * Double.BYTES);
            eval.applyBulk(in, out);

            for (int i = 0; i < n; i++) {
                double actual = out.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                assertEquals(erfRef(x[i]), actual, D_LOOSE_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegDouble_MultiBlock() throws Throwable {
        try (var eval = compile("x - 1"); Arena arena = Arena.ofShared()) {
            int n = BLOCK_SIZE * 2 + 5;
            double[] x = fillLinearD(n, 0.0, 0.01);

            MemorySegment in = arena.allocate((long) n * Double.BYTES);
            MemorySegment.copy(x, 0, in, ValueLayout.JAVA_DOUBLE, 0L, n);

            MemorySegment out = arena.allocate((long) n * Double.BYTES);
            eval.applyBulk(in, out);

            for (int i = 0; i < n; i++) {
                double actual = out.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                assertEquals(x[i] - 1.0, actual, D_TIGHT_EPS, "index " + i);
            }
        }
    }

    // =====================================================================
    // 8. MemorySegment[] (one segment per variable, zero-copy) path
    // =====================================================================

    @Test
    void testMemSegArrayDouble_Add() throws Throwable {
        try (var eval = compile("x + y"); Arena arena = Arena.ofShared()) {
            int n = 90;
            double[] x = fillLinearD(n, 1.0, 0.1);
            double[] y = fillLinearD(n, 2.0, 0.2);

            MemorySegment segX = arena.allocate((long) n * Double.BYTES);
            MemorySegment segY = arena.allocate((long) n * Double.BYTES);
            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_DOUBLE, 0L, n);
            MemorySegment.copy(y, 0, segY, ValueLayout.JAVA_DOUBLE, 0L, n);

            MemorySegment out = arena.allocate((long) n * Double.BYTES);
            eval.applyBulk(new MemorySegment[]{segX, segY}, out);

            for (int i = 0; i < n; i++) {
                double actual = out.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                assertEquals(x[i] + y[i], actual, D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegArrayDouble_Mul() throws Throwable {
        try (var eval = compile("x * y - x"); Arena arena = Arena.ofShared()) {
            int n = 55;
            double[] x = fillLinearD(n, 2.0, 0.05);
            double[] y = fillLinearD(n, 3.0, 0.03);

            MemorySegment segX = arena.allocate((long) n * Double.BYTES);
            MemorySegment segY = arena.allocate((long) n * Double.BYTES);
            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_DOUBLE, 0L, n);
            MemorySegment.copy(y, 0, segY, ValueLayout.JAVA_DOUBLE, 0L, n);

            MemorySegment out = arena.allocate((long) n * Double.BYTES);
            eval.applyBulk(new MemorySegment[]{segX, segY}, out);

            for (int i = 0; i < n; i++) {
                double actual = out.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double expected = x[i] * y[i] - x[i];
                assertEquals(expected, actual, D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegArrayDouble_Parallel() throws Throwable {
        try (var eval = compile("x + y * 2"); Arena arena = Arena.ofShared()) {
            int n = 200_000;
            double[] x = fillLinearD(n, 0.1, 0.00005);
            double[] y = fillLinearD(n, -0.1, 0.00003);

            MemorySegment segX = arena.allocate((long) n * Double.BYTES);
            MemorySegment segY = arena.allocate((long) n * Double.BYTES);
            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_DOUBLE, 0L, n);
            MemorySegment.copy(y, 0, segY, ValueLayout.JAVA_DOUBLE, 0L, n);

            MemorySegment out = arena.allocate((long) n * Double.BYTES);
            eval.applyBulkParallel(new MemorySegment[]{segX, segY}, out);

            for (int i = 0; i < n; i += 997) {
                double actual = out.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double expected = x[i] + y[i] * 2.0;
                assertEquals(expected, actual, D_TIGHT_EPS, "index " + i);
            }
        }
    }

    @Test
    void testMemSegArrayDouble_BareVariable() throws Throwable {
        // Edge case: the whole expression is just "x", so OP_LOAD's
        // segment-backed push is never consumed by any op that would
        // otherwise force materialization — exercises the explicit
        // materialize() fallback in applyBulkInternal(MemorySegment[]...).
        try (var eval = compile("x"); Arena arena = Arena.ofShared()) {
            int n = 48;
            double[] x = fillLinearD(n, -10.0, 0.4);

            MemorySegment segX = arena.allocate((long) n * Double.BYTES);
            MemorySegment.copy(x, 0, segX, ValueLayout.JAVA_DOUBLE, 0L, n);

            MemorySegment out = arena.allocate((long) n * Double.BYTES);
            eval.applyBulk(new MemorySegment[]{segX}, out);

            for (int i = 0; i < n; i++) {
                double actual = out.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                assertEquals(x[i], actual, D_TIGHT_EPS, "index " + i);
            }
        }
    }

    // =====================================================================
    // Reference erf() implementation (double precision) used only to check
    // the erf/gelu float approximation against a trustworthy baseline.
    // =====================================================================
    private static double erfRef(double x) {
        // Abramowitz & Stegun 7.1.26, evaluated in double precision as the
        // "ground truth" for the float approximation under test.
        double sign = x < 0 ? -1.0 : 1.0;
        double ax = Math.abs(x);
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
                a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        double t = 1.0 / (1.0 + p * ax);
        double poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
        double y = 1.0 - poly * Math.exp(-ax * ax);
        return sign * y;
      //  return Maths.erf(x);
    }
}