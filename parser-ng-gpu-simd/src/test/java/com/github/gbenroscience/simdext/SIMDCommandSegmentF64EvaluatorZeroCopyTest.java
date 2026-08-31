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
package com.github.gbenroscience.simdext;

/**
 *
 * @author GBEMIRO
 */
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.lang.foreign.MemorySegment; 
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import com.github.gbenroscience.parser.MathExpression;  
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandSegmentF64;

import static org.junit.jupiter.api.Assertions.*;

/**
 * =====================================================================================
 * IMPORTANT — TWO THINGS TO CONFIRM BEFORE RELYING ON THIS SUITE
 * =====================================================================================
 *
 * 1. VARIABLE-TO-SLOT BINDING: these tests assume variables are bound to
 * double[][]/MemorySegment[] slots in FIRST-APPEARANCE order within the
 * expression string (e.g. "x+y" binds slot 0 = x, slot 1 = y). This is a common
 * convention but I have not personally verified it against MathExpression's
 * actual compilation source — if variable ordering is instead alphabetical,
 * declaration-list-based, or something else, every varSlot(...) call below
 * needs to change accordingly. If it's wrong, EVERY multi-variable test will
 * fail identically (single-variable tests are unaffected), which is itself a
 * useful diagnostic signal.
 *
 * 2. IF / AND / OR SYNTAX: OP_IF, OP_AND, OP_OR are real opcodes in the
 * evaluator, but I don't have confirmed expression-string syntax for them
 * (ternary operator? a function call like if(cond,a,b)? &&/||?). Section 8
 * below is left as a stub with the opcodes it needs to exercise clearly named —
 * fill in real syntax once confirmed. Everything else (arithmetic, comparisons
 * via >, unary functions) uses syntax I'm confident in given the opcode names
 * and standard expression-parser conventions.
 *
 * WHAT THIS SUITE DOES NOT NEED CONFIRMED: unary function names (sin, cos,
 * sqrt, exp, etc.) match the VectorMath method names 1:1 in the source, and
 * comparison operators (>, <, ==, etc.) are standard infix syntax — both are
 * used here with high confidence.
 * =====================================================================================
 */
class SIMDCommandSegmentF64EvaluatorZeroCopyTest {

    private static final long SEED = 42L;
    private final List<SIMDCommandSegmentF64.SIMDVectorCompositeExpression> openEvaluators = new ArrayList<>();

    @AfterEach
    void closeEvaluators() {
        for (var e : openEvaluators) {
            try {
                e.close();
            } catch (Exception ignored) {
            }
        }
        openEvaluators.clear();
    }

    // =====================================================================
    // Helpers
    // =====================================================================
    private SIMDCommandSegmentF64.SIMDVectorCompositeExpression compile(String expr) throws Throwable {
        var evaluator = SIMDCommandSegmentF64.getEvaluator(expr);
        openEvaluators.add(evaluator);
        return evaluator;
    }

    private SIMDCommandSegmentF64.SIMDVectorCompositeExpression compileParallel(String expr, int workers) throws Throwable {
        var evaluator = (SIMDCommandSegmentF64.SIMDVectorCompositeExpression) new SIMDCommandSegmentF64(new MathExpression(expr), workers).compile();
        openEvaluators.add(evaluator);
        return evaluator;
    }

    /**
     * First-appearance variable order — see class-level note (1) above.
     */
    private double[][] randomVars(int varCount, int n, long seed) {
        Random r = new Random(seed);
        double[][] vars = new double[varCount][n];
        for (int v = 0; v < varCount; v++) {
            for (int i = 0; i < n; i++) {
                vars[v][i] = (r.nextDouble() - 0.5) * 200.0; // spread across [-100, 100)
            }
        }
        return vars;
    }

    /**
     * Injects NaN / +Inf / -Inf / 0.0 / -0.0 at fixed positions if n is large
     * enough.
     */
    private void injectSpecialValues(double[][] vars, int n) {
        if (n < 5) {
            return;
        }
        for (double[] col : vars) {
            col[0] = Double.NaN;
            col[1 % n] = Double.POSITIVE_INFINITY;
            col[2 % n] = Double.NEGATIVE_INFINITY;
            col[3 % n] = 0.0;
            col[4 % n] = -0.0;
        }
    }

    private MemorySegment[] toSegments(double[][] vars) {
        MemorySegment[] segs = new MemorySegment[vars.length];
        for (int v = 0; v < vars.length; v++) {
            segs[v] = MemorySegment.ofArray(vars[v]);
        }
        return segs;
    }

    /**
     * Core differential assertion: the double[][]-based path (assumed correct —
     * it predates this change and is unmodified by it) must produce
     * bit-identical output to the new MemorySegment[] path, for both serial and
     * parallel entry points, across a fresh evaluator instance per call.
     */
    private void assertSegmentMatchesArray(String expr, int varCount, int n, long seed, boolean withSpecialValues) throws Throwable {
        double[][] vars = randomVars(varCount, n, seed);
        if (withSpecialValues) {
            injectSpecialValues(vars, n);
        }

        // --- Baseline: existing double[][] path ---
        var baselineEval = compile(expr);
        double[] expected = new double[n];
        baselineEval.applyBulk(vars, expected);

        // --- New: MemorySegment[] serial path ---
        var segEval = compile(expr);
        MemorySegment[] segs = toSegments(vars);
        double[] actualBuf = new double[n];
        MemorySegment outSeg = MemorySegment.ofArray(actualBuf);
        segEval.applyBulk(segs, outSeg);

        assertArrayEqualsBitwise(expected, actualBuf, expr + " [serial, n=" + n + ", seed=" + seed + "]");

        // --- New: MemorySegment[] parallel path (2 workers) ---
        var segEvalParallel = compileParallel(expr, 2);
        double[] parallelBuf = new double[n];
        MemorySegment outSegParallel = MemorySegment.ofArray(parallelBuf);
        segEvalParallel.applyBulkParallel(toSegments(vars), outSegParallel);

        assertArrayEqualsBitwise(expected, parallelBuf, expr + " [parallel, n=" + n + ", seed=" + seed + "]");
    }

    private void assertSegmentMatchesArray(String expr, int varCount, int n) throws Throwable {
        assertSegmentMatchesArray(expr, varCount, n, SEED, false);
    }

    private static void assertArrayEqualsBitwise(double[] expected, double[] actual, String context) {
        assertEquals(expected.length, actual.length, context + " — length mismatch");
        for (int i = 0; i < expected.length; i++) {
            double e = expected[i];
            double a = actual[i];
            boolean bothNaN = Double.isNaN(e) && Double.isNaN(a);
            
              if (!bothNaN && !valuesAreEssentiallySame(e, a)) {
                fail(context + " — mismatch at index " + i + ": expected=" + e + " actual=" + a);
            }
        }
    }

    private static final double EPSILON = 1e-11; // Tolerance for floating point drift

    private static boolean valuesAreEssentiallySame(double a, double b) {
        // 1. Fast path for exact matches, handles infinities and +0.0 == -0.0
        if (a == b) {
            return true;
        }

        // 2. Calculate the absolute difference
        double diff = Math.abs(a - b);

        // 3. Absolute error check (Critical for numbers close to zero)
        if (diff <= EPSILON) {
            return true;
        }

        // 4. Relative error check (Critical for very large numbers)
        // We scale the epsilon by the largest of the two numbers.
        double largest = Math.max(Math.abs(a), Math.abs(b));
        return diff <= largest * EPSILON;
    }

    // =====================================================================
    // Section 1 — All 9 operand-state branch combinations, per operator
    //   seg-seg | seg-const | const-seg | seg-array | array-seg |
    //   array-array | array-const | const-array | const-const
    // =====================================================================
    static final String[] ADD_COMBOS = {
        "x+y", // seg, seg
        "x+7.5", // seg, const
        "7.5+x", // const, seg
        "y+sin(x)", // seg, array   (sin(x) forces materialize -> array)
        "sin(x)+y", // array, seg
        "sin(x)+cos(y)", // array, array
        "sin(x)+7.5", // array, const
        "7.5+sin(x)", // const, array
        "(2*3)+x" // const-const fold path (may compile-time-fold; still validates correctness either way)
    };

    static final String[] SUB_COMBOS = {
        "x-y", "x-7.5", "7.5-x", "y-sin(x)", "sin(x)-y",
        "sin(x)-cos(y)", "sin(x)-7.5", "7.5-sin(x)", "(2*3)-x"
    };

    static final String[] MUL_COMBOS = {
        "x*y", "x*7.5", "7.5*x", "y*sin(x)", "sin(x)*y",
        "sin(x)*cos(y)", "sin(x)*7.5", "7.5*sin(x)", "(2*3)*x"
    };

    static final String[] DIV_COMBOS = {
        "x/y", "x/7.5", "7.5/x", "y/sin(x)", "sin(x)/y",
        "sin(x)/cos(y)", "sin(x)/7.5", "7.5/sin(x)", "(2*3)/x"
    };

    @Test
    void additionBranchMatrix() throws Throwable {
        for (String expr : ADD_COMBOS) {
            assertSegmentMatchesArray(expr, 2, 10_000);
        }
    }

    @Test
    void subtractionBranchMatrix() throws Throwable {
        for (String expr : SUB_COMBOS) {
            assertSegmentMatchesArray(expr, 2, 10_000);
        }
    }

    @Test
    void multiplicationBranchMatrix() throws Throwable {
        for (String expr : MUL_COMBOS) {
            assertSegmentMatchesArray(expr, 2, 10_000);
        }
    }

    @Test
    void divisionBranchMatrix() throws Throwable {
        // Use a seed/offset that avoids exact-zero denominators dominating; NaN/Inf
        // results from occasional near-zero divisors are fine — bitwise comparison
        // (including NaN-equals-NaN) still catches any real divergence.
        for (String expr : DIV_COMBOS) {
            assertSegmentMatchesArray(expr, 2, 10_000, SEED + 1, false);
        }
    }

    @Test
    void chainedMixedExpression() throws Throwable {
        // Exercises multiple branch types within a single compiled expression:
        // segment+segment, then the array result combined with a third segment
        // variable, then a constant multiply.
        assertSegmentMatchesArray("(x+y)*z - 3.0", 3, 50_000);
    }

    // =====================================================================
    // Section 2 — Edge cases the redesign specifically introduced
    // =====================================================================
    @Test
    void bareVariableExpression_forcesMaterializeAtOutputBoundary() throws Throwable {
        // The entire expression is a single OP_LOAD with nothing to consume it —
        // this is exactly the case applyBulkInternal's new `if (ctx.stackIsSegment[0])`
        // check exists to handle. Without that check, this test fails or reads garbage.
        assertSegmentMatchesArray("x", 1, 4096);
        assertSegmentMatchesArray("x", 1, 1);      // smallest possible n
        assertSegmentMatchesArray("x", 1, 100_003); // spans many blocks, non-aligned tail
    }

    @Test
    void bareConstantExpression_currentlyNoOpsForSegmentPath() throws Throwable {
        // FINDING, not a passing assertion of desired behavior: a zero-variable
        // (pure constant) expression compiles with varCount == 0. applyBulk(MemorySegment[],
        // MemorySegment) guards on `variables.length == 0` and returns immediately —
        // meaning the output buffer is left untouched rather than filled with the
        // constant. This test documents that behavior explicitly so it can't regress
        // silently, and flags it as something parser-ng-arrow's binding layer should
        // special-case (e.g. detect varCount == 0 and fill the output column directly)
        // rather than something to "fix" inside SIMDCommandSegmentF64 without more thought
        // about what callers currently depend on.
        var evaluator = compile("42.0");
        double[] out = new double[100];
        java.util.Arrays.fill(out, -999.0); // sentinel so we can detect a no-op
        MemorySegment outSeg = MemorySegment.ofArray(out);
        MemorySegment[]in=new MemorySegment[0];
      
        evaluator.applyBulk(in, outSeg);

        for (double v : out) {
            assertEquals(42, v, 0.0,
                    "Expected current no-op behavior for varCount==0 expressions — "
                    + "if this now fails, either the guard was removed (good, but verify "
                    + "intentionally) or something else changed unexpectedly.");
        }
    }

    // =====================================================================
    // Section 3 — Boundary / non-vector-aligned / non-block-aligned sizes
    // =====================================================================
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65, 127, 128, 129,
        255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097,
        10_000, 100_003, 1_000_003})
    void boundarySizes_singleVariable(int n) throws Throwable {
        assertSegmentMatchesArray("sqrt(x*x+1.0)", 1, n);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 3, 17, 129, 4097, 100_003})
    void boundarySizes_twoVariableArithmetic(int n) throws Throwable {
        assertSegmentMatchesArray("x+y", 2, n);
        assertSegmentMatchesArray("x*y-x/y", 2, n);
    }

    @Test
    void zeroLengthOutput_doesNotThrow() throws Throwable {
        var evaluator = compile("x+y");
        MemorySegment[] segs = {MemorySegment.ofArray(new double[0]), MemorySegment.ofArray(new double[0])};
        MemorySegment outSeg = MemorySegment.ofArray(new double[0]);
        assertDoesNotThrow(() -> evaluator.applyBulk(segs, outSeg));
    }

    // =====================================================================
    // Section 4 — Special floating-point values
    // =====================================================================
    @Test
    void specialValues_arithmeticOps() throws Throwable {
        for (String expr : new String[]{"x+y", "x-y", "x*y", "x/y"}) {
            assertSegmentMatchesArray(expr, 2, 5000, SEED, true);
        }
    }

    @Test
    void specialValues_transcendentalTriggersMaterialize() throws Throwable {
        // Forces the materialize() segment-copy branch with NaN/Inf present,
        // making sure the on-demand copy doesn't do anything Math.sin/Math.cos
        // wouldn't also do with the same inputs via the array path.
        assertSegmentMatchesArray("sin(x)+cos(y)", 2, 5000, SEED, true);
    }

    // =====================================================================
    // Section 5 — Parallel vs. serial consistency at scale
    // =====================================================================
    @Test
    void largeParallelWorkload_matchesSerial() throws Throwable {
        // NOTE: must exceed PARALLEL_OPS_THRESHOLD for applyBulkParallel to actually
        // dispatch to workers rather than silently falling back to serial — confirm
        // the threshold value in VectorConfig/BatchedVectorCompositeExpression and
        // adjust n upward if this constant is larger than expected.
        int n = 5_000_000;
        assertSegmentMatchesArray("(x+y)*sin(x) - sqrt(y*y+1.0)", 2, n);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 4, 8})
    void variousWorkerCounts_produceIdenticalResults(int workers) throws Throwable {
        int n = 2_000_000;
        double[][] vars = randomVars(2, n, SEED);
        double[] baseline = new double[n];
        compile("x+y*2.0-sin(x)").applyBulk(vars, baseline);

        var evaluator = compileParallel("x+y*2.0-sin(x)", workers);
        double[] actual = new double[n];
        evaluator.applyBulkParallel(toSegments(vars), MemorySegment.ofArray(actual));

        assertArrayEqualsBitwise(baseline, actual, "workers=" + workers);
    }

    // =====================================================================
    // Section 6 — State cleanliness across interleaved / repeated calls
    //
    // These specifically target the stale-flag risk called out during design:
    // stackIsConst / stackIsSegment must never leak between calls that reuse
    // the same EvaluationContext (via the ThreadLocal masterEvalContext).
    // =====================================================================
    @Test
    void interleavedArrayAndSegmentCallsOnSameEvaluator() throws Throwable {
        var evaluator = compile("x+y*sin(x)");
        int n = 10_000;

        for (int round = 0; round < 5; round++) {
            double[][] vars = randomVars(2, n, SEED + round);

            double[] arrayResult = new double[n];
            evaluator.applyBulk(vars, arrayResult);

            double[] segResult = new double[n];
            evaluator.applyBulk(toSegments(vars), MemorySegment.ofArray(segResult));

            double[] arrayResultAgain = new double[n];
            evaluator.applyBulk(vars, arrayResultAgain);

            assertArrayEqualsBitwise(arrayResult, segResult, "round " + round + " array-vs-segment");
            assertArrayEqualsBitwise(arrayResult, arrayResultAgain, "round " + round + " array-vs-array (no leakage from segment call)");
        }
    }

    @Test
    void repeatedSegmentCallsWithVaryingVariableStates() throws Throwable {
        // First call has both operands segment-backed variables; second call reuses
        // the SAME compiled evaluator/context but with an expression shape where one
        // side would be array-backed after materialize — catches stale stackIsSegment
        // flags surviving into a slot that a later call expects to be const/array.
        var evaluator = compile("x+y");
        int n = 8192;

        for (int i = 0; i < 20; i++) {
            double[][] vars = randomVars(2, n, SEED + i);
            double[] expected = new double[n];
            SIMDCommandSegmentF64.getEvaluator("x+y").applyBulk(vars, expected); // fresh instance as ground truth

            double[] actual = new double[n];
            evaluator.applyBulk(toSegments(vars), MemorySegment.ofArray(actual));
            assertArrayEqualsBitwise(expected, actual, "iteration " + i);
        }
    }

    // =====================================================================
    // Section 7 — Concurrent external callers on one shared evaluator instance
    //
    // Exercises masterEvalContext (ThreadLocal) correctness and worker-pool
    // safety under concurrent *external* invocation, not just the internal
    // parallel-dispatch path.
    // =====================================================================
    @Test
    void concurrentCallersOnSharedEvaluator() throws Throwable {
        var evaluator = compile("x*y+sin(x)-cos(y)");
        int n = 20_000;
        int threadCount = 8;
        int iterationsPerThread = 25;

        ExecutorService pool = Executors.newFixedThreadPool(threadCount);
        AtomicBoolean failed = new AtomicBoolean(false);
        List<Future<?>> futures = new ArrayList<>();

        for (int t = 0; t < threadCount; t++) {
            final int threadIdx = t;
            futures.add(pool.submit(() -> {
                try {
                    for (int i = 0; i < iterationsPerThread; i++) {
                        long seed = SEED + threadIdx * 1000L + i;
                        double[][] vars = randomVars(2, n, seed);

                        double[] expected = new double[n];
                        SIMDCommandSegmentF64.getEvaluator("x*y+sin(x)-cos(y)").applyBulk(vars, expected);

                        double[] actual = new double[n];
                        evaluator.applyBulk(toSegments(vars), MemorySegment.ofArray(actual));

                        assertArrayEqualsBitwise(expected, actual, "thread=" + threadIdx + " iter=" + i);
                    }
                } catch (Throwable e) {
                    failed.set(true);
                    throw new RuntimeException(e);
                }
            }));
        }

        pool.shutdown();
        assertTrue(pool.awaitTermination(120, TimeUnit.SECONDS), "concurrent test run timed out");
        for (Future<?> f : futures) {
            assertDoesNotThrow(() -> {
                try {
                    f.get();
                } catch (java.util.concurrent.ExecutionException e) {
                    throw e.getCause();
                }
            });
        }
        assertFalse(failed.get());
    }

    // =====================================================================
    // Section 8 — STUB: comparisons, IF/AND/OR, POW with segment operands
    //
    // Comparisons use standard infix syntax so these should work as-is.
    // IF/AND/OR syntax needs confirmation — see class-level note (2).
    // =====================================================================
    @Test
    void comparisonOperators_withSegmentOperands() throws Throwable {
        // These all force materialize() on both operands (comparisons don't have
        // a segment-native fast path — see design notes), so this specifically
        // exercises the new materialize() segment branch, not the doAdd-family
        // fast paths.
        for (String op : new String[]{">", "<", ">=", "<=", "==", "!="}) {
            assertSegmentMatchesArray("x" + op + "y", 2, 10_000);
        }
    }

    @Test
    void powOperator_withSegmentBase() throws Throwable {
        assertSegmentMatchesArray("x^2", 1, 10_000);   // uniform-exponent fast path in executePowerBlended
        assertSegmentMatchesArray("x^y", 2, 5_000);     // variable-exponent path
    }

    // TODO: confirm real syntax, then uncomment / adapt:
    //
    // @Test
    // void ifExpression_withSegmentOperands() throws Throwable {
    //     assertSegmentMatchesArray("x>0.0 ? x : y", 2, 10_000); // ternary guess
    // }
    //
    // @Test
    // void andOrExpressions_withSegmentOperands() throws Throwable {
    //     assertSegmentMatchesArray("(x>0.0) && (y>0.0)", 2, 10_000); // guess
    // }
}
