package com.github.gbenroscience.simd;

/**
 *
 * @author oluwagbemirojiboye
 */
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.SIMDEngineEvaluator;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class MemorySegmentEvaluatorTest {

    private static final double EPSILON = 1e-11;
    private final boolean active = true;

    // Dummy method to match your original test's structural dependency
    private void logDetails(MathExpression me, SIMDEngineEvaluator.SIMDVectorCompositeExpression eval, boolean flag) {
        // Logging implementation
    }

    /**
     * Test API Call: Serial MemorySegment Bulk Execution Uses
     * Arena.ofConfined() since processing stays on the calling thread.
     */
    @Test
    public void testMathematicalPrecisionVsNativeMemorySegmentBulk() throws Throwable {
        MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        SIMDEngineEvaluator.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(me).compile();

        logDetails(me, evaluator, !active);

        // 17 datapoints to trigger both vector lane and tail scalar loop remainders
        long totalElements = 17;
        int varCount = 3; // x1, x2, x3

        // Use confined arena for single-threaded evaluation
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment inputSegment = arena.allocate(varCount * totalElements * 8L);
            MemorySegment outputSegment = arena.allocate(totalElements * 8L);

            // Populate the memory segment using stride offsets
            for (long i = 0; i < totalElements; i++) {
                double x1Val = 1.5 + (i * 0.1);
                double x2Val = 2.0 + (i * 0.5);
                double x3Val = 0.5;

                inputSegment.set(ValueLayout.JAVA_DOUBLE, ((0 * totalElements) + i) * 8L, x1Val); // x1
                inputSegment.set(ValueLayout.JAVA_DOUBLE, ((1 * totalElements) + i) * 8L, x2Val); // x2
                inputSegment.set(ValueLayout.JAVA_DOUBLE, ((2 * totalElements) + i) * 8L, x3Val); // x3
            }

            // Execute Serial Segment Path
            evaluator.applyBulk(inputSegment, outputSegment);

            // Verify mathematical equality against standard Java scalar paths
            for (long i = 0; i < totalElements; i++) {
                double x1 = inputSegment.get(ValueLayout.JAVA_DOUBLE, ((0 * totalElements) + i) * 8L);
                double x2 = inputSegment.get(ValueLayout.JAVA_DOUBLE, ((1 * totalElements) + i) * 8L);
                double x3 = inputSegment.get(ValueLayout.JAVA_DOUBLE, ((2 * totalElements) + i) * 8L);

                double expected = (1.0 / (x1 * Math.sqrt(2.0 * 3.14159)))
                        * Math.exp((-Math.pow((x2 - x3), 2.0)) / (2.0 * Math.pow(x1, 2.0)));

                double actual = outputSegment.get(ValueLayout.JAVA_DOUBLE, i * 8L);
                assertEquals(expected, actual, EPSILON, "SIMD MemorySegment (Serial) math drifted at index: " + i);
            }
        }
    }

    /**
     * Test API Call: Parallel MemorySegment Bulk Execution Uses
     * Arena.ofShared() to allow worker threads to access the memory segments.
     */
    @Test
    public void testMathematicalPrecisionVsNativeMemorySegmentBulkParallel() throws Throwable {
        MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        // Instantiate with 4 workers to ensure parallel execution pool is created
        SIMDEngineEvaluator.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(me, 4).compile();

        logDetails(me, evaluator, !active);

        // Use a dataset large enough to exceed PARALLEL_OPS_THRESHOLD
        long totalElements = 300_000_000L;
        int varCount = 3; // x1, x2, x3
        double start = System.nanoTime();
        // MUST use shared arena for multi-threaded memory access
        try (Arena arena = Arena.ofShared()) {
            MemorySegment inputSegment = arena.allocate(varCount * totalElements * 8L);
            MemorySegment outputSegment = arena.allocate(totalElements * 8L);
            System.out.println("Started loading input data");
            for (long i = 0; i < totalElements; i++) {
                double x1Val = 1.5 + (i * 0.001);
                double x2Val = 2.0 + (i * 0.005);
                double x3Val = 0.5;

                inputSegment.set(ValueLayout.JAVA_DOUBLE, ((0 * totalElements) + i) * 8L, x1Val);
                inputSegment.set(ValueLayout.JAVA_DOUBLE, ((1 * totalElements) + i) * 8L, x2Val);
                inputSegment.set(ValueLayout.JAVA_DOUBLE, ((2 * totalElements) + i) * 8L, x3Val);
            }
            double t0 = System.nanoTime() - start;
            System.out.println("Done loading input data in " + t0 + "ns");

            start = System.nanoTime();
            // Execute Parallel Segment Path
            evaluator.applyBulkParallel(inputSegment, outputSegment);
            double t1 = System.nanoTime() - start;
            System.out.println("evalTime = " + t1 + "ns");

            System.out.println("Started testing output results");
            start = System.nanoTime();
            for (long i = 0; i < totalElements; i++) {
                double x1 = inputSegment.get(ValueLayout.JAVA_DOUBLE, ((0 * totalElements) + i) * 8L);
                double x2 = inputSegment.get(ValueLayout.JAVA_DOUBLE, ((1 * totalElements) + i) * 8L);
                double x3 = inputSegment.get(ValueLayout.JAVA_DOUBLE, ((2 * totalElements) + i) * 8L);

                double expected = (1.0 / (x1 * Math.sqrt(2.0 * 3.14159)))
                        * Math.exp((-Math.pow((x2 - x3), 2.0)) / (2.0 * Math.pow(x1, 2.0)));

                double actual = outputSegment.get(ValueLayout.JAVA_DOUBLE, i * 8L);
                assertEquals(expected, actual, EPSILON, "SIMD MemorySegment (Parallel) math drifted at index: " + i);
            }
            double t2 = System.nanoTime() - start;
            System.out.println("Ended testing output results in "+t2+"ns");
        }
    }

    /**
     * Test API Call: Edge Case - Null / Empty Segments Validates that the
     * evaluator fails gracefully or bypasses correctly without attempting
     * illegal memory dereferencing.
     */
    @Test
    public void testMemorySegmentBulkHandlesNullGracefully() throws Throwable {
        MathExpression me = new MathExpression("x * 2");
        SIMDEngineEvaluator.SIMDVectorCompositeExpression evaluator
                = (SIMDEngineEvaluator.SIMDVectorCompositeExpression) new SIMDEngineEvaluator(me).compile();

        try (Arena arena = Arena.ofConfined()) {
            // Test with null segments
            evaluator.applyBulk((MemorySegment) null, null);
            evaluator.applyBulkParallel((MemorySegment) null, null);

            // Test with valid empty segments (0 bytes)
            MemorySegment emptyInput = arena.allocate(0L);
            MemorySegment emptyOutput = arena.allocate(0L);

            evaluator.applyBulk(emptyInput, emptyOutput);
            evaluator.applyBulkParallel(emptyInput, emptyOutput);

            // If we reach here without NullPointerException or IndexOutOfBounds, the guard checks work
            assertEquals(0, emptyOutput.byteSize());
        }
    }
}
