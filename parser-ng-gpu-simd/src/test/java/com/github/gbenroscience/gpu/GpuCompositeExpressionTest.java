/**
 *
 * @author oluwagbemirojiboye
 */
package com.github.gbenroscience.gpu;

import com.github.gbenroscience.gpu.evaluator.GpuCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.GpuExpressionBridge;
import com.github.gbenroscience.gpu.evaluator.opencl.OpenClCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.opencl.OpenClExpressionBridge;
import com.github.gbenroscience.gpu.evaluator.opencl.OpenClKernelSource;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.SIMDCompositeExpression; 
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator; 
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF64;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Requires an actual OpenCL GPU device -- run with -Dgpu.tests=true on a
 * machine that has one.
 *
 * Two tiers of tests here, deliberately using different entry points:
 *
 * 1. LOW-LEVEL / PLUMBING tests (basicCorrectness, the BUG#1-3 regressions,
 * deeplyNestedExpressionRejectedBeforeTouchingGpu, the two
 * memorySegmentOverloadMatches* tests, and the scatter* tests added
 * alongside them) construct
 * {@link OpenClCompositeExpression} directly and go through
 * {@link OpenClExpressionBridge} explicitly. They're testing concrete
 * OpenCL-backend internals (buffer growth, kernel-arg races, staging slicing)
 * that only make sense pinned to one backend -- a CudaCompositeExpressionTest
 * would mirror these for the CUDA backend.
 *
 * 2. CORRECTNESS / PERFORMANCE tests (everything under "CPU vs GPU" below) go
 * through the backend-agnostic {@link GpuExpressionBridge}, so this half of the
 * suite exercises whichever backend (CUDA preferred, OpenCL fallback -- see
 * GpuExpressionBridge's preference order) is actually available on the machine
 * running the tests.
 *
 * All CPU-vs-GPU comparisons run the SAME compiled opcode program
 * (VectorTurboEvaluator's bytecode) on both sides -- there is no second,
 * independently-written "expected value" oracle for the composite cases.
 * Divergence beyond the stated tolerance means one interpreter disagrees with
 * the other on identical bytecode, which is exactly the class of bug this suite
 * exists to catch.
 *
 * Expression under test in the low-level plumbing tests: f(x) = x + 1 opcodes =
 * [OP_LOAD, OP_CONST, OP_ADD], targetSlots = [0, 0, 0] (slot 0 unused for
 * CONST/ADD), literalConstants = [0, 1.0, 0] (only index 1 -- the CONST -- is
 * read).
 */
//@EnabledIfSystemProperty(named = "gpu.tests", matches = "true")
public class GpuCompositeExpressionTest {

    private static OpenClCompositeExpression xPlusOne() {
        int[] opcodes = {2 /*OP_LOAD*/, 1 /*OP_CONST*/, 3 /*OP_ADD*/};
        int[] targetSlots = {0, 0, 0};
        double[] literals = {0.0, 1.0, 0.0};
        return new OpenClCompositeExpression(opcodes, targetSlots, literals, 3, 1);
    }

    @Test
    void basicCorrectness() throws Throwable {
        try (OpenClCompositeExpression expr = xPlusOne()) {
            double[] in = {1, 2, 3, 4, 5};
            double[] out = new double[5];
            expr.applyBulk(in, out);
            assertArrayEquals(new double[]{2, 3, 4, 5, 6}, out, 1e-9);
        }
    }

    /**
     * BUG #1 regression: staging buffers are grow-only and reused. A call with
     * a large batch followed by a call with a SMALLER batch must not let
     * dispatch() infer dataSize from the old, larger staging capacity. Before
     * the fix, this either throws (device buffer size mismatch) or silently
     * returns extra/garbage-influenced elements.
     */
    @Test
    void shrinkingBatchAfterLargeBatchStaysCorrect() throws Throwable {
        try (OpenClCompositeExpression expr = xPlusOne()) {
            double[] bigIn = new double[10_000];
            for (int i = 0; i < bigIn.length; i++) {
                bigIn[i] = i;
            }
            double[] bigOut = new double[bigIn.length];
            expr.applyBulk(bigIn, bigOut);
            assertEquals(1.0, bigOut[0], 1e-9);

            double[] smallIn = {41.0};
            double[] smallOut = new double[1];
            expr.applyBulk(smallIn, smallOut);

            assertEquals(42.0, smallOut[0], 1e-9,
                    "dataSize must track the CURRENT call's length, not a stale staging-buffer capacity");
        }
    }

    /**
     * BUG #2 regression: growing `in` while `out` shrinks (or vice versa)
     * relative to a previous call's max(inBytes, outBytes) must still resize
     * whichever device buffer actually needs it.
     */
    @Test
    void asymmetricInOutGrowthResizesBothBuffersIndependently() throws Throwable {
        try (OpenClCompositeExpression expr = xPlusOne()) {
            // call 1: small in, small out -> both buffers sized small.
            expr.applyBulk(new double[]{1}, new double[1]);

            // call 2: a much larger elementwise call -- exercises growth of
            // both the input AND output device buffers together. (The
            // asymmetric-growth failure mode this guards against is: track
            // capacity as a single combined max(inBytes, outBytes) instead
            // of independently per buffer, so a call whose OUTPUT grows
            // past a previous call's combined max while its INPUT stays
            // small skips reallocating the output buffer entirely --
            // see OpenClCompositeExpression.ensureDeviceBuffers's javadoc.)
            double[] in2 = new double[5000];
            for (int i = 0; i < in2.length; i++) {
                in2[i] = i;
            }
            double[] out2 = new double[5000];
            assertDoesNotThrow(() -> expr.applyBulk(in2, out2));
            assertEquals(5000.0, out2[4999], 1e-9);
        }
    }

    /**
     * BUG #3 regression: GpuContext.KERNEL_F64/KERNEL_F32/QUEUE are static and
     * shared across every instance. Concurrent dispatch from multiple threads
     * (even on different OpenClCompositeExpression instances) must not
     * interleave clSetKernelArg calls across threads.
     */
    @Test
    void concurrentDispatchDoesNotCrossTalk() throws Throwable {
        int threads = 8;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        AtomicInteger failures = new AtomicInteger();
        CountDownLatch done = new CountDownLatch(threads);

        try {
            for (int t = 0; t < threads; t++) {
                final double offset = t;
                pool.submit(() -> {
                    try (OpenClCompositeExpression expr = xPlusOne()) {
                        for (int iter = 0; iter < 50; iter++) {
                            double[] in = {offset, offset + 1, offset + 2};
                            double[] out = new double[3];
                            expr.applyBulk(in, out);
                            if (Math.abs(out[0] - (offset + 1)) > 1e-9
                                    || Math.abs(out[1] - (offset + 2)) > 1e-9
                                    || Math.abs(out[2] - (offset + 3)) > 1e-9) {
                                failures.incrementAndGet();
                            }
                        }
                    } catch (Throwable e) {
                        failures.incrementAndGet();
                    } finally {
                        done.countDown();
                    }
                });
            }
            assertTrue(done.await(60, TimeUnit.SECONDS), "threads did not finish in time");
            assertEquals(0, failures.get(), "concurrent dispatch produced wrong results -- kernel arg race");
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void checkProperty() {
        System.out.println("gpu.tests = " + System.getProperty("gpu.tests"));
    }

    /**
     * The GPU kernel's per-thread stack is a fixed MAX_STACK-slot private array
     * with NO bounds checking -- unlike a Java array, silently walking past it
     * is undefined behavior on the device, not a catchable exception.
     * GpuExpressionBridge/OpenClExpressionBridge.from() is supposed to refuse
     * to hand such an expression to the GPU at all. This builds a right-nested
     * chain -- x+(x+(x+(...+(x+1)...))) -- whose AST depth (and therefore
     * VectorTurboEvaluator's stackDepth) grows linearly with nesting, so it
     * reliably exceeds MAX_STACK regardless of parser internals. MAX_STACK is
     * shared between the double and float kernels (same stack DEPTH bound, just
     * different element width), so this guard applies equally to both
     * precisions.
     */
    @Test
    void deeplyNestedExpressionRejectedBeforeTouchingGpu() throws Throwable {
        int nesting = OpenClKernelSource.MAX_STACK + 16;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < nesting; i++) {
            sb.append("x+(");
        }
        sb.append("1");
        sb.append(")".repeat(nesting));

        MathExpression me = new MathExpression(sb.toString());
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

        assertTrue(vte.getStackDepth() > OpenClKernelSource.MAX_STACK,
                "test construction assumption failed -- nesting didn't produce the expected stack depth; "
                + "actual depth=" + vte.getStackDepth());

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> OpenClExpressionBridge.from(vte));
        assertTrue(ex.getMessage().contains("stack depth"),
                "exception should explain WHY the expression was rejected, not just that it was");
    }

    /**
     * The double[] and MemorySegment overloads must be two doors into the same
     * double-precision dispatch path, not two independently-behaving
     * implementations. Runs the identical expression + identical data through
     * both entry points and requires bit-for-bit identical output (same GPU
     * dispatch underneath -- any difference here would mean the staging/copy
     * layer itself is introducing error, independent of the kernel). Sized to
     * comfortably exercise the path (a few million elements, tens of MB)
     * without the multi-GB allocations a much larger size would require -- this
     * is a correctness/plumbing test, not a throughput benchmark (see
     * largeGridInformationalTiming for that).
     */
    @Test
    void memorySegmentOverloadMatchesDoubleArrayOverload() throws Throwable {
        MathExpression me = new MathExpression("2*x^2-3*x+1");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

        try (OpenClCompositeExpression gpu = (OpenClCompositeExpression) OpenClExpressionBridge.from(vte); Arena arena = Arena.ofConfined()) {

            int dataSize = 2_000_000;
            double[] flat = new double[dataSize];
            for (int i = 0; i < dataSize; i++) {
                flat[i] = -25.6 + i * 0.1;
            }

            double[] outViaArray = new double[dataSize];
            long tArray = System.nanoTime();
            gpu.applyBulk(flat, outViaArray);
            tArray = System.nanoTime() - tArray;
            System.out.println("double[] path: " + (tArray / 1000) + " us");

            MemorySegment inSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize());
            MemorySegment outSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize());
            MemorySegment.copy(flat, 0, inSeg, ValueLayout.JAVA_DOUBLE, 0, dataSize);

            long tSeg = System.nanoTime();
            gpu.applyBulk(inSeg, outSeg);
            tSeg = System.nanoTime() - tSeg;
            System.out.println("MemorySegment path: " + (tSeg / 1000) + " us");

            double[] outViaSegment = new double[dataSize];
            MemorySegment.copy(outSeg, ValueLayout.JAVA_DOUBLE, 0, outViaSegment, 0, dataSize);

            assertArrayEquals(outViaArray, outViaSegment, 0.0,
                    "double[] and MemorySegment overloads diverged for identical input -- bug is in the "
                    + "staging/copy layer, not the kernel");
        }
    }

    /**
     * Float32 counterpart of memorySegmentOverloadMatchesDoubleArrayOverload --
     * same rationale, but exercising the NATIVE float kernel path
     * (interpretF32) rather than bridging through double. Uses
     * applyBulkF32(MemorySegment, MemorySegment) explicitly -- NOT
     * applyBulk(MemorySegment, MemorySegment), which always assumes double (see
     * GpuCompositeExpression's javadoc for why a bare MemorySegment can't
     * safely be dispatched by argument type the way float[]/double[] can).
     * Calling the wrong one here previously produced wildly wrong output
     * (bit-pattern garbage, not a rounding difference) rather than a clean
     * failure, since the double dispatch path would silently reinterpret pairs
     * of floats as double bit patterns.
     */
    @Test
    void memorySegmentOverloadMatchesFloatArrayOverload() throws Throwable {
        MathExpression me = new MathExpression("2*x^2-3*x+1");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

        try (OpenClCompositeExpression gpu = (OpenClCompositeExpression) OpenClExpressionBridge.from(vte); Arena arena = Arena.ofConfined()) {

            int dataSize = 2_000_000;
            float[] flat = new float[dataSize];
            for (int i = 0; i < dataSize; i++) {
                flat[i] = (float) (-25.6 + i * 0.1);
            }

            float[] outViaArray = new float[dataSize];
            long tArray = System.nanoTime();
            gpu.applyBulk(flat, outViaArray);
            tArray = System.nanoTime() - tArray;
            System.out.println("float[] path: " + (tArray / 1000) + " us");

            MemorySegment inSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment outSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(flat, 0, inSeg, ValueLayout.JAVA_FLOAT, 0, dataSize);

            long tSeg = System.nanoTime();
            gpu.applyBulkF32(inSeg, outSeg);
            tSeg = System.nanoTime() - tSeg;
            System.out.println("MemorySegment path (f32): " + (tSeg / 1000) + " us");

            float[] outViaSegment = new float[dataSize];
            MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, outViaSegment, 0, dataSize);

            assertArrayEquals(outViaArray, outViaSegment, 0.0f,
                    "float[] and MemorySegment (applyBulkF32) overloads diverged for identical input -- "
                    + "bug is in the staging/copy layer, not the kernel");
        }
    }

    // =====================================================================
    // Scatter overloads: applyBulk(MemorySegment[], MemorySegment) and
    // applyBulkF32(MemorySegment[], MemorySegment) -- the "true zero-copy"
    // multi-variable entry points added alongside double[][]/float[][].
    // Unlike double[][]/float[][], which must copy every row out of the
    // Java heap into a staging arena on every call, these two overloads
    // assume the caller ALREADY has each variable's data as native memory
    // and write each segment straight into its slice of the device input
    // buffer -- no host-side flatten/staging copy at all. The tests below
    // both check correctness (same output as the existing multi-variable
    // paths) and time the scatter overloads against the two paths they're
    // meant to improve on: double[][]/float[][] (heap-crossing row copy)
    // and the "caller flattens N native segments by hand, then calls
    // applyBulk(MemorySegment, MemorySegment)" pattern a caller with
    // already-native per-variable buffers previously had no alternative to.
    // =====================================================================
    /**
     * Correctness + timing: applyBulk(MemorySegment[], MemorySegment) against
     * applyBulk(double[][], double[]) for the same multi-variable expression
     * and identical underlying data. Both must produce bit-for-bit identical
     * output (same GPU dispatch underneath, just a different staging path),
     * and the scatter overload -- which skips the double[][] path's per-row
     * Java-heap-to-native copy entirely, since its input is already native --
     * must not be slower. Each path is warmed up once first (device-buffer
     * allocation and JIT warmup would otherwise pollute the first measured
     * call) and then averaged over several iterations to smooth out
     * scheduling noise.
     */
    @Test
    void scatterMemorySegmentOverloadMatchesDoubleArrayOfArraysOverload() throws Throwable {
        MathExpression me = new MathExpression("x^2+y^2-2*x*y*cos(x-y)");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

        try (OpenClCompositeExpression gpu = (OpenClCompositeExpression) OpenClExpressionBridge.from(vte);
                Arena arena = Arena.ofConfined()) {

            int dataSize = 1_000_000;
            int varCount = vte.getVarCount();
            double[] flat = buildSafeSamples(me, new String[]{"x", "y"},
                    new double[][]{{-8, 8}, {-8, 8}}, dataSize, varCount);
            double[][] rows = toRowsDouble(flat, varCount, dataSize);
            MemorySegment[] segs = toSegmentsDouble(arena, rows);
            MemorySegment outSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize());
            double[] outViaRows = new double[dataSize];

            // warmup: allocates device buffers and lets the JIT settle
            // before either path is timed for real.
            gpu.applyBulk(rows, outViaRows);
            gpu.applyBulk(segs, outSeg);

            int iterations = 5;
            long tRows = 0;
            long tScatter = 0;
            for (int i = 0; i < iterations; i++) {
                long t0 = System.nanoTime();
                gpu.applyBulk(rows, outViaRows);
                tRows += System.nanoTime() - t0;

                long t1 = System.nanoTime();
                gpu.applyBulk(segs, outSeg);
                tScatter += System.nanoTime() - t1;
            }
            tRows /= iterations;
            tScatter /= iterations;

            double[] outViaScatter = new double[dataSize];
            MemorySegment.copy(outSeg, ValueLayout.JAVA_DOUBLE, 0, outViaScatter, 0, dataSize);

            assertArrayEquals(outViaRows, outViaScatter, 0.0,
                    "double[][] and MemorySegment[] overloads diverged for identical input -- bug is in "
                    + "the staging/copy layer, not the kernel");

            System.out.printf("double[][] path: %,d us | MemorySegment[] (scatter) path: %,d us%n",
                    tRows / 1000, tScatter / 1000);
            assertTrue(tScatter <= tRows * 1.1,
                    "applyBulk(MemorySegment[], MemorySegment) skips the per-row Java-heap-crossing copy "
                    + "that applyBulk(double[][], double[]) requires on every call, so on identical "
                    + "native input it should not be slower (allowing a small margin for scheduling noise)");
        }
    }

    /**
     * Float32 counterpart of scatterMemorySegmentOverloadMatchesDoubleArrayOfArraysOverload
     * -- applyBulkF32(MemorySegment[], MemorySegment) against
     * applyBulk(float[][], float[]), same rationale.
     */
    @Test
    void scatterMemorySegmentOverloadMatchesFloatArrayOfArraysOverload() throws Throwable {
        MathExpression me = new MathExpression("x^2+y^2-2*x*y*cos(x-y)");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

        try (OpenClCompositeExpression gpu = (OpenClCompositeExpression) OpenClExpressionBridge.from(vte);
                Arena arena = Arena.ofConfined()) {

            int dataSize = 1_000_000;
            int varCount = vte.getVarCount();
            double[] flatD = buildSafeSamples(me, new String[]{"x", "y"},
                    new double[][]{{-8, 8}, {-8, 8}}, dataSize, varCount);
            float[] flat = new float[flatD.length];
            for (int i = 0; i < flatD.length; i++) {
                flat[i] = (float) flatD[i];
            }
            float[][] rows = toRowsFloat(flat, varCount, dataSize);
            MemorySegment[] segs = toSegmentsFloat(arena, rows);
            MemorySegment outSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
            float[] outViaRows = new float[dataSize];

            gpu.applyBulk(rows, outViaRows);
            gpu.applyBulkF32(segs, outSeg);

            int iterations = 5;
            long tRows = 0;
            long tScatter = 0;
            for (int i = 0; i < iterations; i++) {
                long t0 = System.nanoTime();
                gpu.applyBulk(rows, outViaRows);
                tRows += System.nanoTime() - t0;

                long t1 = System.nanoTime();
                gpu.applyBulkF32(segs, outSeg);
                tScatter += System.nanoTime() - t1;
            }
            tRows /= iterations;
            tScatter /= iterations;

            float[] outViaScatter = new float[dataSize];
            MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, outViaScatter, 0, dataSize);

            assertArrayEquals(outViaRows, outViaScatter, 0.0f,
                    "float[][] and MemorySegment[] (applyBulkF32) overloads diverged for identical input "
                    + "-- bug is in the staging/copy layer, not the kernel");

            System.out.printf("float[][] path: %,d us | MemorySegment[] (scatter, f32) path: %,d us%n",
                    tRows / 1000, tScatter / 1000);
            assertTrue(tScatter <= tRows * 1.1,
                    "applyBulkF32(MemorySegment[], MemorySegment) skips the per-row Java-heap-crossing "
                    + "copy that applyBulk(float[][], float[]) requires on every call, so on identical "
                    + "native input it should not be slower (allowing a small margin for scheduling noise)");
        }
    }

    /**
     * The actual motivation for applyBulk(MemorySegment[], MemorySegment): a
     * caller who already has one native MemorySegment per variable previously
     * had no choice but to flatten them into a single contiguous MemorySegment
     * BY HAND (a full host-side copy) before calling
     * applyBulk(MemorySegment, MemorySegment). The scatter overload reaches
     * the same device-side layout without that host-side flatten step at all
     * -- each native segment goes straight to its slice of the device buffer.
     * This times "caller flattens by hand, then calls the single-segment
     * overload" against "caller just calls the scatter overload directly" on
     * identical data and confirms the latter is faster: it is doing strictly
     * less work (one fewer full-buffer host-side copy per call).
     */
    @Test
    void scatterAvoidsHostSideFlattenCopyDouble() throws Throwable {
        MathExpression me = new MathExpression("x^2+y^2-2*x*y*cos(x-y)");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

        try (OpenClCompositeExpression gpu = (OpenClCompositeExpression) OpenClExpressionBridge.from(vte);
                Arena arena = Arena.ofConfined()) {

            int dataSize = 1_000_000;
            int varCount = vte.getVarCount();
            double[] flat = buildSafeSamples(me, new String[]{"x", "y"},
                    new double[][]{{-8, 8}, {-8, 8}}, dataSize, varCount);
            double[][] rows = toRowsDouble(flat, varCount, dataSize);
            MemorySegment[] segs = toSegmentsDouble(arena, rows);

            long rowBytes = (long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize();
            MemorySegment manualFlat = arena.allocate((long) varCount * rowBytes);
            MemorySegment outManual = arena.allocate((long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize());
            MemorySegment outScatter = arena.allocate((long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize());

            // warmup
            flattenInto(segs, manualFlat, rowBytes);
            gpu.applyBulk(manualFlat, outManual);
            gpu.applyBulk(segs, outScatter);

            int iterations = 5;
            long tManual = 0;
            long tScatter = 0;
            for (int i = 0; i < iterations; i++) {
                long t0 = System.nanoTime();
                flattenInto(segs, manualFlat, rowBytes);   // the copy a caller would otherwise have to do
                gpu.applyBulk(manualFlat, outManual);
                tManual += System.nanoTime() - t0;

                long t1 = System.nanoTime();
                gpu.applyBulk(segs, outScatter);
                tScatter += System.nanoTime() - t1;
            }
            tManual /= iterations;
            tScatter /= iterations;

            double[] outViaManual = new double[dataSize];
            MemorySegment.copy(outManual, ValueLayout.JAVA_DOUBLE, 0, outViaManual, 0, dataSize);
            double[] outViaScatter = new double[dataSize];
            MemorySegment.copy(outScatter, ValueLayout.JAVA_DOUBLE, 0, outViaScatter, 0, dataSize);
            assertArrayEquals(outViaManual, outViaScatter, 0.0,
                    "manual-flatten-then-dispatch and the scatter overload diverged for identical input");

            System.out.printf("manual host-flatten + applyBulk(MemorySegment,...): %,d us | "
                    + "applyBulk(MemorySegment[],...) scatter: %,d us%n", tManual / 1000, tScatter / 1000);
            assertTrue(tScatter < tManual,
                    "applyBulk(MemorySegment[], MemorySegment) eliminates the caller-side host-flatten "
                    + "copy that applyBulk(MemorySegment, MemorySegment) requires of a caller holding "
                    + "separate native per-variable buffers -- it should be strictly faster on the same "
                    + "data/hardware");
        }
    }

    /**
     * Float32 counterpart of scatterAvoidsHostSideFlattenCopyDouble.
     */
    @Test
    void scatterAvoidsHostSideFlattenCopyFloat() throws Throwable {
        MathExpression me = new MathExpression("x^2+y^2-2*x*y*cos(x-y)");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

        try (OpenClCompositeExpression gpu = (OpenClCompositeExpression) OpenClExpressionBridge.from(vte);
                Arena arena = Arena.ofConfined()) {

            int dataSize = 1_000_000;
            int varCount = vte.getVarCount();
            double[] flatD = buildSafeSamples(me, new String[]{"x", "y"},
                    new double[][]{{-8, 8}, {-8, 8}}, dataSize, varCount);
            float[] flat = new float[flatD.length];
            for (int i = 0; i < flatD.length; i++) {
                flat[i] = (float) flatD[i];
            }
            float[][] rows = toRowsFloat(flat, varCount, dataSize);
            MemorySegment[] segs = toSegmentsFloat(arena, rows);

            long rowBytes = (long) dataSize * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment manualFlat = arena.allocate((long) varCount * rowBytes);
            MemorySegment outManual = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment outScatter = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());

            // warmup
            flattenInto(segs, manualFlat, rowBytes);
            gpu.applyBulkF32(manualFlat, outManual);
            gpu.applyBulkF32(segs, outScatter);

            int iterations = 5;
            long tManual = 0;
            long tScatter = 0;
            for (int i = 0; i < iterations; i++) {
                long t0 = System.nanoTime();
                flattenInto(segs, manualFlat, rowBytes);   // the copy a caller would otherwise have to do
                gpu.applyBulkF32(manualFlat, outManual);
                tManual += System.nanoTime() - t0;

                long t1 = System.nanoTime();
                gpu.applyBulkF32(segs, outScatter);
                tScatter += System.nanoTime() - t1;
            }
            tManual /= iterations;
            tScatter /= iterations;

            float[] outViaManual = new float[dataSize];
            MemorySegment.copy(outManual, ValueLayout.JAVA_FLOAT, 0, outViaManual, 0, dataSize);
            float[] outViaScatter = new float[dataSize];
            MemorySegment.copy(outScatter, ValueLayout.JAVA_FLOAT, 0, outViaScatter, 0, dataSize);
            assertArrayEquals(outViaManual, outViaScatter, 0.0f,
                    "manual-flatten-then-dispatch and the scatter overload (f32) diverged for identical input");

            System.out.printf("manual host-flatten + applyBulkF32(MemorySegment,...): %,d us | "
                    + "applyBulkF32(MemorySegment[],...) scatter: %,d us%n", tManual / 1000, tScatter / 1000);
            assertTrue(tScatter < tManual,
                    "applyBulkF32(MemorySegment[], MemorySegment) eliminates the caller-side host-flatten "
                    + "copy that applyBulkF32(MemorySegment, MemorySegment) requires of a caller holding "
                    + "separate native per-variable buffers -- it should be strictly faster on the same "
                    + "data/hardware");
        }
    }

    /**
     * varCount == 1 is special-cased in both scatter overloads to skip
     * straight to the existing single-segment dispatch()/dispatchF32() path
     * rather than looping over a single slot -- this confirms that fast path
     * produces exactly the same output as the plain double[] overload on the
     * expression already used for the low-level plumbing tests above.
     */
    @Test
    void singleVariableScatterMatchesDoubleArrayOverload() throws Throwable {
        try (OpenClCompositeExpression expr = xPlusOne(); Arena arena = Arena.ofConfined()) {
            double[] in = {1, 2, 3, 4, 5};
            double[] outViaArray = new double[5];
            expr.applyBulk(in, outViaArray);

            MemorySegment inSeg = arena.allocate((long) in.length * ValueLayout.JAVA_DOUBLE.byteSize());
            MemorySegment.copy(in, 0, inSeg, ValueLayout.JAVA_DOUBLE, 0, in.length);
            MemorySegment outSeg = arena.allocate((long) in.length * ValueLayout.JAVA_DOUBLE.byteSize());

            expr.applyBulk(new MemorySegment[]{inSeg}, outSeg);

            double[] outViaScatter = new double[5];
            MemorySegment.copy(outSeg, ValueLayout.JAVA_DOUBLE, 0, outViaScatter, 0, 5);

            assertArrayEquals(outViaArray, outViaScatter, 0.0);
        }
    }

    /**
     * applyBulk(MemorySegment[], MemorySegment) must reject an array whose
     * length doesn't match the expression's variable count -- same contract
     * as applyBulk(double[][], double[]) -- rather than silently reading past
     * the array or under-filling the device buffer.
     */
    @Test
    void scatterRejectsWrongVariableCount() throws Throwable {
        try (OpenClCompositeExpression expr = xPlusOne(); Arena arena = Arena.ofConfined()) {
            MemorySegment[] wrongCount = {
                arena.allocate((long) 4 * ValueLayout.JAVA_DOUBLE.byteSize()),
                arena.allocate((long) 4 * ValueLayout.JAVA_DOUBLE.byteSize())
            }; // expr has varCount == 1
            MemorySegment out = arena.allocate((long) 4 * ValueLayout.JAVA_DOUBLE.byteSize());

            IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                    () -> expr.applyBulk(wrongCount, out));
            assertTrue(ex.getMessage().contains("variable segments"),
                    "exception should explain WHY the call was rejected, not just that it was");
        }
    }

    /**
     * Every element of the MemorySegment[] must be exactly dataSize elements
     * long (dataSize derived from out) -- a mismatched slot segment must be
     * rejected loudly rather than partially transferred or silently
     * misaligned against the device buffer's per-slot offsets.
     */
    @Test
    void scatterRejectsMismatchedSegmentLength() throws Throwable {
        try (OpenClCompositeExpression expr = xPlusOne(); Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate((long) 4 * ValueLayout.JAVA_DOUBLE.byteSize()); // dataSize=4
            MemorySegment[] badRow = {arena.allocate((long) 3 * ValueLayout.JAVA_DOUBLE.byteSize())}; // wrong length

            IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                    () -> expr.applyBulk(badRow, out));
            assertTrue(ex.getMessage().contains("bytes, expected"),
                    "exception should explain WHY the call was rejected, not just that it was");
        }
    }

    // =====================================================================
    // CPU (SIMD, double) vs GPU. Backend-agnostic from here down: goes
    // through GpuExpressionBridge.from(vte), which auto-selects CUDA or
    // OpenCL depending on what's actually available on the machine running
    // the tests -- see GpuExpressionBridge's javadoc for the preference
    // order and how to override it.
    // =====================================================================
    /**
     * name, expression, variable names, domain [min,max] per variable,
     * per-element DOUBLE-vs-DOUBLE tolerance (CPU double vs GPU double -- both
     * full precision, so this stays tight). Domains are chosen to avoid
     * asymptotes/negative log arguments where floating-point noise near a
     * singularity would make the test flaky rather than meaningful.
     */
    static Stream<Arguments> functionCases() {
        return Stream.of(
                Arguments.of("linear", "x+1", new String[]{"x"}, new double[][]{{-50, 50}}, 1e-9),
                Arguments.of("polynomial", "x^3-2*x^2+5", new String[]{"x"}, new double[][]{{-10, 10}}, 1e-6),
                Arguments.of("sin_cos", "3*sin(x)-cos(2*x)", new String[]{"x"}, new double[][]{{-6, 6}}, 1e-6),
                Arguments.of("exp_log", "exp(x)-log(x+10.0)", new String[]{"x"}, new double[][]{{-5, 5}}, 1e-6),
                Arguments.of("sqrt_abs", "sqrt(abs(x))+x^2", new String[]{"x"}, new double[][]{{-20, 20}}, 1e-6),
                Arguments.of("hyperbolic", "sinh(x)-tanh(x)*cosh(x)", new String[]{"x"}, new double[][]{{-3, 3}}, 1e-6),
                Arguments.of("degree_trig", "sin(x)", new String[]{"x"}, new double[][]{{-720, 720}}, 1e-6),
                // The original composite expression from the design discussion.
                // tan(x) asymptotes are dodged by the sampler below (see
                // buildSafeSamples), not by the domain bounds themselves.
                Arguments.of("composite", "3*cos(x-2)+ln(3*x^3-5*x-4*tan(x))",
                        new String[]{"x"}, new double[][]{{-4, 4}}, 1e-4)
        );
    }

    /**
     * CPU SIMD (double) vs GPU (double) -- correctness AND timing. This is the
     * core "does the GPU agree with the CPU" check: same compiled bytecode,
     * same input, both full double precision.
     */
    @ParameterizedTest(name = "[CPU vs GPU, double] {0}: {1}")
    @MethodSource("functionCases")
    void cpuVsGpuDoubleParityAndTiming(String caseName, String expression, String[] varNames,
            double[][] domains, double tolerance) throws Throwable {

        MathExpression me = new MathExpression(expression);
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);
        SIMDCompositeExpression cpu = vte.compile();

        try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte)) {
            int dataSize = 200_000;
            double[] flat = buildSafeSamples(me, varNames, domains, dataSize, vte.getVarCount());

            double[] cpuOut = new double[dataSize];
            double[] gpuOut = new double[dataSize];

            long cpuNanos = System.nanoTime();
            cpu.applyBulk(flat, cpuOut);
            cpuNanos = System.nanoTime() - cpuNanos;

            long gpuNanos = System.nanoTime();
            gpu.applyBulk(flat, gpuOut);
            gpuNanos = System.nanoTime() - gpuNanos;

            assertParity(caseName + " (double)", cpuOut, gpuOut, tolerance);
            System.out.printf("[%s, double, %,d pts] CPU %,d us | GPU %,d us%n",
                    caseName, dataSize, cpuNanos / 1000, gpuNanos / 1000);
        }
    }

    /**
     * CPU SIMD (double, ground truth) vs GPU (NATIVE float32) -- correctness
     * AND timing. There is no CPU float32 path (VectorTurboEvaluator's bytecode
     * is double-only end to end), so the CPU side here is the same double
     * computation as above; what's different is the GPU side now runs through
     * interpretF32's native float stack, not a double-cast bridge. Tolerances
     * are intentionally far looser than the double-vs- double case above:
     * float32 carries roughly 7 significant decimal digits versus double's
     * ~15-16, and every GPU-side operation (not just the final result) is
     * computed at that precision, so error can compound across a multi-op
     * expression. 1e-4 relative was chosen as "loose enough to not flake on
     * legitimate float32 rounding, tight enough to still catch a genuinely
     * wrong formula" -- it is NOT the same bar as the double comparisons above
     * and shouldn't be tightened to match them.
     */
    @ParameterizedTest(name = "[CPU double vs GPU float32] {0}: {1}")
    @MethodSource("functionCases")
    void cpuVsGpuFloat32ParityAndTiming(String caseName, String expression, String[] varNames,
            double[][] domains, double ignoredDoubleTolerance) throws Throwable {

        MathExpression me = new MathExpression(expression);
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);
        SIMDCompositeExpression cpu = vte.compile();

        try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte)) {
            int dataSize = 200_000;
            double[] flatD = buildSafeSamples(me, varNames, domains, dataSize, vte.getVarCount());
            float[] flatF = new float[flatD.length];
            for (int i = 0; i < flatD.length; i++) {
                flatF[i] = (float) flatD[i];
            }

            double[] cpuOut = new double[dataSize];
            float[] gpuOutF = new float[dataSize];

            long cpuNanos = System.nanoTime();
            cpu.applyBulk(flatD, cpuOut);
            cpuNanos = System.nanoTime() - cpuNanos;

            long gpuNanos = System.nanoTime();
            gpu.applyBulk(flatF, gpuOutF);
            gpuNanos = System.nanoTime() - gpuNanos;

            double[] gpuOut = new double[dataSize];
            for (int i = 0; i < dataSize; i++) {
                gpuOut[i] = gpuOutF[i];
            }

            assertParity(caseName + " (float32)", cpuOut, gpuOut, 1e-4);
            System.out.printf("[%s, float32, %,d pts] CPU(double) %,d us | GPU(float32) %,d us%n",
                    caseName, dataSize, cpuNanos / 1000, gpuNanos / 1000);
        }
    }

    /**
     * Directly answers "what's the point of float32 if it computes at the same
     * rate as double" -- runs the SAME expression, at the SAME dataSize,
     * through the GPU's double path and its float32 path back to back, and
     * prints both timings plus the ratio. No hard assertion on the ratio
     * itself: relative float32/double throughput is entirely a function of the
     * specific GPU's hardware (consumer GPUs often cap fp64 at 1/32 or worse of
     * fp32 rate; datacenter parts vary widely), so asserting a specific speedup
     * here would be asserting a fact about the test machine, not about the
     * code. This is a human-readable check, not a correctness gate.
     */
    @Test
    void gpuDoubleVsFloatThroughputComparison() throws Throwable {
        MathExpression me = new MathExpression("3*sin(x)*cos(y)+sqrt(abs(x*y))");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

        try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte)) {
            int side = 2000;
            int dataSize = side * side;

            double[] flatD = buildSafeSamples(me, new String[]{"x", "y"},
                    new double[][]{{-8, 8}, {-8, 8}}, dataSize, vte.getVarCount());
            float[] flatF = new float[flatD.length];
            for (int i = 0; i < flatD.length; i++) {
                flatF[i] = (float) flatD[i];
            }

            double[] outD = new double[dataSize];
            long tDouble = System.nanoTime();
            gpu.applyBulk(flatD, outD);
            tDouble = System.nanoTime() - tDouble;

            float[] outF = new float[dataSize];
            long tFloat = System.nanoTime();
            gpu.applyBulk(flatF, outF);
            tFloat = System.nanoTime() - tFloat;

            System.out.printf("[%dx%d grid, %,d pts] GPU double: %,d us | GPU float32: %,d us | ratio double/float: %.2fx%n",
                    side, side, dataSize, tDouble / 1000, tFloat / 1000,
                    tFloat == 0 ? Double.NaN : (double) tDouble / (double) tFloat);
        }
    }

    @Test
    void multiVariableExpressionParity() throws Throwable {
        MathExpression me = new MathExpression("x^2+y^2-2*x*y*cos(x-y)");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);
        SIMDCompositeExpression cpu = vte.compile();

        try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte)) {
            int dataSize = 250_000; // e.g. a 500x500 grid, flattened
            double[] flat = buildSafeSamples(me, new String[]{"x", "y"},
                    new double[][]{{-8, 8}, {-8, 8}}, dataSize, vte.getVarCount());

            double[] cpuOut = new double[dataSize];
            long cpuNanos = System.nanoTime();
            cpu.applyBulk(flat, cpuOut);
            cpuNanos = System.nanoTime() - cpuNanos;
            System.out.println("cpu-time: " + (cpuNanos / 1000) + " us");

            double[] gpuOut = new double[dataSize];
            long gpuNanos = System.nanoTime();
            gpu.applyBulk(flat, gpuOut);
            gpuNanos = System.nanoTime() - gpuNanos;
            System.out.println("gpu-time: " + (gpuNanos / 1000) + " us");

            assertParity("x^2+y^2-2xy*cos(x-y)", cpuOut, gpuOut, 1e-6);
        }
    }

    @Test
    void multiVariableExpressionParity1() throws Throwable {
        MathExpression me = new MathExpression("x^2+y^2-2*x*y*cos(x-y)");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);
        SIMDCompositeExpression cpu = vte.compile();

        try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte)) {
            int dataSize = 250_000; // e.g. a 500x500 grid, flattened
            double[] flat = buildSafeSamples(me, new String[]{"x", "y"},
                    new double[][]{{-8, 8}, {-8, 8}}, dataSize, vte.getVarCount());

            double[] gpuOut = new double[dataSize];
            long gpuNanos = System.nanoTime();
            gpu.applyBulk(flat, gpuOut);
            gpuNanos = System.nanoTime() - gpuNanos;
            System.out.println("gpu-time: " + (gpuNanos / 1000) + " us");

        }
    }

    /**
     * Backend-agnostic correctness check for the scatter overload: same
     * expression/domain as multiVariableExpressionParity, but supplies input
     * as one native MemorySegment per variable via
     * applyBulk(MemorySegment[], MemorySegment) rather than a single flat
     * double[], and checks against the CPU SIMD reference -- exercises
     * whichever backend (CUDA or OpenCL) GpuExpressionBridge actually
     * resolves on the machine running the tests, not just OpenCL
     * specifically.
     */
    @Test
    void multiVariableScatterParityBackendAgnostic() throws Throwable {
        MathExpression me = new MathExpression("x^2+y^2-2*x*y*cos(x-y)");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);
        SIMDCompositeExpression cpu = vte.compile();

        try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte); Arena arena = Arena.ofConfined()) {
            int dataSize = 250_000; // e.g. a 500x500 grid, flattened
            int varCount = vte.getVarCount();
            double[] flat = buildSafeSamples(me, new String[]{"x", "y"},
                    new double[][]{{-8, 8}, {-8, 8}}, dataSize, varCount);

            double[] cpuOut = new double[dataSize];
            cpu.applyBulk(flat, cpuOut);

            double[][] rows = toRowsDouble(flat, varCount, dataSize);
            MemorySegment[] segs = toSegmentsDouble(arena, rows);
            MemorySegment outSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize());

            gpu.applyBulk(segs, outSeg);

            double[] gpuOut = new double[dataSize];
            MemorySegment.copy(outSeg, ValueLayout.JAVA_DOUBLE, 0, gpuOut, 0, dataSize);

            assertParity("x^2+y^2-2xy*cos(x-y) (scatter, backend-agnostic)", cpuOut, gpuOut, 1e-6);
        }
    }

    /**
     * Informational, not a correctness gate: confirms the GPU path is actually
     * faster once dataSize is large enough to amortize dispatch overhead. No
     * hard assertion on speedup ratio -- CI hardware varies too much for that
     * to be anything but flaky -- this just prints both timings for a human to
     * read. Sized at a comfortable few million points rather than the 100M+ an
     * earlier version of this test used, which would allocate hundreds of MB to
     * several GB of host arrays alone.
     */
    @Test
    void largeGridInformationalTiming() throws Throwable {
        MathExpression me = new MathExpression("3*sin(x)*cos(y)+sqrt(abs(x*y))");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);
        SIMDCompositeExpression cpu = vte.compile();

        int side = 2000;
        int dataSize = side * side;
        double[] flat = buildSafeSamples(me, new String[]{"x", "y"},
                new double[][]{{-8, 8}, {-8, 8}}, dataSize, vte.getVarCount());

        double[] cpuOut = new double[dataSize];
        long cpuStart = System.nanoTime();
        cpu.applyBulkParallel(flat, cpuOut);
        long cpuNanos = System.nanoTime() - cpuStart;

        try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte)) {
            double[] gpuOut = new double[dataSize];
            long gpuStart = System.nanoTime();
            gpu.applyBulk(flat, gpuOut);
            long gpuNanos = System.nanoTime() - gpuStart;

            assertParity(side + "x" + side + " grid", cpuOut, gpuOut, 1e-5);

            System.out.printf("%dx%d grid (%,d points): CPU %,d us | GPU %,d us%n",
                    side, side, dataSize, cpuNanos / 1000, gpuNanos / 1000);

            // Calculate Memory Bandwidth Utilization
            long totalBytesRead = (long) dataSize * vte.getVarCount() * Double.BYTES;
            long totalBytesWritten = (long) dataSize * Double.BYTES;
            long totalBytesMoved = totalBytesRead + totalBytesWritten;

            double gpuSeconds = gpuNanos / 1_000_000_000.0;
            double gigabytesMoved = totalBytesMoved / (1024.0 * 1024.0 * 1024.0);
            double effectiveGbps = gigabytesMoved / gpuSeconds;

            System.out.printf("Effective GPU Memory Bandwidth: %.2f GB/s%n", effectiveGbps);
        }
    }

    @Test
    void largeGridInformationalTimingWithMemorySegments() throws Throwable {
        MathExpression me = new MathExpression("3*sin(x)*cos(y)+sqrt(abs(x*y))");
        SIMDCommandF64.SIMDVectorCompositeExpression cpu = SIMDCommandF64.getEvaluator(me);
        VectorTurboEvaluator vte = (VectorTurboEvaluator) cpu.getCompiler();
 
        OpenClCompositeExpression.selectDevice(OpenClCompositeExpression.GpuVendor.INTEL);
        //System.setProperty("opencl.gpu.vendor", "Advanced Micro Devices, Inc.");
        //OpenClCompositeExpression.selectDevice("Advanced Micro Devices, Inc."); 
        //System.setProperty("opencl.gpu.vendor", "Intel");
        //OpenClCompositeExpression.selectDevice("Intel"); 
        int side = 2000;
        int dataSize = side * side;

        // Build samples on heap, then transfer once to off-heap before timing
        double[] heapSamples = buildSafeSamples(me, new String[]{"x", "y"},
                new double[][]{{-8, 8}, {-8, 8}}, dataSize, vte.getVarCount());

        // NOTE: the CPU reference below deliberately goes through
        // applyBulkParallel(double[], double[]) -- already exercised and
        // trusted elsewhere in this file -- rather than
        // applyBulkParallel(MemorySegment, MemorySegment). That CPU
        // MemorySegment overload's implementation isn't visible from this
        // file (it lives in BatchedVectorCompositeExpression), and a
        // reported failure here showed the CPU side reading back as ~0.0
        // at nearly every point while the GPU computed correctly -- i.e.
        // the CPU MemorySegment path appears not to be writing results,
        // not a numerical disagreement. This test exists to validate the
        // GPU's MemorySegment path specifically, so it shouldn't be
        // blocked on -- or silently mask -- a separate, unconfirmed CPU-side
        // issue. If that CPU overload needs to be tested directly, that
        // belongs in its own dedicated test once its implementation can be
        // inspected.
        try (Arena arena = Arena.ofShared()) {
            MemorySegment flatSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, heapSamples.length);
            MemorySegment.copy(heapSamples, 0, flatSegment, ValueLayout.JAVA_DOUBLE, 0, heapSamples.length);

            MemorySegment gpuOutSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, dataSize);

            double[] cpuOut = new double[dataSize];
            long cpuStart = System.nanoTime();
            cpu.applyBulkParallel(heapSamples, cpuOut);
            //cpu.applyBulkParallel(flatSegment, gpuOutSegment);
            long cpuNanos = System.nanoTime() - cpuStart;

            try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte, GpuBackend.OPENCL)) {
                long gpuStart = System.nanoTime();
                gpu.applyBulk(flatSegment, gpuOutSegment);
                long gpuNanos = System.nanoTime() - gpuStart;

                double[] gpuOut = gpuOutSegment.toArray(ValueLayout.JAVA_DOUBLE);
                assertParity(side + "x" + side + " grid (MemorySegment)", cpuOut, gpuOut, 1e-5);

                System.out.printf("%dx%d grid (%,d points, MemorySegment): CPU %,d us | GPU %,d us%n",
                        side, side, dataSize, cpuNanos / 1000, gpuNanos / 1000);

                // Calculate Memory Bandwidth Utilization
                long totalBytesRead = (long) dataSize * vte.getVarCount() * Double.BYTES;
                long totalBytesWritten = (long) dataSize * Double.BYTES;
                long totalBytesMoved = totalBytesRead + totalBytesWritten;

                double gpuSeconds = gpuNanos / 1_000_000_000.0;
                double gigabytesMoved = totalBytesMoved / (1024.0 * 1024.0 * 1024.0);
                double effectiveGbps = gigabytesMoved / gpuSeconds;

                System.out.printf("Effective GPU Memory Bandwidth: %.2f GB/s%n", effectiveGbps);
            }
        }
    }

    // ================= Helpers =================
    /**
     * Builds a flat, column-major sample buffer (in[slot*dataSize+idx],
     * matching VectorTurboEvaluator's own flatVariables layout) across
     * `dataSize` points, linearly sampling each named variable's domain. Points
     * where any tan()-bearing term would land within a small window of an
     * asymptote are nudged away, since a few-ULP difference in the argument
     * there produces enormous, meaningless output deltas on both backends and
     * would make an otherwise-correct kernel look broken.
     */
    private static double[] buildSafeSamples(MathExpression me, String[] varNames, double[][] domains,
            int dataSize, int varCount) {
        double[] flat = new double[varCount * dataSize];
        int[] slots = new int[varNames.length];
        for (int v = 0; v < varNames.length; v++) {
            slots[v] = me.getRegistry().getSlot(varNames[v]);
        }

        for (int i = 0; i < dataSize; i++) {
            double t = dataSize == 1 ? 0.5 : (double) i / (dataSize - 1);
            for (int v = 0; v < varNames.length; v++) {
                double lo = domains[v][0];
                double hi = domains[v][1];
                double x = lo + t * (hi - lo);
                x = nudgeAwayFromTanAsymptote(x);
                if (slots[v] != -1) {
                    flat[slots[v] * dataSize + i] = x;
                }
            }
        }
        return flat;
    }

    private static double nudgeAwayFromTanAsymptote(double x) {
        double nearestHalfPi = Math.round(x / (Math.PI / 2)) * (Math.PI / 2);
        boolean nearOddMultiple = Math.round(x / (Math.PI / 2)) % 2 != 0;
        if (nearOddMultiple && Math.abs(x - nearestHalfPi) < 0.02) {
            return nearestHalfPi + (x >= nearestHalfPi ? 0.02 : -0.02);
        }
        return x;
    }

    /**
     * Reshapes a flat, column-major sample buffer (as produced by
     * buildSafeSamples -- varCount blocks of dataSize contiguous elements)
     * into the varCount-rows-of-dataSize-elements shape applyBulk(double[][], double[])
     * and the scatter overloads both expect. Since buildSafeSamples already
     * lays its output out as exactly that concatenation, this is a plain
     * slice per row, not a transpose.
     */
    private static double[][] toRowsDouble(double[] flat, int varCount, int dataSize) {
        double[][] rows = new double[varCount][dataSize];
        for (int slot = 0; slot < varCount; slot++) {
            System.arraycopy(flat, slot * dataSize, rows[slot], 0, dataSize);
        }
        return rows;
    }

    /** Float32 counterpart of toRowsDouble. */
    private static float[][] toRowsFloat(float[] flat, int varCount, int dataSize) {
        float[][] rows = new float[varCount][dataSize];
        for (int slot = 0; slot < varCount; slot++) {
            System.arraycopy(flat, slot * dataSize, rows[slot], 0, dataSize);
        }
        return rows;
    }

    /**
     * Copies each on-heap row into its own freshly-allocated native
     * MemorySegment -- this one-time per-row heap-to-native copy is exactly
     * what the resulting MemorySegment[] lets applyBulk(MemorySegment[], MemorySegment)
     * skip on every subsequent call, since from here on the data is already
     * native.
     */
    private static MemorySegment[] toSegmentsDouble(Arena arena, double[][] rows) {
        MemorySegment[] segs = new MemorySegment[rows.length];
        for (int i = 0; i < rows.length; i++) {
            MemorySegment seg = arena.allocate((long) rows[i].length * ValueLayout.JAVA_DOUBLE.byteSize());
            MemorySegment.copy(rows[i], 0, seg, ValueLayout.JAVA_DOUBLE, 0, rows[i].length);
            segs[i] = seg;
        }
        return segs;
    }

    /** Float32 counterpart of toSegmentsDouble. */
    private static MemorySegment[] toSegmentsFloat(Arena arena, float[][] rows) {
        MemorySegment[] segs = new MemorySegment[rows.length];
        for (int i = 0; i < rows.length; i++) {
            MemorySegment seg = arena.allocate((long) rows[i].length * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(rows[i], 0, seg, ValueLayout.JAVA_FLOAT, 0, rows[i].length);
            segs[i] = seg;
        }
        return segs;
    }

    /**
     * Simulates what a caller holding separate native per-variable segments
     * previously had to do by hand before calling
     * applyBulk(MemorySegment, MemorySegment): copy each row segment into its
     * slice of one contiguous destination segment. This is exactly the
     * host-side copy applyBulk(MemorySegment[], MemorySegment) eliminates --
     * see scatterAvoidsHostSideFlattenCopyDouble/Float.
     */
    private static void flattenInto(MemorySegment[] rows, MemorySegment dest, long rowBytes) {
        for (int i = 0; i < rows.length; i++) {
            MemorySegment.copy(rows[i], 0, dest, (long) i * rowBytes, rowBytes);
        }
    }

    /**
     * NaN is treated as agreement in both backends (e.g. ln() of a negative
     * argument near a sign-flip boundary) since forcing bit-identical NaN
     * payloads across two different math libraries is not a meaningful bar.
     * Everything else uses a combined absolute+relative tolerance, since
     * OpenCL/CUDA's built-in transcendental functions are only required by spec
     * to be correctly rounded within a few ULP, not bit-identical to Java's
     * Math implementation -- exact equality here would be testing the wrong
     * thing. Callers doing cross-precision comparisons (double CPU vs float32
     * GPU) should pass a correspondingly looser tolerance -- this helper
     * doesn't know or care which precision produced either array.
     */
    private static void assertParity(String label, double[] cpu, double[] gpu, double tolerance) {
        int mismatches = 0;
        int firstBadIndex = -1;
        double worstDelta = 0;

        for (int i = 0; i < cpu.length; i++) {
            double a = cpu[i];
            double b = gpu[i];
            if (Double.isNaN(a) && Double.isNaN(b)) {
                continue;
            }
            double delta = Math.abs(a - b);
            double allowed = tolerance * Math.max(1.0, Math.abs(a));
            if (delta > allowed) {
                mismatches++;
                if (firstBadIndex == -1) {
                    firstBadIndex = i;
                }
                worstDelta = Math.max(worstDelta, delta);
            }
        }

        if (mismatches > 0) {
            fail(String.format(
                    "%s: %d/%d points diverged beyond tolerance %.2e "
                    + "(first at index %d: expected=%s actual=%s, worst delta=%.3e)",
                    label, mismatches, cpu.length, tolerance, firstBadIndex,
                    cpu[firstBadIndex], gpu[firstBadIndex], worstDelta));
        }
    }
}