package com.github.gbenroscience.gpu;

import java.lang.foreign.MemorySegment;

/**
 * Backend-agnostic contract for a single compiled expression running on a
 * GPU. Implemented by both
 * com.github.gbenroscience.gpu.opencl.OpenClCompositeExpression and
 * com.github.gbenroscience.gpu.cuda.CudaCompositeExpression -- same apply
 * surface either way, so code written against this interface doesn't need
 * to know or care which backend actually ran the expression.
 *
 * The overloads mirror the CPU-side BatchedVectorCompositeExpression apply
 * surface this whole GPU effort was built to parallel, plus one that has
 * no CPU counterpart (applyBulkF32 -- see below):
 *   - applyBulk(MemorySegment, MemorySegment): the primary DOUBLE path,
 *     zero host-array copy (see the implementing classes' javadocs for the
 *     unified-memory caveat).
 *   - applyBulkF32(MemorySegment, MemorySegment): the FLOAT32 counterpart.
 *     This is a DISTINCT method, not an overload distinguished by argument
 *     type, because a bare MemorySegment carries no element-type
 *     information at the Java level -- a segment of N*4 bytes is
 *     indistinguishable from "N floats" versus "N/2 doubles" by inspecting
 *     it alone. A single applyBulk(MemorySegment, MemorySegment) cannot
 *     safely serve both precisions: it has to pick one by convention, and
 *     the convention here is double (matching every other MemorySegment
 *     path in this codebase). Callers holding a float-filled segment MUST
 *     call applyBulkF32 explicitly -- calling applyBulk on it will
 *     mis-dispatch through the double kernel against a buffer sized/laid
 *     out for floats, silently reinterpreting pairs of floats as double
 *     bit patterns rather than failing loudly.
 *   - double[] / float[] overloads: convenience for callers already
 *     holding on-heap arrays; copy into a persistent, grow-only staging
 *     buffer internally. These ARE safely overloadable by argument type,
 *     since float[] and double[] are distinct types the compiler
 *     disambiguates at the call site -- unlike the two MemorySegment
 *     methods above.
 *   - double[][] / float[][] overloads: one row per variable slot,
 *     flattened internally into the same column-major layout the kernels
 *     expect.
 *
 * Extends AutoCloseable because every implementation owns off-heap/device
 * resources (device buffers, a staging Arena) that must be released
 * deterministically -- always use try-with-resources.
 */
public interface GpuCompositeExpression extends AutoCloseable {

    /** Double-precision MemorySegment path. See class javadoc: assumes the segment holds doubles. */
    void applyBulk(MemorySegment in, MemorySegment out) throws Throwable;

    /** Float32-precision MemorySegment path. See class javadoc for why this can't just be an applyBulk overload. */
    void applyBulkF32(MemorySegment in, MemorySegment out) throws Throwable;

    void applyBulk(double[] in, double[] out) throws Throwable;

    void applyBulk(double[][] in, double[] out) throws Throwable;

    void applyBulk(float[] in, float[] out) throws Throwable;

    void applyBulk(float[][] in, float[] out) throws Throwable;

    /**
     * Narrows AutoCloseable.close()'s `throws Exception` to no checked
     * exception -- both implementations do best-effort cleanup internally
     * and never throw from close(), so callers shouldn't have to handle
     * one just because the interface theoretically allows it.
     */
    @Override
    void close();
}