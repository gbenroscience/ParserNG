package com.github.gbenroscience.gpu.evaluator;

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
 *     expect. Because a double[]/float[] row lives on the Java heap, this
 *     flattening necessarily crosses the heap/native boundary once per
 *     row -- there's no way around that copy for on-heap callers.
 *   - MemorySegment[] / MemorySegment overloads (applyBulk(MemorySegment[], MemorySegment)
 *     and applyBulkF32(MemorySegment[], MemorySegment)): the TRUE zero-copy
 *     entry point for multi-variable input. Each element of the array is
 *     one variable slot's data, already resident in native memory (e.g.
 *     produced by a prior native computation stage, or a pinned/mapped
 *     buffer) -- unlike the double[][]/float[][] overloads, there is no
 *     Java-heap staging step here at all: each slot's segment is
 *     transferred (scattered) straight into its slice of the device input
 *     buffer, one native-to-device transfer per variable, with no
 *     intermediate host-side flatten/staging buffer. This is strictly
 *     less copying than applyBulk(MemorySegment, MemorySegment) requires
 *     of a caller who currently has to flatten N native buffers into one
 *     contiguous MemorySegment themselves before calling it -- that
 *     caller-side flatten is exactly the copy these two methods eliminate.
 *     Every element must be exactly dataSize elements long (dataSize is
 *     derived from out's size), and the array length must equal the
 *     expression's variable count, same contract as double[][]/float[][].
 *
 * Extends AutoCloseable because every implementation owns off-heap/device
 * resources (device buffers, a staging Arena) that must be released
 * deterministically -- always use try-with-resources.
 */
public interface GpuCompositeExpression extends AutoCloseable {

    /** Double-precision MemorySegment path. See class javadoc: assumes the segment holds doubles.
     * @param in
     * @param out
     * @throws java.lang.Throwable */
    void applyBulk(MemorySegment in, MemorySegment out) throws Throwable;

    /** Float32-precision MemorySegment path. See class javadoc for why this can't just be an applyBulk overload.
     * @param in
     * @param out
     * @throws java.lang.Throwable */
    void applyBulkF32(MemorySegment in, MemorySegment out) throws Throwable;

    /**
     * True zero-copy, multi-variable double path. {@code in[slot]} is the
     * native-memory buffer for variable slot {@code slot}, exactly
     * {@code dataSize} doubles long (dataSize derived from out); {@code
     * in.length} must equal this expression's variable count. Each slot is
     * transferred directly from its own segment into the device input
     * buffer -- no host-side flatten/staging copy, unlike the
     * double[][] overload below.
     * @param in
     * @param out
     * @throws java.lang.Throwable */
    void applyBulk(MemorySegment[] in, MemorySegment out) throws Throwable;

    /**
     * True zero-copy, multi-variable float32 path. Same contract as
     * {@link #applyBulk(MemorySegment[], MemorySegment)}, just float32
     * throughout -- see that method and the class javadoc.
     * @param in
     * @param out
     * @throws java.lang.Throwable */
    void applyBulkF32(MemorySegment[] in, MemorySegment out) throws Throwable;

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