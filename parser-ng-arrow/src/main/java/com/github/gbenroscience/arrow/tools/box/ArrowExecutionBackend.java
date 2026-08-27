package com.github.gbenroscience.arrow.tools.box;

/**
 * Which execution engine an {@link ArrowExpressionEvaluator} runs on. Pass
 * one of these to {@link ArrowExpressionEvaluators#compile} to choose a
 * backend without touching the call site again if you later switch it.
 *
 * <h2>{@code GPU_AUTO} vs pinning a backend</h2>
 * {@code GPU_AUTO} defers backend selection to {@code GpuExpressionBridge},
 * which prefers CUDA and falls back to OpenCL (see that class's javadoc for
 * the exact preference order and its system-property override). Compiling
 * with {@code GPU_AUTO} on a machine with neither backend available throws;
 * it does not fall back to {@link #CPU_SIMD} — for that, use
 * {@link ArrowExpressionEvaluators#compilePreferGpu} instead, which tries
 * GPU_AUTO first and only falls back to {@code CPU_SIMD} if no GPU backend
 * bootstraps at all.
 *
 * <p>{@link #GPU_CUDA} and {@link #GPU_OPENCL} pin a specific backend:
 * compilation fails loudly if that exact backend cannot be bootstrapped on
 * this machine, rather than silently trying the other one. Pick a pinned
 * backend when you need reproducible-across-machines behavior (e.g. a fleet
 * where you know exactly which accelerator is installed) or want a hard
 * failure signal instead of a silent substitution; pick {@code GPU_AUTO} when
 * you just want "the best GPU available, whichever it is".
 */
public enum ArrowExecutionBackend {

    /** {@link ArrowBulkEvaluator} — CPU, SIMD-vectorized. Always available. */
    CPU_SIMD,

    /**
     * {@link ArrowGpuBulkEvaluator}, auto-selecting CUDA or OpenCL — whichever
     * {@code GpuExpressionBridge} finds bootstrapable first. Use
     * {@link ArrowGpuBulkEvaluator#actualBackend()} after compiling to find
     * out which one was actually picked.
     */
    GPU_AUTO,

    /** {@link ArrowGpuBulkEvaluator} pinned to the CUDA backend. */
    GPU_CUDA,

    /** {@link ArrowGpuBulkEvaluator} pinned to the OpenCL backend. */
    GPU_OPENCL
}