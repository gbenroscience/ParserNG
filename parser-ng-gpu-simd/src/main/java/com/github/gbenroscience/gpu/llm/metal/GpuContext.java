package com.github.gbenroscience.gpu.llm.metal;

/**
 * Metal counterpart of {@code com.github.gbenroscience.gpu.llm.cuda.GpuContext}
 * / {@code com.github.gbenroscience.gpu.llm.opencl.GpuContext}. Bootstrap
 * sequence mirrors both: resolve a device, compile the whole kernel
 * source once, resolve every kernel entry point up front and hold it for
 * this context's lifetime.
 *
 * TRANSLATION NOTES vs the CUDA version:
 *   - {@code CUcontext}/{@code cuCtxSetCurrent} has no Metal analogue --
 *     Metal objects (device, queue, pipeline states) are not
 *     thread-affine or "current-context" scoped the way a CUDA driver
 *     context is; every {@code id<MTLCommandQueue>} can be used from any
 *     thread. {@link LlamaLayer}'s dispatch methods consequently do NOT
 *     need the CUDA port's {@code cuCtxSetCurrent} calls, though they
 *     keep an equivalent {@link #dispatchLock} for the same reason the
 *     CUDA port does: serializing encoder-build + commit against this
 *     context's single command queue keeps per-layer dispatch ordering
 *     predictable without needing a queue-per-thread scheme.
 *   - {@code CUmodule}/{@code cuModuleLoadData} (load precompiled PTX)
 *     becomes {@code id<MTLLibrary>} compiled directly from MSL SOURCE at
 *     construction time via {@link MetalBindings#compileLibrary} -- Metal
 *     has no separate "compile ahead of time to an IR, then load the IR"
 *     step exposed at this level the way NVRTC-then-cuModuleLoadData
 *     does; {@code newLibraryWithSource:options:error:} does both.
 *   - Each {@code CUfunction} becomes a resolved
 *     {@code id<MTLComputePipelineState>} (function lookup AND
 *     device-specific pipeline finalization done together, unlike CUDA
 *     where {@code cuModuleGetFunction} is comparatively cheap and any
 *     "finalize for this device" work already happened at PTX-JIT time
 *     inside the driver) -- held as a {@code long} the same way the CUDA
 *     port holds {@code MemorySegment} CUfunction handles.
 *
 * UNVERIFIED, same standing caveat as every file in this codebase: no
 * Metal GPU/toolchain was available while writing this.
 */
public final class GpuContext implements AutoCloseable {

    public final MetalBindings mtl;

    public final long device;          // id<MTLDevice>
    public final String selectedDeviceDescription;
    public final long commandQueue;    // id<MTLCommandQueue>
    public final long library;         // id<MTLLibrary>

    // ---- decode-path kernels ----
    public final long kQuantizeI8;
    public final long kQuantizeActivationQ8_0;
    public final long kQ8_0GemvSplit;
    public final long kQ8_0GemvPlain;
    public final long kRopeApplySplit;
    public final long kRmsnormPartialSumsq;
    public final long kRmsnormApply;
    public final long kAttnScores;
    public final long kSoftmaxInplace;
    public final long kAttnWeightedSum;
    public final long kSwigluActivate;
    public final long kResidualAdd;
    public final long kF32Gemv;

    // ---- FFN activation alternatives ----
    public final long kGeluActivate;
    public final long kGegluActivate;

    // ---- batched prefill kernels ----
    public final long kQ8_0GemmTiled;
    public final long kF32GemmTiled;
    public final long kRmsnormPartialSumsqRows;
    public final long kRmsnormApplyRows;
    public final long kRopeApplyPairwiseRows;
    public final long kAttnScoresCausalBatched;
    public final long kSoftmaxInplaceRows;
    public final long kAttnWeightedSumCausalBatched;

    /** Serializes encoder-build + commit against the shared command queue -- see class javadoc's "no cuCtxSetCurrent needed, but still one lock" note. */
    public final Object dispatchLock = new Object();

    public GpuContext() {
        this.mtl = new MetalBindings();

        try {
            MetalDeviceSelector.SelectedDevice chosen = MetalDeviceSelector.resolve();
            this.device = chosen.deviceId();
            this.selectedDeviceDescription = chosen.describe();

            this.commandQueue = mtl.newCommandQueue(device);
            if (commandQueue == 0L) {
                throw new IllegalStateException("newCommandQueue returned nil for device " + chosen.describe());
            }

            this.library = mtl.compileLibrary(device, MetalKernelSource.METAL_SOURCE);

            this.kQuantizeI8 = pipeline(MetalKernelSource.KERNEL_QUANTIZE_I8);
            this.kQuantizeActivationQ8_0 = pipeline(MetalKernelSource.KERNEL_QUANTIZE_ACTIVATION_Q8_0);
            this.kQ8_0GemvSplit = pipeline(MetalKernelSource.KERNEL_Q8_0_GEMV_SPLIT);
            this.kQ8_0GemvPlain = pipeline(MetalKernelSource.KERNEL_Q8_0_GEMV_PLAIN);
            this.kRopeApplySplit = pipeline(MetalKernelSource.KERNEL_ROPE_APPLY_SPLIT);
            this.kRmsnormPartialSumsq = pipeline(MetalKernelSource.KERNEL_RMSNORM_PARTIAL_SUMSQ);
            this.kRmsnormApply = pipeline(MetalKernelSource.KERNEL_RMSNORM_APPLY);
            this.kAttnScores = pipeline(MetalKernelSource.KERNEL_ATTN_SCORES);
            this.kSoftmaxInplace = pipeline(MetalKernelSource.KERNEL_SOFTMAX_INPLACE);
            this.kAttnWeightedSum = pipeline(MetalKernelSource.KERNEL_ATTN_WEIGHTED_SUM);
            this.kSwigluActivate = pipeline(MetalKernelSource.KERNEL_SWIGLU_ACTIVATE);
            this.kResidualAdd = pipeline(MetalKernelSource.KERNEL_RESIDUAL_ADD);
            this.kF32Gemv = pipeline(MetalKernelSource.KERNEL_F32_GEMV);

            this.kGeluActivate = pipeline(MetalKernelSource.KERNEL_GELU_ACTIVATE);
            this.kGegluActivate = pipeline(MetalKernelSource.KERNEL_GEGLU_ACTIVATE);

            this.kQ8_0GemmTiled = pipeline(MetalKernelSource.KERNEL_Q8_0_GEMM_TILED);
            this.kF32GemmTiled = pipeline(MetalKernelSource.KERNEL_F32_GEMM_TILED);
            this.kRmsnormPartialSumsqRows = pipeline(MetalKernelSource.KERNEL_RMSNORM_PARTIAL_SUMSQ_ROWS);
            this.kRmsnormApplyRows = pipeline(MetalKernelSource.KERNEL_RMSNORM_APPLY_ROWS);
            this.kRopeApplyPairwiseRows = pipeline(MetalKernelSource.KERNEL_ROPE_APPLY_PAIRWISE_ROWS);
            this.kAttnScoresCausalBatched = pipeline(MetalKernelSource.KERNEL_ATTN_SCORES_CAUSAL_BATCHED);
            this.kSoftmaxInplaceRows = pipeline(MetalKernelSource.KERNEL_SOFTMAX_INPLACE_ROWS);
            this.kAttnWeightedSumCausalBatched = pipeline(MetalKernelSource.KERNEL_ATTN_WEIGHTED_SUM_CAUSAL_BATCHED);

            System.err.println("[ParserNG LLM-Metal] " + chosen.describe());

        } catch (Throwable t) {
            throw new RuntimeException("Failed to bootstrap GpuContext (Metal)", t);
        }
    }

    private long pipeline(String functionName) {
        long fn = mtl.newFunction(library, functionName);
        if (fn == 0L) {
            throw new IllegalStateException("Kernel function not found in compiled MSL library: " + functionName);
        }
        return mtl.newComputePipelineState(device, fn);
    }

    private final long[] allPipelines() {
        return new long[]{
                kQuantizeI8, kQuantizeActivationQ8_0, kQ8_0GemvSplit, kQ8_0GemvPlain,
                kRopeApplySplit, kRmsnormPartialSumsq, kRmsnormApply, kAttnScores,
                kSoftmaxInplace, kAttnWeightedSum, kSwigluActivate, kResidualAdd, kF32Gemv,
                kGeluActivate, kGegluActivate,
                kQ8_0GemmTiled, kF32GemmTiled, kRmsnormPartialSumsqRows, kRmsnormApplyRows,
                kRopeApplyPairwiseRows, kAttnScoresCausalBatched, kSoftmaxInplaceRows,
                kAttnWeightedSumCausalBatched
        };
    }

    /** Releases every pipeline state, the compiled library, and the command queue -- the Metal analogue of cuModuleUnload/cuDevicePrimaryCtxRelease. The system device itself is intentionally NOT released: MTLCreateSystemDefaultDevice/MTLCopyAllDevices hand back the OS-owned singleton device object(s), which Apple's own docs describe as safe to hold indefinitely and not meant to be released by the caller. */
    @Override
    public void close() {
        try {
            for (long pso : allPipelines()) {
                mtl.release(pso);
            }
            mtl.release(library);
            mtl.release(commandQueue);
        } catch (Throwable t) {
            // best-effort cleanup, mirrors the CUDA/OpenCL ports' close()
        }
    }
}