package com.github.gbenroscience.gpu.llm.cuda;

import com.github.gbenroscience.gpu.evaluator.cuda.CudaBindings;
import com.github.gbenroscience.gpu.evaluator.cuda.NvrtcBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * CUDA counterpart of {@code com.github.gbenroscience.gpu.llm.LlmGpuContext}
 * -- v2, extended for KernelSource's larger kernel set (GeLU/GeGLU
 * activations + the batched prefill path). Bootstrap sequence and
 * ownership model are unchanged from v1 -- see that version's javadoc for
 * the full rationale (primary-context retain, single NVRTC compile,
 * "cuda.device.index" system property). This version just resolves 8
 * additional CUfunction handles out of the same module.
 *
 * UNVERIFIED: no CUDA driver, GPU, or NVRTC toolchain were available
 * while writing this. Same caveat as v1 and as every kernel file in this
 * codebase.
 */
public final class GpuContext implements AutoCloseable {

    public final CudaBindings cu;
    public final NvrtcBindings nvrtc;

    public final int device;
    public final String selectedDeviceDescription;
    public final MemorySegment context; // CUcontext -- the device's primary context
    public final MemorySegment module;  // CUmodule -- holds all kernels

    // ---- decode-path kernels ----
    public final MemorySegment kQuantizeI8;
    public final MemorySegment kQuantizeActivationQ8_0;
    public final MemorySegment kQ8_0GemvSplit;
    public final MemorySegment kQ8_0GemvPlain;
    public final MemorySegment kRopeApplySplit;
    public final MemorySegment kRmsnormPartialSumsq;
    public final MemorySegment kRmsnormApply;
    public final MemorySegment kAttnScores;
    public final MemorySegment kSoftmaxInplace;
    public final MemorySegment kAttnWeightedSum;
    public final MemorySegment kSwigluActivate;
    public final MemorySegment kResidualAdd;
    public final MemorySegment kF32Gemv;

    // ---- FFN activation alternatives ----
    public final MemorySegment kGeluActivate;
    public final MemorySegment kGegluActivate;

    // ---- batched prefill kernels ----
    public final MemorySegment kQ8_0GemmTiled;
    public final MemorySegment kF32GemmTiled;
    public final MemorySegment kRmsnormPartialSumsqRows;
    public final MemorySegment kRmsnormApplyRows;
    public final MemorySegment kRopeApplyPairwiseRows;
    public final MemorySegment kAttnScoresCausalBatched;
    public final MemorySegment kSoftmaxInplaceRows;
    public final MemorySegment kAttnWeightedSumCausalBatched;

    /** Serializes kernelParams-build + launch against the shared primary context. */
    public final Object dispatchLock = new Object();

    public GpuContext() {
        this.cu = new CudaBindings();
        this.nvrtc = new NvrtcBindings();

        try (Arena bootstrap = Arena.ofConfined()) {
            CudaDeviceSelector.SelectedDevice chosen = CudaDeviceSelector.resolve();
            this.device = chosen.cuDevice();
            this.selectedDeviceDescription = chosen.describe();
            int major = chosen.major();
            int minor = chosen.minor();

            MemorySegment ctxBuf = bootstrap.allocate(ValueLayout.ADDRESS);
            check((int) cu.cuDevicePrimaryCtxRetain.invoke(ctxBuf, device), "cuDevicePrimaryCtxRetain");
            this.context = ctxBuf.get(ValueLayout.ADDRESS, 0);
            check((int) cu.cuCtxSetCurrent.invoke(context), "cuCtxSetCurrent");

            String ptx = compileToPtx(bootstrap, major, minor);

            MemorySegment ptxSrc = bootstrap.allocateFrom(ptx);
            MemorySegment moduleBuf = bootstrap.allocate(ValueLayout.ADDRESS);
            check((int) cu.cuModuleLoadData.invoke(moduleBuf, ptxSrc), "cuModuleLoadData");
            this.module = moduleBuf.get(ValueLayout.ADDRESS, 0);

            this.kQuantizeI8 = getFunction(bootstrap, KernelSource.KERNEL_QUANTIZE_I8);
            this.kQuantizeActivationQ8_0 = getFunction(bootstrap, KernelSource.KERNEL_QUANTIZE_ACTIVATION_Q8_0);
            this.kQ8_0GemvSplit = getFunction(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_SPLIT);
            this.kQ8_0GemvPlain = getFunction(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_PLAIN);
            this.kRopeApplySplit = getFunction(bootstrap, KernelSource.KERNEL_ROPE_APPLY_SPLIT);
            this.kRmsnormPartialSumsq = getFunction(bootstrap, KernelSource.KERNEL_RMSNORM_PARTIAL_SUMSQ);
            this.kRmsnormApply = getFunction(bootstrap, KernelSource.KERNEL_RMSNORM_APPLY);
            this.kAttnScores = getFunction(bootstrap, KernelSource.KERNEL_ATTN_SCORES);
            this.kSoftmaxInplace = getFunction(bootstrap, KernelSource.KERNEL_SOFTMAX_INPLACE);
            this.kAttnWeightedSum = getFunction(bootstrap, KernelSource.KERNEL_ATTN_WEIGHTED_SUM);
            this.kSwigluActivate = getFunction(bootstrap, KernelSource.KERNEL_SWIGLU_ACTIVATE);
            this.kResidualAdd = getFunction(bootstrap, KernelSource.KERNEL_RESIDUAL_ADD);
            this.kF32Gemv = getFunction(bootstrap, KernelSource.KERNEL_F32_GEMV);

            this.kGeluActivate = getFunction(bootstrap, KernelSource.KERNEL_GELU_ACTIVATE);
            this.kGegluActivate = getFunction(bootstrap, KernelSource.KERNEL_GEGLU_ACTIVATE);

            this.kQ8_0GemmTiled = getFunction(bootstrap, KernelSource.KERNEL_Q8_0_GEMM_TILED);
            this.kF32GemmTiled = getFunction(bootstrap, KernelSource.KERNEL_F32_GEMM_TILED);
            this.kRmsnormPartialSumsqRows = getFunction(bootstrap, KernelSource.KERNEL_RMSNORM_PARTIAL_SUMSQ_ROWS);
            this.kRmsnormApplyRows = getFunction(bootstrap, KernelSource.KERNEL_RMSNORM_APPLY_ROWS);
            this.kRopeApplyPairwiseRows = getFunction(bootstrap, KernelSource.KERNEL_ROPE_APPLY_PAIRWISE_ROWS);
            this.kAttnScoresCausalBatched = getFunction(bootstrap, KernelSource.KERNEL_ATTN_SCORES_CAUSAL_BATCHED);
            this.kSoftmaxInplaceRows = getFunction(bootstrap, KernelSource.KERNEL_SOFTMAX_INPLACE_ROWS);
            this.kAttnWeightedSumCausalBatched = getFunction(bootstrap, KernelSource.KERNEL_ATTN_WEIGHTED_SUM_CAUSAL_BATCHED);

            System.err.println("[ParserNG LLM-CUDA] " + chosen.describe());

        } catch (Throwable t) {
            throw new RuntimeException("Failed to bootstrap GpuContext", t);
        }
    }

    private MemorySegment getFunction(Arena arena, String name) throws Throwable {
        MemorySegment nameSeg = arena.allocateFrom(name, StandardCharsets.UTF_8);
        MemorySegment fnBuf = arena.allocate(ValueLayout.ADDRESS);
        check((int) cu.cuModuleGetFunction.invoke(fnBuf, module, nameSeg), "cuModuleGetFunction(" + name + ")");
        return fnBuf.get(ValueLayout.ADDRESS, 0);
    }

    private String compileToPtx(Arena arena, int major, int minor) throws Throwable {
        MemorySegment src = arena.allocateFrom(KernelSource.CUDA_SOURCE, StandardCharsets.UTF_8);
        MemorySegment name = arena.allocateFrom("llm_decoder_kernels.cu", StandardCharsets.UTF_8);

        MemorySegment progBuf = arena.allocate(ValueLayout.ADDRESS);
        checkNvrtc((int) nvrtc.nvrtcCreateProgram.invoke(progBuf, src, name, 0,
                MemorySegment.NULL, MemorySegment.NULL), "nvrtcCreateProgram");
        MemorySegment program = progBuf.get(ValueLayout.ADDRESS, 0);

        MemorySegment archOpt = arena.allocateFrom(
                "--gpu-architecture=compute_" + major + minor, StandardCharsets.UTF_8);
        MemorySegment optionsArr = arena.allocate(ValueLayout.ADDRESS, 1);
        optionsArr.setAtIndex(ValueLayout.ADDRESS, 0, archOpt);

        int compileStatus = (int) nvrtc.nvrtcCompileProgram.invoke(program, 1, optionsArr);
        if (compileStatus != NvrtcBindings.NVRTC_SUCCESS) {
            throw new IllegalStateException(
                    "NVRTC compile of LLM decoder kernels failed (" + compileStatus + "): "
                            + fetchCompileLog(arena, program));
        }

        MemorySegment ptxSizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        checkNvrtc((int) nvrtc.nvrtcGetPTXSize.invoke(program, ptxSizeBuf), "nvrtcGetPTXSize");
        long ptxSize = ptxSizeBuf.get(ValueLayout.JAVA_LONG, 0);

        MemorySegment ptxBuf = arena.allocate(ptxSize);
        checkNvrtc((int) nvrtc.nvrtcGetPTX.invoke(program, ptxBuf), "nvrtcGetPTX");

        checkNvrtc((int) nvrtc.nvrtcDestroyProgram.invoke(progBuf), "nvrtcDestroyProgram");

        return ptxBuf.getString(0, StandardCharsets.UTF_8);
    }

    private String fetchCompileLog(Arena arena, MemorySegment program) throws Throwable {
        MemorySegment logSizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        nvrtc.nvrtcGetProgramLogSize.invoke(program, logSizeBuf);
        long logSize = logSizeBuf.get(ValueLayout.JAVA_LONG, 0);
        if (logSize <= 1) {
            return "(no compile log)";
        }
        MemorySegment logBuf = arena.allocate(logSize);
        nvrtc.nvrtcGetProgramLog.invoke(program, logBuf);
        return logBuf.getString(0, StandardCharsets.UTF_8);
    }

    static void check(int status, String call) {
        if (status != CudaBindings.CUDA_SUCCESS) {
            throw new IllegalStateException("CUDA error in " + call + ": code " + status);
        }
    }

    private static void checkNvrtc(int status, String call) {
        if (status != NvrtcBindings.NVRTC_SUCCESS) {
            throw new IllegalStateException("NVRTC error in " + call + ": code " + status);
        }
    }

    @Override
    public void close() {
        try {
            cu.cuModuleUnload.invoke(module);
            cu.cuDevicePrimaryCtxRelease.invoke(device);
        } catch (Throwable t) {
            // best-effort cleanup
        }
    }
}