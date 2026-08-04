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
package com.github.gbenroscience.gpu.cuda.llm;
 

import com.github.gbenroscience.gpu.cuda.CudaBindings;
import com.github.gbenroscience.gpu.cuda.NvrtcBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * @author GBEMIRO
 * CUDA counterpart of {@code com.github.gbenroscience.gpu.llm.LlmGpuContext}.
Bootstraps and holds the CUDA primary context / module / kernel-function
handles for GPU LLM inference. One instance per selected device -- same
"build one per process, reuse across every GpuLlamaLayer call" contract
as the OpenCL version.

Follows CudaCompositeExpression's established pattern exactly rather
than inventing a new one:
  - cuInit once, device chosen via a system property (here
    "cuda.device.index", same property name CudaCompositeExpression
    already reads -- so both this and the interpreter path pick up the
    same -D flag if a process uses both).
  - The device's PRIMARY context is retained (cuDevicePrimaryCtxRetain)
    rather than an explicitly created one, so it stays cheaply shareable
    across threads via cuCtxSetCurrent -- see CudaCompositeExpression's
    class javadoc for why.
  - ONE NVRTC compile of KernelSource.CUDA_SOURCE, targeting the
    device's actual compute capability (queried via
    cuDeviceGetAttribute, same as the interpreter path), producing ONE
    PTX module with all 13 kernel entry points.

DIFFERENCE FROM CudaCompositeExpression: that class holds its CUcontext/
CUmodule/CUfunction handles as a lazily-initialized static singleton
shared process-wide, because the interpreter path has no natural
"owner" object. LlmGpuContext (OpenCL) is instead an explicit,
caller-managed, closeable instance -- so this mirrors THAT shape:
instance fields, an explicit close(). Nothing stops an application from
only ever constructing one, same as the OpenCL version's own
documented usage.

UNVERIFIED: no CUDA driver, no GPU, and no NVRTC toolchain were
available in the environment this was written in. Traced against the
exact FFM call shapes CudaCompositeExpression already uses for the
interpreter kernels (which share the same NvrtcBindings/CudaBindings),
but treat as an untested first draft.
 */
public final class GpuContext implements AutoCloseable {

    public final CudaBindings cu;
    public final NvrtcBindings nvrtc;

    public final int device;
    public final MemorySegment context; // CUcontext -- the device's primary context
    public final MemorySegment module;  // CUmodule -- holds all 13 kernels

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

    /** Serializes kernelParams-build + launch against the shared primary context, same rationale as CudaCompositeExpression.DISPATCH_LOCK. */
    public final Object dispatchLock = new Object();

    public GpuContext() {
        this.cu = new CudaBindings();
        this.nvrtc = new NvrtcBindings();

        try (Arena bootstrap = Arena.ofConfined()) {
            check((int) cu.cuInit.invoke(0), "cuInit");

            MemorySegment countBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
            check((int) cu.cuDeviceGetCount.invoke(countBuf), "cuDeviceGetCount");
            if (countBuf.get(ValueLayout.JAVA_INT, 0) < 1) {
                throw new IllegalStateException("No CUDA devices found");
            }

            int deviceIndex = Integer.getInteger("cuda.device.index", 0);
            MemorySegment deviceBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
            check((int) cu.cuDeviceGet.invoke(deviceBuf, deviceIndex), "cuDeviceGet");
            this.device = deviceBuf.get(ValueLayout.JAVA_INT, 0);

            MemorySegment majorBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
            MemorySegment minorBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
            check((int) cu.cuDeviceGetAttribute.invoke(majorBuf,
                    CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                    "cuDeviceGetAttribute(major)");
            check((int) cu.cuDeviceGetAttribute.invoke(minorBuf,
                    CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
                    "cuDeviceGetAttribute(minor)");
            int major = majorBuf.get(ValueLayout.JAVA_INT, 0);
            int minor = minorBuf.get(ValueLayout.JAVA_INT, 0);

            MemorySegment ctxBuf = bootstrap.allocate(ValueLayout.ADDRESS);
            check((int) cu.cuDevicePrimaryCtxRetain.invoke(ctxBuf, device), "cuDevicePrimaryCtxRetain");
            this.context = ctxBuf.get(ValueLayout.ADDRESS, 0);
            check((int) cu.cuCtxSetCurrent.invoke(context), "cuCtxSetCurrent");

            // ONE NVRTC compile -- the resulting PTX module contains all
            // 13 LLM decoder kernels (see KernelSource).
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

            System.err.println("[ParserNG LLM-CUDA] using device " + device
                    + " (compute capability " + major + "." + minor + ")");

        } catch (Throwable t) {
            throw new RuntimeException("Failed to bootstrap LlmCudaContext", t);
        }
    }

    private MemorySegment getFunction(Arena arena, String name) throws Throwable {
        MemorySegment nameSeg = arena.allocateFrom(name, StandardCharsets.UTF_8);
        MemorySegment fnBuf = arena.allocate(ValueLayout.ADDRESS);
        check((int) cu.cuModuleGetFunction.invoke(fnBuf, module, nameSeg), "cuModuleGetFunction(" + name + ")");
        return fnBuf.get(ValueLayout.ADDRESS, 0);
    }

    /** Same two-stage NVRTC-then-driver compile CudaCompositeExpression uses, just against KernelSource.CUDA_SOURCE. */
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