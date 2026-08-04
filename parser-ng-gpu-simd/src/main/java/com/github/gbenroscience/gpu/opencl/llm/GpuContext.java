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
package com.github.gbenroscience.gpu.opencl.llm;
 

import com.github.gbenroscience.gpu.opencl.OpenClBindings;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * @author GBEMIRO
Bootstraps and holds the OpenCL context/queue/program/kernels for GPU LLM
inference. One instance per selected device -- unlike
OpenClCompositeExpression's per-device registry (which needed to support
many independently-compiled expressions sharing a device), this class
only ever compiles ONE fixed program (KernelSource.OPENCL_SOURCE),
since the LLM kernel set is fixed, not user-expression-driven. Build one
GpuContext per process and reuse it across every GpuLlamaLayer call.

Device selection reuses the exact same -Dopencl.gpu.vendor /
-Dopencl.platform.index system properties OpenClCompositeExpression
already established -- see that class if you need to pin a specific GPU
(e.g. multiple GPUs installed) before constructing this.

UNVERIFIED: no OpenCL driver or GPU available in the environment this was
written in. Traced carefully against the same FFM call shapes already
proven out in OpenClCompositeExpression, but treat as an untested first
draft.
 */
public final class GpuContext implements AutoCloseable {

    public final OpenClBindings cl;
    public final MemorySegment platform;
    public final MemorySegment device;
    public final MemorySegment context;
    public final MemorySegment queue;
    public final MemorySegment program;

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

    /** Serializes kernel-arg-set + dispatch, same rationale as OpenClCompositeExpression's per-context lock. */
    public final Object dispatchLock = new Object();

    public GpuContext() {
        this.cl = new OpenClBindings();
        try (Arena bootstrap = Arena.ofConfined()) {
            MemorySegment errBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
            MemorySegment countBuf = bootstrap.allocate(ValueLayout.JAVA_INT);

            // --- platform/device selection: same properties as OpenClCompositeExpression ---
            check((int) cl.clGetPlatformIDs.invoke(0, MemorySegment.NULL, countBuf), "clGetPlatformIDs(count)");
            int platformCount = countBuf.get(ValueLayout.JAVA_INT, 0);
            if (platformCount < 1) {
                throw new IllegalStateException("No OpenCL platforms found");
            }
            MemorySegment platformArr = bootstrap.allocate(ValueLayout.ADDRESS, platformCount);
            check((int) cl.clGetPlatformIDs.invoke(platformCount, platformArr, MemorySegment.NULL),
                    "clGetPlatformIDs(list)");

            MemorySegment chosenPlatform = null;
            MemorySegment chosenDevice = null;
            String vendorProp = System.getProperty("opencl.gpu.vendor");
            String platformIndexProp = System.getProperty("opencl.platform.index");

            outer:
            for (int p = 0; p < platformCount; p++) {
                MemorySegment plat = platformArr.getAtIndex(ValueLayout.ADDRESS, p);
                if (platformIndexProp != null && p != Integer.parseInt(platformIndexProp.trim())) {
                    continue;
                }
                int devStatus = (int) cl.clGetDeviceIDs.invoke(
                        plat, OpenClBindings.CL_DEVICE_TYPE_GPU, 0, MemorySegment.NULL, countBuf);
                if (devStatus == OpenClBindings.CL_DEVICE_NOT_FOUND) {
                    continue;
                }
                check(devStatus, "clGetDeviceIDs(count)");
                int devCount = countBuf.get(ValueLayout.JAVA_INT, 0);
                if (devCount < 1) {
                    continue;
                }
                MemorySegment devArr = bootstrap.allocate(ValueLayout.ADDRESS, devCount);
                check((int) cl.clGetDeviceIDs.invoke(plat, OpenClBindings.CL_DEVICE_TYPE_GPU,
                        devCount, devArr, MemorySegment.NULL), "clGetDeviceIDs(list)");

                for (int d = 0; d < devCount; d++) {
                    MemorySegment dev = devArr.getAtIndex(ValueLayout.ADDRESS, d);
                    if (vendorProp == null || vendorProp.isBlank()) {
                        chosenPlatform = plat;
                        chosenDevice = dev;
                        break outer;
                    }
                    String vendor = deviceInfoString(bootstrap, dev, OpenClBindings.CL_DEVICE_VENDOR).toLowerCase();
                    String name = deviceInfoString(bootstrap, dev, OpenClBindings.CL_DEVICE_NAME).toLowerCase();
                    String needle = vendorProp.trim().toLowerCase();
                    if (vendor.contains(needle) || name.contains(needle)) {
                        chosenPlatform = plat;
                        chosenDevice = dev;
                        break outer;
                    }
                }
            }
            if (chosenDevice == null) {
                throw new IllegalStateException(
                        "No OpenCL GPU device found matching opencl.gpu.vendor=\"" + vendorProp
                        + "\" / opencl.platform.index=" + platformIndexProp);
            }
            this.platform = chosenPlatform;
            this.device = chosenDevice;

            // --- context / queue ---
            MemorySegment devicesForCtx = bootstrap.allocate(ValueLayout.ADDRESS);
            devicesForCtx.set(ValueLayout.ADDRESS, 0, device);
            this.context = (MemorySegment) cl.clCreateContext.invoke(
                    MemorySegment.NULL, 1, devicesForCtx, MemorySegment.NULL, MemorySegment.NULL, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateContext");

            this.queue = (MemorySegment) cl.clCreateCommandQueue.invoke(context, device, 0L, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateCommandQueue");

            // --- program: ONE build, all 11 kernels ---
            MemorySegment src = bootstrap.allocateFrom(KernelSource.OPENCL_SOURCE);
            MemorySegment srcPtrArr = bootstrap.allocate(ValueLayout.ADDRESS);
            srcPtrArr.set(ValueLayout.ADDRESS, 0, src);

            this.program = (MemorySegment) cl.clCreateProgramWithSource.invoke(
                    context, 1, srcPtrArr, MemorySegment.NULL, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateProgramWithSource");

            int buildStatus = (int) cl.clBuildProgram.invoke(program, 1, devicesForCtx,
                    MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL);
            if (buildStatus != OpenClBindings.CL_SUCCESS) {
                throw new IllegalStateException(
                        "OpenCL LLM kernel build failed (" + buildStatus + "): " + fetchBuildLog(bootstrap));
            }

            this.kQuantizeI8 = createKernel(bootstrap, KernelSource.KERNEL_QUANTIZE_I8, errBuf);
            this.kQuantizeActivationQ8_0 = createKernel(bootstrap, KernelSource.KERNEL_QUANTIZE_ACTIVATION_Q8_0, errBuf);
            this.kQ8_0GemvSplit = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_SPLIT, errBuf);
            this.kQ8_0GemvPlain = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_PLAIN, errBuf);
            this.kRopeApplySplit = createKernel(bootstrap, KernelSource.KERNEL_ROPE_APPLY_SPLIT, errBuf);
            this.kRmsnormPartialSumsq = createKernel(bootstrap, KernelSource.KERNEL_RMSNORM_PARTIAL_SUMSQ, errBuf);
            this.kRmsnormApply = createKernel(bootstrap, KernelSource.KERNEL_RMSNORM_APPLY, errBuf);
            this.kAttnScores = createKernel(bootstrap, KernelSource.KERNEL_ATTN_SCORES, errBuf);
            this.kSoftmaxInplace = createKernel(bootstrap, KernelSource.KERNEL_SOFTMAX_INPLACE, errBuf);
            this.kAttnWeightedSum = createKernel(bootstrap, KernelSource.KERNEL_ATTN_WEIGHTED_SUM, errBuf);
            this.kSwigluActivate = createKernel(bootstrap, KernelSource.KERNEL_SWIGLU_ACTIVATE, errBuf);
            this.kResidualAdd = createKernel(bootstrap, KernelSource.KERNEL_RESIDUAL_ADD, errBuf);
            this.kF32Gemv = createKernel(bootstrap, KernelSource.KERNEL_F32_GEMV, errBuf);

            System.err.println("[ParserNG LLM-GPU] using " + deviceInfoString(bootstrap, device, OpenClBindings.CL_DEVICE_VENDOR)
                    + " " + deviceInfoString(bootstrap, device, OpenClBindings.CL_DEVICE_NAME));

        } catch (Throwable t) {
            throw new RuntimeException("Failed to bootstrap LlmGpuContext", t);
        }
    }

    private MemorySegment createKernel(Arena arena, String name, MemorySegment errBuf) throws Throwable {
        MemorySegment nameSeg = arena.allocateFrom(name, StandardCharsets.UTF_8);
        MemorySegment kernel = (MemorySegment) cl.clCreateKernel.invoke(program, nameSeg, errBuf);
        check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateKernel(" + name + ")");
        return kernel;
    }

    private String deviceInfoString(Arena arena, MemorySegment dev, int param) throws Throwable {
        MemorySegment sizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        cl.clGetDeviceInfo.invoke(dev, param, 0L, MemorySegment.NULL, sizeBuf);
        long size = sizeBuf.get(ValueLayout.JAVA_LONG, 0);
        MemorySegment buf = arena.allocate(Math.max(size, 1));
        cl.clGetDeviceInfo.invoke(dev, param, size, buf, MemorySegment.NULL);
        return buf.getString(0, StandardCharsets.UTF_8);
    }

    private String fetchBuildLog(Arena arena) throws Throwable {
        MemorySegment sizeRet = arena.allocate(ValueLayout.JAVA_LONG);
        cl.clGetProgramBuildInfo.invoke(program, device, OpenClBindings.CL_PROGRAM_BUILD_LOG,
                0L, MemorySegment.NULL, sizeRet);
        long logSize = sizeRet.get(ValueLayout.JAVA_LONG, 0);
        if (logSize <= 0) {
            return "(no build log)";
        }
        MemorySegment logBuf = arena.allocate(logSize);
        cl.clGetProgramBuildInfo.invoke(program, device, OpenClBindings.CL_PROGRAM_BUILD_LOG,
                logSize, logBuf, MemorySegment.NULL);
        return logBuf.getString(0, StandardCharsets.UTF_8);
    }

    private static void check(int status, String call) {
        if (status != OpenClBindings.CL_SUCCESS) {
            throw new IllegalStateException("OpenCL error in " + call + ": code " + status);
        }
    }

    @Override
    public void close() {
        try {
            cl.clReleaseKernel.invoke(kQuantizeI8);
            cl.clReleaseKernel.invoke(kQuantizeActivationQ8_0);
            cl.clReleaseKernel.invoke(kQ8_0GemvSplit);
            cl.clReleaseKernel.invoke(kQ8_0GemvPlain);
            cl.clReleaseKernel.invoke(kRopeApplySplit);
            cl.clReleaseKernel.invoke(kRmsnormPartialSumsq);
            cl.clReleaseKernel.invoke(kRmsnormApply);
            cl.clReleaseKernel.invoke(kAttnScores);
            cl.clReleaseKernel.invoke(kSoftmaxInplace);
            cl.clReleaseKernel.invoke(kAttnWeightedSum);
            cl.clReleaseKernel.invoke(kSwigluActivate);
            cl.clReleaseKernel.invoke(kResidualAdd);
            cl.clReleaseKernel.invoke(kF32Gemv);
            cl.clReleaseProgram.invoke(program);
            cl.clReleaseCommandQueue.invoke(queue);
            cl.clReleaseContext.invoke(context);
        } catch (Throwable t) {
            // best-effort cleanup
        }
    }
}