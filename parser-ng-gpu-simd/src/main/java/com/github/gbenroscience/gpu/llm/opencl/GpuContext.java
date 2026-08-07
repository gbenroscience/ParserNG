package com.github.gbenroscience.gpu.llm.opencl;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * OpenCL counterpart of {@code com.github.gbenroscience.gpu.llm.cuda.GpuContext}.
 * Bootstraps one platform + one device + one in-order command queue +
 * one built program, then resolves all 23 kernels KernelSource declares.
 *
 * DEVICE SELECTION: delegates to {@link OpenCLDeviceSelector} (this same
 * package's own copy -- distinct from
 * {@code com.github.gbenroscience.gpu.evaluator.opencl}'s selector, see
 * that class's javadoc for why they're kept separate rather than merged).
 * {@code OpenCLDeviceSelector.selectDevice(GpuVendor.AMD)} (or the
 * equivalent system property -- "opencl.gpu.vendor", "opencl.platform.index",
 * "opencl.device.index", "opencl.device.type") governs every device this
 * class constructs after that call -- see OpenCLDeviceSelector's javadoc
 * for the full precedence rules, including CPU-device selection via
 * "opencl.device.type=CPU", which this class now supports (a GpuContext
 * built against a CPU OpenCL device works exactly the same way as one
 * built against a GPU -- same kernels, same dispatch code -- just runs on
 * whatever throughput a CPU OpenCL runtime like Intel's or PoCL provides).
 *
 * QUEUE ORDERING: created with properties=0, i.e. the OpenCL-default
 * IN-ORDER queue (out-of-order execution requires explicitly requesting
 * CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, which this never does). This is
 * what LlamaLayer's "no sync between back-to-back kernel launches" sync
 * discipline depends on -- see LlamaLayer's class javadoc, same rationale
 * as the CUDA port's default-stream argument.
 *
 * BUILD: clBuildProgram compiles AND links KernelSource.CL_SOURCE against
 * this context's one device in a single call -- there is no OpenCL
 * equivalent of NVRTC as a separate library; the ICD's own compiler
 * (whatever the vendor driver ships) does it. Build failures surface the
 * full compiler log via clGetProgramBuildInfo, same pattern the CUDA port
 * uses for nvrtcGetProgramLog.
 *
 * UNVERIFIED: no OpenCL platform/device was available while writing this
 * port -- carefully traced against the OpenCL 1.2 spec's exact function
 * signatures and against the already-working CUDA sibling's structure,
 * but treat every kernel here as unrun until diffed against known-good
 * per-layer activations, same standing caveat the CUDA files in this
 * codebase carry.
 */
public final class GpuContext implements AutoCloseable {

    public final OpenCLBindings cl;

    public final MemorySegment platform;
    public final MemorySegment device;
    public final String selectedDeviceDescription;
    public final MemorySegment context;
    public final MemorySegment queue;
    public final MemorySegment program;

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

    /** Serializes kernelArgs-set + enqueue against the shared in-order queue -- same role as the CUDA port's dispatchLock. */
    public final Object dispatchLock = new Object();

    public GpuContext() {
        this.cl = new OpenCLBindings();

        try (Arena bootstrap = Arena.ofConfined()) {
           // OpenCLDeviceSelector.selectDevice(OpenCLDeviceSelector.DeviceType.GPU); 
            OpenCLDeviceSelector.SelectedDevice chosen = OpenCLDeviceSelector.resolve();
            this.platform = chosen.platform();
            this.device = chosen.device();
            this.selectedDeviceDescription = chosen.describe();

            // cl_context clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret)
            MemorySegment devicesArr = bootstrap.allocate(ValueLayout.ADDRESS, 1);
            devicesArr.setAtIndex(ValueLayout.ADDRESS, 0, device);
            MemorySegment errBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
            MemorySegment ctxResult = (MemorySegment) cl.clCreateContext.invoke(
                    MemorySegment.NULL, 1, devicesArr, MemorySegment.NULL, MemorySegment.NULL, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateContext");
            this.context = ctxResult;

            errBuf.set(ValueLayout.JAVA_INT, 0, 0);
            MemorySegment queueResult = (MemorySegment) cl.clCreateCommandQueue.invoke(
                    context, device, 0L, errBuf); // properties=0 -> in-order queue, see class javadoc
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateCommandQueue");
            this.queue = queueResult;

            this.program = buildProgram(bootstrap);

            this.kQuantizeI8 = createKernel(bootstrap, KernelSource.KERNEL_QUANTIZE_I8);
            this.kQuantizeActivationQ8_0 = createKernel(bootstrap, KernelSource.KERNEL_QUANTIZE_ACTIVATION_Q8_0);
            this.kQ8_0GemvSplit = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_SPLIT);
            this.kQ8_0GemvPlain = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_PLAIN);
            this.kRopeApplySplit = createKernel(bootstrap, KernelSource.KERNEL_ROPE_APPLY_SPLIT);
            this.kRmsnormPartialSumsq = createKernel(bootstrap, KernelSource.KERNEL_RMSNORM_PARTIAL_SUMSQ);
            this.kRmsnormApply = createKernel(bootstrap, KernelSource.KERNEL_RMSNORM_APPLY);
            this.kAttnScores = createKernel(bootstrap, KernelSource.KERNEL_ATTN_SCORES);
            this.kSoftmaxInplace = createKernel(bootstrap, KernelSource.KERNEL_SOFTMAX_INPLACE);
            this.kAttnWeightedSum = createKernel(bootstrap, KernelSource.KERNEL_ATTN_WEIGHTED_SUM);
            this.kSwigluActivate = createKernel(bootstrap, KernelSource.KERNEL_SWIGLU_ACTIVATE);
            this.kResidualAdd = createKernel(bootstrap, KernelSource.KERNEL_RESIDUAL_ADD);
            this.kF32Gemv = createKernel(bootstrap, KernelSource.KERNEL_F32_GEMV);

            this.kGeluActivate = createKernel(bootstrap, KernelSource.KERNEL_GELU_ACTIVATE);
            this.kGegluActivate = createKernel(bootstrap, KernelSource.KERNEL_GEGLU_ACTIVATE);

            this.kQ8_0GemmTiled = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMM_TILED);
            this.kF32GemmTiled = createKernel(bootstrap, KernelSource.KERNEL_F32_GEMM_TILED);
            this.kRmsnormPartialSumsqRows = createKernel(bootstrap, KernelSource.KERNEL_RMSNORM_PARTIAL_SUMSQ_ROWS);
            this.kRmsnormApplyRows = createKernel(bootstrap, KernelSource.KERNEL_RMSNORM_APPLY_ROWS);
            this.kRopeApplyPairwiseRows = createKernel(bootstrap, KernelSource.KERNEL_ROPE_APPLY_PAIRWISE_ROWS);
            this.kAttnScoresCausalBatched = createKernel(bootstrap, KernelSource.KERNEL_ATTN_SCORES_CAUSAL_BATCHED);
            this.kSoftmaxInplaceRows = createKernel(bootstrap, KernelSource.KERNEL_SOFTMAX_INPLACE_ROWS);
            this.kAttnWeightedSumCausalBatched = createKernel(bootstrap, KernelSource.KERNEL_ATTN_WEIGHTED_SUM_CAUSAL_BATCHED);

            System.err.println("[ParserNG LLM-OpenCL] " + selectedDeviceDescription);

        } catch (Throwable t) {
            throw new RuntimeException("Failed to bootstrap OpenCL GpuContext", t);
        }
    }

    private MemorySegment buildProgram(Arena arena) throws Throwable {
        MemorySegment src = arena.allocateFrom(KernelSource.CL_SOURCE, StandardCharsets.UTF_8);
        MemorySegment stringsArr = arena.allocate(ValueLayout.ADDRESS, 1);
        stringsArr.setAtIndex(ValueLayout.ADDRESS, 0, src);
        // lengths=NULL -> each string is treated as a NUL-terminated C string, which allocateFrom guarantees.
        MemorySegment errBuf = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment prog = (MemorySegment) cl.clCreateProgramWithSource.invoke(
                context, 1, stringsArr, MemorySegment.NULL, errBuf);
        check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateProgramWithSource");

        MemorySegment devicesArr = arena.allocate(ValueLayout.ADDRESS, 1);
        devicesArr.setAtIndex(ValueLayout.ADDRESS, 0, device);
        int buildStatus = (int) cl.clBuildProgram.invoke(
                prog, 1, devicesArr, MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL);
        if (buildStatus != OpenCLBindings.CL_SUCCESS) {
            throw new IllegalStateException(
                    "OpenCL build of LLM decoder kernels failed (" + OpenCLBindings.errorString(buildStatus) + "): "
                            + fetchBuildLog(arena, prog));
        }
        return prog;
    }

    private String fetchBuildLog(Arena arena, MemorySegment prog) throws Throwable {
        MemorySegment sizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        int status = (int) cl.clGetProgramBuildInfo.invoke(
                prog, device, OpenCLBindings.CL_PROGRAM_BUILD_LOG, 0L, MemorySegment.NULL, sizeBuf);
        if (status != OpenCLBindings.CL_SUCCESS) {
            return "(no build log available)";
        }
        long size = sizeBuf.get(ValueLayout.JAVA_LONG, 0);
        if (size <= 1) {
            return "(empty build log)";
        }
        MemorySegment logBuf = arena.allocate(size);
        cl.clGetProgramBuildInfo.invoke(prog, device, OpenCLBindings.CL_PROGRAM_BUILD_LOG, size, logBuf, MemorySegment.NULL);
        return logBuf.getString(0, StandardCharsets.UTF_8);
    }

    private MemorySegment createKernel(Arena arena, String name) throws Throwable {
        MemorySegment nameSeg = arena.allocateFrom(name, StandardCharsets.UTF_8);
        MemorySegment errBuf = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment kernel = (MemorySegment) cl.clCreateKernel.invoke(program, nameSeg, errBuf);
        check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateKernel(" + name + ")");
        return kernel;
    }

    static void check(int status, String call) {
        if (status != OpenCLBindings.CL_SUCCESS) {
            throw new IllegalStateException("OpenCL error in " + call + ": " + OpenCLBindings.errorString(status));
        }
    }

    @Override
    public void close() {
        try {
            releaseKernelQuietly(kQuantizeI8);
            releaseKernelQuietly(kQuantizeActivationQ8_0);
            releaseKernelQuietly(kQ8_0GemvSplit);
            releaseKernelQuietly(kQ8_0GemvPlain);
            releaseKernelQuietly(kRopeApplySplit);
            releaseKernelQuietly(kRmsnormPartialSumsq);
            releaseKernelQuietly(kRmsnormApply);
            releaseKernelQuietly(kAttnScores);
            releaseKernelQuietly(kSoftmaxInplace);
            releaseKernelQuietly(kAttnWeightedSum);
            releaseKernelQuietly(kSwigluActivate);
            releaseKernelQuietly(kResidualAdd);
            releaseKernelQuietly(kF32Gemv);
            releaseKernelQuietly(kGeluActivate);
            releaseKernelQuietly(kGegluActivate);
            releaseKernelQuietly(kQ8_0GemmTiled);
            releaseKernelQuietly(kF32GemmTiled);
            releaseKernelQuietly(kRmsnormPartialSumsqRows);
            releaseKernelQuietly(kRmsnormApplyRows);
            releaseKernelQuietly(kRopeApplyPairwiseRows);
            releaseKernelQuietly(kAttnScoresCausalBatched);
            releaseKernelQuietly(kSoftmaxInplaceRows);
            releaseKernelQuietly(kAttnWeightedSumCausalBatched);
            cl.clReleaseProgram.invoke(program);
            cl.clReleaseCommandQueue.invoke(queue);
            cl.clReleaseContext.invoke(context);
        } catch (Throwable t) {
            // best-effort cleanup
        }
    }

    private void releaseKernelQuietly(MemorySegment kernel) {
        try {
            if (kernel != null) {
                cl.clReleaseKernel.invoke(kernel);
            }
        } catch (Throwable t) {
            // best-effort cleanup
        }
    }
}