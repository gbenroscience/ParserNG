package com.github.gbenroscience.gpu.llm.opencl;


import com.github.gbenroscience.gpu.llm.core.AllocationTracker;
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

    private static final System.Logger LOG = System.getLogger(GpuContext.class.getName());
    /** Every live device allocation made through this context (see {@link AllocationTracker}). */
    public final AllocationTracker allocations = new AllocationTracker();
    private boolean closed;
    /** Number of device buffers still live when close() ran (0 = every owner released its buffers explicitly). */
    public volatile int sweptAtClose = -1;

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
    public final MemorySegment kRopeApplyPairs;
    public final MemorySegment kRmsnormPartialSumsq;
    public final MemorySegment kRmsnormApply;
    public final MemorySegment kAttnScores;
    public final MemorySegment kSoftmaxInplace;
    public final MemorySegment kAttnWeightedSum;
    public final MemorySegment kAttnDecodeFused;
    public final MemorySegment kQ8_0GemvPlainLegacy;
    public final MemorySegment kF32GemvLegacy;
    public final MemorySegment kRmsnormFused;
    final MemorySegment kQ8_0GemvVloadf;
    final MemorySegment kQ8_0GemvU32;
    final MemorySegment kQ8_0GemmRows;
    final MemorySegment kF32GemmRows;
    /** Decode GEMV variant used when -Dllm.gemv is not forced: 0 = wg (vload16), 1 = vloadf (float dp4), 2 = u32 (aligned words). Set by GemvAutotune. */
    volatile int gemvVariant = 0;
    /** Prefill GEMM choice (set by GemvAutotune.runGemm; true = row-wise kernels, false = the original one-thread-per-output kernels). */
    volatile boolean q8GemmRows = true;
    volatile boolean f32GemmRows = true;
    final MemorySegment kBwU8;
    final MemorySegment kBwU32;
    final MemorySegment kBwU128;
    final MemorySegment kBwV16Off2;
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

    /** Profiling follows -Dllm.profile=true. */
    public GpuContext() {
        this(DeviceProfiler.ON);
    }

    /** Category recorded with every buffer allocated from now on (for the memory report); owners set it before allocating. */
    volatile String allocCategory = "other";

    /** Event profiler of this context, or null when profiling is off. */
    final DeviceProfiler prof;

    /** @param profile capture OpenCL events for every command (see {@link DeviceProfiler}); adds a little overhead. */
    GpuContext(boolean profile) {
        this.prof = profile ? new DeviceProfiler() : null;
        this.cl = new OpenCLBindings(prof);
        final long tCtx0 = LoadTimer.start();

        try (Arena bootstrap = Arena.ofConfined()) {
           // OpenCLDeviceSelector.selectDevice(OpenCLDeviceSelector.DeviceType.GPU); 
            long tSel = LoadTimer.start();
            OpenCLDeviceSelector.SelectedDevice chosen = OpenCLDeviceSelector.resolve();
            LoadTimer.stop("opencl: platform/device selection", tSel);
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
                    context, device, prof != null ? 2L : 0L, errBuf); // 0 = plain in-order queue (see class javadoc); 2 = CL_QUEUE_PROFILING_ENABLE
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateCommandQueue");
            this.queue = queueResult;

            long tBuild = LoadTimer.start();
            this.program = buildProgram(bootstrap);
            LoadTimer.stop("opencl: clBuildProgram (kernel JIT)", tBuild);
            long tKern = LoadTimer.start();

            this.kQuantizeI8 = createKernel(bootstrap, KernelSource.KERNEL_QUANTIZE_I8);
            this.kQuantizeActivationQ8_0 = createKernel(bootstrap, KernelSource.KERNEL_QUANTIZE_ACTIVATION_Q8_0);
            this.kQ8_0GemvSplit = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_SPLIT);
            this.kQ8_0GemvPlain = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_PLAIN);
            this.kRopeApplyPairs = createKernel(bootstrap, KernelSource.KERNEL_ROPE_APPLY_PAIRS);
            this.kRmsnormPartialSumsq = createKernel(bootstrap, KernelSource.KERNEL_RMSNORM_PARTIAL_SUMSQ);
            this.kRmsnormApply = createKernel(bootstrap, KernelSource.KERNEL_RMSNORM_APPLY);
            this.kAttnScores = createKernel(bootstrap, KernelSource.KERNEL_ATTN_SCORES);
            this.kSoftmaxInplace = createKernel(bootstrap, KernelSource.KERNEL_SOFTMAX_INPLACE);
            this.kAttnWeightedSum = createKernel(bootstrap, KernelSource.KERNEL_ATTN_WEIGHTED_SUM);
            this.kAttnDecodeFused = createKernel(bootstrap, KernelSource.KERNEL_ATTN_DECODE_FUSED);
            this.kQ8_0GemvPlainLegacy = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_PLAIN_LEGACY);
            this.kF32GemvLegacy = createKernel(bootstrap, KernelSource.KERNEL_F32_GEMV_LEGACY);
            this.kRmsnormFused = createKernel(bootstrap, KernelSource.KERNEL_RMSNORM_FUSED);
            this.kQ8_0GemvVloadf = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_VLOADF);
            this.kQ8_0GemvU32 = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMV_U32);
            this.kQ8_0GemmRows = createKernel(bootstrap, KernelSource.KERNEL_Q8_0_GEMM_ROWS);
            this.kF32GemmRows = createKernel(bootstrap, KernelSource.KERNEL_F32_GEMM_ROWS);
            this.kBwU8 = createKernel(bootstrap, KernelSource.KERNEL_BW_U8);
            this.kBwU32 = createKernel(bootstrap, KernelSource.KERNEL_BW_U32);
            this.kBwU128 = createKernel(bootstrap, KernelSource.KERNEL_BW_U128);
            this.kBwV16Off2 = createKernel(bootstrap, KernelSource.KERNEL_BW_V16_OFF2);
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
            LoadTimer.stop("opencl: clCreateKernel x all", tKern);

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
        if (prof != null) {
            prof.nameKernel(kernel, name);
        }
        return kernel;
    }

    static void check(int status, String call) {
        if (status != OpenCLBindings.CL_SUCCESS) {
            throw new IllegalStateException("OpenCL error in " + call + ": " + OpenCLBindings.errorString(status));
        }
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        long[] leaked = allocations.drain();
        sweptAtClose = leaked.length;
        if (leaked.length > 0) {
            LOG.log(System.Logger.Level.WARNING, "closing OpenCL context with {0} device buffers still allocated; releasing them", leaked.length);
        }
        for (long addr : leaked) {
            try {
                cl.clReleaseMemObject.invoke(java.lang.foreign.MemorySegment.ofAddress(addr));
            } catch (Throwable t) {
                LOG.log(System.Logger.Level.ERROR, "clReleaseMemObject failed for 0x" + Long.toHexString(addr), t);
            }
        }
        try {
            releaseKernelQuietly(kQuantizeI8);
            releaseKernelQuietly(kQuantizeActivationQ8_0);
            releaseKernelQuietly(kQ8_0GemvSplit);
            releaseKernelQuietly(kQ8_0GemvPlain);
            releaseKernelQuietly(kRopeApplyPairs);
            releaseKernelQuietly(kRmsnormPartialSumsq);
            releaseKernelQuietly(kRmsnormApply);
            releaseKernelQuietly(kAttnScores);
            releaseKernelQuietly(kSoftmaxInplace);
            releaseKernelQuietly(kAttnWeightedSum);
            releaseKernelQuietly(kAttnDecodeFused);
            releaseKernelQuietly(kQ8_0GemvPlainLegacy);
            releaseKernelQuietly(kF32GemvLegacy);
            releaseKernelQuietly(kRmsnormFused);
            releaseKernelQuietly(kQ8_0GemvVloadf);
            releaseKernelQuietly(kQ8_0GemvU32);
            releaseKernelQuietly(kQ8_0GemmRows);
            releaseKernelQuietly(kF32GemmRows);
            releaseKernelQuietly(kBwU8);
            releaseKernelQuietly(kBwU32);
            releaseKernelQuietly(kBwU128);
            releaseKernelQuietly(kBwV16Off2);
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
            LOG.log(System.Logger.Level.ERROR, "failed releasing OpenCL program/queue/context", t);
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

    // ---------------------------------------------------------------- device report

    private String infoString(int param) throws Throwable {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment size = a.allocate(ValueLayout.JAVA_LONG);
            check((int) cl.clGetDeviceInfo.invoke(device, param, 0L, MemorySegment.NULL, size), "clGetDeviceInfo(size)");
            long n = size.get(ValueLayout.JAVA_LONG, 0);
            MemorySegment buf = a.allocate(Math.max(n, 1));
            check((int) cl.clGetDeviceInfo.invoke(device, param, n, buf, MemorySegment.NULL), "clGetDeviceInfo");
            return n <= 1 ? "" : buf.getString(0);
        }
    }

    private long infoNumber(int param, int bytes) throws Throwable {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment buf = a.allocate(8);
            check((int) cl.clGetDeviceInfo.invoke(device, param, (long) bytes, buf, MemorySegment.NULL), "clGetDeviceInfo");
            return bytes == 4 ? Integer.toUnsignedLong(buf.get(ValueLayout.JAVA_INT, 0)) : buf.get(ValueLayout.JAVA_LONG, 0);
        }
    }

    /** Human-readable hardware / driver description (name, driver, OpenCL version, memory, compute units, ...). */
    public String deviceInfo() {
        try {
            String ext = infoString(0x1030);
            StringBuilder sb = new StringBuilder();
            sb.append("selected: ").append(selectedDeviceDescription).append('\n');
            sb.append("CL_DEVICE_NAME:            ").append(infoString(0x102B)).append('\n');
            sb.append("CL_DEVICE_VENDOR:          ").append(infoString(0x102C)).append('\n');
            sb.append("CL_DRIVER_VERSION:         ").append(infoString(0x102D)).append('\n');
            sb.append("CL_DEVICE_VERSION:         ").append(infoString(0x102F)).append('\n');
            sb.append("CL_DEVICE_OPENCL_C_VERSION:").append(' ').append(infoString(0x103D)).append('\n');
            sb.append("compute units / max clock: ").append(infoNumber(0x1002, 4)).append(" / ").append(infoNumber(0x100C, 4)).append(" MHz\n");
            sb.append(String.format("global memory:             %.2f GB (max single allocation %.2f GB, local memory %d KB, host-unified=%d)%n",
                    infoNumber(0x101F, 8) / 1e9, infoNumber(0x1010, 8) / 1e9, infoNumber(0x1023, 8) / 1024, infoNumber(0x1035, 4)));
            sb.append("max work-group size:       ").append(infoNumber(0x1004, 8)).append('\n');
            sb.append("extensions of interest:    ");
            for (String e : new String[]{"cl_khr_fp16", "cl_khr_subgroups", "cl_intel_subgroups", "cl_intel_required_subgroup_size",
                "cl_khr_integer_dot_product", "cl_intel_subgroups_char", "cl_khr_int64_base_atomics"}) {
                if (ext.contains(e)) {
                    sb.append(e).append(' ');
                }
            }
            return sb.append('\n').toString();
        } catch (Throwable t) {
            return "device info unavailable: " + t;
        }
    }
}
