package com.github.gbenroscience.gpu.llm.opencl;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.List;

/**
 * Raw FFM downcalls into the OpenCL ICD loader (libOpenCL.so.1 on Linux,
 * OpenCL.dll on Windows, the OpenCL.framework on macOS). This is the
 * vendor-neutral ICD dispatch library every conformant OpenCL
 * implementation (NVIDIA, AMD, Intel, PoCL, etc.) installs -- it forwards
 * every call here to whichever vendor driver actually owns the chosen
 * platform/device, so nothing in this file is vendor-specific.
 *
 * Placed in {@code com.github.gbenroscience.gpu.opencl} (one package level
 * above {@code .llm}), mirroring where {@code CudaBindings}/
 * {@code NvrtcBindings} sit relative to {@code com.github.gbenroscience.gpu.llm.cuda}
 * -- see CudaBindings' class javadoc, which already references this class
 * by name. This is a fresh, self-contained implementation (no dependency
 * on an existing OpenCLBindings elsewhere in the tree) so the LLM port is
 * drop-in testable on its own; if your project already has an
 * OpenCLBindings class, reconcile the two rather than shipping both.
 *
 * Unlike the CUDA driver, OpenCL needs no separate NVRTC-style runtime
 * compiler binding: clBuildProgram compiles source text directly, so this
 * one file covers everything GpuContext needs (platform/device
 * enumeration, context/queue, program build, kernel/buffer lifecycle,
 * enqueue calls).
 *
 * NAMING NOTE: cl_mem/cl_kernel/cl_program/cl_command_queue/cl_context/
 * cl_device_id/cl_platform_id are all opaque native pointers (8 bytes on
 * every platform OpenCL runs on) -- represented here as
 * {@link MemorySegment} (an ADDRESS-layout value), never as raw {@code long}
 * device addresses the way CUDA's CUdeviceptr is. There is no pointer
 * arithmetic on a cl_mem handle in OpenCL; sub-region access goes through
 * clEnqueueCopyBuffer/clEnqueueReadBuffer/clEnqueueWriteBuffer's explicit
 * byte-offset parameters instead (see GpuState/LlamaLayer for where that
 * matters -- KV-cache writes and batch-row extraction).
 *
 * cl_device_type and cl_mem_flags/cl_command_queue_properties are
 * cl_bitfield, which the OpenCL spec fixes at 64 bits (cl_ulong) on every
 * platform -- passed as JAVA_LONG throughout, not JAVA_INT.
 */
public final class OpenCLBindings {

    public static final int CL_SUCCESS = 0;

    // ---- cl_device_type bitfield values (cl_bitfield / cl_ulong) ----
    public static final long CL_DEVICE_TYPE_DEFAULT = 1L << 0;
    public static final long CL_DEVICE_TYPE_CPU = 1L << 1;
    public static final long CL_DEVICE_TYPE_GPU = 1L << 2;
    public static final long CL_DEVICE_TYPE_ACCELERATOR = 1L << 3;
    public static final long CL_DEVICE_TYPE_ALL = 0xFFFFFFFFL;

    // ---- cl_mem_flags bitfield values used by this codebase ----
    public static final long CL_MEM_READ_WRITE = 1L << 0;
    public static final long CL_MEM_READ_ONLY = 1L << 2;

    // ---- cl_bool ----
    public static final int CL_TRUE = 1;
    public static final int CL_FALSE = 0;

    // ---- cl_platform_info / cl_device_info queries used for logging ----
    public static final int CL_PLATFORM_NAME = 0x0902;
    public static final int CL_DEVICE_TYPE = 0x1000;
    public static final int CL_DEVICE_VENDOR = 0x102C;
    public static final int CL_DEVICE_NAME = 0x102B;
    public static final int CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;

    // ---- cl_program_build_info ----
    public static final int CL_PROGRAM_BUILD_LOG = 0x1183;
    public static final int CL_PROGRAM_BUILD_STATUS = 0x1181;

    private final Linker linker = Linker.nativeLinker();
    private final SymbolLookup lookup;

    public final MethodHandle clGetPlatformIDs;
    public final MethodHandle clGetPlatformInfo;
    public final MethodHandle clGetDeviceIDs;
    public final MethodHandle clGetDeviceInfo;
    public final MethodHandle clCreateContext;
    public final MethodHandle clCreateCommandQueue;
    public final MethodHandle clCreateProgramWithSource;
    public final MethodHandle clBuildProgram;
    public final MethodHandle clGetProgramBuildInfo;
    public final MethodHandle clCreateKernel;
    public final MethodHandle clSetKernelArg;
    public final MethodHandle clCreateBuffer;
    public final MethodHandle clEnqueueWriteBuffer;
    public final MethodHandle clEnqueueReadBuffer;
    public final MethodHandle clEnqueueCopyBuffer;
    public final MethodHandle clEnqueueNDRangeKernel;
    public final MethodHandle clFinish;
    public final MethodHandle clReleaseMemObject;
    public final MethodHandle clReleaseKernel;
    public final MethodHandle clReleaseProgram;
    public final MethodHandle clReleaseCommandQueue;
    public final MethodHandle clReleaseContext;

    private static List<String> candidateLibraryNames() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return List.of("OpenCL", "OpenCL.dll");
        }
        if (os.contains("mac") || os.contains("darwin")) {
            return List.of("/System/Library/Frameworks/OpenCL.framework/OpenCL", "OpenCL");
        }
        // Same dlopen-is-literal caveat CudaBindings/OpenCLBindings' own
        // javadoc calls out for libcuda: try the versioned ICD-loader name
        // (guaranteed by every distro's ocl-icd/vendor package) before the
        // unversioned/dev-style name.
        return List.of("libOpenCL.so.1", "libOpenCL.so", "OpenCL");
    }

    public OpenCLBindings() {
        List<String> candidates = candidateLibraryNames();
        SymbolLookup resolved = null;
        StringBuilder attempts = new StringBuilder();
        for (String candidate : candidates) {
            try {
                resolved = SymbolLookup.libraryLookup(candidate, Arena.global());
                break;
            } catch (IllegalArgumentException e) {
                attempts.append(candidate).append(": ").append(e.getMessage()).append("; ");
            }
        }
        if (resolved == null) {
            throw new UnsatisfiedLinkError(
                    "Could not load the OpenCL ICD loader. Tried: " + attempts
                    + " -- install an OpenCL ICD loader plus at least one vendor driver "
                    + "(NVIDIA/AMD/Intel GPU driver, or a CPU runtime such as PoCL) for your platform.");
        }
        this.lookup = resolved;

        ValueLayout.OfInt CI = ValueLayout.JAVA_INT;
        ValueLayout.OfLong CL = ValueLayout.JAVA_LONG;
        AddressLayout PTR = ValueLayout.ADDRESS;

        clGetPlatformIDs = downcall("clGetPlatformIDs", FunctionDescriptor.of(CI, CI, PTR, PTR));

        clGetPlatformInfo = downcall("clGetPlatformInfo",
                FunctionDescriptor.of(CI, PTR, CI, CL, PTR, PTR));

        clGetDeviceIDs = downcall("clGetDeviceIDs",
                FunctionDescriptor.of(CI, PTR, CL, CI, PTR, PTR));

        clGetDeviceInfo = downcall("clGetDeviceInfo",
                FunctionDescriptor.of(CI, PTR, CI, CL, PTR, PTR));

        // cl_context clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret)
        clCreateContext = downcall("clCreateContext",
                FunctionDescriptor.of(PTR, PTR, CI, PTR, PTR, PTR, PTR));

        // cl_command_queue clCreateCommandQueue(context, device, properties, errcode_ret)
        // Deprecated in the OpenCL 2.0 headers in favor of
        // clCreateCommandQueueWithProperties, but every ICD loader still
        // exports this symbol for 1.x-era ABI compatibility -- using it
        // directly (rather than the 2.0 variant) keeps this binding
        // working unmodified against OpenCL 1.1/1.2 devices too, which is
        // still common on older GPUs.
        clCreateCommandQueue = downcall("clCreateCommandQueue",
                FunctionDescriptor.of(PTR, PTR, PTR, CL, PTR));

        clCreateProgramWithSource = downcall("clCreateProgramWithSource",
                FunctionDescriptor.of(PTR, PTR, CI, PTR, PTR, PTR));

        clBuildProgram = downcall("clBuildProgram",
                FunctionDescriptor.of(CI, PTR, CI, PTR, PTR, PTR, PTR));

        clGetProgramBuildInfo = downcall("clGetProgramBuildInfo",
                FunctionDescriptor.of(CI, PTR, PTR, CI, CL, PTR, PTR));

        clCreateKernel = downcall("clCreateKernel", FunctionDescriptor.of(PTR, PTR, PTR, PTR));

        clSetKernelArg = downcall("clSetKernelArg",
                FunctionDescriptor.of(CI, PTR, CI, CL, PTR));

        clCreateBuffer = downcall("clCreateBuffer",
                FunctionDescriptor.of(PTR, PTR, CL, CL, PTR, PTR));

        clEnqueueWriteBuffer = downcall("clEnqueueWriteBuffer",
                FunctionDescriptor.of(CI, PTR, PTR, CI, CL, CL, PTR, CI, PTR, PTR));

        clEnqueueReadBuffer = downcall("clEnqueueReadBuffer",
                FunctionDescriptor.of(CI, PTR, PTR, CI, CL, CL, PTR, CI, PTR, PTR));

        clEnqueueCopyBuffer = downcall("clEnqueueCopyBuffer",
                FunctionDescriptor.of(CI, PTR, PTR, PTR, CL, CL, CL, CI, PTR, PTR));

        clEnqueueNDRangeKernel = downcall("clEnqueueNDRangeKernel",
                FunctionDescriptor.of(CI, PTR, PTR, CI, PTR, PTR, PTR, CI, PTR, PTR));

        clFinish = downcall("clFinish", FunctionDescriptor.of(CI, PTR));

        clReleaseMemObject = downcall("clReleaseMemObject", FunctionDescriptor.of(CI, PTR));
        clReleaseKernel = downcall("clReleaseKernel", FunctionDescriptor.of(CI, PTR));
        clReleaseProgram = downcall("clReleaseProgram", FunctionDescriptor.of(CI, PTR));
        clReleaseCommandQueue = downcall("clReleaseCommandQueue", FunctionDescriptor.of(CI, PTR));
        clReleaseContext = downcall("clReleaseContext", FunctionDescriptor.of(CI, PTR));
    }

    private MethodHandle downcall(String symbol, FunctionDescriptor fd) {
        MemorySegment addr = lookup.find(symbol)
                .orElseThrow(() -> new UnsatisfiedLinkError("OpenCL symbol not found: " + symbol));
        return linker.downcallHandle(addr, fd);
    }

    /**
     * Best-effort cl_int -> human string, since the OpenCL spec (unlike
     * NVRTC's nvrtcGetErrorString) defines no runtime error-string call.
     * Covers the codes GpuContext/LlamaLayer can actually hit; unknown
     * codes fall back to the bare numeric value.
     */
    public static String errorString(int code) {
        return switch (code) {
            case 0 -> "CL_SUCCESS";
            case -1 -> "CL_DEVICE_NOT_FOUND";
            case -2 -> "CL_DEVICE_NOT_AVAILABLE";
            case -3 -> "CL_COMPILER_NOT_AVAILABLE";
            case -4 -> "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case -5 -> "CL_OUT_OF_RESOURCES";
            case -6 -> "CL_OUT_OF_HOST_MEMORY";
            case -11 -> "CL_BUILD_PROGRAM_FAILURE";
            case -12 -> "CL_MAP_FAILURE";
            case -30 -> "CL_INVALID_VALUE";
            case -31 -> "CL_INVALID_DEVICE_TYPE";
            case -32 -> "CL_INVALID_PLATFORM";
            case -33 -> "CL_INVALID_DEVICE";
            case -34 -> "CL_INVALID_CONTEXT";
            case -35 -> "CL_INVALID_QUEUE_PROPERTIES";
            case -36 -> "CL_INVALID_COMMAND_QUEUE";
            case -38 -> "CL_INVALID_MEM_OBJECT";
            case -40 -> "CL_INVALID_IMAGE_SIZE";
            case -44 -> "CL_INVALID_PROGRAM";
            case -45 -> "CL_INVALID_PROGRAM_EXECUTABLE";
            case -46 -> "CL_INVALID_KERNEL_NAME";
            case -47 -> "CL_INVALID_KERNEL_DEFINITION";
            case -48 -> "CL_INVALID_KERNEL";
            case -49 -> "CL_INVALID_ARG_INDEX";
            case -50 -> "CL_INVALID_ARG_VALUE";
            case -51 -> "CL_INVALID_ARG_SIZE";
            case -52 -> "CL_INVALID_KERNEL_ARGS";
            case -53 -> "CL_INVALID_WORK_DIMENSION";
            case -54 -> "CL_INVALID_WORK_GROUP_SIZE";
            case -55 -> "CL_INVALID_WORK_ITEM_SIZE";
            case -56 -> "CL_INVALID_GLOBAL_OFFSET";
            case -57 -> "CL_INVALID_EVENT_WAIT_LIST";
            case -59 -> "CL_INVALID_KERNEL_ARGS_MISSING"; // not standard, placeholder guard
            case -63 -> "CL_INVALID_GLOBAL_WORK_SIZE";
            default -> "cl_int(" + code + ")";
        };
    }
}