package com.github.gbenroscience.gpu.opencl;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Raw FFM downcalls into the platform's OpenCL ICD loader
 * (libOpenCL.so on Linux, OpenCL.dll on Windows, OpenCL.framework on macOS
 * -- macOS needs the framework path variant, see {@link #libraryName()}).
 *
 * This is deliberately NOT a full OpenCL binding -- only the calls the
 * generic interpreter kernel path needs: platform/device discovery,
 * context/queue/program/kernel lifecycle, buffer create/write/read, and
 * NDRange dispatch. No JAR, no JNI stub you compile yourself -- the only
 * "native code" here is the driver every OpenCL-capable machine already
 * ships.
 *
 * Every handle-returning call follows the OpenCL C convention of writing an
 * int error code through an out-pointer; those out-pointers are plain
 * MemorySegments allocated from a caller-supplied Arena, never Java arrays,
 * consistent with the zero-alloc-on-hot-path rule (bootstrap/compile-time
 * calls here allocate; the hot apply() path later does not).
 */
public final class OpenClBindings {

    // --- Common OpenCL status/type codes this scaffold cares about ---
    public static final int CL_SUCCESS = 0;
    public static final int CL_DEVICE_NOT_FOUND = -1; // returned by clGetDeviceIDs when a platform has zero matching devices -- not a real error during enumeration
    public static final int CL_DEVICE_TYPE_GPU = 1 << 2;
    public static final int CL_PROGRAM_BUILD_LOG = 0x1183;
    public static final int CL_MEM_READ_ONLY = 1 << 2;
    public static final int CL_MEM_WRITE_ONLY = 1 << 1;
    public static final int CL_MEM_READ_WRITE = 1 << 0;
    public static final int CL_TRUE = 1;
    public static final int CL_FALSE = 0;

    // cl_platform_info / cl_device_info query codes used for identifying
    // WHICH platform/device is which (needed to distinguish an Intel GPU
    // from an AMD GPU when both are installed -- see
    // OpenClCompositeExpression's device-selection logic).
    public static final int CL_PLATFORM_NAME = 0x0902;
    public static final int CL_PLATFORM_VENDOR = 0x0903;
    public static final int CL_DEVICE_NAME = 0x102B;
    public static final int CL_DEVICE_VENDOR = 0x102C;

    private final Linker linker = Linker.nativeLinker();
    private final SymbolLookup lookup;

    public final MethodHandle clGetPlatformIDs;
    public final MethodHandle clGetPlatformInfo;
    public final MethodHandle clGetDeviceIDs;
    public final MethodHandle clGetDeviceInfo;
    public final MethodHandle clCreateContext;
    public final MethodHandle clCreateCommandQueue; // OpenCL 1.2 form: universally supported, incl. macOS
    public final MethodHandle clCreateBuffer;
    public final MethodHandle clCreateProgramWithSource;
    public final MethodHandle clBuildProgram;
    public final MethodHandle clGetProgramBuildInfo;
    public final MethodHandle clCreateKernel;
    public final MethodHandle clSetKernelArg;
    public final MethodHandle clEnqueueWriteBuffer;
    public final MethodHandle clEnqueueReadBuffer;
    public final MethodHandle clEnqueueNDRangeKernel;
    public final MethodHandle clFinish;
    public final MethodHandle clReleaseMemObject;
    public final MethodHandle clReleaseKernel;
    public final MethodHandle clReleaseProgram;
    public final MethodHandle clReleaseCommandQueue;
    public final MethodHandle clReleaseContext;

    private static String libraryName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return "OpenCL";
        }
        if (os.contains("mac")) {
            // Frameworks aren't found via the normal .so/.dll search path;
            // this needs SymbolLookup against the framework's actual binary,
            // typically /System/Library/Frameworks/OpenCL.framework/OpenCL.
            // (Apple also deprecated OpenCL in favor of Metal -- it still
            // works today, but this path is the first thing to re-check if
            // a future macOS removes it. The Metal backend is the intended
            // long-term replacement on Apple hardware, not this loader.)
            return "/System/Library/Frameworks/OpenCL.framework/OpenCL";
        }
        return "OpenCL"; // libOpenCL.so via standard linker search / ldconfig
    }

    public OpenClBindings() {
        this.lookup = SymbolLookup.libraryLookup(libraryName(), Arena.global());

        ValueLayout.OfInt CI = ValueLayout.JAVA_INT;
        ValueLayout.OfLong CL = ValueLayout.JAVA_LONG;
        AddressLayout PTR = ValueLayout.ADDRESS;

        clGetPlatformIDs = downcall("clGetPlatformIDs",
                FunctionDescriptor.of(CI, CI, PTR, PTR));

        // (platform, param_name, param_value_size, param_value, param_value_size_ret)
        clGetPlatformInfo = downcall("clGetPlatformInfo",
                FunctionDescriptor.of(CI, PTR, CI, CL, PTR, PTR));

        // device_type is cl_device_type == cl_bitfield == cl_ulong (8 bytes), NOT cl_uint.
        clGetDeviceIDs = downcall("clGetDeviceIDs",
                FunctionDescriptor.of(CI, PTR, CL, CI, PTR, PTR));

        // (device, param_name, param_value_size, param_value, param_value_size_ret)
        clGetDeviceInfo = downcall("clGetDeviceInfo",
                FunctionDescriptor.of(CI, PTR, CI, CL, PTR, PTR));

        clCreateContext = downcall("clCreateContext",
                FunctionDescriptor.of(PTR, PTR, CI, PTR, PTR, PTR, PTR));

        clCreateCommandQueue = downcall("clCreateCommandQueue",
                FunctionDescriptor.of(PTR, PTR, PTR, CL, PTR));

        clCreateBuffer = downcall("clCreateBuffer",
                FunctionDescriptor.of(PTR, PTR, CL, CL, PTR, PTR));

        clCreateProgramWithSource = downcall("clCreateProgramWithSource",
                FunctionDescriptor.of(PTR, PTR, CI, PTR, PTR, PTR));

        clBuildProgram = downcall("clBuildProgram",
                FunctionDescriptor.of(CI, PTR, CI, PTR, PTR, PTR, PTR));

        clGetProgramBuildInfo = downcall("clGetProgramBuildInfo",
                FunctionDescriptor.of(CI, PTR, PTR, CI, CL, PTR, PTR));

        clCreateKernel = downcall("clCreateKernel",
                FunctionDescriptor.of(PTR, PTR, PTR, PTR));

        clSetKernelArg = downcall("clSetKernelArg",
                FunctionDescriptor.of(CI, PTR, CI, CL, PTR));

        clEnqueueWriteBuffer = downcall("clEnqueueWriteBuffer",
                FunctionDescriptor.of(CI, PTR, PTR, CI, CL, CL, PTR, CI, PTR, PTR));

        clEnqueueReadBuffer = downcall("clEnqueueReadBuffer",
                FunctionDescriptor.of(CI, PTR, PTR, CI, CL, CL, PTR, CI, PTR, PTR));

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
}