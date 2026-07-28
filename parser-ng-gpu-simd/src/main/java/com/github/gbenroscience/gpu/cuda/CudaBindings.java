package com.github.gbenroscience.gpu.cuda;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.List;

/**
 * Raw FFM downcalls into the CUDA Driver API (libcuda.so.1 on Linux,
 * nvcuda.dll on Windows -- this is the driver-shipped library, NOT the CUDA
 * Toolkit's libcudart; no toolkit installation is required on the target
 * machine, only an NVIDIA driver).
 *
 * Only the calls the generic interpreter kernel path needs: init, device
 * discovery, primary-context retain, module load (from NVRTC-compiled PTX
 * -- see NvrtcBindings), buffer alloc/copy, and kernel launch.
 *
 * Mirrors OpenCLBindings' shape and its resilient-library-lookup fix: on a
 * stock Linux box the NVIDIA driver installer usually DOES create the
 * unversioned libcuda.so symlink (unlike the OpenCL ICD loader packages),
 * but we still try the versioned name first since ldconfig always knows
 * about that one regardless of how the driver was packaged.
 */
public final class CudaBindings {

    public static final int CUDA_SUCCESS = 0;

    // CUdevice_attribute values needed to pick an NVRTC --gpu-architecture
    // option that matches the actual device (see cuda.h).
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    private final Linker linker = Linker.nativeLinker();
    private final SymbolLookup lookup;

    public final MethodHandle cuInit;
    public final MethodHandle cuDeviceGetCount;
    public final MethodHandle cuDeviceGet;
    public final MethodHandle cuDeviceGetAttribute;
    public final MethodHandle cuDevicePrimaryCtxRetain;
    public final MethodHandle cuDevicePrimaryCtxRelease;
    public final MethodHandle cuCtxSetCurrent;
    public final MethodHandle cuCtxSynchronize;
    public final MethodHandle cuModuleLoadData;
    public final MethodHandle cuModuleGetFunction;
    public final MethodHandle cuModuleUnload;
    public final MethodHandle cuMemAlloc;
    public final MethodHandle cuMemFree;
    public final MethodHandle cuMemcpyHtoD;
    public final MethodHandle cuMemcpyDtoH;
    public final MethodHandle cuLaunchKernel;
    public final MethodHandle cuGetErrorString;

    private static List<String> candidateLibraryNames() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return List.of("nvcuda", "nvcuda.dll");
        }
        // Same dlopen-is-literal caveat as OpenCLBindings: try the
        // versioned name (what ldconfig/the driver package guarantees)
        // before the unversioned/dev-style name.
        return List.of("libcuda.so.1", "libcuda.so", "cuda");
    }

    public CudaBindings() {
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
                    "Could not load the CUDA driver library. Tried: " + attempts
                    + " -- an NVIDIA GPU driver must be installed (this links against "
                    + "the driver's libcuda/nvcuda, not the CUDA Toolkit runtime).");
        }
        this.lookup = resolved;

        ValueLayout.OfInt CI = ValueLayout.JAVA_INT;
        ValueLayout.OfLong CL = ValueLayout.JAVA_LONG;
        AddressLayout PTR = ValueLayout.ADDRESS;

        cuInit = downcall("cuInit", FunctionDescriptor.of(CI, CI));

        cuDeviceGetCount = downcall("cuDeviceGetCount", FunctionDescriptor.of(CI, PTR));

        cuDeviceGet = downcall("cuDeviceGet", FunctionDescriptor.of(CI, PTR, CI));

        cuDeviceGetAttribute = downcall("cuDeviceGetAttribute",
                FunctionDescriptor.of(CI, PTR, CI, CI));

        cuDevicePrimaryCtxRetain = downcall("cuDevicePrimaryCtxRetain",
                FunctionDescriptor.of(CI, PTR, CI));

        cuDevicePrimaryCtxRelease = downcall("cuDevicePrimaryCtxRelease",
                FunctionDescriptor.of(CI, CI));

        cuCtxSetCurrent = downcall("cuCtxSetCurrent", FunctionDescriptor.of(CI, PTR));

        cuCtxSynchronize = downcall("cuCtxSynchronize", FunctionDescriptor.of(CI));

        // Simpler sibling of cuModuleLoadDataEx -- no JIT options needed
        // for a plain PTX/cubin image loaded from memory.
        cuModuleLoadData = downcall("cuModuleLoadData", FunctionDescriptor.of(CI, PTR, PTR));

        cuModuleGetFunction = downcall("cuModuleGetFunction",
                FunctionDescriptor.of(CI, PTR, PTR, PTR));

        cuModuleUnload = downcall("cuModuleUnload", FunctionDescriptor.of(CI, PTR));

        // CUdeviceptr is an 8-byte unsigned handle (not a host pointer);
        // cuMemAlloc writes it out through a pointer-to-long, cuMemFree/
        // cuMemcpyHtoD/cuMemcpyDtoH take it BY VALUE as a JAVA_LONG.
        // NOTE: cuda.h's cuMemAlloc/cuMemFree/cuMemcpyHtoD/cuMemcpyDtoH are
        // preprocessor macros that alias to the *_v2 symbols -- since we're
        // doing raw dynamic symbol lookup (no header, no macro expansion),
        // we have to ask for the real exported names directly.
        cuMemAlloc = downcall("cuMemAlloc_v2", FunctionDescriptor.of(CI, PTR, CL));

        cuMemFree = downcall("cuMemFree_v2", FunctionDescriptor.of(CI, CL));

        cuMemcpyHtoD = downcall("cuMemcpyHtoD_v2", FunctionDescriptor.of(CI, CL, PTR, CL));

        cuMemcpyDtoH = downcall("cuMemcpyDtoH_v2", FunctionDescriptor.of(CI, PTR, CL, CL));

        cuLaunchKernel = downcall("cuLaunchKernel",
                FunctionDescriptor.of(CI, PTR, CI, CI, CI, CI, CI, CI, CI, PTR, PTR, PTR));

        cuGetErrorString = downcall("cuGetErrorString", FunctionDescriptor.of(CI, CI, PTR));
    }

    private MethodHandle downcall(String symbol, FunctionDescriptor fd) {
        MemorySegment addr = lookup.find(symbol)
                .orElseThrow(() -> new UnsatisfiedLinkError("CUDA driver symbol not found: " + symbol));
        return linker.downcallHandle(addr, fd);
    }
}