package com.github.gbenroscience.gpu.cuda;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.List;

/**
 * Raw FFM downcalls into NVRTC (NVIDIA's runtime compiler). The CUDA driver
 * API only loads pre-compiled PTX/cubin -- it has no equivalent of OpenCL's
 * clBuildProgram that takes source text directly. NVRTC is what closes that
 * gap: it compiles CudaKernelSource.CUDA_SOURCE to PTX at process startup,
 * the same "ship source, compile once, run everywhere" shape as the OpenCL
 * path, just via a second native library instead of the driver alone.
 *
 * Library name is the awkward part here: on Linux it's typically
 * "libnvrtc.so.<version>" with an unversioned "libnvrtc.so" symlink only if
 * the -dev package is present (same class of problem OpenCLBindings hit).
 * On Windows the DLL name embeds the CUDA version, e.g.
 * "nvrtc64_120_0.dll" for CUDA 12.0 -- there's no way to guess that
 * reliably across installs, so this tries a short list of common versions
 * and also honors a "cuda.nvrtc.library" system property so a caller can
 * pin the exact name for their environment.
 */
public final class NvrtcBindings {

    public static final int NVRTC_SUCCESS = 0;

    private final Linker linker = Linker.nativeLinker();
    private final SymbolLookup lookup;

    public final MethodHandle nvrtcCreateProgram;
    public final MethodHandle nvrtcCompileProgram;
    public final MethodHandle nvrtcGetProgramLogSize;
    public final MethodHandle nvrtcGetProgramLog;
    public final MethodHandle nvrtcGetPTXSize;
    public final MethodHandle nvrtcGetPTX;
    public final MethodHandle nvrtcDestroyProgram;
    public final MethodHandle nvrtcGetErrorString;

    private static List<String> candidateLibraryNames() {
        List<String> candidates = new ArrayList<>();
        String override = System.getProperty("cuda.nvrtc.library");
        if (override != null && !override.isBlank()) {
            candidates.add(override);
        }
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            // Most likely current versions first; not exhaustive -- set
            // -Dcuda.nvrtc.library=nvrtc64_XXX_0 if your install isn't here.
            candidates.addAll(List.of(
                    "nvrtc64_120_0", "nvrtc64_121_0", "nvrtc64_122_0",
                    "nvrtc64_110_0", "nvrtc64_112_0",
                    "nvrtc64_102_0", "nvrtc"));
        } else {
            candidates.addAll(List.of(
                    "libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11", "nvrtc"));
        }
        return candidates;
    }

    public NvrtcBindings() {
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
                    "Could not load NVRTC. Tried: " + attempts
                    + " -- install the CUDA Toolkit (NVRTC ships with it, not with the "
                    + "driver alone), or set -Dcuda.nvrtc.library=<exact name> for your version.");
        }
        this.lookup = resolved;

        ValueLayout.OfInt CI = ValueLayout.JAVA_INT;
        AddressLayout PTR = ValueLayout.ADDRESS;

        nvrtcCreateProgram = downcall("nvrtcCreateProgram",
                FunctionDescriptor.of(CI, PTR, PTR, PTR, CI, PTR, PTR));

        nvrtcCompileProgram = downcall("nvrtcCompileProgram",
                FunctionDescriptor.of(CI, PTR, CI, PTR));

        nvrtcGetProgramLogSize = downcall("nvrtcGetProgramLogSize",
                FunctionDescriptor.of(CI, PTR, PTR));

        nvrtcGetProgramLog = downcall("nvrtcGetProgramLog",
                FunctionDescriptor.of(CI, PTR, PTR));

        nvrtcGetPTXSize = downcall("nvrtcGetPTXSize",
                FunctionDescriptor.of(CI, PTR, PTR));

        nvrtcGetPTX = downcall("nvrtcGetPTX",
                FunctionDescriptor.of(CI, PTR, PTR));

        nvrtcDestroyProgram = downcall("nvrtcDestroyProgram",
                FunctionDescriptor.of(CI, PTR));

        // Unlike every CUresult-returning driver call, this one returns the
        // message pointer directly rather than writing it through an
        // out-param -- no CUresult-style wrapper here.
        nvrtcGetErrorString = downcall("nvrtcGetErrorString",
                FunctionDescriptor.of(PTR, CI));
    }

    private MethodHandle downcall(String symbol, FunctionDescriptor fd) {
        MemorySegment addr = lookup.find(symbol)
                .orElseThrow(() -> new UnsatisfiedLinkError("NVRTC symbol not found: " + symbol));
        return linker.downcallHandle(addr, fd);
    }
}