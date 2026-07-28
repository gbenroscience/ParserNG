package com.github.gbenroscience.gpu.cuda;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import com.github.gbenroscience.gpu.GpuCompositeExpression;

/**
 * CUDA-backed evaluator for a single compiled expression -- the CUDA
 * counterpart of com.github.gbenroscience.gpu.opencl.OpenClCompositeExpression.
 * Native dual precision: ONE NVRTC compile of CudaKernelSource.CUDA_SOURCE
 * produces a single PTX module containing BOTH entry points ("interpret"
 * for double, "interpretF32" for float); cuModuleGetFunction pulls each
 * out of that one module. Every double-path call flows through the double
 * function with double device buffers throughout; every float-path call
 * flows through the float function with float device buffers throughout.
 * No cross-precision conversion happens anywhere in either call path --
 * that's what makes the float path a genuine throughput win (half the
 * PCIe/memory traffic per element, full-rate execution even on consumer
 * GPUs where fp64 throughput is deliberately capped well below fp32).
 *
 * Structural differences from the OpenCL version, all forced by the CUDA
 * driver API's shape rather than by choice:
 *
 * - Device buffers are CUdeviceptr values (plain 8-byte handles), not
 *   MemorySegments -- CUDA device memory isn't host-addressable the way an
 *   OpenCL cl_mem's underlying MemorySegment reference is treated here, so
 *   opcodesDevice/inputDevice/outputDevice/etc. are `long`, not MemorySegment.
 *
 * - Compilation is a two-stage NVRTC-then-driver process (source -> PTX via
 *   NvrtcBindings, PTX -> loaded module via CudaBindings.cuModuleLoadData)
 *   instead of OpenCL's single clBuildProgram call.
 *
 * - Kernel arguments are passed as a `void** kernelParams` array built by
 *   hand (each element points at a small buffer holding that argument's
 *   value) rather than OpenCL's one-arg-at-a-time clSetKernelArg.
 *
 * - CUDA contexts are per-thread (a thread must have a context "current"
 *   before it can call most driver functions). This uses the *primary*
 *   context (cuDevicePrimaryCtxRetain) rather than an explicitly created
 *   one specifically so it can be shared: cuCtxSetCurrent(sameContext) is
 *   cheap and thread-safe to call from any thread, unlike juggling
 *   independently-created contexts across threads.
 */
public final class CudaCompositeExpression implements GpuCompositeExpression {

    private static final class CudaContext {

        static final CudaBindings CU = new CudaBindings();
        static final NvrtcBindings NVRTC = new NvrtcBindings();

        static final int DEVICE;
        static final MemorySegment CONTEXT;   // CUcontext (opaque handle)
        static final MemorySegment MODULE;    // CUmodule -- holds BOTH kernels
        static final MemorySegment FUNCTION_F64; // CUfunction "interpret"
        static final MemorySegment FUNCTION_F32; // CUfunction "interpretF32"

        static {
            try (Arena bootstrap = Arena.ofConfined()) {
                check((int) CU.cuInit.invoke(0), "cuInit");

                MemorySegment countBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
                check((int) CU.cuDeviceGetCount.invoke(countBuf), "cuDeviceGetCount");
                if (countBuf.get(ValueLayout.JAVA_INT, 0) < 1) {
                    throw new IllegalStateException("No CUDA devices found");
                }

                int deviceIndex = Integer.getInteger("cuda.device.index", 0);
                MemorySegment deviceBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
                check((int) CU.cuDeviceGet.invoke(deviceBuf, deviceIndex), "cuDeviceGet");
                DEVICE = deviceBuf.get(ValueLayout.JAVA_INT, 0);

                MemorySegment majorBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
                MemorySegment minorBuf = bootstrap.allocate(ValueLayout.JAVA_INT);
                check((int) CU.cuDeviceGetAttribute.invoke(majorBuf,
                        CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, DEVICE),
                        "cuDeviceGetAttribute(major)");
                check((int) CU.cuDeviceGetAttribute.invoke(minorBuf,
                        CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, DEVICE),
                        "cuDeviceGetAttribute(minor)");
                int major = majorBuf.get(ValueLayout.JAVA_INT, 0);
                int minor = minorBuf.get(ValueLayout.JAVA_INT, 0);

                MemorySegment ctxBuf = bootstrap.allocate(ValueLayout.ADDRESS);
                check((int) CU.cuDevicePrimaryCtxRetain.invoke(ctxBuf, DEVICE),
                        "cuDevicePrimaryCtxRetain");
                CONTEXT = ctxBuf.get(ValueLayout.ADDRESS, 0);
                check((int) CU.cuCtxSetCurrent.invoke(CONTEXT), "cuCtxSetCurrent");

                // ONE NVRTC compile -- the resulting PTX module contains
                // BOTH "interpret" and "interpretF32" (see CudaKernelSource).
                String ptx = compileToPtx(bootstrap, major, minor);

                MemorySegment ptxSrc = bootstrap.allocateFrom(ptx);
                MemorySegment moduleBuf = bootstrap.allocate(ValueLayout.ADDRESS);
                check((int) CU.cuModuleLoadData.invoke(moduleBuf, ptxSrc), "cuModuleLoadData");
                MODULE = moduleBuf.get(ValueLayout.ADDRESS, 0);

                MemorySegment fnNameF64 = bootstrap.allocateFrom(
                        CudaKernelSource.KERNEL_NAME_F64, StandardCharsets.UTF_8);
                MemorySegment fnBufF64 = bootstrap.allocate(ValueLayout.ADDRESS);
                check((int) CU.cuModuleGetFunction.invoke(fnBufF64, MODULE, fnNameF64),
                        "cuModuleGetFunction(interpret)");
                FUNCTION_F64 = fnBufF64.get(ValueLayout.ADDRESS, 0);

                MemorySegment fnNameF32 = bootstrap.allocateFrom(
                        CudaKernelSource.KERNEL_NAME_F32, StandardCharsets.UTF_8);
                MemorySegment fnBufF32 = bootstrap.allocate(ValueLayout.ADDRESS);
                check((int) CU.cuModuleGetFunction.invoke(fnBufF32, MODULE, fnNameF32),
                        "cuModuleGetFunction(interpretF32)");
                FUNCTION_F32 = fnBufF32.get(ValueLayout.ADDRESS, 0);

            } catch (Throwable t) {
                throw new ExceptionInInitializerError(t);
            }
        }

        private static String compileToPtx(Arena arena, int major, int minor) throws Throwable {
            MemorySegment src = arena.allocateFrom(CudaKernelSource.CUDA_SOURCE, StandardCharsets.UTF_8);
            MemorySegment name = arena.allocateFrom("interpreter_kernel.cu", StandardCharsets.UTF_8);

            MemorySegment progBuf = arena.allocate(ValueLayout.ADDRESS);
            checkNvrtc((int) NVRTC.nvrtcCreateProgram.invoke(progBuf, src, name, 0,
                    MemorySegment.NULL, MemorySegment.NULL), "nvrtcCreateProgram");
            MemorySegment program = progBuf.get(ValueLayout.ADDRESS, 0);

            MemorySegment archOpt = arena.allocateFrom(
                    "--gpu-architecture=compute_" + major + minor, StandardCharsets.UTF_8);
            MemorySegment optionsArr = arena.allocate(ValueLayout.ADDRESS, 1);
            optionsArr.setAtIndex(ValueLayout.ADDRESS, 0, archOpt);

            int compileStatus = (int) NVRTC.nvrtcCompileProgram.invoke(program, 1, optionsArr);
            if (compileStatus != NvrtcBindings.NVRTC_SUCCESS) {
                throw new IllegalStateException(
                        "NVRTC compile failed (" + compileStatus + "): " + fetchCompileLog(arena, program));
            }

            MemorySegment ptxSizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
            checkNvrtc((int) NVRTC.nvrtcGetPTXSize.invoke(program, ptxSizeBuf), "nvrtcGetPTXSize");
            long ptxSize = ptxSizeBuf.get(ValueLayout.JAVA_LONG, 0);

            MemorySegment ptxBuf = arena.allocate(ptxSize);
            checkNvrtc((int) NVRTC.nvrtcGetPTX.invoke(program, ptxBuf), "nvrtcGetPTX");

            checkNvrtc((int) NVRTC.nvrtcDestroyProgram.invoke(progBuf), "nvrtcDestroyProgram");

            return ptxBuf.getString(0, StandardCharsets.UTF_8);
        }

        private static String fetchCompileLog(Arena arena, MemorySegment program) throws Throwable {
            MemorySegment logSizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
            NVRTC.nvrtcGetProgramLogSize.invoke(program, logSizeBuf);
            long logSize = logSizeBuf.get(ValueLayout.JAVA_LONG, 0);
            if (logSize <= 1) {
                return "(no compile log)";
            }
            MemorySegment logBuf = arena.allocate(logSize);
            NVRTC.nvrtcGetProgramLog.invoke(program, logBuf);
            return logBuf.getString(0, StandardCharsets.UTF_8);
        }

        private static void check(int status, String call) {
            if (status != CudaBindings.CUDA_SUCCESS) {
                throw new IllegalStateException("CUDA error in " + call + ": code " + status);
            }
        }

        private static void checkNvrtc(int status, String call) {
            if (status != NvrtcBindings.NVRTC_SUCCESS) {
                throw new IllegalStateException("NVRTC error in " + call + ": code " + status);
            }
        }
    }

    // CudaContext.FUNCTION_F64/FUNCTION_F32/CONTEXT are static singletons
    // shared by every instance and thread. Setting up kernelParams and
    // launching against a shared function is a check-then-act sequence
    // that must be serialized -- same rationale as the OpenCL path's
    // DISPATCH_LOCK. One lock covers both precisions (see that class's
    // javadoc for the same tradeoff note): simpler, at the cost of
    // serializing double- and float-path dispatches against each other too.
    private static final Object DISPATCH_LOCK = new Object();

    private static final int DEFAULT_BLOCK_SIZE = 256;

    private final CudaBindings cu = CudaContext.CU;

    private final Arena arena = Arena.ofShared();
    private final long opcodesDevice;
    private final long targetSlotsDevice;
    private final long literalConstantsDevice;    // double, feeds FUNCTION_F64
    private final long literalConstantsDeviceF32;  // float, feeds FUNCTION_F32
    private final int instructionCount;
    private final int varCount;

    // ---- double-path device/staging state ----
    private long inputDevice = 0L;
    private long outputDevice = 0L;
    private MemorySegment stagingIn = MemorySegment.NULL;
    private MemorySegment stagingOut = MemorySegment.NULL;
    private long deviceInCapacityBytes = 0;
    private long deviceOutCapacityBytes = 0;
    private long stagingInCapacityBytes = 0;
    private long stagingOutCapacityBytes = 0;

    // ---- float-path device/staging state (fully independent of the double
    // ---- path above -- separate buffers, separate capacities, never
    // ---- shared or resized together) ----
    private long inputDeviceF32 = 0L;
    private long outputDeviceF32 = 0L;
    private MemorySegment stagingInF32 = MemorySegment.NULL;
    private MemorySegment stagingOutF32 = MemorySegment.NULL;
    private long deviceInCapacityBytesF32 = 0;
    private long deviceOutCapacityBytesF32 = 0;
    private long stagingInCapacityBytesF32 = 0;
    private long stagingOutCapacityBytesF32 = 0;

    public CudaCompositeExpression(int[] opcodes, int[] targetSlots, double[] literalConstants,
            int instructionCount, int varCount) {
        this.instructionCount = instructionCount;
        this.varCount = varCount;

        try {
            this.opcodesDevice = uploadIntArray(opcodes);
            this.targetSlotsDevice = uploadIntArray(targetSlots);
            this.literalConstantsDevice = uploadDoubleArray(literalConstants);

            // Converted ONCE, at construction, from the same source values
            // the double path uses -- not recomputed per call. The float
            // function's literalConstants buffer is genuinely float32 on
            // device from here on.
            float[] literalConstantsF32 = new float[literalConstants.length];
            for (int i = 0; i < literalConstants.length; i++) {
                literalConstantsF32[i] = (float) literalConstants[i];
            }
            this.literalConstantsDeviceF32 = uploadFloatArray(literalConstantsF32);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to upload GPU program buffers", t);
        }
    }

    // ================= double precision path (unchanged) =================

    public void applyBulk(MemorySegment in, MemorySegment out) throws Throwable {
        long dataSize = out.byteSize() / ValueLayout.JAVA_DOUBLE.byteSize();
        dispatch(in, out, (int) dataSize);
    }

    public void applyBulk(double[] in, double[] out) throws Throwable {
        ensureStaging(in.length, out.length);
        MemorySegment.copy(in, 0, stagingIn, ValueLayout.JAVA_DOUBLE, 0, in.length);
        MemorySegment inSlice = stagingIn.asSlice(0, (long) in.length * ValueLayout.JAVA_DOUBLE.byteSize());
        MemorySegment outSlice = stagingOut.asSlice(0, (long) out.length * ValueLayout.JAVA_DOUBLE.byteSize());
        dispatch(inSlice, outSlice, out.length);
        MemorySegment.copy(outSlice, ValueLayout.JAVA_DOUBLE, 0, out, 0, out.length);
    }

    public void applyBulk(double[][] in, double[] out) throws Throwable {
        if (in.length != varCount) {
            throw new IllegalArgumentException(
                    "Expected " + varCount + " variable rows, got " + in.length);
        }
        int dataSize = out.length;
        int flatLen = varCount * dataSize;
        ensureStaging(flatLen, dataSize);

        for (int slot = 0; slot < in.length; slot++) {
            MemorySegment.copy(in[slot], 0, stagingIn, ValueLayout.JAVA_DOUBLE,
                    (long) slot * dataSize * ValueLayout.JAVA_DOUBLE.byteSize(), dataSize);
        }
        MemorySegment inSlice = stagingIn.asSlice(0, (long) flatLen * ValueLayout.JAVA_DOUBLE.byteSize());
        MemorySegment outSlice = stagingOut.asSlice(0, (long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize());
        dispatch(inSlice, outSlice, dataSize);
        MemorySegment.copy(outSlice, ValueLayout.JAVA_DOUBLE, 0, out, 0, out.length);
    }

    private void dispatch(MemorySegment in, MemorySegment out, int dataSize) throws Throwable {
        synchronized (DISPATCH_LOCK) {
            ensureDeviceBuffers(in.byteSize(), out.byteSize());

            try {
                check((int) cu.cuCtxSetCurrent.invoke(CudaContext.CONTEXT), "cuCtxSetCurrent");

                check((int) cu.cuMemcpyHtoD.invoke(inputDevice, in, in.byteSize()), "cuMemcpyHtoD(in)");

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment kernelParams = buildKernelParams(tmp,
                            literalConstantsDevice, inputDevice, outputDevice, dataSize);

                    int gridDimX = (dataSize + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
                    check((int) cu.cuLaunchKernel.invoke(
                            CudaContext.FUNCTION_F64,
                            gridDimX, 1, 1,
                            DEFAULT_BLOCK_SIZE, 1, 1,
                            0,
                            MemorySegment.NULL,   // default stream
                            kernelParams,
                            MemorySegment.NULL),
                            "cuLaunchKernel(interpret)");
                }

                check((int) cu.cuCtxSynchronize.invoke(), "cuCtxSynchronize");

                check((int) cu.cuMemcpyDtoH.invoke(out, outputDevice, out.byteSize()), "cuMemcpyDtoH(out)");
            } catch (Throwable t) {
                throw new RuntimeException("GPU dispatch failed", t);
            }
        }
    }

    // ================= float32 precision path (native, not bridged) =================

    @Override
    public void applyBulkF32(MemorySegment in, MemorySegment out) throws Throwable {
        long dataSize = out.byteSize() / ValueLayout.JAVA_FLOAT.byteSize();
        dispatchF32(in, out, (int) dataSize);
    }

    @Override
    public void applyBulk(float[] in, float[] out) throws Throwable {
        ensureStagingF32(in.length, out.length);
        MemorySegment.copy(in, 0, stagingInF32, ValueLayout.JAVA_FLOAT, 0, in.length);
        MemorySegment inSlice = stagingInF32.asSlice(0, (long) in.length * ValueLayout.JAVA_FLOAT.byteSize());
        MemorySegment outSlice = stagingOutF32.asSlice(0, (long) out.length * ValueLayout.JAVA_FLOAT.byteSize());
        dispatchF32(inSlice, outSlice, out.length);
        MemorySegment.copy(outSlice, ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
    }

    @Override
    public void applyBulk(float[][] in, float[] out) throws Throwable {
        if (in.length != varCount) {
            throw new IllegalArgumentException(
                    "Expected " + varCount + " variable rows, got " + in.length);
        }
        int dataSize = out.length;
        int flatLen = varCount * dataSize;
        ensureStagingF32(flatLen, dataSize);

        for (int slot = 0; slot < in.length; slot++) {
            MemorySegment.copy(in[slot], 0, stagingInF32, ValueLayout.JAVA_FLOAT,
                    (long) slot * dataSize * ValueLayout.JAVA_FLOAT.byteSize(), dataSize);
        }
        MemorySegment inSlice = stagingInF32.asSlice(0, (long) flatLen * ValueLayout.JAVA_FLOAT.byteSize());
        MemorySegment outSlice = stagingOutF32.asSlice(0, (long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
        dispatchF32(inSlice, outSlice, dataSize);
        MemorySegment.copy(outSlice, ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
    }

    private void dispatchF32(MemorySegment in, MemorySegment out, int dataSize) throws Throwable {
        synchronized (DISPATCH_LOCK) {
            ensureDeviceBuffersF32(in.byteSize(), out.byteSize());

            try {
                check((int) cu.cuCtxSetCurrent.invoke(CudaContext.CONTEXT), "cuCtxSetCurrent");

                check((int) cu.cuMemcpyHtoD.invoke(inputDeviceF32, in, in.byteSize()), "cuMemcpyHtoD(in, f32)");

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment kernelParams = buildKernelParams(tmp,
                            literalConstantsDeviceF32, inputDeviceF32, outputDeviceF32, dataSize);

                    int gridDimX = (dataSize + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
                    check((int) cu.cuLaunchKernel.invoke(
                            CudaContext.FUNCTION_F32,
                            gridDimX, 1, 1,
                            DEFAULT_BLOCK_SIZE, 1, 1,
                            0,
                            MemorySegment.NULL,   // default stream
                            kernelParams,
                            MemorySegment.NULL),
                            "cuLaunchKernel(interpretF32)");
                }

                check((int) cu.cuCtxSynchronize.invoke(), "cuCtxSynchronize");

                check((int) cu.cuMemcpyDtoH.invoke(out, outputDeviceF32, out.byteSize()), "cuMemcpyDtoH(out, f32)");
            } catch (Throwable t) {
                throw new RuntimeException("GPU dispatch failed (f32)", t);
            }
        }
    }

    /**
     * Builds the {@code void** kernelParams} array cuLaunchKernel expects:
     * one pointer per argument, each pointing at a small buffer holding
     * that argument's actual value. Kernel signature/order (must match
     * CudaKernelSource.CUDA_SOURCE's "interpret"/"interpretF32" exactly):
     *   (opcodes, targetSlots, literalConstants, instructionCount,
     *    in, dataSize, varCount, out)
     *
     * Shared by BOTH precisions -- the buffer layout (8-byte CUdeviceptr
     * handles for opcodes/targetSlots/literalConstants/in/out, 4-byte ints
     * for instructionCount/dataSize/varCount) is identical whether the
     * device pointers underneath happen to reference double or float
     * memory; only which CUdeviceptr values and which CUfunction get
     * passed in differs between dispatch()/dispatchF32().
     */
    private MemorySegment buildKernelParams(Arena tmp, long literalConstantsBuf,
            long inBuf, long outBuf, int dataSize) {
        MemorySegment pOpcodes = tmp.allocate(ValueLayout.JAVA_LONG);
        pOpcodes.set(ValueLayout.JAVA_LONG, 0, opcodesDevice);
        MemorySegment pTargetSlots = tmp.allocate(ValueLayout.JAVA_LONG);
        pTargetSlots.set(ValueLayout.JAVA_LONG, 0, targetSlotsDevice);
        MemorySegment pLiterals = tmp.allocate(ValueLayout.JAVA_LONG);
        pLiterals.set(ValueLayout.JAVA_LONG, 0, literalConstantsBuf);
        MemorySegment pInstrCount = tmp.allocate(ValueLayout.JAVA_INT);
        pInstrCount.set(ValueLayout.JAVA_INT, 0, instructionCount);
        MemorySegment pIn = tmp.allocate(ValueLayout.JAVA_LONG);
        pIn.set(ValueLayout.JAVA_LONG, 0, inBuf);
        MemorySegment pDataSize = tmp.allocate(ValueLayout.JAVA_INT);
        pDataSize.set(ValueLayout.JAVA_INT, 0, dataSize);
        MemorySegment pVarCount = tmp.allocate(ValueLayout.JAVA_INT);
        pVarCount.set(ValueLayout.JAVA_INT, 0, varCount);
        MemorySegment pOut = tmp.allocate(ValueLayout.JAVA_LONG);
        pOut.set(ValueLayout.JAVA_LONG, 0, outBuf);

        MemorySegment paramPtrs = tmp.allocate(ValueLayout.ADDRESS, 8);
        paramPtrs.setAtIndex(ValueLayout.ADDRESS, 0, pOpcodes);
        paramPtrs.setAtIndex(ValueLayout.ADDRESS, 1, pTargetSlots);
        paramPtrs.setAtIndex(ValueLayout.ADDRESS, 2, pLiterals);
        paramPtrs.setAtIndex(ValueLayout.ADDRESS, 3, pInstrCount);
        paramPtrs.setAtIndex(ValueLayout.ADDRESS, 4, pIn);
        paramPtrs.setAtIndex(ValueLayout.ADDRESS, 5, pDataSize);
        paramPtrs.setAtIndex(ValueLayout.ADDRESS, 6, pVarCount);
        paramPtrs.setAtIndex(ValueLayout.ADDRESS, 7, pOut);
        return paramPtrs;
    }

    // ================= double-path buffer/staging management =================
    // Independent in/out capacity tracking -- a single combined max()
    // capacity is unsafe across calls with asymmetric growth (see
    // OpenClCompositeExpression's ensureDeviceBuffers javadoc for the
    // concrete failure case).
    private void ensureDeviceBuffers(long inBytes, long outBytes) throws Throwable {
        boolean needsIn = inBytes > deviceInCapacityBytes || inputDevice == 0L;
        boolean needsOut = outBytes > deviceOutCapacityBytes || outputDevice == 0L;
        if (!needsIn && !needsOut) {
            return;
        }

        try (Arena tmp = Arena.ofConfined()) {
            if (needsIn) {
                if (inputDevice != 0L) {
                    cu.cuMemFree.invoke(inputDevice);
                }
                MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
                check((int) cu.cuMemAlloc.invoke(ptrBuf, inBytes), "cuMemAlloc(in)");
                inputDevice = ptrBuf.get(ValueLayout.JAVA_LONG, 0);
                deviceInCapacityBytes = inBytes;
            }
            if (needsOut) {
                if (outputDevice != 0L) {
                    cu.cuMemFree.invoke(outputDevice);
                }
                MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
                check((int) cu.cuMemAlloc.invoke(ptrBuf, outBytes), "cuMemAlloc(out)");
                outputDevice = ptrBuf.get(ValueLayout.JAVA_LONG, 0);
                deviceOutCapacityBytes = outBytes;
            }
        }
    }

    private void ensureStagingBytes(long inBytes, long outBytes) {
        if (inBytes > stagingInCapacityBytes) {
            stagingIn = arena.allocate(inBytes);
            stagingInCapacityBytes = inBytes;
        }
        if (outBytes > stagingOutCapacityBytes) {
            stagingOut = arena.allocate(outBytes);
            stagingOutCapacityBytes = outBytes;
        }
    }

    private void ensureStaging(int inLen, int outLen) {
        ensureStagingBytes((long) inLen * ValueLayout.JAVA_DOUBLE.byteSize(),
                           (long) outLen * ValueLayout.JAVA_DOUBLE.byteSize());
    }

    // ================= float-path buffer/staging management (independent of double path) =================

    private void ensureDeviceBuffersF32(long inBytes, long outBytes) throws Throwable {
        boolean needsIn = inBytes > deviceInCapacityBytesF32 || inputDeviceF32 == 0L;
        boolean needsOut = outBytes > deviceOutCapacityBytesF32 || outputDeviceF32 == 0L;
        if (!needsIn && !needsOut) {
            return;
        }

        try (Arena tmp = Arena.ofConfined()) {
            if (needsIn) {
                if (inputDeviceF32 != 0L) {
                    cu.cuMemFree.invoke(inputDeviceF32);
                }
                MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
                check((int) cu.cuMemAlloc.invoke(ptrBuf, inBytes), "cuMemAlloc(in, f32)");
                inputDeviceF32 = ptrBuf.get(ValueLayout.JAVA_LONG, 0);
                deviceInCapacityBytesF32 = inBytes;
            }
            if (needsOut) {
                if (outputDeviceF32 != 0L) {
                    cu.cuMemFree.invoke(outputDeviceF32);
                }
                MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
                check((int) cu.cuMemAlloc.invoke(ptrBuf, outBytes), "cuMemAlloc(out, f32)");
                outputDeviceF32 = ptrBuf.get(ValueLayout.JAVA_LONG, 0);
                deviceOutCapacityBytesF32 = outBytes;
            }
        }
    }

    private void ensureStagingF32(int inLen, int outLen) {
        long inBytes = (long) inLen * ValueLayout.JAVA_FLOAT.byteSize();
        long outBytes = (long) outLen * ValueLayout.JAVA_FLOAT.byteSize();
        if (inBytes > stagingInCapacityBytesF32) {
            stagingInF32 = arena.allocate(inBytes);
            stagingInCapacityBytesF32 = inBytes;
        }
        if (outBytes > stagingOutCapacityBytesF32) {
            stagingOutF32 = arena.allocate(outBytes);
            stagingOutCapacityBytesF32 = outBytes;
        }
    }

    // ================= program-buffer upload helpers =================

    private long uploadIntArray(int[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_INT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);

            MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
            check((int) cu.cuMemAlloc.invoke(ptrBuf, host.byteSize()), "cuMemAlloc(int[])");
            long device = ptrBuf.get(ValueLayout.JAVA_LONG, 0);

            check((int) cu.cuMemcpyHtoD.invoke(device, host, host.byteSize()), "cuMemcpyHtoD(int[])");
            return device;
        }
    }

    private long uploadDoubleArray(double[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_DOUBLE.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_DOUBLE, 0, data.length);

            MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
            check((int) cu.cuMemAlloc.invoke(ptrBuf, host.byteSize()), "cuMemAlloc(double[])");
            long device = ptrBuf.get(ValueLayout.JAVA_LONG, 0);

            check((int) cu.cuMemcpyHtoD.invoke(device, host, host.byteSize()), "cuMemcpyHtoD(double[])");
            return device;
        }
    }

    private long uploadFloatArray(float[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);

            MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
            check((int) cu.cuMemAlloc.invoke(ptrBuf, host.byteSize()), "cuMemAlloc(float[])");
            long device = ptrBuf.get(ValueLayout.JAVA_LONG, 0);

            check((int) cu.cuMemcpyHtoD.invoke(device, host, host.byteSize()), "cuMemcpyHtoD(float[])");
            return device;
        }
    }

    private static void check(int status, String call) {
        if (status != CudaBindings.CUDA_SUCCESS) {
            throw new IllegalStateException("CUDA error in " + call + ": code " + status);
        }
    }

    @Override
    public void close() {
        try {
            if (inputDevice != 0L) {
                cu.cuMemFree.invoke(inputDevice);
            }
            if (outputDevice != 0L) {
                cu.cuMemFree.invoke(outputDevice);
            }
            if (inputDeviceF32 != 0L) {
                cu.cuMemFree.invoke(inputDeviceF32);
            }
            if (outputDeviceF32 != 0L) {
                cu.cuMemFree.invoke(outputDeviceF32);
            }
            cu.cuMemFree.invoke(opcodesDevice);
            cu.cuMemFree.invoke(targetSlotsDevice);
            cu.cuMemFree.invoke(literalConstantsDevice);
            cu.cuMemFree.invoke(literalConstantsDeviceF32);
        } catch (Throwable t) {
            // best-effort cleanup
        } finally {
            arena.close();
        }
    }
}