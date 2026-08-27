package com.github.gbenroscience.gpu.evaluator.cuda;


import com.github.gbenroscience.gpu.evaluator.GpuCompositeExpression;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;

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
 * DEVICE BINDING MODEL: each {@link CudaCompositeExpression} instance is
 * bound, at construction time, to whichever device {@link #selectDevice}
 * currently resolves to. That binding is fixed for the instance's whole
 * lifetime -- an in-flight expression's GPU never changes under it. The
 * underlying per-device resources (primary context, loaded PTX module,
 * both kernel functions) are cached in a small registry keyed by device
 * index and shared by every instance bound to that device, so switching
 * selection back and forth (e.g. across test methods) does NOT rebuild or
 * recompile anything after the first time each distinct device is used.
 * This mirrors {@code OpenClCompositeExpression}'s registry exactly, and is
 * the direct fix for device selection previously being a one-shot,
 * JVM-wide decision baked into a static initializer: now it's "which
 * device will the NEXT constructed instance use", not "which device may
 * this JVM ever use, once, forever". The {@code cuda.device.index} system
 * property keeps working exactly as before for anyone already using it --
 * it's just read fresh on every construction now instead of once.
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
 *   independently-created contexts across threads. Every dispatch call
 *   re-asserts its own instance's context as current on the calling thread
 *   before touching the driver, which is what makes per-instance/per-device
 *   binding safe even when one thread interleaves calls across instances
 *   bound to different devices.
 *
 * - CUDA has no OpenCL-style "platform" layer -- devices are just indexed
 *   0..N-1 by the driver directly, so {@link #selectDevice(int)} takes a
 *   single device index rather than a (platform, device) pair, and there
 *   is no vendor-selection overload (every CUDA device is, definitionally,
 *   NVIDIA hardware).
 */
public final class CudaCompositeExpression implements GpuCompositeExpression {

    /** One enumerated CUDA device with its human-readable identity. */
    private record CudaDeviceCandidate(int deviceIndex, int deviceHandle, String deviceName,
            int major, int minor) {
        String describe() {
            return "[cuda device " + deviceIndex + "] " + deviceName
                    + " (compute capability " + major + "." + minor + ")";
        }
    }

    /**
     * Lists every CUDA device this process can currently see, as plain
     * human-readable descriptions -- e.g. {@code "[cuda device 0] NVIDIA
     * GeForce RTX 4080 (compute capability 8.9)"}. Call this first, before
     * guessing at a name substring or index to pass to {@link #selectDevice}.
     *
     * Deliberately independent of the context registry: this method does
     * its own lightweight device enumeration and never retains a primary
     * context, compiles PTX, or touches the registry -- safe to call any
     * number of times, at any point, purely for inspection.
     */
    public static java.util.List<String> listAvailableDevices() {
        try (Arena arena = Arena.ofConfined()) {
            java.util.List<CudaDeviceCandidate> candidates = enumerateDevices(arena);
            java.util.List<String> descriptions = new java.util.ArrayList<>();
            for (CudaDeviceCandidate c : candidates) {
                descriptions.add(c.describe());
            }
            return descriptions;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to enumerate CUDA devices", t);
        }
    }

    private static java.util.List<CudaDeviceCandidate> enumerateDevices(Arena arena) throws Throwable {
        check((int) CU.cuInit.invoke(0), "cuInit");

        MemorySegment countBuf = arena.allocate(ValueLayout.JAVA_INT);
        check((int) CU.cuDeviceGetCount.invoke(countBuf), "cuDeviceGetCount");
        int count = countBuf.get(ValueLayout.JAVA_INT, 0);

        java.util.List<CudaDeviceCandidate> candidates = new java.util.ArrayList<>();
        for (int i = 0; i < count; i++) {
            MemorySegment deviceBuf = arena.allocate(ValueLayout.JAVA_INT);
            check((int) CU.cuDeviceGet.invoke(deviceBuf, i), "cuDeviceGet");
            int device = deviceBuf.get(ValueLayout.JAVA_INT, 0);

            MemorySegment nameBuf = arena.allocate(256);
            check((int) CU.cuDeviceGetName.invoke(nameBuf, 256, device), "cuDeviceGetName");
            String name = nameBuf.getString(0, StandardCharsets.UTF_8);

            MemorySegment majorBuf = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment minorBuf = arena.allocate(ValueLayout.JAVA_INT);
            check((int) CU.cuDeviceGetAttribute.invoke(majorBuf,
                    CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                    "cuDeviceGetAttribute(major)");
            check((int) CU.cuDeviceGetAttribute.invoke(minorBuf,
                    CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
                    "cuDeviceGetAttribute(minor)");
            int major = majorBuf.get(ValueLayout.JAVA_INT, 0);
            int minor = minorBuf.get(ValueLayout.JAVA_INT, 0);

            candidates.add(new CudaDeviceCandidate(i, device, name, major, minor));
        }
        return candidates;
    }

    private static String describeAll(java.util.List<CudaDeviceCandidate> candidates) {
        StringBuilder sb = new StringBuilder();
        for (CudaDeviceCandidate c : candidates) {
            sb.append("  ").append(c.describe()).append('\n');
        }
        return sb.toString();
    }

    /**
     * Selection precedence, most to least specific -- read fresh every
     * time a context is resolved (see {@link #resolveContext()}), which is
     * what makes {@link #selectDevice} affect only instances constructed
     * after it's called, rather than a single JVM-wide decision:
     *   1. {@code -Dcuda.device.index=N} (or {@link #selectDevice(int)}) --
     *      exact device index. This is the same property this class has
     *      always honored; it's just re-read on every construction now
     *      instead of once, ever, in a static initializer.
     *   2. {@code -Dcuda.gpu.name=<substring>} (or
     *      {@link #selectDevice(String)}) -- case-insensitive substring
     *      match against the device's name (e.g. "4080", "A100").
     *   3. Default: device 0 -- the "just work on whatever's there"
     *      behavior for a single-GPU machine.
     */
    private static CudaDeviceCandidate selectCandidate(java.util.List<CudaDeviceCandidate> candidates) {
        String indexProp = System.getProperty("cuda.device.index");
        if (indexProp != null && !indexProp.isBlank()) {
            int deviceIndex = Integer.parseInt(indexProp.trim());
            for (CudaDeviceCandidate c : candidates) {
                if (c.deviceIndex() == deviceIndex) {
                    return c;
                }
            }
            throw new IllegalStateException(
                    "No CUDA device at cuda.device.index=" + deviceIndex + ". Available devices:\n"
                    + describeAll(candidates));
        }

        String nameProp = System.getProperty("cuda.gpu.name");
        if (nameProp != null && !nameProp.isBlank()) {
            String needle = nameProp.trim().toLowerCase(java.util.Locale.ROOT);
            for (CudaDeviceCandidate c : candidates) {
                if (c.deviceName().toLowerCase(java.util.Locale.ROOT).contains(needle)) {
                    return c;
                }
            }
            throw new IllegalStateException(
                    "No CUDA device matching cuda.gpu.name=\"" + nameProp + "\" found. Available devices:\n"
                    + describeAll(candidates));
        }

        return candidates.get(0);
    }

    // ================= device selection =================

    /**
     * Selects, by exact device index, which GPU the NEXT constructed
     * {@link CudaCompositeExpression} (or the next
     * {@link CudaExpressionBridge}/{@code GpuExpressionBridge} call) will
     * bind to. Use once you've already seen the available devices (e.g.
     * via {@link #listAvailableDevices()}).
     *
     * Safe to call as many times as you like, at any point in a JVM's
     * lifetime -- selection is resolved fresh on every construction, not
     * locked in once and forever. The FIRST time a given device index is
     * selected, its primary context and PTX module are built and cached;
     * every later {@code selectDevice} call back to that same index reuses
     * the cached context rather than rebuilding it.
     *
     * What this does NOT do: change the device an ALREADY-CONSTRUCTED
     * instance is using. Device binding is fixed per-instance at
     * construction time and never changes afterward -- an in-flight
     * expression's GPU never moves under it. This also isn't a per-call or
     * per-thread setting: it's a plain JVM system property under the hood
     * (the same {@code cuda.device.index} property this class has always
     * read), so don't call it concurrently from one thread while another
     * thread is mid-construction expecting a different device -- treat it
     * as "set the default for whatever gets constructed next", called from
     * one thread at a time.
     */
    public static void selectDevice(int deviceIndex) {
        System.clearProperty("cuda.gpu.name");
        System.setProperty("cuda.device.index", String.valueOf(deviceIndex));
    }

    /**
     * Selects, by case-insensitive substring match against the device
     * name (e.g. "4080", "A100", "RTX"), which GPU the NEXT constructed
     * instance will use. Same rules as {@link #selectDevice(int)} apply.
     * Use {@link #listAvailableDevices()} first if you're not sure what
     * substring to pass.
     */
    public static void selectDevice(String nameSubstring) {
        System.clearProperty("cuda.device.index");
        System.setProperty("cuda.gpu.name", nameSubstring);
    }

    /**
     * Clears any explicit selection previously set via {@link #selectDevice},
     * reverting to the default (device 0) for instances constructed after
     * this call. Existing instances are unaffected either way -- their
     * device binding was already fixed at their own construction time.
     */
    public static void clearDeviceSelection() {
        System.clearProperty("cuda.device.index");
        System.clearProperty("cuda.gpu.name");
    }

    // ================= per-device context registry =================

    /**
     * FFM bindings, shared by every instance and every device -- nothing
     * device-specific here, just resolved function pointers into the
     * driver/NVRTC libraries.
     */
    private static final CudaBindings CU = new CudaBindings();
    private static final NvrtcBindings NVRTC = new NvrtcBindings();

    /**
     * The GPU context registry: one entry per distinct device index
     * actually used so far, built the FIRST time that device is selected
     * and reused (primary context, loaded PTX module, both kernel
     * functions) by every instance bound to it afterward. This is what
     * makes repeated {@link #selectDevice} calls cheap after the first use
     * of each device -- switching back to a previously-used device never
     * recompiles anything.
     */
    private static final ConcurrentHashMap<Integer, CudaContext> CONTEXT_REGISTRY =
            new ConcurrentHashMap<>();

    /**
     * Resolves (building and caching if necessary) the CudaContext for
     * whichever device {@link #selectCandidate} currently points at.
     * Called once per {@link CudaCompositeExpression} construction -- the
     * resolved context is then fixed for that instance's entire lifetime.
     */
    private static CudaContext resolveContext() {
        try (Arena arena = Arena.ofConfined()) {
            java.util.List<CudaDeviceCandidate> candidates = enumerateDevices(arena);
            if (candidates.isEmpty()) {
                throw new IllegalStateException("No CUDA devices found");
            }
            CudaDeviceCandidate chosen = selectCandidate(candidates);
            return CONTEXT_REGISTRY.computeIfAbsent(chosen.deviceIndex(), k -> buildContext(chosen));
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) {
                throw re;
            }
            throw new RuntimeException("Failed to resolve a CUDA context", t);
        }
    }

    private static CudaContext buildContext(CudaDeviceCandidate chosen) {
        try {
            return new CudaContext(chosen);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to bootstrap CUDA context for " + chosen.describe(), t);
        }
    }

    /**
     * Everything needed to dispatch against ONE specific CUDA device: the
     * retained primary context, the loaded PTX module, both kernel
     * functions, and a dispatch lock scoped to just this device. Previously
     * a JVM-wide singleton built once in a static initializer (one
     * CudaContext, ever, for the whole process); now one instance per
     * distinct device index actually used, cached in {@link #CONTEXT_REGISTRY}
     * and shared by every {@link CudaCompositeExpression} bound to that
     * device -- exactly mirroring {@code OpenClCompositeExpression}'s
     * {@code GpuContext}.
     */
    private static final class CudaContext {

        final int device;
        final MemorySegment CONTEXT;   // CUcontext (opaque handle)
        final MemorySegment MODULE;    // CUmodule -- holds BOTH kernels
        final MemorySegment FUNCTION_F64; // CUfunction "interpret"
        final MemorySegment FUNCTION_F32; // CUfunction "interpretF32"
        final String selectedDeviceDescription;

        // CudaContext instances are shared by every CudaCompositeExpression
        // bound to the same device. cuCtxSynchronize() blocks until ALL
        // preceding work on this context has completed (context-wide, not
        // stream- or caller-specific), so kernel dispatch against a shared
        // context must be serialized per-device -- same rationale as the
        // OpenCL path's per-GpuContext dispatchLock. Scoped per-CudaContext
        // (not one global lock across every device) so dispatches to
        // DIFFERENT devices don't serialize against each other unnecessarily.
        final Object dispatchLock = new Object();

        CudaContext(CudaDeviceCandidate chosen) throws Throwable {
            try (Arena bootstrap = Arena.ofConfined()) {
                this.device = chosen.deviceHandle();
                this.selectedDeviceDescription = chosen.describe();

                MemorySegment ctxBuf = bootstrap.allocate(ValueLayout.ADDRESS);
                check((int) CU.cuDevicePrimaryCtxRetain.invoke(ctxBuf, device),
                        "cuDevicePrimaryCtxRetain");
                CONTEXT = ctxBuf.get(ValueLayout.ADDRESS, 0);
                check((int) CU.cuCtxSetCurrent.invoke(CONTEXT), "cuCtxSetCurrent");

                // ONE NVRTC compile -- the resulting PTX module contains
                // BOTH "interpret" and "interpretF32" (see CudaKernelSource).
                String ptx = compileToPtx(bootstrap, chosen.major(), chosen.minor());

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

                // Printed once per DISTINCT device the first time it's
                // built (not once per instance -- computeIfAbsent in
                // resolveContext ensures this constructor only runs once
                // per registry key), so it's obvious from program output
                // which of possibly several installed GPUs is in play.
                System.err.println("[ParserNG GPU] CUDA using " + selectedDeviceDescription);
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

    private static final int DEFAULT_BLOCK_SIZE = 256;

    /**
     * A human-readable description of the exact CUDA device this instance
     * is bound to -- e.g. {@code "[cuda device 0] NVIDIA GeForce RTX 4080
     * (compute capability 8.9)"}. Fixed for this instance's whole lifetime
     * (see class javadoc's "DEVICE BINDING MODEL"); confirms which device
     * a prior {@link #selectDevice} call actually resolved to.
     * @return
     */
    public String getDeviceDescription() {
        return ctx.selectedDeviceDescription;
    }

    private final CudaContext ctx;
    private final CudaBindings cu = CU;

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
        // Resolved ONCE, here, and fixed for this instance's whole
        // lifetime -- see class javadoc's "DEVICE BINDING MODEL".
        this.ctx = resolveContext();

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

    /**
     * True zero-copy, multi-variable double path -- see
     * GpuCompositeExpression's interface javadoc. {@code in[slot]} is
     * already native memory, so unlike applyBulk(double[][], double[])
     * there is no Java-heap-crossing staging copy here: each slot is
     * copied straight from its own segment to its slice of the device
     * input buffer -- CUDA device pointers are plain {@code long} handles,
     * so "its slice" is just {@code inputDevice + slot*rowBytes}, no
     * separate offset-write call shape needed the way OpenCL's
     * clEnqueueWriteBuffer requires. varCount == 1 is special-cased to
     * skip straight to the existing single-segment dispatch() path.
     */
    public void applyBulk(MemorySegment[] in, MemorySegment out) throws Throwable {
        if (in.length != varCount) {
            throw new IllegalArgumentException(
                    "Expected " + varCount + " variable segments, got " + in.length);
        }
        long dataSize = out.byteSize() / ValueLayout.JAVA_DOUBLE.byteSize();
        long rowBytes = dataSize * ValueLayout.JAVA_DOUBLE.byteSize();
        for (int slot = 0; slot < in.length; slot++) {
            if (in[slot].byteSize() != rowBytes) {
                throw new IllegalArgumentException(
                        "Variable segment " + slot + " has " + in[slot].byteSize()
                                + " bytes, expected " + rowBytes + " (dataSize=" + dataSize + ")");
            }
        }
        if (varCount == 1) {
            dispatch(in[0], out, (int) dataSize);
            return;
        }
        dispatchScatter(in, out, rowBytes, (int) dataSize);
    }

    private void dispatchScatter(MemorySegment[] in, MemorySegment out, long rowBytes, int dataSize) throws Throwable {
        synchronized (ctx.dispatchLock) {
            ensureDeviceBuffers((long) varCount * rowBytes, out.byteSize());

            try {
                check((int) cu.cuCtxSetCurrent.invoke(ctx.CONTEXT), "cuCtxSetCurrent");

                for (int slot = 0; slot < varCount; slot++) {
                    long offset = (long) slot * rowBytes;
                    check((int) cu.cuMemcpyHtoD.invoke(inputDevice + offset, in[slot], rowBytes),
                            "cuMemcpyHtoD(in[" + slot + "])");
                }

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment kernelParams = buildKernelParams(tmp,
                            literalConstantsDevice, inputDevice, outputDevice, dataSize);

                    int gridDimX = (dataSize + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
                    check((int) cu.cuLaunchKernel.invoke(
                            ctx.FUNCTION_F64,
                            gridDimX, 1, 1,
                            DEFAULT_BLOCK_SIZE, 1, 1,
                            0,
                            MemorySegment.NULL,   // default stream
                            kernelParams,
                            MemorySegment.NULL),
                            "cuLaunchKernel(interpret, scatter)");
                }

                check((int) cu.cuCtxSynchronize.invoke(), "cuCtxSynchronize");

                check((int) cu.cuMemcpyDtoH.invoke(out, outputDevice, out.byteSize()), "cuMemcpyDtoH(out, scatter)");
            } catch (Throwable t) {
                throw new RuntimeException("GPU dispatch failed (scatter)", t);
            }
        }
    }

    private void dispatch(MemorySegment in, MemorySegment out, int dataSize) throws Throwable {
        synchronized (ctx.dispatchLock) {
            ensureDeviceBuffers(in.byteSize(), out.byteSize());

            try {
                check((int) cu.cuCtxSetCurrent.invoke(ctx.CONTEXT), "cuCtxSetCurrent");

                check((int) cu.cuMemcpyHtoD.invoke(inputDevice, in, in.byteSize()), "cuMemcpyHtoD(in)");

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment kernelParams = buildKernelParams(tmp,
                            literalConstantsDevice, inputDevice, outputDevice, dataSize);

                    int gridDimX = (dataSize + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
                    check((int) cu.cuLaunchKernel.invoke(
                            ctx.FUNCTION_F64,
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

    /**
     * True zero-copy, multi-variable float32 path -- float32 counterpart
     * of {@link #applyBulk(MemorySegment[], MemorySegment)}. See that
     * method's javadoc; same contract, just float throughout.
     */
    @Override
    public void applyBulkF32(MemorySegment[] in, MemorySegment out) throws Throwable {
        if (in.length != varCount) {
            throw new IllegalArgumentException(
                    "Expected " + varCount + " variable segments, got " + in.length);
        }
        long dataSize = out.byteSize() / ValueLayout.JAVA_FLOAT.byteSize();
        long rowBytes = dataSize * ValueLayout.JAVA_FLOAT.byteSize();
        for (int slot = 0; slot < in.length; slot++) {
            if (in[slot].byteSize() != rowBytes) {
                throw new IllegalArgumentException(
                        "Variable segment " + slot + " has " + in[slot].byteSize()
                                + " bytes, expected " + rowBytes + " (dataSize=" + dataSize + ")");
            }
        }
        if (varCount == 1) {
            dispatchF32(in[0], out, (int) dataSize);
            return;
        }
        dispatchScatterF32(in, out, rowBytes, (int) dataSize);
    }

    private void dispatchScatterF32(MemorySegment[] in, MemorySegment out, long rowBytes, int dataSize) throws Throwable {
        synchronized (ctx.dispatchLock) {
            ensureDeviceBuffersF32((long) varCount * rowBytes, out.byteSize());

            try {
                check((int) cu.cuCtxSetCurrent.invoke(ctx.CONTEXT), "cuCtxSetCurrent");

                for (int slot = 0; slot < varCount; slot++) {
                    long offset = (long) slot * rowBytes;
                    check((int) cu.cuMemcpyHtoD.invoke(inputDeviceF32 + offset, in[slot], rowBytes),
                            "cuMemcpyHtoD(in[" + slot + "], f32)");
                }

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment kernelParams = buildKernelParams(tmp,
                            literalConstantsDeviceF32, inputDeviceF32, outputDeviceF32, dataSize);

                    int gridDimX = (dataSize + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
                    check((int) cu.cuLaunchKernel.invoke(
                            ctx.FUNCTION_F32,
                            gridDimX, 1, 1,
                            DEFAULT_BLOCK_SIZE, 1, 1,
                            0,
                            MemorySegment.NULL,   // default stream
                            kernelParams,
                            MemorySegment.NULL),
                            "cuLaunchKernel(interpretF32, scatter)");
                }

                check((int) cu.cuCtxSynchronize.invoke(), "cuCtxSynchronize");

                check((int) cu.cuMemcpyDtoH.invoke(out, outputDeviceF32, out.byteSize()), "cuMemcpyDtoH(out, f32, scatter)");
            } catch (Throwable t) {
                throw new RuntimeException("GPU dispatch failed (f32, scatter)", t);
            }
        }
    }

    private void dispatchF32(MemorySegment in, MemorySegment out, int dataSize) throws Throwable {
        synchronized (ctx.dispatchLock) {
            ensureDeviceBuffersF32(in.byteSize(), out.byteSize());

            try {
                check((int) cu.cuCtxSetCurrent.invoke(ctx.CONTEXT), "cuCtxSetCurrent");

                check((int) cu.cuMemcpyHtoD.invoke(inputDeviceF32, in, in.byteSize()), "cuMemcpyHtoD(in, f32)");

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment kernelParams = buildKernelParams(tmp,
                            literalConstantsDeviceF32, inputDeviceF32, outputDeviceF32, dataSize);

                    int gridDimX = (dataSize + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
                    check((int) cu.cuLaunchKernel.invoke(
                            ctx.FUNCTION_F32,
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