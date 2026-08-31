package com.github.gbenroscience.gpu.evaluator.metal;

import com.github.gbenroscience.gpu.evaluator.GpuCompositeExpression;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Native single-precision Metal evaluator -- macOS ONLY. Compiles ONE MSL
 * kernel ({@link MetalKernelSource#KERNEL_NAME_F32}) once per distinct
 * {@code MTLDevice} the first time that device is used, exactly mirroring
 * OpenClCompositeExpression's per-device context registry and
 * {@code selectDevice}/{@code GpuVendor} API surface (same method names,
 * same precedence rules for {@code -Dmetal.device.index=}/{@code
 * -Dmetal.gpu.vendor=}), so callers already using the OpenCL backend can
 * switch backends without relearning device selection.
 *
 * WHAT IS DELIBERATELY DIFFERENT FROM OpenClCompositeExpression, AND WHY:
 *
 *  1. NO DOUBLE-PRECISION PATH. Metal Shading Language has no double type
 *     for GPU compute on any Apple GPU (see MetalKernelSource's javadoc) --
 *     this is a hardware ceiling, not a missing feature. Every
 *     {@code GpuCompositeExpression} method that takes/returns
 *     {@code double}s ({@link #applyBulk(MemorySegment, MemorySegment)},
 *     {@link #applyBulk(double[], double[])}, {@link #applyBulk(double[][], double[])})
 *     throws {@link UnsupportedOperationException} with a message pointing
 *     at the float-path equivalent. Callers that need double precision on
 *     Apple hardware should use the OpenCL backend instead (Apple's OpenCL
 *     driver is deprecated but, as of this writing, still present and still
 *     supports {@code cl_khr_fp64} on the CPU device at minimum) or fall
 *     back to the CPU evaluator.
 *
 *  2. NO EXPLICIT WRITE-BUFFER/READ-BUFFER ENQUEUE CALLS. Buffers are
 *     created with {@code MTLResourceStorageModeShared}, Metal's
 *     CPU-and-GPU-visible unified-memory mode: {@code contents} returns a
 *     raw pointer this code can {@link MemorySegment#copy} into and out of
 *     directly. There is no OpenCL-style {@code clEnqueueWriteBuffer}/
 *     {@code clEnqueueReadBuffer} step -- the copy IS the write/read, and it
 *     must happen strictly before {@code commit} / strictly after
 *     {@code waitUntilCompleted} respectively, which {@link #dispatchF32}
 *     enforces by ordering.
 *
 *  3. SCALAR KERNEL ARGUMENTS use {@code setBytes:length:atIndex:} against a
 *     {@code constant int&} kernel parameter rather than OpenCL's
 *     {@code clSetKernelArg} value-copy -- functionally equivalent (Metal
 *     copies the bytes into command-buffer-managed storage immediately,
 *     same as clSetKernelArg does), just a different call for the same job.
 *
 *  4. DEVICE MEMORY MANAGEMENT IS MANUAL REFERENCE COUNTING (no ARC, no GC
 *     for Objective-C objects) -- see MetalBindings' class javadoc. Every
 *     Metal object this class creates directly (buffers, command buffers,
 *     encoders, NSStrings) is released as soon as this code is done with
 *     it; the exceptions are the per-device {@code GpuContext} fields
 *     (device, queue, library, pipeline state), which are intentionally
 *     retained for the JVM's lifetime and never released, exactly like
 *     OpenClCompositeExpression never tears down its cached context/program/
 *     kernels either.
 *
 * DEVICE BINDING MODEL: identical to OpenClCompositeExpression -- each
 * instance is bound, at construction time, to whichever device
 * {@link #selectDevice} currently resolves to, for that instance's entire
 * lifetime. Per-device resources are cached in {@link #CONTEXT_REGISTRY} and
 * shared by every instance bound to the same device.
 */
public final class MetalCompositeExpression implements GpuCompositeExpression {

    /**
     * Closed, typed vendor choice mirroring OpenClCompositeExpression.GpuVendor.
     * On current Macs the overwhelmingly common case is a single
     * {@code APPLE} (Apple Silicon integrated) device; AMD/INTEL cover older
     * Intel Macs with a discrete or integrated GPU from those vendors, which
     * Metal can still drive. NVIDIA is included only for completeness --
     * Apple has not shipped an NVIDIA-capable Metal driver in years, so this
     * will simply never match on any machine actually running this code.
     */
    public enum GpuVendor {
        APPLE, AMD, INTEL, NVIDIA
    }

    /** One enumerated MTLDevice with its human-readable identity. */
    private record Candidate(MemorySegment device, int index, String name) {
        String describe() {
            return "[device " + index + ": " + name + "]";
        }

        String registryKey() {
            return Integer.toString(index);
        }
    }

    private static final MetalBindings MTL = new MetalBindings();

    /**
     * Per-device context registry -- same purpose and lifetime contract as
     * OpenClCompositeExpression.CONTEXT_REGISTRY.
     */
    private static final ConcurrentHashMap<String, GpuContext> CONTEXT_REGISTRY = new ConcurrentHashMap<>();

    /**
     * Lists every GPU Metal can currently see, as human-readable
     * descriptions (e.g. {@code "[device 0: Apple M3 Max]"}). Deliberately
     * independent of the context registry -- never builds a command queue,
     * compiles a library, or touches the registry; safe to call any number
     * of times purely for inspection.
     */
    public static java.util.List<String> listAvailableDevices() {
        try (Arena arena = Arena.ofConfined()) {
            java.util.List<Candidate> candidates = enumerateCandidates(arena);
            java.util.List<String> descriptions = new java.util.ArrayList<>();
            for (Candidate c : candidates) {
                descriptions.add(c.describe());
            }
            return descriptions;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to enumerate Metal devices", t);
        }
    }

    private static java.util.List<Candidate> enumerateCandidates(Arena arena) throws Throwable {
        MemorySegment array = (MemorySegment) MTL.MTLCopyAllDevices.invoke();
        if (array.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("MTLCopyAllDevices returned NULL -- no Metal-capable GPU on this machine");
        }
        long count = MTL.longMsgSend(array, MTL.sel(arena, "count"));

        java.util.List<Candidate> candidates = new java.util.ArrayList<>();
        for (long i = 0; i < count; i++) {
            MemorySegment device = MTL.idMsgSendL(array, MTL.sel(arena, "objectAtIndex:"), i);
            MemorySegment nameObj = MTL.idMsgSend(device, MTL.sel(arena, "name"));
            String name = MTL.utf8String(arena, nameObj);
            candidates.add(new Candidate(device, (int) i, name == null ? "(unnamed)" : name));
        }
        MTL.release(arena, array);
        return candidates;
    }

    /**
     * Selection precedence, identical structure to
     * OpenClCompositeExpression.selectCandidate:
     *   1. -Dmetal.device.index=N -- exact index into listAvailableDevices().
     *   2. -Dmetal.gpu.vendor=<substring> -- case-insensitive substring
     *      match against the device name (e.g. "Apple", "AMD", "Radeon",
     *      "Intel"), first match wins. No alias table is needed here the way
     *      OpenClCompositeExpression needs one for CL_DEVICE_VENDOR: Metal's
     *      {@code name} property already reports consumer-recognizable
     *      strings ("Apple M3 Max", "AMD Radeon Pro 5500M") rather than
     *      legal-entity vendor strings like OpenCL's "Advanced Micro
     *      Devices, Inc.", so a plain substring match is sufficient.
     *   3. Default: {@code MTLCreateSystemDefaultDevice()}'s device if it
     *      appears in the candidate list, else the first candidate found.
     */
    private static Candidate selectCandidate(java.util.List<Candidate> candidates, Arena arena) throws Throwable {
        String indexProp = System.getProperty("metal.device.index");
        if (indexProp != null) {
            int index = Integer.parseInt(indexProp.trim());
            for (Candidate c : candidates) {
                if (c.index() == index) {
                    return c;
                }
            }
            throw new IllegalStateException(
                    "No GPU found at metal.device.index=" + index + ". Available devices:\n" + describeAll(candidates));
        }

        String vendorProp = System.getProperty("metal.gpu.vendor");
        if (vendorProp != null && !vendorProp.isBlank()) {
            String needle = vendorProp.trim().toLowerCase(java.util.Locale.ROOT);
            for (Candidate c : candidates) {
                if (c.name().toLowerCase(java.util.Locale.ROOT).contains(needle)) {
                    return c;
                }
            }
            throw new IllegalStateException(
                    "No GPU matching metal.gpu.vendor=\"" + vendorProp + "\" found. Available devices:\n"
                    + describeAll(candidates));
        }

        MemorySegment defaultDevice = (MemorySegment) MTL.MTLCreateSystemDefaultDevice.invoke();
        if (!defaultDevice.equals(MemorySegment.NULL)) {
            MemorySegment defaultName = MTL.idMsgSend(defaultDevice, MTL.sel(arena, "name"));
            String dn = MTL.utf8String(arena, defaultName);
            MTL.release(arena, defaultDevice);
            if (dn != null) {
                for (Candidate c : candidates) {
                    if (c.name().equals(dn)) {
                        return c;
                    }
                }
            }
        }

        return candidates.get(0);
    }

    private static String describeAll(java.util.List<Candidate> candidates) {
        StringBuilder sb = new StringBuilder();
        for (Candidate c : candidates) {
            sb.append("  ").append(c.describe()).append('\n');
        }
        return sb.toString();
    }

    private static GpuContext resolveContext() {
        try (Arena arena = Arena.ofConfined()) {
            java.util.List<Candidate> candidates = enumerateCandidates(arena);
            if (candidates.isEmpty()) {
                throw new IllegalStateException("No Metal-capable GPU devices found on this machine");
            }
            Candidate chosen = selectCandidate(candidates, arena);
            return CONTEXT_REGISTRY.computeIfAbsent(chosen.registryKey(), k -> buildContext(chosen));
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) {
                throw re;
            }
            throw new RuntimeException("Failed to resolve a Metal GPU context", t);
        }
    }

    private static GpuContext buildContext(Candidate chosen) {
        try {
            return new GpuContext(chosen);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to bootstrap Metal context for " + chosen.describe(), t);
        }
    }

    /**
     * Everything needed to dispatch against ONE specific MTLDevice: the
     * device, its command queue, the compiled library, the compute pipeline
     * state for {@code interpretF32}, and a per-device dispatch lock -- same
     * role and same "never torn down" lifetime as OpenClCompositeExpression's
     * GpuContext.
     */
    private static final class GpuContext {
        final MemorySegment device;
        final MemorySegment commandQueue;
        final MemorySegment library;
        final MemorySegment pipelineStateF32;
        final long maxTotalThreadsPerThreadgroup;
        final String selectedDeviceDescription;
        final Object dispatchLock = new Object();

        GpuContext(Candidate chosen) throws Throwable {
            try (Arena bootstrap = Arena.ofConfined()) {
                this.selectedDeviceDescription = chosen.describe();
                this.device = chosen.device();

                this.commandQueue = MTL.idMsgSend(device, MTL.sel(bootstrap, "newCommandQueue"));
                if (commandQueue.equals(MemorySegment.NULL)) {
                    throw new IllegalStateException("newCommandQueue returned nil for " + selectedDeviceDescription);
                }

                MemorySegment source = MTL.nsString(bootstrap, MetalKernelSource.METAL_SOURCE);
                MemorySegment errorPtrPtr = bootstrap.allocate(ValueLayout.ADDRESS);
                errorPtrPtr.set(ValueLayout.ADDRESS, 0, MemorySegment.NULL);

                this.library = MTL.idMsgSend(device,
                        MTL.sel(bootstrap, "newLibraryWithSource:options:error:"),
                        source, MemorySegment.NULL, errorPtrPtr);
                MTL.release(bootstrap, source);

                if (library.equals(MemorySegment.NULL)) {
                    throw new IllegalStateException("Metal shader compilation failed on "
                            + selectedDeviceDescription + ": " + describeNsError(bootstrap, errorPtrPtr));
                }

                MemorySegment fnName = MTL.nsString(bootstrap, MetalKernelSource.KERNEL_NAME_F32);
                MemorySegment function = MTL.idMsgSend(library, MTL.sel(bootstrap, "newFunctionWithName:"), fnName);
                MTL.release(bootstrap, fnName);
                if (function.equals(MemorySegment.NULL)) {
                    throw new IllegalStateException("Kernel function \"" + MetalKernelSource.KERNEL_NAME_F32
                            + "\" not found in compiled library on " + selectedDeviceDescription);
                }

                errorPtrPtr.set(ValueLayout.ADDRESS, 0, MemorySegment.NULL);
                this.pipelineStateF32 = MTL.idMsgSend(device,
                        MTL.sel(bootstrap, "newComputePipelineStateWithFunction:error:"),
                        function, errorPtrPtr);
                MTL.release(bootstrap, function);

                if (pipelineStateF32.equals(MemorySegment.NULL)) {
                    throw new IllegalStateException("Failed to create compute pipeline state on "
                            + selectedDeviceDescription + ": " + describeNsError(bootstrap, errorPtrPtr));
                }

                this.maxTotalThreadsPerThreadgroup = MTL.longMsgSend(
                        pipelineStateF32, MTL.sel(bootstrap, "maxTotalThreadsPerThreadgroup"));

                // Printed once per DISTINCT device the first time it's built
                // (computeIfAbsent in resolveContext ensures this
                // constructor runs once per registry key) -- mirrors
                // OpenClCompositeExpression's equivalent log line.
                System.err.println("[ParserNG GPU] Metal using " + selectedDeviceDescription);
            }
        }

        private static String describeNsError(Arena arena, MemorySegment errorPtrPtr) {
            MemorySegment error = errorPtrPtr.get(ValueLayout.ADDRESS, 0);
            if (error.equals(MemorySegment.NULL)) {
                return "(no NSError provided)";
            }
            MemorySegment desc = MTL.idMsgSend(error, MTL.sel(arena, "localizedDescription"));
            String s = MTL.utf8String(arena, desc);
            return s == null ? "(no localizedDescription)" : s;
        }
    }

    // ================= device selection =================

    /** Selects which GPU the NEXT constructed instance binds to, by name substring. See {@link #selectCandidate}. */
    public static void selectDevice(String vendorOrNameSubstring) {
        System.setProperty("metal.gpu.vendor", vendorOrNameSubstring);
    }

    /** Recommended entry point over {@link #selectDevice(String)} -- see OpenClCompositeExpression.GpuVendor's javadoc for the rationale. */
    public static void selectDevice(GpuVendor vendor) {
        selectDevice(vendor.name());
    }

    /** Exact-index counterpart of {@link #selectDevice(String)}, indexing into {@link #listAvailableDevices()}. */
    public static void selectDevice(int deviceIndex) {
        System.setProperty("metal.device.index", String.valueOf(deviceIndex));
    }

    /** Clears any explicit selection previously set via {@link #selectDevice}. */
    public static void clearDeviceSelection() {
        System.clearProperty("metal.gpu.vendor");
        System.clearProperty("metal.device.index");
    }

    // ================= instance state =================

    public String getDeviceDescription() {
        return ctx.selectedDeviceDescription;
    }

    private final GpuContext ctx;
    private final Arena arena = Arena.ofShared();

    private final MemorySegment opcodesDevice;
    private final MemorySegment targetSlotsDevice;
    private final MemorySegment literalConstantsDeviceF32;
    private final int instructionCount;
    private final int varCount;

    private MemorySegment inputDeviceF32 = MemorySegment.NULL;
    private MemorySegment outputDeviceF32 = MemorySegment.NULL;
    private long deviceInCapacityBytesF32 = 0;
    private long deviceOutCapacityBytesF32 = 0;

    /**
     * @param opcodes same bytecode VectorTurboEvaluator/OpenClCompositeExpression consume
     * @param targetSlots same bytecode VectorTurboEvaluator/OpenClCompositeExpression consume
     * @param literalConstants double[] for call-site parity with the OpenCL constructor and
     *        VectorTurboEvaluator's own output type; narrowed to float[] here (and ONLY here --
     *        no double buffer is ever uploaded) since Metal has no double kernel to feed. Values
     *        outside float range/precision will lose precision exactly as any double-to-float
     *        narrowing cast does; callers needing exact double precision should use the OpenCL
     *        backend instead (see class javadoc point 1).
     */
    public MetalCompositeExpression(int[] opcodes, int[] targetSlots, double[] literalConstants,
            int instructionCount, int varCount) {
        this.instructionCount = instructionCount;
        this.varCount = varCount;
        this.ctx = resolveContext();

        try {
            this.opcodesDevice = uploadIntArray(opcodes);
            this.targetSlotsDevice = uploadIntArray(targetSlots);

            float[] literalConstantsF32 = new float[literalConstants.length];
            for (int i = 0; i < literalConstants.length; i++) {
                literalConstantsF32[i] = (float) literalConstants[i];
            }
            this.literalConstantsDeviceF32 = uploadFloatArray(literalConstantsF32);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to upload GPU program buffers", t);
        }
    }

    // ================= double-precision path: unsupported on Metal =================

    private static UnsupportedOperationException noDoubleSupport() {
        return new UnsupportedOperationException(
                "MetalCompositeExpression has no double-precision path -- Metal Shading Language has no "
                + "double type on any Apple GPU. Use applyBulkF32/applyBulk(float[]...) instead, or use "
                + "OpenClCompositeExpression if exact double precision is required.");
    }

    @Override
    public void applyBulk(MemorySegment in, MemorySegment out) throws Throwable {
        throw noDoubleSupport();
    }

    @Override
    public void applyBulk(double[] in, double[] out) throws Throwable {
        throw noDoubleSupport();
    }

    @Override
    public void applyBulk(double[][] in, double[] out) throws Throwable {
        throw noDoubleSupport();
    }

    /** Segment-based counterpart of {@link #applyBulk(double[][], double[])} -- same reason, same message. */
    @Override
    public void applyBulk(MemorySegment[] in, MemorySegment out) throws Throwable {
        throw noDoubleSupport();
    }

    // ================= float path =================

    @Override
    public void applyBulkF32(MemorySegment in, MemorySegment out) throws Throwable {
        long dataSize = out.byteSize() / ValueLayout.JAVA_FLOAT.byteSize();
        dispatchF32(in, out, (int) dataSize);
    }

    /**
     * Segment-based counterpart of {@link #applyBulk(float[][], float[])}: one
     * MemorySegment per variable row (each expected to hold {@code dataSize}
     * floats, {@code dataSize} derived from {@code out}), flattened into a
     * single varCount-major input buffer the same way the {@code float[][]}
     * overload flattens Java arrays -- just copying from segments instead of
     * from heap arrays, so callers already holding off-heap row data (e.g.
     * from a prior GPU/native step) can skip a heap round-trip entirely.
     */
    @Override
    public void applyBulkF32(MemorySegment[] in, MemorySegment out) throws Throwable {
        if (in.length != varCount) {
            throw new IllegalArgumentException("Expected " + varCount + " variable rows, got " + in.length);
        }
        long dataSize = out.byteSize() / ValueLayout.JAVA_FLOAT.byteSize();
        long rowBytes = dataSize * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment inSeg = arena.allocate((long) varCount * rowBytes);
        for (int slot = 0; slot < in.length; slot++) {
            MemorySegment row = in[slot];
            if (row.byteSize() != rowBytes) {
                throw new IllegalArgumentException("Row " + slot + " has " + row.byteSize()
                        + " bytes, expected " + rowBytes + " (dataSize=" + dataSize + " floats)");
            }
            MemorySegment.copy(row, 0, inSeg, (long) slot * rowBytes, rowBytes);
        }
        dispatchF32(inSeg, out, (int) dataSize);
    }

    @Override
    public void applyBulk(float[] in, float[] out) throws Throwable {
        MemorySegment inSeg = arena.allocate((long) in.length * ValueLayout.JAVA_FLOAT.byteSize());
        MemorySegment.copy(in, 0, inSeg, ValueLayout.JAVA_FLOAT, 0, in.length);
        MemorySegment outSeg = arena.allocate((long) out.length * ValueLayout.JAVA_FLOAT.byteSize());
        dispatchF32(inSeg, outSeg, out.length);
        MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
    }

    @Override
    public void applyBulk(float[][] in, float[] out) throws Throwable {
        if (in.length != varCount) {
            throw new IllegalArgumentException("Expected " + varCount + " variable rows, got " + in.length);
        }
        int dataSize = out.length;
        int flatLen = varCount * dataSize;

        MemorySegment inSeg = arena.allocate((long) flatLen * ValueLayout.JAVA_FLOAT.byteSize());
        for (int slot = 0; slot < in.length; slot++) {
            MemorySegment.copy(in[slot], 0, inSeg, ValueLayout.JAVA_FLOAT,
                    (long) slot * dataSize * ValueLayout.JAVA_FLOAT.byteSize(), dataSize);
        }
        MemorySegment outSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
        dispatchF32(inSeg, outSeg, dataSize);
        MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
    }

    private void dispatchF32(MemorySegment in, MemorySegment out, int dataSize) throws Throwable {
        synchronized (ctx.dispatchLock) {
            try (Arena tmp = Arena.ofConfined()) {
                ensureDeviceBuffersF32(in.byteSize(), out.byteSize(), tmp);

                // --- host -> device: direct copy into the shared-storage
                // buffer's mapped memory, done BEFORE this dispatch's
                // command buffer is committed (see class javadoc point 2). ---
                MemorySegment inContents = MTL.idMsgSend(inputDeviceF32, MTL.sel(tmp, "contents"));
                MemorySegment.copy(in, 0, inContents.reinterpret(in.byteSize()), 0, in.byteSize());

                MemorySegment commandBuffer = MTL.idMsgSend(ctx.commandQueue, MTL.sel(tmp, "commandBuffer"));
                MemorySegment encoder = MTL.idMsgSend(commandBuffer, MTL.sel(tmp, "computeCommandEncoder"));

                MTL.voidMsgSend(encoder, MTL.sel(tmp, "setComputePipelineState:"), ctx.pipelineStateF32);

                MTL.voidMsgSendPtrLL(encoder, MTL.sel(tmp, "setBuffer:offset:atIndex:"), opcodesDevice, 0L, 0L);
                MTL.voidMsgSendPtrLL(encoder, MTL.sel(tmp, "setBuffer:offset:atIndex:"), targetSlotsDevice, 0L, 1L);
                MTL.voidMsgSendPtrLL(encoder, MTL.sel(tmp, "setBuffer:offset:atIndex:"), literalConstantsDeviceF32, 0L, 2L);
                setBytesInt(tmp, encoder, instructionCount, 3L);
                MTL.voidMsgSendPtrLL(encoder, MTL.sel(tmp, "setBuffer:offset:atIndex:"), inputDeviceF32, 0L, 4L);
                setBytesInt(tmp, encoder, dataSize, 5L);
                setBytesInt(tmp, encoder, varCount, 6L);
                MTL.voidMsgSendPtrLL(encoder, MTL.sel(tmp, "setBuffer:offset:atIndex:"), outputDeviceF32, 0L, 7L);

                long threadsPerGroup = Math.max(1L, Math.min(ctx.maxTotalThreadsPerThreadgroup, dataSize));
                MemorySegment gridSize = MTL.mtlSize(tmp, dataSize, 1, 1);
                MemorySegment groupSize = MTL.mtlSize(tmp, threadsPerGroup, 1, 1);
                MTL.voidMsgSendDispatch(encoder, MTL.sel(tmp, "dispatchThreads:threadsPerThreadgroup:"),
                        gridSize, groupSize);

                MTL.voidMsgSend(encoder, MTL.sel(tmp, "endEncoding"));
                MTL.voidMsgSend(commandBuffer, MTL.sel(tmp, "commit"));
                MTL.voidMsgSend(commandBuffer, MTL.sel(tmp, "waitUntilCompleted"));

                // --- device -> host: only safe to read after
                // waitUntilCompleted returns (see class javadoc point 2). ---
                MemorySegment outContents = MTL.idMsgSend(outputDeviceF32, MTL.sel(tmp, "contents"));
                MemorySegment.copy(outContents.reinterpret(out.byteSize()), 0, out, 0, out.byteSize());
            } catch (Throwable t) {
                throw new RuntimeException("GPU dispatch failed (Metal, f32)", t);
            }
        }
    }

    private void setBytesInt(Arena tmp, MemorySegment encoder, int value, long atIndex) {
        MemorySegment holder = tmp.allocate(ValueLayout.JAVA_INT);
        holder.set(ValueLayout.JAVA_INT, 0, value);
        MTL.voidMsgSendPtrLL(encoder, MTL.sel(tmp, "setBytes:length:atIndex:"),
                holder, ValueLayout.JAVA_INT.byteSize(), atIndex);
    }

    private void ensureDeviceBuffersF32(long inBytes, long outBytes, Arena tmp) {
        boolean needsIn = inBytes > deviceInCapacityBytesF32 || inputDeviceF32.equals(MemorySegment.NULL);
        boolean needsOut = outBytes > deviceOutCapacityBytesF32 || outputDeviceF32.equals(MemorySegment.NULL);
        if (!needsIn && !needsOut) {
            return;
        }

        if (needsIn) {
            if (!inputDeviceF32.equals(MemorySegment.NULL)) {
                MTL.release(tmp, inputDeviceF32);
            }
            inputDeviceF32 = MTL.idMsgSendLL(ctx.device, MTL.sel(tmp, "newBufferWithLength:options:"),
                    inBytes, MetalBindings.MTLResourceOptionsDefault);
            if (inputDeviceF32.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("newBufferWithLength:options: returned nil (in)");
            }
            deviceInCapacityBytesF32 = inBytes;
        }

        if (needsOut) {
            if (!outputDeviceF32.equals(MemorySegment.NULL)) {
                MTL.release(tmp, outputDeviceF32);
            }
            outputDeviceF32 = MTL.idMsgSendLL(ctx.device, MTL.sel(tmp, "newBufferWithLength:options:"),
                    outBytes, MetalBindings.MTLResourceOptionsDefault);
            if (outputDeviceF32.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("newBufferWithLength:options: returned nil (out)");
            }
            deviceOutCapacityBytesF32 = outBytes;
        }
    }

    private MemorySegment uploadIntArray(int[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_INT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);
            MemorySegment buf = MTL.idMsgSendPtrLL(ctx.device, MTL.sel(tmp, "newBufferWithBytes:length:options:"),
                    host, host.byteSize(), MetalBindings.MTLResourceOptionsDefault);
            if (buf.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("newBufferWithBytes:length:options: returned nil (int[])");
            }
            return buf;
        }
    }

    private MemorySegment uploadFloatArray(float[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            MemorySegment buf = MTL.idMsgSendPtrLL(ctx.device, MTL.sel(tmp, "newBufferWithBytes:length:options:"),
                    host, host.byteSize(), MetalBindings.MTLResourceOptionsDefault);
            if (buf.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("newBufferWithBytes:length:options: returned nil (float[])");
            }
            return buf;
        }
    }

    @Override
    public void close() {
        // Only per-instance resources are released here. ctx (device, queue,
        // library, pipeline state) is SHARED via CONTEXT_REGISTRY and
        // deliberately never torn down by an individual instance's close(),
        // exactly like OpenClCompositeExpression's ctx.
        try (Arena tmp = Arena.ofConfined()) {
            if (!inputDeviceF32.equals(MemorySegment.NULL)) {
                MTL.release(tmp, inputDeviceF32);
            }
            if (!outputDeviceF32.equals(MemorySegment.NULL)) {
                MTL.release(tmp, outputDeviceF32);
            }
            MTL.release(tmp, opcodesDevice);
            MTL.release(tmp, targetSlotsDevice);
            MTL.release(tmp, literalConstantsDeviceF32);
        } catch (Throwable t) {
            // best-effort cleanup
        } finally {
            arena.close();
        }
    }
}