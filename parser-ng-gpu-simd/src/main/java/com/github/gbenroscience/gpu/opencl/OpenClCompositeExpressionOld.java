
package com.github.gbenroscience.gpu.opencl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import com.github.gbenroscience.gpu.GpuCompositeExpression;

/**
 * Native dual-precision OpenCL evaluator. Two kernels ("interpret" for
 * double, "interpretF32" for float -- see OpenClKernelSource) are compiled
 * ONCE from a single combined program at class-init. Every double-path
 * call flows through the double kernel with double buffers throughout;
 * every float-path call flows through the float kernel with float buffers
 * throughout. There is no cross-precision conversion anywhere in either
 * call path -- that's what makes the float path a genuine throughput win
 * rather than a convenience wrapper: half the memory traffic per element,
 * and full-speed execution on GPUs with weak or absent fp64 hardware.
 */
public final class OpenClCompositeExpressionOld implements GpuCompositeExpression {

    /**
     * Common GPU vendors as a closed, typed choice -- the primary,
     * recommended way to call {@link #selectDevice}. Internally maps to
     * the same alias-expanded string matching {@link #selectDevice(String)}
     * uses (see {@link #expandVendorAliases}), so callers never need to
     * know or guess the actual driver-reported vendor string (which is
     * NOT standardized across vendors, operating systems, or driver
     * stacks -- e.g. AMD's is typically "Advanced Micro Devices, Inc.",
     * which doesn't even contain the substring "amd").
     *
     * Covers only the three vendors common enough to bake in as an enum.
     * For anything else (Apple, Qualcomm Adreno, ARM Mali, an exotic Mesa
     * driver name, etc.) use {@link #selectDevice(String)} directly with
     * whatever substring you find via {@link #listAvailableDevices()}.
     */
    public enum GpuVendor {
        AMD, INTEL, NVIDIA
    }

    /** One enumerated (platform, device) pair with its human-readable identity. */
    private record Candidate(MemorySegment platform, MemorySegment device,
            int platformIndex, int deviceIndex, String platformName,
            String deviceVendor, String deviceName) {
        String describe() {
            return "[platform " + platformIndex + ": " + platformName + "] "
                    + "[device " + deviceIndex + ": " + deviceVendor + " " + deviceName + "]";
        }
    }

    /**
     * Lists every GPU OpenCL can currently see on this machine, across
     * every installed platform, as plain human-readable descriptions --
     * e.g. {@code "[platform 0: Intel(R) OpenCL Graphics] [device 0:
     * Intel(R) Corporation Intel(R) Iris(R) Xe Graphics]"}.
     *
     * This is the answer to "how do I know what string to pass
     * selectDevice" -- don't guess, call this first (e.g. print it at
     * startup, or show it in a settings UI) and either pick a
     * {@link GpuVendor} that matches what you see, or copy an exact
     * substring into {@link #selectDevice(String)}.
     *
     * Deliberately independent of {@link GpuContext}: this method does its
     * own lightweight platform/device enumeration via a fresh
     * {@link OpenClBindings} instance and touches nothing that would
     * trigger the heavy bootstrap (context creation, program build, kernel
     * compile) or set the "already bootstrapped" flag {@link #selectDevice}
     * checks. Safe to call any number of times, at any point, purely for
     * inspection -- it never commits to a device.
     */
    public static java.util.List<String> listAvailableDevices() {
        try (Arena arena = Arena.ofConfined()) {
            OpenClBindings cl = new OpenClBindings();
            java.util.List<Candidate> candidates = enumerateGpuCandidates(cl, arena);
            java.util.List<String> descriptions = new java.util.ArrayList<>();
            for (Candidate c : candidates) {
                descriptions.add(c.describe());
            }
            return descriptions;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to enumerate OpenCL GPU devices", t);
        }
    }

    /**
     * Enumerates every (platform, GPU device) pair across ALL installed
     * OpenCL platforms -- not just platform index 0. With multiple vendors
     * installed (e.g. Intel's OpenCL runtime and AMD's), each typically
     * registers as its own platform, each exposing its own GPU device(s).
     * clGetDeviceIDs returning CL_DEVICE_NOT_FOUND is a normal "zero
     * matching devices on this platform" outcome during enumeration, not a
     * real error -- handled as such below, unlike every other OpenCL call
     * in this class where any non-CL_SUCCESS status is fatal.
     *
     * Takes an explicit OpenClBindings/Arena rather than reaching for
     * GpuContext's -- callable both from GpuContext's own bootstrap AND
     * from the standalone {@link #listAvailableDevices()}, without either
     * one depending on the other's lifecycle.
     */
    private static java.util.List<Candidate> enumerateGpuCandidates(OpenClBindings cl, Arena arena) throws Throwable {
        MemorySegment countBuf = arena.allocate(ValueLayout.JAVA_INT);

        check((int) cl.clGetPlatformIDs.invoke(0, MemorySegment.NULL, countBuf), "clGetPlatformIDs(count)");
        int platformCount = countBuf.get(ValueLayout.JAVA_INT, 0);
        if (platformCount < 1) {
            throw new IllegalStateException(
                    "No OpenCL platforms found (ICD loader present but no vendor registered)");
        }

        MemorySegment platformArr = arena.allocate(ValueLayout.ADDRESS, platformCount);
        check((int) cl.clGetPlatformIDs.invoke(platformCount, platformArr, MemorySegment.NULL),
                "clGetPlatformIDs(list)");

        java.util.List<Candidate> candidates = new java.util.ArrayList<>();

        for (int p = 0; p < platformCount; p++) {
            MemorySegment platform = platformArr.getAtIndex(ValueLayout.ADDRESS, p);
            String platformName = getPlatformInfoString(cl, arena, platform, OpenClBindings.CL_PLATFORM_NAME);

            int deviceCountStatus = (int) cl.clGetDeviceIDs.invoke(
                    platform, OpenClBindings.CL_DEVICE_TYPE_GPU, 0, MemorySegment.NULL, countBuf);
            if (deviceCountStatus == OpenClBindings.CL_DEVICE_NOT_FOUND) {
                continue; // this platform simply has no GPU device -- normal, not an error
            }
            check(deviceCountStatus, "clGetDeviceIDs(count) on platform " + p);
            int deviceCount = countBuf.get(ValueLayout.JAVA_INT, 0);
            if (deviceCount < 1) {
                continue;
            }

            MemorySegment deviceArr = arena.allocate(ValueLayout.ADDRESS, deviceCount);
            check((int) cl.clGetDeviceIDs.invoke(platform, OpenClBindings.CL_DEVICE_TYPE_GPU,
                    deviceCount, deviceArr, MemorySegment.NULL), "clGetDeviceIDs(list) on platform " + p);

            for (int d = 0; d < deviceCount; d++) {
                MemorySegment device = deviceArr.getAtIndex(ValueLayout.ADDRESS, d);
                String vendor = getDeviceInfoString(cl, arena, device, OpenClBindings.CL_DEVICE_VENDOR);
                String name = getDeviceInfoString(cl, arena, device, OpenClBindings.CL_DEVICE_NAME);
                candidates.add(new Candidate(platform, device, p, d, platformName, vendor, name));
            }
        }

        return candidates;
    }

    /**
     * Selection precedence, most to least specific:
     *   1. -Dopencl.platform.index=P (optionally + -Dopencl.device.index=D,
     *      default 0 within that platform) -- exact control for anyone
     *      who has already enumerated their system and knows the indices.
     *   2. -Dopencl.gpu.vendor=<substring> -- e.g. "AMD", "Intel", "NVIDIA"
     *      (or set programmatically via selectDevice(GpuVendor) /
     *      selectDevice(String)), matched case-insensitively, WITH known
     *      alias expansion (see expandVendorAliases), against BOTH the
     *      device's vendor string and its name. First match wins.
     *   3. Default (neither property set): the first candidate found, in
     *      enumeration order -- i.e. the first platform that actually
     *      exposes a GPU, first device on it. This is the "just work on
     *      whatever's there" behavior for a single-GPU machine.
     */
    private static Candidate selectCandidate(java.util.List<Candidate> candidates) {
        String platformIndexProp = System.getProperty("opencl.platform.index");
        if (platformIndexProp != null) {
            int platformIndex = Integer.parseInt(platformIndexProp.trim());
            int deviceIndex = Integer.getInteger("opencl.device.index", 0);
            for (Candidate c : candidates) {
                if (c.platformIndex() == platformIndex && c.deviceIndex() == deviceIndex) {
                    return c;
                }
            }
            throw new IllegalStateException(
                    "No GPU found at opencl.platform.index=" + platformIndex
                    + ", opencl.device.index=" + deviceIndex + ". Available devices:\n"
                    + describeAll(candidates));
        }

        String vendorProp = System.getProperty("opencl.gpu.vendor");
        if (vendorProp != null && !vendorProp.isBlank()) {
            String needle = vendorProp.trim().toLowerCase(java.util.Locale.ROOT);
            java.util.List<String> needles = expandVendorAliases(needle);
            for (Candidate c : candidates) {
                String vendor = c.deviceVendor().toLowerCase(java.util.Locale.ROOT);
                String name = c.deviceName().toLowerCase(java.util.Locale.ROOT);
                for (String alt : needles) {
                    if (matchesAlias(vendor, alt) || matchesAlias(name, alt)) {
                        return c;
                    }
                }
            }
            throw new IllegalStateException(
                    "No GPU matching opencl.gpu.vendor=\"" + vendorProp + "\" found "
                    + "(also tried aliases " + needles + "). Available devices:\n"
                    + describeAll(candidates));
        }

        return candidates.get(0);
    }

    /**
     * Real vendor strings from clGetDeviceInfo(CL_DEVICE_VENDOR) rarely
     * match the short name people actually type, and are NOT standardized
     * across vendors, operating systems, or driver stacks. Notably: AMD's
     * is typically "Advanced Micro Devices, Inc." -- which does NOT
     * contain the literal substring "amd" anywhere -- so a plain
     * needle.contains("amd") check silently matches nothing on real AMD
     * hardware, throwing "no GPU matching" even though an AMD GPU is right
     * there in the candidate list. Expanding a short vendor name to its
     * known long-form/marketing aliases (checked against both vendor AND
     * device name, see selectCandidate) closes that gap without requiring
     * the caller to know the exact driver string -- this is exactly what
     * {@link GpuVendor} is built on top of. Not exhaustive -- add entries
     * here as other mismatches surface, or use listAvailableDevices() plus
     * selectDevice(String) with an exact substring for anything not
     * covered.
     */
    private static java.util.List<String> expandVendorAliases(String needle) {
        java.util.List<String> aliases = new java.util.ArrayList<>();
        aliases.add(needle);
        switch (needle) {
            case "amd" -> aliases.addAll(java.util.List.of(
                    "advanced micro devices", "radeon", "ati"));
            case "intel" -> aliases.addAll(java.util.List.of(
                    "iris", "arc", "uhd graphics"));
            case "nvidia" -> aliases.addAll(java.util.List.of(
                    "geforce", "quadro", "tesla", "rtx", "gtx"));
            default -> { /* use the literal needle as typed, no known aliases */ }
        }
        return aliases;
    }

    /**
     * Whether {@code alias} appears in {@code haystack} as a standalone
     * token, not merely as a substring embedded inside a larger word.
     *
     * CONFIRMED BUG THIS FIXES: a plain {@code haystack.contains(alias)}
     * check let the short AMD alias "ati" (from "ATI Technologies", AMD's
     * older brand) match INSIDE THE WORD "Corporation" -- "Intel(R)
     * Corporation", lowercased, contains the literal substring "ati" at
     * "corpor-ATI-on". Since selectCandidate() returns the FIRST candidate
     * that matches any alias, and Intel happened to enumerate before AMD
     * on the affected machine, selectDevice(GpuVendor.AMD) was silently
     * selecting the Intel device instead -- confirmed empirically: the
     * enum selector picked Intel, while the exact literal vendor string
     * ("Advanced Micro Devices, Inc.") correctly picked AMD, since that
     * full-string comparison never went through alias expansion at all.
     *
     * \b in the regex requires an actual transition between a word
     * character and a non-word character (or a string boundary) on each
     * side of the match. "ati" inside "corporation" has ordinary letters
     * on both sides (...por-ati-on...) -- no transition, no boundary, so
     * this correctly refuses to match there, while still correctly
     * matching a standalone occurrence like "ATI" in a hypothetical
     * "ATI Technologies Inc." vendor string (bounded by the string start
     * and a space). Multi-word aliases (e.g. "advanced micro devices")
     * work the same way -- the \b anchors only apply to the outer edges
     * of the whole quoted phrase, not between the individual words inside
     * it, which is exactly the intended behavior.
     *
     * This closes the whole CLASS of bug, not just the one alias that
     * happened to trigger it -- any future short/generic alias is safe
     * from the same kind of accidental mid-word collision.
     */
    private static boolean matchesAlias(String haystackLower, String aliasLower) {
        return java.util.regex.Pattern
                .compile("\\b" + java.util.regex.Pattern.quote(aliasLower) + "\\b")
                .matcher(haystackLower)
                .find();
    }

    private static String describeAll(java.util.List<Candidate> candidates) {
        StringBuilder sb = new StringBuilder();
        for (Candidate c : candidates) {
            sb.append("  ").append(c.describe()).append('\n');
        }
        return sb.toString();
    }

    private static String getPlatformInfoString(OpenClBindings cl, Arena arena, MemorySegment platform, int paramName) throws Throwable {
        MemorySegment sizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        check((int) cl.clGetPlatformInfo.invoke(platform, paramName, 0L, MemorySegment.NULL, sizeBuf),
                "clGetPlatformInfo(size)");
        long size = sizeBuf.get(ValueLayout.JAVA_LONG, 0);
        MemorySegment buf = arena.allocate(Math.max(size, 1));
        check((int) cl.clGetPlatformInfo.invoke(platform, paramName, size, buf, MemorySegment.NULL),
                "clGetPlatformInfo(value)");
        return buf.getString(0, StandardCharsets.UTF_8);
    }

    private static String getDeviceInfoString(OpenClBindings cl, Arena arena, MemorySegment device, int paramName) throws Throwable {
        MemorySegment sizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        check((int) cl.clGetDeviceInfo.invoke(device, paramName, 0L, MemorySegment.NULL, sizeBuf),
                "clGetDeviceInfo(size)");
        long size = sizeBuf.get(ValueLayout.JAVA_LONG, 0);
        MemorySegment buf = arena.allocate(Math.max(size, 1));
        check((int) cl.clGetDeviceInfo.invoke(device, paramName, size, buf, MemorySegment.NULL),
                "clGetDeviceInfo(value)");
        return buf.getString(0, StandardCharsets.UTF_8);
    }

    private static final class GpuContext {

        static final OpenClBindings CL = new OpenClBindings();
        static final MemorySegment PLATFORM;
        static final MemorySegment DEVICE;
        static final MemorySegment CONTEXT;
        static final MemorySegment QUEUE;
        static final MemorySegment PROGRAM;
        static final MemorySegment KERNEL_F64;
        static final MemorySegment KERNEL_F32;
        static final String SELECTED_DEVICE_DESCRIPTION;

        static {
            try (Arena bootstrap = Arena.ofConfined()) {
                java.util.List<Candidate> candidates = enumerateGpuCandidates(CL, bootstrap);
                if (candidates.isEmpty()) {
                    throw new IllegalStateException(
                            "No OpenCL GPU devices found on any platform (ICD loader present but "
                            + "no platform exposed a GPU device)");
                }

                Candidate chosen = selectCandidate(candidates);
                SELECTED_DEVICE_DESCRIPTION = chosen.describe();
                PLATFORM = chosen.platform();
                DEVICE = chosen.device();

                MemorySegment errBuf = bootstrap.allocate(ValueLayout.JAVA_INT);

                MemorySegment devicesForCtx = bootstrap.allocate(ValueLayout.ADDRESS);
                devicesForCtx.set(ValueLayout.ADDRESS, 0, DEVICE);
                CONTEXT = (MemorySegment) CL.clCreateContext.invoke(
                        MemorySegment.NULL, 1, devicesForCtx,
                        MemorySegment.NULL, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateContext");

                QUEUE = (MemorySegment) CL.clCreateCommandQueue.invoke(CONTEXT, DEVICE, 0L, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateCommandQueue");

                // ONE combined source, ONE build -- both kernel entry points
                // (interpret/interpretF32) come out of this single program.
                MemorySegment src = bootstrap.allocateFrom(OpenClKernelSource.OPENCL_SOURCE);

                MemorySegment srcPtrArr = bootstrap.allocate(ValueLayout.ADDRESS);
                srcPtrArr.set(ValueLayout.ADDRESS, 0, src);

                PROGRAM = (MemorySegment) CL.clCreateProgramWithSource.invoke(
                        CONTEXT, 1, srcPtrArr, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateProgramWithSource");

                int buildStatus = (int) CL.clBuildProgram.invoke(PROGRAM, 1, devicesForCtx,
                        MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL);
                if (buildStatus != OpenClBindings.CL_SUCCESS) {
                    throw new IllegalStateException(
                            "OpenCL build failed (" + buildStatus + ") on " + SELECTED_DEVICE_DESCRIPTION
                            + ": " + fetchBuildLog(bootstrap));
                }

                MemorySegment kernelNameF64 = bootstrap.allocateFrom(
                        OpenClKernelSource.KERNEL_NAME_F64, StandardCharsets.UTF_8);
                KERNEL_F64 = (MemorySegment) CL.clCreateKernel.invoke(PROGRAM, kernelNameF64, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateKernel(interpret)");

                MemorySegment kernelNameF32 = bootstrap.allocateFrom(
                        OpenClKernelSource.KERNEL_NAME_F32, StandardCharsets.UTF_8);
                KERNEL_F32 = (MemorySegment) CL.clCreateKernel.invoke(PROGRAM, kernelNameF32, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateKernel(interpretF32)");

                // Printed once at bootstrap so it's obvious from program
                // output which of possibly several installed GPUs got
                // picked -- especially useful with a multi-vendor system
                // (e.g. Intel iGPU + AMD discrete) where silence here would
                // leave the choice invisible until something went wrong.
                System.err.println("[ParserNG GPU] OpenCL using " + SELECTED_DEVICE_DESCRIPTION);

                // Must be the LAST statement in this block, and only reached
                // on a fully successful bootstrap -- this is exactly the
                // flag selectDevice()'s guard checks to refuse late calls.
                gpuContextBootstrapped = true;

            } catch (Throwable t) {
                throw new ExceptionInInitializerError(t);
            }
        }

        private static String fetchBuildLog(Arena arena) throws Throwable {
            MemorySegment sizeRet = arena.allocate(ValueLayout.JAVA_LONG);
            CL.clGetProgramBuildInfo.invoke(PROGRAM, DEVICE, OpenClBindings.CL_PROGRAM_BUILD_LOG,
                    0L, MemorySegment.NULL, sizeRet);
            long logSize = sizeRet.get(ValueLayout.JAVA_LONG, 0);
            if (logSize <= 0) {
                return "(no build log)";
            }

            MemorySegment logBuf = arena.allocate(logSize);

            CL.clGetProgramBuildInfo.invoke(PROGRAM, DEVICE, OpenClBindings.CL_PROGRAM_BUILD_LOG,
                    logSize, logBuf, MemorySegment.NULL);

            return logBuf.getString(0, StandardCharsets.UTF_8);
        }

        private static void check(int status, String call) {
            if (status != OpenClBindings.CL_SUCCESS) {
                throw new IllegalStateException("OpenCL error in " + call + ": code " + status);
            }
        }
    }

    // GpuContext.KERNEL_F64/KERNEL_F32/QUEUE are process-wide singletons
    // (one compiled program, reused by every OpenClCompositeExpressionOld
    // instance). Setting kernel args and enqueueing the dispatch that
    // consumes them is a check-then-act sequence on whichever kernel
    // object is in play, so it must be serialized -- otherwise two threads
    // calling applyBulk concurrently can interleave clSetKernelArg calls
    // and each read back the other's operands/results. One lock covers
    // both kernels: simpler and still correct, at the cost of serializing
    // double- and float-path dispatches against each other too (a
    // deliberate simplicity/throughput tradeoff, not an oversight -- split
    // into two locks later if double/float dispatches from different
    // threads turn out to need to run concurrently).
    private static final Object DISPATCH_LOCK = new Object();

    // Set as the LAST action of GpuContext's static initializer (see
    // GpuContext's static block, bottom). Deliberately lives on THIS class,
    // not inside GpuContext -- merely reading this flag must never itself
    // trigger GpuContext's class-init, or the guard below couldn't
    // distinguish "not yet bootstrapped" from "about to bootstrap because
    // you just asked".
    private static volatile boolean gpuContextBootstrapped = false;

    /**
     * Programmatic alternative to -Dopencl.gpu.vendor=... -- selects which
     * GPU the OpenCL backend will bootstrap onto, by a case-insensitive
     * substring match against the device's vendor or name (e.g. "AMD",
     * "Intel", "Radeon", "NVIDIA").
     *
     * MUST be called before the first use of any GPU-facing class in this
     * JVM (OpenClCompositeExpressionOld, OpenClExpressionBridge,
     * GpuExpressionBridge, or GpuExpressionBridge.isAvailable(...) probing
     * a backend) -- device selection is a one-time, process-wide decision
     * resolved inside GpuContext's static initializer, which runs exactly
     * once, lazily, the first time anything actually touches it. Calling
     * this method after that point throws IllegalStateException rather
     * than silently doing nothing: a setter that only works if you call it
     * early, with no signal when you didn't, would be worse than the plain
     * system property it wraps.
     *
     * Practically: call this as close to the start of main() as you can,
     * before any GPU work, and don't call it from more than one thread --
     * there's no lock between "this method runs" and "some other thread's
     * first GPU call triggers the static init that reads it", so treat
     * device selection as single-threaded startup configuration, not a
     * runtime toggle.
     *
     * This selects the GPU for the WHOLE JVM (GpuContext is a
     * process-wide singleton, shared by every OpenClCompositeExpressionOld
     * instance) -- it is not a per-call or per-instance choice. Running
     * different expressions on different GPUs simultaneously within one
     * JVM would need a real architectural change (GpuContext as a registry
     * keyed by device, not a singleton), which this method does not
     * attempt.
     */
    public static void selectDevice(String vendorOrNameSubstring) {
        requireNotYetBootstrapped();
        System.setProperty("opencl.gpu.vendor", vendorOrNameSubstring);
    }

    /**
     * Recommended entry point over {@link #selectDevice(String)} for the
     * common case: picks by a closed, typed vendor choice instead of a raw
     * string, so the caller never needs to know (or guess, or look up)
     * what the installed driver actually reports via clGetDeviceInfo --
     * see {@link GpuVendor}'s javadoc for why that string is not something
     * to assume a caller, however technical, would already know. Same
     * timing rule and same process-wide scope as {@link #selectDevice(String)}
     * apply (this just delegates to it).
     */
    public static void selectDevice(GpuVendor vendor) {
        selectDevice(vendor.name());
    }

    /**
     * Exact-index counterpart of {@link #selectDevice(String)} -- same
     * timing rules and same one-time, process-wide scope apply. Use once
     * you've already seen the available devices (e.g. from a prior run's
     * "no GPU matching..." error message, which lists them) and want to
     * pin an exact platform/device pair rather than match by name.
     * @param platformIndex
     * @param deviceIndex
     */
    public static void selectDevice(int platformIndex, int deviceIndex) {
        requireNotYetBootstrapped();
        System.setProperty("opencl.platform.index", String.valueOf(platformIndex));
        System.setProperty("opencl.device.index", String.valueOf(deviceIndex));
    }

    private static void requireNotYetBootstrapped() {
        if (gpuContextBootstrapped) {
            throw new IllegalStateException(
                    "Cannot select a GPU device -- the OpenCL backend has already bootstrapped onto "
                    + getSelectedDeviceDescription() + " earlier in this JVM run. selectDevice(...) "
                    + "must be called before the first use of any GPU-facing class (OpenClCompositeExpressionOld, "
                    + "OpenClExpressionBridge, GpuExpressionBridge). Device selection is a one-time, "
                    + "process-wide decision -- it cannot be changed after the fact without restarting the JVM.");
        }
    }

    /**
     * Which GPU (platform + device, vendor/name) ended up selected for the
     * OpenCL backend this JVM run -- e.g. to confirm "-Dopencl.gpu.vendor=AMD"
     * actually picked the AMD device and not the Intel one. Resolved once,
     * at class-init, same lifetime as everything else in GpuContext.
     */
    public static String getSelectedDeviceDescription() {
        return GpuContext.SELECTED_DEVICE_DESCRIPTION;
    }

    private final OpenClBindings cl = GpuContext.CL;

    private final Arena arena = Arena.ofShared();
    private final MemorySegment opcodesDevice;
    private final MemorySegment targetSlotsDevice;
    private final MemorySegment literalConstantsDevice;   // double, feeds KERNEL_F64
    private final MemorySegment literalConstantsDeviceF32; // float, feeds KERNEL_F32
    private final int instructionCount;
    private final int varCount;

    // ---- double-path device/staging state ----
    private MemorySegment inputDevice = MemorySegment.NULL;
    private MemorySegment outputDevice = MemorySegment.NULL;
    private MemorySegment stagingIn = MemorySegment.NULL;
    private MemorySegment stagingOut = MemorySegment.NULL;
    private long deviceInCapacityBytes = 0;
    private long deviceOutCapacityBytes = 0;
    private long stagingInCapacityBytes = 0;
    private long stagingOutCapacityBytes = 0;

    // ---- float-path device/staging state (fully independent of the double
    // ---- path above -- separate buffers, separate capacities, never
    // ---- shared or resized together) ----
    private MemorySegment inputDeviceF32 = MemorySegment.NULL;
    private MemorySegment outputDeviceF32 = MemorySegment.NULL;
    private MemorySegment stagingInF32 = MemorySegment.NULL;
    private MemorySegment stagingOutF32 = MemorySegment.NULL;
    private long deviceInCapacityBytesF32 = 0;
    private long deviceOutCapacityBytesF32 = 0;
    private long stagingInCapacityBytesF32 = 0;
    private long stagingOutCapacityBytesF32 = 0;

    public OpenClCompositeExpressionOld(int[] opcodes, int[] targetSlots, double[] literalConstants,
            int instructionCount, int varCount) {
        this.instructionCount = instructionCount;
        this.varCount = varCount;

        try {
            this.opcodesDevice = uploadIntArray(opcodes);
            this.targetSlotsDevice = uploadIntArray(targetSlots);
            this.literalConstantsDevice = uploadDoubleArray(literalConstants);

            // Converted ONCE, at construction, from the same source values
            // the double kernel uses -- not recomputed per call. The float
            // kernel's literalConstants buffer is genuinely float32 on
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
        // Slice down to the CURRENT call's exact length: stagingIn/stagingOut
        // are grow-only and may still hold a larger capacity left over from
        // a previous, bigger call. Passing the raw (oversized) segments into
        // dispatch() would make it infer dataSize from stale capacity rather
        // than this call's actual element count.
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
                check((int) cl.clEnqueueWriteBuffer.invoke(GpuContext.QUEUE, inputDevice,
                        OpenClBindings.CL_TRUE, 0L, in.byteSize(), in, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueWriteBuffer(in)");

                setKernelArgs(GpuContext.KERNEL_F64, literalConstantsDevice, inputDevice, outputDevice, dataSize);

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment globalWorkSize = tmp.allocate(ValueLayout.JAVA_LONG);
                    globalWorkSize.set(ValueLayout.JAVA_LONG, 0, (long) dataSize);

                    check((int) cl.clEnqueueNDRangeKernel.invoke(GpuContext.QUEUE, GpuContext.KERNEL_F64,
                            1, MemorySegment.NULL, globalWorkSize, MemorySegment.NULL,
                            0, MemorySegment.NULL, MemorySegment.NULL),
                            "clEnqueueNDRangeKernel(interpret)");
                }

                check((int) cl.clEnqueueReadBuffer.invoke(GpuContext.QUEUE, outputDevice,
                        OpenClBindings.CL_TRUE, 0L, out.byteSize(), out, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueReadBuffer(out)");

                check((int) cl.clFinish.invoke(GpuContext.QUEUE), "clFinish");
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
                check((int) cl.clEnqueueWriteBuffer.invoke(GpuContext.QUEUE, inputDeviceF32,
                        OpenClBindings.CL_TRUE, 0L, in.byteSize(), in, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueWriteBuffer(in, f32)");

                setKernelArgs(GpuContext.KERNEL_F32, literalConstantsDeviceF32,
                        inputDeviceF32, outputDeviceF32, dataSize);

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment globalWorkSize = tmp.allocate(ValueLayout.JAVA_LONG);
                    globalWorkSize.set(ValueLayout.JAVA_LONG, 0, (long) dataSize);

                    check((int) cl.clEnqueueNDRangeKernel.invoke(GpuContext.QUEUE, GpuContext.KERNEL_F32,
                            1, MemorySegment.NULL, globalWorkSize, MemorySegment.NULL,
                            0, MemorySegment.NULL, MemorySegment.NULL),
                            "clEnqueueNDRangeKernel(interpretF32)");
                }

                check((int) cl.clEnqueueReadBuffer.invoke(GpuContext.QUEUE, outputDeviceF32,
                        OpenClBindings.CL_TRUE, 0L, out.byteSize(), out, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueReadBuffer(out, f32)");

                check((int) cl.clFinish.invoke(GpuContext.QUEUE), "clFinish");
            } catch (Throwable t) {
                throw new RuntimeException("GPU dispatch failed (f32)", t);
            }
        }
    }

    // ================= shared kernel-arg plumbing (parameterized by which kernel/buffers) =================

    private void setKernelArgs(MemorySegment kernel, MemorySegment literalConstantsBuf,
            MemorySegment inBuf, MemorySegment outBuf, int dataSize) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(tmp, kernel, 0, opcodesDevice);
            setArgPtr(tmp, kernel, 1, targetSlotsDevice);
            setArgPtr(tmp, kernel, 2, literalConstantsBuf);
            setArgInt(tmp, kernel, 3, instructionCount);
            setArgPtr(tmp, kernel, 4, inBuf);
            setArgInt(tmp, kernel, 5, dataSize);
            setArgInt(tmp, kernel, 6, varCount);
            setArgPtr(tmp, kernel, 7, outBuf);
        }
    }

    private void setArgPtr(Arena tmp, MemorySegment kernel, int index, MemorySegment value) throws Throwable {
        MemorySegment holder = tmp.allocate(ValueLayout.ADDRESS);
        holder.set(ValueLayout.ADDRESS, 0, value);
        check((int) cl.clSetKernelArg.invoke(kernel, index,
                ValueLayout.ADDRESS.byteSize(), holder), "clSetKernelArg[" + index + "]");
    }

    private void setArgInt(Arena tmp, MemorySegment kernel, int index, int value) throws Throwable {
        MemorySegment holder = tmp.allocate(ValueLayout.JAVA_INT);
        holder.set(ValueLayout.JAVA_INT, 0, value);
        check((int) cl.clSetKernelArg.invoke(kernel, index,
                ValueLayout.JAVA_INT.byteSize(), holder), "clSetKernelArg[" + index + "]");
    }

    // ================= double-path buffer/staging management =================

    private void ensureDeviceBuffers(long inBytes, long outBytes) throws Throwable {
        // Input and output capacities are tracked and grown independently.
        // A single combined max(inBytes, outBytes) capacity is not safe:
        // e.g. call 1 = (in=800B, out=100B) sizes both buffers off a
        // required=800B; call 2 = (in=200B, out=750B) has required=750B
        // <= 800B, so growth would be skipped entirely even though the
        // *output* buffer is still only 100B -- an out-of-bounds
        // clEnqueueReadBuffer. Track each independently instead.
        boolean needsIn = inBytes > deviceInCapacityBytes || inputDevice.equals(MemorySegment.NULL);
        boolean needsOut = outBytes > deviceOutCapacityBytes || outputDevice.equals(MemorySegment.NULL);
        if (!needsIn && !needsOut) {
            return;
        }

        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);

            if (needsIn) {
                if (!inputDevice.equals(MemorySegment.NULL)) {
                    cl.clReleaseMemObject.invoke(inputDevice);
                }
                inputDevice = (MemorySegment) cl.clCreateBuffer.invoke(GpuContext.CONTEXT,
                        OpenClBindings.CL_MEM_READ_ONLY, inBytes, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(in)");
                deviceInCapacityBytes = inBytes;
            }

            if (needsOut) {
                if (!outputDevice.equals(MemorySegment.NULL)) {
                    cl.clReleaseMemObject.invoke(outputDevice);
                }
                outputDevice = (MemorySegment) cl.clCreateBuffer.invoke(GpuContext.CONTEXT,
                        OpenClBindings.CL_MEM_WRITE_ONLY, outBytes, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(out)");
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

    private void releaseDeviceBuffers() throws Throwable {
        if (!inputDevice.equals(MemorySegment.NULL)) {
            cl.clReleaseMemObject.invoke(inputDevice);
        }
        if (!outputDevice.equals(MemorySegment.NULL)) {
            cl.clReleaseMemObject.invoke(outputDevice);
        }
    }

    // ================= float-path buffer/staging management (independent of double path) =================

    private void ensureDeviceBuffersF32(long inBytes, long outBytes) throws Throwable {
        boolean needsIn = inBytes > deviceInCapacityBytesF32 || inputDeviceF32.equals(MemorySegment.NULL);
        boolean needsOut = outBytes > deviceOutCapacityBytesF32 || outputDeviceF32.equals(MemorySegment.NULL);
        if (!needsIn && !needsOut) {
            return;
        }

        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);

            if (needsIn) {
                if (!inputDeviceF32.equals(MemorySegment.NULL)) {
                    cl.clReleaseMemObject.invoke(inputDeviceF32);
                }
                inputDeviceF32 = (MemorySegment) cl.clCreateBuffer.invoke(GpuContext.CONTEXT,
                        OpenClBindings.CL_MEM_READ_ONLY, inBytes, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(in, f32)");
                deviceInCapacityBytesF32 = inBytes;
            }

            if (needsOut) {
                if (!outputDeviceF32.equals(MemorySegment.NULL)) {
                    cl.clReleaseMemObject.invoke(outputDeviceF32);
                }
                outputDeviceF32 = (MemorySegment) cl.clCreateBuffer.invoke(GpuContext.CONTEXT,
                        OpenClBindings.CL_MEM_WRITE_ONLY, outBytes, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(out, f32)");
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

    private void releaseDeviceBuffersF32() throws Throwable {
        if (!inputDeviceF32.equals(MemorySegment.NULL)) {
            cl.clReleaseMemObject.invoke(inputDeviceF32);
        }
        if (!outputDeviceF32.equals(MemorySegment.NULL)) {
            cl.clReleaseMemObject.invoke(outputDeviceF32);
        }
    }

    // ================= program-buffer upload helpers =================

    private MemorySegment uploadIntArray(int[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_INT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);

            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
            MemorySegment device = (MemorySegment) cl.clCreateBuffer.invoke(GpuContext.CONTEXT,
                    OpenClBindings.CL_MEM_READ_ONLY, host.byteSize(), MemorySegment.NULL, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(int[])");

            check((int) cl.clEnqueueWriteBuffer.invoke(GpuContext.QUEUE, device, OpenClBindings.CL_TRUE,
                    0L, host.byteSize(), host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueWriteBuffer(int[])");
            return device;
        }
    }

    private MemorySegment uploadDoubleArray(double[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_DOUBLE.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_DOUBLE, 0, data.length);

            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
            MemorySegment device = (MemorySegment) cl.clCreateBuffer.invoke(GpuContext.CONTEXT,
                    OpenClBindings.CL_MEM_READ_ONLY, host.byteSize(), MemorySegment.NULL, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(double[])");

            check((int) cl.clEnqueueWriteBuffer.invoke(GpuContext.QUEUE, device, OpenClBindings.CL_TRUE,
                    0L, host.byteSize(), host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueWriteBuffer(double[])");
            return device;
        }
    }

    private MemorySegment uploadFloatArray(float[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);

            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
            MemorySegment device = (MemorySegment) cl.clCreateBuffer.invoke(GpuContext.CONTEXT,
                    OpenClBindings.CL_MEM_READ_ONLY, host.byteSize(), MemorySegment.NULL, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(float[])");

            check((int) cl.clEnqueueWriteBuffer.invoke(GpuContext.QUEUE, device, OpenClBindings.CL_TRUE,
                    0L, host.byteSize(), host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueWriteBuffer(float[])");
            return device;
        }
    }

    private static void check(int status, String call) {
        if (status != OpenClBindings.CL_SUCCESS) {
            throw new IllegalStateException("OpenCL error in " + call + ": code " + status);
        }
    }

    @Override
    public void close() {
        try {
            releaseDeviceBuffers();
            releaseDeviceBuffersF32();
            cl.clReleaseMemObject.invoke(opcodesDevice);
            cl.clReleaseMemObject.invoke(targetSlotsDevice);
            cl.clReleaseMemObject.invoke(literalConstantsDevice);
            cl.clReleaseMemObject.invoke(literalConstantsDeviceF32);
        } catch (Throwable t) {
            // best-effort cleanup
        } finally {
            arena.close();
        }
    }
}