package com.github.gbenroscience.gpu.evaluator.opencl;


import com.github.gbenroscience.gpu.evaluator.GpuCompositeExpression;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;


/**
 * Native dual-precision OpenCL evaluator. Two kernels ("interpret" for
 * double, "interpretF32" for float -- see OpenClKernelSource) are compiled
 * ONCE per distinct GPU device the first time that device is used. Every
 * double-path call flows through the double kernel with double buffers
 * throughout; every float-path call flows through the float kernel with
 * float buffers throughout. There is no cross-precision conversion anywhere
 * in either call path -- that's what makes the float path a genuine
 * throughput win rather than a convenience wrapper: half the memory traffic
 * per element, and full-speed execution on GPUs with weak or absent fp64
 * hardware.
 *
 * DEVICE BINDING MODEL: each {@link OpenClCompositeExpression} instance is
 * bound, at construction time, to whichever device {@link #selectDevice}
 * currently resolves to. That binding is fixed for the instance's whole
 * lifetime -- an in-flight expression's GPU never changes under it. The
 * underlying per-device resources (context, command queue, compiled
 * program, both kernels) are cached in a small registry keyed by device and
 * shared by every instance bound to that device, so switching selection
 * back and forth (e.g. across test methods -- "AMD" for this test, "Intel"
 * for that one) does NOT rebuild/recompile anything after the first time
 * each distinct device is used. This is the direct fix for device selection
 * previously being a one-shot, JVM-wide decision: now it's "which device
 * will the NEXT constructed instance use", not "which device may this JVM
 * ever use, once, forever".
 */
public final class OpenClCompositeExpression implements GpuCompositeExpression {

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

        /** Stable registry key for this exact (platform, device) pair. */
        String registryKey() {
            return platformIndex + ":" + deviceIndex;
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
     * Deliberately independent of the context registry: this method does
     * its own lightweight platform/device enumeration and never builds a
     * context, compiles a program, or touches the registry -- safe to call
     * any number of times, at any point, purely for inspection.
     */
    public static java.util.List<String> listAvailableDevices() {
        try (Arena arena = Arena.ofConfined()) {
            java.util.List<Candidate> candidates = enumerateGpuCandidates(CL, arena);
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
     *
     * Read fresh, every time a context is resolved -- see resolveContext().
     * This is what makes selectDevice() affect only instances constructed
     * after it's called, rather than a single JVM-wide decision.
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
     * "corpor-ATI-on". \b requires an actual transition between a word
     * character and a non-word character (or a string boundary) on each
     * side of the match, so this correctly refuses to match there while
     * still correctly matching a standalone occurrence like "ATI" in a
     * hypothetical "ATI Technologies Inc." vendor string. Multi-word
     * aliases (e.g. "advanced micro devices") work the same way -- the \b
     * anchors only apply to the outer edges of the whole quoted phrase.
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

    // ================= per-device context registry =================

    /**
     * One driver-facing binding set, shared by every instance and every
     * device (the OpenCL ICD loader itself routes each call to the correct
     * vendor driver based on the platform/context/device handle passed in
     * at call time -- one resolved set of function pointers works for
     * every installed vendor simultaneously, nothing device-specific here).
     */
    private static final OpenClBindings CL = new OpenClBindings();

    /**
     * The GPU context registry: one entry per distinct (platform, device)
     * pair actually used so far, built the FIRST time that device is
     * selected and reused (context, queue, compiled program, both kernels)
     * by every instance bound to it afterward. This is what makes
     * repeated selectDevice() calls cheap after the first use of each
     * device -- switching back to a previously-used device never
     * recompiles anything.
     */
    private static final ConcurrentHashMap<String, GpuContext> CONTEXT_REGISTRY = new ConcurrentHashMap<>();

    /**
     * Resolves (building and caching if necessary) the GpuContext for
     * whichever device selectCandidate() currently points at. Called once
     * per OpenClCompositeExpression construction -- the resolved context
     * is then fixed for that instance's entire lifetime.
     */
    private static GpuContext resolveContext() {
        try (Arena arena = Arena.ofConfined()) {
            java.util.List<Candidate> candidates = enumerateGpuCandidates(CL, arena);
            if (candidates.isEmpty()) {
                throw new IllegalStateException(
                        "No OpenCL GPU devices found on any platform (ICD loader present but "
                        + "no platform exposed a GPU device)");
            }
            Candidate chosen = selectCandidate(candidates);
            return CONTEXT_REGISTRY.computeIfAbsent(chosen.registryKey(), k -> buildContext(chosen));
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) {
                throw re;
            }
            throw new RuntimeException("Failed to resolve an OpenCL GPU context", t);
        }
    }

    private static GpuContext buildContext(Candidate chosen) {
        try {
            return new GpuContext(CL, chosen);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to bootstrap OpenCL context for " + chosen.describe(), t);
        }
    }

    /**
     * Everything needed to dispatch against ONE specific GPU device:
     * context, command queue, the compiled program, both kernel handles,
     * and a dispatch lock scoped to just this device. Previously a
     * JVM-wide singleton (one GpuContext, ever); now one instance per
     * distinct (platform, device) pair actually used, cached in
     * CONTEXT_REGISTRY and shared by every OpenClCompositeExpression bound
     * to that device.
     */
    private static final class GpuContext {

        final MemorySegment platform;
        final MemorySegment device;
        final MemorySegment context;
        final MemorySegment queue;
        final MemorySegment program;
        final MemorySegment kernelF64;
        final MemorySegment kernelF32;
        final String selectedDeviceDescription;

        // GpuContext instances are shared by every OpenClCompositeExpression
        // bound to the same device. Setting kernel args and enqueueing the
        // dispatch that consumes them is a check-then-act sequence on this
        // context's kernel objects, so it must be serialized per-device --
        // otherwise two threads dispatching against the SAME device could
        // interleave clSetKernelArg calls and each read back the other's
        // operands/results. Scoped per-GpuContext (not one global lock
        // across every device) so dispatches to DIFFERENT devices -- e.g.
        // one thread on AMD, another on Intel -- don't serialize against
        // each other unnecessarily.
        final Object dispatchLock = new Object();

        GpuContext(OpenClBindings cl, Candidate chosen) throws Throwable {
            try (Arena bootstrap = Arena.ofConfined()) {
                this.selectedDeviceDescription = chosen.describe();
                this.platform = chosen.platform();
                this.device = chosen.device();

                MemorySegment errBuf = bootstrap.allocate(ValueLayout.JAVA_INT);

                MemorySegment devicesForCtx = bootstrap.allocate(ValueLayout.ADDRESS);
                devicesForCtx.set(ValueLayout.ADDRESS, 0, device);
                this.context = (MemorySegment) cl.clCreateContext.invoke(
                        MemorySegment.NULL, 1, devicesForCtx,
                        MemorySegment.NULL, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateContext");

                this.queue = (MemorySegment) cl.clCreateCommandQueue.invoke(context, device, 0L, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateCommandQueue");

                // ONE combined source, ONE build -- both kernel entry points
                // (interpret/interpretF32) come out of this single program.
                MemorySegment src = bootstrap.allocateFrom(OpenClKernelSource.OPENCL_SOURCE);

                MemorySegment srcPtrArr = bootstrap.allocate(ValueLayout.ADDRESS);
                srcPtrArr.set(ValueLayout.ADDRESS, 0, src);

                this.program = (MemorySegment) cl.clCreateProgramWithSource.invoke(
                        context, 1, srcPtrArr, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateProgramWithSource");

                int buildStatus = (int) cl.clBuildProgram.invoke(program, 1, devicesForCtx,
                        MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL);
                if (buildStatus != OpenClBindings.CL_SUCCESS) {
                    throw new IllegalStateException(
                            "OpenCL build failed (" + buildStatus + ") on " + selectedDeviceDescription
                            + ": " + fetchBuildLog(cl, bootstrap, program, device));
                }

                MemorySegment kernelNameF64 = bootstrap.allocateFrom(
                        OpenClKernelSource.KERNEL_NAME_F64, StandardCharsets.UTF_8);
                this.kernelF64 = (MemorySegment) cl.clCreateKernel.invoke(program, kernelNameF64, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateKernel(interpret)");

                MemorySegment kernelNameF32 = bootstrap.allocateFrom(
                        OpenClKernelSource.KERNEL_NAME_F32, StandardCharsets.UTF_8);
                this.kernelF32 = (MemorySegment) cl.clCreateKernel.invoke(program, kernelNameF32, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateKernel(interpretF32)");

                // Printed once per DISTINCT device the first time it's
                // built (not once per instance -- computeIfAbsent in
                // resolveContext ensures this constructor only runs once
                // per registry key), so it's obvious from program output
                // which of possibly several installed GPUs is in play.
                System.err.println("[ParserNG GPU] OpenCL using " + selectedDeviceDescription);
            }
        }
    }

    private static String fetchBuildLog(OpenClBindings cl, Arena arena, MemorySegment program, MemorySegment device) throws Throwable {
        MemorySegment sizeRet = arena.allocate(ValueLayout.JAVA_LONG);
        cl.clGetProgramBuildInfo.invoke(program, device, OpenClBindings.CL_PROGRAM_BUILD_LOG,
                0L, MemorySegment.NULL, sizeRet);
        long logSize = sizeRet.get(ValueLayout.JAVA_LONG, 0);
        if (logSize <= 0) {
            return "(no build log)";
        }

        MemorySegment logBuf = arena.allocate(logSize);

        cl.clGetProgramBuildInfo.invoke(program, device, OpenClBindings.CL_PROGRAM_BUILD_LOG,
                logSize, logBuf, MemorySegment.NULL);

        return logBuf.getString(0, StandardCharsets.UTF_8);
    }

    // ================= device selection =================

    /**
     * Programmatic alternative to -Dopencl.gpu.vendor=... -- selects which
     * GPU device the NEXT constructed OpenClCompositeExpression (or the
     * next auto-detection probe, or the next OpenClExpressionBridge/
     * GpuExpressionBridge call) will bind to, by a case-insensitive
     * substring match against the device's vendor or name (e.g. "AMD",
     * "Intel", "Radeon", "NVIDIA").
     *
     * Safe to call as many times as you like, at any point in a JVM's
     * lifetime -- selection is resolved fresh on every construction (see
     * resolveContext()), not locked in once and forever. Calling this
     * between two test methods to target a different GPU for each is a
     * fully supported pattern: the FIRST time a given device is selected,
     * its context/program/kernels are built and cached; every later
     * selectDevice(...) call back to that same device reuses the cached
     * context rather than rebuilding it.
     *
     * What this does NOT do: change the device an ALREADY-CONSTRUCTED
     * instance is using. Device binding is fixed per-instance at
     * construction time and never changes afterward -- an in-flight
     * expression's GPU never moves under it. This also isn't a per-call
     * or per-thread setting: it's a plain JVM system property under the
     * hood, so don't call it concurrently from one thread while another
     * thread is mid-construction expecting a different device -- treat it
     * as "set the default for whatever gets constructed next", called from
     * one thread at a time.
     */
    public static void selectDevice(String vendorOrNameSubstring) {
        System.setProperty("opencl.gpu.vendor", vendorOrNameSubstring);
    }

    /**
     * Recommended entry point over {@link #selectDevice(String)} for the
     * common case: picks by a closed, typed vendor choice instead of a raw
     * string, so the caller never needs to know (or guess, or look up)
     * what the installed driver actually reports via clGetDeviceInfo --
     * see {@link GpuVendor}'s javadoc for why that string is not something
     * to assume a caller, however technical, would already know. Same
     * rules as {@link #selectDevice(String)} apply (this just delegates
     * to it).
     */
    public static void selectDevice(GpuVendor vendor) {
        selectDevice(vendor.name());
    }

    /**
     * Exact-index counterpart of {@link #selectDevice(String)} -- same
     * rules apply. Use once you've already seen the available devices
     * (e.g. via {@link #listAvailableDevices()}, or from a prior
     * "no GPU matching..." error message, which lists them) and want to
     * pin an exact platform/device pair rather than match by name.
     */
    public static void selectDevice(int platformIndex, int deviceIndex) {
        System.setProperty("opencl.platform.index", String.valueOf(platformIndex));
        System.setProperty("opencl.device.index", String.valueOf(deviceIndex));
    }

    /**
     * Clears any explicit selection previously set via
     * {@link #selectDevice}, reverting to the default (first platform that
     * exposes a GPU, first device on it) for instances constructed after
     * this call. Existing instances are unaffected either way -- their
     * device binding was already fixed at their own construction time.
     */
    public static void clearDeviceSelection() {
        System.clearProperty("opencl.gpu.vendor");
        System.clearProperty("opencl.platform.index");
        System.clearProperty("opencl.device.index");
    }

    private static void check(int status, String call) {
        if (status != OpenClBindings.CL_SUCCESS) {
            throw new IllegalStateException("OpenCL error in " + call + ": code " + status);
        }
    }

    // ================= instance state =================

    /**
     * Which GPU (platform + device, vendor/name) THIS instance is bound
     * to -- e.g. to confirm a prior selectDevice(...) call actually took
     * effect for this particular expression. Fixed at construction time,
     * for this instance's whole lifetime.
     *
     * NOTE: this used to be a static method answering "which GPU is the
     * JVM using" -- that question no longer has a single coherent answer
     * now that different instances can be bound to different devices
     * simultaneously (that's the whole point of this change). Callers
     * relying on the old static getSelectedDeviceDescription() need to
     * switch to calling this on a specific instance instead.
     */
    public String getDeviceDescription() {
        return ctx.selectedDeviceDescription;
    }

    private final GpuContext ctx;
    private final OpenClBindings cl = CL;

    private final Arena arena = Arena.ofShared();
    private final MemorySegment opcodesDevice;
    private final MemorySegment targetSlotsDevice;
    private final MemorySegment literalConstantsDevice;   // double, feeds kernelF64
    private final MemorySegment literalConstantsDeviceF32; // float, feeds kernelF32
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

    public OpenClCompositeExpression(int[] opcodes, int[] targetSlots, double[] literalConstants,
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

            float[] literalConstantsF32 = new float[literalConstants.length];
            for (int i = 0; i < literalConstants.length; i++) {
                literalConstantsF32[i] = (float) literalConstants[i];
            }
            this.literalConstantsDeviceF32 = uploadFloatArray(literalConstantsF32);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to upload GPU program buffers", t);
        }
    }

    /**
     *
     * @param in
     * @param out
     * @throws Throwable
     */
    @Override
    public void applyBulk(MemorySegment in, MemorySegment out) throws Throwable {
        long dataSize = out.byteSize() / ValueLayout.JAVA_DOUBLE.byteSize();
        dispatch(in, out, (int) dataSize);
    }

    /**
     *
     * @param in
     * @param out
     * @throws Throwable
     */
    @Override
    public void applyBulk(double[] in, double[] out) throws Throwable {
        ensureStaging(in.length, out.length);
        MemorySegment.copy(in, 0, stagingIn, ValueLayout.JAVA_DOUBLE, 0, in.length);
        MemorySegment inSlice = stagingIn.asSlice(0, (long) in.length * ValueLayout.JAVA_DOUBLE.byteSize());
        MemorySegment outSlice = stagingOut.asSlice(0, (long) out.length * ValueLayout.JAVA_DOUBLE.byteSize());
        dispatch(inSlice, outSlice, out.length);
        MemorySegment.copy(outSlice, ValueLayout.JAVA_DOUBLE, 0, out, 0, out.length);
    }

    /**
     *
     * @param in
     * @param out
     * @throws Throwable
     */
    @Override
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
        synchronized (ctx.dispatchLock) {
            ensureDeviceBuffers(in.byteSize(), out.byteSize());

            try {
                check((int) cl.clEnqueueWriteBuffer.invoke(ctx.queue, inputDevice,
                        OpenClBindings.CL_TRUE, 0L, in.byteSize(), in, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueWriteBuffer(in)");

                setKernelArgs(ctx.kernelF64, literalConstantsDevice, inputDevice, outputDevice, dataSize);

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment globalWorkSize = tmp.allocate(ValueLayout.JAVA_LONG);
                    globalWorkSize.set(ValueLayout.JAVA_LONG, 0, (long) dataSize);

                    check((int) cl.clEnqueueNDRangeKernel.invoke(ctx.queue, ctx.kernelF64,
                            1, MemorySegment.NULL, globalWorkSize, MemorySegment.NULL,
                            0, MemorySegment.NULL, MemorySegment.NULL),
                            "clEnqueueNDRangeKernel(interpret)");
                }

                check((int) cl.clEnqueueReadBuffer.invoke(ctx.queue, outputDevice,
                        OpenClBindings.CL_TRUE, 0L, out.byteSize(), out, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueReadBuffer(out)");

                check((int) cl.clFinish.invoke(ctx.queue), "clFinish");
            } catch (Throwable t) {
                throw new RuntimeException("GPU dispatch failed", t);
            }
        }
    }

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
        synchronized (ctx.dispatchLock) {
            ensureDeviceBuffersF32(in.byteSize(), out.byteSize());

            try {
                check((int) cl.clEnqueueWriteBuffer.invoke(ctx.queue, inputDeviceF32,
                        OpenClBindings.CL_TRUE, 0L, in.byteSize(), in, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueWriteBuffer(in, f32)");

                setKernelArgs(ctx.kernelF32, literalConstantsDeviceF32,
                        inputDeviceF32, outputDeviceF32, dataSize);

                try (Arena tmp = Arena.ofConfined()) {
                    MemorySegment globalWorkSize = tmp.allocate(ValueLayout.JAVA_LONG);
                    globalWorkSize.set(ValueLayout.JAVA_LONG, 0, (long) dataSize);

                    check((int) cl.clEnqueueNDRangeKernel.invoke(ctx.queue, ctx.kernelF32,
                            1, MemorySegment.NULL, globalWorkSize, MemorySegment.NULL,
                            0, MemorySegment.NULL, MemorySegment.NULL),
                            "clEnqueueNDRangeKernel(interpretF32)");
                }

                check((int) cl.clEnqueueReadBuffer.invoke(ctx.queue, outputDeviceF32,
                        OpenClBindings.CL_TRUE, 0L, out.byteSize(), out, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueReadBuffer(out, f32)");

                check((int) cl.clFinish.invoke(ctx.queue), "clFinish");
            } catch (Throwable t) {
                throw new RuntimeException("GPU dispatch failed (f32)", t);
            }
        }
    }

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

    private void ensureDeviceBuffers(long inBytes, long outBytes) throws Throwable {
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
                inputDevice = (MemorySegment) cl.clCreateBuffer.invoke(ctx.context,
                        OpenClBindings.CL_MEM_READ_ONLY, inBytes, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(in)");
                deviceInCapacityBytes = inBytes;
            }

            if (needsOut) {
                if (!outputDevice.equals(MemorySegment.NULL)) {
                    cl.clReleaseMemObject.invoke(outputDevice);
                }
                outputDevice = (MemorySegment) cl.clCreateBuffer.invoke(ctx.context,
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
                inputDeviceF32 = (MemorySegment) cl.clCreateBuffer.invoke(ctx.context,
                        OpenClBindings.CL_MEM_READ_ONLY, inBytes, MemorySegment.NULL, errBuf);
                check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(in, f32)");
                deviceInCapacityBytesF32 = inBytes;
            }

            if (needsOut) {
                if (!outputDeviceF32.equals(MemorySegment.NULL)) {
                    cl.clReleaseMemObject.invoke(outputDeviceF32);
                }
                outputDeviceF32 = (MemorySegment) cl.clCreateBuffer.invoke(ctx.context,
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

    private MemorySegment uploadIntArray(int[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_INT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);

            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
            MemorySegment device = (MemorySegment) cl.clCreateBuffer.invoke(ctx.context,
                    OpenClBindings.CL_MEM_READ_ONLY, host.byteSize(), MemorySegment.NULL, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(int[])");

            check((int) cl.clEnqueueWriteBuffer.invoke(ctx.queue, device, OpenClBindings.CL_TRUE,
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
            MemorySegment device = (MemorySegment) cl.clCreateBuffer.invoke(ctx.context,
                    OpenClBindings.CL_MEM_READ_ONLY, host.byteSize(), MemorySegment.NULL, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(double[])");

            check((int) cl.clEnqueueWriteBuffer.invoke(ctx.queue, device, OpenClBindings.CL_TRUE,
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
            MemorySegment device = (MemorySegment) cl.clCreateBuffer.invoke(ctx.context,
                    OpenClBindings.CL_MEM_READ_ONLY, host.byteSize(), MemorySegment.NULL, errBuf);
            check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(float[])");

            check((int) cl.clEnqueueWriteBuffer.invoke(ctx.queue, device, OpenClBindings.CL_TRUE,
                    0L, host.byteSize(), host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueWriteBuffer(float[])");
            return device;
        }
    }

    @Override
    public void close() {
        // Only per-instance resources are released here. ctx (context,
        // queue, program, both kernels) is SHARED -- cached in
        // CONTEXT_REGISTRY and potentially reused by other instances bound
        // to the same device -- and deliberately never torn down by an
        // individual instance's close(). Same lifetime assumption as the
        // original single-context design: contexts live for the JVM's
        // duration, now just potentially several of them instead of one.
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
