package com.github.gbenroscience.gpu.llm.opencl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Vendor/device selection for OpenCL, factored out so every OpenCL
 * consumer in this codebase -- the math-expression evaluator's
 * {@code OpenClCompositeExpression} and the Llama runner's
 * {@code com.github.gbenroscience.gpu.opencl.llm.GpuContext} alike -- picks
 * a device the same way, with the same system properties, the same vendor
 * aliasing, and the same enumeration bug fixes (see {@link #matchesAlias}).
 *
 * DELIBERATELY REUSES {@code OpenClCompositeExpression}'s property names --
 * {@code opencl.gpu.vendor}, {@code opencl.platform.index},
 * {@code opencl.device.index} -- rather than inventing Llama-specific ones.
 * One {@code -Dopencl.gpu.vendor=AMD}, or one call to
 * {@link #selectDevice(GpuVendor)}, now governs BOTH the math evaluator and
 * the Llama runner's next-constructed instances. That's the point: "which
 * GPU" should be one JVM-wide answer, not a per-feature setting a caller
 * has to remember to set twice.
 *
 * IMPORTANT NOTE ON BINDINGS: this class calls
 * {@code com.github.gbenroscience.gpu.opencl.OpenCLBindings} (this
 * repository's Llama-port raw FFM bindings). Your math evaluator's
 * {@code OpenClCompositeExpression} already has its own near-identical
 * {@code OpenClBindings} class (note the capitalization difference:
 * "OpenCLBindings" vs "OpenClBindings") in this SAME package, built
 * independently before that file was shared with me. They almost
 * certainly duplicate each other's entire API surface. I'm keeping them
 * separate for now rather than guessing at OpenClBindings' exact method
 * signatures and silently breaking your working math evaluator -- if you
 * want them consolidated onto one binding class, share OpenClBindings.java
 * and I'll merge them and delete the duplicate.
 *
 * Selection precedence (identical to OpenClCompositeExpression's, see its
 * {@code selectCandidate} javadoc for the full rationale):
 *   1. -Dopencl.platform.index=P (+ optional -Dopencl.device.index=D)
 *   2. -Dopencl.gpu.vendor=&lt;substring&gt; (or {@link #selectDevice(GpuVendor)}),
 *      case-insensitive, word-boundary matched, with known alias expansion
 *   3. Default: the first platform that exposes a GPU, first device on it
 *
 * NOTE ON SCOPE vs OpenClCompositeExpression: that class also enumerates
 * only CL_DEVICE_TYPE_GPU with no CPU/ALL fallback -- this class matches
 * that exactly (no "opencl.device.type=ALL" escape hatch), for
 * consistency. If you need to test against a CPU OpenCL runtime (e.g.
 * PoCL) with no GPU installed, pass the exact platform/device index via
 * -Dopencl.platform.index/-Dopencl.device.index once you've located it
 * some other way -- this class's own enumeration will not surface it.
 */
public final class OpenCLDeviceSelector {

    private OpenCLDeviceSelector() {
    }

    /** See OpenClCompositeExpression.GpuVendor's javadoc for why this exists instead of requiring the caller to know the raw driver-reported vendor string. */
    public enum GpuVendor {
        AMD, INTEL, NVIDIA
    }

    /** One enumerated (platform, device) pair actually resolved for use. */
    public record SelectedDevice(MemorySegment platform, MemorySegment device,
            int platformIndex, int deviceIndex, String platformName,
            String deviceVendor, String deviceName) {
        public String describe() {
            return "[platform " + platformIndex + ": " + platformName + "] "
                    + "[device " + deviceIndex + ": " + deviceVendor + " " + deviceName + "]";
        }

        /** Stable key for a per-device resource cache, for callers that want one (this class itself caches nothing). */
        public String registryKey() {
            return platformIndex + ":" + deviceIndex;
        }
    }

    private static final OpenCLBindings CL = new OpenCLBindings();

    /**
     * Lists every GPU OpenCL can currently see, across every installed
     * platform, as human-readable descriptions -- call this to find out
     * what string to pass {@link #selectDevice(String)}, or which
     * {@link GpuVendor}/index pair matches your hardware, rather than
     * guessing. Independent of any device actually being selected or a
     * context being built -- safe to call any number of times.
     */
    public static List<String> listAvailableDevices() {
        try (Arena arena = Arena.ofConfined()) {
            List<SelectedDevice> candidates = enumerateGpuCandidates(arena);
            List<String> descriptions = new ArrayList<>();
            for (SelectedDevice c : candidates) {
                descriptions.add(c.describe());
            }
            return descriptions;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to enumerate OpenCL GPU devices", t);
        }
    }

    /**
     * Resolves whichever device the currently-set selection properties
     * point at. Read fresh every call -- a prior {@link #selectDevice}
     * only affects resolutions that happen AFTER it's called, never one
     * already in progress or already resolved.
     */
    public static SelectedDevice resolve() {
        try (Arena arena = Arena.ofConfined()) {
            List<SelectedDevice> candidates = enumerateGpuCandidates(arena);
            if (candidates.isEmpty()) {
                throw new IllegalStateException(
                        "No OpenCL GPU devices found on any platform (ICD loader present but "
                        + "no platform exposed a GPU device)");
            }
            return selectCandidate(candidates);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) {
                throw re;
            }
            throw new RuntimeException("Failed to resolve an OpenCL GPU device", t);
        }
    }

    // ================= enumeration =================

    private static List<SelectedDevice> enumerateGpuCandidates(Arena arena) throws Throwable {
        MemorySegment countBuf = arena.allocate(ValueLayout.JAVA_INT);

        check((int) CL.clGetPlatformIDs.invoke(0, MemorySegment.NULL, countBuf), "clGetPlatformIDs(count)");
        int platformCount = countBuf.get(ValueLayout.JAVA_INT, 0);
        if (platformCount < 1) {
            throw new IllegalStateException(
                    "No OpenCL platforms found (ICD loader present but no vendor registered)");
        }

        MemorySegment platformArr = arena.allocate(ValueLayout.ADDRESS, platformCount);
        check((int) CL.clGetPlatformIDs.invoke(platformCount, platformArr, MemorySegment.NULL),
                "clGetPlatformIDs(list)");

        List<SelectedDevice> candidates = new ArrayList<>();

        for (int p = 0; p < platformCount; p++) {
            MemorySegment platform = platformArr.getAtIndex(ValueLayout.ADDRESS, p);
            String platformName = getPlatformInfoString(arena, platform, OpenCLBindings.CL_PLATFORM_NAME);

            MemorySegment devCountBuf = arena.allocate(ValueLayout.JAVA_INT);
            int deviceCountStatus = (int) CL.clGetDeviceIDs.invoke(
                    platform, OpenCLBindings.CL_DEVICE_TYPE_GPU, 0, MemorySegment.NULL, devCountBuf);
            if (deviceCountStatus != OpenCLBindings.CL_SUCCESS) {
                continue; // this platform has no GPU device (or errored enumerating) -- normal, not fatal
            }
            int deviceCount = devCountBuf.get(ValueLayout.JAVA_INT, 0);
            if (deviceCount < 1) {
                continue;
            }

            MemorySegment deviceArr = arena.allocate(ValueLayout.ADDRESS, deviceCount);
            check((int) CL.clGetDeviceIDs.invoke(platform, OpenCLBindings.CL_DEVICE_TYPE_GPU,
                    deviceCount, deviceArr, MemorySegment.NULL), "clGetDeviceIDs(list) on platform " + p);

            for (int d = 0; d < deviceCount; d++) {
                MemorySegment device = deviceArr.getAtIndex(ValueLayout.ADDRESS, d);
                String vendor = getDeviceInfoString(arena, device, OpenCLBindings.CL_DEVICE_VENDOR);
                String name = getDeviceInfoString(arena, device, OpenCLBindings.CL_DEVICE_NAME);
                candidates.add(new SelectedDevice(platform, device, p, d, platformName, vendor, name));
            }
        }

        return candidates;
    }

    // ================= selection precedence =================

    private static SelectedDevice selectCandidate(List<SelectedDevice> candidates) {
        String platformIndexProp = System.getProperty("opencl.platform.index");
        if (platformIndexProp != null) {
            int platformIndex = Integer.parseInt(platformIndexProp.trim());
            int deviceIndex = Integer.getInteger("opencl.device.index", 0);
            for (SelectedDevice c : candidates) {
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
            String needle = vendorProp.trim().toLowerCase(Locale.ROOT);
            List<String> needles = expandVendorAliases(needle);
            for (SelectedDevice c : candidates) {
                String vendor = c.deviceVendor().toLowerCase(Locale.ROOT);
                String name = c.deviceName().toLowerCase(Locale.ROOT);
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

    /** See OpenClCompositeExpression.expandVendorAliases' javadoc -- same table, same rationale (e.g. AMD's real vendor string doesn't contain the substring "amd" at all). */
    private static List<String> expandVendorAliases(String needle) {
        List<String> aliases = new ArrayList<>();
        aliases.add(needle);
        switch (needle) {
            case "amd" -> aliases.addAll(List.of("advanced micro devices", "radeon", "ati"));
            case "intel" -> aliases.addAll(List.of("iris", "arc", "uhd graphics"));
            case "nvidia" -> aliases.addAll(List.of("geforce", "quadro", "tesla", "rtx", "gtx"));
            default -> { /* use the literal needle as typed, no known aliases */ }
        }
        return aliases;
    }

    /** See OpenClCompositeExpression.matchesAlias's javadoc -- word-boundary match, not a bare substring check, to avoid e.g. "ati" matching inside "Corporation". */
    private static boolean matchesAlias(String haystackLower, String aliasLower) {
        return Pattern.compile("\\b" + Pattern.quote(aliasLower) + "\\b").matcher(haystackLower).find();
    }

    private static String describeAll(List<SelectedDevice> candidates) {
        StringBuilder sb = new StringBuilder();
        for (SelectedDevice c : candidates) {
            sb.append("  ").append(c.describe()).append('\n');
        }
        return sb.toString();
    }

    private static String getPlatformInfoString(Arena arena, MemorySegment platform, int paramName) throws Throwable {
        MemorySegment sizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        check((int) CL.clGetPlatformInfo.invoke(platform, paramName, 0L, MemorySegment.NULL, sizeBuf),
                "clGetPlatformInfo(size)");
        long size = sizeBuf.get(ValueLayout.JAVA_LONG, 0);
        MemorySegment buf = arena.allocate(Math.max(size, 1));
        check((int) CL.clGetPlatformInfo.invoke(platform, paramName, size, buf, MemorySegment.NULL),
                "clGetPlatformInfo(value)");
        return buf.getString(0, StandardCharsets.UTF_8);
    }

    private static String getDeviceInfoString(Arena arena, MemorySegment device, int paramName) throws Throwable {
        MemorySegment sizeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        check((int) CL.clGetDeviceInfo.invoke(device, paramName, 0L, MemorySegment.NULL, sizeBuf),
                "clGetDeviceInfo(size)");
        long size = sizeBuf.get(ValueLayout.JAVA_LONG, 0);
        MemorySegment buf = arena.allocate(Math.max(size, 1));
        check((int) CL.clGetDeviceInfo.invoke(device, paramName, size, buf, MemorySegment.NULL),
                "clGetDeviceInfo(value)");
        return buf.getString(0, StandardCharsets.UTF_8);
    }

    private static void check(int status, String call) {
        if (status != OpenCLBindings.CL_SUCCESS) {
            throw new IllegalStateException("OpenCL error in " + call + ": " + OpenCLBindings.errorString(status));
        }
    }

    // ================= public selection API =================

    /**
     * Selects which GPU the NEXT-resolved device (in this codebase: the
     * next constructed Llama {@code GpuContext}, AND -- since this is the
     * same {@code opencl.gpu.vendor} property -- the next constructed
     * {@code OpenClCompositeExpression}) will bind to, by case-insensitive
     * substring match against the device's vendor or name. See class
     * javadoc for precedence and the shared-property rationale.
     */
    public static void selectDevice(String vendorOrNameSubstring) {
        System.setProperty("opencl.gpu.vendor", vendorOrNameSubstring);
    }

    /** Typed convenience over {@link #selectDevice(String)} -- see {@link GpuVendor}'s javadoc. */
    public static void selectDevice(GpuVendor vendor) {
        selectDevice(vendor.name());
    }

    /** Exact-index selection -- see {@link #listAvailableDevices()} to find the indices. */
    public static void selectDevice(int platformIndex, int deviceIndex) {
        System.setProperty("opencl.platform.index", String.valueOf(platformIndex));
        System.setProperty("opencl.device.index", String.valueOf(deviceIndex));
    }

    /** Reverts to the default (first platform exposing a GPU, first device on it) for anything resolved after this call. */
    public static void clearDeviceSelection() {
        System.clearProperty("opencl.gpu.vendor");
        System.clearProperty("opencl.platform.index");
        System.clearProperty("opencl.device.index");
    }
}