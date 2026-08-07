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
 * Device selection for the Llama runner's OpenCL backend. Distinct from
 * {@code com.github.gbenroscience.gpu.evaluator.opencl.OpenClBindings}'
 * device-selection logic in {@code OpenClCompositeExpression} -- separate
 * package, separate {@link com.github.gbenroscience.gpu.llm.opencl.OpenCLBindings}
 * copy (see that class's javadoc for why they're kept apart rather than
 * merged), so its own device selector too.
 *
 * <b>CPU SELECTION:</b> unlike {@code com.github.gbenroscience.gpu.llm.cuda.CudaDeviceSelector}
 * (which can only ever target NVIDIA GPUs -- see that class's javadoc for
 * why CPU selection fails there), OpenCL genuinely supports running
 * against a CPU device: {@code CL_DEVICE_TYPE_CPU} is a real, standard
 * OpenCL device type, exposed by e.g. Intel's OpenCL CPU runtime or PoCL.
 * {@link DeviceType#CPU} here actually works, filtering enumeration to
 * CPU devices instead of throwing.
 *
 * Selection precedence (device-type filter narrows the search space FIRST,
 * then the same vendor/index precedence
 * {@code com.github.gbenroscience.gpu.evaluator.opencl}'s selector uses
 * applies within it):
 *   1. -Dopencl.platform.index=P (+ optional -Dopencl.device.index=D) --
 *      an exact pick always wins, regardless of any device-type filter.
 *   2. -Dopencl.device.type=GPU (default) | CPU | ANY -- narrows which
 *      devices are even candidates.
 *   3. -Dopencl.gpu.vendor=&lt;substring&gt; (or {@link #selectDevice(GpuVendor)}),
 *      case-insensitive, word-boundary matched, with known alias
 *      expansion -- applied within the type-filtered candidate set.
 *   4. Default: the first device of the selected type on the first
 *      platform that has one.
 */
public final class OpenCLDeviceSelector {

    private OpenCLDeviceSelector() {
    }

    public enum GpuVendor {
        AMD, INTEL, NVIDIA
    }

    public enum DeviceType {
        GPU, CPU, ANY
    }

    /** One enumerated (platform, device) pair actually resolved for use. */
    public record SelectedDevice(MemorySegment platform, MemorySegment device,
            int platformIndex, int deviceIndex, String platformName,
            String deviceVendor, String deviceName, boolean isCpu) {
        public String describe() {
            return "[platform " + platformIndex + ": " + platformName + "] "
                    + "[device " + deviceIndex + ": " + deviceVendor + " " + deviceName
                    + (isCpu ? " (CPU)" : " (GPU)") + "]";
        }

        public String registryKey() {
            return platformIndex + ":" + deviceIndex;
        }
    }

    private static final OpenCLBindings CL = new OpenCLBindings();

    /** Lists every OpenCL device (GPU AND CPU, regardless of any currently-set type filter -- this is discovery, not selection) this driver can currently see, across every installed platform. */
    public static List<String> listAvailableDevices() {
        try (Arena arena = Arena.ofConfined()) {
            List<SelectedDevice> candidates = enumerateCandidates(arena, OpenCLBindings.CL_DEVICE_TYPE_ALL);
            List<String> descriptions = new ArrayList<>();
            for (SelectedDevice c : candidates) {
                descriptions.add(c.describe());
            }
            return descriptions;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to enumerate OpenCL devices", t);
        }
    }

    /** Resolves whichever device the currently-set selection properties point at. Read fresh every call. */
    public static SelectedDevice resolve() {
        try (Arena arena = Arena.ofConfined()) {
            long typeFilter = resolveTypeFilter();
            List<SelectedDevice> candidates = enumerateCandidates(arena, typeFilter);
            if (candidates.isEmpty()) {
                throw new IllegalStateException(
                        "No OpenCL " + deviceTypeLabel(typeFilter) + " devices found on any platform "
                        + "(ICD loader present but no platform exposed one)");
            }
            return selectCandidate(candidates);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) {
                throw re;
            }
            throw new RuntimeException("Failed to resolve an OpenCL device", t);
        }
    }

    private static long resolveTypeFilter() {
        String typeProp = System.getProperty("opencl.device.type", "GPU").trim().toUpperCase(Locale.ROOT);
        return switch (typeProp) {
            case "CPU" -> OpenCLBindings.CL_DEVICE_TYPE_CPU;
            case "ANY", "ALL" -> OpenCLBindings.CL_DEVICE_TYPE_ALL;
            default -> OpenCLBindings.CL_DEVICE_TYPE_GPU;
        };
    }

    private static String deviceTypeLabel(long typeFilter) {
        if (typeFilter == OpenCLBindings.CL_DEVICE_TYPE_CPU) {
            return "CPU";
        }
        if (typeFilter == OpenCLBindings.CL_DEVICE_TYPE_ALL) {
            return "GPU or CPU";
        }
        return "GPU";
    }

    // ================= enumeration =================

    private static List<SelectedDevice> enumerateCandidates(Arena arena, long typeFilter) throws Throwable {
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
                    platform, typeFilter, 0, MemorySegment.NULL, devCountBuf);
            if (deviceCountStatus != OpenCLBindings.CL_SUCCESS) {
                continue; // this platform has no device of the requested type -- normal, not fatal
            }
            int deviceCount = devCountBuf.get(ValueLayout.JAVA_INT, 0);
            if (deviceCount < 1) {
                continue;
            }

            MemorySegment deviceArr = arena.allocate(ValueLayout.ADDRESS, deviceCount);
            check((int) CL.clGetDeviceIDs.invoke(platform, typeFilter,
                    deviceCount, deviceArr, MemorySegment.NULL), "clGetDeviceIDs(list) on platform " + p);

            for (int d = 0; d < deviceCount; d++) {
                MemorySegment device = deviceArr.getAtIndex(ValueLayout.ADDRESS, d);
                String vendor = getDeviceInfoString(arena, device, OpenCLBindings.CL_DEVICE_VENDOR);
                String name = getDeviceInfoString(arena, device, OpenCLBindings.CL_DEVICE_NAME);
                boolean isCpu = isCpuDevice(arena, device);
                candidates.add(new SelectedDevice(platform, device, p, d, platformName, vendor, name, isCpu));
            }
        }

        return candidates;
    }

    /** Distinguishes actual device type in the -Dopencl.device.type=ANY case, where the enumeration itself mixes GPU and CPU devices together and {@code describe()}/downstream logging needs to say which is which. */
    private static boolean isCpuDevice(Arena arena, MemorySegment device) throws Throwable {
        MemorySegment typeBuf = arena.allocate(ValueLayout.JAVA_LONG);
        int status = (int) CL.clGetDeviceInfo.invoke(device, OpenCLBindings.CL_DEVICE_TYPE, 8L, typeBuf, MemorySegment.NULL);
        if (status != OpenCLBindings.CL_SUCCESS) {
            return false;
        }
        long type = typeBuf.get(ValueLayout.JAVA_LONG, 0);
        return (type & OpenCLBindings.CL_DEVICE_TYPE_CPU) != 0;
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
                    "No device found at opencl.platform.index=" + platformIndex
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
                    "No device matching opencl.gpu.vendor=\"" + vendorProp + "\" found "
                    + "(also tried aliases " + needles + "). Available devices:\n"
                    + describeAll(candidates));
        }

        return candidates.get(0);
    }

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

    /** Selects which GPU the NEXT-resolved device will bind to, by case-insensitive substring match against the device's vendor or name. Does not change the device-TYPE filter -- combine with {@link #selectDevice(DeviceType)} if you also want to restrict to CPU/GPU/ANY. */
    public static void selectDevice(String vendorOrNameSubstring) {
        System.setProperty("opencl.gpu.vendor", vendorOrNameSubstring);
    }

    /** Typed convenience over {@link #selectDevice(String)}. */
    public static void selectDevice(GpuVendor vendor) {
        selectDevice(vendor.name());
    }

    /**
     * Restricts the NEXT-resolved device to the given type -- {@link DeviceType#CPU}
     * actually works here (see class javadoc), unlike
     * {@code com.github.gbenroscience.gpu.llm.cuda.CudaDeviceSelector}'s
     * version of this method. Does not change any vendor/index selection
     * already set -- combine as needed.
     */
    public static void selectDevice(DeviceType type) {
        System.setProperty("opencl.device.type", type.name());
    }

    /** Exact-index selection -- see {@link #listAvailableDevices()} to find the indices. Overrides any type/vendor filter. */
    public static void selectDevice(int platformIndex, int deviceIndex) {
        System.setProperty("opencl.platform.index", String.valueOf(platformIndex));
        System.setProperty("opencl.device.index", String.valueOf(deviceIndex));
    }

    /** Reverts to the default (GPU, first platform that has one, first device on it) for anything resolved after this call. */
    public static void clearDeviceSelection() {
        System.clearProperty("opencl.gpu.vendor");
        System.clearProperty("opencl.device.type");
        System.clearProperty("opencl.platform.index");
        System.clearProperty("opencl.device.index");
    }
}