package com.github.gbenroscience.gpu.evaluator.cuda;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Device selection for CUDA. No vendor concept here -- CUDA only ever
 * targets NVIDIA hardware, so "which GPU" reduces entirely to "which
 * index", unlike {@link com.github.gbenroscience.gpu.opencl.OpenClDeviceSelector}'s
 * vendor/alias matching across possibly several installed OpenCL ICDs.
 * Kept as its own class (rather than folded into GpuContext) purely for
 * API symmetry with the OpenCL side and the math evaluator's pattern --
 * {@link #listAvailableDevices()} to see what's there,
 * {@link #selectDevice(int)} to pick one, {@link #clearDeviceSelection()}
 * to go back to the default.
 *
 * Uses the SAME "cuda.device.index" property the CUDA LLM {@code GpuContext}
 * already read directly before this class existed -- this is a drop-in
 * factoring-out, not a behavior change; anything already setting
 * -Dcuda.device.index continues to work unmodified.
 */
public final class CudaDeviceSelector {

    private CudaDeviceSelector() {
    }

    public record SelectedDevice(int deviceIndex, int cuDevice, String name, int major, int minor) {
        public String describe() {
            return "[device " + deviceIndex + ": " + name + " (compute capability " + major + "." + minor + ")]";
        }
    }

    private static final CudaBindings CU = new CudaBindings();
    private static volatile boolean initialized = false;

    private static synchronized void ensureInit() throws Throwable {
        if (!initialized) {
            check((int) CU.cuInit.invoke(0), "cuInit");
            initialized = true;
        }
    }

    /** Lists every CUDA device this driver can see, with name and compute capability -- call this to find out what index to pass {@link #selectDevice(int)}. */
    public static List<String> listAvailableDevices() {
        try (Arena arena = Arena.ofConfined()) {
            List<SelectedDevice> devices = enumerateDevices(arena);
            List<String> descriptions = new ArrayList<>();
            for (SelectedDevice d : devices) {
                descriptions.add(d.describe());
            }
            return descriptions;
        } catch (Throwable t) {
            throw new RuntimeException("Failed to enumerate CUDA devices", t);
        }
    }

    /** Resolves whichever device index is currently selected (default 0), with its name/compute-capability for logging. */
    public static SelectedDevice resolve() {
        try (Arena arena = Arena.ofConfined()) {
            List<SelectedDevice> devices = enumerateDevices(arena);
            if (devices.isEmpty()) {
                throw new IllegalStateException("No CUDA devices found -- an NVIDIA GPU driver must be installed.");
            }
            int index = Integer.getInteger("cuda.device.index", 0);
            for (SelectedDevice d : devices) {
                if (d.deviceIndex() == index) {
                    return d;
                }
            }
            StringBuilder sb = new StringBuilder();
            for (SelectedDevice d : devices) {
                sb.append("  ").append(d.describe()).append('\n');
            }
            throw new IllegalStateException(
                    "No CUDA device found at cuda.device.index=" + index + ". Available devices:\n" + sb);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) {
                throw re;
            }
            throw new RuntimeException("Failed to resolve a CUDA device", t);
        }
    }

    private static List<SelectedDevice> enumerateDevices(Arena arena) throws Throwable {
        ensureInit();

        MemorySegment countBuf = arena.allocate(ValueLayout.JAVA_INT);
        check((int) CU.cuDeviceGetCount.invoke(countBuf), "cuDeviceGetCount");
        int count = countBuf.get(ValueLayout.JAVA_INT, 0);

        List<SelectedDevice> devices = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            MemorySegment deviceBuf = arena.allocate(ValueLayout.JAVA_INT);
            check((int) CU.cuDeviceGet.invoke(deviceBuf, i), "cuDeviceGet");
            int device = deviceBuf.get(ValueLayout.JAVA_INT, 0);

            MemorySegment majorBuf = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment minorBuf = arena.allocate(ValueLayout.JAVA_INT);
            check((int) CU.cuDeviceGetAttribute.invoke(majorBuf,
                    CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute(major)");
            check((int) CU.cuDeviceGetAttribute.invoke(minorBuf,
                    CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute(minor)");

            MemorySegment nameBuf = arena.allocate(256);
            check((int) CU.cuDeviceGetName.invoke(nameBuf, 256, device), "cuDeviceGetName");
            String name = nameBuf.getString(0, StandardCharsets.UTF_8);

            devices.add(new SelectedDevice(i, device, name,
                    majorBuf.get(ValueLayout.JAVA_INT, 0), minorBuf.get(ValueLayout.JAVA_INT, 0)));
        }
        return devices;
    }

    private static void check(int status, String call) {
        if (status != CudaBindings.CUDA_SUCCESS) {
            throw new IllegalStateException("CUDA error in " + call + ": code " + status);
        }
    }

    /** Selects which CUDA device index the NEXT-resolved GpuContext will bind to. Same "set the default for whatever gets constructed next" contract as OpenClDeviceSelector.selectDevice -- see its javadoc. */
    public static void selectDevice(int deviceIndex) {
        System.setProperty("cuda.device.index", String.valueOf(deviceIndex));
    }

    /** Reverts to the default (device index 0) for anything resolved after this call. */
    public static void clearDeviceSelection() {
        System.clearProperty("cuda.device.index");
    }
}