package com.github.gbenroscience.gpu.llm.cuda;

import com.github.gbenroscience.gpu.evaluator.cuda.CudaBindings;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Device selection for the Llama runner's CUDA backend. Distinct from
 * {@code com.github.gbenroscience.gpu.evaluator.cuda.CudaDeviceSelector}
 * (the math evaluator's, GPU-only for now) -- this copy additionally
 * accepts a {@link DeviceType} selection, since the Llama runner's device
 * selectors are being extended to CPU-awareness first (see class javadoc
 * on why CPU fails here specifically, and
 * {@code com.github.gbenroscience.gpu.llm.opencl.OpenCLDeviceSelector} for
 * where CPU selection actually WORKS).
 *
 * <b>WHY CPU SELECTION FAILS HERE, HONESTLY, RATHER THAN BEING SILENTLY
 * UNSUPPORTED:</b> the CUDA driver API has no CPU device concept at all --
 * {@code cuDeviceGetCount}/{@code cuDeviceGet} only ever enumerate NVIDIA
 * GPUs; there is no CUDA equivalent of OpenCL's {@code CL_DEVICE_TYPE_CPU}.
 * {@link DeviceType} is accepted here anyway (rather than only existing on
 * the OpenCL selector) purely for API symmetry with
 * {@code OpenCLDeviceSelector}, in case calling code is written generically
 * against "a selector" without caring which backend it's driving. Selecting
 * {@link DeviceType#CPU} records the request same as any other property
 * (cheap, no validation), but {@link #resolve()} throws a clear, actionable
 * {@link UnsupportedOperationException} rather than either quietly falling
 * back to a GPU (wrong -- ignores what the caller asked for) or quietly
 * doing nothing (equally wrong -- looks like it worked). If you actually
 * need CPU execution, select {@code GpuBackend.OPENCL} with an OpenCL CPU
 * runtime (Intel's, or PoCL) via {@code OpenCLDeviceSelector}, or -- once
 * it exists -- a CPU-native (non-GPU-API) Llama backend entirely.
 */
public final class CudaDeviceSelector {

    private CudaDeviceSelector() {
    }

    public enum DeviceType {
        GPU, CPU
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

    /** Lists every CUDA device this driver can see, with name and compute capability -- call this to find out what index to pass {@link #selectDevice(int)}. Always GPUs -- see class javadoc for why there is no CPU entry to list. */
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

    /** Resolves whichever device index is currently selected (default 0). Throws {@link UnsupportedOperationException} if {@link DeviceType#CPU} was selected -- see class javadoc. */
    public static SelectedDevice resolve() {
        String typeProp = System.getProperty("cuda.device.type", "GPU");
        if ("CPU".equalsIgnoreCase(typeProp.trim())) {
            throw new UnsupportedOperationException(
                    "CudaDeviceSelector was asked to select a CPU device, but the CUDA driver API has no "
                    + "CPU device concept -- cuDeviceGetCount only ever enumerates NVIDIA GPUs. "
                    + "Select GpuBackend.OPENCL with an OpenCL CPU runtime (via "
                    + "com.github.gbenroscience.gpu.llm.opencl.OpenCLDeviceSelector.selectDevice(DeviceType.CPU)) "
                    + "if you need CPU execution.");
        }

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

    /** Selects which CUDA device index the NEXT-resolved GpuContext will bind to. */
    public static void selectDevice(int deviceIndex) {
        System.setProperty("cuda.device.type", "GPU");
        System.setProperty("cuda.device.index", String.valueOf(deviceIndex));
    }

    /**
     * Records a device-TYPE preference. {@link DeviceType#GPU} behaves
     * exactly as before (the default). {@link DeviceType#CPU} is accepted
     * -- cheaply, no validation here, same "properties are free, resolve()
     * does the real work" contract every selector in this codebase
     * follows -- but {@link #resolve()} will throw when actually invoked.
     * See class javadoc for why.
     */
    public static void selectDevice(DeviceType type) {
        System.setProperty("cuda.device.type", type.name());
    }

    /** Reverts to the default (GPU, device index 0) for anything resolved after this call. */
    public static void clearDeviceSelection() {
        System.clearProperty("cuda.device.type");
        System.clearProperty("cuda.device.index");
    }
}