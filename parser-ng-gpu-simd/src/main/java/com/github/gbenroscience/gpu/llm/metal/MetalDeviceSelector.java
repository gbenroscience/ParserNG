package com.github.gbenroscience.gpu.llm.metal;

import java.util.ArrayList;
import java.util.List;

/**
 * Device selection for the Llama runner's Metal backend -- counterpart of
 * {@code com.github.gbenroscience.gpu.llm.cuda.CudaDeviceSelector} /
 * {@code com.github.gbenroscience.gpu.llm.opencl.OpenCLDeviceSelector}.
 * Same "properties are free, resolve() does the real work" contract, same
 * {@link DeviceType} symmetry with the other two selectors' APIs.
 *
 * <b>WHY CPU SELECTION FAILS HERE TOO, HONESTLY:</b> exactly the same
 * reasoning as {@code CudaDeviceSelector}'s javadoc -- {@code MTLCopyAllDevices}
 * enumerates GPUs (integrated and/or discrete), never a CPU compute
 * device; Metal has no CPU-device concept the way OpenCL's
 * {@code CL_DEVICE_TYPE_CPU} does. {@link DeviceType#CPU} is accepted for
 * API symmetry with the OpenCL selector, cheaply recorded, but
 * {@link #resolve()} throws a clear {@link UnsupportedOperationException}
 * rather than silently ignoring the request or silently falling back to a
 * GPU.
 *
 * <b>ONE ADDITIONAL CHECK METAL-SPECIFIC TO THIS PORT:</b>
 * {@link #resolve()} also rejects a device that reports
 * {@code hasUnifiedMemory == false} (a discrete/eGPU Mac) with a similarly
 * explicit exception, rather than silently using
 * {@code MTLResourceStorageModeShared} against a device where that mode
 * is legal but slow (falls back to a PCIe-style transfer under the hood)
 * -- see {@link MetalBindings}'s class javadoc for why this port's whole
 * memory model assumes unified memory. If you need discrete-GPU support,
 * {@link LlamaLayer}'s upload/download helpers need a
 * {@code StorageModeManaged}-aware rewrite first; not attempted here.
 */
public final class MetalDeviceSelector {

    private MetalDeviceSelector() {
    }

    public enum DeviceType {
        GPU, CPU
    }

    public record SelectedDevice(int deviceIndex, long deviceId, String name, boolean unifiedMemory) {
        public String describe() {
            return "[device " + deviceIndex + ": " + name + (unifiedMemory ? " (unified memory)" : " (DISCRETE -- unsupported by this port)") + "]";
        }
    }

    private static final MetalBindings MTL = new MetalBindings();

    /** Lists every Metal device this process can see, with name and unified-memory status -- call this to find out what index to pass {@link #selectDevice(int)}. */
    public static List<String> listAvailableDevices() {
        List<SelectedDevice> devices = enumerateDevices();
        List<String> descriptions = new ArrayList<>();
        for (SelectedDevice d : devices) {
            descriptions.add(d.describe());
        }
        return descriptions;
    }

    /** Resolves whichever device index is currently selected (default 0, i.e. the system default device). Throws {@link UnsupportedOperationException} if {@link DeviceType#CPU} was selected, or {@link IllegalStateException} if the resolved device lacks unified memory -- see class javadoc. */
    public static SelectedDevice resolve() {
        String typeProp = System.getProperty("metal.device.type", "GPU");
        if ("CPU".equalsIgnoreCase(typeProp.trim())) {
            throw new UnsupportedOperationException(
                    "MetalDeviceSelector was asked to select a CPU device, but Metal has no CPU device "
                    + "concept -- MTLCopyAllDevices only ever enumerates GPUs. Select "
                    + "GpuBackend.OPENCL with an OpenCL CPU runtime (via "
                    + "com.github.gbenroscience.gpu.llm.opencl.OpenCLDeviceSelector.selectDevice(DeviceType.CPU)) "
                    + "if you need CPU execution.");
        }

        List<SelectedDevice> devices = enumerateDevices();
        if (devices.isEmpty()) {
            throw new IllegalStateException("No Metal devices found -- this process must be running on a Mac with a Metal-capable GPU.");
        }

        int index = Integer.getInteger("metal.device.index", -1);
        SelectedDevice chosen;
        if (index < 0) {
            // Default: MTLCreateSystemDefaultDevice's pick (index 0 in our enumeration order by convention).
            chosen = devices.get(0);
        } else {
            chosen = null;
            for (SelectedDevice d : devices) {
                if (d.deviceIndex() == index) {
                    chosen = d;
                    break;
                }
            }
            if (chosen == null) {
                StringBuilder sb = new StringBuilder();
                for (SelectedDevice d : devices) {
                    sb.append("  ").append(d.describe()).append('\n');
                }
                throw new IllegalStateException(
                        "No Metal device found at metal.device.index=" + index + ". Available devices:\n" + sb);
            }
        }

        if (!chosen.unifiedMemory()) {
            throw new IllegalStateException(
                    "Selected Metal device \"" + chosen.name() + "\" reports hasUnifiedMemory=false (a discrete "
                    + "or external GPU). This port's memory model (MetalBuffer / MetalBindings, see their "
                    + "class javadocs) assumes MTLResourceStorageModeShared unified memory throughout and has "
                    + "not been extended to StorageModeManaged staging for discrete GPUs -- select an "
                    + "integrated-GPU device index instead, or extend LlamaLayer's upload/download helpers "
                    + "before using this device.");
        }
        return chosen;
    }

    private static List<SelectedDevice> enumerateDevices() {
        long[] ids = MTL.copyAllDevices();
        List<SelectedDevice> devices = new ArrayList<>();
        if (ids.length == 0) {
            long def = MTL.createSystemDefaultDevice();
            if (def != 0L) {
                devices.add(new SelectedDevice(0, def, MTL.deviceName(def), MTL.hasUnifiedMemory(def)));
            }
            return devices;
        }
        for (int i = 0; i < ids.length; i++) {
            long id = ids[i];
            devices.add(new SelectedDevice(i, id, MTL.deviceName(id), MTL.hasUnifiedMemory(id)));
        }
        return devices;
    }

    /** Selects which Metal device index the NEXT-resolved GpuContext will bind to. */
    public static void selectDevice(int deviceIndex) {
        System.setProperty("metal.device.type", "GPU");
        System.setProperty("metal.device.index", String.valueOf(deviceIndex));
    }

    /**
     * Records a device-TYPE preference. {@link DeviceType#GPU} behaves
     * exactly as before (the default). {@link DeviceType#CPU} is accepted
     * cheaply here but {@link #resolve()} will throw when actually
     * invoked -- see class javadoc.
     */
    public static void selectDevice(DeviceType type) {
        System.setProperty("metal.device.type", type.name());
    }

    /** Reverts to the default (system default device) for anything resolved after this call. */
    public static void clearDeviceSelection() {
        System.clearProperty("metal.device.type");
        System.clearProperty("metal.device.index");
    }
}