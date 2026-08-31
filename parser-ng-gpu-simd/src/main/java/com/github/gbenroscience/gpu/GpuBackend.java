package com.github.gbenroscience.gpu;

/**
 * Which GPU compute backend to compile/dispatch an expression against.
 * Passed to GpuExpressionBridge to pick explicitly; GpuExpressionBridge's
 * no-argument overloads use AUTO's preference order instead (see its
 * javadoc for what "preference order" means and why it's a JVM-lifetime
 * decision, not a per-call one).
 */
public enum GpuBackend {
    /** Vendor-neutral -- runs on Intel/AMD/NVIDIA GPUs (and CPUs, via an ICD like POCL). */
    OPENCL,
    /** NVIDIA-only, requires the CUDA Toolkit's NVRTC in addition to the driver. */
    CUDA, METAL 
}