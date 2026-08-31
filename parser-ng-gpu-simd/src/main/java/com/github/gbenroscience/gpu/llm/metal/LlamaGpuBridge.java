package com.github.gbenroscience.gpu.llm;

import com.github.gbenroscience.gpu.GpuBackend;
import com.github.gbenroscience.gpu.llm.cuda.CudaLlamaEngine;
import com.github.gbenroscience.gpu.llm.metal.MetalLlamaEngine;
import com.github.gbenroscience.gpu.llm.opencl.OpenCLLlamaEngine;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Single entry point for building a GPU-backed Llama engine without the
 * caller needing to know or choose between
 * {@code com.github.gbenroscience.gpu.llm.opencl.OpenCLLlamaEngine},
 * {@code com.github.gbenroscience.gpu.llm.cuda.CudaLlamaEngine}, and
 * {@code com.github.gbenroscience.gpu.llm.metal.MetalLlamaEngine}. Direct
 * counterpart of {@link com.github.gbenroscience.gpu.GpuExpressionBridge} --
 * same auto-detection model, same caching/probe behavior, same
 * {@code gpu.backend.preference} property (deliberately the SAME property,
 * not a Llama-specific one -- see below). Use this unless a specific
 * backend is a hard requirement.
 *
 * <pre>
  try (LlamaGpuEngine llm = LlamaGpuBridge.load(new File("model.gguf"))) {       // auto-picks a backend
      String out = llm.generate("Once upon a time,", new LlamaGenerationConfig());
  }

  try (LlamaGpuEngine llm = LlamaGpuBridge.load(new File("model.gguf"), GpuBackend.OPENCL)) {  // explicit
      String out = llm.generate("Once upon a time,", new LlamaGenerationConfig());
  }
 * </pre>
 *
 * <b>GPU VENDOR/DEVICE SELECTION</b> is a layer BELOW backend selection and
 * is not this class's job -- exactly like GpuExpressionBridge, this only
 * decides OpenCL-vs-CUDA-vs-Metal. Once a backend is chosen (or resolved by
 * auto-detection), WHICH device on that backend is used is controlled the
 * same way it is everywhere else in this codebase:
 *   - OpenCL: {@code com.github.gbenroscience.gpu.opencl.OpenClDeviceSelector.selectDevice(...)}
 *     (vendor, exact index, or raw substring) -- the SAME selector and SAME
 *     system properties ({@code opencl.gpu.vendor}, {@code opencl.platform.index},
 *     {@code opencl.device.index}) the math evaluator's OpenClCompositeExpression uses.
 *   - CUDA: {@code com.github.gbenroscience.gpu.cuda.CudaDeviceSelector.selectDevice(int)}
 *     ({@code cuda.device.index} property).
 *   - Metal: {@code com.github.gbenroscience.gpu.llm.metal.MetalDeviceSelector.selectDevice(int)}
 *     ({@code metal.device.index} property) -- see that selector's javadoc for why the
 *     selected device must additionally report unified memory.
 * Call the relevant selector BEFORE {@link #load}, same "sets the default
 * for whatever gets constructed next" contract those classes document.
 *
 * <b>ON PROBING:</b> unlike GpuExpressionBridge's probe (a trivial
 * one-instruction expression -- cheap because a compiled expression is
 * cheap), probing a Llama backend here does NOT load any model weights --
 * it constructs a bare backend {@code GpuContext} (device discovery,
 * context/queue creation, kernel program build) and closes it immediately.
 * That's the right-sized probe: it genuinely exercises the same bootstrap
 * {@link #load} would hit, without the cost of reading a multi-gigabyte
 * GGUF file just to answer "is this backend usable".
 *
 * <b>On auto-detection and class-init:</b> see GpuExpressionBridge's
 * javadoc for the full NoClassDefFoundError-on-static-init-failure
 * rationale -- identical mechanism and identical caching behavior here.
 */
public final class LlamaGpuBridge {

    private static final Map<GpuBackend, Boolean> AVAILABILITY = new ConcurrentHashMap<>();
    private static final Map<GpuBackend, Throwable> LAST_FAILURE = new ConcurrentHashMap<>();
    private static volatile GpuBackend resolvedAutoBackend;
    private static final Object RESOLVE_LOCK = new Object();

    private LlamaGpuBridge() {
    }

    // ================= Explicit backend =================
    public static LlamaGpuEngine load(File ggufPath, GpuBackend backend) throws Throwable {
        return switch (backend) {
            case OPENCL -> new OpenCLLlamaEngine(ggufPath);
            case CUDA -> new CudaLlamaEngine(ggufPath);
            case METAL -> new MetalLlamaEngine(ggufPath);
        };
    }

    // ================= Auto-detected backend =================
    public static LlamaGpuEngine load(File ggufPath) throws Throwable {
        return load(ggufPath, resolveAutoBackend());
    }

    /**
     * Whether a given backend's bootstrap has succeeded (or would succeed,
     * probing it now if it hasn't been tried yet this JVM run). Safe to
     * call speculatively -- results are cached, so repeat calls are free.
     */
    public static boolean isAvailable(GpuBackend backend) {
        return AVAILABILITY.computeIfAbsent(backend, LlamaGpuBridge::probeQuietly);
    }

    // ================= Backend resolution =================
    private static GpuBackend resolveAutoBackend() {
        GpuBackend cached = resolvedAutoBackend;
        if (cached != null) {
            return cached;
        }
        synchronized (RESOLVE_LOCK) {
            if (resolvedAutoBackend != null) {
                return resolvedAutoBackend;
            }
            List<GpuBackend> order = preferenceOrder();
            for (GpuBackend candidate : order) {
                if (isAvailable(candidate)) {
                    resolvedAutoBackend = candidate;
                    return candidate;
                }
            }
            IllegalStateException summary = new IllegalStateException(
                    "No usable GPU backend found for the Llama runner (tried " + order + " in that order). "
                            + "Check that either an OpenCL ICD or an NVIDIA/CUDA driver+NVRTC is actually "
                            + "installed, or select a backend explicitly with "
                            + "LlamaGpuBridge.load(ggufPath, GpuBackend.OPENCL/CUDA) to see the real "
                            + "bootstrap error instead of this summary. The actual per-backend failure is "
                            + "attached below as a suppressed exception for each backend that was tried.");
            for (GpuBackend candidate : order) {
                Throwable cause = LAST_FAILURE.get(candidate);
                if (cause != null) {
                    summary.addSuppressed(new RuntimeException("[" + candidate + "] real failure", cause));
                }
            }
            throw summary;
        }
    }

    /**
     * Same {@code -Dgpu.backend.preference=cuda,opencl} property
     * GpuExpressionBridge reads -- deliberately shared, not
     * Llama-specific, so one flag sets the whole application's backend
     * preference consistently. If you actually want the math evaluator on
     * one backend and the Llama runner on another, call {@link #load(File, GpuBackend)}
     * explicitly for this one rather than relying on auto-detection.
     */
    private static List<GpuBackend> preferenceOrder() {
        String pref = System.getProperty("gpu.backend.preference", "cuda,metal,opencl");
        List<GpuBackend> order = new ArrayList<>();
        for (String token : pref.split(",")) {
            switch (token.trim().toLowerCase(Locale.ROOT)) {
                case "cuda" -> order.add(GpuBackend.CUDA);
                case "opencl" -> order.add(GpuBackend.OPENCL);
                case "metal" -> order.add(GpuBackend.METAL);
                default -> { /* ignore unrecognized token */ }
            }
        }
        if (order.isEmpty()) {
            order.add(GpuBackend.CUDA);
            order.add(GpuBackend.METAL);
            order.add(GpuBackend.OPENCL);
        }
        return order;
    }

    private static boolean probeQuietly(GpuBackend backend) {
        try {
            probe(backend);
            LAST_FAILURE.remove(backend);
            return true;
        } catch (Throwable t) {
            LAST_FAILURE.put(backend, t);
            return false;
        }
    }

    /**
     * Forces the backend's real bootstrap (device discovery, context/queue
     * creation, kernel program build) to run now, via a bare GpuContext
     * that's closed immediately -- see class javadoc's "ON PROBING" note
     * for why this is the right-sized probe for a Llama backend (no model
     * weights loaded).
     */
    private static void probe(GpuBackend backend) throws Throwable {
        switch (backend) {
            case OPENCL -> new com.github.gbenroscience.gpu.llm.opencl.GpuContext().close();
            case CUDA -> new com.github.gbenroscience.gpu.llm.cuda.GpuContext().close();
            case METAL -> new com.github.gbenroscience.gpu.llm.metal.GpuContext().close();
        }
    }
}