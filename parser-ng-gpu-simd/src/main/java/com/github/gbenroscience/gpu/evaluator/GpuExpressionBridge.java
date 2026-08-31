package com.github.gbenroscience.gpu.evaluator;

import com.github.gbenroscience.gpu.GpuBackend;
import com.github.gbenroscience.gpu.evaluator.cuda.CudaCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.cuda.CudaExpressionBridge;
import com.github.gbenroscience.gpu.evaluator.metal.MetalCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.metal.MetalExpressionBridge;
import com.github.gbenroscience.gpu.evaluator.opencl.OpenClCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.opencl.OpenClExpressionBridge;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Single entry point for building a GPU-backed evaluator without the
 * caller needing to know or choose between
 * com.github.gbenroscience.gpu.opencl.OpenClExpressionBridge,
 * com.github.gbenroscience.gpu.cuda.CudaExpressionBridge, and
 * com.github.gbenroscience.gpu.metal.MetalExpressionBridge. Use this unless
 * a specific backend is a hard requirement -- go to the backend-specific
 * bridge directly only when you need that.
 *
 * <pre>
  MathExpression me = new MathExpression("3*cos(x-2)+ln(3x^3-5x-4*tan(x))");
  VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

  try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte)) {       // auto-picks a backend
      gpu.applyBulk(flatVariables, output);
  }

  try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte, GpuBackend.OPENCL)) {  // explicit
      gpu.applyBulk(flatVariables, output);
  }
</pre>
 *
 * <b>On METAL specifically:</b> Metal is macOS-only and, per
 * MetalCompositeExpression's javadoc, has no double-precision path at all
 * (every {@code double}-taking {@code GpuCompositeExpression} method throws
 * {@link UnsupportedOperationException} on that backend). Auto-detection
 * here treats METAL exactly like any other backend for the purpose of
 * "does its bootstrap succeed" -- on Linux/Windows the Metal.framework/
 * libobjc lookups in MetalBindings simply fail to load and
 * {@link #probeQuietly} records that as unavailable, the same way a
 * missing OpenCL ICD or missing NVRTC does. Auto-detection does NOT know
 * or care whether the caller's expression will later be evaluated in
 * double precision -- if that matters, pick the backend explicitly with
 * {@code GpuBackend.OPENCL} or {@code GpuBackend.CUDA} rather than relying
 * on auto-detection, since METAL winning the race on a Mac would silently
 * make every double-precision call on that evaluator throw.
 *
 * <b>On auto-detection and class-init:</b> OpenClCompositeExpression,
 * CudaCompositeExpression, and MetalCompositeExpression each bootstrap their
 * backend (device discovery, context creation, kernel build) exactly once
 * per JVM, in a private static nested class's static initializer, the first
 * time that backend is actually used. If that bootstrap fails, the JVM marks
 * the nested class permanently erroneous -- every later reference throws
 * NoClassDefFoundError, not a fresh retry of the failure. That's exactly
 * the behavior auto-detection wants (don't keep re-probing a backend
 * that's already been shown not to work), so {@link #from(VectorTurboEvaluator)}
 * leans on it directly rather than working around it: the first probe of
 * each backend actually triggers its real bootstrap, the result is cached
 * here too (to skip even the fast NoClassDefFoundError path on repeat
 * calls), and that decision holds for the rest of the JVM's life --
 * a backend that fails once (e.g. a driver briefly not ready) will not be
 * retried without a process restart.
 */
public final class GpuExpressionBridge {

    private static final int PROBE_OP_LOAD = 2; // matches OP_LOAD in all three kernels' shared opcode numbering

    private static final Map<GpuBackend, Boolean> AVAILABILITY = new ConcurrentHashMap<>();
    // Retains the REAL exception each backend's probe failed with, not just
    // the boolean outcome. Previously probeQuietly() discarded the actual
    // Throwable entirely, so a genuine configuration error (e.g. a
    // selectDevice()/opencl.gpu.vendor or metal.gpu.vendor value that
    // matched no installed device) was indistinguishable from "that backend
    // just isn't installed" -- both produced the same opaque summary message
    // below. Surfacing the real cause (see resolveAutoBackend) fixes that
    // without changing the caching/preference-order behavior at all.
    private static final Map<GpuBackend, Throwable> LAST_FAILURE = new ConcurrentHashMap<>();
    private static volatile GpuBackend resolvedAutoBackend;
    private static final Object RESOLVE_LOCK = new Object();

    private GpuExpressionBridge() {
    }

    // ================= Explicit backend =================
    public static GpuCompositeExpression from(VectorTurboEvaluator vte, GpuBackend backend) {
        return switch (backend) {
            case OPENCL -> OpenClExpressionBridge.from(vte);
            case CUDA -> CudaExpressionBridge.from(vte);
            case METAL -> MetalExpressionBridge.from(vte);
        };
    }

    public static GpuCompositeExpression compile(MathExpression me, GpuBackend backend) throws Throwable {
        return from(new VectorTurboEvaluator(me), backend);
    }

    // ================= Auto-detected backend =================
    public static GpuCompositeExpression from(VectorTurboEvaluator vte) {
        return from(vte, resolveAutoBackend());
    }

    public static GpuCompositeExpression compile(MathExpression me) throws Throwable {
        return from(new VectorTurboEvaluator(me));
    }

    /**
     * Whether a given backend's bootstrap has succeeded (or would succeed,
     * probing it now if it hasn't been tried yet this JVM run). Safe to
     * call speculatively -- results are cached, so repeat calls are free.
     */
    public static boolean isAvailable(GpuBackend backend) {
        return AVAILABILITY.computeIfAbsent(backend, GpuExpressionBridge::probeQuietly);
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
                    "No usable GPU backend found (tried " + order + " in that order). "
                            + "Check that either an OpenCL ICD, an NVIDIA/CUDA driver+NVRTC, or "
                            + "(on macOS) Metal.framework is actually installed/available, or select "
                            + "a backend explicitly with GpuExpressionBridge.from(vte, "
                            + "GpuBackend.OPENCL/CUDA/METAL) to see the real bootstrap error instead "
                            + "of this summary. The actual per-backend failure (e.g. a bad "
                            + "-Dopencl.gpu.vendor/-Dmetal.gpu.vendor value, not necessarily a missing "
                            + "driver) is attached below as a suppressed exception for each backend "
                            + "that was tried.");
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
     * Preference order for auto-detection, configurable via
     * -Dgpu.backend.preference=cuda,opencl,metal (comma-separated, any
     * order/subset). Unrecognized tokens are ignored; an empty/unset
     * property falls back to a platform-aware default: METAL is only worth
     * trying first on macOS (it's the only backend of the three that is
     * NOT deprecated there, and it's the only one of the three that never
     * runs anywhere else), everywhere else the default is CUDA then OpenCL
     * exactly as before this backend was added. METAL is still appended as
     * a low-priority fallback on non-mac default orders too -- its probe
     * simply fails fast there (see class javadoc), so including it costs
     * nothing and covers the odd case of a JVM property misconfiguration
     * masking as "not on a Mac".
     */
    private static List<GpuBackend> preferenceOrder() {
        String pref = System.getProperty("gpu.backend.preference");
        if (pref == null || pref.isBlank()) {
            return defaultPreferenceOrder();
        }
        List<GpuBackend> order = new ArrayList<>();
        for (String token : pref.split(",")) {
            switch (token.trim().toLowerCase(Locale.ROOT)) {
                case "cuda" -> order.add(GpuBackend.CUDA);
                case "opencl" -> order.add(GpuBackend.OPENCL);
                case "metal" -> order.add(GpuBackend.METAL);
                default -> { /* ignore unrecognized token */ }
            }
        }
        return order.isEmpty() ? defaultPreferenceOrder() : order;
    }

    private static List<GpuBackend> defaultPreferenceOrder() {
        List<GpuBackend> order = new ArrayList<>();
        if (isMacOs()) {
            order.add(GpuBackend.METAL);
            order.add(GpuBackend.OPENCL);
            order.add(GpuBackend.CUDA);
        } else {
            order.add(GpuBackend.CUDA);
            order.add(GpuBackend.OPENCL);
            order.add(GpuBackend.METAL);
        }
        return order;
    }

    private static boolean isMacOs() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
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
     * Forces the backend's real static bootstrap to run now (device
     * discovery, context/queue creation, kernel build -- see each
     * concrete class's javadoc) via a trivial one-instruction expression,
     * rather than deferring to the caller's first real dispatch. This is
     * what makes auto-detection test actual availability instead of
     * guessing from e.g. whether a library file merely exists on disk.
     *
     * For METAL this exercises exactly the same bootstrap path
     * {@code applyBulkF32}/{@code applyBulk(float[]...)} would later use --
     * the probe expression's single literal is a double 0.0 narrowed to
     * float internally by MetalCompositeExpression's constructor, same as
     * any real caller's literals would be, so this is a faithful
     * availability check even though METAL never touches the double path.
     */
    private static void probe(GpuBackend backend) throws Throwable {
        int[] opcodes = {PROBE_OP_LOAD};
        int[] targetSlots = {0};
        double[] literals = {0.0};

        GpuCompositeExpression expr = switch (backend) {
            case OPENCL -> new OpenClCompositeExpression(opcodes, targetSlots, literals, 1, 1);
            case CUDA -> new CudaCompositeExpression(opcodes, targetSlots, literals, 1, 1);
            case METAL -> new MetalCompositeExpression(opcodes, targetSlots, literals, 1, 1);
        };
        expr.close();
    }
}