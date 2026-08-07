package com.github.gbenroscience.gpu.evaluator;

import com.github.gbenroscience.gpu.GpuBackend;
import com.github.gbenroscience.gpu.evaluator.cuda.CudaCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.cuda.CudaExpressionBridge;
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
 * com.github.gbenroscience.gpu.opencl.OpenClExpressionBridge and
 * com.github.gbenroscience.gpu.cuda.CudaExpressionBridge. Use this unless
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
 * <b>On auto-detection and class-init:</b> both OpenClCompositeExpression
 * and CudaCompositeExpression bootstrap their backend (device discovery,
 * context creation, kernel build) exactly once per JVM, in a private
 * static nested class's static initializer, the first time that backend
 * is actually used. If that bootstrap fails, the JVM marks the nested
 * class permanently erroneous -- every later reference throws
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

    private static final int PROBE_OP_LOAD = 2; // matches OP_LOAD in both kernels' shared opcode numbering

    private static final Map<GpuBackend, Boolean> AVAILABILITY = new ConcurrentHashMap<>();
    // Retains the REAL exception each backend's probe failed with, not just
    // the boolean outcome. Previously probeQuietly() discarded the actual
    // Throwable entirely, so a genuine configuration error (e.g. a
    // selectDevice()/opencl.gpu.vendor value that matched no installed
    // device) was indistinguishable from "OpenCL just isn't installed" --
    // both produced the same opaque summary message below. Surfacing the
    // real cause (see resolveAutoBackend) fixes that without changing the
    // caching/preference-order behavior at all.
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
                            + "Check that either an OpenCL ICD or an NVIDIA/CUDA driver+NVRTC "
                            + "is actually installed, or select a backend explicitly with "
                            + "GpuExpressionBridge.from(vte, GpuBackend.OPENCL/CUDA) to see the "
                            + "real bootstrap error instead of this summary. The actual per-backend "
                            + "failure (e.g. a bad -Dopencl.gpu.vendor value, not necessarily a "
                            + "missing driver) is attached below as a suppressed exception for "
                            + "each backend that was tried.");
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
     * -Dgpu.backend.preference=cuda,opencl (default) or opencl,cuda, etc.
     * Unrecognized tokens are ignored; an empty/unset property falls back
     * to the default order.
     */
    private static List<GpuBackend> preferenceOrder() {
        String pref = System.getProperty("gpu.backend.preference", "cuda,opencl");
        List<GpuBackend> order = new ArrayList<>();
        for (String token : pref.split(",")) {
            switch (token.trim().toLowerCase(Locale.ROOT)) {
                case "cuda" -> order.add(GpuBackend.CUDA);
                case "opencl" -> order.add(GpuBackend.OPENCL);
                default -> { /* ignore unrecognized token */ }
            }
        }
        if (order.isEmpty()) {
            order.add(GpuBackend.CUDA);
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
     * Forces the backend's real static bootstrap to run now (device
     * discovery, context/queue creation, kernel build -- see each
     * concrete class's javadoc) via a trivial one-instruction expression,
     * rather than deferring to the caller's first real dispatch. This is
     * what makes auto-detection test actual availability instead of
     * guessing from e.g. whether a library file merely exists on disk.
     */
    private static void probe(GpuBackend backend) throws Throwable {
        int[] opcodes = {PROBE_OP_LOAD};
        int[] targetSlots = {0};
        double[] literals = {0.0};

        GpuCompositeExpression expr = switch (backend) {
            case OPENCL -> new OpenClCompositeExpression(opcodes, targetSlots, literals, 1, 1);
            case CUDA -> new CudaCompositeExpression(opcodes, targetSlots, literals, 1, 1);
        };
        expr.close();
    }
}