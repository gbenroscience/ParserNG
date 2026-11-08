package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.core.CpuReference;

import java.lang.foreign.MemorySegment;
import java.util.Random;

/**
 * Chooses, on the actual device, the fastest of the decode Q8_0 GEMV loops (see {@link GpuContext#gemvVariant}):
 * 0 = wg (vload16), 1 = vloadf (same loads, float dp4), 2 = u32 (aligned 4-byte words over 68-byte block pairs).
 *
 * <p>Safety first: a candidate is only eligible if its output on the calibration problem matches the reference
 * variant (0) to a relative 1e-5 -- the loops are mathematically identical, differing only in the order lanes are
 * combined -- and contains no NaN. Timing then uses the median of several launches (wall clock around enqueue +
 * clFinish; a few ms each, so the ~0.1 ms call overhead is irrelevant). The result and every timing go into the
 * {@link LoadTimer} report. Skipped when {@code -Dllm.gemv} forces a variant.
 */
final class GemvAutotune {

    private GemvAutotune() {
    }

    private static final String[] NAMES = {"wg (vload16)", "vloadf (vload16 + float dp4)", "u32 (aligned words)"};

    static double run(GpuContext ctx, int k, int n) throws Throwable {
        final long t0 = System.nanoTime();
        final int rows = Math.max(256, n);
        Random r = new Random(11);
        double[] bestOut = {Double.MAX_VALUE};
        MemorySegment dw = null, dx = null, dout = null;
        StringBuilder note = new StringBuilder("GEMV autotune (K=" + k + ", N=" + rows + ", median of 7): ");
        try {
            dw = LlamaLayer.uploadBytes(ctx, GemvSweep.fakeQ8((long) rows * (k / 32), r));
            float[] x = new float[k];
            for (int i = 0; i < k; i++) {
                x[i] = (float) r.nextGaussian();
            }
            dx = LlamaLayer.uploadBytes(ctx, CpuReference.quantizeQ8_0(x));
            dout = LlamaLayer.allocFloats(ctx, rows);
            float[] ref = null;
            int best = 0;
            double bestMs = Double.MAX_VALUE;
            for (int v = 0; v < 3; v++) {
                if (v == 2 && (k / 32) % 2 != 0) {
                    note.append("u32 n/a (odd block count); ");
                    continue;
                }
                ctx.gemvVariant = v;
                LlamaLayer.q8_0GemvPlain(ctx, dx, dw, dout, rows, k);
                LlamaLayer.finish(ctx);
                float[] got = LlamaLayer.downloadFloats(ctx, dout, rows);
                if (v == 0) {
                    ref = got;
                } else if (!matches(ref, got)) {
                    note.append(NAMES[v]).append(" REJECTED (output differs from the reference variant); ");
                    continue;
                }
                double[] ms = new double[7];
                for (int rep = 0; rep < 7; rep++) {
                    long a = System.nanoTime();
                    LlamaLayer.q8_0GemvPlain(ctx, dx, dw, dout, rows, k);
                    LlamaLayer.finish(ctx);
                    ms[rep] = (System.nanoTime() - a) / 1e6;
                }
                java.util.Arrays.sort(ms);
                double med = ms[3];
                note.append(String.format("%s %.2f ms (%.1f GB/s); ", NAMES[v], med, (double) rows * (k / 32) * 34 / 1e6 / med));
                if (med < bestMs) {
                    bestMs = med;
                    best = v;
                }
            }
            ctx.gemvVariant = best;
            bestOut[0] = bestMs;
            note.append("-> chose ").append(NAMES[best]).append(String.format(" (autotune took %.0f ms)", (System.nanoTime() - t0) / 1e6));
        } finally {
            LlamaLayer.freeQuietly(ctx, dw);
            LlamaLayer.freeQuietly(ctx, dx);
            LlamaLayer.freeQuietly(ctx, dout);
        }
        LoadTimer.note(note.toString());
        return bestOut[0];
    }

    /**
     * Times the original one-thread-per-output prefill GEMMs against the row-wise ones on a 16-token chunk and keeps the
     * faster, provided its output matches the legacy kernel's (relative 1e-4: the reduction order differs). Q8_0 uses
     * (T=16, N=rows, K=k); F32 (the wo projection) uses (T=16, N=k, K=k).
     */
    static void runGemm(GpuContext ctx, int k, int n) throws Throwable {
        final long t0 = System.nanoTime();
        final int T = 16;
        final int rows = Math.max(256, n);
        Random r = new Random(13);
        StringBuilder note = new StringBuilder("GEMM autotune (T=" + T + ", K=" + k + "): ");
        MemorySegment dw = null, dxq = null, dout = null, daF = null, dwF = null, doutF = null;
        try {
            dw = LlamaLayer.uploadBytes(ctx, GemvSweep.fakeQ8((long) rows * (k / 32), r));
            byte[] xq = new byte[T * (k / 32) * 34];
            for (int t = 0; t < T; t++) {
                float[] x = new float[k];
                for (int i = 0; i < k; i++) {
                    x[i] = (float) r.nextGaussian();
                }
                byte[] q = CpuReference.quantizeQ8_0(x);
                System.arraycopy(q, 0, xq, t * q.length, q.length);
            }
            dxq = LlamaLayer.uploadBytes(ctx, xq);
            dout = LlamaLayer.allocFloats(ctx, (long) T * rows);
            if ((k / 32) % 2 == 0) {
                final MemorySegment qx = dxq, qw = dw, qo = dout;
                ctx.q8GemmRows = true;
                double fast = timeGemm(ctx, () -> LlamaLayer.q8_0GemmTiled(ctx, qx, qw, qo, T, rows, k));
                float[] got = LlamaLayer.downloadFloats(ctx, dout, T * rows);
                ctx.q8GemmRows = false;
                double legacy = timeOnce(ctx, () -> LlamaLayer.q8_0GemmTiled(ctx, qx, qw, qo, T, rows, k));
                float[] ref = LlamaLayer.downloadFloats(ctx, dout, T * rows);
                boolean ok = matches(ref, got, 1e-4);
                ctx.q8GemmRows = ok && fast < legacy;
                note.append(String.format("Q8_0 legacy %.2f ms (1 launch) vs row-wise %.2f ms%s -> %s; ", legacy, fast, ok ? "" : " (REJECTED: output differs)",
                        ctx.q8GemmRows ? "row-wise" : "legacy"));
            } else {
                ctx.q8GemmRows = false;
                note.append("Q8_0 row-wise n/a (odd block count) -> legacy; ");
            }
            if (k % 4 == 0) {
                float[] w = new float[k * k];
                for (int i = 0; i < w.length; i += 7) {
                    w[i] = (float) r.nextGaussian() * 0.05f;
                }
                float[] a = new float[T * k];
                for (int i = 0; i < a.length; i++) {
                    a[i] = (float) r.nextGaussian();
                }
                dwF = LlamaLayer.uploadFloats(ctx, w);
                daF = LlamaLayer.uploadFloats(ctx, a);
                doutF = LlamaLayer.allocFloats(ctx, (long) T * k);
                final MemorySegment fw = dwF, fa = daF, fo = doutF;
                ctx.f32GemmRows = true;
                double fast = timeGemm(ctx, () -> LlamaLayer.f32GemmTiled(ctx, fa, fw, fo, T, k, k));
                float[] got = LlamaLayer.downloadFloats(ctx, doutF, T * k);
                ctx.f32GemmRows = false;
                double legacy = timeOnce(ctx, () -> LlamaLayer.f32GemmTiled(ctx, fa, fw, fo, T, k, k));
                float[] ref = LlamaLayer.downloadFloats(ctx, doutF, T * k);
                boolean ok = matches(ref, got, 1e-4);
                ctx.f32GemmRows = ok && fast < legacy;
                note.append(String.format("F32 legacy %.2f ms (1 launch) vs row-wise %.2f ms%s -> %s", legacy, fast, ok ? "" : " (REJECTED: output differs)",
                        ctx.f32GemmRows ? "row-wise" : "legacy"));
            } else {
                ctx.f32GemmRows = false;
                note.append("F32 row-wise n/a (K % 4 != 0) -> legacy");
            }
            note.append(String.format(" (autotune took %.0f ms)", (System.nanoTime() - t0) / 1e6));
        } finally {
            LlamaLayer.freeQuietly(ctx, dw);
            LlamaLayer.freeQuietly(ctx, dxq);
            LlamaLayer.freeQuietly(ctx, dout);
            LlamaLayer.freeQuietly(ctx, daF);
            LlamaLayer.freeQuietly(ctx, dwF);
            LlamaLayer.freeQuietly(ctx, doutF);
        }
        LoadTimer.note(note.toString());
    }

    // ======================================================================================================
    // Q8_0 layout selection (runs BEFORE any weight is uploaded, because it decides how they are uploaded)
    // ======================================================================================================

    /**
     * SoA is chosen unless the best AoS kernel is more than this much faster in the head-to-head measurement.
     * The first version required SoA to win by 8 %; the launch-to-launch noise of the measurement is larger than
     * that (the same AoS kernel measured 1.16, 1.21, 1.35 and 2.69 ms on one Hainan in four starts), so identical
     * code picked AoS in one run (decode 6.9 tok/s, prefill 24 tok/s) and SoA in the next (7.95 and 44 tok/s). SoA was
     * the better layout on both test GPUs for decode and prefill, so the burden of proof is on AoS.
     */
    static final double SOA_TOLERANCE = 1.10;

    /** The layout decision from per-launch times (ms) of the best AoS kernel and the best SoA kernel. */
    static boolean preferSoa(double aosMs, double soaMs) {
        return soaMs <= aosMs * SOA_TOLERANCE;
    }

    /**
     * Decides the Q8_0 device layout of this context and the per-K lanes-per-row of the SoA GEMV, on the real device:
     * <ol>
     *   <li>-Dllm.layout=aos, or SoA kernels not compiled: AoS (with the usual GEMV variant autotune).</li>
     *   <li>otherwise time the best AoS decode kernel and every SoA lanes-per-row variant on the same synthetic
     *       problem, then head-to-head (interleaved, batched); SoA is chosen unless AoS is more than {@link #SOA_TOLERANCE}-1 faster (always, when -Dllm.layout=soa).</li>
     *   <li>every SoA kernel used must reproduce a CPU reference (GEMV for each tuned K; the prefill GEMM against
     *       the same reference) to a relative 1e-4. If any check fails the context falls back to AoS (or, when SoA was
     *       forced, throws): a miscompiled kernel must never silently produce wrong text.</li>
     * </ol>
     * Sets {@code ctx.q8Soa}, {@code ctx.soaLprByK} and (for AoS) {@code ctx.gemvVariant}; everything is reported via {@link LoadTimer}.
     */
    static void selectLayout(GpuContext ctx, LlamaLayer.Config cfg) throws Throwable {
        final String req = Variants.layoutRequest();
        ctx.q8Soa = false;
        if (req.equals("soa") && !ctx.soaAvailable) {
            throw new IllegalStateException("-Dllm.layout=soa was requested but the SoA kernels did not compile on this device");
        }
        final boolean aosTune = Variants.autotuneGemv() && !Variants.legacyGemv();
        final int rows = Math.max(256, Math.min(cfg.hidden_dim, 4096));
        double aosMs = Double.MAX_VALUE;
        if (!req.equals("soa") && aosTune) {
            aosMs = run(ctx, cfg.dim, Math.min(cfg.hidden_dim, 4096));
        }
        if (req.equals("aos") || !ctx.soaAvailable) {
            LoadTimer.note("Q8_0 layout: AoS (" + (req.equals("aos") ? "requested" : "SoA kernels unavailable") + ")");
            return;
        }
        StringBuilder note = new StringBuilder("Q8_0 layout selection: ");
        boolean ok = true;
        double soaMs = Double.MAX_VALUE;
        java.util.Set<Integer> ks = new java.util.LinkedHashSet<>();
        ks.add(cfg.dim);
        ks.add(cfg.hidden_dim);
        ks.add(cfg.num_heads * cfg.head_dim);
        try {
            ctx.q8Soa = true;
            for (int k : ks) {
                if (k % 32 != 0) {
                    ok = false;
                    note.append("K=").append(k).append(" is not a multiple of 32; ");
                    break;
                }
                int nRows = (k == cfg.dim) ? rows : Math.max(256, Math.min(cfg.dim, 2048));
                double[] ms = new double[1];
                int lpr = tuneSoaLpr(ctx, k, nRows, note, ms);
                if (lpr < 0) {
                    ok = false;
                    break;
                }
                ctx.soaLprByK.put(k, lpr);
                if (k == cfg.dim) {
                    soaMs = ms[0];
                    ctx.soaLpr = lpr;
                }
            }
            if (ok && !validateSoaGemm(ctx, cfg.dim, rows, note)) {
                ok = false;
            }
            if (ok && !req.equals("soa")) {
                double[] h2h = headToHead(ctx, cfg.dim, rows, note);
                if (h2h != null) {
                    aosMs = h2h[0];
                    soaMs = h2h[1];
                }
            }
        } finally {
            ctx.q8Soa = false;
        }
        boolean forced = req.equals("soa");
        if (!ok) {
            ctx.soaLprByK.clear();
            if (forced) {
                LoadTimer.note(note.toString());
                throw new IllegalStateException("-Dllm.layout=soa was requested but the SoA kernels failed validation on this device: " + note);
            }
            note.append("-> AoS (SoA failed validation)");
        } else if (forced || preferSoa(aosMs, soaMs)) {
            ctx.q8Soa = true;
            note.append(String.format("-> SoA (%.2f ms vs best AoS %s)", soaMs, aosMs == Double.MAX_VALUE ? "n/a" : String.format("%.2f ms", aosMs)));
        } else {
            note.append(String.format("-> AoS (best AoS %.2f ms is more than %.0f%% faster than SoA %.2f ms)", aosMs, (SOA_TOLERANCE - 1) * 100, soaMs));
            ctx.soaLprByK.clear();
        }
        LoadTimer.note(note.toString());
    }

    /**
     * Times the best AoS kernel and the chosen SoA kernel on the same problem, alternating between them so clock and
     * thermal drift hits both equally, with eight launches per synchronisation so the launch round trip is amortised
     * (a single launch of this size takes about a millisecond, the same order as the round trip). Returns
     * {aosMs, soaMs} per launch (median of five rounds), or null if anything fails. ctx.q8Soa must be true on entry
     * and is true on exit.
     */
    private static double[] headToHead(GpuContext ctx, int k, int rows, StringBuilder note) {
        MemorySegment dwA = null, dwS = null, dxf = null, dxA = null, dxS = null, dout = null;
        try {
            Random r = new Random(41 + k);
            byte[] aos = GemvSweep.fakeQ8((long) rows * (k / 32), r);
            float[] x = new float[k];
            for (int i = 0; i < k; i++) {
                x[i] = (float) r.nextGaussian();
            }
            dwA = LlamaLayer.uploadBytes(ctx, aos);
            dwS = LlamaLayer.uploadBytes(ctx, Q8Layout.repackWeightsSoa(aos, rows, k));
            dxf = LlamaLayer.uploadFloats(ctx, x);
            dxA = LlamaLayer.allocBytes(ctx, Q8Layout.activationBytes(false, k));
            dxS = LlamaLayer.allocBytes(ctx, Q8Layout.activationBytes(true, k));
            dout = LlamaLayer.allocFloats(ctx, rows);
            ctx.q8Soa = false;
            LlamaLayer.quantizeActivationQ8_0(ctx, dxf, dxA, k);
            ctx.q8Soa = true;
            LlamaLayer.quantizeActivationQ8_0(ctx, dxf, dxS, k);
            final int rounds = 5, batch = 8;
            double[] a = new double[rounds], b = new double[rounds];
            for (int round = 0; round < rounds + 1; round++) { // round 0 is a warm-up
                boolean aosFirst = round % 2 == 0;
                for (int pass = 0; pass < 2; pass++) {
                    boolean aosNow = (pass == 0) == aosFirst;
                    ctx.q8Soa = !aosNow;
                    long t0 = System.nanoTime();
                    for (int i = 0; i < batch; i++) {
                        if (aosNow) {
                            LlamaLayer.q8_0GemvPlain(ctx, dxA, dwA, dout, rows, k);
                        } else {
                            LlamaLayer.q8_0GemvPlain(ctx, dxS, dwS, dout, rows, k);
                        }
                    }
                    LlamaLayer.finish(ctx);
                    double ms = (System.nanoTime() - t0) / 1e6 / batch;
                    if (round > 0) {
                        (aosNow ? a : b)[round - 1] = ms;
                    }
                }
            }
            java.util.Arrays.sort(a);
            java.util.Arrays.sort(b);
            note.append(String.format("head-to-head at K=%d, N=%d (5 alternating rounds x 8 launches): AoS %.2f ms vs SoA %.2f ms per launch; ", k, rows, a[rounds / 2], b[rounds / 2]));
            return new double[]{a[rounds / 2], b[rounds / 2]};
        } catch (Throwable t) {
            note.append("head-to-head measurement failed (").append(t).append("); ");
            return null;
        } finally {
            ctx.q8Soa = true;
            for (MemorySegment m : new MemorySegment[]{dwA, dwS, dxf, dxA, dxS, dout}) {
                LlamaLayer.freeQuietly(ctx, m);
            }
        }
    }

    /**
     * Times each SoA GEMV lanes-per-row variant on (K=k, N=rows), checks it against the CPU reference, and returns the
     * fastest valid lanes-per-row (or -1 if none validates). ctx.q8Soa must be true. Median wall time of 7 launches.
     */
    private static int tuneSoaLpr(GpuContext ctx, int k, int rows, StringBuilder note, double[] bestMsOut) throws Throwable {
        Random r = new Random(17 + k);
        MemorySegment dw = null, dxf = null, dx = null, dout = null;
        note.append("SoA GEMV (K=").append(k).append(", N=").append(rows).append("): ");
        try {
            byte[] aos = GemvSweep.fakeQ8((long) rows * (k / 32), r);
            dw = LlamaLayer.uploadBytes(ctx, Q8Layout.repackWeightsSoa(aos, rows, k));
            float[] x = new float[k];
            for (int i = 0; i < k; i++) {
                x[i] = (float) r.nextGaussian();
            }
            float[] ref = CpuReference.q8Gemv(CpuReference.quantizeQ8_0(x), aos, rows, k);
            dxf = LlamaLayer.uploadFloats(ctx, x);
            dx = LlamaLayer.allocBytes(ctx, Q8Layout.activationBytes(true, k));
            dout = LlamaLayer.allocFloats(ctx, rows);
            LlamaLayer.quantizeActivationQ8_0(ctx, dxf, dx, k);
            int best = -1;
            double bestMs = Double.MAX_VALUE;
            for (int lpr : KernelSourceSoa.GEMV_LPR) {
                ctx.soaLprByK.put(k, lpr);
                LlamaLayer.q8_0GemvSoa(ctx, dx, dw, dout, rows, k);
                LlamaLayer.finish(ctx);
                float[] got = LlamaLayer.downloadFloats(ctx, dout, rows);
                if (!matches(ref, got, 1e-4)) {
                    note.append("l").append(lpr).append(" REJECTED (differs from CPU reference); ");
                    continue;
                }
                double[] ms = new double[7];
                for (int rep = 0; rep < ms.length; rep++) {
                    long a = System.nanoTime();
                    LlamaLayer.q8_0GemvSoa(ctx, dx, dw, dout, rows, k);
                    LlamaLayer.finish(ctx);
                    ms[rep] = (System.nanoTime() - a) / 1e6;
                }
                java.util.Arrays.sort(ms);
                double med = ms[3];
                note.append(String.format("l%d %.2f ms (%.1f GB/s); ", lpr, med, (double) Q8Layout.weightBytes(true, rows, k) / 1e6 / med));
                if (med < bestMs) {
                    bestMs = med;
                    best = lpr;
                }
            }
            bestMsOut[0] = bestMs;
            if (best < 0) {
                note.append("no variant validated; ");
            } else {
                note.append("best l").append(best).append("; ");
            }
            return best;
        } finally {
            ctx.soaLprByK.remove(k);
            LlamaLayer.freeQuietly(ctx, dw);
            LlamaLayer.freeQuietly(ctx, dxf);
            LlamaLayer.freeQuietly(ctx, dx);
            LlamaLayer.freeQuietly(ctx, dout);
        }
    }

    /** Checks the SoA prefill GEMM (T=16 tokens, N=rows, K=k) against the CPU reference token by token. ctx.q8Soa must be true. */
    private static boolean validateSoaGemm(GpuContext ctx, int k, int rows, StringBuilder note) throws Throwable {
        final int T = 16;
        Random r = new Random(23);
        MemorySegment dw = null, dxf = null, dx = null, dout = null;
        try {
            byte[] aos = GemvSweep.fakeQ8((long) rows * (k / 32), r);
            dw = LlamaLayer.uploadBytes(ctx, Q8Layout.repackWeightsSoa(aos, rows, k));
            float[] x = new float[T * k];
            for (int i = 0; i < x.length; i++) {
                x[i] = (float) r.nextGaussian();
            }
            dxf = LlamaLayer.uploadFloats(ctx, x);
            dx = LlamaLayer.allocBytes(ctx, Q8Layout.activationBytes(true, (long) T * k));
            dout = LlamaLayer.allocFloats(ctx, (long) T * rows);
            LlamaLayer.quantizeActivationQ8_0(ctx, dxf, dx, T * k);
            LlamaLayer.q8_0GemmSoa(ctx, dx, dw, dout, T, rows, k);
            LlamaLayer.finish(ctx);
            float[] got = LlamaLayer.downloadFloats(ctx, dout, T * rows);
            for (int t = 0; t < T; t++) {
                float[] xt = java.util.Arrays.copyOfRange(x, t * k, (t + 1) * k);
                float[] ref = CpuReference.q8Gemv(CpuReference.quantizeQ8_0(xt), aos, rows, k);
                if (!matches(ref, java.util.Arrays.copyOfRange(got, t * rows, (t + 1) * rows), 1e-4)) {
                    note.append("SoA prefill GEMM REJECTED (token ").append(t).append(" differs from CPU reference); ");
                    return false;
                }
            }
            note.append("SoA prefill GEMM validated (T=16); ");
            return true;
        } finally {
            LlamaLayer.freeQuietly(ctx, dw);
            LlamaLayer.freeQuietly(ctx, dxf);
            LlamaLayer.freeQuietly(ctx, dx);
            LlamaLayer.freeQuietly(ctx, dout);
        }
    }

    private interface Launch {

        void run() throws Throwable;
    }

    /** Single timed launch (no warm-up): used for the legacy kernel, which can be 100x slower and need not be repeated. */
    private static double timeOnce(GpuContext ctx, Launch l) throws Throwable {
        long a = System.nanoTime();
        l.run();
        LlamaLayer.finish(ctx);
        return (System.nanoTime() - a) / 1e6;
    }

    /** Median wall time (ms) of 5 launches after one warm-up. */
    private static double timeGemm(GpuContext ctx, Launch l) throws Throwable {
        l.run();
        LlamaLayer.finish(ctx);
        double[] ms = new double[5];
        for (int i = 0; i < ms.length; i++) {
            long a = System.nanoTime();
            l.run();
            LlamaLayer.finish(ctx);
            ms[i] = (System.nanoTime() - a) / 1e6;
        }
        java.util.Arrays.sort(ms);
        return ms[2];
    }

    private static boolean matches(float[] ref, float[] got, double rel) {
        double scale = 0;
        for (float v : ref) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < ref.length; i++) {
            if (Float.isNaN(got[i]) || Math.abs(got[i] - ref[i]) > rel * Math.max(scale, 1e-6)) {
                return false;
            }
        }
        return true;
    }

    private static boolean matches(float[] ref, float[] got) {
        double scale = 0;
        for (float v : ref) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < ref.length; i++) {
            if (Float.isNaN(got[i]) || Math.abs(got[i] - ref[i]) > 1e-5 * Math.max(scale, 1e-6)) {
                return false;
            }
        }
        return true;
    }
}
