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

    static void run(GpuContext ctx, int k, int n) throws Throwable {
        final long t0 = System.nanoTime();
        final int rows = Math.max(256, n);
        Random r = new Random(11);
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
            note.append("-> chose ").append(NAMES[best]).append(String.format(" (autotune took %.0f ms)", (System.nanoTime() - t0) / 1e6));
        } finally {
            LlamaLayer.freeQuietly(ctx, dw);
            LlamaLayer.freeQuietly(ctx, dx);
            LlamaLayer.freeQuietly(ctx, dout);
        }
        LoadTimer.note(note.toString());
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
                ctx.q8GemmRows = false;
                double legacy = timeGemm(ctx, () -> LlamaLayer.q8_0GemmTiled(ctx, qx, qw, qo, T, rows, k));
                float[] ref = LlamaLayer.downloadFloats(ctx, dout, T * rows);
                ctx.q8GemmRows = true;
                double fast = timeGemm(ctx, () -> LlamaLayer.q8_0GemmTiled(ctx, qx, qw, qo, T, rows, k));
                boolean ok = matches(ref, LlamaLayer.downloadFloats(ctx, dout, T * rows), 1e-4);
                ctx.q8GemmRows = ok && fast < legacy;
                note.append(String.format("Q8_0 legacy %.2f ms vs row-wise %.2f ms%s -> %s; ", legacy, fast, ok ? "" : " (REJECTED: output differs)",
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
                ctx.f32GemmRows = false;
                double legacy = timeGemm(ctx, () -> LlamaLayer.f32GemmTiled(ctx, fa, fw, fo, T, k, k));
                float[] ref = LlamaLayer.downloadFloats(ctx, doutF, T * k);
                ctx.f32GemmRows = true;
                double fast = timeGemm(ctx, () -> LlamaLayer.f32GemmTiled(ctx, fa, fw, fo, T, k, k));
                boolean ok = matches(ref, LlamaLayer.downloadFloats(ctx, doutF, T * k), 1e-4);
                ctx.f32GemmRows = ok && fast < legacy;
                note.append(String.format("F32 legacy %.2f ms vs row-wise %.2f ms%s -> %s", legacy, fast, ok ? "" : " (REJECTED: output differs)",
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

    private interface Launch {

        void run() throws Throwable;
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
