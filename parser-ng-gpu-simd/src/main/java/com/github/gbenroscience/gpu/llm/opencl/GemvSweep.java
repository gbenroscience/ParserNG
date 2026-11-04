package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.core.CpuReference;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Model-free, event-timed benchmark of the projection kernels at the exact shapes of a transformer, to find out
 * whether GEMV time scales with BYTES read (memory-bound) or with the NUMBER OF OUTPUT ROWS (per-work-group /
 * per-launch overhead). It also times the LM head in isolation. All times are OpenCL device timestamps
 * (CL_PROFILING_COMMAND_START/END), not host wall clock.
 *
 * <p>Method: for every shape, 3 warm-up launches, then {@code reps} timed launches whose events are read after a
 * clFinish. Both the current work-group-per-row kernel and the original one-thread-per-row kernel are timed.
 * An ordinary-least-squares fit {@code t_us = a + b*rows + c*MB} over all Q8_0 shapes separates the fixed launch
 * cost (a), the cost per output row (b) and the cost per megabyte of weights (c, i.e. 1000/c GB/s).
 */
final class GemvSweep {

    private GemvSweep() {
    }

    private static final class Shape {

        final String name;
        final int n;
        final int k;
        final boolean f32;

        Shape(String name, int n, int k, boolean f32) {
            this.name = name;
            this.n = n;
            this.k = k;
            this.f32 = f32;
        }

        long bytes() {
            return f32 ? (long) n * k * 4 : (long) n * (k / 32) * 34;
        }
    }

    /** Valid Q8_0 data (finite scale, random int8 quants) generated cheaply; the values do not affect GPU timing. */
    static byte[] fakeQ8(long blocks, Random r) {
        byte[] b = new byte[(int) (blocks * 34)];
        int scale = com.github.gbenroscience.gpu.llm.core.Fp16.floatToHalfBits(0.01f);
        byte[] q = new byte[32 * 1024];
        r.nextBytes(q);
        for (long i = 0; i < blocks; i++) {
            int o = (int) (i * 34);
            b[o] = (byte) scale;
            b[o + 1] = (byte) (scale >> 8);
            System.arraycopy(q, (int) ((i * 32) % (q.length - 32)), b, o + 2, 32);
        }
        return b;
    }

    /** Median device time (microseconds) of {@code reps} launches of one shape with the given kernel variant. */
    private static double timeShape(GpuContext ctx, Shape sh, boolean legacy, int reps, Random r) throws Throwable {
        System.setProperty("llm.gemv", legacy ? "legacy" : "wg");
        MemorySegment dw, dx, dout = LlamaLayer.allocFloats(ctx, sh.n);
        if (sh.f32) {
            float[] w = new float[sh.n * sh.k];
            for (int i = 0; i < w.length; i += 997) {
                w[i] = r.nextFloat();
            }
            dw = LlamaLayer.uploadFloats(ctx, w);
            dx = LlamaLayer.uploadFloats(ctx, new float[sh.k]);
        } else {
            dw = LlamaLayer.uploadBytes(ctx, fakeQ8((long) sh.n * (sh.k / 32), r));
            dx = LlamaLayer.uploadBytes(ctx, CpuReference.quantizeQ8_0(randomFloats(sh.k, r)));
        }
        try {
            String label = sh.name + (legacy ? " [legacy]" : " [wg]");
            ctx.prof.phase = "sweep";
            for (int i = 0; i < 3; i++) {
                launch(ctx, sh, dx, dw, dout);
            }
            LlamaLayer.finish(ctx);
            ctx.prof.flush();
            ctx.prof.resetStats();
            double[] us = new double[reps];
            for (int i = 0; i < reps; i++) {
                ctx.prof.stage = label + "#" + i;
                launch(ctx, sh, dx, dw, dout);
            }
            LlamaLayer.finish(ctx);
            ctx.prof.flush();
            for (int i = 0; i < reps; i++) {
                us[i] = ctx.prof.stageStat("sweep", label + "#" + i).deviceNs / 1e3;
            }
            java.util.Arrays.sort(us);
            return us[reps / 2];
        } finally {
            System.clearProperty("llm.gemv");
            LlamaLayer.freeQuietly(ctx, dw);
            LlamaLayer.freeQuietly(ctx, dx);
            LlamaLayer.freeQuietly(ctx, dout);
        }
    }

    private static float[] randomFloats(int n, Random r) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) r.nextGaussian();
        }
        return x;
    }

    private static void launch(GpuContext ctx, Shape sh, MemorySegment dx, MemorySegment dw, MemorySegment dout) throws Throwable {
        if (sh.f32) {
            LlamaLayer.f32Gemv(ctx, dx, dw, dout, sh.k, sh.n);
        } else {
            LlamaLayer.q8_0GemvPlain(ctx, dx, dw, dout, sh.n, sh.k);
        }
    }

    /** Least squares for y = a + b*x1 + c*x2 via the 3x3 normal equations; returns {a, b, c, r2}. */
    static double[] fit(List<double[]> rows) {
        double[][] m = new double[3][4];
        double sy = 0, syy = 0;
        for (double[] r : rows) {
            double[] x = {1, r[0], r[1]};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    m[i][j] += x[i] * x[j];
                }
                m[i][3] += x[i] * r[2];
            }
            sy += r[2];
            syy += r[2] * r[2];
        }
        for (int i = 0; i < 3; i++) {
            int p = i;
            for (int k = i + 1; k < 3; k++) {
                if (Math.abs(m[k][i]) > Math.abs(m[p][i])) {
                    p = k;
                }
            }
            double[] t = m[i];
            m[i] = m[p];
            m[p] = t;
            if (Math.abs(m[i][i]) < 1e-12) {
                return new double[]{Double.NaN, Double.NaN, Double.NaN, Double.NaN};
            }
            for (int k = 0; k < 3; k++) {
                if (k != i) {
                    double f = m[k][i] / m[i][i];
                    for (int j = i; j < 4; j++) {
                        m[k][j] -= f * m[i][j];
                    }
                }
            }
        }
        double a = m[0][3] / m[0][0], b = m[1][3] / m[1][1], c = m[2][3] / m[2][2];
        double mean = sy / rows.size(), ssTot = syy - rows.size() * mean * mean, ssRes = 0;
        for (double[] r : rows) {
            double e = r[2] - (a + b * r[0] + c * r[1]);
            ssRes += e * e;
        }
        return new double[]{a, b, c, ssTot <= 0 ? 1.0 : 1.0 - ssRes / ssTot};
    }

    static void run(GpuContext ctx, int dim, int hidden, int heads, int kvHeads, int headDim, int layers, int vocab, boolean includeF32Head) throws Throwable {
        if (ctx.prof == null) {
            throw new IllegalStateException("the sweep needs an event-profiling context");
        }
        Random r = new Random(7);
        final int qd = heads * headDim, kvd = kvHeads * headDim;
        List<Shape> model = new ArrayList<>();
        model.add(new Shape("q proj      N=" + qd + " K=" + dim, qd, dim, false));
        model.add(new Shape("k/v proj    N=" + kvd + " K=" + dim, kvd, dim, false));
        model.add(new Shape("gate/up     N=" + hidden + " K=" + dim, hidden, dim, false));
        model.add(new Shape("down        N=" + dim + " K=" + hidden, dim, hidden, false));
        model.add(new Shape("wo (F32)    N=" + dim + " K=" + qd, dim, qd, true));
        model.add(new Shape("lm head Q8  N=" + vocab + " K=" + dim, vocab, dim, false));
        if (includeF32Head && Runtime.getRuntime().maxMemory() > 3L << 30) {
            model.add(new Shape("lm head F32 N=" + vocab + " K=" + dim, vocab, dim, true));
        }
        // equal-bytes series: the same ~18 MB of Q8_0 weights arranged as few long rows or many short rows
        List<Shape> series = new ArrayList<>();
        long elems = 2048L * 8192;
        for (int k : new int[]{32768, 8192, 2048, 512, 128}) {
            int n = (int) (elems / k);
            series.add(new Shape("equal-bytes  N=" + n + " K=" + k, n, k, false));
        }
        final int reps = 11;
        System.out.printf("%n==================== GEMV sweep (device time from OpenCL events, median of %d) ====================%n", reps);
        System.out.printf("%-34s %10s %12s %12s %9s %12s %9s%n", "shape", "MB", "wg-per-row us", "GB/s", "legacy us", "legacy GB/s", "speedup");
        double projectionsMs = 0;
        List<double[]> fitRows = new ArrayList<>();
        List<Shape> all = new ArrayList<>(model);
        all.addAll(series);
        double headUs = Double.NaN;
        for (Shape sh : all) {
            double wg = timeShape(ctx, sh, false, reps, r);
            double lg = timeShape(ctx, sh, true, reps, r);
            double mb = sh.bytes() / 1e6;
            System.out.printf("%-34s %10.2f %12.1f %12.2f %9.1f %12.2f %8.2fx%n", sh.name, mb, wg, sh.bytes() / 1e3 / wg, lg, sh.bytes() / 1e3 / lg, lg / wg);
            if (!sh.f32) {
                fitRows.add(new double[]{sh.n, mb, wg});
            }
            if (sh.name.startsWith("lm head Q8")) {
                headUs = wg;
            }
            if (sh.name.startsWith("q proj")) {
                projectionsMs += layers * wg / 1e3;
            } else if (sh.name.startsWith("k/v")) {
                projectionsMs += layers * 2 * wg / 1e3;
            } else if (sh.name.startsWith("gate/up")) {
                projectionsMs += layers * 2 * wg / 1e3;
            } else if (sh.name.startsWith("down") || sh.name.startsWith("wo")) {
                projectionsMs += layers * wg / 1e3;
            }
        }
        double[] f = fit(fitRows);
        System.out.printf("%nfit over the wg-per-row Q8_0 shapes:  t_us = %.1f + %.4f*rows + %.2f*MB   (R^2 = %.3f)%n", f[0], f[1], f[2], f[3]);
        System.out.printf("  fixed cost per launch %.1f us; cost per output row (one work-group) %.4f us; cost per MB of weights %.2f us (= %.2f GB/s)%n",
                f[0], f[1], f[2], 1000.0 / f[2]);
        System.out.printf("  => for a 2048x8192 Q8_0 matrix: %.0f us from rows, %.0f us from bytes%n", f[1] * 2048, f[2] * 2048 * 8192 / 32 * 34 / 1e6);
        System.out.printf("%nprojections-only reconstruction of one decode token (%d layers x q,k,v,wo,gate,up,down + LM head Q8): %.1f ms%n",
                layers, projectionsMs + (Double.isNaN(headUs) ? 0 : headUs / 1e3));
        System.out.println("(compare with the measured 'logits + readback wait' per token; the remainder is attention, norms, rope, activations, launch gaps)");
    }
}
