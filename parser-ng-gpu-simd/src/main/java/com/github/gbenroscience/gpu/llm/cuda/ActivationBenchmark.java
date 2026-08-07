package com.github.gbenroscience.gpu.llm.cuda;

import java.util.Locale;
import java.util.Random;

/**
 * CUDA counterpart of {@code com.github.gbenroscience.gpu.opencl.llm.ActivationBenchmark}.
 * Same tests, same tolerance, same benchmark structure -- the OpenCL
 * version's class javadoc (why three timing numbers, why the tolerance
 * isn't zero) applies unchanged here. The only things that changed are
 * mechanical: device buffers are {@code long} CUdeviceptr values instead of
 * {@code MemorySegment} cl_mem handles, and "does the kernel actually run"
 * is synced with {@code cuCtxSynchronize} (LlamaLayer.finish) instead of
 * {@code clFinish}.
 *
 * Run with (adjust classpath):
 *   java --enable-preview -cp &lt;your-classpath&gt; com.github.gbenroscience.gpu.cuda.llm.CudaActivationBenchmark
 */
public final class ActivationBenchmark {

    private ActivationBenchmark() {
    }

    private static final float TOLERANCE_ABS = 2e-3f;
    private static final int[] CORRECTNESS_SIZES = {1, 7, 32, 255, 256, 257, 1024, 11008};
    private static final int[] BENCHMARK_SIZES = {4096, 11008, 14336};
    private static final int WARMUP_ITERS = 20;
    private static final int TIMED_ITERS = 200;

    public static void main(String[] args) throws Throwable {
        GpuContext ctx = new GpuContext();
        try {
            System.out.println("=== Correctness: GPU kernels vs CPU reference (tolerance=" + TOLERANCE_ABS + ") ===");
            boolean allPassed = true;
            allPassed &= runCorrectness(ctx, Activation.SWIGLU);
            allPassed &= runCorrectness(ctx, Activation.GEGLU);
            allPassed &= runCorrectness(ctx, Activation.GELU);

            if (!allPassed) {
                System.err.println();
                System.err.println("One or more correctness checks FAILED -- fix before trusting benchmark numbers "
                        + "or any real model output from this CUDA port.");
                System.exit(1);
            }

            System.out.println();
            System.out.println("=== Benchmark: GPU (kernel-only / round-trip) vs single-threaded CPU ===");
            System.out.printf(Locale.ROOT, "%-8s %8s %14s %16s %16s %10s %10s%n",
                    "kernel", "size", "CPU(ms)", "GPU-kernel(ms)", "GPU-roundtrip(ms)", "spd(k)", "spd(rt)");
            for (int size : BENCHMARK_SIZES) {
                benchmark(ctx, Activation.SWIGLU, size);
                benchmark(ctx, Activation.GEGLU, size);
                benchmark(ctx, Activation.GELU, size);
            }
        } finally {
            ctx.close();
        }
    }

    private enum Activation {
        SWIGLU, GEGLU, GELU
    }

    // =====================================================================
    // ===================== CORRECTNESS ==================================
    // =====================================================================

    private static boolean runCorrectness(GpuContext ctx, Activation act) throws Throwable {
        boolean allPassed = true;
        Random rnd = new Random(42);
        for (int len : CORRECTNESS_SIZES) {
            float[] gate = randomActivations(rnd, len);
            float[] up = (act != Activation.GELU) ? randomActivations(rnd, len) : null;

            float[] cpuOut = cpuReference(act, gate, up);
            float[] gpuOut = gpuReference(ctx, act, gate, up);

            float maxAbsDiff = 0f;
            int worstIdx = -1;
            for (int i = 0; i < len; i++) {
                float diff = Math.abs(cpuOut[i] - gpuOut[i]);
                if (diff > maxAbsDiff) {
                    maxAbsDiff = diff;
                    worstIdx = i;
                }
            }
            boolean pass = maxAbsDiff <= TOLERANCE_ABS;
            allPassed &= pass;
            System.out.printf(Locale.ROOT, "  %-8s len=%-6d maxAbsDiff=%-12.6g %s%s%n",
                    act, len, maxAbsDiff, pass ? "PASS" : "FAIL",
                    pass ? "" : String.format(Locale.ROOT, "  (worst at i=%d: cpu=%.6f gpu=%.6f)", worstIdx, cpuOut[worstIdx], gpuOut[worstIdx]));
        }
        return allPassed;
    }

    private static float[] randomActivations(Random rnd, int len) {
        float[] out = new float[len];
        for (int i = 0; i < len; i++) {
            out[i] = (float) (rnd.nextGaussian() * 3.0);
        }
        return out;
    }

    private static float[] gpuReference(GpuContext ctx, Activation act, float[] gate, float[] up) throws Throwable {
        int len = gate.length;
        long gateDevice = LlamaLayer.uploadFloats(ctx, gate);
        long upDevice = (up != null) ? LlamaLayer.uploadFloats(ctx, up) : 0L;
        long outDevice = LlamaLayer.allocFloats(ctx, len);
        try {
            dispatch(ctx, act, gateDevice, upDevice, outDevice, len);
            LlamaLayer.finish(ctx);
            return LlamaLayer.downloadFloats(ctx, outDevice, len);
        } finally {
            LlamaLayer.freeQuietly(ctx, gateDevice);
            if (upDevice != 0L) {
                LlamaLayer.freeQuietly(ctx, upDevice);
            }
            LlamaLayer.freeQuietly(ctx, outDevice);
        }
    }

    private static void dispatch(GpuContext ctx, Activation act, long gateDevice, long upDevice, long outDevice, int len) throws Throwable {
        switch (act) {
            case SWIGLU -> LlamaLayer.swigluActivate(ctx, gateDevice, upDevice, outDevice, len);
            case GEGLU -> LlamaLayer.gegluActivate(ctx, gateDevice, upDevice, outDevice, len);
            case GELU -> LlamaLayer.geluActivate(ctx, gateDevice, outDevice, len);
        }
    }

    // =====================================================================
    // ===================== CPU REFERENCE ================================
    // =====================================================================

    private static float[] cpuReference(Activation act, float[] gate, float[] up) {
        int len = gate.length;
        float[] out = new float[len];
        switch (act) {
            case SWIGLU -> {
                for (int i = 0; i < len; i++) {
                    out[i] = gate[i] * cpuSigmoidClamped(gate[i]) * up[i];
                }
            }
            case GEGLU -> {
                for (int i = 0; i < len; i++) {
                    out[i] = cpuGelu(gate[i]) * up[i];
                }
            }
            case GELU -> {
                for (int i = 0; i < len; i++) {
                    out[i] = cpuGelu(gate[i]);
                }
            }
        }
        return out;
    }

    private static float cpuSigmoidClamped(float g) {
        float gClamped = Math.max(g, -88.0f);
        return 1.0f / (1.0f + (float) Math.exp(-gClamped));
    }

    private static float cpuGelu(float x) {
        return (float) (0.5 * x * (1.0 + erf(x * 0.70710678118654752440)));
    }

    private static double erf(double x) {
        double sign = (x < 0) ? -1.0 : 1.0;
        x = Math.abs(x);
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
                a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return sign * y;
    }

    // =====================================================================
    // ===================== BENCHMARK =====================================
    // =====================================================================

    private static void benchmark(GpuContext ctx, Activation act, int len) throws Throwable {
        Random rnd = new Random(7);
        float[] gate = randomActivations(rnd, len);
        float[] up = (act != Activation.GELU) ? randomActivations(rnd, len) : null;

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cpuReference(act, gate, up);
        }
        long cpuStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            cpuReference(act, gate, up);
        }
        double cpuMs = (System.nanoTime() - cpuStart) / 1e6 / TIMED_ITERS;

        long gateDevice = LlamaLayer.uploadFloats(ctx, gate);
        long upDevice = (up != null) ? LlamaLayer.uploadFloats(ctx, up) : 0L;
        long outDevice = LlamaLayer.allocFloats(ctx, len);
        double gpuKernelMs;
        try {
            for (int i = 0; i < WARMUP_ITERS; i++) {
                dispatch(ctx, act, gateDevice, upDevice, outDevice, len);
            }
            LlamaLayer.finish(ctx);

            long gpuStart = System.nanoTime();
            for (int i = 0; i < TIMED_ITERS; i++) {
                dispatch(ctx, act, gateDevice, upDevice, outDevice, len);
            }
            LlamaLayer.finish(ctx);
            gpuKernelMs = (System.nanoTime() - gpuStart) / 1e6 / TIMED_ITERS;
        } finally {
            LlamaLayer.freeQuietly(ctx, gateDevice);
            if (upDevice != 0L) {
                LlamaLayer.freeQuietly(ctx, upDevice);
            }
            LlamaLayer.freeQuietly(ctx, outDevice);
        }

        for (int i = 0; i < WARMUP_ITERS; i++) {
            gpuReference(ctx, act, gate, up);
        }
        long rtStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            gpuReference(ctx, act, gate, up);
        }
        double gpuRoundTripMs = (System.nanoTime() - rtStart) / 1e6 / TIMED_ITERS;

        double speedupKernel = cpuMs / gpuKernelMs;
        double speedupRoundTrip = cpuMs / gpuRoundTripMs;

        System.out.printf(Locale.ROOT, "%-8s %8d %14.4f %16.4f %16.4f %9.2fx %9.2fx%n",
                act, len, cpuMs, gpuKernelMs, gpuRoundTripMs, speedupKernel, speedupRoundTrip);
    }
}