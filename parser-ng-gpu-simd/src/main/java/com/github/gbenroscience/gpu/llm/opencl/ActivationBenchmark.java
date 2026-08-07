/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.gpu.llm.opencl;

import java.lang.foreign.MemorySegment;
import java.util.Locale;
import java.util.Random;

/**
 * Standalone correctness + benchmark harness for the three FFN activation
 * kernels (KERNEL_SWIGLU_ACTIVATE, KERNEL_GEGLU_ACTIVATE,
 * KERNEL_GELU_ACTIVATE) -- exercised directly, without needing a real GGUF
 * model, a tokenizer, or the rest of the decoder pipeline.
 *
 * WHY THIS EXISTS AS ITS OWN CLASS rather than folded into OpenCLDemo:
 * activations are the cheapest, most isolated kernels in this codebase
 * (no weights, no KV cache, no multi-kernel dependency chain) -- ideal
 * for a first "does my OpenCL bootstrap actually work end to end" smoke
 * test before spending an hour debugging GEMV/attention against a real
 * model file. Run this FIRST on a new device/driver combination.
 *
 * WHAT "CORRECT" MEANS HERE: bit-for-bit equality is not the bar -- the
 * GPU kernels use OpenCL's built-in {@code erf()}, which is whatever the
 * vendor ICD ships (every conformant implementation is accurate to a few
 * ULPs, but not necessarily the SAME few ULPs as this class's own
 * Abramowitz-Stegun host approximation). TOLERANCE_ABS below is set loose
 * enough to absorb that without also being loose enough to hide an actual
 * bug (wrong offset, wrong activation formula, uninitialized memory).
 *
 * WHAT THE BENCHMARK MEASURES, AND WHY THREE NUMBERS PER SIZE, NOT ONE:
 * activations in the real decode/prefill path never round-trip through
 * the host between kernels -- gate/up are produced by a GEMV/GEMM already
 * sitting in device memory, consumed by swiglu_activate, and the result
 * is consumed by the next GEMV, still on the device. So "GPU kernel-only"
 * time (data already resident, clFinish brackets just the kernel) is the
 * number that predicts real inference throughput. "GPU round-trip" (host
 * array -> device -> kernel -> host array) is reported too because it's
 * the honest cost of calling an activation kernel in ISOLATION the way
 * this benchmark itself does -- if you only look at round-trip you will
 * see the GPU lose badly at small sizes to PCIe/USB transfer latency
 * that a real forward pass never pays.
 *
 * Run with (adjust classpath):
 *   java --enable-preview -cp &lt;your-classpath&gt; com.github.gbenroscience.gpu.opencl.llm.ActivationBenchmark
 */
public final class ActivationBenchmark {

    private ActivationBenchmark() {
    }

    /** Absolute-difference tolerance for GPU-vs-CPU agreement -- see class javadoc for why this isn't (and shouldn't be) zero. */
    private static final float TOLERANCE_ABS = 2e-3f;

    /**
     * Sizes exercising: a single element, a partial warp/wavefront, exactly
     * one work-group (DEFAULT_BLOCK_SIZE=256), one element over a work-group
     * boundary (this is the important one -- it's the only thing that
     * actually tests launch1D's "round global size up, bound-check inside
     * the kernel" logic; every size below 256 or exactly 256 would pass
     * even if that bound check were missing or broken), and two realistic
     * Llama-family FFN hidden_dim values.
     */
    private static final int[] CORRECTNESS_SIZES = {1, 7, 32, 255, 256, 257, 1024, 11008};

    /** Representative FFN hidden_dim sizes for benchmarking -- 11008 (Llama-2 7B), 14336 (Llama-3 8B / Mistral 7B). */
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
                        + "or any real model output from this OpenCL port.");
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
            // Gaussian*3 -> mostly [-9,9], occasionally further out; the
            // realistic magnitude range for a post-GEMV pre-activation
            // value, and wide enough to exercise sigmoid/erf's tails.
            out[i] = (float) (rnd.nextGaussian() * 3.0);
        }
        return out;
    }

    private static float[] gpuReference(GpuContext ctx, Activation act, float[] gate, float[] up) throws Throwable {
        int len = gate.length;
        MemorySegment gateDevice = LlamaLayer.uploadFloats(ctx, gate);
        MemorySegment upDevice = (up != null) ? LlamaLayer.uploadFloats(ctx, up) : null;
        MemorySegment outDevice = LlamaLayer.allocFloats(ctx, len);
        try {
            dispatch(ctx, act, gateDevice, upDevice, outDevice, len);
            LlamaLayer.finish(ctx);
            return LlamaLayer.downloadFloats(ctx, outDevice, len);
        } finally {
            LlamaLayer.freeQuietly(ctx, gateDevice);
            if (upDevice != null) {
                LlamaLayer.freeQuietly(ctx, upDevice);
            }
            LlamaLayer.freeQuietly(ctx, outDevice);
        }
    }

    private static void dispatch(GpuContext ctx, Activation act, MemorySegment gateDevice, MemorySegment upDevice,
            MemorySegment outDevice, int len) throws Throwable {
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

    /** Mirrors swiglu_activate's exact clamp: fmax(g, -88) before exp(-g), to avoid exp() overflow for very negative gate values. */
    private static float cpuSigmoidClamped(float g) {
        float gClamped = Math.max(g, -88.0f);
        return 1.0f / (1.0f + (float) Math.exp(-gClamped));
    }

    /** Exact-erf GeLU, matching gpu_gelu_f's formula in KernelSource. */
    private static float cpuGelu(float x) {
        return (float) (0.5 * x * (1.0 + erf(x * 0.70710678118654752440)));
    }

    /** Abramowitz & Stegun 7.1.26 -- max absolute error ~1.5e-7, comfortably below TOLERANCE_ABS and below float32 precision at these magnitudes. */
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

        // ---- CPU: single-threaded, warm the JIT first ----
        for (int i = 0; i < WARMUP_ITERS; i++) {
            cpuReference(act, gate, up);
        }
        long cpuStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            cpuReference(act, gate, up);
        }
        double cpuMs = (System.nanoTime() - cpuStart) / 1e6 / TIMED_ITERS;

        // ---- GPU kernel-only: data uploaded ONCE, only the kernel+clFinish is timed ----
        MemorySegment gateDevice = LlamaLayer.uploadFloats(ctx, gate);
        MemorySegment upDevice = (up != null) ? LlamaLayer.uploadFloats(ctx, up) : null;
        MemorySegment outDevice = LlamaLayer.allocFloats(ctx, len);
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
            LlamaLayer.finish(ctx); // one finish after the whole batch -- see below for why
            gpuKernelMs = (System.nanoTime() - gpuStart) / 1e6 / TIMED_ITERS;
        } finally {
            LlamaLayer.freeQuietly(ctx, gateDevice);
            if (upDevice != null) {
                LlamaLayer.freeQuietly(ctx, upDevice);
            }
            LlamaLayer.freeQuietly(ctx, outDevice);
        }

        // ---- GPU round-trip: upload + kernel + download, every iteration ----
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