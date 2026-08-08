package com.github.gbenroscience.gpu.llm.opencl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Locale;
import java.util.Random;

/**
 * Correctness + benchmark harness for the kernels that actually dominate
 * per-token latency: Q8_0/F32 GEMV (decode path), Q8_0/F32 GEMM (batched
 * prefill path), RMSNorm, RoPE, and the three-kernel attention pipeline
 * (scores -&gt; softmax -&gt; weighted sum), both single-token and batched-causal.
 * Same spirit as ActivationBenchmark -- run correctness first, refuse to
 * trust benchmark numbers if anything fails -- but these kernels are
 * where an offset bug or a quantization mistake would actually corrupt
 * model output, so treat any FAIL here as a stop-everything signal.
 *
 * WHY Q8_0 CORRECTNESS IS TESTED AGAINST "THE SAME QUANTIZED BYTES", NOT
 * AGAINST UNQUANTIZED GROUND TRUTH: for the GEMV/GEMM kernels (which
 * consume ALREADY-quantized x and W and do integer dot products + a
 * float scale multiply), this harness quantizes x and W on the HOST once,
 * uploads those exact bytes, and compares the GPU's output against a CPU
 * reference that decodes and dot-products those SAME bytes the same way
 * the kernel does. That isolates "does the kernel's dot-product/offset/
 * scale-decode arithmetic match its own quantized inputs" from "how much
 * error does Q8_0 quantization itself introduce" -- the latter is a
 * property of the quantization SCHEME, not of this GPU port, and mixing
 * the two into one tolerance would hide a real kernel bug behind
 * quantization noise. The quantize_activation_q8_0_blocks KERNEL itself
 * (the thing that produces x's quantized bytes at runtime) gets its own
 * separate correctness check below, against a host quantizer using the
 * identical block algorithm.
 *
 * DIMENSIONS: chosen to match Llama-2-7B (dim=4096, hidden_dim=11008,
 * 32 query heads, 32 KV heads i.e. plain MHA not GQA, head_dim=128) so
 * the benchmark numbers map onto a model you can actually go run. Swap
 * DIM/HIDDEN/HEADS/HEAD_DIM at the top if you're targeting a different
 * model size.
 *
 * Run with (adjust classpath):
 *   java --enable-preview -cp &lt;your-classpath&gt; com.github.gbenroscience.gpu.llm.opencl.CoreKernelBenchmark
 */
public final class CoreKernelBenchmark {

    private CoreKernelBenchmark() {
    }

    // ---- Llama-2-7B-shaped dimensions ----
    private static final int DIM = 4096;
    private static final int HIDDEN = 11008;
    private static final int NUM_HEADS = 32;
    private static final int HEAD_DIM = 128; // DIM / NUM_HEADS
    private static final int VOCAB = 32000;

    private static final int WARMUP_ITERS = 10;
    private static final int TIMED_ITERS = 30;

    /** Q8_0 dot-product/GEMV/GEMM tolerance -- see class javadoc for why this compares against a same-bytes CPU reference rather than unquantized ground truth. Loose enough for float32 accumulation-order noise across up to ~1000 blocks, tight enough to catch a real offset/formula bug. */
    private static final float TOLERANCE_REL = 1e-2f;
    private static final float TOLERANCE_ABS = 1e-2f;

    public static void main(String[] args) throws Throwable {
        GpuContext ctx = new GpuContext();
        try {
            // Prime every kernel this file touches before anything is
            // measured or checked -- see the note ActivationBenchmark's
            // results turned up about first-dispatch JIT/clock-ramp skew.
            prime(ctx);

            System.out.println("=== Correctness ===");
            boolean ok = true;
            ok &= testQuantizeKernel(ctx);
            ok &= testQ8_0GemvPlain(ctx);
            ok &= testQ8_0GemvSplit(ctx);
            ok &= testF32Gemv(ctx);
            ok &= testRmsNorm(ctx);
            ok &= testRopeSplit(ctx);
            ok &= testAttentionDecode(ctx);
            ok &= testQ8_0GemmTiled(ctx);
            ok &= testF32GemmTiled(ctx);
            ok &= testRmsNormRows(ctx);
            ok &= testRopePairwiseRows(ctx);
            ok &= testAttentionCausalBatched(ctx);

            if (!ok) {
                System.err.println();
                System.err.println("One or more correctness checks FAILED -- do not trust benchmark numbers "
                        + "or any real model output from this OpenCL port until this is fixed.");
                System.exit(1);
            }

            System.out.println();
            System.out.println("=== Benchmark: decode-path GEMV (single token) ===");
            System.out.printf(Locale.ROOT, "%-28s %10s %10s %14s %16s %10s%n",
                    "op", "N", "K", "CPU(ms)", "GPU-kernel(ms)", "speedup");
            benchQ8_0Gemv("FFN gate/up (Q8_0 GEMV)", ctx, HIDDEN, DIM);
            benchQ8_0Gemv("FFN down (Q8_0 GEMV)", ctx, DIM, HIDDEN);
            benchF32Gemv("attn output proj (F32 GEMV)", ctx, DIM, DIM);
            benchF32Gemv("LM head (F32 GEMV)", ctx, DIM, VOCAB);
            benchQ8_0GemvSplit(ctx);

            System.out.println();
            System.out.println("=== Benchmark: decode-path attention (scores+softmax+weightedsum), scaling with context length ===");
            System.out.printf(Locale.ROOT, "%-10s %14s %16s %10s%n", "posIncl", "CPU(ms)", "GPU-kernel(ms)", "speedup");
            for (int posInclusive : new int[]{128, 512, 2048, 4096}) {
                benchAttentionDecode(ctx, posInclusive);
            }

            System.out.println();
            System.out.println("=== Benchmark: batched prefill GEMM, scaling with batch size T ===");
            System.out.printf(Locale.ROOT, "%-28s %6s %10s %10s %14s %16s %10s%n",
                    "op", "T", "N", "K", "CPU(ms)", "GPU-kernel(ms)", "speedup");
            for (int T : new int[]{8, 32, 128}) {
                benchQ8_0Gemm("FFN gate/up (Q8_0 GEMM)", ctx, T, HIDDEN, DIM);
                benchF32Gemm("attn output proj (F32 GEMM)", ctx, T, DIM, DIM);
            }

            System.out.println();
            System.out.println("=== Benchmark: batched-causal attention, scaling with batch size T ===");
            System.out.printf(Locale.ROOT, "%-10s %14s %16s %10s%n", "T", "CPU(ms)", "GPU-kernel(ms)", "speedup");
            for (int T : new int[]{8, 32, 128}) {
                benchAttentionCausalBatched(ctx, T);
            }
        } finally {
            ctx.close();
        }
    }

    /** Dispatches every kernel this file exercises a couple hundred times before any timing/correctness begins, so the first REAL measurement doesn't inherit driver JIT and GPU clock-ramp cost the way ActivationBenchmark's SWIGLU row did. */
    private static void prime(GpuContext ctx) throws Throwable {
        Random rnd = new Random(1);
        float[] x = randomFloats(rnd, DIM, 3.0);
        byte[] xQ8 = quantizeRowQ8_0(x);
        float[] w = randomFloats(rnd, DIM, 1.0);
        byte[] wQ8 = quantizeRowQ8_0(w);

        MemorySegment xDev = LlamaLayer.uploadFloats(ctx, x);
        MemorySegment xQ8Dev = LlamaLayer.uploadBytes(ctx, xQ8);
        MemorySegment wQ8Dev = LlamaLayer.uploadBytes(ctx, wQ8);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, DIM);
        MemorySegment partials = LlamaLayer.allocFloats(ctx, (DIM + 255) / 256);
        try {
            for (int i = 0; i < 200; i++) {
                LlamaLayer.quantizeActivationQ8_0(ctx, xDev, xQ8Dev, DIM);
                LlamaLayer.q8_0GemvPlain(ctx, xQ8Dev, wQ8Dev, outDev, 1, DIM);
                LlamaLayer.f32Gemv(ctx, xDev, xDev, outDev, DIM, 1);
                LlamaLayer.rmsNorm(ctx, xDev, xDev, outDev, DIM, 1e-6, partials);
            }
            LlamaLayer.finish(ctx);
        } finally {
            LlamaLayer.freeQuietly(ctx, xDev);
            LlamaLayer.freeQuietly(ctx, xQ8Dev);
            LlamaLayer.freeQuietly(ctx, wQ8Dev);
            LlamaLayer.freeQuietly(ctx, outDev);
            LlamaLayer.freeQuietly(ctx, partials);
        }
    }

    // =====================================================================
    // ===================== CORRECTNESS ==================================
    // =====================================================================

    private static boolean testQuantizeKernel(GpuContext ctx) throws Throwable {
        Random rnd = new Random(11);
        int len = 4096;
        float[] x = randomFloats(rnd, len, 4.0);

        MemorySegment xDev = LlamaLayer.uploadFloats(ctx, x);
        MemorySegment outQ8Dev = LlamaLayer.allocBytes(ctx, q8_0Bytes(len));
        byte[] gpuQ8;
        try {
            LlamaLayer.quantizeActivationQ8_0(ctx, xDev, outQ8Dev, len);
            LlamaLayer.finish(ctx);
            gpuQ8 = downloadBytes(ctx, outQ8Dev, q8_0Bytes(len));
        } finally {
            LlamaLayer.freeQuietly(ctx, xDev);
            LlamaLayer.freeQuietly(ctx, outQ8Dev);
        }

        byte[] cpuQ8 = quantizeRowQ8_0(x);

        // Compare DECODED values rather than raw bytes -- the fp16 scale
        // encoding can legitimately round the very last bit differently
        // between two independently-written encoders (see
        // KernelSource.encode_fp16_bits' round-half-up note) without that
        // meaning either quantization is wrong; decoding both back to
        // float and comparing sidesteps that non-issue entirely.
        float[] gpuDecoded = dequantizeQ8_0(gpuQ8, len);
        float[] cpuDecoded = dequantizeQ8_0(cpuQ8, len);
        float maxDiff = maxAbsDiff(cpuDecoded, gpuDecoded);
        boolean pass = maxDiff <= TOLERANCE_ABS;
        report("quantize_activation_q8_0_blocks", "len=" + len, maxDiff, pass);
        return pass;
    }

    private static boolean testQ8_0GemvPlain(GpuContext ctx) throws Throwable {
        boolean allPassed = true;
        for (int[] shape : new int[][]{{HIDDEN, DIM}, {DIM, HIDDEN}}) {
            int N = shape[0], K = shape[1];
            Random rnd = new Random(21);
            byte[] xQ8 = quantizeRowQ8_0(randomFloats(rnd, K, 2.0));
            byte[] wQ8 = quantizeMatrixQ8_0(rnd, N, K, 1.0);

            float[] gpuOut = gpuQ8_0GemvPlain(ctx, xQ8, wQ8, N, K);
            float[] cpuOut = cpuQ8_0GemvPlain(xQ8, wQ8, N, K);

            float maxDiff = maxRelevantDiff(cpuOut, gpuOut);
            boolean pass = maxDiff <= TOLERANCE_ABS;
            allPassed &= pass;
            report("q8_0_gemv_plain", "N=" + N + " K=" + K, maxDiff, pass);
        }
        return allPassed;
    }

    private static boolean testQ8_0GemvSplit(GpuContext ctx) throws Throwable {
        int heads = NUM_HEADS, headDim = HEAD_DIM, K = DIM;
        int N = heads * headDim;
        Random rnd = new Random(31);
        byte[] xQ8 = quantizeRowQ8_0(randomFloats(rnd, K, 2.0));
        byte[] wQ8 = quantizeMatrixQ8_0(rnd, N, K, 1.0);

        float[] gpuOut = gpuQ8_0GemvSplit(ctx, xQ8, wQ8, heads, headDim, K);
        float[] cpuOut = cpuQ8_0GemvSplit(xQ8, wQ8, heads, headDim, K);

        float maxDiff = maxRelevantDiff(cpuOut, gpuOut);
        boolean pass = maxDiff <= TOLERANCE_ABS;
        report("q8_0_gemv_split", "heads=" + heads + " headDim=" + headDim + " K=" + K, maxDiff, pass);
        return pass;
    }

    private static boolean testF32Gemv(GpuContext ctx) throws Throwable {
        int K = DIM, N = DIM;
        Random rnd = new Random(41);
        float[] a = randomFloats(rnd, K, 1.0);
        float[] B = randomFloats(rnd, (long) K * N, 0.5);

        float[] gpuOut = gpuF32Gemv(ctx, a, B, K, N);
        float[] cpuOut = cpuF32Gemv(a, B, K, N);

        float maxDiff = maxRelevantDiff(cpuOut, gpuOut);
        boolean pass = maxDiff <= TOLERANCE_ABS;
        report("f32_gemv", "K=" + K + " N=" + N, maxDiff, pass);
        return pass;
    }

    private static boolean testRmsNorm(GpuContext ctx) throws Throwable {
        boolean allPassed = true;
        for (int len : new int[]{32, 255, 256, 257, DIM}) {
            Random rnd = new Random(51);
            float[] x = randomFloats(rnd, len, 3.0);
            float[] gamma = randomFloats(rnd, len, 1.0, 0.5); // gamma centered near 1.0, like real RMSNorm weights

            MemorySegment xDev = LlamaLayer.uploadFloats(ctx, x);
            MemorySegment gammaDev = LlamaLayer.uploadFloats(ctx, gamma);
            MemorySegment outDev = LlamaLayer.allocFloats(ctx, len);
            MemorySegment partials = LlamaLayer.allocFloats(ctx, (len + 255) / 256);
            float[] gpuOut;
            try {
                LlamaLayer.rmsNorm(ctx, xDev, gammaDev, outDev, len, 1e-6, partials);
                LlamaLayer.finish(ctx);
                gpuOut = LlamaLayer.downloadFloats(ctx, outDev, len);
            } finally {
                LlamaLayer.freeQuietly(ctx, xDev);
                LlamaLayer.freeQuietly(ctx, gammaDev);
                LlamaLayer.freeQuietly(ctx, outDev);
                LlamaLayer.freeQuietly(ctx, partials);
            }

            float[] cpuOut = cpuRmsNorm(x, gamma, 1e-6);
            float maxDiff = maxAbsDiff(cpuOut, gpuOut);
            boolean pass = maxDiff <= TOLERANCE_ABS;
            allPassed &= pass;
            report("rmsnorm (partial_sumsq+apply)", "len=" + len, maxDiff, pass);
        }
        return allPassed;
    }

    private static boolean testRopeSplit(GpuContext ctx) throws Throwable {
        int heads = NUM_HEADS, headDim = HEAD_DIM, halfDim = headDim / 2;
        Random rnd = new Random(61);
        float[] buf = randomFloats(rnd, (long) heads * headDim, 2.0);
        int pos = 17;
        float[] cosVals = new float[halfDim];
        float[] sinVals = new float[halfDim];
        for (int i = 0; i < halfDim; i++) {
            double freq = 1.0 / Math.pow(10000.0, (2.0 * i) / headDim);
            double angle = pos * freq;
            cosVals[i] = (float) Math.cos(angle);
            sinVals[i] = (float) Math.sin(angle);
        }

        MemorySegment bufDev = LlamaLayer.uploadFloats(ctx, buf);
        MemorySegment cosDev = LlamaLayer.uploadFloats(ctx, cosVals);
        MemorySegment sinDev = LlamaLayer.uploadFloats(ctx, sinVals);
        float[] gpuOut;
        try {
            LlamaLayer.ropeApplySplit(ctx, bufDev, cosDev, sinDev, heads, headDim, 0);
            LlamaLayer.finish(ctx);
            gpuOut = LlamaLayer.downloadFloats(ctx, bufDev, heads * headDim);
        } finally {
            LlamaLayer.freeQuietly(ctx, bufDev);
            LlamaLayer.freeQuietly(ctx, cosDev);
            LlamaLayer.freeQuietly(ctx, sinDev);
        }

        float[] cpuOut = cpuRopeSplit(buf, cosVals, sinVals, heads, headDim);
        float maxDiff = maxAbsDiff(cpuOut, gpuOut);
        boolean pass = maxDiff <= TOLERANCE_ABS;
        report("rope_apply_split", "heads=" + heads + " headDim=" + headDim + " pos=" + pos, maxDiff, pass);
        return pass;
    }

    private static boolean testAttentionDecode(GpuContext ctx) throws Throwable {
        boolean allPassed = true;
        for (int posInclusive : new int[]{1, 33, 256, 513}) {
            Random rnd = new Random(71);
            float[] q = randomFloats(rnd, HEAD_DIM, 1.0);
            float[] kCache = randomFloats(rnd, (long) posInclusive * HEAD_DIM, 1.0);
            float[] vCache = randomFloats(rnd, (long) posInclusive * HEAD_DIM, 1.0);

            float[] gpuOut = gpuAttentionDecode(ctx, q, kCache, vCache, posInclusive, HEAD_DIM);
            float[] cpuOut = cpuAttentionDecode(q, kCache, vCache, posInclusive, HEAD_DIM);

            float maxDiff = maxAbsDiff(cpuOut, gpuOut);
            boolean pass = maxDiff <= TOLERANCE_ABS;
            allPassed &= pass;
            report("attn_scores+softmax+weighted_sum", "posInclusive=" + posInclusive, maxDiff, pass);
        }
        return allPassed;
    }

    private static boolean testQ8_0GemmTiled(GpuContext ctx) throws Throwable {
        int T = 17, N = 300, K = DIM; // deliberately not multiples of GEMM_TILE_N/RMSNORM_WORKGROUP_SIZE
        Random rnd = new Random(81);
        byte[] xQ8Batch = quantizeMatrixQ8_0(rnd, T, K, 2.0);
        byte[] wQ8 = quantizeMatrixQ8_0(rnd, N, K, 1.0);

        MemorySegment xDev = LlamaLayer.uploadBytes(ctx, xQ8Batch);
        MemorySegment wDev = LlamaLayer.uploadBytes(ctx, wQ8);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, (long) T * N);
        float[] gpuOut;
        try {
            LlamaLayer.q8_0GemmTiled(ctx, xDev, wDev, outDev, T, N, K);
            LlamaLayer.finish(ctx);
            gpuOut = LlamaLayer.downloadFloats(ctx, outDev, T * N);
        } finally {
            LlamaLayer.freeQuietly(ctx, xDev);
            LlamaLayer.freeQuietly(ctx, wDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }

        float[] cpuOut = cpuQ8_0GemmTiled(xQ8Batch, wQ8, T, N, K);
        float maxDiff = maxRelevantDiff(cpuOut, gpuOut);
        boolean pass = maxDiff <= TOLERANCE_ABS;
        report("q8_0_gemm_tiled", "T=" + T + " N=" + N + " K=" + K, maxDiff, pass);
        return pass;
    }

    private static boolean testF32GemmTiled(GpuContext ctx) throws Throwable {
        int T = 17, K = DIM, N = 300;
        Random rnd = new Random(91);
        float[] A = randomFloats(rnd, (long) T * K, 1.0);
        float[] B = randomFloats(rnd, (long) K * N, 0.5);

        MemorySegment aDev = LlamaLayer.uploadFloats(ctx, A);
        MemorySegment bDev = LlamaLayer.uploadFloats(ctx, B);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, (long) T * N);
        float[] gpuOut;
        try {
            LlamaLayer.f32GemmTiled(ctx, aDev, bDev, outDev, T, K, N);
            LlamaLayer.finish(ctx);
            gpuOut = LlamaLayer.downloadFloats(ctx, outDev, T * N);
        } finally {
            LlamaLayer.freeQuietly(ctx, aDev);
            LlamaLayer.freeQuietly(ctx, bDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }

        float[] cpuOut = cpuF32GemmTiled(A, B, T, K, N);
        float maxDiff = maxRelevantDiff(cpuOut, gpuOut);
        boolean pass = maxDiff <= TOLERANCE_ABS;
        report("f32_gemm_tiled", "T=" + T + " K=" + K + " N=" + N, maxDiff, pass);
        return pass;
    }

    private static boolean testRmsNormRows(GpuContext ctx) throws Throwable {
        int T = 13, features = DIM;
        Random rnd = new Random(101);
        float[] x = randomFloats(rnd, (long) T * features, 3.0);
        float[] gamma = randomFloats(rnd, features, 1.0, 0.5);

        int wgSize = KernelSource.RMSNORM_WORKGROUP_SIZE;
        int numGroups = (features + wgSize - 1) / wgSize;

        MemorySegment xDev = LlamaLayer.uploadFloats(ctx, x);
        MemorySegment gammaDev = LlamaLayer.uploadFloats(ctx, gamma);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, (long) T * features);
        MemorySegment partialsBatch = LlamaLayer.allocFloats(ctx, (long) T * numGroups);
        MemorySegment rmsRowValues = LlamaLayer.allocFloats(ctx, T);
        float[] gpuOut;
        try {
            LlamaLayer.rmsNormRows(ctx, xDev, gammaDev, outDev, features, 1e-6, T, partialsBatch, rmsRowValues);
            LlamaLayer.finish(ctx);
            gpuOut = LlamaLayer.downloadFloats(ctx, outDev, T * features);
        } finally {
            LlamaLayer.freeQuietly(ctx, xDev);
            LlamaLayer.freeQuietly(ctx, gammaDev);
            LlamaLayer.freeQuietly(ctx, outDev);
            LlamaLayer.freeQuietly(ctx, partialsBatch);
            LlamaLayer.freeQuietly(ctx, rmsRowValues);
        }

        float[] cpuOut = new float[T * features];
        for (int row = 0; row < T; row++) {
            float[] rowIn = new float[features];
            System.arraycopy(x, row * features, rowIn, 0, features);
            float[] rowOut = cpuRmsNorm(rowIn, gamma, 1e-6);
            System.arraycopy(rowOut, 0, cpuOut, row * features, features);
        }

        float maxDiff = maxAbsDiff(cpuOut, gpuOut);
        boolean pass = maxDiff <= TOLERANCE_ABS;
        report("rmsnorm_partial_sumsq_rows+apply_rows", "T=" + T + " features=" + features, maxDiff, pass);
        return pass;
    }

    private static boolean testRopePairwiseRows(GpuContext ctx) throws Throwable {
        int T = 5, heads = NUM_HEADS, headDim = HEAD_DIM, halfDim = headDim / 2;
        Random rnd = new Random(111);
        float[] buf = randomFloats(rnd, (long) T * heads * headDim, 2.0);
        int[] positions = {3, 4, 5, 6, 7}; // arbitrary, non-zero-based, exercises the position-indexed cos/sin lookup
        int maxSeq = 16;
        float[] cosTable = new float[maxSeq * halfDim];
        float[] sinTable = new float[maxSeq * halfDim];
        for (int p = 0; p < maxSeq; p++) {
            for (int i = 0; i < halfDim; i++) {
                double freq = 1.0 / Math.pow(10000.0, (2.0 * i) / headDim);
                double angle = p * freq;
                cosTable[p * halfDim + i] = (float) Math.cos(angle);
                sinTable[p * halfDim + i] = (float) Math.sin(angle);
            }
        }

        MemorySegment bufDev = LlamaLayer.uploadFloats(ctx, buf);
        MemorySegment cosDev = LlamaLayer.uploadFloats(ctx, cosTable);
        MemorySegment sinDev = LlamaLayer.uploadFloats(ctx, sinTable);
        MemorySegment positionsDev = LlamaLayer.uploadInts(ctx, positions);
        float[] gpuOut;
        try {
            LlamaLayer.ropeApplyPairwiseRows(ctx, bufDev, cosDev, sinDev, heads, headDim, positionsDev, T);
            LlamaLayer.finish(ctx);
            gpuOut = LlamaLayer.downloadFloats(ctx, bufDev, T * heads * headDim);
        } finally {
            LlamaLayer.freeQuietly(ctx, bufDev);
            LlamaLayer.freeQuietly(ctx, cosDev);
            LlamaLayer.freeQuietly(ctx, sinDev);
            LlamaLayer.freeQuietly(ctx, positionsDev);
        }

        float[] cpuOut = cpuRopePairwiseRows(buf, cosTable, sinTable, positions, T, heads, headDim);
        float maxDiff = maxAbsDiff(cpuOut, gpuOut);
        boolean pass = maxDiff <= TOLERANCE_ABS;
        report("rope_apply_pairwise_rows", "T=" + T + " heads=" + heads + " headDim=" + headDim, maxDiff, pass);
        return pass;
    }

    private static boolean testAttentionCausalBatched(GpuContext ctx) throws Throwable {
        int T = 11, headDim = HEAD_DIM; // T deliberately not a power of 2 -- exercises softmax_inplace_rows' nextPow2 sizing
        Random rnd = new Random(121);
        float[] q = randomFloats(rnd, (long) T * headDim, 1.0);
        float[] k = randomFloats(rnd, (long) T * headDim, 1.0);
        float[] v = randomFloats(rnd, (long) T * headDim, 1.0);

        float[] gpuOut = gpuAttentionCausalBatched(ctx, q, k, v, T, headDim);
        float[] cpuOut = cpuAttentionCausalBatched(q, k, v, T, headDim);

        float maxDiff = maxAbsDiff(cpuOut, gpuOut);
        boolean pass = maxDiff <= TOLERANCE_ABS;
        report("attn_scores/softmax/weightedsum_causal_batched", "T=" + T + " headDim=" + headDim, maxDiff, pass);
        return pass;
    }

    private static void report(String kernel, String shape, float maxDiff, boolean pass) {
        System.out.printf(Locale.ROOT, "  %-42s %-30s maxDiff=%-12.6g %s%n", kernel, shape, maxDiff, pass ? "PASS" : "FAIL");
    }

    // =====================================================================
    // ===================== GPU CALL WRAPPERS ============================
    // =====================================================================

    private static float[] gpuQ8_0GemvPlain(GpuContext ctx, byte[] xQ8, byte[] wQ8, int N, int K) throws Throwable {
        MemorySegment xDev = LlamaLayer.uploadBytes(ctx, xQ8);
        MemorySegment wDev = LlamaLayer.uploadBytes(ctx, wQ8);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, N);
        try {
            LlamaLayer.q8_0GemvPlain(ctx, xDev, wDev, outDev, N, K);
            LlamaLayer.finish(ctx);
            return LlamaLayer.downloadFloats(ctx, outDev, N);
        } finally {
            LlamaLayer.freeQuietly(ctx, xDev);
            LlamaLayer.freeQuietly(ctx, wDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }
    }

    private static float[] gpuQ8_0GemvSplit(GpuContext ctx, byte[] xQ8, byte[] wQ8, int heads, int headDim, int K) throws Throwable {
        MemorySegment xDev = LlamaLayer.uploadBytes(ctx, xQ8);
        MemorySegment wDev = LlamaLayer.uploadBytes(ctx, wQ8);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, (long) heads * headDim);
        try {
            LlamaLayer.q8_0GemvSplit(ctx, xDev, wDev, outDev, heads, headDim, K);
            LlamaLayer.finish(ctx);
            return LlamaLayer.downloadFloats(ctx, outDev, heads * headDim);
        } finally {
            LlamaLayer.freeQuietly(ctx, xDev);
            LlamaLayer.freeQuietly(ctx, wDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }
    }

    private static float[] gpuF32Gemv(GpuContext ctx, float[] a, float[] B, int K, int N) throws Throwable {
        MemorySegment aDev = LlamaLayer.uploadFloats(ctx, a);
        MemorySegment bDev = LlamaLayer.uploadFloats(ctx, B);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, N);
        try {
            LlamaLayer.f32Gemv(ctx, aDev, bDev, outDev, K, N);
            LlamaLayer.finish(ctx);
            return LlamaLayer.downloadFloats(ctx, outDev, N);
        } finally {
            LlamaLayer.freeQuietly(ctx, aDev);
            LlamaLayer.freeQuietly(ctx, bDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }
    }

    private static float[] gpuAttentionDecode(GpuContext ctx, float[] q, float[] kCache, float[] vCache, int posInclusive, int headDim) throws Throwable {
        float rsqrtD = (float) (1.0 / Math.sqrt(headDim));
        MemorySegment qDev = LlamaLayer.uploadFloats(ctx, q);
        MemorySegment kDev = LlamaLayer.uploadFloats(ctx, kCache);
        MemorySegment vDev = LlamaLayer.uploadFloats(ctx, vCache);
        MemorySegment scoresDev = LlamaLayer.allocFloats(ctx, posInclusive);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, headDim);
        try {
            LlamaLayer.attnScores(ctx, qDev, kDev, scoresDev, 0, headDim, headDim, 0, posInclusive, rsqrtD);
            LlamaLayer.softmaxInplace(ctx, scoresDev, posInclusive);
            LlamaLayer.attnWeightedSum(ctx, scoresDev, vDev, outDev, 0, headDim, headDim, 0, posInclusive);
            LlamaLayer.finish(ctx);
            return LlamaLayer.downloadFloats(ctx, outDev, headDim);
        } finally {
            LlamaLayer.freeQuietly(ctx, qDev);
            LlamaLayer.freeQuietly(ctx, kDev);
            LlamaLayer.freeQuietly(ctx, vDev);
            LlamaLayer.freeQuietly(ctx, scoresDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }
    }

    private static float[] gpuAttentionCausalBatched(GpuContext ctx, float[] q, float[] k, float[] v, int T, int headDim) throws Throwable {
        float rsqrtD = (float) (1.0 / Math.sqrt(headDim));
        MemorySegment qDev = LlamaLayer.uploadFloats(ctx, q);
        MemorySegment kDev = LlamaLayer.uploadFloats(ctx, k);
        MemorySegment vDev = LlamaLayer.uploadFloats(ctx, v);
        MemorySegment scoresDev = LlamaLayer.allocFloats(ctx, (long) T * T);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, (long) T * headDim);
        try {
            LlamaLayer.attnScoresCausalBatched(ctx, qDev, kDev, scoresDev, headDim, headDim, 0, 0, headDim, T, rsqrtD);
            LlamaLayer.softmaxInplaceRows(ctx, scoresDev, T);
            LlamaLayer.attnWeightedSumCausalBatched(ctx, scoresDev, vDev, outDev, headDim, headDim, 0, 0, headDim, T);
            LlamaLayer.finish(ctx);
            return LlamaLayer.downloadFloats(ctx, outDev, T * headDim);
        } finally {
            LlamaLayer.freeQuietly(ctx, qDev);
            LlamaLayer.freeQuietly(ctx, kDev);
            LlamaLayer.freeQuietly(ctx, vDev);
            LlamaLayer.freeQuietly(ctx, scoresDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }
    }

    // =====================================================================
    // ===================== CPU REFERENCE ================================
    // =====================================================================

    private static float[] cpuQ8_0GemvPlain(byte[] xQ8, byte[] wQ8, int N, int K) {
        int numBlocks = K / 32;
        int rowStride = numBlocks * 34;
        float[] out = new float[N];
        for (int n = 0; n < N; n++) {
            out[n] = q8_0Dot(xQ8, 0, wQ8, n * rowStride, numBlocks);
        }
        return out;
    }

    private static float[] cpuQ8_0GemvSplit(byte[] xQ8, byte[] wQ8, int heads, int headDim, int K) {
        int numBlocks = K / 32;
        int rowStride = numBlocks * 34;
        int halfDim = headDim / 2;
        float[] out = new float[heads * headDim];
        for (int h = 0; h < heads; h++) {
            int evenOff = h * headDim;
            int oddOff = evenOff + halfDim;
            for (int i = 0; i < halfDim; i++) {
                int n0 = h * headDim + 2 * i;
                int n1 = n0 + 1;
                out[evenOff + i] = q8_0Dot(xQ8, 0, wQ8, n0 * rowStride, numBlocks);
                out[oddOff + i] = q8_0Dot(xQ8, 0, wQ8, n1 * rowStride, numBlocks);
            }
        }
        return out;
    }

    private static float[] cpuQ8_0GemmTiled(byte[] xQ8Batch, byte[] wQ8, int T, int N, int K) {
        int numBlocks = K / 32;
        int rowStride = numBlocks * 34;
        float[] out = new float[T * N];
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                out[t * N + n] = q8_0Dot(xQ8Batch, t * rowStride, wQ8, n * rowStride, numBlocks);
            }
        }
        return out;
    }

    /** B is [N, K] row-major -- matches the corrected f32_gemv kernel and GGUF's native Linear-weight layout. */
    private static float[] cpuF32Gemv(float[] a, float[] B, int K, int N) {
        float[] out = new float[N];
        for (int n = 0; n < N; n++) {
            int rowOff = n * K;
            float acc = 0f;
            for (int k = 0; k < K; k++) {
                acc += a[k] * B[rowOff + k];
            }
            out[n] = acc;
        }
        return out;
    }

    /** B is [N, K] row-major -- matches the corrected f32_gemm_tiled kernel. */
    private static float[] cpuF32GemmTiled(float[] A, float[] B, int T, int K, int N) {
        float[] out = new float[T * N];
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                int bRowOff = n * K;
                float acc = 0f;
                for (int k = 0; k < K; k++) {
                    acc += A[t * K + k] * B[bRowOff + k];
                }
                out[t * N + n] = acc;
            }
        }
        return out;
    }

    private static float[] cpuRmsNorm(float[] x, float[] gamma, double eps) {
        int n = x.length;
        double sumSq = 0.0;
        for (float v : x) {
            sumSq += (double) v * v;
        }
        float rms = (float) (1.0 / Math.sqrt(sumSq / n + eps));
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = x[i] * rms * gamma[i];
        }
        return out;
    }

    private static float[] cpuRopeSplit(float[] buf, float[] cosVals, float[] sinVals, int heads, int headDim) {
        int halfDim = headDim / 2;
        float[] out = buf.clone();
        for (int h = 0; h < heads; h++) {
            int headOff = h * headDim;
            for (int i = 0; i < halfDim; i++) {
                float c = cosVals[i], s = sinVals[i];
                float x0 = buf[headOff + i], x1 = buf[headOff + halfDim + i];
                out[headOff + i] = x0 * c - x1 * s;
                out[headOff + halfDim + i] = x0 * s + x1 * c;
            }
        }
        return out;
    }

    /** Different convention from cpuRopeSplit -- ADJACENT pairs (2i,2i+1), not split halves. Mirrors rope_apply_pairwise_rows exactly, see that kernel's javadoc note about the two conventions coexisting in this codebase. */
    private static float[] cpuRopePairwiseRows(float[] buf, float[] cosTable, float[] sinTable, int[] positions, int T, int heads, int headDim) {
        int halfDim = headDim / 2;
        float[] out = buf.clone();
        for (int row = 0; row < T; row++) {
            int pos = positions[row];
            int rowOff = row * heads * headDim;
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < halfDim; i++) {
                    int pairOff = rowOff + h * headDim + 2 * i;
                    float c = cosTable[pos * halfDim + i], s = sinTable[pos * halfDim + i];
                    float x0 = buf[pairOff], x1 = buf[pairOff + 1];
                    out[pairOff] = x0 * c - x1 * s;
                    out[pairOff + 1] = x0 * s + x1 * c;
                }
            }
        }
        return out;
    }

    private static float[] cpuAttentionDecode(float[] q, float[] kCache, float[] vCache, int T, int headDim) {
        float rsqrtD = (float) (1.0 / Math.sqrt(headDim));
        float[] scores = new float[T];
        float maxScore = -Float.MAX_VALUE;
        for (int j = 0; j < T; j++) {
            float dot = 0f;
            for (int d = 0; d < headDim; d++) {
                dot += q[d] * kCache[j * headDim + d];
            }
            scores[j] = dot * rsqrtD;
            maxScore = Math.max(maxScore, scores[j]);
        }
        float sum = 0f;
        for (int j = 0; j < T; j++) {
            scores[j] = (float) Math.exp(scores[j] - maxScore);
            sum += scores[j];
        }
        for (int j = 0; j < T; j++) {
            scores[j] /= sum;
        }
        float[] out = new float[headDim];
        for (int d = 0; d < headDim; d++) {
            float acc = 0f;
            for (int j = 0; j < T; j++) {
                acc += scores[j] * vCache[j * headDim + d];
            }
            out[d] = acc;
        }
        return out;
    }

    private static float[] cpuAttentionCausalBatched(float[] q, float[] k, float[] v, int T, int headDim) {
        float rsqrtD = (float) (1.0 / Math.sqrt(headDim));
        float[] out = new float[T * headDim];
        for (int t = 0; t < T; t++) {
            float[] scores = new float[t + 1];
            float maxScore = -Float.MAX_VALUE;
            for (int j = 0; j <= t; j++) {
                float dot = 0f;
                for (int d = 0; d < headDim; d++) {
                    dot += q[t * headDim + d] * k[j * headDim + d];
                }
                scores[j] = dot * rsqrtD;
                maxScore = Math.max(maxScore, scores[j]);
            }
            float sum = 0f;
            for (int j = 0; j <= t; j++) {
                scores[j] = (float) Math.exp(scores[j] - maxScore);
                sum += scores[j];
            }
            for (int j = 0; j <= t; j++) {
                scores[j] /= sum;
            }
            for (int d = 0; d < headDim; d++) {
                float acc = 0f;
                for (int j = 0; j <= t; j++) {
                    acc += scores[j] * v[j * headDim + d];
                }
                out[t * headDim + d] = acc;
            }
        }
        return out;
    }

    // =====================================================================
    // ===================== Q8_0 HOST CODEC ===============================
    // (mirrors KernelSource's decode_fp16/encode_fp16_bits and the
    //  quantize_activation_q8_0_blocks kernel's block algorithm exactly)
    // =====================================================================

    private static int q8_0Bytes(int len) {
        return (len / 32) * 34;
    }

    private static byte[] quantizeRowQ8_0(float[] row) {
        if (row.length % 32 != 0) {
            throw new IllegalArgumentException("Q8_0 requires a length that's a multiple of 32, got " + row.length);
        }
        int numBlocks = row.length / 32;
        byte[] out = new byte[numBlocks * 34];
        for (int b = 0; b < numBlocks; b++) {
            int off = b * 32;
            float absmax = 0f;
            for (int j = 0; j < 32; j++) {
                absmax = Math.max(absmax, Math.abs(row[off + j]));
            }
            float scale = absmax / 127.0f;
            float invScale = (scale > 0f) ? (1.0f / scale) : 0f;
            int outOff = b * 34;
            short bits = encodeFp16Bits(scale);
            out[outOff] = (byte) (bits & 0xFF);
            out[outOff + 1] = (byte) ((bits >>> 8) & 0xFF);
            for (int j = 0; j < 32; j++) {
                int q = (int) Math.rint(row[off + j] * invScale);
                q = Math.max(-127, Math.min(127, q));
                out[outOff + 2 + j] = (byte) q;
            }
        }
        return out;
    }

    /** Quantizes an N x K row-major float matrix into N independently-quantized Q8_0 rows, concatenated -- exactly the layout real GGUF Q8_0 weight tensors use, and what q8_0_gemv_plain/split/gemm_tiled expect for W. */
    private static byte[] quantizeMatrixQ8_0(Random rnd, int N, int K, double scaleHint) {
        int rowStride = q8_0Bytes(K);
        byte[] out = new byte[N * rowStride];
        for (int n = 0; n < N; n++) {
            float[] row = randomFloats(rnd, K, scaleHint);
            byte[] rowQ8 = quantizeRowQ8_0(row);
            System.arraycopy(rowQ8, 0, out, n * rowStride, rowStride);
        }
        return out;
    }

    private static float[] dequantizeQ8_0(byte[] q8, int len) {
        int numBlocks = len / 32;
        float[] out = new float[len];
        for (int b = 0; b < numBlocks; b++) {
            int blockOff = b * 34;
            float scale = decodeFp16(q8[blockOff], q8[blockOff + 1]);
            for (int j = 0; j < 32; j++) {
                out[b * 32 + j] = q8[blockOff + 2 + j] * scale;
            }
        }
        return out;
    }

    /** One block's contribution: decode both sides' fp16 scale, integer dot-product the 32 int8 values (exact, no rounding), multiply by the combined float scale -- exactly q8_0_gemv_plain's inner loop body, generalized to an arbitrary row offset on each side (used directly for gemv, and with a nonzero xRowOff for the batched gemm). */
    private static float q8_0Dot(byte[] xQ8, int xRowOff, byte[] wQ8, int wRowOff, int numBlocks) {
        float acc = 0f;
        for (int b = 0; b < numBlocks; b++) {
            int xBlockOff = xRowOff + b * 34;
            int wBlockOff = wRowOff + b * 34;
            float xScale = decodeFp16(xQ8[xBlockOff], xQ8[xBlockOff + 1]);
            float wScale = decodeFp16(wQ8[wBlockOff], wQ8[wBlockOff + 1]);
            float scale = xScale * wScale;
            int iacc = 0;
            for (int j = 0; j < 32; j++) {
                int xv = xQ8[xBlockOff + 2 + j]; // Java byte is signed -- matches the kernel's (char) cast directly
                int wv = wQ8[wBlockOff + 2 + j];
                iacc += xv * wv;
            }
            acc += iacc * scale;
        }
        return acc;
    }

    private static short encodeFp16Bits(float value) {
        int x = Float.floatToRawIntBits(value);
        int sign = (x >>> 16) & 0x8000;
        int mantissa = x & 0x007FFFFF;
        int exponent = ((x >>> 23) & 0xFF) - 127;

        if (((x >>> 23) & 0xFF) == 0xFF) {
            return (short) (sign | 0x7C00 | (mantissa != 0 ? 0x200 : 0));
        }
        if (exponent > 15) {
            return (short) (sign | 0x7C00);
        }
        if (exponent < -14) {
            if (exponent < -24) {
                return (short) sign;
            }
            mantissa |= 0x00800000;
            int shift = -14 - exponent;
            int halfMant = mantissa >>> (13 + shift);
            int roundBit = (mantissa >>> (12 + shift)) & 1;
            halfMant += roundBit;
            return (short) (sign | halfMant);
        }
        int halfExp = exponent + 15;
        int halfMant = mantissa >>> 13;
        int roundBit = (mantissa >>> 12) & 1;
        int rounded = halfMant + roundBit;
        if (rounded == 0x400) {
            rounded = 0;
            halfExp += 1;
        }
        return (short) (sign | (halfExp << 10) | rounded);
    }

    private static float decodeFp16(byte lo, byte hi) {
        int bits = (lo & 0xFF) | ((hi & 0xFF) << 8);
        int sign = (bits >>> 15) & 0x1;
        int exp = (bits >>> 10) & 0x1F;
        int mant = bits & 0x3FF;
        int f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign << 31;
            } else {
                int e = 1;
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    e--;
                }
                mant &= 0x3FF;
                int fexp = e - 15 + 127;
                f = (sign << 31) | (fexp << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            f = (sign << 31) | (0xFF << 23) | (mant << 13);
        } else {
            int fexp = exp - 15 + 127;
            f = (sign << 31) | (fexp << 23) | (mant << 13);
        }
        return Float.intBitsToFloat(f);
    }

    // =====================================================================
    // ===================== MISC HELPERS ==================================
    // =====================================================================

    private static float[] randomFloats(Random rnd, long len, double gaussianScale) {
        return randomFloats(rnd, len, gaussianScale, 0.0);
    }

    private static float[] randomFloats(Random rnd, long len, double gaussianScale, double mean) {
        float[] out = new float[(int) len];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) (mean + rnd.nextGaussian() * gaussianScale);
        }
        return out;
    }

    private static byte[] downloadBytes(GpuContext ctx, MemorySegment mem, int count) throws Throwable {
        byte[] out = new byte[count];
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate(Math.max(count, 1));
            GpuContext.check((int) ctx.cl.clEnqueueReadBuffer.invoke(
                    ctx.queue, mem, OpenCLBindings.CL_TRUE, 0L, count, host, 0,
                    MemorySegment.NULL, MemorySegment.NULL), "clEnqueueReadBuffer(bytes)");
            MemorySegment.copy(host, ValueLayout.JAVA_BYTE, 0, out, 0, count);
        }
        return out;
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float max = 0f;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, Math.abs(a[i] - b[i]));
        }
        return max;
    }

    /** Combined absolute/relative tolerance check, reported as a single "effective diff" number: 0 if within tolerance, otherwise the actual diff at the worst offending index. Used for GEMV/GEMM outputs, whose magnitude scales with K/N and so a pure absolute tolerance would be too strict for large reductions and too loose for small ones. */
    private static float maxRelevantDiff(float[] cpu, float[] gpu) {
        float worst = 0f;
        for (int i = 0; i < cpu.length; i++) {
            float diff = Math.abs(cpu[i] - gpu[i]);
            float allowed = Math.max(TOLERANCE_ABS, TOLERANCE_REL * Math.abs(cpu[i]));
            if (diff > allowed) {
                worst = Math.max(worst, diff);
            }
        }
        return worst;
    }

    // =====================================================================
    // ===================== BENCHMARKS ====================================
    // =====================================================================

    private static void benchQ8_0Gemv(String label, GpuContext ctx, int N, int K) throws Throwable {
        Random rnd = new Random(201);
        byte[] xQ8 = quantizeRowQ8_0(randomFloats(rnd, K, 2.0));
        byte[] wQ8 = quantizeMatrixQ8_0(rnd, N, K, 1.0);

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cpuQ8_0GemvPlain(xQ8, wQ8, N, K);
        }
        long cpuStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            cpuQ8_0GemvPlain(xQ8, wQ8, N, K);
        }
        double cpuMs = (System.nanoTime() - cpuStart) / 1e6 / TIMED_ITERS;

        MemorySegment xDev = LlamaLayer.uploadBytes(ctx, xQ8);
        MemorySegment wDev = LlamaLayer.uploadBytes(ctx, wQ8);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, N);
        double gpuMs;
        try {
            for (int i = 0; i < WARMUP_ITERS; i++) {
                LlamaLayer.q8_0GemvPlain(ctx, xDev, wDev, outDev, N, K);
            }
            LlamaLayer.finish(ctx);
            long gpuStart = System.nanoTime();
            for (int i = 0; i < TIMED_ITERS; i++) {
                LlamaLayer.q8_0GemvPlain(ctx, xDev, wDev, outDev, N, K);
            }
            LlamaLayer.finish(ctx);
            gpuMs = (System.nanoTime() - gpuStart) / 1e6 / TIMED_ITERS;
        } finally {
            LlamaLayer.freeQuietly(ctx, xDev);
            LlamaLayer.freeQuietly(ctx, wDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }

        System.out.printf(Locale.ROOT, "%-28s %10d %10d %14.4f %16.4f %9.2fx%n", label, N, K, cpuMs, gpuMs, cpuMs / gpuMs);
    }

    private static void benchQ8_0GemvSplit(GpuContext ctx) throws Throwable {
        int heads = NUM_HEADS, headDim = HEAD_DIM, K = DIM;
        int N = heads * headDim;
        Random rnd = new Random(211);
        byte[] xQ8 = quantizeRowQ8_0(randomFloats(rnd, K, 2.0));
        byte[] wQ8 = quantizeMatrixQ8_0(rnd, N, K, 1.0);

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cpuQ8_0GemvSplit(xQ8, wQ8, heads, headDim, K);
        }
        long cpuStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            cpuQ8_0GemvSplit(xQ8, wQ8, heads, headDim, K);
        }
        double cpuMs = (System.nanoTime() - cpuStart) / 1e6 / TIMED_ITERS;

        MemorySegment xDev = LlamaLayer.uploadBytes(ctx, xQ8);
        MemorySegment wDev = LlamaLayer.uploadBytes(ctx, wQ8);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, N);
        double gpuMs;
        try {
            for (int i = 0; i < WARMUP_ITERS; i++) {
                LlamaLayer.q8_0GemvSplit(ctx, xDev, wDev, outDev, heads, headDim, K);
            }
            LlamaLayer.finish(ctx);
            long gpuStart = System.nanoTime();
            for (int i = 0; i < TIMED_ITERS; i++) {
                LlamaLayer.q8_0GemvSplit(ctx, xDev, wDev, outDev, heads, headDim, K);
            }
            LlamaLayer.finish(ctx);
            gpuMs = (System.nanoTime() - gpuStart) / 1e6 / TIMED_ITERS;
        } finally {
            LlamaLayer.freeQuietly(ctx, xDev);
            LlamaLayer.freeQuietly(ctx, wDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }

        System.out.printf(Locale.ROOT, "%-28s %10d %10d %14.4f %16.4f %9.2fx%n",
                "QKV proj (Q8_0 GEMV split)", N, K, cpuMs, gpuMs, cpuMs / gpuMs);
    }

    private static void benchF32Gemv(String label, GpuContext ctx, int K, int N) throws Throwable {
        Random rnd = new Random(221);
        float[] a = randomFloats(rnd, K, 1.0);
        float[] B = randomFloats(rnd, (long) K * N, 0.5);

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cpuF32Gemv(a, B, K, N);
        }
        long cpuStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            cpuF32Gemv(a, B, K, N);
        }
        double cpuMs = (System.nanoTime() - cpuStart) / 1e6 / TIMED_ITERS;

        MemorySegment aDev = LlamaLayer.uploadFloats(ctx, a);
        MemorySegment bDev = LlamaLayer.uploadFloats(ctx, B);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, N);
        double gpuMs;
        try {
            for (int i = 0; i < WARMUP_ITERS; i++) {
                LlamaLayer.f32Gemv(ctx, aDev, bDev, outDev, K, N);
            }
            LlamaLayer.finish(ctx);
            long gpuStart = System.nanoTime();
            for (int i = 0; i < TIMED_ITERS; i++) {
                LlamaLayer.f32Gemv(ctx, aDev, bDev, outDev, K, N);
            }
            LlamaLayer.finish(ctx);
            gpuMs = (System.nanoTime() - gpuStart) / 1e6 / TIMED_ITERS;
        } finally {
            LlamaLayer.freeQuietly(ctx, aDev);
            LlamaLayer.freeQuietly(ctx, bDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }

        System.out.printf(Locale.ROOT, "%-28s %10d %10d %14.4f %16.4f %9.2fx%n", label, N, K, cpuMs, gpuMs, cpuMs / gpuMs);
    }

    private static void benchAttentionDecode(GpuContext ctx, int posInclusive) throws Throwable {
        Random rnd = new Random(231);
        float[] q = randomFloats(rnd, HEAD_DIM, 1.0);
        float[] kCache = randomFloats(rnd, (long) posInclusive * HEAD_DIM, 1.0);
        float[] vCache = randomFloats(rnd, (long) posInclusive * HEAD_DIM, 1.0);

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cpuAttentionDecode(q, kCache, vCache, posInclusive, HEAD_DIM);
        }
        long cpuStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            cpuAttentionDecode(q, kCache, vCache, posInclusive, HEAD_DIM);
        }
        double cpuMs = (System.nanoTime() - cpuStart) / 1e6 / TIMED_ITERS;

        float rsqrtD = (float) (1.0 / Math.sqrt(HEAD_DIM));
        MemorySegment qDev = LlamaLayer.uploadFloats(ctx, q);
        MemorySegment kDev = LlamaLayer.uploadFloats(ctx, kCache);
        MemorySegment vDev = LlamaLayer.uploadFloats(ctx, vCache);
        MemorySegment scoresDev = LlamaLayer.allocFloats(ctx, posInclusive);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, HEAD_DIM);
        double gpuMs;
        try {
            for (int i = 0; i < WARMUP_ITERS; i++) {
                LlamaLayer.attnScores(ctx, qDev, kDev, scoresDev, 0, HEAD_DIM, HEAD_DIM, 0, posInclusive, rsqrtD);
                LlamaLayer.softmaxInplace(ctx, scoresDev, posInclusive);
                LlamaLayer.attnWeightedSum(ctx, scoresDev, vDev, outDev, 0, HEAD_DIM, HEAD_DIM, 0, posInclusive);
            }
            LlamaLayer.finish(ctx);
            long gpuStart = System.nanoTime();
            for (int i = 0; i < TIMED_ITERS; i++) {
                LlamaLayer.attnScores(ctx, qDev, kDev, scoresDev, 0, HEAD_DIM, HEAD_DIM, 0, posInclusive, rsqrtD);
                LlamaLayer.softmaxInplace(ctx, scoresDev, posInclusive);
                LlamaLayer.attnWeightedSum(ctx, scoresDev, vDev, outDev, 0, HEAD_DIM, HEAD_DIM, 0, posInclusive);
            }
            LlamaLayer.finish(ctx);
            gpuMs = (System.nanoTime() - gpuStart) / 1e6 / TIMED_ITERS;
        } finally {
            LlamaLayer.freeQuietly(ctx, qDev);
            LlamaLayer.freeQuietly(ctx, kDev);
            LlamaLayer.freeQuietly(ctx, vDev);
            LlamaLayer.freeQuietly(ctx, scoresDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }

        System.out.printf(Locale.ROOT, "%-10d %14.4f %16.4f %9.2fx%n", posInclusive, cpuMs, gpuMs, cpuMs / gpuMs);
    }

    private static void benchQ8_0Gemm(String label, GpuContext ctx, int T, int N, int K) throws Throwable {
        Random rnd = new Random(241);
        byte[] xQ8Batch = quantizeMatrixQ8_0(rnd, T, K, 2.0);
        byte[] wQ8 = quantizeMatrixQ8_0(rnd, N, K, 1.0);

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cpuQ8_0GemmTiled(xQ8Batch, wQ8, T, N, K);
        }
        long cpuStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            cpuQ8_0GemmTiled(xQ8Batch, wQ8, T, N, K);
        }
        double cpuMs = (System.nanoTime() - cpuStart) / 1e6 / TIMED_ITERS;

        MemorySegment xDev = LlamaLayer.uploadBytes(ctx, xQ8Batch);
        MemorySegment wDev = LlamaLayer.uploadBytes(ctx, wQ8);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, (long) T * N);
        double gpuMs;
        try {
            for (int i = 0; i < WARMUP_ITERS; i++) {
                LlamaLayer.q8_0GemmTiled(ctx, xDev, wDev, outDev, T, N, K);
            }
            LlamaLayer.finish(ctx);
            long gpuStart = System.nanoTime();
            for (int i = 0; i < TIMED_ITERS; i++) {
                LlamaLayer.q8_0GemmTiled(ctx, xDev, wDev, outDev, T, N, K);
            }
            LlamaLayer.finish(ctx);
            gpuMs = (System.nanoTime() - gpuStart) / 1e6 / TIMED_ITERS;
        } finally {
            LlamaLayer.freeQuietly(ctx, xDev);
            LlamaLayer.freeQuietly(ctx, wDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }

        System.out.printf(Locale.ROOT, "%-28s %6d %10d %10d %14.4f %16.4f %9.2fx%n", label, T, N, K, cpuMs, gpuMs, cpuMs / gpuMs);
    }

    private static void benchF32Gemm(String label, GpuContext ctx, int T, int K, int N) throws Throwable {
        Random rnd = new Random(251);
        float[] A = randomFloats(rnd, (long) T * K, 1.0);
        float[] B = randomFloats(rnd, (long) K * N, 0.5);

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cpuF32GemmTiled(A, B, T, K, N);
        }
        long cpuStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            cpuF32GemmTiled(A, B, T, K, N);
        }
        double cpuMs = (System.nanoTime() - cpuStart) / 1e6 / TIMED_ITERS;

        MemorySegment aDev = LlamaLayer.uploadFloats(ctx, A);
        MemorySegment bDev = LlamaLayer.uploadFloats(ctx, B);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, (long) T * N);
        double gpuMs;
        try {
            for (int i = 0; i < WARMUP_ITERS; i++) {
                LlamaLayer.f32GemmTiled(ctx, aDev, bDev, outDev, T, K, N);
            }
            LlamaLayer.finish(ctx);
            long gpuStart = System.nanoTime();
            for (int i = 0; i < TIMED_ITERS; i++) {
                LlamaLayer.f32GemmTiled(ctx, aDev, bDev, outDev, T, K, N);
            }
            LlamaLayer.finish(ctx);
            gpuMs = (System.nanoTime() - gpuStart) / 1e6 / TIMED_ITERS;
        } finally {
            LlamaLayer.freeQuietly(ctx, aDev);
            LlamaLayer.freeQuietly(ctx, bDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }

        System.out.printf(Locale.ROOT, "%-28s %6d %10d %10d %14.4f %16.4f %9.2fx%n", label, T, N, K, cpuMs, gpuMs, cpuMs / gpuMs);
    }

    private static void benchAttentionCausalBatched(GpuContext ctx, int T) throws Throwable {
        Random rnd = new Random(261);
        float[] q = randomFloats(rnd, (long) T * HEAD_DIM, 1.0);
        float[] k = randomFloats(rnd, (long) T * HEAD_DIM, 1.0);
        float[] v = randomFloats(rnd, (long) T * HEAD_DIM, 1.0);

        for (int i = 0; i < WARMUP_ITERS; i++) {
            cpuAttentionCausalBatched(q, k, v, T, HEAD_DIM);
        }
        long cpuStart = System.nanoTime();
        for (int i = 0; i < TIMED_ITERS; i++) {
            cpuAttentionCausalBatched(q, k, v, T, HEAD_DIM);
        }
        double cpuMs = (System.nanoTime() - cpuStart) / 1e6 / TIMED_ITERS;

        float rsqrtD = (float) (1.0 / Math.sqrt(HEAD_DIM));
        MemorySegment qDev = LlamaLayer.uploadFloats(ctx, q);
        MemorySegment kDev = LlamaLayer.uploadFloats(ctx, k);
        MemorySegment vDev = LlamaLayer.uploadFloats(ctx, v);
        MemorySegment scoresDev = LlamaLayer.allocFloats(ctx, (long) T * T);
        MemorySegment outDev = LlamaLayer.allocFloats(ctx, (long) T * HEAD_DIM);
        double gpuMs;
        try {
            for (int i = 0; i < WARMUP_ITERS; i++) {
                LlamaLayer.attnScoresCausalBatched(ctx, qDev, kDev, scoresDev, HEAD_DIM, HEAD_DIM, 0, 0, HEAD_DIM, T, rsqrtD);
                LlamaLayer.softmaxInplaceRows(ctx, scoresDev, T);
                LlamaLayer.attnWeightedSumCausalBatched(ctx, scoresDev, vDev, outDev, HEAD_DIM, HEAD_DIM, 0, 0, HEAD_DIM, T);
            }
            LlamaLayer.finish(ctx);
            long gpuStart = System.nanoTime();
            for (int i = 0; i < TIMED_ITERS; i++) {
                LlamaLayer.attnScoresCausalBatched(ctx, qDev, kDev, scoresDev, HEAD_DIM, HEAD_DIM, 0, 0, HEAD_DIM, T, rsqrtD);
                LlamaLayer.softmaxInplaceRows(ctx, scoresDev, T);
                LlamaLayer.attnWeightedSumCausalBatched(ctx, scoresDev, vDev, outDev, HEAD_DIM, HEAD_DIM, 0, 0, HEAD_DIM, T);
            }
            LlamaLayer.finish(ctx);
            gpuMs = (System.nanoTime() - gpuStart) / 1e6 / TIMED_ITERS;
        } finally {
            LlamaLayer.freeQuietly(ctx, qDev);
            LlamaLayer.freeQuietly(ctx, kDev);
            LlamaLayer.freeQuietly(ctx, vDev);
            LlamaLayer.freeQuietly(ctx, scoresDev);
            LlamaLayer.freeQuietly(ctx, outDev);
        }

        System.out.printf(Locale.ROOT, "%-10d %14.4f %16.4f %9.2fx%n", T, cpuMs, gpuMs, cpuMs / gpuMs);
    }
}