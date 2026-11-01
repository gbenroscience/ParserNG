package com.github.gbenroscience.gpu.llm.opencl;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.github.gbenroscience.gpu.llm.core.CpuReference;
import com.github.gbenroscience.gpu.llm.core.Fp16;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Runs the REAL OpenCL kernels (through the project's FFM bindings) against the
 * backend-independent CPU reference. Skipped -- never passed -- when no OpenCL
 * device is visible. The device (and whether it is a CPU or a GPU) is printed
 * so a PoCL/CPU run can never be mistaken for GPU verification.
 */
class OpenCLConformanceTest {

    private static GpuContext ctx;
    private static String skipReason;

    @BeforeAll
    static void open() {
        try {
            ctx = new GpuContext();
            System.out.println("[OpenCLConformanceTest] device = " + ctx.selectedDeviceDescription);
        } catch (Throwable t) {
            skipReason = "no usable OpenCL device: " + t;
        }
    }

    @AfterAll
    static void close() {
        if (ctx != null) {
            ctx.close();
        }
    }

    private static void needDevice() {
        assumeTrue(ctx != null, skipReason);
    }

    private static byte[] runQuantize(float[] x) throws Throwable {
        MemorySegment dx = LlamaLayer.uploadFloats(ctx, x);
        MemorySegment dq = LlamaLayer.allocBytes(ctx, (CpuReference.q8Bytes(x.length) + 3) / 4 * 4L);
        try {
            LlamaLayer.quantizeActivationQ8_0(ctx, dx, dq, x.length);
            LlamaLayer.finish(ctx);
            return download(dq, CpuReference.q8Bytes(x.length));
        } finally {
            LlamaLayer.freeQuietly(ctx, dx);
            LlamaLayer.freeQuietly(ctx, dq);
        }
    }

    private static byte[] download(MemorySegment dev, int bytes) throws Throwable {
        // downloadFloats copies raw 4-byte words; reinterpret them as bytes
        float[] words = LlamaLayer.downloadFloats(ctx, dev, (bytes + 3) / 4);
        byte[] out = new byte[bytes];
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(words.length * 4).order(java.nio.ByteOrder.nativeOrder());
        for (float w : words) {
            bb.putInt(Float.floatToRawIntBits(w));
        }
        System.arraycopy(bb.array(), 0, out, 0, bytes);
        return out;
    }

    @Test
    void quantizeKernelIsByteIdenticalToReferenceOnRandomData() throws Throwable {
        needDevice();
        Random r = new Random(10);
        for (int len : new int[]{32, 4096, 11008}) {
            float[] x = new float[len];
            for (int i = 0; i < len; i++) {
                double mag = Math.pow(10, r.nextInt(9) - 5); // 1e-5 .. 1e3
                x[i] = (float) (r.nextGaussian() * mag);
            }
            assertArrayEquals(CpuReference.quantizeQ8_0(x), runQuantize(x), "len=" + len);
        }
    }

    @Test
    void quantizeKernelRoundsFp16ScaleTiesToEven() throws Throwable {
        needDevice();
        Random r = new Random(11);
        // build blocks whose scale = absmax/127 lands exactly on an fp16 rounding tie
        float[] x = new float[32 * 4000];
        int tieBlocks = 0;
        for (int b = 0; b < 4000; b++) {
            int mant = (r.nextInt(1 << 10) << 13) | 0x1000; // exact halfway between two halves
            int exp = 100 + r.nextInt(60); // float exponent field -> normal half range
            float t = Float.intBitsToFloat((exp << 23) | mant);
            float absmax = t * 127f;
            if (absmax / 127f == t && Float.isFinite(absmax) && Fp16.floatToHalfBits(t) != Fp16.floatToHalfBits(Math.nextUp(t))) {
                tieBlocks++;
            }
            x[b * 32] = absmax;
            for (int j = 1; j < 32; j++) {
                x[b * 32 + j] = (float) (r.nextGaussian() * absmax * 0.1);
            }
        }
        assertTrue(tieBlocks > 500, "test must actually construct ties, got " + tieBlocks);
        assertArrayEquals(CpuReference.quantizeQ8_0(x), runQuantize(x));
    }

    // ------------------------------------------------------------------ GEMV kernels
    private static float[] randn(Random r, int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = scale * (float) r.nextGaussian();
        }
        return v;
    }

    /** Work-group-per-row Q8_0 GEMV vs the reference, for shapes around the 64-lane work-group boundary. */
    @Test
    void q8GemvMatchesReferenceForEdgeShapes() throws Throwable {
        needDevice();
        // K/32 blocks: 1, 2, 63, 64 (== lanes), 65, 129 (not a multiple of the lane count), 256
        int[][] shapes = {{1, 32}, {3, 64}, {5, 2016}, {7, 2048}, {9, 2080}, {33, 4128}, {64, 8192}, {257, 96}};
        for (int[] nk : shapes) {
            int N = nk[0], K = nk[1];
            Random r = new Random(N * 7919L + K);
            byte[] w = CpuReference.quantizeQ8_0(randn(r, N * K, 0.05f));
            byte[] xq = CpuReference.quantizeQ8_0(randn(r, K, 1f));
            MemorySegment dx = LlamaLayer.uploadBytes(ctx, xq);
            MemorySegment dw = LlamaLayer.uploadBytes(ctx, w);
            MemorySegment dout = LlamaLayer.allocFloats(ctx, N);
            try {
                LlamaLayer.q8_0GemvPlain(ctx, dx, dw, dout, N, K);
                LlamaLayer.finish(ctx);
                float[] got = LlamaLayer.downloadFloats(ctx, dout, N);
                com.github.gbenroscience.gpu.llm.core.ErrorStats st = com.github.gbenroscience.gpu.llm.core.ErrorStats
                        .compare(got, CpuReference.q8Gemv(xq, w, N, K), 1e-4, 1e-5);
                assertTrue(st.passed(), "q8 N=" + N + " K=" + K + " " + st);
            } finally {
                LlamaLayer.freeQuietly(ctx, dx);
                LlamaLayer.freeQuietly(ctx, dw);
                LlamaLayer.freeQuietly(ctx, dout);
            }
        }
    }

    /** F32 GEMV including K % 4 != 0 (scalar path) and K % 4 == 0 (float4 path), and K smaller than the lane count. */
    @Test
    void f32GemvMatchesReferenceForEdgeShapes() throws Throwable {
        needDevice();
        int[][] shapes = {{1, 1}, {2, 3}, {5, 30}, {4, 4}, {6, 64}, {7, 257}, {9, 2048}, {130, 2052}, {33, 1000}};
        for (int[] nk : shapes) {
            int N = nk[0], K = nk[1];
            Random r = new Random(N * 104729L + K);
            float[] w = randn(r, N * K, 0.1f);
            float[] x = randn(r, K, 1f);
            MemorySegment dx = LlamaLayer.uploadFloats(ctx, x);
            MemorySegment dw = LlamaLayer.uploadFloats(ctx, w);
            MemorySegment dout = LlamaLayer.allocFloats(ctx, N);
            try {
                LlamaLayer.f32Gemv(ctx, dx, dw, dout, K, N);
                LlamaLayer.finish(ctx);
                float[] got = LlamaLayer.downloadFloats(ctx, dout, N);
                com.github.gbenroscience.gpu.llm.core.ErrorStats st = com.github.gbenroscience.gpu.llm.core.ErrorStats
                        .compare(got, CpuReference.matvecExact(w, x, N, K), 1e-4, 1e-4);
                assertTrue(st.passed(), "f32 N=" + N + " K=" + K + " " + st);
            } finally {
                LlamaLayer.freeQuietly(ctx, dx);
                LlamaLayer.freeQuietly(ctx, dw);
                LlamaLayer.freeQuietly(ctx, dout);
            }
        }
    }

    /** The Java launch geometry and the kernel's compile-time work-group size must not drift apart. */
    @Test
    void gemvWorkgroupConstantMatchesKernelSource() throws Exception {
        int javaValue = KernelSource.GEMV_WORKGROUP;
        String src = null;
        for (java.lang.reflect.Field f : KernelSource.class.getDeclaredFields()) {
            if (f.getType() == String.class && java.lang.reflect.Modifier.isStatic(f.getModifiers())) {
                f.setAccessible(true);
                String v = (String) f.get(null);
                if (v != null && v.contains("__kernel") && v.contains("q8_0_gemv_plain")) {
                    src = v;
                }
            }
        }
        assertNotNull(src, "kernel source text not found in KernelSource");
        assertTrue(src.contains("#define GEMV_WG " + javaValue), "kernel GEMV_WG must equal KernelSource.GEMV_WORKGROUP=" + javaValue);
    }

    /** Fused all-heads decode attention vs the double-precision two-pass reference (and the tiled online-softmax spec). */
    @Test
    void fusedDecodeAttentionMatchesReference() throws Throwable {
        needDevice();
        // {heads, kvHeads, headDim, nCtx}: single position, GQA, multi-tile contexts (> 256), odd headDim, MQA, widest head
        int[][] shapes = {{4, 2, 16, 1}, {8, 2, 64, 37}, {32, 8, 64, 300}, {6, 3, 24, 513}, {4, 4, 128, 256}, {4, 4, 128, 257},
            {2, 1, 256, 700}, {8, 8, 8, 1000}};
        for (int[] sh : shapes) {
            int heads = sh[0], kv = sh[1], hd = sh[2], nCtx = sh[3];
            int kvDim = kv * hd;
            Random r = new Random(heads * 31L + nCtx);
            float[] q = randn(r, heads * hd, 1.5f);
            float[] k = randn(r, nCtx * kvDim, 1.5f);
            float[] v = randn(r, nCtx * kvDim, 1f);
            MemorySegment dq = LlamaLayer.uploadFloats(ctx, q);
            MemorySegment dk = LlamaLayer.uploadFloats(ctx, k);
            MemorySegment dv = LlamaLayer.uploadFloats(ctx, v);
            MemorySegment dout = LlamaLayer.allocFloats(ctx, heads * hd);
            try {
                LlamaLayer.attnDecodeFused(ctx, dq, dk, dv, dout, heads, kv, hd, nCtx, (float) (1.0 / Math.sqrt(hd)));
                LlamaLayer.finish(ctx);
                float[] got = LlamaLayer.downloadFloats(ctx, dout, heads * hd);
                float[] ref = CpuReference.attentionDecode(q, k, v, heads, kv, hd, nCtx);
                com.github.gbenroscience.gpu.llm.core.ErrorStats st = com.github.gbenroscience.gpu.llm.core.ErrorStats.compare(got, ref, 1e-5, 1e-4);
                assertTrue(st.passed(), "heads=" + heads + " kv=" + kv + " hd=" + hd + " nCtx=" + nCtx + " " + st);
            } finally {
                LlamaLayer.freeQuietly(ctx, dq);
                LlamaLayer.freeQuietly(ctx, dk);
                LlamaLayer.freeQuietly(ctx, dv);
                LlamaLayer.freeQuietly(ctx, dout);
            }
        }
    }

    /** Huge scores must not overflow: this is the point of the max-subtracted online softmax. */
    @Test
    void fusedDecodeAttentionIsStableForHugeScores() throws Throwable {
        needDevice();
        int heads = 2, kv = 1, hd = 16, nCtx = 300;
        Random r = new Random(5);
        float[] q = randn(r, heads * hd, 60f);
        float[] k = randn(r, nCtx * kv * hd, 60f);
        float[] v = randn(r, nCtx * kv * hd, 1f);
        MemorySegment dq = LlamaLayer.uploadFloats(ctx, q);
        MemorySegment dk = LlamaLayer.uploadFloats(ctx, k);
        MemorySegment dv = LlamaLayer.uploadFloats(ctx, v);
        MemorySegment dout = LlamaLayer.allocFloats(ctx, heads * hd);
        try {
            LlamaLayer.attnDecodeFused(ctx, dq, dk, dv, dout, heads, kv, hd, nCtx, (float) (1.0 / Math.sqrt(hd)));
            LlamaLayer.finish(ctx);
            float[] got = LlamaLayer.downloadFloats(ctx, dout, heads * hd);
            com.github.gbenroscience.gpu.llm.core.ErrorStats st = com.github.gbenroscience.gpu.llm.core.ErrorStats
                    .compare(got, CpuReference.attentionDecode(q, k, v, heads, kv, hd, nCtx), 1e-4, 1e-3);
            assertTrue(st.passed(), st.toString());
        } finally {
            LlamaLayer.freeQuietly(ctx, dq);
            LlamaLayer.freeQuietly(ctx, dk);
            LlamaLayer.freeQuietly(ctx, dv);
            LlamaLayer.freeQuietly(ctx, dout);
        }
    }

    @Test
    void fusedAttentionRejectsUnsupportedGeometry() {
        assertFalse(LlamaLayer.fusedAttentionSupported(6));
        assertFalse(LlamaLayer.fusedAttentionSupported(260));
        assertTrue(LlamaLayer.fusedAttentionSupported(64));
        assertTrue(LlamaLayer.fusedAttentionSupported(256));
    }

    /** Fused single-launch RMSNorm vs the double-precision reference, incl. sizes that are not multiples of the work-group. */
    @Test
    void fusedRmsNormMatchesReference() throws Throwable {
        needDevice();
        for (int n : new int[]{1, 3, 31, 63, 64, 65, 255, 256, 257, 2048, 4097}) {
            for (float scale : new float[]{1e-6f, 1f, 3e3f}) {
                Random r = new Random(n * 13L + (long) scale);
                float[] x = randn(r, n, scale);
                float[] g = new float[n];
                for (int i = 0; i < n; i++) {
                    g[i] = 1f + 0.1f * (float) r.nextGaussian();
                }
                MemorySegment dx = LlamaLayer.uploadFloats(ctx, x);
                MemorySegment dg = LlamaLayer.uploadFloats(ctx, g);
                MemorySegment dout = LlamaLayer.allocFloats(ctx, n);
                MemorySegment dpart = LlamaLayer.allocFloats(ctx, (n + 255) / 256);
                try {
                    LlamaLayer.rmsNorm(ctx, dx, dg, dout, n, 1e-5, dpart);
                    LlamaLayer.finish(ctx);
                    float[] got = LlamaLayer.downloadFloats(ctx, dout, n);
                    com.github.gbenroscience.gpu.llm.core.ErrorStats st = com.github.gbenroscience.gpu.llm.core.ErrorStats
                            .compare(got, CpuReference.rmsNorm(x, g, 1e-5), 1e-6 * Math.max(scale, 1f), 1e-5);
                    assertTrue(st.passed(), "n=" + n + " scale=" + scale + " " + st);
                } finally {
                    LlamaLayer.freeQuietly(ctx, dx);
                    LlamaLayer.freeQuietly(ctx, dg);
                    LlamaLayer.freeQuietly(ctx, dout);
                    LlamaLayer.freeQuietly(ctx, dpart);
                }
            }
        }
    }

    /** The legacy RMSNorm path stays available and numerically equivalent. */
    @Test
    void legacyRmsNormStillMatchesReference() throws Throwable {
        needDevice();
        System.setProperty("llm.rmsnorm", "legacy");
        try {
            Random r = new Random(3);
            int n = 2048;
            float[] x = randn(r, n, 2f);
            float[] g = new float[n];
            java.util.Arrays.fill(g, 1f);
            MemorySegment dx = LlamaLayer.uploadFloats(ctx, x);
            MemorySegment dg = LlamaLayer.uploadFloats(ctx, g);
            MemorySegment dout = LlamaLayer.allocFloats(ctx, n);
            MemorySegment dpart = LlamaLayer.allocFloats(ctx, (n + 255) / 256);
            try {
                LlamaLayer.rmsNorm(ctx, dx, dg, dout, n, 1e-5, dpart);
                LlamaLayer.finish(ctx);
                assertTrue(com.github.gbenroscience.gpu.llm.core.ErrorStats.compare(LlamaLayer.downloadFloats(ctx, dout, n),
                        CpuReference.rmsNorm(x, g, 1e-5), 1e-6, 1e-5).passed());
            } finally {
                LlamaLayer.freeQuietly(ctx, dx);
                LlamaLayer.freeQuietly(ctx, dg);
                LlamaLayer.freeQuietly(ctx, dout);
                LlamaLayer.freeQuietly(ctx, dpart);
            }
        } finally {
            System.clearProperty("llm.rmsnorm");
        }
    }
}
