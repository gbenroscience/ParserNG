package com.github.gbenroscience.gpu.llm.core;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;
import org.junit.jupiter.api.Test;

class CpuReferenceTest {

    private static float[] randn(Random r, int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = scale * (float) r.nextGaussian();
        }
        return v;
    }

    // ------------------------------------------------------------------- Q8_0
    @Test
    void q8RoundTripWithinHalfStep() {
        Random r = new Random(1);
        float[] x = randn(r, 4096, 2f);
        float[] d = CpuReference.dequantizeQ8_0(CpuReference.quantizeQ8_0(x), x.length);
        for (int b = 0; b < x.length / 32; b++) {
            float amax = 0;
            for (int j = 0; j < 32; j++) {
                amax = Math.max(amax, Math.abs(x[b * 32 + j]));
            }
            // half a quantization step plus the fp16 rounding of the scale (<= 2^-11 relative on 127 steps)
            double tol = amax / 127.0 * 0.5 + amax * 1.0 / 2048 + 1e-12;
            for (int j = 0; j < 32; j++) {
                assertTrue(Math.abs(x[b * 32 + j] - d[b * 32 + j]) <= tol, "block " + b + " elt " + j);
            }
        }
    }

    @Test
    void q8ZeroBlockAndLargeValues() {
        float[] x = new float[64];
        x[40] = 1e6f;
        x[41] = -1e6f;
        float[] d = CpuReference.dequantizeQ8_0(CpuReference.quantizeQ8_0(x), 64);
        for (int i = 0; i < 32; i++) {
            assertEquals(0f, d[i]);
        }
        assertEquals(1e6f, d[40], 1e6f * 2e-3f);
        assertEquals(-1e6f, d[41], 1e6f * 2e-3f);
    }

    /** Documented range limit shared by all backends and llama.cpp: the Q8_0 scale is stored as fp16 (max 65504), so a block with |x| > 65504*127 (~8.3e6) has an infinite scale. */
    @Test
    void q8ScaleOverflowsFp16BeyondDocumentedRange() {
        float[] x = new float[32];
        x[0] = 1e9f;
        float[] d = CpuReference.dequantizeQ8_0(CpuReference.quantizeQ8_0(x), 32);
        assertTrue(Float.isInfinite(d[0]) || Float.isNaN(d[0]));
        x[0] = 8.0e6f;
        d = CpuReference.dequantizeQ8_0(CpuReference.quantizeQ8_0(x), 32);
        assertEquals(8.0e6f, d[0], 8.0e6f * 2e-3f);
    }

    @Test
    void q8GemvMatchesExactMathOnDequantizedOperands() {
        for (int[] nk : new int[][]{{1, 32}, {7, 96}, {33, 4096}, {300, 11008}}) {
            Random r = new Random(nk[0] * 31 + nk[1]);
            int N = nk[0], K = nk[1];
            byte[] w = CpuReference.quantizeQ8_0(randn(r, N * K, 0.05f));
            float[] x = randn(r, K, 1f);
            byte[] xq = CpuReference.quantizeQ8_0(x);
            float[] dev = CpuReference.q8Gemv(xq, w, N, K);
            float[] exact = CpuReference.matvecExact(CpuReference.dequantizeQ8_0(w, N * K), CpuReference.dequantizeQ8_0(xq, K), N, K);
            ErrorStats s = ErrorStats.compare(dev, exact, 1e-4, 1e-4);
            assertTrue(s.passed(), "N=" + N + " K=" + K + " " + s);
        }
    }

    @Test
    void q8SchemeErrorBudgetAgainstF32GroundTruth() {
        Random r = new Random(5);
        int N = 512, K = 4096;
        float[] wf = randn(r, N * K, 0.05f);
        float[] x = randn(r, K, 1f);
        float[] truth = CpuReference.matvecExact(wf, x, N, K);
        float[] dev = CpuReference.q8Gemv(CpuReference.quantizeQ8_0(x), CpuReference.quantizeQ8_0(wf), N, K);
        double se = 0, st = 0;
        for (int i = 0; i < N; i++) {
            se += (dev[i] - truth[i]) * (double) (dev[i] - truth[i]);
            st += truth[i] * (double) truth[i];
        }
        double rel = Math.sqrt(se / st);
        System.out.printf("Q8_0 W+A scheme relative RMS error vs F32 truth: %.4f%%%n", rel * 100);
        assertTrue(rel < 0.01, "relative RMS error " + rel);
    }

    // ---------------------------------------------------------------- RMSNorm
    @Test
    void rmsNormProperties() {
        double eps = 1e-5;
        for (int n : new int[]{1, 3, 31, 32, 255, 256, 257, 4097}) {
            Random r = new Random(n);
            float[] x = randn(r, n, 3f);
            float[] g = new float[n];
            java.util.Arrays.fill(g, 1f);
            float[] y = CpuReference.rmsNorm(x, g, eps);
            double ms = 0, oy = 0;
            for (float v : x) {
                ms += (double) v * v;
            }
            ms /= n;
            for (float v : y) {
                oy += (double) v * v;
            }
            oy /= n;
            assertEquals(ms / (ms + eps), oy, 1e-6, "n=" + n);
        }
    }

    @Test
    void rmsNormExtremeMagnitudes() {
        float[] big = {1e18f, -1e18f, 1e18f, 1e18f};
        float[] one = {1f, 1f, 1f, 1f};
        float[] y = CpuReference.rmsNorm(big, one, 1e-5);
        for (int i = 0; i < 4; i++) {
            assertEquals(Math.signum(big[i]), y[i], 1e-6);
        }
        float[] tiny = {1e-30f, 1e-30f};
        float[] z = CpuReference.rmsNorm(tiny, new float[]{1f, 1f}, 1e-5);
        assertEquals(1e-30 / Math.sqrt(1e-5), z[0], 1e-33, "epsilon dominates near zero");
        float[] zero = CpuReference.rmsNorm(new float[8], new float[8], 1e-5);
        for (float v : zero) {
            assertEquals(0f, v);
        }
    }

    // ------------------------------------------------------------------- RoPE
    private static final RopeTables.Params PLAIN = new RopeTables.Params();

    @Test
    void ropeKnownValueAndIdentityAndNorm() {
        float[] x = {1f, 0f};
        float[] y = CpuReference.ropeInterleaved(x, 1, 2, 2, 1, 10000.0, PLAIN);
        assertEquals(Math.cos(1.0), y[0], 1e-7);
        assertEquals(Math.sin(1.0), y[1], 1e-7);
        Random r = new Random(3);
        float[] q = randn(r, 4 * 16, 1f);
        assertArrayEquals(q, CpuReference.ropeInterleaved(q, 4, 16, 16, 0, 10000.0, PLAIN), 1e-7f);
        float[] rot = CpuReference.ropeInterleaved(q, 4, 16, 16, 37, 10000.0, PLAIN);
        for (int h = 0; h < 4; h++) {
            for (int i = 0; i < 8; i++) {
                double a = Math.hypot(q[h * 16 + 2 * i], q[h * 16 + 2 * i + 1]);
                double b = Math.hypot(rot[h * 16 + 2 * i], rot[h * 16 + 2 * i + 1]);
                assertEquals(a, b, 1e-6);
            }
        }
    }

    @Test
    void neoxEqualsInterleavedUnderLayoutPermutation() {
        Random r = new Random(4);
        float[] x = randn(r, 8 * 32, 1f);
        float[] a = CpuReference.toSplitLayout(CpuReference.ropeInterleaved(x, 8, 32, 32, 123, 500000.0, PLAIN), 8, 32);
        float[] b = CpuReference.ropeNeox(CpuReference.toSplitLayout(x, 8, 32), 8, 32, 32, 123, 500000.0, PLAIN);
        ErrorStats s = ErrorStats.compare(b, a, 1e-6, 1e-6);
        assertTrue(s.passed(), s.toString());
    }

    @Test
    void ropeRelativePositionProperty() {
        Random r = new Random(6);
        float[] q = randn(r, 32, 1f);
        float[] k = randn(r, 32, 1f);
        double d1 = dot(CpuReference.ropeInterleaved(q, 1, 32, 32, 50, 10000.0, PLAIN), CpuReference.ropeInterleaved(k, 1, 32, 32, 20, 10000.0, PLAIN));
        double d2 = dot(CpuReference.ropeInterleaved(q, 1, 32, 32, 1050, 10000.0, PLAIN), CpuReference.ropeInterleaved(k, 1, 32, 32, 1020, 10000.0, PLAIN));
        assertEquals(d1, d2, 1e-4);
    }

    @Test
    void partialRotaryLeavesTailUntouched() {
        Random r = new Random(7);
        float[] x = randn(r, 2 * 16, 1f);
        float[] y = CpuReference.ropeInterleaved(x, 2, 16, 8, 9, 10000.0, PLAIN);
        for (int h = 0; h < 2; h++) {
            for (int d = 8; d < 16; d++) {
                assertEquals(x[h * 16 + d], y[h * 16 + d]);
            }
        }
        assertNotEquals(x[0], y[0]);
    }

    @Test
    void ropeScalingRules() {
        RopeTables.Params lin = new RopeTables.Params();
        lin.scaling = RopeTables.Scaling.LINEAR;
        lin.factor = 4;
        assertEquals(RopeTables.frequency(3, 16, 10000, PLAIN) / 4, RopeTables.frequency(3, 16, 10000, lin), 1e-15);
        RopeTables.Params ff = new RopeTables.Params();
        ff.freqFactors = new float[]{1, 2, 4, 8, 1, 1, 1, 1};
        assertEquals(RopeTables.frequency(2, 16, 10000, PLAIN) / 4, RopeTables.frequency(2, 16, 10000, ff), 1e-15);
        RopeTables.Params l3 = new RopeTables.Params();
        l3.scaling = RopeTables.Scaling.LLAMA3;
        l3.factor = 8;
        int hd = 128;
        // highest-frequency pair: wavelength 2pi << 8192/4 -> unchanged; lowest: wavelength > 8192 -> /8
        assertEquals(RopeTables.baseFrequency(0, hd, 500000), RopeTables.frequency(0, hd, 500000, l3), 1e-15);
        int last = hd / 2 - 1;
        assertEquals(RopeTables.baseFrequency(last, hd, 500000) / 8, RopeTables.frequency(last, hd, 500000, l3), 1e-18);
        // monotone non-increasing frequencies after scaling
        double prev = Double.MAX_VALUE;
        for (int i = 0; i < hd / 2; i++) {
            double f = RopeTables.frequency(i, hd, 500000, l3);
            assertTrue(f <= prev + 1e-18);
            prev = f;
        }
    }

    @Test
    void ropeTableMatchesDirectRotation() {
        int maxSeq = 40, hd = 16, half = 8;
        float[] c = new float[maxSeq * half];
        float[] s = new float[maxSeq * half];
        RopeTables.fill(c, s, maxSeq, hd, 10000.0, PLAIN);
        for (int pos : new int[]{0, 1, 17, 39}) {
            for (int i = 0; i < half; i++) {
                double ang = pos * RopeTables.baseFrequency(i, hd, 10000.0);
                assertEquals((float) Math.cos(ang), c[pos * half + i]);
                assertEquals((float) Math.sin(ang), s[pos * half + i]);
            }
        }
        assertThrows(IllegalArgumentException.class, () -> RopeTables.fill(new float[3], new float[3], 40, 16, 1e4, PLAIN));
    }

    private static double dot(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (double) a[i] * b[i];
        }
        return s;
    }

    // -------------------------------------------------------------- attention
    @Test
    void onlineSoftmaxEqualsTwoPassForAllTilesAndShapes() {
        int[][] shapes = {{4, 4, 8}, {8, 2, 16}, {32, 8, 128}, {6, 3, 12}, {5, 1, 10}};
        for (int[] sh : shapes) {
            int heads = sh[0], kv = sh[1], hd = sh[2];
            for (int nCtx : new int[]{1, 2, 7, 33, 130}) {
                Random r = new Random(heads * 1000 + nCtx);
                float[] q = randn(r, heads * hd, 1.5f);
                float[] k = randn(r, nCtx * kv * hd, 1.5f);
                float[] v = randn(r, nCtx * kv * hd, 1f);
                float[] ref = CpuReference.attentionDecode(q, k, v, heads, kv, hd, nCtx);
                for (int tile : new int[]{1, 3, 16, 64, 1000}) {
                    float[] on = CpuReference.attentionDecodeOnline(q, k, v, heads, kv, hd, nCtx, tile);
                    ErrorStats s = ErrorStats.compare(on, ref, 1e-6, 1e-6);
                    assertTrue(s.passed(), "heads=" + heads + " kv=" + kv + " hd=" + hd + " nCtx=" + nCtx + " tile=" + tile + " " + s);
                }
            }
        }
    }

    @Test
    void groupedQueryHeadMapping() {
        int heads = 8, kv = 2, hd = 4;
        float[] q = new float[heads * hd];
        float[] k = new float[kv * hd];
        float[] v = new float[kv * hd];
        for (int i = 0; i < v.length; i++) {
            v[i] = i + 1; // kv head 0 -> 1..4, kv head 1 -> 5..8
        }
        float[] out = CpuReference.attentionDecode(q, k, v, heads, kv, hd, 1);
        for (int h = 0; h < heads; h++) {
            int expectKv = h / (heads / kv);
            for (int d = 0; d < hd; d++) {
                assertEquals(v[expectKv * hd + d], out[h * hd + d], 1e-6, "head " + h);
            }
        }
        assertThrows(IllegalArgumentException.class, () -> CpuReference.attentionDecode(q, k, v, 8, 3, 4, 1));
    }

    @Test
    void softmaxIsStableForHugeScores() {
        int hd = 4;
        float[] q = {1e4f, 0, 0, 0};
        float[] k = {1e4f, 0, 0, 0, -1e4f, 0, 0, 0, 5e3f, 0, 0, 0};
        float[] v = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        for (float[] o : new float[][]{CpuReference.attentionDecode(q, k, v, 1, 1, hd, 3),
            CpuReference.attentionDecodeOnline(q, k, v, 1, 1, hd, 3, 2)}) {
            assertEquals(1f, o[0], 1e-6);
            assertEquals(0f, o[1], 1e-6);
            assertFalse(Float.isNaN(o[2]));
        }
    }

    @Test
    void chunkedPrefillEqualsPerTokenDecodeAndIsChunkSizeInvariant() {
        int heads = 4, kv = 2, hd = 8, total = 21;
        Random r = new Random(11);
        float[] k = randn(r, total * kv * hd, 1f);
        float[] v = randn(r, total * kv * hd, 1f);
        float[] q = randn(r, total * heads * hd, 1f);
        float[] whole = CpuReference.attentionChunk(q, k, v, heads, kv, hd, 0, total);
        int qRow = heads * hd;
        for (int chunk : new int[]{1, 4, 5, 8, 21}) {
            float[] assembled = new float[total * qRow];
            for (int s = 0; s < total; s += chunk) {
                int t = Math.min(chunk, total - s);
                float[] part = CpuReference.attentionChunk(java.util.Arrays.copyOfRange(q, s * qRow, (s + t) * qRow), k, v, heads, kv, hd, s, t);
                System.arraycopy(part, 0, assembled, s * qRow, t * qRow);
            }
            assertArrayEquals(whole, assembled, 0f, "chunk=" + chunk);
        }
    }

    // -------------------------------------------------------------------- FFN
    @Test
    void erfAndGeluAccuracy() {
        assertEquals(0.5204998778130465, CpuReference.erf(0.5), 1e-13);
        assertEquals(0.8427007929497149, CpuReference.erf(1.0), 1e-13);
        assertEquals(0.9953222650189527, CpuReference.erf(2.0), 1e-13);
        assertEquals(0.9999779095030014, CpuReference.erf(3.0), 1e-12);
        assertEquals(-0.8427007929497149, CpuReference.erf(-1.0), 1e-13);
        assertEquals(1.0, CpuReference.erf(10), 0);
        assertEquals(0.0, CpuReference.gelu(0, false), 0);
        assertEquals(0.8413447460685429 , CpuReference.gelu(1.0, false), 1e-12);
        for (double x = -6; x <= 6; x += 0.25) {
            assertEquals(CpuReference.gelu(x, false), CpuReference.gelu(x, true), 2e-3, "tanh approximation differs by <2e-3");
        }
    }

    @Test
    void swigluMatchesDefinition() {
        float[] g = {-100f, -1f, 0f, 1f, 100f};
        float[] u = {2f, 2f, 2f, 2f, 2f};
        float[] y = CpuReference.swiglu(g, u);
        assertEquals(0f, y[0], 1e-30f);
        assertEquals(-1.0 / (1 + Math.E) * 2, y[1], 1e-7);
        assertEquals(0f, y[2]);
        assertEquals(1.0 / (1 + Math.exp(-1)) * 2, y[3], 1e-7);
        assertEquals(200f, y[4], 1e-3);
    }

    // ------------------------------------------------------------- ErrorStats
    @Test
    void errorStatsReportsEverything() {
        float[] e = {1f, 2f, 3f, 4f};
        float[] a = {1f, 2.5f, Float.NaN, Float.POSITIVE_INFINITY};
        ErrorStats s = ErrorStats.compare(a, e, 1e-3, 1e-3);
        assertEquals(1, s.nanCount);
        assertEquals(1, s.infCount);
        assertEquals(3, s.mismatchCount);
        assertFalse(s.passed());
        assertEquals(0.5, s.maxAbsError, 1e-12);
        ErrorStats ok = ErrorStats.compare(new float[]{1f, 2f}, new float[]{1f, 2.0001f}, 1e-3, 0);
        assertTrue(ok.passed());
        assertEquals(1, ok.worstIndex);
    }
}
