package com.github.gbenroscience.gpu.llm.core;

/**
 * Independent CPU reference implementations of every operation in the
 * inference path. Written for clarity and accuracy, not speed: reductions
 * accumulate in {@code double}, softmax is max-subtracted, and nothing here
 * shares code with any GPU kernel (only {@link RopeTables}, which is itself
 * the single source of RoPE frequencies, and {@link Fp16}).
 *
 * Two flavours exist for the quantized matrix-vector product on purpose:
 * {@link #q8Gemv} mirrors the device arithmetic (int8 x int8 per block,
 * float scale product) so kernels can be compared bit-for-bit-ish, and
 * {@link #matvecExact} is the F32 ground truth so the quantization error
 * budget of the whole scheme can be measured, not assumed.
 */
public final class CpuReference {

    public static final int Q8_GROUP = 32;
    public static final int Q8_BLOCK_BYTES = 34;

    private CpuReference() {
    }

    // ------------------------------------------------------------------ Q8_0

    public static int q8Bytes(int len) {
        if (len % Q8_GROUP != 0) {
            throw new IllegalArgumentException("length " + len + " is not a multiple of " + Q8_GROUP);
        }
        return (len / Q8_GROUP) * Q8_BLOCK_BYTES;
    }

    /** Same arithmetic as the device kernel quantize_activation_q8_0_blocks (fp16 scale RNE, rint, clamp +/-127). */
    public static byte[] quantizeQ8_0(float[] x) {
        byte[] out = new byte[q8Bytes(x.length)];
        for (int b = 0; b < x.length / Q8_GROUP; b++) {
            int off = b * Q8_GROUP;
            float absmax = 0f;
            for (int j = 0; j < Q8_GROUP; j++) {
                absmax = Math.max(absmax, Math.abs(x[off + j]));
            }
            float scale = absmax / 127.0f;
            float inv = (scale > 0f) ? (1.0f / scale) : 0f;
            int bits = Fp16.floatToHalfBits(scale);
            int o = b * Q8_BLOCK_BYTES;
            out[o] = (byte) (bits & 0xFF);
            out[o + 1] = (byte) ((bits >> 8) & 0xFF);
            for (int j = 0; j < Q8_GROUP; j++) {
                int q = (int) Math.rint((double) (x[off + j] * inv));
                q = Math.max(-127, Math.min(127, q));
                out[o + 2 + j] = (byte) q;
            }
        }
        return out;
    }

    public static float[] dequantizeQ8_0(byte[] q, int len) {
        float[] out = new float[len];
        for (int b = 0; b < len / Q8_GROUP; b++) {
            int o = b * Q8_BLOCK_BYTES;
            float scale = Fp16.halfBitsToFloat((q[o] & 0xFF) | ((q[o + 1] & 0xFF) << 8));
            for (int j = 0; j < Q8_GROUP; j++) {
                out[b * Q8_GROUP + j] = q[o + 2 + j] * scale;
            }
        }
        return out;
    }

    /** y[n] = sum_k W[n,k] * x[k] using the device's arithmetic: int8 dot per 32-block, float scale product, float accumulation. */
    public static float[] q8Gemv(byte[] xQ8, byte[] wQ8, int N, int K) {
        int nb = K / Q8_GROUP;
        int stride = nb * Q8_BLOCK_BYTES;
        if (xQ8.length != stride || wQ8.length != (long) N * stride) {
            throw new IllegalArgumentException("Q8_0 buffer sizes do not match N=" + N + " K=" + K);
        }
        float[] y = new float[N];
        for (int n = 0; n < N; n++) {
            float acc = 0f;
            for (int b = 0; b < nb; b++) {
                int xo = b * Q8_BLOCK_BYTES;
                int wo = n * stride + b * Q8_BLOCK_BYTES;
                float xs = Fp16.halfBitsToFloat((xQ8[xo] & 0xFF) | ((xQ8[xo + 1] & 0xFF) << 8));
                float ws = Fp16.halfBitsToFloat((wQ8[wo] & 0xFF) | ((wQ8[wo + 1] & 0xFF) << 8));
                float scale = xs * ws;
                int iacc = 0;
                for (int j = 0; j < Q8_GROUP; j++) {
                    iacc += xQ8[xo + 2 + j] * wQ8[wo + 2 + j];
                }
                acc += iacc * scale;
            }
            y[n] = acc;
        }
        return y;
    }

    /** F32 ground truth: W is [N, K] row-major, double accumulation. */
    public static float[] matvecExact(float[] W, float[] x, int N, int K) {
        float[] y = new float[N];
        for (int n = 0; n < N; n++) {
            double acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (double) W[n * K + k] * x[k];
            }
            y[n] = (float) acc;
        }
        return y;
    }

    // --------------------------------------------------------------- RMSNorm

    /** out = x / sqrt(mean(x^2) + eps) * gamma, accumulated in double. */
    public static float[] rmsNorm(float[] x, float[] gamma, double eps) {
        double ss = 0;
        for (float v : x) {
            ss += (double) v * v;
        }
        double inv = 1.0 / Math.sqrt(ss / x.length + eps);
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (float) (x[i] * inv * gamma[i]);
        }
        return out;
    }

    // ------------------------------------------------------------------ RoPE

    /**
     * GGUF/llama "normal" convention: rotate adjacent pairs (2i, 2i+1) inside
     * the first nRot dims of every head. Returns a new array.
     */
    public static float[] ropeInterleaved(float[] x, int heads, int headDim, int nRot, int pos, double theta, RopeTables.Params p) {
        return rope(x, heads, headDim, nRot, pos, theta, p, false);
    }

    /** NEOX / HF convention: rotate (i, i + nRot/2). Returns a new array. */
    public static float[] ropeNeox(float[] x, int heads, int headDim, int nRot, int pos, double theta, RopeTables.Params p) {
        return rope(x, heads, headDim, nRot, pos, theta, p, true);
    }

    private static float[] rope(float[] x, int heads, int headDim, int nRot, int pos, double theta, RopeTables.Params p, boolean neox) {
        float[] out = x.clone();
        int halfRot = nRot / 2;
        for (int h = 0; h < heads; h++) {
            int base = h * headDim;
            for (int i = 0; i < halfRot; i++) {
                // frequency exponent uses nRot (ggml n_dims), not headDim
                double freq = RopeTables.frequency(i, nRot, theta, p);
                double angle = pos * freq;
                double c = Math.cos(angle);
                double s = Math.sin(angle);
                int a = neox ? base + i : base + 2 * i;
                int b = neox ? base + i + halfRot : base + 2 * i + 1;
                double x0 = x[a];
                double x1 = x[b];
                out[a] = (float) (x0 * c - x1 * s);
                out[b] = (float) (x0 * s + x1 * c);
            }
        }
        return out;
    }

    /** Maps interleaved layout [x0 y0 x1 y1 ...] of every head to the device "split" layout [x0 x1 ... | y0 y1 ...]. */
    public static float[] toSplitLayout(float[] x, int heads, int headDim) {
        float[] out = new float[x.length];
        int half = headDim / 2;
        for (int h = 0; h < heads; h++) {
            for (int i = 0; i < half; i++) {
                out[h * headDim + i] = x[h * headDim + 2 * i];
                out[h * headDim + half + i] = x[h * headDim + 2 * i + 1];
            }
        }
        return out;
    }

    // ------------------------------------------------------------- attention

    /**
     * Decode-step attention for ONE query position over nCtx cached positions
     * (the current token already written at index nCtx-1). Grouped-query:
     * query head h reads KV head h / (heads/kvHeads). Two-pass, max-subtracted
     * softmax in double.
     */
    public static float[] attentionDecode(float[] q, float[] kCache, float[] vCache, int heads, int kvHeads, int headDim, int nCtx) {
        checkGqa(heads, kvHeads);
        int kvDim = kvHeads * headDim;
        int group = heads / kvHeads;
        double inv = 1.0 / Math.sqrt(headDim);
        float[] out = new float[heads * headDim];
        double[] s = new double[nCtx];
        for (int h = 0; h < heads; h++) {
            int kvOff = (h / group) * headDim;
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < nCtx; j++) {
                double dot = 0;
                for (int d = 0; d < headDim; d++) {
                    dot += (double) q[h * headDim + d] * kCache[j * kvDim + kvOff + d];
                }
                s[j] = dot * inv;
                max = Math.max(max, s[j]);
            }
            double sum = 0;
            for (int j = 0; j < nCtx; j++) {
                s[j] = Math.exp(s[j] - max);
                sum += s[j];
            }
            for (int d = 0; d < headDim; d++) {
                double acc = 0;
                for (int j = 0; j < nCtx; j++) {
                    acc += s[j] * vCache[j * kvDim + kvOff + d];
                }
                out[h * headDim + d] = (float) (acc / sum);
            }
        }
        return out;
    }

    /**
     * Same result as {@link #attentionDecode} computed FlashAttention-style:
     * the context is consumed in tiles with a running max / normaliser
     * (online softmax), never materialising all scores. This is the
     * mathematical specification a fused GPU attention kernel must match.
     */
    public static float[] attentionDecodeOnline(float[] q, float[] kCache, float[] vCache, int heads, int kvHeads, int headDim, int nCtx, int tile) {
        checkGqa(heads, kvHeads);
        if (tile <= 0) {
            throw new IllegalArgumentException("tile must be > 0");
        }
        int kvDim = kvHeads * headDim;
        int group = heads / kvHeads;
        double inv = 1.0 / Math.sqrt(headDim);
        float[] out = new float[heads * headDim];
        for (int h = 0; h < heads; h++) {
            int kvOff = (h / group) * headDim;
            double m = Double.NEGATIVE_INFINITY;
            double l = 0;
            double[] acc = new double[headDim];
            for (int t0 = 0; t0 < nCtx; t0 += tile) {
                int t1 = Math.min(nCtx, t0 + tile);
                double[] s = new double[t1 - t0];
                double tm = Double.NEGATIVE_INFINITY;
                for (int j = t0; j < t1; j++) {
                    double dot = 0;
                    for (int d = 0; d < headDim; d++) {
                        dot += (double) q[h * headDim + d] * kCache[j * kvDim + kvOff + d];
                    }
                    s[j - t0] = dot * inv;
                    tm = Math.max(tm, s[j - t0]);
                }
                double mNew = Math.max(m, tm);
                double corr = (m == Double.NEGATIVE_INFINITY) ? 0.0 : Math.exp(m - mNew);
                l *= corr;
                for (int d = 0; d < headDim; d++) {
                    acc[d] *= corr;
                }
                for (int j = t0; j < t1; j++) {
                    double p = Math.exp(s[j - t0] - mNew);
                    l += p;
                    for (int d = 0; d < headDim; d++) {
                        acc[d] += p * vCache[j * kvDim + kvOff + d];
                    }
                }
                m = mNew;
            }
            for (int d = 0; d < headDim; d++) {
                out[h * headDim + d] = (float) (acc[d] / l);
            }
        }
        return out;
    }

    /**
     * Prefill attention for a CHUNK of T query rows that start at absolute
     * position startPos: row t attends to cache positions [0, startPos+t]
     * (causal, including everything cached by earlier chunks). Q is
     * [T, heads*headDim]; K/V caches are [>= startPos+T, kvDim] with the
     * chunk's own K/V already written. This is the semantics chunked prefill
     * must reproduce, and it equals single-token decode applied row by row.
     */
    public static float[] attentionChunk(float[] qChunk, float[] kCache, float[] vCache, int heads, int kvHeads, int headDim, int startPos, int T) {
        int qRow = heads * headDim;
        float[] out = new float[T * qRow];
        for (int t = 0; t < T; t++) {
            float[] q = java.util.Arrays.copyOfRange(qChunk, t * qRow, (t + 1) * qRow);
            float[] o = attentionDecode(q, kCache, vCache, heads, kvHeads, headDim, startPos + t + 1);
            System.arraycopy(o, 0, out, t * qRow, qRow);
        }
        return out;
    }

    private static void checkGqa(int heads, int kvHeads) {
        if (kvHeads <= 0 || heads % kvHeads != 0) {
            throw new IllegalArgumentException("heads (" + heads + ") must be a multiple of kvHeads (" + kvHeads + ")");
        }
    }

    // ------------------------------------------------------------------- FFN

    public static float[] swiglu(float[] gate, float[] up) {
        float[] out = new float[gate.length];
        for (int i = 0; i < out.length; i++) {
            double g = gate[i];
            out[i] = (float) (g / (1.0 + Math.exp(-g)) * up[i]);
        }
        return out;
    }

    public static float[] geglu(float[] gate, float[] up, boolean tanhApprox) {
        float[] out = new float[gate.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) (gelu(gate[i], tanhApprox) * up[i]);
        }
        return out;
    }

    public static double gelu(double x, boolean tanhApprox) {
        if (tanhApprox) {
            return 0.5 * x * (1.0 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
        }
        return 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
    }

    /** erf to ~1e-12: Taylor series for |x| &lt; 2.5, continued fraction for erfc beyond. */
    public static double erf(double x) {
        double ax = Math.abs(x);
        double r;
        if (ax < 2.5) {
            double term = ax;
            double sum = ax;
            for (int n = 1; n < 80; n++) {
                term *= -ax * ax / n;
                sum += term / (2 * n + 1);
            }
            r = sum * 1.1283791670955126; // 2/sqrt(pi)
        } else if (ax > 6.5) {
            r = 1.0;
        } else {
            double t = ax;
            for (int k = 80; k >= 1; k--) {
                t = ax + (k / 2.0) / t;
            }
            r = 1.0 - Math.exp(-ax * ax) / (Math.sqrt(Math.PI) * t);
        }
        return x < 0 ? -r : r;
    }
}
