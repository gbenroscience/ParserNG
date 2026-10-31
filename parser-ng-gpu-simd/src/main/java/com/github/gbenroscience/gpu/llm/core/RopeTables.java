package com.github.gbenroscience.gpu.llm.core;

/**
 * Single source of truth for the RoPE cos/sin tables. All three backends call
 * this instead of carrying their own copy of the frequency maths, so they cannot
 * drift apart. Tables are laid out [position][pair] (pair = 0..headDim/2-1) and
 * computed in double precision, then rounded once to float.
 *
 * The rotation angle for pair i at position p is
 * {@code p * freq_i}, with {@code freq_i = theta^(-2i/headDim) / freqFactor_i}
 * and then the context-scaling rule applied (ggml semantics: a
 * {@code rope_freqs.weight} tensor DIVIDES the base frequency).
 */
public final class RopeTables {

    private RopeTables() {
    }

    public enum Scaling {
        /** Plain RoPE. */
        NONE,
        /** All frequencies divided by {@code factor} (GGUF "linear"). */
        LINEAR,
        /** Llama-3.x NTK-by-parts (Meta's reference {@code apply_scaling}). */
        LLAMA3
    }

    public static final class Params {

        public Scaling scaling = Scaling.NONE;
        public double factor = 1.0;
        public double lowFreqFactor = 1.0;
        public double highFreqFactor = 4.0;
        public double origContextLength = 8192.0;
        /** Optional per-pair divisors ({@code rope_freqs.weight}); length headDim/2, or null. */
        public float[] freqFactors = null;
    }

    public static double baseFrequency(int pair, int headDim, double theta) {
        return 1.0 / Math.pow(theta, (2.0 * pair) / headDim);
    }

    public static double frequency(int pair, int headDim, double theta, Params p) {
        double freq = baseFrequency(pair, headDim, theta);
        if (p.freqFactors != null) {
            freq /= p.freqFactors[pair];
        }
        switch (p.scaling) {
            case NONE:
                return freq;
            case LINEAR:
                return freq / p.factor;
            case LLAMA3: {
                double lowWavelen = p.origContextLength / p.lowFreqFactor;
                double highWavelen = p.origContextLength / p.highFreqFactor;
                double wavelen = 2.0 * Math.PI / freq;
                if (wavelen < highWavelen) {
                    return freq;
                }
                if (wavelen > lowWavelen) {
                    return freq / p.factor;
                }
                double smooth = (p.origContextLength / wavelen - p.lowFreqFactor)
                        / (p.highFreqFactor - p.lowFreqFactor);
                return (1.0 - smooth) * (freq / p.factor) + smooth * freq;
            }
            default:
                throw new IllegalStateException("unreachable " + p.scaling);
        }
    }

    public static void fill(float[] cosOut, float[] sinOut, int maxSeq, int headDim, double theta, Params p) {
        int half = headDim / 2;
        if ((long) maxSeq * half != cosOut.length || cosOut.length != sinOut.length) {
            throw new IllegalArgumentException("table length " + cosOut.length + " != maxSeq*headDim/2 = " + (long) maxSeq * half);
        }
        if (p.freqFactors != null && p.freqFactors.length != half) {
            throw new IllegalArgumentException("freqFactors length " + p.freqFactors.length + " != headDim/2 = " + half);
        }
        double[] freq = new double[half];
        for (int i = 0; i < half; i++) {
            freq[i] = frequency(i, headDim, theta, p);
        }
        for (int pos = 0; pos < maxSeq; pos++) {
            for (int i = 0; i < half; i++) {
                double angle = pos * freq[i];
                cosOut[pos * half + i] = (float) Math.cos(angle);
                sinOut[pos * half + i] = (float) Math.sin(angle);
            }
        }
    }
}
