package com.github.gbenroscience.gpu.llm.core;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Host-side token sampler shared by all three backends (their per-package
 * {@code Sampler} classes delegate here, so sampling semantics cannot differ
 * between CUDA, OpenCL and Metal).
 *
 * <h3>Pipeline (in this exact order)</h3>
 * <ol>
 *   <li>Reject NaN / +Inf logits (a corrupted forward pass must not turn into a
 *       plausible-looking token). -Inf is legal (masked token).</li>
 *   <li>Repetition penalty on the last {@code repetitionPenaltyWindow} history
 *       tokens, each distinct token penalised once (positive logit / p,
 *       negative logit * p). Applied in greedy mode too.</li>
 *   <li>{@code temperature <= 0}: greedy argmax (lowest index wins ties).</li>
 *   <li>Temperature scaling, then <b>top-k</b> restriction (0 = off).</li>
 *   <li>Softmax <b>over the top-k candidates only</b> (max-subtracted, in
 *       double), i.e. the distribution is renormalised after top-k.</li>
 *   <li><b>top-p</b> on that renormalised distribution: the smallest prefix of
 *       the descending-probability list whose cumulative mass reaches topP
 *       (at least one token).</li>
 *   <li>Sample from the kept tokens, renormalised.</li>
 * </ol>
 * Temperature is applied before top-k/top-p (HF/OpenAI order). llama.cpp's
 * default chain applies temperature last; the two agree for greedy decoding
 * and differ in which tokens top-p keeps at temperature != 1.
 *
 * Complexity: O(V log k) selection plus O(k log k); no boxing, no full-vocab
 * sort unless top-k is disabled. Reproducible for a fixed seed.
 */
public final class Sampler {

    public static class Config {

        public float temperature = 0.8f;
        /** 0 disables top-k. */
        public int topK = 40;
        /** Must be in (0, 1]; 1.0 disables top-p. */
        public float topP = 0.95f;
        /** 1.0 disables; must be &gt; 0. */
        public float repetitionPenalty = 1.1f;
        /** 0 disables the penalty. */
        public int repetitionPenaltyWindow = 64;
        public long seed = System.nanoTime();

        public void validate() {
            if (Float.isNaN(temperature) || Float.isInfinite(temperature)) {
                throw new IllegalArgumentException("temperature must be finite, got " + temperature);
            }
            if (topK < 0) {
                throw new IllegalArgumentException("topK must be >= 0 (0 = disabled), got " + topK);
            }
            if (!(topP > 0f && topP <= 1f)) {
                throw new IllegalArgumentException("topP must be in (0, 1], got " + topP);
            }
            if (!(repetitionPenalty > 0f) || Float.isInfinite(repetitionPenalty)) {
                throw new IllegalArgumentException("repetitionPenalty must be a finite value > 0, got " + repetitionPenalty);
            }
            if (repetitionPenaltyWindow < 0) {
                throw new IllegalArgumentException("repetitionPenaltyWindow must be >= 0, got " + repetitionPenaltyWindow);
            }
        }
    }

    private final Config cfg;
    private final Random random;

    public Sampler(Config cfg) {
        cfg.validate();
        this.cfg = cfg;
        this.random = new Random(cfg.seed);
    }

    /** Never mutates {@code logits}. */
    public int sample(float[] logits, List<Integer> history) {
        final int vocab = logits.length;
        if (vocab == 0) {
            throw new IllegalArgumentException("empty logits");
        }
        for (int i = 0; i < vocab; i++) {
            float v = logits[i];
            if (Float.isNaN(v)) {
                throw new IllegalStateException("NaN logit at token " + i + " -- the forward pass produced invalid values");
            }
            if (v == Float.POSITIVE_INFINITY) {
                throw new IllegalStateException("+Inf logit at token " + i + " -- the forward pass produced invalid values");
            }
        }

        float[] work = logits.clone();
        if (cfg.repetitionPenalty != 1.0f && cfg.repetitionPenaltyWindow > 0 && history != null && !history.isEmpty()) {
            applyRepetitionPenalty(work, history);
        }
        if (cfg.temperature <= 0f) {
            return argmax(work);
        }

        float invT = 1.0f / cfg.temperature;
        int k = (cfg.topK > 0) ? Math.min(cfg.topK, vocab) : vocab;

        // Candidate selection: indices of the k largest (temperature scaling is
        // monotone, so selecting on raw penalised logits is equivalent).
        long[] keys = selectTopKeys(work, k);   // ascending by (value, -index)
        int n = keys.length;
        int[] idx = new int[n];
        double[] prob = new double[n];
        // iterate from the end: descending value, ascending index on ties
        for (int j = 0; j < n; j++) {
            idx[j] = Integer.MAX_VALUE - (int) (keys[n - 1 - j] & 0xFFFFFFFFL);
        }
        double max = (double) work[idx[0]] * invT;
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            double z = (double) work[idx[j]] * invT - max;
            double e = (z == Double.NEGATIVE_INFINITY) ? 0.0 : Math.exp(z);
            prob[j] = e;
            sum += e;
        }
        for (int j = 0; j < n; j++) {
            prob[j] /= sum; // renormalised over the top-k candidates
        }

        int keep = n;
        if (cfg.topP < 1.0f) {
            double cumulative = 0.0;
            for (int j = 0; j < n; j++) {
                cumulative += prob[j];
                if (cumulative >= cfg.topP) {
                    keep = j + 1;
                    break;
                }
            }
        }
        keep = Math.max(keep, 1);

        double kept = 0.0;
        for (int j = 0; j < keep; j++) {
            kept += prob[j];
        }
        double roll = random.nextDouble() * kept;
        double running = 0.0;
        for (int j = 0; j < keep; j++) {
            running += prob[j];
            if (running > roll) {
                return idx[j];
            }
        }
        return idx[keep - 1];
    }

    /**
     * Returns the keys of the k largest logits sorted ascending, where
     * key = (order-preserving bits of the value) &lt;&lt; 32 | (Integer.MAX_VALUE - index).
     */
    private static long[] selectTopKeys(float[] v, int k) {
        if (k >= v.length) {
            long[] all = new long[v.length];
            for (int i = 0; i < all.length; i++) {
                all[i] = key(v[i], i);
            }
            Arrays.sort(all);
            return all;
        }
        // binary min-heap of size k on keys
        long[] heap = new long[k];
        for (int i = 0; i < k; i++) {
            heap[i] = key(v[i], i);
        }
        for (int i = k / 2 - 1; i >= 0; i--) {
            siftDown(heap, i, k);
        }
        for (int i = k; i < v.length; i++) {
            long kk = key(v[i], i);
            if (kk > heap[0]) {
                heap[0] = kk;
                siftDown(heap, 0, k);
            }
        }
        Arrays.sort(heap);
        return heap;
    }

    private static void siftDown(long[] h, int i, int n) {
        while (true) {
            int l = 2 * i + 1;
            if (l >= n) {
                return;
            }
            int m = (l + 1 < n && h[l + 1] < h[l]) ? l + 1 : l;
            if (h[m] >= h[i]) {
                return;
            }
            long t = h[i];
            h[i] = h[m];
            h[m] = t;
            i = m;
        }
    }

    private static long key(float value, int index) {
        int b = Float.floatToIntBits(value);
        b ^= (b >> 31) & 0x7FFFFFFF; // signed-comparable ordering of floats (-0 < +0, which is fine here)
        return ((long) b << 32) | (long) (Integer.MAX_VALUE - index);
    }

    private static int argmax(float[] v) {
        int best = 0;
        float bv = v[0];
        for (int i = 1; i < v.length; i++) {
            if (v[i] > bv) {
                bv = v[i];
                best = i;
            }
        }
        return best;
    }

    private void applyRepetitionPenalty(float[] logits, List<Integer> history) {
        int from = Math.max(0, history.size() - cfg.repetitionPenaltyWindow);
        boolean[] seen = new boolean[logits.length];
        for (int i = from; i < history.size(); i++) {
            int id = history.get(i);
            if (id < 0 || id >= logits.length || seen[id]) {
                continue;
            }
            seen[id] = true;
            float v = logits[id];
            logits[id] = (v > 0f) ? (v / cfg.repetitionPenalty) : (v * cfg.repetitionPenalty);
        }
    }
}
