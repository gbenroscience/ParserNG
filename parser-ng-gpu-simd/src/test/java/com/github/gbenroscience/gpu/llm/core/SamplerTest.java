package com.github.gbenroscience.gpu.llm.core;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class SamplerTest {

    private static Sampler.Config cfg(float t, int k, float p, float pen, int win, long seed) {
        Sampler.Config c = new Sampler.Config();
        c.temperature = t;
        c.topK = k;
        c.topP = p;
        c.repetitionPenalty = pen;
        c.repetitionPenaltyWindow = win;
        c.seed = seed;
        return c;
    }

    @Test
    void topPIsAppliedToRenormalisedTopKDistribution() {
        // full-vocab softmax: p0=.493 p1=.181; renormalised over top-2: .731 / .269.
        float[] logits = new float[100];
        logits[0] = 5f;
        logits[1] = 4f;
        Sampler s = new Sampler(cfg(1f, 2, 0.7f, 1f, 0, 42));
        for (int i = 0; i < 3000; i++) {
            assertEquals(0, s.sample(logits, List.of()), "renormalised cumulative .731 >= .7 must keep ONLY token 0");
        }
    }

    @Test
    void topPKeepsMoreWhenThresholdAboveFirstRenormalisedMass() {
        float[] logits = new float[100];
        logits[0] = 5f;
        logits[1] = 4f;
        Sampler s = new Sampler(cfg(1f, 2, 0.9f, 1f, 0, 7));
        boolean saw1 = false;
        for (int i = 0; i < 3000; i++) {
            int t = s.sample(logits, List.of());
            assertTrue(t == 0 || t == 1);
            saw1 |= t == 1;
        }
        assertTrue(saw1);
    }

    @Test
    void samplingFrequenciesMatchDistribution() {
        float[] logits = {0f, (float) Math.log(2), (float) Math.log(3)};
        Sampler s = new Sampler(cfg(1f, 0, 1f, 1f, 0, 123));
        int[] c = new int[3];
        int n = 120000;
        for (int i = 0; i < n; i++) {
            c[s.sample(logits, List.of())]++;
        }
        assertEquals(1.0 / 6, c[0] / (double) n, 0.006);
        assertEquals(2.0 / 6, c[1] / (double) n, 0.006);
        assertEquals(3.0 / 6, c[2] / (double) n, 0.006);
    }

    @Test
    void temperatureSharpens() {
        float[] logits = {1f, 2f, 3f};
        Sampler s = new Sampler(cfg(0.05f, 0, 1f, 1f, 0, 5));
        for (int i = 0; i < 500; i++) {
            assertEquals(2, s.sample(logits, List.of()));
        }
    }

    @Test
    void greedyAndTies() {
        assertEquals(1, new Sampler(cfg(0f, 40, 0.9f, 1f, 0, 1)).sample(new float[]{1f, 3f, 2f}, List.of()));
        assertEquals(0, new Sampler(cfg(0f, 0, 1f, 1f, 0, 1)).sample(new float[]{2f, 2f, 2f}, List.of()), "lowest index wins ties");
        assertEquals(0, new Sampler(cfg(1f, 1, 1f, 1f, 0, 1)).sample(new float[]{2f, 2f, 1f}, List.of()), "topK=1 -> argmax, lowest index");
    }

    @Test
    void repetitionPenaltyAppliesInGreedyOncePerDistinctTokenAndRespectsWindow() {
        assertEquals(1, new Sampler(cfg(0f, 0, 1f, 1.2f, 10, 1)).sample(new float[]{2.0f, 1.9f}, List.of(0)));
        // negative logits are multiplied: -1.0*1.1=-1.1 < -1.05
        assertEquals(1, new Sampler(cfg(0f, 0, 1f, 1.1f, 10, 1)).sample(new float[]{-1.0f, -1.05f}, List.of(0)));
        // once, not per occurrence: 2/1.1=1.818 > 1.7 (thrice would give 1.503 < 1.7)
        assertEquals(0, new Sampler(cfg(0f, 0, 1f, 1.1f, 10, 1)).sample(new float[]{2.0f, 1.7f}, List.of(0, 0, 0)));
        // window=2 only sees the last two entries (1,1), token 0 is outside it
        assertEquals(0, new Sampler(cfg(0f, 0, 1f, 2.0f, 2, 1)).sample(new float[]{2.0f, 1.5f}, List.of(0, 1, 1)));
    }

    @Test
    void doesNotMutateInput() {
        float[] l = {1f, 2f, 3f};
        float[] copy = l.clone();
        new Sampler(cfg(1f, 2, 0.5f, 1.5f, 5, 1)).sample(l, List.of(0, 1, 2));
        assertArrayEquals(copy, l);
    }

    @Test
    void invalidLogitsAreRejected() {
        Sampler s = new Sampler(cfg(1f, 0, 1f, 1f, 0, 1));
        assertThrows(IllegalStateException.class, () -> s.sample(new float[]{1f, Float.NaN}, List.of()));
        assertThrows(IllegalStateException.class, () -> s.sample(new float[]{1f, Float.POSITIVE_INFINITY}, List.of()));
        assertThrows(IllegalStateException.class, () -> new Sampler(cfg(0f, 0, 1f, 1f, 0, 1)).sample(new float[]{Float.NaN}, List.of()));
        assertEquals(0, s.sample(new float[]{1f, Float.NEGATIVE_INFINITY}, List.of()), "-Inf is a legal masked token");
    }

    @Test
    void invalidConfigIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Sampler(cfg(Float.NaN, 0, 1f, 1f, 0, 1)));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(cfg(1f, -1, 1f, 1f, 0, 1)));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(cfg(1f, 0, 0f, 1f, 0, 1)));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(cfg(1f, 0, 1.5f, 1f, 0, 1)));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(cfg(1f, 0, Float.NaN, 1f, 0, 1)));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(cfg(1f, 0, 1f, 0f, 0, 1)));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(cfg(1f, 0, 1f, 1f, -1, 1)));
    }

    @Test
    void seedReproducibility() {
        float[] logits = new float[50];
        for (int i = 0; i < logits.length; i++) {
            logits[i] = (i % 7) * 0.1f;
        }
        int[] a = new int[100];
        int[] b = new int[100];
        int[] c = new int[100];
        Sampler s1 = new Sampler(cfg(1f, 20, 0.95f, 1f, 0, 99));
        Sampler s2 = new Sampler(cfg(1f, 20, 0.95f, 1f, 0, 99));
        Sampler s3 = new Sampler(cfg(1f, 20, 0.95f, 1f, 0, 100));
        for (int i = 0; i < 100; i++) {
            a[i] = s1.sample(logits, List.of());
            b[i] = s2.sample(logits, List.of());
            c[i] = s3.sample(logits, List.of());
        }
        assertArrayEquals(a, b);
        assertFalse(Arrays.equals(a, c));
    }

    @Test
    void topKLargerThanVocabAndLargeVocabSelection() {
        float[] l = new float[200000];
        for (int i = 0; i < l.length; i++) {
            l[i] = (float) Math.sin(i * 0.37) * 3f;
        }
        l[123456] = 9f;
        assertEquals(123456, new Sampler(cfg(0.01f, 40, 0.9f, 1f, 0, 3)).sample(l, List.of()));
        assertEquals(123456, new Sampler(cfg(0.01f, 1_000_000, 1f, 1f, 0, 3)).sample(l, List.of()));
    }
}
