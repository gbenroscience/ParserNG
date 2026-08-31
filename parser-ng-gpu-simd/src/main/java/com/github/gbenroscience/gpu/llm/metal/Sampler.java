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
package com.github.gbenroscience.gpu.llm.metal;

/**
 *
 * @author GBEMIRO
 */ 

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Host-side, pure-Java sampling over the logits LlamaLayer.finalLogits (this package's Metal LlamaLayer)
 * already copied off the device. No GPU work here at all -- vocab-sized
 * arrays (tens of thousands of floats) are cheap to sort/scan on the CPU,
 * and doing it there avoids a whole extra kernel (or several: temperature
 * scale, penalty, sort/select for top-k, cumulative-sum for top-p) for
 * work that's a rounding error next to the GEMVs/GEMMs it sits beside.
 *
 * Order of operations (matches the conventional llama.cpp-style pipeline):
 *   1. repetition penalty (divide/multiply logits for recently-seen tokens)
 *   2. temperature scaling
 *   3. softmax -> probabilities
 *   4. top-k filter (keep only the K highest-probability tokens)
 *   5. top-p / nucleus filter (keep the smallest prefix whose cumulative
 *      probability >= topP, applied AFTER top-k narrows the candidate set)
 *   6. renormalize the surviving probabilities, sample from them
 *
 * temperature <= 0 short-circuits everything above and returns pure
 * argmax (deterministic greedy), same behavior the CUDA/OpenCL ports' old
 * generate_token had before Sampler existed.
 */
public final class Sampler {

    public static final class Config {
        public float temperature = 0.8f;
        /** 0 disables top-k filtering (all tokens remain candidates going into top-p). */
        public int topK = 40;
        /** 1.0 disables top-p filtering. */
        public float topP = 0.95f;
        /**
         * Multiplicative penalty applied to logits of tokens already seen
         * in the supplied history. 1.0 disables it. Standard llama.cpp
         * convention: positive logits are DIVIDED by the penalty (pushed
         * toward zero, less likely), negative logits are MULTIPLIED by it
         * (pushed more negative, also less likely) -- this asymmetry is
         * intentional so the penalty always discourages repetition
         * regardless of the logit's sign.
         */
        public float repetitionPenalty = 1.1f;
        /**
         * How many of the most recent generated/prompt tokens count
         * toward the repetition penalty. 0 disables the penalty
         * regardless of repetitionPenalty's value.
         */
        public int repetitionPenaltyWindow = 64;
        public long seed = System.nanoTime();
    }

    private final Config cfg;
    private final Random random;

    public Sampler(Config cfg) {
        this.cfg = cfg;
        this.random = new Random(cfg.seed);
    }

    /**
     * Samples the next token id from a fresh copy of `logits` (never
     * mutates the caller's array). `history` is the full token sequence
     * so far (prompt + generated) -- only the last
     * cfg.repetitionPenaltyWindow entries are consulted.
     */
    public int sample(float[] logits, List<Integer> history) {
        if (cfg.temperature <= 0f) {
            return argmax(logits);
        }

        float[] working = logits.clone();

        if (cfg.repetitionPenalty != 1.0f && cfg.repetitionPenaltyWindow > 0 && !history.isEmpty()) {
            applyRepetitionPenalty(working, history, cfg.repetitionPenalty, cfg.repetitionPenaltyWindow);
        }

        applyTemperature(working, cfg.temperature);

        float[] probs = softmax(working);

        // Candidate index list, sorted by probability descending -- both
        // top-k and top-p operate on this same sorted order.
        Integer[] order = new Integer[probs.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, Comparator.comparingDouble((Integer i) -> probs[i]).reversed());

        int keep = probs.length;
        if (cfg.topK > 0) {
            keep = Math.min(keep, cfg.topK);
        }
        if (cfg.topP < 1.0f) {
            double cumulative = 0.0;
            int cutoff = keep;
            for (int i = 0; i < keep; i++) {
                cumulative += probs[order[i]];
                if (cumulative >= cfg.topP) {
                    cutoff = i + 1;
                    break;
                }
            }
            keep = Math.min(keep, cutoff);
        }
        keep = Math.max(keep, 1); // always leave at least one candidate

        double renormSum = 0.0;
        for (int i = 0; i < keep; i++) {
            renormSum += probs[order[i]];
        }

        double roll = random.nextDouble() * renormSum;
        double running = 0.0;
        for (int i = 0; i < keep; i++) {
            running += probs[order[i]];
            if (running >= roll) {
                return order[i];
            }
        }
        return order[keep - 1]; // floating-point fallback, should be unreachable
    }

    private static int argmax(float[] logits) {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > bestVal) {
                bestVal = logits[i];
                best = i;
            }
        }
        return best;
    }

    private static void applyRepetitionPenalty(float[] logits, List<Integer> history, float penalty, int window) {
        int from = Math.max(0, history.size() - window);
        // A token may appear more than once in the window; the penalty is
        // idempotent per application, not cumulative per occurrence
        // (matches llama.cpp's default), so track which ids were already
        // penalized rather than re-penalizing on every repeat.
        List<Integer> seen = new ArrayList<>();
        for (int i = from; i < history.size(); i++) {
            int id = history.get(i);
            if (seen.contains(id)) {
                continue;
            }
            seen.add(id);
            if (id < 0 || id >= logits.length) {
                continue;
            }
            float v = logits[id];
            logits[id] = (v > 0f) ? (v / penalty) : (v * penalty);
        }
    }

    private static void applyTemperature(float[] logits, float temperature) {
        for (int i = 0; i < logits.length; i++) {
            logits[i] /= temperature;
        }
    }

    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            if (v > max) {
                max = v;
            }
        }
        float[] out = new float[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            double e = Math.exp(logits[i] - max);
            out[i] = (float) e;
            sum += e;
        }
        float invSum = (float) (1.0 / sum);
        for (int i = 0; i < out.length; i++) {
            out[i] *= invSum;
        }
        return out;
    }
}