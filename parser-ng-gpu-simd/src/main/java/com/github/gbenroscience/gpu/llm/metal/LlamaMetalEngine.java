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

import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Metal counterpart of {@code com.github.gbenroscience.gpu.llm.cuda.LlamaCudaEngine}
 * -- tokenizer -> embedding lookup -> batched prefill (capped at
 * cfg.max_prefill_batch) -> decode-tail fallback for prompt overflow ->
 * per-token decode + sampling loop -> detokenize. Ties together every
 * other file in this package.
 *
 * PROMPT LENGTH HANDLING, THREADING: identical contract to the CUDA
 * port -- see {@code LlamaCudaEngine}'s class javadoc for the full
 * discussion (prefill_layer's single-call-only restriction and this
 * class's fallback strategy for prompts longer than max_prefill_batch;
 * one engine owns one sequence's GpuState, not safe for concurrent
 * generate() calls against the same instance).
 *
 * TRANSLATION NOTE: the CUDA port's {@code processPrompt} computes the
 * "last row" pointer for the overflow-free case via raw pointer
 * arithmetic ({@code xBatch + (prefillLen-1)*dim*4}). Here that becomes
 * {@link MetalBuffer#withOffset(long)} -- see that class's javadoc for
 * why a bare {@code long} can't represent a Metal buffer + offset the
 * way a CUdeviceptr can.
 *
 * UNVERIFIED, same standing caveat as the rest of this codebase: no
 * Metal GPU/toolchain was available while writing this.
 */
public final class LlamaMetalEngine implements AutoCloseable {

    private final GpuContext ctx;
    private final LlamaLayer.Config cfg;
    private final LlamaLayer.GpuWeights[] layers;
    private final LlamaLayer.GpuState[] states;
    private final EmbeddingTable embeddings;
    private final BpeTokenizer tokenizer;
    private final MetalBuffer finalNormGammaDevice;
    private final MetalBuffer lmHeadDevice;
    private final int vocabSize;

    private final MetalBuffer xSingle; // [dim] decode-path scratch, persistent for this engine's lifetime
    private final MetalBuffer xBatch;  // [max_prefill_batch, dim] prefill-path scratch

    public LlamaMetalEngine(
            GpuContext ctx,
            LlamaLayer.Config cfg,
            LlamaLayer.GpuWeights[] layers,
            LlamaLayer.GpuState[] states,
            EmbeddingTable embeddings,
            BpeTokenizer tokenizer,
            MetalBuffer finalNormGammaDevice,
            MetalBuffer lmHeadDevice,
            int vocabSize) throws Throwable {

        if (layers.length != states.length) {
            throw new IllegalArgumentException("layers.length (" + layers.length + ") != states.length (" + states.length + ")");
        }
        if (embeddings.dim() != cfg.dim) {
            throw new IllegalArgumentException("EmbeddingTable dim (" + embeddings.dim() + ") != Config.dim (" + cfg.dim + ")");
        }

        this.ctx = ctx;
        this.cfg = cfg;
        this.layers = layers;
        this.states = states;
        this.embeddings = embeddings;
        this.tokenizer = tokenizer;
        this.finalNormGammaDevice = finalNormGammaDevice;
        this.lmHeadDevice = lmHeadDevice;
        this.vocabSize = vocabSize;

        this.xSingle = LlamaLayer.allocFloats(ctx, cfg.dim);
        this.xBatch = (cfg.max_prefill_batch > 0)
                ? LlamaLayer.allocFloats(ctx, (long) cfg.max_prefill_batch * cfg.dim)
                : MetalBuffer.NULL;
    }

    public static final class GenerationConfig {
        public int maxNewTokens = 256;
        public Sampler.Config sampler = new Sampler.Config();
        /** -1 means "use tokenizer.getEosId()". */
        public int eosTokenId = -1;
    }

    /**
     * Runs the full prompt through the model, then samples up to
     * genCfg.maxNewTokens new tokens, stopping early on EOS. Returns the
     * DETOKENIZED generated continuation only (not the echoed prompt).
     */
    public String generate(String prompt, GenerationConfig genCfg) throws Throwable {
        int[] promptIds = tokenizer.encode(prompt, true);
        if (promptIds.length == 0) {
            throw new IllegalArgumentException("Prompt tokenized to zero tokens");
        }

        List<Integer> history = new ArrayList<>(promptIds.length + genCfg.maxNewTokens);
        for (int id : promptIds) {
            history.add(id);
        }

        MetalBuffer lastRow = processPrompt(promptIds);

        int eosId = genCfg.eosTokenId >= 0 ? genCfg.eosTokenId : tokenizer.getEosId();
        Sampler sampler = new Sampler(genCfg.sampler);
        List<Integer> generated = new ArrayList<>(genCfg.maxNewTokens);

        float[] logits = LlamaLayer.finalLogits(lastRow, finalNormGammaDevice, lmHeadDevice, cfg, ctx, vocabSize);
        int nextId = sampler.sample(logits, history);

        for (int step = 0; step < genCfg.maxNewTokens; step++) {
            if (nextId == eosId) {
                break;
            }
            generated.add(nextId);
            history.add(nextId);

            embeddings.embedRow(nextId, xSingle, ctx);
            for (int l = 0; l < layers.length; l++) {
                LlamaLayer.forward_layer(xSingle, layers[l], states[l], cfg, ctx);
            }
            logits = LlamaLayer.finalLogits(xSingle, finalNormGammaDevice, lmHeadDevice, cfg, ctx, vocabSize);
            nextId = sampler.sample(logits, history);
        }

        int[] generatedIds = new int[generated.size()];
        for (int i = 0; i < generatedIds.length; i++) {
            generatedIds[i] = generated.get(i);
        }
        return tokenizer.decode(generatedIds);
    }

    /**
     * Feeds the whole prompt through every layer -- batched prefill for
     * the first min(T, max_prefill_batch) tokens, then a one-at-a-time
     * forward_layer decode fallback for anything beyond that cap. Returns
     * a MetalBuffer pointing at the LAST processed position's final-layer
     * hidden state, ready for finalLogits.
     */
    private MetalBuffer processPrompt(int[] promptIds) throws Throwable {
        int T = promptIds.length;
        int prefillLen = Math.min(T, cfg.max_prefill_batch);

        if (prefillLen > 0) {
            int[] prefillIds = Arrays.copyOfRange(promptIds, 0, prefillLen);
            embeddings.embedRows(prefillIds, xBatch, ctx);
            for (int l = 0; l < layers.length; l++) {
                LlamaLayer.prefill_layer(xBatch, layers[l], states[l], cfg, ctx, 0, prefillLen);
            }
        }

        if (T == prefillLen) {
            // Whole prompt fit in the batched call -- last row lives in xBatch.
            return xBatch.withOffset((long) (prefillLen - 1) * cfg.dim * ValueLayout.JAVA_FLOAT.byteSize());
        }

        // Overflow fallback: remaining tokens, one at a time through the
        // decode path. Each iteration leaves its result in xSingle; the
        // LAST iteration's xSingle is what the caller needs.
        for (int i = prefillLen; i < T; i++) {
            embeddings.embedRow(promptIds[i], xSingle, ctx);
            for (int l = 0; l < layers.length; l++) {
                LlamaLayer.forward_layer(xSingle, layers[l], states[l], cfg, ctx);
            }
        }
        return xSingle;
    }

    @Override
    public void close() {
        LlamaLayer.freeQuietly(ctx, xSingle);
        LlamaLayer.freeQuietly(ctx, xBatch);
    }
}