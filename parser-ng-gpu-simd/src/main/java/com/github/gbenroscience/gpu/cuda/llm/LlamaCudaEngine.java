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
package com.github.gbenroscience.gpu.cuda.llm;

 

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author GBEMIRO
 * Top-level Llama-style CUDA inference engine: tokenizer -> embedding
 * lookup -> batched prefill (one call, capped at cfg.max_prefill_batch) ->
 * decode-tail fallback for any prompt overflow -> per-token decode +
 * sampling loop -> detokenize. Ties together every other file in this
 * package; this is the class an application actually calls.
 *
 * PROMPT LENGTH HANDLING: prefill_layer only supports a single batched
 * call (see its javadoc for why cross-chunk causal attention isn't
 * implemented). This engine handles prompts longer than
 * cfg.max_prefill_batch by batching the first max_prefill_batch tokens
 * (fast path) and then feeding the remainder through forward_layer one
 * token at a time (correct, just not batched-fast) -- exactly the
 * fallback prefill_layer's javadoc points callers toward. Both paths are
 * exercised by generate() automatically; the caller doesn't need to know
 * which one ran.
 *
 * THREADING: one LlamaCudaEngine owns one sequence's worth of GpuState
 * (KV cache + pos). It is NOT safe to call generate() concurrently from
 * multiple threads against the same engine instance -- construct one
 * engine (sharing the same GpuContext/GpuWeights across engines is
 * fine and intended; GpuState is the only per-sequence part) per
 * concurrent conversation.
 *
 * UNVERIFIED, same standing caveat as the rest of this codebase: no CUDA
 * GPU, driver, or NVRTC toolchain were available while writing this.
 */
public final class LlamaCudaEngine implements AutoCloseable {

    private final GpuContext ctx;
    private final LlamaLayer.Config cfg;
    private final LlamaLayer.GpuWeights[] layers;
    private final LlamaLayer.GpuState[] states;
    private final EmbeddingTable embeddings;
    private final BpeTokenizer tokenizer;
    private final long finalNormGammaDevice;
    private final long lmHeadDevice;
    private final int vocabSize;

    private final long xSingle; // [dim] decode-path scratch, persistent for this engine's lifetime
    private final long xBatch;  // [max_prefill_batch, dim] prefill-path scratch

    public LlamaCudaEngine(
            GpuContext ctx,
            LlamaLayer.Config cfg,
            LlamaLayer.GpuWeights[] layers,
            LlamaLayer.GpuState[] states,
            EmbeddingTable embeddings,
            BpeTokenizer tokenizer,
            long finalNormGammaDevice,
            long lmHeadDevice,
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

        this.xSingle = allocFloats(ctx, cfg.dim);
        this.xBatch = (cfg.max_prefill_batch > 0)
                ? allocFloats(ctx, (long) cfg.max_prefill_batch * cfg.dim)
                : 0L;
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

        long lastRow = processPrompt(promptIds);

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
     * forward_layer decode fallback for anything beyond that cap (see
     * class javadoc). Returns a device pointer to the LAST processed
     * position's final-layer hidden state, ready for finalLogits.
     */
    private long processPrompt(int[] promptIds) throws Throwable {
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
            return xBatch + (long) (prefillLen - 1) * cfg.dim * ValueLayout.JAVA_FLOAT.byteSize();
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

    public static long allocFloats(GpuContext ctx, long count) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
            GpuContext.check((int) ctx.cu.cuMemAlloc.invoke(ptrBuf, count * ValueLayout.JAVA_FLOAT.byteSize()),
                    "cuMemAlloc(engine scratch)");
            return ptrBuf.get(ValueLayout.JAVA_LONG, 0);
        }
    }

    @Override
    public void close() {
        try {
            if (xSingle != 0L) {
                ctx.cu.cuMemFree.invoke(xSingle);
            }
            if (xBatch != 0L) {
                ctx.cu.cuMemFree.invoke(xBatch);
            }
        } catch (Throwable t) {
            // best-effort cleanup
        }
    }
}