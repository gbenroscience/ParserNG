package com.github.gbenroscience.gpu.llm.opencl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * OpenCL counterpart of {@code com.github.gbenroscience.gpu.llm.cuda.LlamaCudaEngine}.
 * Same top-level shape: tokenizer -> embedding lookup -> batched prefill
 * (capped at cfg.max_prefill_batch) -> decode-tail fallback for prompt
 * overflow -> per-token decode + sampling loop -> detokenize. See the CUDA
 * original's class javadoc for the algorithmic rationale (prompt-length
 * handling, threading model) -- unchanged here.
 *
 * THE ONE REAL DIFFERENCE: processPrompt. The CUDA original returns a
 * device pointer INTO the middle of xBatch ({@code xBatch + (T-1)*dim*sizeof(float)})
 * when the whole prompt fit in one batched call, and feeds that offset
 * pointer straight into finalLogits. OpenCL's cl_mem has no such pointer
 * arithmetic (see LlamaLayer's class javadoc) -- so this port instead
 * extracts that one row into a dedicated [dim]-sized scratch buffer via
 * clEnqueueCopyBuffer (a single small device-to-device copy, enqueued on
 * the same in-order queue as everything else, so no extra host sync is
 * needed) and passes THAT to finalLogits. This copy happens once per
 * generate() call (for the prompt's last token), never per decode step,
 * so it costs nothing measurable against the GEMV/GEMM work around it.
 *
 * THREADING: same contract as the CUDA original -- one LlamaOpenCLEngine
 * owns one sequence's GpuState; not safe to call generate() concurrently
 * against the same instance. Sharing one GpuContext/GpuWeights[] across
 * several engines (several concurrent conversations) is fine and intended.
 *
 * UNVERIFIED: no OpenCL platform/device was available while writing this
 * port -- see GpuContext's class javadoc for the same standing caveat.
 */
public final class LlamaOpenCLEngine implements AutoCloseable {

    private final GpuContext ctx;
    private final LlamaLayer.Config cfg;
    private final LlamaLayer.GpuWeights[] layers;
    private final LlamaLayer.GpuState[] states;
    private final EmbeddingTable embeddings;
    private final BpeTokenizer tokenizer;
    private final MemorySegment finalNormGammaDevice;
    private final MemorySegment lmHeadDevice;
    private final int vocabSize;

    private final MemorySegment xSingle;  // [dim] decode-path scratch, persistent for this engine's lifetime
    private final MemorySegment xBatch;   // [max_prefill_batch, dim] prefill-path scratch
    private final MemorySegment lastRow;  // [dim] scratch used only to extract xBatch's final row for finalLogits -- see class javadoc

    public LlamaOpenCLEngine(
            GpuContext ctx,
            LlamaLayer.Config cfg,
            LlamaLayer.GpuWeights[] layers,
            LlamaLayer.GpuState[] states,
            EmbeddingTable embeddings,
            BpeTokenizer tokenizer,
            MemorySegment finalNormGammaDevice,
            MemorySegment lmHeadDevice,
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
                : null;
        this.lastRow = LlamaLayer.allocFloats(ctx, cfg.dim);
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

        MemorySegment lastRowResult = processPrompt(promptIds);

        int eosId = genCfg.eosTokenId >= 0 ? genCfg.eosTokenId : tokenizer.getEosId();
        Sampler sampler = new Sampler(genCfg.sampler);
        List<Integer> generated = new ArrayList<>(genCfg.maxNewTokens);

        float[] logits = LlamaLayer.finalLogits(lastRowResult, finalNormGammaDevice, lmHeadDevice, cfg, ctx, vocabSize);
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
     * class javadoc). Returns a [dim]-sized device buffer holding the
     * LAST processed position's final-layer hidden state, ready for
     * finalLogits.
     */
    private MemorySegment processPrompt(int[] promptIds) throws Throwable {
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
            // Whole prompt fit in the batched call -- last row lives in
            // xBatch at row (prefillLen-1). Extract it into the dedicated
            // lastRow buffer (device-to-device copy, no host round trip;
            // see class javadoc for why this replaces the CUDA original's
            // raw pointer-plus-offset trick).
            long srcOffsetBytes = (long) (prefillLen - 1) * cfg.dim * ValueLayout.JAVA_FLOAT.byteSize();
            long byteSize = (long) cfg.dim * ValueLayout.JAVA_FLOAT.byteSize();
            synchronized (ctx.dispatchLock) {
                GpuContext.check((int) ctx.cl.clEnqueueCopyBuffer.invoke(
                        ctx.queue, xBatch, lastRow, srcOffsetBytes, 0L, byteSize, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueCopyBuffer(extract last prefill row)");
            }
            return lastRow;
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
        if (xBatch != null) {
            LlamaLayer.freeQuietly(ctx, xBatch);
        }
        LlamaLayer.freeQuietly(ctx, lastRow);
    }
}