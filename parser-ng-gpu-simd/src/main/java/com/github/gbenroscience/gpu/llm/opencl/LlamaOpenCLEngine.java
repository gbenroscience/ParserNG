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
    private final boolean lmHeadQ8;
    private final LlamaLayer.LogitsScratch logitsScratch;
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
        this(ctx, cfg, layers, states, embeddings, tokenizer, finalNormGammaDevice, lmHeadDevice, false, vocabSize);
    }

    /**
     * @param lmHeadQ8 true when {@code lmHeadDevice} holds the output projection as Q8_0 blocks (4x less memory
     *                 traffic per token than the F32 form); false for a row-major F32 matrix.
     */
    public LlamaOpenCLEngine(
            GpuContext ctx,
            LlamaLayer.Config cfg,
            LlamaLayer.GpuWeights[] layers,
            LlamaLayer.GpuState[] states,
            EmbeddingTable embeddings,
            BpeTokenizer tokenizer,
            MemorySegment finalNormGammaDevice,
            MemorySegment lmHeadDevice,
            boolean lmHeadQ8,
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
        this.lmHeadQ8 = lmHeadQ8;
        this.vocabSize = vocabSize;

        ctx.allocCategory = "engine scratch";
        this.xSingle = LlamaLayer.allocFloats(ctx, cfg.dim);
        this.xBatch = (cfg.max_prefill_batch > 0)
                ? LlamaLayer.allocFloats(ctx, (long) cfg.max_prefill_batch * cfg.dim)
                : null;
        this.lastRow = LlamaLayer.allocFloats(ctx, cfg.dim);
        LlamaLayer.LogitsScratch scratch;
        try {
            scratch = new LlamaLayer.LogitsScratch(ctx, cfg, vocabSize);
        } catch (Throwable t) {
            close();
            throw t;
        }
        this.logitsScratch = scratch;
    }

    /** Timings of the most recent generate() call (wall clock, milliseconds). */
    public static final class RunStats {

        public int promptTokens;
        public int decodeForwards;
        public double tokenizeMs;
        /** Prompt processing plus the first logits readback and sample (= time to first token, excluding tokenization). */
        public double prefillMs;
        public double decodeMs;
        /** Host wall time inside the decode loop, split by phase (their sum is approximately decodeMs). */
        public double embedMs, layerSubmitMs, logitsWaitMs, sampleMs;
    }

    private RunStats lastRunStats = new RunStats();

    public RunStats lastRunStats() {
        return lastRunStats;
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
        final long tTok0 = System.nanoTime();
        int[] promptIds = tokenizer.encode(prompt, true);
        final double tokenizeMs = (System.nanoTime() - tTok0) / 1e6;
        if (promptIds.length == 0) {
            throw new IllegalArgumentException("Prompt tokenized to zero tokens");
        }
        if (genCfg.maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must be >= 0, got " + genCfg.maxNewTokens);
        }
        // Positions written to the KV cache / RoPE tables: the prompt plus every generated token that is
        // fed back (the last sampled token is never forwarded). Exceeding max_seq would index past the
        // device KV cache, RoPE tables and score buffer, so refuse BEFORE touching the device.
        long positionsNeeded = (long) promptIds.length + Math.max(0, genCfg.maxNewTokens - 1);
        if (positionsNeeded > cfg.max_seq) {
            throw new IllegalArgumentException("Prompt (" + promptIds.length + " tokens) + maxNewTokens (" + genCfg.maxNewTokens
                    + ") needs " + positionsNeeded + " KV positions but the model context is " + cfg.max_seq);
        }
        // Every generate() starts a fresh sequence: reset the KV position of every layer explicitly rather
        // than relying on prefill_layer to overwrite it (it is skipped when max_prefill_batch == 0).
        for (LlamaLayer.GpuState layerState : states) {
            layerState.pos = 0;
        }
        if (genCfg.maxNewTokens == 0) {
            return "";
        }

        List<Integer> history = new ArrayList<>(promptIds.length + genCfg.maxNewTokens);
        for (int id : promptIds) {
            history.add(id);
        }

        final DeviceProfiler prof = ctx.prof;
        final boolean timing = Boolean.getBoolean("llm.timing") || prof != null;
        final int prefillLenForStats = Math.min(promptIds.length, cfg.max_prefill_batch);
        final int prefillPasses = (prefillLenForStats > 0 ? 1 : 0) + (promptIds.length - prefillLenForStats);
        if (prof != null) {
            prof.phase = "prefill";
            prof.layer = -1;
        }
        final long tStart = System.nanoTime();
        MemorySegment lastRowResult = processPrompt(promptIds);

        int eosId = genCfg.eosTokenId >= 0 ? genCfg.eosTokenId : tokenizer.getEosId();
        Sampler sampler = new Sampler(genCfg.sampler);
        List<Integer> generated = new ArrayList<>(genCfg.maxNewTokens);

        if (prof != null) {
            prof.layer = -1;
        }
        float[] logits = LlamaLayer.finalLogits(logitsScratch, lmHeadQ8, lastRowResult, finalNormGammaDevice, lmHeadDevice, cfg, ctx);
        if (prof != null) {
            prof.flush();
        }
        int nextId = sampler.sample(logits, history);
        final long tFirstToken = System.nanoTime();
        int decodeForwards = 0;
        long nsEmbed = 0, nsSubmit = 0, nsLogits = 0, nsSample = 0;
        if (prof != null) {
            prof.phase = "decode";
        }

        for (int step = 0; step < genCfg.maxNewTokens; step++) {
            if (nextId == eosId) {
                break;
            }
            generated.add(nextId);
            history.add(nextId);
            if (step == genCfg.maxNewTokens - 1) {
                break; // the final token needs no forward pass: its successor would never be used
            }

            final long h0 = System.nanoTime();
            if (prof != null) {
                prof.layer = -1;
            }
            LlamaLayer.stage(ctx, "embedding upload");
            embeddings.embedRowAsync(nextId, xSingle, ctx); // non-blocking: the logits read below is the only sync of the step
            final long h1 = System.nanoTime();
            for (int l = 0; l < layers.length; l++) {
                if (prof != null) {
                    prof.layer = l;
                }
                LlamaLayer.forward_layer(xSingle, layers[l], states[l], cfg, ctx);
            }
            final long h2 = System.nanoTime();
            if (prof != null) {
                prof.layer = -1;
            }
            logits = LlamaLayer.finalLogits(logitsScratch, lmHeadQ8, xSingle, finalNormGammaDevice, lmHeadDevice, cfg, ctx);
            final long h3 = System.nanoTime();
            if (prof != null) {
                prof.flush(); // queue is idle after the blocking logits read
            }
            final long h3b = System.nanoTime();
            nextId = sampler.sample(logits, history);
            final long h4 = System.nanoTime();
            nsEmbed += h1 - h0;
            nsSubmit += h2 - h1;
            nsLogits += h3 - h2;
            nsSample += h4 - h3b;
            decodeForwards++;
        }
        final long tEnd = System.nanoTime();
        RunStats st = new RunStats();
        st.promptTokens = promptIds.length;
        st.decodeForwards = decodeForwards;
        st.tokenizeMs = tokenizeMs;
        st.prefillMs = (tFirstToken - tStart) / 1e6;
        st.decodeMs = (tEnd - tFirstToken) / 1e6;
        st.embedMs = nsEmbed / 1e6;
        st.layerSubmitMs = nsSubmit / 1e6;
        st.logitsWaitMs = nsLogits / 1e6;
        st.sampleMs = nsSample / 1e6;
        lastRunStats = st;
        if (timing) {
            System.out.printf("[llm.timing] tokenize=%.2f ms; prompt=%d tokens, time-to-first-token=%.1f ms (%.2f tok/s prefill); decode=%d tokens in %.1f ms = %.3f tok/s (%.1f ms/token)%n",
                    st.tokenizeMs, st.promptTokens, st.prefillMs, st.promptTokens * 1000.0 / Math.max(st.prefillMs, 1e-9),
                    st.decodeForwards, st.decodeMs, st.decodeForwards * 1000.0 / Math.max(st.decodeMs, 1e-9), st.decodeMs / Math.max(st.decodeForwards, 1));
            if (decodeForwards > 0) {
                System.out.printf("[llm.timing] decode host wall per token: embed=%.2f ms, enqueue all layers=%.2f ms, logits+readback wait=%.2f ms, sample=%.2f ms%n",
                        st.embedMs / decodeForwards, st.layerSubmitMs / decodeForwards, st.logitsWaitMs / decodeForwards, st.sampleMs / decodeForwards);
            }
            System.out.printf("[llm.timing] device memory live %.1f MB (peak %.1f MB)%n", ctx.allocations.liveBytes() / 1e6, ctx.allocations.peakBytes() / 1e6);
            if (prof != null) {
                System.out.print(prof.report(java.util.Map.of("prefill", prefillPasses, "decode", decodeForwards)));
            }
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
            LlamaLayer.stage(ctx, "embedding upload");
            embeddings.embedRows(prefillIds, xBatch, ctx);
            for (int l = 0; l < layers.length; l++) {
                if (ctx.prof != null) {
                    ctx.prof.layer = l;
                }
                LlamaLayer.prefill_layer(xBatch, layers[l], states[l], cfg, ctx, 0, prefillLen);
            }
            if (ctx.prof != null) {
                ctx.prof.layer = -1;
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
            LlamaLayer.stage(ctx, "embedding upload");
            embeddings.embedRow(promptIds[i], xSingle, ctx);
            for (int l = 0; l < layers.length; l++) {
                if (ctx.prof != null) {
                    ctx.prof.layer = l;
                }
                LlamaLayer.forward_layer(xSingle, layers[l], states[l], cfg, ctx);
            }
            if (ctx.prof != null) {
                ctx.prof.layer = -1;
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
        if (logitsScratch != null) {
            logitsScratch.free(ctx);
        }
    }
}