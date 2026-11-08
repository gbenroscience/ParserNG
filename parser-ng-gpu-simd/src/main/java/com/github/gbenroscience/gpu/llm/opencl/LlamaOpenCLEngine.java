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
    /** Queue hand-off interval during decode, in layers (0 = off); see {@link Variants#flushEvery()}. */
    private final int flushEvery = Variants.flushEvery();
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
        /** Tokens generated by the last call. */
        public int generatedTokens;
        /** The ids of those tokens (a stop token is not included). */
        public int[] generatedIds = new int[0];
        /** Why the last call ended. */
        public com.github.gbenroscience.gpu.llm.StopReason stopReason = com.github.gbenroscience.gpu.llm.StopReason.MAX_TOKENS;
        /** Streaming only: milliseconds from the start of the call (before the prompt) to the first piece handed to the listener; -1 if none was. */
        public double firstPieceMs = -1;
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
        return generate(prompt, genCfg, null);
    }

    /** As {@link #generate(String, GenerationConfig)}, delivering text to {@code listener} as it is produced (null = none). */
    public String generate(String prompt, GenerationConfig genCfg, com.github.gbenroscience.gpu.llm.TokenListener listener) throws Throwable {
        final long tTok0 = System.nanoTime();
        int[] promptIds = tokenizer.encode(prompt, true);
        final double tokenizeMs = (System.nanoTime() - tTok0) / 1e6;
        int eosId = genCfg.eosTokenId >= 0 ? genCfg.eosTokenId : tokenizer.getEosId();
        return generateIds(promptIds, tokenizeMs, genCfg, new int[]{eosId}, listener);
    }

    /**
     * Applies the Llama-3 chat template to {@code messages} and generates the assistant's reply, stopping at any
     * end-of-generation token ({@code <|eot_id|>}, {@code <|eom_id|>}, {@code <|end_of_text|>} or the metadata EOS; an
     * explicit {@code genCfg.eosTokenId} is added to that set). Throws {@link UnsupportedOperationException} if the
     * model's vocabulary is not a Llama-3 one. Every call starts a fresh sequence: pass the whole conversation.
     */
    public String chat(List<com.github.gbenroscience.gpu.llm.core.ChatTemplate.Message> messages, GenerationConfig genCfg) throws Throwable {
        return chat(messages, genCfg, null);
    }

    /** As {@link #chat(List, GenerationConfig)}, delivering the reply to {@code listener} as it is produced (null = none). */
    public String chat(List<com.github.gbenroscience.gpu.llm.core.ChatTemplate.Message> messages, GenerationConfig genCfg,
            com.github.gbenroscience.gpu.llm.TokenListener listener) throws Throwable {
        final long tTok0 = System.nanoTime();
        int[] promptIds = com.github.gbenroscience.gpu.llm.core.ChatTemplate.llama3(tokenizer, messages);
        final double tokenizeMs = (System.nanoTime() - tTok0) / 1e6;
        int[] stops = com.github.gbenroscience.gpu.llm.core.ChatTemplate.stopIds(tokenizer);
        if (genCfg.eosTokenId >= 0) {
            stops = Arrays.copyOf(stops, stops.length + 1);
            stops[stops.length - 1] = genCfg.eosTokenId;
        }
        return generateIds(promptIds, tokenizeMs, genCfg, stops, listener);
    }

    /** True when {@link #chat} can be used with this model's vocabulary. */
    public boolean supportsChat() {
        return com.github.gbenroscience.gpu.llm.core.ChatTemplate.supportsLlama3(tokenizer);
    }

    private static boolean isStop(int id, int[] stopIds) {
        for (int s : stopIds) {
            if (s >= 0 && s == id) {
                return true;
            }
        }
        return false;
    }

    /** True while a generation is running; a listener that calls back into the same engine would corrupt its state. */
    private boolean busy;

    private String generateIds(int[] promptIds, double tokenizeMs, GenerationConfig genCfg, int[] stopIds,
            com.github.gbenroscience.gpu.llm.TokenListener listener) throws Throwable {
        if (busy) {
            throw new IllegalStateException("generate()/chat() was called from inside a streaming callback of the same engine");
        }
        busy = true;
        try {
            return generateIdsUnguarded(promptIds, tokenizeMs, genCfg, stopIds, listener);
        } finally {
            busy = false;
        }
    }

    private String generateIdsUnguarded(int[] promptIds, double tokenizeMs, GenerationConfig genCfg, int[] stopIds,
            com.github.gbenroscience.gpu.llm.TokenListener listener) throws Throwable {
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
            RunStats none = new RunStats();
            none.promptTokens = promptIds.length;
            lastRunStats = none;
            if (listener != null) {
                listener.onComplete(com.github.gbenroscience.gpu.llm.StopReason.MAX_TOKENS, 0);
            }
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

        com.github.gbenroscience.gpu.llm.StopReason stopReason = com.github.gbenroscience.gpu.llm.StopReason.MAX_TOKENS;
        final BpeTokenizer.StreamDecoder streamer = listener != null ? tokenizer.newStreamDecoder() : null;
        final StringBuilder delivered = listener != null ? new StringBuilder() : null;
        double firstPieceMs = -1;

        for (int step = 0; step < genCfg.maxNewTokens; step++) {
            if (isStop(nextId, stopIds)) {
                stopReason = com.github.gbenroscience.gpu.llm.StopReason.STOP_TOKEN;
                break;
            }
            generated.add(nextId);
            history.add(nextId);
            // decoded now, delivered after the next step has been submitted so the callback overlaps device compute
            final String piece = streamer != null ? streamer.push(nextId) : "";
            if (step == genCfg.maxNewTokens - 1) {
                // the final token needs no forward pass: its successor would never be used
                if (!piece.isEmpty()) {
                    if (firstPieceMs < 0) {
                        firstPieceMs = (System.nanoTime() - tStart) / 1e6;
                    }
                    delivered.append(piece);
                    if (!listener.onText(piece)) {
                        stopReason = com.github.gbenroscience.gpu.llm.StopReason.CANCELLED;
                    }
                }
                break;
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
                if (flushEvery > 0 && (l + 1) % flushEvery == 0 && l + 1 < layers.length) {
                    LlamaLayer.flushQueue(ctx); // let the device start while the host enqueues the remaining layers
                }
            }
            final long h2 = System.nanoTime();
            if (!piece.isEmpty()) {
                // The step for this token is already enqueued: hand the queue to the device, then run the callback while it computes.
                if (firstPieceMs < 0) {
                    firstPieceMs = (System.nanoTime() - tStart) / 1e6;
                }
                delivered.append(piece);
                boolean keepGoing;
                try {
                    LlamaLayer.flushQueue(ctx);
                    keepGoing = listener.onText(piece);
                } catch (Throwable t) {
                    drainQuietly(t);
                    throw t;
                }
                if (!keepGoing) {
                    LlamaLayer.finish(ctx); // leave no work in flight; the engine is reusable straight away
                    stopReason = com.github.gbenroscience.gpu.llm.StopReason.CANCELLED;
                    break;
                }
            }
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
        if (streamer != null && stopReason != com.github.gbenroscience.gpu.llm.StopReason.CANCELLED) {
            String tail = streamer.finish(); // a character cut off by the end of the generation
            if (!tail.isEmpty()) {
                if (firstPieceMs < 0) {
                    firstPieceMs = (System.nanoTime() - tStart) / 1e6;
                }
                delivered.append(tail);
                if (!listener.onText(tail)) {
                    stopReason = com.github.gbenroscience.gpu.llm.StopReason.CANCELLED;
                }
            }
        }
        RunStats st = new RunStats();
        st.generatedTokens = generated.size();
        st.generatedIds = generated.stream().mapToInt(Integer::intValue).toArray();
        st.stopReason = stopReason;
        st.firstPieceMs = firstPieceMs;
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
        if (listener != null) {
            listener.onComplete(stopReason, generated.size());
        }
        // cancelled: exactly what the listener has been given; otherwise the whole text (the same string as without a listener)
        return stopReason == com.github.gbenroscience.gpu.llm.StopReason.CANCELLED && delivered != null
                ? delivered.toString() : tokenizer.decode(generatedIds);
    }

    /** Waits for in-flight device work after a failed callback so the failure does not leave the queue running; never throws. */
    private void drainQuietly(Throwable cause) {
        try {
            LlamaLayer.finish(ctx);
        } catch (Throwable t) {
            cause.addSuppressed(t);
        }
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