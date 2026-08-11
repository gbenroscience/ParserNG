package com.github.gbenroscience.gpu.llm.cuda;

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.NoSuchElementException;

/**
 * CUDA counterpart of {@code com.github.gbenroscience.gpu.llm.GpuLlamaLayer}
 * -- v2. Adds a batched prefill path (prefill_layer) alongside the
 * original single-token forward_layer, activation-type selection
 * (SwiGLU / GeLU / GeGLU), and a corrected launch/sync discipline. See
 * KernelSource's class javadoc for what "batched" means here and
 * what it deliberately isn't (not cuBLAS-tier tiling, not fused
 * attention).
 *
 * SYNC DISCIPLINE, CHANGED FROM v1: v1 called cuCtxSynchronize after
 * EVERY kernel launch, mirroring the OpenCL version's per-kernel
 * clFinish. That's unnecessary and was the single biggest throughput
 * problem in this port. Kernels launched on the same stream (here: the
 * default stream, MemorySegment.NULL, for every launch in this file)
 * already execute in issued order without host intervention -- the
 * driver enforces that. The synchronous driver-API memcpy calls
 * (cuMemcpyHtoD_v2/cuMemcpyDtoH_v2, i.e. cu.cuMemcpyHtoD/cuMemcpyDtoH)
 * are themselves ordered against that same default stream and block
 * until complete, so they act as the synchronization point wherever the
 * HOST actually needs a result: the RMSNorm partial-sum readback, the
 * KV-cache write round-trip, and the final logits readback. Everywhere
 * else, kernels are simply launched back to back with no intervening
 * sync. A single explicit cuCtxSynchronize is kept at the very end of
 * forward_layer/prefill_layer -- not required for correctness, kept for
 * predictable per-token/per-batch timing if the caller profiles calls to
 * these methods.
 *
 * NOT fixed here (still future work, flagged rather than silently
 * dropped): attention remains three unfused launches per head; there is
 * still no multi-stream pipelining (everything is one context, one
 * implicit stream); the prefill GEMMs are not shared-memory tiled.
 *
 * UNVERIFIED, same standing caveat as the rest of this codebase: no CUDA
 * GPU, driver, or NVRTC toolchain were available while writing this.
 */
public final class LlamaLayer {

    private LlamaLayer() {
    }

    /** Which FFN activation this layer's weights expect. */
    public enum ActivationType {
        /** out = gate * sigmoid(gate) * up. Llama/Mistral/Qwen-family default. */
        SWIGLU,
        /**
         * out = gelu(gate). UNGATED -- the "up" projection is still
         * computed by GpuWeights (fromGguf always loads ffn_up.weight)
         * but is NOT read by this activation; only ffn_gate.weight's
         * projection feeds gelu_activate. If your GGUF genuinely has no
         * separate gate tensor for a true ungated model, map your single
         * hidden-projection tensor to ffn_gate.weight when building
         * GpuWeights -- GpuWeights.fromGguf's tensor-name lookups are
         * unchanged from v1 and still require both attn_gate.weight and
         * attn_up.weight to exist in the GGUF file.
         */
        GELU,
        /** out = gelu(gate) * up. Same gated shape as SWIGLU, GeLU gate instead of SiLU. */
        GEGLU
    }

    /** Mirrors GpuLlamaLayer.Config / LlamaLayerInt8.Config, plus activationType and prefill batch sizing. */
    public static final class Config {
        public int n_layers = 32;
        public int dim = 4096;
        public int hidden_dim = 11008;
        public int num_heads = 32;
        public int kv_heads = 32;
        public int head_dim = 128;
        public int max_seq = 4096;
        public double norm_eps = 1e-6;
        public double rope_theta = 10000.0;
        /** Llama-3/3.1/3.2-style RoPE frequency scaling -- see com.github.gbenroscience.gpu.llm.opencl.LlamaLayer's Config for the full javadoc (identical here). Defaults are all no-ops. */
        public double rope_scaling_factor = 1.0;
        public double rope_scaling_low_freq_factor = 1.0;
        public double rope_scaling_high_freq_factor = 4.0;
        public double rope_scaling_orig_context_length = 8192.0;

        public ActivationType activationType = ActivationType.SWIGLU;

        /**
         * Upper bound on prefill batch size (T) this layer's GpuState
         * will allocate scratch buffers for -- determines device memory
         * committed to the batched-prefill path up front. 0 disables
         * prefill entirely (prefill_layer throws if called); GpuState
         * skips allocating the batch buffers in that case, saving the
         * memory for decode-only use.
         */
        public int max_prefill_batch = 512;

        public void validate() {
            if (dim % 32 != 0) {
                throw new IllegalArgumentException("dim must be a multiple of 32 (Q8_0 block size), got " + dim);
            }
            if (hidden_dim % 32 != 0) {
                throw new IllegalArgumentException("hidden_dim must be a multiple of 32 (Q8_0 block size), got " + hidden_dim);
            }
            if (head_dim % 2 != 0) {
                throw new IllegalArgumentException("head_dim must be even, got " + head_dim);
            }
            if (num_heads % kv_heads != 0) {
                throw new IllegalArgumentException(
                        "num_heads (" + num_heads + ") must be an exact multiple of kv_heads (" + kv_heads + ") for GQA grouping");
            }
            if (max_prefill_batch < 0) {
                throw new IllegalArgumentException("max_prefill_batch must be >= 0, got " + max_prefill_batch);
            }
            if (n_layers <= 0) {
                throw new IllegalArgumentException("n_layers must be > 0, got " + n_layers);
            }
        }

        /**
         * Reads every architecture-derived field from a loaded GGUF
         * file's metadata section (GGUFFile.metadata, distinct from
         * GGUFFile.tensors -- see BpeTokenizer.fromGguf for the same
         * distinction on the tokenizer side). GGUF's convention is that
         * architecture-specific keys are prefixed by whatever string
         * "general.architecture" holds (e.g. "llama", "qwen2", "mistral"
         * all use "{arch}.attention.head_count" etc. under their own
         * arch prefix) -- this reads that prefix first, then everything
         * else relative to it:
         *
         *   n_layers    <- {arch}.block_count
         *   dim         <- {arch}.embedding_length
         *   hidden_dim  <- {arch}.feed_forward_length
         *   num_heads   <- {arch}.attention.head_count
         *   kv_heads    <- {arch}.attention.head_count_kv  (falls back to num_heads -- absent means no GQA, every model has this key or doesn't use GQA)
         *   head_dim    <- {arch}.attention.key_length      (falls back to dim/num_heads -- not every GGUF file writes this key explicitly; the fallback is the standard relationship whenever it's absent)
         *   max_seq     <- {arch}.context_length
         *   norm_eps    <- {arch}.attention.layer_norm_rms_epsilon
         *   rope_theta  <- {arch}.rope.freq_base
         *
         * NOT read from metadata, deliberately -- these aren't
         * architecture facts, they're inference-time/engine choices, and
         * GGUF has no reliable key for either:
         *   - activationType: stays at its class default (SWIGLU). GGUF
         *     doesn't carry an explicit "FFN activation function" key in
         *     general use -- SwiGLU is simply what "general.architecture"
         *     values like "llama"/"mistral"/"qwen2" imply by convention,
         *     not something written down. If your model uses GeLU/GeGLU
         *     (e.g. some GPT-2/GPT-NeoX-family GGUF files), set
         *     cfg.activationType AFTER calling this method.
         *   - max_prefill_batch: stays at its class default (512). This
         *     is a memory/throughput tradeoff for YOUR deployment, not a
         *     property of the model file.
         *
         * Calls validate() before returning, so a malformed or
         * unexpected GGUF file (e.g. head_count_kv that doesn't evenly
         * divide head_count) fails here with a clear message rather than
         * later inside GpuState construction.
         */
        public static Config fromGguf(GGUFLoader.GGUFFile gguf) {
            Config cfg = new Config();

            String arch = stringMetadata(gguf, "general.architecture", "llama");

            cfg.n_layers = intMetadata(gguf, arch + ".block_count", cfg.n_layers);
            cfg.dim = intMetadata(gguf, arch + ".embedding_length", cfg.dim);
            cfg.hidden_dim = intMetadata(gguf, arch + ".feed_forward_length", cfg.hidden_dim);
            cfg.num_heads = intMetadata(gguf, arch + ".attention.head_count", cfg.num_heads);
            cfg.kv_heads = intMetadata(gguf, arch + ".attention.head_count_kv", cfg.num_heads);
            cfg.max_seq = intMetadata(gguf, arch + ".context_length", cfg.max_seq);
            cfg.norm_eps = doubleMetadata(gguf, arch + ".attention.layer_norm_rms_epsilon", cfg.norm_eps);
            cfg.rope_theta = doubleMetadata(gguf, arch + ".rope.freq_base", cfg.rope_theta);
            cfg.rope_scaling_factor = doubleMetadata(gguf, arch + ".rope.scaling.factor", cfg.rope_scaling_factor);
            cfg.rope_scaling_low_freq_factor = doubleMetadata(gguf, arch + ".rope.scaling.low_freq_factor", cfg.rope_scaling_low_freq_factor);
            cfg.rope_scaling_high_freq_factor = doubleMetadata(gguf, arch + ".rope.scaling.high_freq_factor", cfg.rope_scaling_high_freq_factor);
            cfg.rope_scaling_orig_context_length = doubleMetadata(gguf, arch + ".rope.scaling.original_context_length", cfg.rope_scaling_orig_context_length);

            int keyLength = intMetadata(gguf, arch + ".attention.key_length", -1);
            cfg.head_dim = (keyLength > 0) ? keyLength : (cfg.dim / Math.max(cfg.num_heads, 1));

            cfg.validate();
            return cfg;
        }

        private static String stringMetadata(GGUFLoader.GGUFFile gguf, String key, String defaultValue) {
            Object v = gguf.metadata.get(key);
            return (v != null) ? String.valueOf(v) : defaultValue;
        }

        private static int intMetadata(GGUFLoader.GGUFFile gguf, String key, int defaultValue) {
            Object v = gguf.metadata.get(key);
            return (v instanceof Number) ? ((Number) v).intValue() : defaultValue;
        }

        private static double doubleMetadata(GGUFLoader.GGUFFile gguf, String key, double defaultValue) {
            Object v = gguf.metadata.get(key);
            return (v instanceof Number) ? ((Number) v).doubleValue() : defaultValue;
        }
    }

    /** GPU-resident weights for ONE decoder layer -- unchanged from v1. */
    public static final class GpuWeights implements AutoCloseable {
        final GpuContext ctx;
        final long wq_q8_0, wk_q8_0, wv_q8_0;
        final long wo_f32;
        final long w_gate_q8_0, w_up_q8_0, w_down_q8_0;
        final long attn_norm_gamma, ffn_norm_gamma;

        public GpuWeights(GpuContext ctx,
                byte[] wqQ8_0, byte[] wkQ8_0, byte[] wvQ8_0,
                float[] woF32,
                byte[] wGateQ8_0, byte[] wUpQ8_0, byte[] wDownQ8_0,
                float[] attnNormGamma, float[] ffnNormGamma) throws Throwable {
            this.ctx = ctx;
            this.wq_q8_0 = uploadBytes(ctx, wqQ8_0);
            this.wk_q8_0 = uploadBytes(ctx, wkQ8_0);
            this.wv_q8_0 = uploadBytes(ctx, wvQ8_0);
            this.wo_f32 = uploadFloats(ctx, woF32);
            this.w_gate_q8_0 = uploadBytes(ctx, wGateQ8_0);
            this.w_up_q8_0 = uploadBytes(ctx, wUpQ8_0);
            this.w_down_q8_0 = uploadBytes(ctx, wDownQ8_0);
            this.attn_norm_gamma = uploadFloats(ctx, attnNormGamma);
            this.ffn_norm_gamma = uploadFloats(ctx, ffnNormGamma);
        }

        public static GpuWeights fromGguf(GpuContext ctx, GGUFLoader.GGUFFile gguf, String layerPrefix) throws Throwable {
            return new GpuWeights(ctx,
                    find(gguf, layerPrefix + "attn_q.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "attn_k.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "attn_v.weight").loadQ8_0(),
                    loadAsFloat(find(gguf, layerPrefix + "attn_output.weight")),
                    find(gguf, layerPrefix + "ffn_gate.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "ffn_up.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "ffn_down.weight").loadQ8_0(),
                    loadAsFloat(find(gguf, layerPrefix + "attn_norm.weight")),
                    loadAsFloat(find(gguf, layerPrefix + "ffn_norm.weight")));
        }

        /** GGML type 0 = F32 (read directly), type 8 = Q8_0 (dequantized on the CPU at load time). wo/norm weights are kept FP32 device-side regardless of which way the GGUF file happens to store them on disk -- see class javadoc's "it's small" rationale. */
        private static float[] loadAsFloat(GGUFLoader.Tensor t) {
            return switch (t.type) {
                case 0 -> t.loadFloat();
                case 8 -> t.loadQ8_0AsFloat();
                default -> throw new IllegalArgumentException(
                        "Tensor '" + t.name + "' has GGML type " + t.type + " -- only F32 (0) and Q8_0 (8) are handled here.");
            };
        }

        private static GGUFLoader.Tensor find(GGUFLoader.GGUFFile gguf, String name) {
            GGUFLoader.Tensor t = gguf.tensors.get(name);
            if (t == null) {
                throw new NoSuchElementException("Required GGUF tensor missing: " + name);
            }
            return t;
        }

        @Override
        public void close() {
            freeQuietly(ctx, wq_q8_0);
            freeQuietly(ctx, wk_q8_0);
            freeQuietly(ctx, wv_q8_0);
            freeQuietly(ctx, wo_f32);
            freeQuietly(ctx, w_gate_q8_0);
            freeQuietly(ctx, w_up_q8_0);
            freeQuietly(ctx, w_down_q8_0);
            freeQuietly(ctx, attn_norm_gamma);
            freeQuietly(ctx, ffn_norm_gamma);
        }
    }

    /**
     * Per-sequence GPU state: KV cache, RoPE tables, decode-path scratch
     * (unchanged from v1), PLUS prefill-path batch scratch sized by
     * cfg.max_prefill_batch (skipped entirely when that's 0).
     */
    public static final class GpuState implements AutoCloseable {
        final GpuContext ctx;
        final int kvDim;
        final int maxBatchT;

        // ---- decode-path (v1, unchanged) ----
        final long k_cache_f32, v_cache_f32;
        final long cos_table, sin_table;
        final long xNorm, xNormQ8;
        final long qSplit, kNew, vNew;
        final long scores;
        final long attnOut, attnProj;
        final long ffnNorm, ffnNormQ8;
        final long gate, up, swigluOut, swigluOutQ8;
        final long ffnDownOut;
        final long rmsPartials;

        // ---- prefill-path batch scratch (new; 0 if maxBatchT == 0) ----
        final long qBatch, kBatch, vBatch;
        final long xNormBatch, xNormQ8Batch;
        final long scoresBatch;
        final long attnOutBatch, attnProjBatch;
        final long ffnNormBatch, ffnNormQ8Batch;
        final long gateBatch, upBatch, swigluOutBatch, swigluOutQ8Batch;
        final long ffnDownOutBatch;
        final long rmsPartialsBatch;
        final long rmsRowValuesBatch;   // [maxBatchT] rms scalar per row, uploaded fresh each rmsNormRows call
        final long positionsBatch;      // [maxBatchT] int, uploaded fresh each prefill_layer call

        public int pos = 0;

        public GpuState(GpuContext ctx, Config cfg) throws Throwable {
            cfg.validate();
            this.ctx = ctx;
            this.kvDim = cfg.kv_heads * cfg.head_dim;
            this.maxBatchT = cfg.max_prefill_batch;

            this.k_cache_f32 = allocFloats(ctx, (long) cfg.max_seq * kvDim);
            this.v_cache_f32 = allocFloats(ctx, (long) cfg.max_seq * kvDim);

            int halfDim = cfg.head_dim / 2;
            float[] cosHost = new float[cfg.max_seq * halfDim];
            float[] sinHost = new float[cfg.max_seq * halfDim];
            precomputeRope(cosHost, sinHost, cfg.max_seq, cfg.head_dim, cfg.rope_theta, cfg);
            this.cos_table = uploadFloats(ctx, cosHost);
            this.sin_table = uploadFloats(ctx, sinHost);

            this.xNorm = allocFloats(ctx, cfg.dim);
            this.xNormQ8 = allocBytes(ctx, q8_0Bytes(cfg.dim));
            this.qSplit = allocFloats(ctx, (long) cfg.num_heads * cfg.head_dim);
            this.kNew = allocFloats(ctx, kvDim);
            this.vNew = allocFloats(ctx, kvDim);
            this.scores = allocFloats(ctx, cfg.max_seq);
            this.attnOut = allocFloats(ctx, cfg.dim);
            this.attnProj = allocFloats(ctx, cfg.dim);
            this.ffnNorm = allocFloats(ctx, cfg.dim);
            this.ffnNormQ8 = allocBytes(ctx, q8_0Bytes(cfg.dim));
            this.gate = allocFloats(ctx, cfg.hidden_dim);
            this.up = allocFloats(ctx, cfg.hidden_dim);
            this.swigluOut = allocFloats(ctx, cfg.hidden_dim);
            this.swigluOutQ8 = allocBytes(ctx, q8_0Bytes(cfg.hidden_dim));
            this.ffnDownOut = allocFloats(ctx, cfg.dim);

            int maxLen = Math.max(cfg.dim, cfg.hidden_dim);
            int maxGroups = (maxLen + KernelSource.RMSNORM_WORKGROUP_SIZE - 1) / KernelSource.RMSNORM_WORKGROUP_SIZE;
            this.rmsPartials = allocFloats(ctx, maxGroups);

            if (maxBatchT > 0) {
                int qkRowWidth = cfg.num_heads * cfg.head_dim;
                this.qBatch = allocFloats(ctx, (long) maxBatchT * qkRowWidth);
                this.kBatch = allocFloats(ctx, (long) maxBatchT * kvDim);
                this.vBatch = allocFloats(ctx, (long) maxBatchT * kvDim);
                this.xNormBatch = allocFloats(ctx, (long) maxBatchT * cfg.dim);
                this.xNormQ8Batch = allocBytes(ctx, (long) maxBatchT * q8_0Bytes(cfg.dim));
                this.scoresBatch = allocFloats(ctx, (long) maxBatchT * maxBatchT);
                this.attnOutBatch = allocFloats(ctx, (long) maxBatchT * cfg.dim);
                this.attnProjBatch = allocFloats(ctx, (long) maxBatchT * cfg.dim);
                this.ffnNormBatch = allocFloats(ctx, (long) maxBatchT * cfg.dim);
                this.ffnNormQ8Batch = allocBytes(ctx, (long) maxBatchT * q8_0Bytes(cfg.dim));
                this.gateBatch = allocFloats(ctx, (long) maxBatchT * cfg.hidden_dim);
                this.upBatch = allocFloats(ctx, (long) maxBatchT * cfg.hidden_dim);
                this.swigluOutBatch = allocFloats(ctx, (long) maxBatchT * cfg.hidden_dim);
                this.swigluOutQ8Batch = allocBytes(ctx, (long) maxBatchT * q8_0Bytes(cfg.hidden_dim));
                this.ffnDownOutBatch = allocFloats(ctx, (long) maxBatchT * cfg.dim);
                this.rmsPartialsBatch = allocFloats(ctx, (long) maxBatchT * maxGroups);
                this.rmsRowValuesBatch = allocFloats(ctx, maxBatchT);
                this.positionsBatch = allocBytes(ctx, (long) maxBatchT * ValueLayout.JAVA_INT.byteSize());
            } else {
                this.qBatch = 0; this.kBatch = 0; this.vBatch = 0;
                this.xNormBatch = 0; this.xNormQ8Batch = 0;
                this.scoresBatch = 0;
                this.attnOutBatch = 0; this.attnProjBatch = 0;
                this.ffnNormBatch = 0; this.ffnNormQ8Batch = 0;
                this.gateBatch = 0; this.upBatch = 0; this.swigluOutBatch = 0; this.swigluOutQ8Batch = 0;
                this.ffnDownOutBatch = 0;
                this.rmsPartialsBatch = 0; this.rmsRowValuesBatch = 0; this.positionsBatch = 0;
            }
        }

        private static int q8_0Bytes(int len) {
            return (len / 32) * 34;
        }

        private static void precomputeRope(float[] cosOut, float[] sinOut, int maxSeq, int headDim, double base, Config cfg) {
            int halfDim = headDim / 2;
            for (int p = 0; p < maxSeq; p++) {
                for (int i = 0; i < halfDim; i++) {
                    double freq = 1.0 / Math.pow(base, (2.0 * i) / headDim);
                    freq = applyRopeScaling(freq, cfg);
                    double angle = p * freq;
                    cosOut[p * halfDim + i] = (float) Math.cos(angle);
                    sinOut[p * halfDim + i] = (float) Math.sin(angle);
                }
            }
        }

        /** Llama-3 NTK-by-parts RoPE frequency correction -- see com.github.gbenroscience.gpu.llm.opencl.LlamaLayer's identical method for the full derivation/javadoc. Identity when rope_scaling_factor==1.0. */
        private static double applyRopeScaling(double freq, Config cfg) {
            if (cfg.rope_scaling_factor == 1.0) {
                return freq;
            }
            double lowFreqWavelen = cfg.rope_scaling_orig_context_length / cfg.rope_scaling_low_freq_factor;
            double highFreqWavelen = cfg.rope_scaling_orig_context_length / cfg.rope_scaling_high_freq_factor;
            double wavelen = 2.0 * Math.PI / freq;

            if (wavelen < highFreqWavelen) {
                return freq;
            }
            if (wavelen > lowFreqWavelen) {
                return freq / cfg.rope_scaling_factor;
            }
            double smooth = (cfg.rope_scaling_orig_context_length / wavelen - cfg.rope_scaling_low_freq_factor)
                    / (cfg.rope_scaling_high_freq_factor - cfg.rope_scaling_low_freq_factor);
            return (1.0 - smooth) * (freq / cfg.rope_scaling_factor) + smooth * freq;
        }

        @Override
        public void close() {
            freeQuietly(ctx, k_cache_f32);
            freeQuietly(ctx, v_cache_f32);
            freeQuietly(ctx, cos_table);
            freeQuietly(ctx, sin_table);
            freeQuietly(ctx, xNorm);
            freeQuietly(ctx, xNormQ8);
            freeQuietly(ctx, qSplit);
            freeQuietly(ctx, kNew);
            freeQuietly(ctx, vNew);
            freeQuietly(ctx, scores);
            freeQuietly(ctx, attnOut);
            freeQuietly(ctx, attnProj);
            freeQuietly(ctx, ffnNorm);
            freeQuietly(ctx, ffnNormQ8);
            freeQuietly(ctx, gate);
            freeQuietly(ctx, up);
            freeQuietly(ctx, swigluOut);
            freeQuietly(ctx, swigluOutQ8);
            freeQuietly(ctx, ffnDownOut);
            freeQuietly(ctx, rmsPartials);
            if (maxBatchT > 0) {
                freeQuietly(ctx, qBatch);
                freeQuietly(ctx, kBatch);
                freeQuietly(ctx, vBatch);
                freeQuietly(ctx, xNormBatch);
                freeQuietly(ctx, xNormQ8Batch);
                freeQuietly(ctx, scoresBatch);
                freeQuietly(ctx, attnOutBatch);
                freeQuietly(ctx, attnProjBatch);
                freeQuietly(ctx, ffnNormBatch);
                freeQuietly(ctx, ffnNormQ8Batch);
                freeQuietly(ctx, gateBatch);
                freeQuietly(ctx, upBatch);
                freeQuietly(ctx, swigluOutBatch);
                freeQuietly(ctx, swigluOutQ8Batch);
                freeQuietly(ctx, ffnDownOutBatch);
                freeQuietly(ctx, rmsPartialsBatch);
                freeQuietly(ctx, rmsRowValuesBatch);
                freeQuietly(ctx, positionsBatch);
            }
        }
    }

    // ================= device alloc/upload helpers (long CUdeviceptr) =================

    public static long allocFloats(GpuContext ctx, long count) throws Throwable {
        return allocBytes(ctx, count * ValueLayout.JAVA_FLOAT.byteSize());
    }

    public static long allocBytes(GpuContext ctx, long byteCount) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
            GpuContext.check((int) ctx.cu.cuMemAlloc.invoke(ptrBuf, byteCount), "cuMemAlloc");
            return ptrBuf.get(ValueLayout.JAVA_LONG, 0);
        }
    }

    public static long uploadBytes(GpuContext ctx, byte[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate(data.length);
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_BYTE, 0, data.length);
            MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
            GpuContext.check((int) ctx.cu.cuMemAlloc.invoke(ptrBuf, host.byteSize()), "cuMemAlloc(weight bytes)");
            long device = ptrBuf.get(ValueLayout.JAVA_LONG, 0);
            GpuContext.check((int) ctx.cu.cuMemcpyHtoD.invoke(device, host, host.byteSize()), "cuMemcpyHtoD(weight bytes)");
            return device;
        }
    }

    public static long uploadFloats(GpuContext ctx, float[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
            GpuContext.check((int) ctx.cu.cuMemAlloc.invoke(ptrBuf, host.byteSize()), "cuMemAlloc(weight floats)");
            long device = ptrBuf.get(ValueLayout.JAVA_LONG, 0);
            GpuContext.check((int) ctx.cu.cuMemcpyHtoD.invoke(device, host, host.byteSize()), "cuMemcpyHtoD(weight floats)");
            return device;
        }
    }

    /** Uploads a freshly-computed int[] (e.g. per-row RoPE positions) to an ALREADY-ALLOCATED device buffer -- no new cuMemAlloc, reuses GpuState's persistent scratch. */
    private static void uploadIntsInto(GpuContext ctx, long device, int[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_INT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);
            GpuContext.check((int) ctx.cu.cuMemcpyHtoD.invoke(device, host, host.byteSize()), "cuMemcpyHtoD(positions)");
        }
    }

    /** Uploads a freshly-computed float[] (e.g. per-row rms values) to an ALREADY-ALLOCATED device buffer. */
    private static void uploadFloatsInto(GpuContext ctx, long device, float[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            GpuContext.check((int) ctx.cu.cuMemcpyHtoD.invoke(device, host, host.byteSize()), "cuMemcpyHtoD(rms rows)");
        }
    }

    public static long uploadInts(GpuContext ctx, int[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_INT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);
            MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
            GpuContext.check((int) ctx.cu.cuMemAlloc.invoke(ptrBuf, host.byteSize()), "cuMemAlloc(ints)");
            long device = ptrBuf.get(ValueLayout.JAVA_LONG, 0);
            GpuContext.check((int) ctx.cu.cuMemcpyHtoD.invoke(device, host, host.byteSize()), "cuMemcpyHtoD(ints)");
            return device;
        }
    }

    /** Blocking readback of a float buffer -- public so callers outside this package (benchmarks, tests, other engine code) can pull a result off the device without reaching into GpuContext's raw CUDA handles. cuMemcpyDtoH is synchronous, so this also acts as a sync point against the default stream, same as every other blocking copy in this file. */
    public static float[] downloadFloats(GpuContext ctx, long devicePtr, int count) throws Throwable {
        float[] out = new float[count];
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) count * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment host = tmp.allocate(Math.max(byteSize, 1));
            GpuContext.check((int) ctx.cu.cuMemcpyDtoH.invoke(host, devicePtr, byteSize), "cuMemcpyDtoH");
            MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, out, 0, count);
        }
        return out;
    }

    /** Blocks until every command already issued on the default stream has completed. Not needed anywhere in forward_layer/prefill_layer (kernel launches on the same stream already execute in issued order), but useful to callers timing a single kernel dispatch in isolation -- a benchmark brackets a kernel call with this before stopping its clock, otherwise it would only be timing host-side launch overhead, not actual GPU execution. */
    public static void finish(GpuContext ctx) throws Throwable {
        GpuContext.check((int) ctx.cu.cuCtxSynchronize.invoke(), "cuCtxSynchronize");
    }

    public static void freeQuietly(GpuContext ctx, long devicePtr) {
        try {
            if (devicePtr != 0L) {
                ctx.cu.cuMemFree.invoke(devicePtr);
            }
        } catch (Throwable t) {
            // best-effort cleanup
        }
    }

    // =====================================================================
    // THE DECODE-PATH DECODER LAYER (single token, M=1) -- v1, unchanged
    // except for sync discipline (see class javadoc).
    // =====================================================================

    /** Runs one decoder layer for the CURRENT token, in place on x. x: device pointer, [dim] floats, in/out. */
    public static void forward_layer(long x, GpuWeights w, GpuState s, Config cfg, GpuContext ctx) throws Throwable {
        final int dim = cfg.dim;
        final int hidden = cfg.hidden_dim;
        final int headDim = cfg.head_dim;
        final int numHeads = cfg.num_heads;
        final int kvHeads = cfg.kv_heads;
        final int kvDim = s.kvDim;
        final int halfDim = headDim / 2;
        final int groupSize = numHeads / kvHeads;
        final float rsqrtD = (float) (1.0 / Math.sqrt(headDim));
        final int posInclusive = s.pos + 1;

        synchronized (ctx.dispatchLock) {
            GpuContext.check((int) ctx.cu.cuCtxSetCurrent.invoke(ctx.context), "cuCtxSetCurrent");

            // === 1. Attention block: x = x + attn(rms_norm(x)) ===
            rmsNorm(ctx, x, w.attn_norm_gamma, s.xNorm, dim, cfg.norm_eps, s.rmsPartials);
            quantizeActivationQ8_0(ctx, s.xNorm, s.xNormQ8, dim);

            q8_0GemvSplit(ctx, s.xNormQ8, w.wq_q8_0, s.qSplit, numHeads, headDim, dim);
            q8_0GemvSplit(ctx, s.xNormQ8, w.wk_q8_0, s.kNew, kvHeads, headDim, dim);
            q8_0GemvSplit(ctx, s.xNormQ8, w.wv_q8_0, s.vNew, kvHeads, headDim, dim);

            ropeApplySplit(ctx, s.qSplit, s.cos_table, s.sin_table, numHeads, headDim, s.pos * halfDim);
            ropeApplySplit(ctx, s.kNew, s.cos_table, s.sin_table, kvHeads, headDim, s.pos * halfDim);

            writeIntoCache(ctx, s.kNew, s.k_cache_f32, (long) s.pos * kvDim, kvDim);
            writeIntoCache(ctx, s.vNew, s.v_cache_f32, (long) s.pos * kvDim, kvDim);

            for (int h = 0; h < numHeads; h++) {
                int kvHead = h / groupSize;
                int qHeadOff = h * headDim;
                int kvHeadOff = kvHead * headDim;

                attnScores(ctx, s.qSplit, s.k_cache_f32, s.scores,
                        qHeadOff, headDim, kvDim, kvHeadOff, posInclusive, rsqrtD);
                softmaxInplace(ctx, s.scores, posInclusive);
                attnWeightedSum(ctx, s.scores, s.v_cache_f32, s.attnOut,
                        qHeadOff, headDim, kvDim, kvHeadOff, posInclusive);
            }

            f32Gemv(ctx, s.attnOut, w.wo_f32, s.attnProj, dim, dim);
            residualAdd(ctx, x, s.attnProj, dim);

            // === 2. FFN block: x = x + ffn(rms_norm(x)) ===
            rmsNorm(ctx, x, w.ffn_norm_gamma, s.ffnNorm, dim, cfg.norm_eps, s.rmsPartials);
            quantizeActivationQ8_0(ctx, s.ffnNorm, s.ffnNormQ8, dim);

            ffnActivateDecode(ctx, s, w, cfg, dim, hidden);

            quantizeActivationQ8_0(ctx, s.swigluOut, s.swigluOutQ8, hidden);
            q8_0GemvPlain(ctx, s.swigluOutQ8, w.w_down_q8_0, s.ffnDownOut, dim, hidden);
            residualAdd(ctx, x, s.ffnDownOut, dim);

            // Not required for correctness (see class javadoc) -- kept for
            // predictable per-token timing if the caller profiles this call.
            GpuContext.check((int) ctx.cu.cuCtxSynchronize.invoke(), "cuCtxSynchronize(end of forward_layer)");
        }

        s.pos++;
    }

    /** Dispatches the FFN activation for the decode path based on cfg.activationType. */
    private static void ffnActivateDecode(GpuContext ctx, GpuState s, GpuWeights w, Config cfg, int dim, int hidden) throws Throwable {
        switch (cfg.activationType) {
            case SWIGLU -> {
                q8_0GemvPlain(ctx, s.ffnNormQ8, w.w_gate_q8_0, s.gate, hidden, dim);
                q8_0GemvPlain(ctx, s.ffnNormQ8, w.w_up_q8_0, s.up, hidden, dim);
                swigluActivate(ctx, s.gate, s.up, s.swigluOut, hidden);
            }
            case GEGLU -> {
                q8_0GemvPlain(ctx, s.ffnNormQ8, w.w_gate_q8_0, s.gate, hidden, dim);
                q8_0GemvPlain(ctx, s.ffnNormQ8, w.w_up_q8_0, s.up, hidden, dim);
                gegluActivate(ctx, s.gate, s.up, s.swigluOut, hidden);
            }
            case GELU -> {
                // Ungated: only the gate projection feeds the activation
                // (the up projection is simply not computed here -- see
                // ActivationType.GELU's javadoc for the GGUF-mapping caveat).
                q8_0GemvPlain(ctx, s.ffnNormQ8, w.w_gate_q8_0, s.gate, hidden, dim);
                geluActivate(ctx, s.gate, s.swigluOut, hidden);
            }
        }
    }

    /**
     * Runs the full model for one token: embedding -> N decoder layers ->
     * final norm -> LM head, greedy argmax. For non-greedy sampling, use
     * generate_logits below and feed the result to Sampler instead.
     */
    public static int generate_token(
            long tokenEmbeddingDevice,
            GpuWeights[] layers,
            long finalNormGammaDevice,
            long lmHeadDevice,
            GpuState[] states,
            Config cfg,
            GpuContext ctx,
            int vocabSize) throws Throwable {

        float[] logits = generate_logits(tokenEmbeddingDevice, layers, finalNormGammaDevice, lmHeadDevice, states, cfg, ctx, vocabSize);

        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < vocabSize; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Same forward pass as generate_token, but returns the raw logits
     * (host float[]) instead of collapsing to argmax -- the entry point
     * for Sampler-based (temperature/top-k/top-p/repetition-penalty)
     * decoding instead of pure greedy.
     */
    public static float[] generate_logits(
            long tokenEmbeddingDevice,
            GpuWeights[] layers,
            long finalNormGammaDevice,
            long lmHeadDevice,
            GpuState[] states,
            Config cfg,
            GpuContext ctx,
            int vocabSize) throws Throwable {

        long x = tokenEmbeddingDevice;

        for (int l = 0; l < layers.length; l++) {
            forward_layer(x, layers[l], states[l], cfg, ctx);
        }

        long normed = allocFloats(ctx, cfg.dim);
        long logitsDevice = allocFloats(ctx, vocabSize);
        int maxGroups = (cfg.dim + KernelSource.RMSNORM_WORKGROUP_SIZE - 1) / KernelSource.RMSNORM_WORKGROUP_SIZE;
        long partials = allocFloats(ctx, maxGroups);

        try {
            synchronized (ctx.dispatchLock) {
                GpuContext.check((int) ctx.cu.cuCtxSetCurrent.invoke(ctx.context), "cuCtxSetCurrent");
                rmsNorm(ctx, x, finalNormGammaDevice, normed, cfg.dim, cfg.norm_eps, partials);
                f32Gemv(ctx, normed, lmHeadDevice, logitsDevice, cfg.dim, vocabSize);
            }

            float[] logits = new float[vocabSize];
            try (Arena tmp = Arena.ofConfined()) {
                MemorySegment logitsHost = tmp.allocate((long) vocabSize * ValueLayout.JAVA_FLOAT.byteSize());
                GpuContext.check((int) ctx.cu.cuMemcpyDtoH.invoke(logitsHost, logitsDevice, logitsHost.byteSize()),
                        "cuMemcpyDtoH(logits)");
                MemorySegment.copy(logitsHost, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
            }

            for (GpuState st : states) {
                st.pos++;
            }

            return logits;
        } finally {
            freeQuietly(ctx, normed);
            freeQuietly(ctx, logitsDevice);
            freeQuietly(ctx, partials);
        }
    }

    /**
     * Final norm + LM head + host readback on an ARBITRARY already-computed
     * row pointer -- does NOT run the decoder layers (forward_layer /
     * prefill_layer must already have been applied to whatever buffer
     * xRow points at). Used by the engine layer both after decode
     * (xRow = the [dim] decode scratch buffer) and after prefill
     * (xRow = a pointer to the LAST row of the [T,dim] batch buffer,
     * i.e. xBatch + (T-1)*dim*sizeof(float)) so both paths share one
     * logits computation instead of duplicating it.
     */
    public static float[] finalLogits(long xRow, long finalNormGammaDevice, long lmHeadDevice,
            Config cfg, GpuContext ctx, int vocabSize) throws Throwable {
        long normed = allocFloats(ctx, cfg.dim);
        long logitsDevice = allocFloats(ctx, vocabSize);
        int maxGroups = (cfg.dim + KernelSource.RMSNORM_WORKGROUP_SIZE - 1) / KernelSource.RMSNORM_WORKGROUP_SIZE;
        long partials = allocFloats(ctx, maxGroups);

        try {
            synchronized (ctx.dispatchLock) {
                GpuContext.check((int) ctx.cu.cuCtxSetCurrent.invoke(ctx.context), "cuCtxSetCurrent");
                rmsNorm(ctx, xRow, finalNormGammaDevice, normed, cfg.dim, cfg.norm_eps, partials);
                f32Gemv(ctx, normed, lmHeadDevice, logitsDevice, cfg.dim, vocabSize);
            }

            float[] logits = new float[vocabSize];
            try (Arena tmp = Arena.ofConfined()) {
                MemorySegment logitsHost = tmp.allocate((long) vocabSize * ValueLayout.JAVA_FLOAT.byteSize());
                GpuContext.check((int) ctx.cu.cuMemcpyDtoH.invoke(logitsHost, logitsDevice, logitsHost.byteSize()),
                        "cuMemcpyDtoH(logits)");
                MemorySegment.copy(logitsHost, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
            }
            return logits;
        } finally {
            freeQuietly(ctx, normed);
            freeQuietly(ctx, logitsDevice);
            freeQuietly(ctx, partials);
        }
    }

    // =====================================================================
    // THE BATCHED PREFILL PATH (new): T tokens at once, one layer.
    // =====================================================================

    /**
     * Runs one decoder layer over a BATCH of T prompt positions at once,
     * writing the results into the persistent KV cache at
     * [startPos, startPos+T) so subsequent forward_layer (decode) calls
     * can attend to them. xBatch is [T, dim] device, row-major, in/out --
     * caller populates it (embedding lookup for all T prompt tokens)
     * before calling, and it holds the post-layer activations after.
     *
     * SINGLE CALL ONLY PER SEQUENCE -- NOT CHUNKABLE, DESPITE startPos
     * BEING A PARAMETER: attn_scores_causal_batched attends ONLY within
     * THIS call's own qBatch/kBatch/vBatch (the current T rows) -- it
     * does not read the persistent k_cache_f32/v_cache_f32 at all during
     * prefill (that cache write happens, via writeIntoCache, purely so
     * LATER forward_layer decode calls can attend back to these
     * positions; nothing in THIS call reads it back). Calling
     * prefill_layer twice with startPos=0 then startPos=T would silently
     * produce WRONG output for the second call: its rows would attend
     * only to each other, never to the first T positions, even though
     * their K/V got written into the cache. startPos here exists only to
     * record the correct absolute RoPE angle and cache offset for a
     * prompt that doesn't start at position 0 (e.g. continuing after a
     * prior single prefill_layer + several forward_layer decode calls) --
     * it is NOT a green light for splitting one prompt across multiple
     * prefill_layer calls. Making cross-chunk attention correct would
     * mean attn_scores_causal_batched reading k_cache_f32 directly
     * (variable per-row valid length startPos+t+1, not the fixed-width
     * T-wide scores buffer this version allocates) -- a real kernel and
     * GpuState-sizing change, not attempted here. Callers with prompts
     * longer than cfg.max_prefill_batch must either raise that config
     * value (more device memory for scoresBatch, O(max_prefill_batch^2))
     * or fall back to T sequential forward_layer decode calls for the
     * overflow, which IS correct (just not batched-fast).
     *
     * @throws IllegalStateException if cfg.max_prefill_batch == 0 (no
     *         batch scratch was allocated) or T exceeds it.
     */
    public static void prefill_layer(long xBatch, GpuWeights w, GpuState s, Config cfg, GpuContext ctx,
            int startPos, int T) throws Throwable {
        if (s.maxBatchT == 0) {
            throw new IllegalStateException("This GpuState was built with max_prefill_batch=0 -- no batch scratch allocated; prefill is unavailable.");
        }
        if (T > s.maxBatchT) {
            throw new IllegalStateException("Prefill batch T=" + T + " exceeds this GpuState's max_prefill_batch=" + s.maxBatchT);
        }
        if (startPos != 0) {
            throw new IllegalArgumentException(
                    "prefill_layer only supports startPos=0 -- attn_scores_causal_batched attends "
                            + "solely within this call's own T rows, not the persistent KV cache, so a "
                            + "nonzero startPos would silently produce wrong attention output for any row "
                            + "that should see earlier cached positions. See this method's javadoc.");
        }

        final int dim = cfg.dim;
        final int hidden = cfg.hidden_dim;
        final int headDim = cfg.head_dim;
        final int numHeads = cfg.num_heads;
        final int kvHeads = cfg.kv_heads;
        final int kvDim = s.kvDim;
        final int qRowStride = numHeads * headDim;
        final int groupSize = numHeads / kvHeads;
        final float rsqrtD = (float) (1.0 / Math.sqrt(headDim));

        int[] positions = new int[T];
        for (int i = 0; i < T; i++) {
            positions[i] = startPos + i;
        }

        synchronized (ctx.dispatchLock) {
            GpuContext.check((int) ctx.cu.cuCtxSetCurrent.invoke(ctx.context), "cuCtxSetCurrent");
            uploadIntsInto(ctx, s.positionsBatch, positions);

            // === 1. Attention block ===
            rmsNormRows(ctx, xBatch, w.attn_norm_gamma, s.xNormBatch, dim, cfg.norm_eps, T, s.rmsPartialsBatch, s.rmsRowValuesBatch);
            quantizeActivationQ8_0(ctx, s.xNormBatch, s.xNormQ8Batch, T * dim);

            q8_0GemmTiled(ctx, s.xNormQ8Batch, w.wq_q8_0, s.qBatch, T, qRowStride, dim);
            q8_0GemmTiled(ctx, s.xNormQ8Batch, w.wk_q8_0, s.kBatch, T, kvDim, dim);
            q8_0GemmTiled(ctx, s.xNormQ8Batch, w.wv_q8_0, s.vBatch, T, kvDim, dim);

            ropeApplyPairwiseRows(ctx, s.qBatch, s.cos_table, s.sin_table, numHeads, headDim, s.positionsBatch, T);
            ropeApplyPairwiseRows(ctx, s.kBatch, s.cos_table, s.sin_table, kvHeads, headDim, s.positionsBatch, T);

            // Bulk cache write -- same host round-trip writeIntoCache uses
            // for the decode path, just T*kvDim elements instead of kvDim.
            writeIntoCache(ctx, s.kBatch, s.k_cache_f32, (long) startPos * kvDim, T * kvDim);
            writeIntoCache(ctx, s.vBatch, s.v_cache_f32, (long) startPos * kvDim, T * kvDim);

            // --- causal batched MHA loop: still one dispatch triple per head ---
            for (int h = 0; h < numHeads; h++) {
                int kvHead = h / groupSize;
                int qHeadOff = h * headDim;
                int kvHeadOff = kvHead * headDim;

                attnScoresCausalBatched(ctx, s.qBatch, s.kBatch, s.scoresBatch,
                        qRowStride, kvDim, qHeadOff, kvHeadOff, headDim, T, rsqrtD);
                softmaxInplaceRows(ctx, s.scoresBatch, T);
                attnWeightedSumCausalBatched(ctx, s.scoresBatch, s.vBatch, s.attnOutBatch,
                        kvDim, qRowStride, kvHeadOff, qHeadOff, headDim, T);
            }

            f32GemmTiled(ctx, s.attnOutBatch, w.wo_f32, s.attnProjBatch, T, dim, dim);
            residualAdd(ctx, xBatch, s.attnProjBatch, T * dim);

            // === 2. FFN block ===
            rmsNormRows(ctx, xBatch, w.ffn_norm_gamma, s.ffnNormBatch, dim, cfg.norm_eps, T, s.rmsPartialsBatch, s.rmsRowValuesBatch);
            quantizeActivationQ8_0(ctx, s.ffnNormBatch, s.ffnNormQ8Batch, T * dim);

            ffnActivateBatched(ctx, s, w, cfg, T, dim, hidden);

            quantizeActivationQ8_0(ctx, s.swigluOutBatch, s.swigluOutQ8Batch, T * hidden);
            q8_0GemmTiled(ctx, s.swigluOutQ8Batch, w.w_down_q8_0, s.ffnDownOutBatch, T, dim, hidden);
            residualAdd(ctx, xBatch, s.ffnDownOutBatch, T * dim);

            GpuContext.check((int) ctx.cu.cuCtxSynchronize.invoke(), "cuCtxSynchronize(end of prefill_layer)");
        }

        s.pos = startPos + T;
    }

    /** Same three-way activation dispatch as ffnActivateDecode, batched-kernel variants (elementwise kernels are reused unmodified with len=T*hidden). */
    private static void ffnActivateBatched(GpuContext ctx, GpuState s, GpuWeights w, Config cfg, int T, int dim, int hidden) throws Throwable {
        switch (cfg.activationType) {
            case SWIGLU -> {
                q8_0GemmTiled(ctx, s.ffnNormQ8Batch, w.w_gate_q8_0, s.gateBatch, T, hidden, dim);
                q8_0GemmTiled(ctx, s.ffnNormQ8Batch, w.w_up_q8_0, s.upBatch, T, hidden, dim);
                swigluActivate(ctx, s.gateBatch, s.upBatch, s.swigluOutBatch, T * hidden);
            }
            case GEGLU -> {
                q8_0GemmTiled(ctx, s.ffnNormQ8Batch, w.w_gate_q8_0, s.gateBatch, T, hidden, dim);
                q8_0GemmTiled(ctx, s.ffnNormQ8Batch, w.w_up_q8_0, s.upBatch, T, hidden, dim);
                gegluActivate(ctx, s.gateBatch, s.upBatch, s.swigluOutBatch, T * hidden);
            }
            case GELU -> {
                q8_0GemmTiled(ctx, s.ffnNormQ8Batch, w.w_gate_q8_0, s.gateBatch, T, hidden, dim);
                geluActivate(ctx, s.gateBatch, s.swigluOutBatch, T * hidden);
            }
        }
    }

    // =====================================================================
    // Kernel dispatch helpers. All assume the caller holds ctx.dispatchLock
    // and has already set the current context. No cuCtxSynchronize inside
    // any of these -- see class javadoc for why that's correct.
    // =====================================================================

    // ---- decode-path dispatch (v1, unchanged apart from dropped syncs) ----

    static void rmsNorm(GpuContext ctx, long x, long gamma, long out,
            int features, double eps, long partials) throws Throwable {
        int wgSize = KernelSource.RMSNORM_WORKGROUP_SIZE;
        int numGroups = (features + wgSize - 1) / wgSize;

        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, x), argL(tmp, partials), argI(tmp, features));
            launch1D(ctx, ctx.kRmsnormPartialSumsq, (long) numGroups * wgSize, wgSize,
                    wgSize * (int) ValueLayout.JAVA_FLOAT.byteSize(), params);
        }

        float[] partialHost = new float[numGroups];
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) numGroups * ValueLayout.JAVA_FLOAT.byteSize());
            GpuContext.check((int) ctx.cu.cuMemcpyDtoH.invoke(host, partials, host.byteSize()),
                    "cuMemcpyDtoH(rmsnorm partials)"); // synchronous -> implicit sync point
            MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, partialHost, 0, numGroups);
        }
        double sumSq = 0.0;
        for (float p : partialHost) {
            sumSq += p;
        }
        float rms = (float) (1.0 / Math.sqrt(sumSq / features + eps));

        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, x), argL(tmp, gamma), argL(tmp, out), argF(tmp, rms), argI(tmp, features));
            launch1D(ctx, ctx.kRmsnormApply, features, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void quantizeActivationQ8_0(GpuContext ctx, long x, long outQ8, int len) throws Throwable {
        int blockSize = KernelSource.QUANTIZE_BLOCK_SIZE;
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, x), argL(tmp, outQ8), argI(tmp, len));
            launch1D(ctx, ctx.kQuantizeActivationQ8_0, len, blockSize,
                    blockSize * (int) ValueLayout.JAVA_FLOAT.byteSize(), params);
        }
    }

    static void q8_0GemvSplit(GpuContext ctx, long xQ8, long wQ8, long out,
            int heads, int headDim, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, xQ8), argL(tmp, wQ8), argL(tmp, out),
                    argI(tmp, heads), argI(tmp, headDim), argI(tmp, K));
            launch1D(ctx, ctx.kQ8_0GemvSplit, (long) heads * (headDim / 2), KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void q8_0GemvPlain(GpuContext ctx, long xQ8, long wQ8, long out,
            int N, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, xQ8), argL(tmp, wQ8), argL(tmp, out), argI(tmp, N), argI(tmp, K));
            launch1D(ctx, ctx.kQ8_0GemvPlain, N, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void ropeApplySplit(GpuContext ctx, long buf, long cosTable, long sinTable,
            int heads, int headDim, int cosSinOffset) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, buf), argL(tmp, cosTable), argL(tmp, sinTable),
                    argI(tmp, heads), argI(tmp, headDim), argI(tmp, cosSinOffset));
            launch1D(ctx, ctx.kRopeApplySplit, (long) heads * (headDim / 2), KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void attnScores(GpuContext ctx, long qAllHeads, long kCache, long scores,
            int qHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive, float rsqrtD) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, qAllHeads), argL(tmp, kCache), argL(tmp, scores),
                    argI(tmp, qHeadOff), argI(tmp, headDim), argI(tmp, kvDim),
                    argI(tmp, kvHeadOff), argI(tmp, posInclusive), argF(tmp, rsqrtD));
            launch1D(ctx, ctx.kAttnScores, posInclusive, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void softmaxInplace(GpuContext ctx, long scores, int len) throws Throwable {
        int localSize = nextPow2(Math.max(len, 1));
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, scores), argI(tmp, len));
            launch1D(ctx, ctx.kSoftmaxInplace, localSize, localSize,
                    localSize * (int) ValueLayout.JAVA_FLOAT.byteSize(), params);
        }
    }

    private static int nextPow2(int v) {
        int p = 1;
        while (p < v) {
            p <<= 1;
        }
        return Math.min(p, 256);
    }

    static void attnWeightedSum(GpuContext ctx, long scores, long vCache, long attnOutAllHeads,
            int outHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, scores), argL(tmp, vCache), argL(tmp, attnOutAllHeads),
                    argI(tmp, outHeadOff), argI(tmp, headDim), argI(tmp, kvDim),
                    argI(tmp, kvHeadOff), argI(tmp, posInclusive));
            launch1D(ctx, ctx.kAttnWeightedSum, headDim, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void swigluActivate(GpuContext ctx, long gate, long up, long out, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, gate), argL(tmp, up), argL(tmp, out), argI(tmp, len));
            launch1D(ctx, ctx.kSwigluActivate, len, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void gegluActivate(GpuContext ctx, long gate, long up, long out, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, gate), argL(tmp, up), argL(tmp, out), argI(tmp, len));
            launch1D(ctx, ctx.kGegluActivate, len, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void geluActivate(GpuContext ctx, long gate, long out, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, gate), argL(tmp, out), argI(tmp, len));
            launch1D(ctx, ctx.kGeluActivate, len, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    private static void residualAdd(GpuContext ctx, long x, long y, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, x), argL(tmp, y), argI(tmp, len));
            launch1D(ctx, ctx.kResidualAdd, len, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void f32Gemv(GpuContext ctx, long a, long B, long out, int K, int N) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, a), argL(tmp, B), argL(tmp, out), argI(tmp, K), argI(tmp, N));
            launch1D(ctx, ctx.kF32Gemv, N, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    /** Host round-trip cache write. Offset is folded directly into the CUdeviceptr value. */
    private static void writeIntoCache(GpuContext ctx, long src, long cache, long elementOffset, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) len * ValueLayout.JAVA_FLOAT.byteSize());
            GpuContext.check((int) ctx.cu.cuMemcpyDtoH.invoke(host, src, host.byteSize()),
                    "cuMemcpyDtoH(cache write, read new K/V)"); // synchronous -> implicit sync point
            long dstOffset = cache + elementOffset * ValueLayout.JAVA_FLOAT.byteSize();
            GpuContext.check((int) ctx.cu.cuMemcpyHtoD.invoke(dstOffset, host, host.byteSize()),
                    "cuMemcpyHtoD(cache write, at offset)");
        }
    }

    // ---- batched prefill dispatch (new) ----

    static void rmsNormRows(GpuContext ctx, long x, long gamma, long out,
            int features, double eps, int T, long partialsBatch, long rmsRowValuesBatch) throws Throwable {
        int wgSize = KernelSource.RMSNORM_WORKGROUP_SIZE;
        int numGroups = (features + wgSize - 1) / wgSize;

        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, x), argL(tmp, partialsBatch), argI(tmp, features), argI(tmp, numGroups));
            launch2D(ctx, ctx.kRmsnormPartialSumsqRows, T, (long) numGroups * wgSize, wgSize,
                    wgSize * (int) ValueLayout.JAVA_FLOAT.byteSize(), params);
        }

        float[] partialHost = new float[T * numGroups];
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) T * numGroups * ValueLayout.JAVA_FLOAT.byteSize());
            GpuContext.check((int) ctx.cu.cuMemcpyDtoH.invoke(host, partialsBatch, host.byteSize()),
                    "cuMemcpyDtoH(rmsnorm partials rows)"); // synchronous -> implicit sync point
            MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, partialHost, 0, T * numGroups);
        }

        float[] rmsPerRow = new float[T];
        for (int row = 0; row < T; row++) {
            double sumSq = 0.0;
            for (int g = 0; g < numGroups; g++) {
                sumSq += partialHost[row * numGroups + g];
            }
            rmsPerRow[row] = (float) (1.0 / Math.sqrt(sumSq / features + eps));
        }
        uploadFloatsInto(ctx, rmsRowValuesBatch, rmsPerRow);

        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, x), argL(tmp, gamma), argL(tmp, out), argL(tmp, rmsRowValuesBatch), argI(tmp, features));
            launch2D(ctx, ctx.kRmsnormApplyRows, T, features, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void q8_0GemmTiled(GpuContext ctx, long xQ8, long wQ8, long out, int T, int N, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, xQ8), argL(tmp, wQ8), argL(tmp, out), argI(tmp, T), argI(tmp, N), argI(tmp, K));
            launch2D(ctx, ctx.kQ8_0GemmTiled, T, N, KernelSource.GEMM_TILE_N, 0, params);
        }
    }

    static void f32GemmTiled(GpuContext ctx, long a, long B, long out, int T, int K, int N) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, a), argL(tmp, B), argL(tmp, out), argI(tmp, T), argI(tmp, K), argI(tmp, N));
            launch2D(ctx, ctx.kF32GemmTiled, T, N, KernelSource.GEMM_TILE_N, 0, params);
        }
    }

    static void ropeApplyPairwiseRows(GpuContext ctx, long buf, long cosTable, long sinTable,
            int heads, int headDim, long positionsBatch, int T) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, buf), argL(tmp, cosTable), argL(tmp, sinTable),
                    argI(tmp, heads), argI(tmp, headDim), argL(tmp, positionsBatch));
            launch2D(ctx, ctx.kRopeApplyPairwiseRows, T, (long) heads * (headDim / 2), KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void attnScoresCausalBatched(GpuContext ctx, long qAll, long kAll, long scoresBatch,
            int qRowStride, int kRowStride, int qHeadOff, int kHeadOff, int headDim, int T, float rsqrtD) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, qAll), argL(tmp, kAll), argL(tmp, scoresBatch),
                    argI(tmp, qRowStride), argI(tmp, kRowStride),
                    argI(tmp, qHeadOff), argI(tmp, kHeadOff), argI(tmp, headDim), argI(tmp, T), argF(tmp, rsqrtD));
            launch2D(ctx, ctx.kAttnScoresCausalBatched, T, T, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    static void softmaxInplaceRows(GpuContext ctx, long scoresBatch, int T) throws Throwable {
        int localSize = nextPow2(Math.max(T, 1));
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, scoresBatch), argI(tmp, T));
            GpuContext.check((int) ctx.cu.cuLaunchKernel.invoke(
                    ctx.kSoftmaxInplaceRows,
                    T, 1, 1,                 // gridDim: one block PER ROW
                    localSize, 1, 1,
                    localSize * (int) ValueLayout.JAVA_FLOAT.byteSize(),
                    MemorySegment.NULL, params, MemorySegment.NULL),
                    "cuLaunchKernel(softmax_inplace_rows)");
        }
    }

    static void attnWeightedSumCausalBatched(GpuContext ctx, long scoresBatch, long vAll, long attnOutBatch,
            int vRowStride, int outRowStride, int vHeadOff, int outHeadOff, int headDim, int T) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, scoresBatch), argL(tmp, vAll), argL(tmp, attnOutBatch),
                    argI(tmp, vRowStride), argI(tmp, outRowStride),
                    argI(tmp, vHeadOff), argI(tmp, outHeadOff), argI(tmp, headDim), argI(tmp, T));
            launch2D(ctx, ctx.kAttnWeightedSumCausalBatched, T, headDim, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    // ================= FFM plumbing: kernelParams construction + launch =================

    private static MemorySegment argL(Arena a, long v) {
        MemorySegment s = a.allocate(ValueLayout.JAVA_LONG);
        s.set(ValueLayout.JAVA_LONG, 0, v);
        return s;
    }

    private static MemorySegment argI(Arena a, int v) {
        MemorySegment s = a.allocate(ValueLayout.JAVA_INT);
        s.set(ValueLayout.JAVA_INT, 0, v);
        return s;
    }

    private static MemorySegment argF(Arena a, float v) {
        MemorySegment s = a.allocate(ValueLayout.JAVA_FLOAT);
        s.set(ValueLayout.JAVA_FLOAT, 0, v);
        return s;
    }

    private static MemorySegment kernelParams(Arena a, MemorySegment... argPtrs) {
        MemorySegment arr = a.allocate(ValueLayout.ADDRESS, argPtrs.length);
        for (int i = 0; i < argPtrs.length; i++) {
            arr.setAtIndex(ValueLayout.ADDRESS, i, argPtrs[i]);
        }
        return arr;
    }

    /** 1D launch: rounds workItems up to a whole number of blocks. No sync -- see class javadoc. */
    private static void launch1D(GpuContext ctx, MemorySegment function, long workItems, int blockSize,
            int sharedMemBytes, MemorySegment kernelParams) throws Throwable {
        int gridDim = (int) ((workItems + blockSize - 1) / blockSize);
        GpuContext.check((int) ctx.cu.cuLaunchKernel.invoke(
                function,
                gridDim, 1, 1,
                blockSize, 1, 1,
                sharedMemBytes,
                MemorySegment.NULL,
                kernelParams,
                MemorySegment.NULL),
                "cuLaunchKernel");
    }

    /** 2D launch: gridDim.y = rows (exact), gridDim.x = ceil(cols/blockSizeX). No sync -- see class javadoc. */
    private static void launch2D(GpuContext ctx, MemorySegment function, int rows, long cols, int blockSizeX,
            int sharedMemBytes, MemorySegment kernelParams) throws Throwable {
        int gridX = (int) ((cols + blockSizeX - 1) / blockSizeX);
        GpuContext.check((int) ctx.cu.cuLaunchKernel.invoke(
                function,
                gridX, rows, 1,
                blockSizeX, 1, 1,
                sharedMemBytes,
                MemorySegment.NULL,
                kernelParams,
                MemorySegment.NULL),
                "cuLaunchKernel(2D)");
    }
}