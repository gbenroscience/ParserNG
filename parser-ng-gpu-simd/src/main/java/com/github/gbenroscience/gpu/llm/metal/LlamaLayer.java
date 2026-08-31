package com.github.gbenroscience.gpu.llm.metal;

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.NoSuchElementException;

/**
 * Metal counterpart of {@code com.github.gbenroscience.gpu.llm.cuda.LlamaLayer}
 * -- same two execution paths (single-token {@link #forward_layer} and
 * batched {@link #prefill_layer}), same Config/GpuWeights/GpuState shape,
 * same activation-type dispatch (SwiGLU/GeLU/GeGLU). See
 * {@code MetalKernelSource}'s class javadoc for the kernel-level
 * translation rules and {@code MetalBuffer}'s javadoc for the
 * device-pointer representation change from the CUDA port's raw
 * {@code long}.
 *
 * <b>SYNC DISCIPLINE -- DELIBERATELY THE CUDA PORT'S "v1", NOT ITS "v2":</b>
 * every kernel dispatch in this file goes through {@link #runKernel1D}/
 * {@link #runKernel2D}, which each build ONE command buffer, encode ONE
 * kernel, commit, and {@code waitUntilCompleted} before returning. The
 * CUDA v2 javadoc (see that file) explicitly flags per-kernel
 * synchronization as "the single biggest throughput problem" in that
 * port's v1 and describes batching same-stream kernels into one launch
 * sequence with sync only at genuine host-readback points as the fix.
 * That same optimization is NOT attempted here, for a Metal-specific
 * reason worth stating plainly: unlike CUDA's default STREAM (kernels
 * enqueued on it execute in issued order with no host action required),
 * a Metal {@code MTLCommandBuffer} is fully ENCODED before any of it
 * runs -- you cannot encode kernel N+1 based on kernel N's output without
 * first committing and waiting for kernel N, because kernel N hasn't run
 * yet at encode time. Batching multiple kernels into one command buffer
 * is still possible and still worthwhile (any sub-sequence with no
 * intervening host readback, e.g. the RMSNorm-apply -> quantize ->
 * gate/up GEMV -> activation chain), but doing it correctly means
 * threading a shared, not-yet-committed command buffer through several
 * dispatch calls and is a real restructuring, not a one-line change --
 * flagged here as the natural next optimization pass, same as the CUDA
 * port flags its own un-tiled GEMM. What IS done here, matching the CUDA
 * port's actual necessity rather than its stream-batching optimization:
 * every point where CUDA's code needs a real host readback (RMSNorm's
 * partial-sum reduction, the KV-cache write, the final logits readback)
 * is exactly where this file's kernel sequence was already forced to
 * split across command buffers regardless -- so the "obviously correct,
 * not obviously fast" per-kernel-commit version is also the version with
 * the least code to get wrong, which is the right tradeoff for a first
 * Metal port.
 *
 * UNVERIFIED, same standing caveat as the rest of this codebase: no
 * Metal GPU/toolchain was available while writing this.
 */
public final class LlamaLayer {

    private LlamaLayer() {
    }

    /** Which FFN activation this layer's weights expect. Identical to the CUDA/OpenCL ports' enum. */
    public enum ActivationType {
        /** out = gate * sigmoid(gate) * up. Llama/Mistral/Qwen-family default. */
        SWIGLU,
        /** out = gelu(gate). UNGATED -- see the CUDA LlamaLayer's ActivationType.GELU javadoc for the GGUF-mapping caveat, unchanged here. */
        GELU,
        /** out = gelu(gate) * up. Same gated shape as SWIGLU, GeLU gate instead of SiLU. */
        GEGLU
    }

    /** Mirrors the CUDA/OpenCL ports' Config exactly -- architecture fields, RoPE scaling, activationType, prefill batch sizing. Field-for-field identical; see the CUDA LlamaLayer.Config javadoc for the full metadata-key mapping this reads from a GGUF file. */
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
        public double rope_scaling_factor = 1.0;
        public double rope_scaling_low_freq_factor = 1.0;
        public double rope_scaling_high_freq_factor = 4.0;
        public double rope_scaling_orig_context_length = 8192.0;

        public ActivationType activationType = ActivationType.SWIGLU;

        /** Upper bound on prefill batch size (T) this layer's GpuState will allocate scratch buffers for. 0 disables prefill entirely. */
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

        /** Reads every architecture-derived field from a loaded GGUF file's metadata section. Identical mapping to the CUDA port's Config.fromGguf -- see that method's javadoc for the full {arch}.* key list. */
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

    /** GPU-resident weights for ONE decoder layer. Same tensor set as the CUDA/OpenCL ports; MetalBuffer in place of long/MemorySegment. */
    public static final class GpuWeights implements AutoCloseable {
        final GpuContext ctx;
        final MetalBuffer wq_q8_0, wk_q8_0, wv_q8_0;
        final MetalBuffer wo_f32;
        final MetalBuffer w_gate_q8_0, w_up_q8_0, w_down_q8_0;
        final MetalBuffer attn_norm_gamma, ffn_norm_gamma;

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

    /** Per-sequence GPU state: KV cache, RoPE tables, decode-path scratch, PLUS prefill-path batch scratch. Same shape as the CUDA/OpenCL ports' GpuState; MetalBuffer in place of long. */
    public static final class GpuState implements AutoCloseable {
        final GpuContext ctx;
        final int kvDim;
        final int maxBatchT;

        // ---- decode-path ----
        final MetalBuffer k_cache_f32, v_cache_f32;
        final MetalBuffer cos_table, sin_table;
        final MetalBuffer xNorm, xNormQ8;
        final MetalBuffer qSplit, kNew, vNew;
        final MetalBuffer scores;
        final MetalBuffer attnOut, attnProj;
        final MetalBuffer ffnNorm, ffnNormQ8;
        final MetalBuffer gate, up, swigluOut, swigluOutQ8;
        final MetalBuffer ffnDownOut;
        final MetalBuffer rmsPartials;

        // ---- prefill-path batch scratch (0/NULL if maxBatchT == 0) ----
        final MetalBuffer qBatch, kBatch, vBatch;
        final MetalBuffer xNormBatch, xNormQ8Batch;
        final MetalBuffer scoresBatch;
        final MetalBuffer attnOutBatch, attnProjBatch;
        final MetalBuffer ffnNormBatch, ffnNormQ8Batch;
        final MetalBuffer gateBatch, upBatch, swigluOutBatch, swigluOutQ8Batch;
        final MetalBuffer ffnDownOutBatch;
        final MetalBuffer rmsPartialsBatch;
        final MetalBuffer rmsRowValuesBatch;
        final MetalBuffer positionsBatch;

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
            int maxGroups = (maxLen + MetalKernelSource.RMSNORM_WORKGROUP_SIZE - 1) / MetalKernelSource.RMSNORM_WORKGROUP_SIZE;
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
                this.qBatch = MetalBuffer.NULL; this.kBatch = MetalBuffer.NULL; this.vBatch = MetalBuffer.NULL;
                this.xNormBatch = MetalBuffer.NULL; this.xNormQ8Batch = MetalBuffer.NULL;
                this.scoresBatch = MetalBuffer.NULL;
                this.attnOutBatch = MetalBuffer.NULL; this.attnProjBatch = MetalBuffer.NULL;
                this.ffnNormBatch = MetalBuffer.NULL; this.ffnNormQ8Batch = MetalBuffer.NULL;
                this.gateBatch = MetalBuffer.NULL; this.upBatch = MetalBuffer.NULL;
                this.swigluOutBatch = MetalBuffer.NULL; this.swigluOutQ8Batch = MetalBuffer.NULL;
                this.ffnDownOutBatch = MetalBuffer.NULL;
                this.rmsPartialsBatch = MetalBuffer.NULL; this.rmsRowValuesBatch = MetalBuffer.NULL; this.positionsBatch = MetalBuffer.NULL;
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

        /** Llama-3 NTK-by-parts RoPE frequency correction -- identical derivation to the CUDA/OpenCL ports. Identity when rope_scaling_factor==1.0. */
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

    // ================= device alloc/upload/download helpers =================
    // Every buffer is MTLResourceStorageModeShared (see MetalBindings' class
    // javadoc) -- "upload" is a direct host-memory copy at allocation time,
    // no separate host->device transfer command exists or is needed.

    public static MetalBuffer allocFloats(GpuContext ctx, long count) throws Throwable {
        return allocBytes(ctx, count * ValueLayout.JAVA_FLOAT.byteSize());
    }

    public static MetalBuffer allocBytes(GpuContext ctx, long byteCount) throws Throwable {
        long safeCount = Math.max(byteCount, 1);
        long id = ctx.mtl.newBufferWithLength(ctx.device, safeCount, MetalBindings.DEFAULT_BUFFER_OPTIONS);
        if (id == 0L) {
            throw new IllegalStateException("newBufferWithLength failed (requested " + byteCount + " bytes)");
        }
        long contents = ctx.mtl.bufferContents(id);
        return new MetalBuffer(id, 0L, contents, safeCount);
    }

    public static MetalBuffer uploadBytes(GpuContext ctx, byte[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate(Math.max(data.length, 1));
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_BYTE, 0, data.length);
            long id = ctx.mtl.newBufferWithBytes(ctx.device, host, host.byteSize(), MetalBindings.DEFAULT_BUFFER_OPTIONS);
            if (id == 0L) {
                throw new IllegalStateException("newBufferWithBytes failed (" + data.length + " bytes)");
            }
            long contents = ctx.mtl.bufferContents(id);
            return new MetalBuffer(id, 0L, contents, host.byteSize());
        }
    }

    public static MetalBuffer uploadFloats(GpuContext ctx, float[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) Math.max(data.length, 1) * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            long id = ctx.mtl.newBufferWithBytes(ctx.device, host, host.byteSize(), MetalBindings.DEFAULT_BUFFER_OPTIONS);
            if (id == 0L) {
                throw new IllegalStateException("newBufferWithBytes failed (" + data.length + " floats)");
            }
            long contents = ctx.mtl.bufferContents(id);
            return new MetalBuffer(id, 0L, contents, host.byteSize());
        }
    }

    public static MetalBuffer uploadInts(GpuContext ctx, int[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) Math.max(data.length, 1) * ValueLayout.JAVA_INT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);
            long id = ctx.mtl.newBufferWithBytes(ctx.device, host, host.byteSize(), MetalBindings.DEFAULT_BUFFER_OPTIONS);
            if (id == 0L) {
                throw new IllegalStateException("newBufferWithBytes failed (" + data.length + " ints)");
            }
            long contents = ctx.mtl.bufferContents(id);
            return new MetalBuffer(id, 0L, contents, host.byteSize());
        }
    }

    /** Re-writes an ALREADY-ALLOCATED shared buffer's contents in place -- a direct host memcpy, the Metal replacement for the CUDA port's uploadIntsInto/uploadFloatsInto (which needed a real cuMemcpyHtoD). */
    private static void writeInto(MetalBuffer dst, MemorySegment src) {
        MemorySegment dstSeg = MemorySegment.ofAddress(dst.hostAddress()).reinterpret(src.byteSize());
        MemorySegment.copy(src, 0, dstSeg, 0, src.byteSize());
    }

    private static void uploadIntsInto(GpuContext ctx, MetalBuffer device, int[] data) {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_INT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);
            writeInto(device, host);
        }
    }

    private static void uploadFloatsInto(GpuContext ctx, MetalBuffer device, float[] data) {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            writeInto(device, host);
        }
    }

    /** Blocking readback of a float buffer. Since every prior kernel dispatch in this file already commits+waits (see class javadoc), this is a plain host read of already-ready memory -- no additional sync needed, but the method stays synchronous/blocking to match the CUDA/OpenCL ports' API contract exactly. */
    public static float[] downloadFloats(GpuContext ctx, MetalBuffer buf, int count) throws Throwable {
        float[] out = new float[count];
        if (count == 0) {
            return out;
        }
        MemorySegment seg = MemorySegment.ofAddress(buf.hostAddress()).reinterpret((long) count * ValueLayout.JAVA_FLOAT.byteSize());
        MemorySegment.copy(seg, ValueLayout.JAVA_FLOAT, 0, out, 0, count);
        return out;
    }

    public static byte[] downloadBytes(GpuContext ctx, MetalBuffer buf, int count) throws Throwable {
        byte[] out = new byte[count];
        if (count == 0) {
            return out;
        }
        MemorySegment seg = MemorySegment.ofAddress(buf.hostAddress()).reinterpret(count);
        MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, 0, out, 0, count);
        return out;
    }

    /** No-op on this backend: every dispatch in this file already blocks on waitUntilCompleted before returning (see class javadoc), so there is never outstanding work to wait for by the time a caller reaches this call. Kept for API parity with the CUDA/OpenCL ports -- benchmark code written against those ports calls LlamaLayer.finish(ctx) after a warmup loop and expects it to mean "everything issued so far has actually finished". */
    public static void finish(GpuContext ctx) throws Throwable {
        // Intentionally empty -- see method javadoc.
    }

    public static void freeQuietly(GpuContext ctx, MetalBuffer buf) {
        try {
            if (buf != null && !buf.isNull()) {
                ctx.mtl.release(buf.id);
            }
        } catch (Throwable t) {
            // best-effort cleanup
        }
    }

    // =====================================================================
    // THE DECODE-PATH DECODER LAYER (single token, M=1)
    // =====================================================================

    /** Runs one decoder layer for the CURRENT token, in place on x. x: MetalBuffer, [dim] floats, in/out. */
    public static void forward_layer(MetalBuffer x, GpuWeights w, GpuState s, Config cfg, GpuContext ctx) throws Throwable {
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
            // === 1. Attention block: x = x + attn(rms_norm(x)) ===
            rmsNorm(ctx, x, w.attn_norm_gamma, s.xNorm, dim, cfg.norm_eps, s.rmsPartials);
            quantizeActivationQ8_0(ctx, s.xNorm, s.xNormQ8, dim);

            q8_0GemvSplit(ctx, s.xNormQ8, w.wq_q8_0, s.qSplit, numHeads, headDim, dim);
            q8_0GemvSplit(ctx, s.xNormQ8, w.wk_q8_0, s.kNew, kvHeads, headDim, dim);
            q8_0GemvSplit(ctx, s.xNormQ8, w.wv_q8_0, s.vNew, kvHeads, headDim, dim);

            ropeApplySplit(ctx, s.qSplit, s.cos_table, s.sin_table, numHeads, headDim, s.pos * halfDim);
            ropeApplySplit(ctx, s.kNew, s.cos_table, s.sin_table, kvHeads, headDim, s.pos * halfDim);

            writeIntoCache(s.kNew, s.k_cache_f32, (long) s.pos * kvDim, kvDim);
            writeIntoCache(s.vNew, s.v_cache_f32, (long) s.pos * kvDim, kvDim);

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
                q8_0GemvPlain(ctx, s.ffnNormQ8, w.w_gate_q8_0, s.gate, hidden, dim);
                geluActivate(ctx, s.gate, s.swigluOut, hidden);
            }
        }
    }

    /** Runs the full model for one token: embedding -> N decoder layers -> final norm -> LM head, greedy argmax. For non-greedy sampling, use generate_logits and feed the result to Sampler instead. */
    public static int generate_token(
            MetalBuffer tokenEmbeddingDevice,
            GpuWeights[] layers,
            MetalBuffer finalNormGammaDevice,
            MetalBuffer lmHeadDevice,
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

    /** Same forward pass as generate_token, but returns the raw logits instead of collapsing to argmax -- the entry point for Sampler-based decoding. */
    public static float[] generate_logits(
            MetalBuffer tokenEmbeddingDevice,
            GpuWeights[] layers,
            MetalBuffer finalNormGammaDevice,
            MetalBuffer lmHeadDevice,
            GpuState[] states,
            Config cfg,
            GpuContext ctx,
            int vocabSize) throws Throwable {

        MetalBuffer x = tokenEmbeddingDevice;

        for (int l = 0; l < layers.length; l++) {
            forward_layer(x, layers[l], states[l], cfg, ctx);
        }

        float[] logits = finalLogits(x, finalNormGammaDevice, lmHeadDevice, cfg, ctx, vocabSize);
        for (GpuState st : states) {
            st.pos++;
        }
        return logits;
    }

    /** Final norm + LM head + host readback on an ARBITRARY already-computed row buffer -- does NOT run the decoder layers. Shared by both the decode path (xRow = the [dim] decode scratch buffer) and the prefill path (xRow = xBatch.withOffset((T-1)*dim*4)). */
    public static float[] finalLogits(MetalBuffer xRow, MetalBuffer finalNormGammaDevice, MetalBuffer lmHeadDevice,
            Config cfg, GpuContext ctx, int vocabSize) throws Throwable {
        MetalBuffer normed = allocFloats(ctx, cfg.dim);
        MetalBuffer logitsDevice = allocFloats(ctx, vocabSize);
        int maxGroups = (cfg.dim + MetalKernelSource.RMSNORM_WORKGROUP_SIZE - 1) / MetalKernelSource.RMSNORM_WORKGROUP_SIZE;
        MetalBuffer partials = allocFloats(ctx, maxGroups);

        try {
            synchronized (ctx.dispatchLock) {
                rmsNorm(ctx, xRow, finalNormGammaDevice, normed, cfg.dim, cfg.norm_eps, partials);
                f32Gemv(ctx, normed, lmHeadDevice, logitsDevice, cfg.dim, vocabSize);
            }
            return downloadFloats(ctx, logitsDevice, vocabSize);
        } finally {
            freeQuietly(ctx, normed);
            freeQuietly(ctx, logitsDevice);
            freeQuietly(ctx, partials);
        }
    }

    // =====================================================================
    // THE BATCHED PREFILL PATH: T tokens at once, one layer.
    // =====================================================================

    /**
     * Runs one decoder layer over a BATCH of T prompt positions at once.
     * Same single-call-only, startPos==0-only contract as the CUDA port
     * -- see {@code com.github.gbenroscience.gpu.llm.cuda.LlamaLayer#prefill_layer}'s
     * javadoc for the full "why not chunkable" rationale, which applies
     * unchanged here (attn_scores_causal_batched attends only within this
     * call's own qBatch/kBatch/vBatch, never the persistent KV cache).
     *
     * @throws IllegalStateException if cfg.max_prefill_batch == 0 or T exceeds it.
     */
    public static void prefill_layer(MetalBuffer xBatch, GpuWeights w, GpuState s, Config cfg, GpuContext ctx,
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
                            + "solely within this call's own T rows, not the persistent KV cache. See this method's javadoc.");
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
            uploadIntsInto(ctx, s.positionsBatch, positions);

            // === 1. Attention block ===
            rmsNormRows(ctx, xBatch, w.attn_norm_gamma, s.xNormBatch, dim, cfg.norm_eps, T, s.rmsPartialsBatch, s.rmsRowValuesBatch);
            quantizeActivationQ8_0(ctx, s.xNormBatch, s.xNormQ8Batch, T * dim);

            q8_0GemmTiled(ctx, s.xNormQ8Batch, w.wq_q8_0, s.qBatch, T, qRowStride, dim);
            q8_0GemmTiled(ctx, s.xNormQ8Batch, w.wk_q8_0, s.kBatch, T, kvDim, dim);
            q8_0GemmTiled(ctx, s.xNormQ8Batch, w.wv_q8_0, s.vBatch, T, kvDim, dim);

            ropeApplyPairwiseRows(ctx, s.qBatch, s.cos_table, s.sin_table, numHeads, headDim, s.positionsBatch, T);
            ropeApplyPairwiseRows(ctx, s.kBatch, s.cos_table, s.sin_table, kvHeads, headDim, s.positionsBatch, T);

            writeIntoCache(s.kBatch, s.k_cache_f32, (long) startPos * kvDim, T * kvDim);
            writeIntoCache(s.vBatch, s.v_cache_f32, (long) startPos * kvDim, T * kvDim);

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
        }

        s.pos = startPos + T;
    }

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
    // Kernel dispatch helpers. All assume the caller holds ctx.dispatchLock.
    // See class javadoc for the per-dispatch commit+wait discipline.
    // =====================================================================

    static void rmsNorm(GpuContext ctx, MetalBuffer x, MetalBuffer gamma, MetalBuffer out,
            int features, double eps, MetalBuffer partials) throws Throwable {
        int wgSize = MetalKernelSource.RMSNORM_WORKGROUP_SIZE;
        int numGroups = (features + wgSize - 1) / wgSize;

        runKernel1D(ctx, ctx.kRmsnormPartialSumsq, (long) numGroups * wgSize, wgSize,
                wgSize * (int) ValueLayout.JAVA_FLOAT.byteSize(),
                buf(x), buf(partials), scalarI(features));

        float[] partialHost = downloadFloats(ctx, partials, numGroups);
        double sumSq = 0.0;
        for (float p : partialHost) {
            sumSq += p;
        }
        float rms = (float) (1.0 / Math.sqrt(sumSq / features + eps));

        runKernel1D(ctx, ctx.kRmsnormApply, features, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(x), buf(gamma), buf(out), scalarF(rms), scalarI(features));
    }

    static void quantizeActivationQ8_0(GpuContext ctx, MetalBuffer x, MetalBuffer outQ8, int len) throws Throwable {
        int blockSize = MetalKernelSource.QUANTIZE_BLOCK_SIZE;
        runKernel1D(ctx, ctx.kQuantizeActivationQ8_0, len, blockSize,
                blockSize * (int) ValueLayout.JAVA_FLOAT.byteSize(),
                buf(x), buf(outQ8), scalarI(len));
    }

    static void q8_0GemvSplit(GpuContext ctx, MetalBuffer xQ8, MetalBuffer wQ8, MetalBuffer out,
            int heads, int headDim, int K) throws Throwable {
        runKernel1D(ctx, ctx.kQ8_0GemvSplit, (long) heads * (headDim / 2), MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(xQ8), buf(wQ8), buf(out), scalarI(heads), scalarI(headDim), scalarI(K));
    }

    static void q8_0GemvPlain(GpuContext ctx, MetalBuffer xQ8, MetalBuffer wQ8, MetalBuffer out,
            int N, int K) throws Throwable {
        runKernel1D(ctx, ctx.kQ8_0GemvPlain, N, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(xQ8), buf(wQ8), buf(out), scalarI(N), scalarI(K));
    }

    static void ropeApplySplit(GpuContext ctx, MetalBuffer buf, MetalBuffer cosTable, MetalBuffer sinTable,
            int heads, int headDim, int cosSinOffset) throws Throwable {
        runKernel1D(ctx, ctx.kRopeApplySplit, (long) heads * (headDim / 2), MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(buf), buf(cosTable), buf(sinTable), scalarI(heads), scalarI(headDim), scalarI(cosSinOffset));
    }

    static void attnScores(GpuContext ctx, MetalBuffer qAllHeads, MetalBuffer kCache, MetalBuffer scores,
            int qHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive, float rsqrtD) throws Throwable {
        runKernel1D(ctx, ctx.kAttnScores, posInclusive, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(qAllHeads), buf(kCache), buf(scores), scalarI(qHeadOff), scalarI(headDim),
                scalarI(kvDim), scalarI(kvHeadOff), scalarI(posInclusive), scalarF(rsqrtD));
    }

    static void softmaxInplace(GpuContext ctx, MetalBuffer scores, int len) throws Throwable {
        int localSize = nextPow2(Math.max(len, 1));
        runKernel1DExact(ctx, ctx.kSoftmaxInplace, localSize, localSize,
                localSize * (int) ValueLayout.JAVA_FLOAT.byteSize(),
                buf(scores), scalarI(len));
    }

    private static int nextPow2(int v) {
        int p = 1;
        while (p < v) {
            p <<= 1;
        }
        return Math.min(p, 256);
    }

    static void attnWeightedSum(GpuContext ctx, MetalBuffer scores, MetalBuffer vCache, MetalBuffer attnOutAllHeads,
            int outHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive) throws Throwable {
        runKernel1D(ctx, ctx.kAttnWeightedSum, headDim, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(scores), buf(vCache), buf(attnOutAllHeads), scalarI(outHeadOff), scalarI(headDim),
                scalarI(kvDim), scalarI(kvHeadOff), scalarI(posInclusive));
    }

    static void swigluActivate(GpuContext ctx, MetalBuffer gate, MetalBuffer up, MetalBuffer out, int len) throws Throwable {
        runKernel1D(ctx, ctx.kSwigluActivate, len, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(gate), buf(up), buf(out), scalarI(len));
    }

    static void gegluActivate(GpuContext ctx, MetalBuffer gate, MetalBuffer up, MetalBuffer out, int len) throws Throwable {
        runKernel1D(ctx, ctx.kGegluActivate, len, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(gate), buf(up), buf(out), scalarI(len));
    }

    static void geluActivate(GpuContext ctx, MetalBuffer gate, MetalBuffer out, int len) throws Throwable {
        runKernel1D(ctx, ctx.kGeluActivate, len, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(gate), buf(out), scalarI(len));
    }

    private static void residualAdd(GpuContext ctx, MetalBuffer x, MetalBuffer y, int len) throws Throwable {
        runKernel1D(ctx, ctx.kResidualAdd, len, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(x), buf(y), scalarI(len));
    }

    static void f32Gemv(GpuContext ctx, MetalBuffer a, MetalBuffer B, MetalBuffer out, int K, int N) throws Throwable {
        runKernel1D(ctx, ctx.kF32Gemv, N, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(a), buf(B), buf(out), scalarI(K), scalarI(N));
    }

    /** Host-side buffer-to-buffer copy at a byte offset -- both src and cache are MTLResourceStorageModeShared, so this is a plain MemorySegment.copy, not a real transfer command. Direct replacement for the CUDA port's writeIntoCache (which needed a genuine device round trip). */
    private static void writeIntoCache(MetalBuffer src, MetalBuffer cache, long elementOffset, int len) {
        long byteLen = (long) len * ValueLayout.JAVA_FLOAT.byteSize();
        MemorySegment srcSeg = MemorySegment.ofAddress(src.hostAddress()).reinterpret(byteLen);
        long dstByteOffset = elementOffset * ValueLayout.JAVA_FLOAT.byteSize();
        MemorySegment dstSeg = MemorySegment.ofAddress(cache.hostAddress() + dstByteOffset).reinterpret(byteLen);
        MemorySegment.copy(srcSeg, 0, dstSeg, 0, byteLen);
    }

    // ---- batched prefill dispatch ----

    static void rmsNormRows(GpuContext ctx, MetalBuffer x, MetalBuffer gamma, MetalBuffer out,
            int features, double eps, int T, MetalBuffer partialsBatch, MetalBuffer rmsRowValuesBatch) throws Throwable {
        int wgSize = MetalKernelSource.RMSNORM_WORKGROUP_SIZE;
        int numGroups = (features + wgSize - 1) / wgSize;

        runKernel2D(ctx, ctx.kRmsnormPartialSumsqRows, (long) numGroups * wgSize, T, wgSize,
                wgSize * (int) ValueLayout.JAVA_FLOAT.byteSize(),
                buf(x), buf(partialsBatch), scalarI(features), scalarI(numGroups));

        float[] partialHost = downloadFloats(ctx, partialsBatch, T * numGroups);
        float[] rmsPerRow = new float[T];
        for (int row = 0; row < T; row++) {
            double sumSq = 0.0;
            for (int g = 0; g < numGroups; g++) {
                sumSq += partialHost[row * numGroups + g];
            }
            rmsPerRow[row] = (float) (1.0 / Math.sqrt(sumSq / features + eps));
        }
        uploadFloatsInto(ctx, rmsRowValuesBatch, rmsPerRow);

        runKernel2D(ctx, ctx.kRmsnormApplyRows, features, T, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(x), buf(gamma), buf(out), buf(rmsRowValuesBatch), scalarI(features));
    }

    static void q8_0GemmTiled(GpuContext ctx, MetalBuffer xQ8, MetalBuffer wQ8, MetalBuffer out, int T, int N, int K) throws Throwable {
        runKernel2D(ctx, ctx.kQ8_0GemmTiled, N, T, MetalKernelSource.GEMM_TILE_N, 0,
                buf(xQ8), buf(wQ8), buf(out), scalarI(T), scalarI(N), scalarI(K));
    }

    static void f32GemmTiled(GpuContext ctx, MetalBuffer a, MetalBuffer B, MetalBuffer out, int T, int K, int N) throws Throwable {
        runKernel2D(ctx, ctx.kF32GemmTiled, N, T, MetalKernelSource.GEMM_TILE_N, 0,
                buf(a), buf(B), buf(out), scalarI(T), scalarI(K), scalarI(N));
    }

    static void ropeApplyPairwiseRows(GpuContext ctx, MetalBuffer buf, MetalBuffer cosTable, MetalBuffer sinTable,
            int heads, int headDim, MetalBuffer positionsBatch, int T) throws Throwable {
        runKernel2D(ctx, ctx.kRopeApplyPairwiseRows, (long) heads * (headDim / 2), T, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(buf), buf(cosTable), buf(sinTable), scalarI(heads), scalarI(headDim), buf(positionsBatch));
    }

    static void attnScoresCausalBatched(GpuContext ctx, MetalBuffer qAll, MetalBuffer kAll, MetalBuffer scoresBatch,
            int qRowStride, int kRowStride, int qHeadOff, int kHeadOff, int headDim, int T, float rsqrtD) throws Throwable {
        runKernel2D(ctx, ctx.kAttnScoresCausalBatched, T, T, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(qAll), buf(kAll), buf(scoresBatch), scalarI(qRowStride), scalarI(kRowStride),
                scalarI(qHeadOff), scalarI(kHeadOff), scalarI(headDim), scalarI(T), scalarF(rsqrtD));
    }

    static void softmaxInplaceRows(GpuContext ctx, MetalBuffer scoresBatch, int T) throws Throwable {
        int localSize = nextPow2(Math.max(T, 1));
        // One threadgroup PER ROW: grid = T*localSize threads total, threadgroup width = localSize.
        runKernel1DExact(ctx, ctx.kSoftmaxInplaceRows, (long) T * localSize, localSize,
                localSize * (int) ValueLayout.JAVA_FLOAT.byteSize(),
                buf(scoresBatch), scalarI(T));
    }

    static void attnWeightedSumCausalBatched(GpuContext ctx, MetalBuffer scoresBatch, MetalBuffer vAll, MetalBuffer attnOutBatch,
            int vRowStride, int outRowStride, int vHeadOff, int outHeadOff, int headDim, int T) throws Throwable {
        runKernel2D(ctx, ctx.kAttnWeightedSumCausalBatched, headDim, T, MetalKernelSource.DEFAULT_BLOCK_SIZE, 0,
                buf(scoresBatch), buf(vAll), buf(attnOutBatch), scalarI(vRowStride), scalarI(outRowStride),
                scalarI(vHeadOff), scalarI(outHeadOff), scalarI(headDim), scalarI(T));
    }

    // =====================================================================
    // ===================== ENCODER PLUMBING =============================
    // =====================================================================

    /** Marker wrapping either a MetalBuffer (kernel buffer argument) or a scalar constant (kernel constant argument), consumed positionally by runKernel1D/2D -- the Metal replacement for the CUDA port's raw void* kernelParams array. */
    private sealed interface Arg permits BufArg, ScalarArg {
    }
    private record BufArg(MetalBuffer value) implements Arg {
    }
    private record ScalarArg(MemorySegment value) implements Arg {
    }

    private static Arg buf(MetalBuffer b) {
        return new BufArg(b);
    }

    private static Arg scalarI(int v) {
        MemorySegment s = Arena.ofAuto().allocate(ValueLayout.JAVA_INT);
        s.set(ValueLayout.JAVA_INT, 0, v);
        return new ScalarArg(s);
    }

    private static Arg scalarF(float v) {
        MemorySegment s = Arena.ofAuto().allocate(ValueLayout.JAVA_FLOAT);
        s.set(ValueLayout.JAVA_FLOAT, 0, v);
        return new ScalarArg(s);
    }

    /** 1D dispatch: rounds nothing -- dispatchThreads handles the ragged edge itself (see MetalBindings.dispatchThreads' javadoc). threadgroupMemBytes>0 binds it at threadgroup(0). */
    private static void runKernel1D(GpuContext ctx, long pipeline, long totalThreads, int threadgroupSize,
            int threadgroupMemBytes, Arg... args) throws Throwable {
        dispatch(ctx, pipeline, totalThreads, 1, 1, threadgroupSize, 1, 1, threadgroupMemBytes, args);
    }

    /** Like runKernel1D but the exact grid size is passed in (used for the two "one threadgroup per row" softmax kernels, where totalThreads must be a whole multiple of threadgroupSize -- T*localSize, not T alone). */
    private static void runKernel1DExact(GpuContext ctx, long pipeline, long totalThreads, int threadgroupSize,
            int threadgroupMemBytes, Arg... args) throws Throwable {
        dispatch(ctx, pipeline, totalThreads, 1, 1, threadgroupSize, 1, 1, threadgroupMemBytes, args);
    }

    /** 2D dispatch: grid.x = cols (exact total, e.g. N or T*localSize), grid.y = rows (exact, e.g. T). threadgroup is (blockSizeX, 1, 1) -- matches the CUDA port's launch2D(rows, cols, blockSizeX) shape with rows/cols swapped into Metal's (x=cols,y=rows) grid convention. */
    private static void runKernel2D(GpuContext ctx, long pipeline, long cols, int rows, int blockSizeX,
            int threadgroupMemBytes, Arg... args) throws Throwable {
        dispatch(ctx, pipeline, cols, rows, 1, blockSizeX, 1, 1, threadgroupMemBytes, args);
    }

    private static void dispatch(GpuContext ctx, long pipeline, long gx, long gy, long gz,
            long tx, long ty, long tz, int threadgroupMemBytes, Arg[] args) throws Throwable {
        long cmdBuf = ctx.mtl.commandBuffer(ctx.commandQueue);
        long encoder = ctx.mtl.computeCommandEncoder(cmdBuf);
        ctx.mtl.setComputePipelineState(encoder, pipeline);

        for (int i = 0; i < args.length; i++) {
            Arg a = args[i];
            if (a instanceof BufArg ba) {
                ctx.mtl.setBuffer(encoder, ba.value().id, ba.value().offset, i);
            } else if (a instanceof ScalarArg sa) {
                ctx.mtl.setBytes(encoder, sa.value(), sa.value().byteSize(), i);
            }
        }
        if (threadgroupMemBytes > 0) {
            ctx.mtl.setThreadgroupMemoryLength(encoder, threadgroupMemBytes, 0);
        }

        ctx.mtl.dispatchThreads(encoder, gx, gy, gz, tx, ty, tz);
        ctx.mtl.endEncoding(encoder);
        ctx.mtl.commit(cmdBuf);
        ctx.mtl.waitUntilCompleted(cmdBuf);

        long status = ctx.mtl.commandBufferStatus(cmdBuf);
        if (status == MetalBindings.MTL_COMMAND_BUFFER_STATUS_ERROR) {
            long error = ctx.mtl.commandBufferError(cmdBuf);
            String msg = (error != 0L) ? ctx.mtl.nsStringToJava(ctx.mtl.send0(error, "localizedDescription")) : "(no NSError)";
            throw new IllegalStateException("Metal command buffer failed: " + msg);
        }
    }
}