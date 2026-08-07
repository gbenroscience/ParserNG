package com.github.gbenroscience.gpu.llm.opencl;
 
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.NoSuchElementException;

/**
 * OpenCL counterpart of {@code com.github.gbenroscience.gpu.cuda.llm.LlamaLayer}.
 * Same two entry points (forward_layer for one token, prefill_layer for a
 * batch of T prompt positions), same per-layer algorithm, same activation-
 * type dispatch. Read this alongside the CUDA original -- the algorithmic
 * commentary there (GQA grouping, RoPE layouts, why prefill_layer only
 * supports startPos=0, what "batched" means) applies unchanged and is not
 * repeated here.
 *
 * WHAT ACTUALLY CHANGED FOR THE PORT (everything else is a mechanical
 * CUDA-driver-API -> OpenCL-API translation):
 *
 * 1. DEVICE BUFFERS ARE {@link MemorySegment} cl_mem HANDLES, NOT {@code long}
 *    ADDRESSES. CUDA's CUdeviceptr is a real device virtual address, so
 *    the CUDA original does raw pointer-plus-byte-offset arithmetic on
 *    the host in two places. OpenCL's cl_mem is an opaque handle with no
 *    such arithmetic available. Both spots are handled differently here:
 *      - KV-cache writes at a growing per-token offset (writeIntoCache):
 *        the CUDA version does a synchronous device->host->device round
 *        trip specifically because that's the tool cuMemcpy gives you for
 *        an offset write. OpenCL's clEnqueueCopyBuffer takes independent
 *        source AND destination byte offsets and copies device-to-device
 *        directly -- no host round trip needed, and (being enqueued on
 *        the same in-order queue as everything else) no explicit sync
 *        needed either. This is a genuine improvement over the CUDA
 *        path, not just a translation, and is called out again at
 *        writeIntoCache below.
 *      - Extracting one row out of a [T,dim] prefill batch buffer to feed
 *        finalLogits (CUDA: {@code xBatch + (T-1)*dim*sizeof(float)}): this
 *        can't be done inside LlamaLayer at all without an offset kernel
 *        argument, so it isn't -- finalLogits here still expects a plain
 *        [dim]-sized buffer at element 0, same as the CUDA version. The
 *        row-copy (also via clEnqueueCopyBuffer) happens one layer up, in
 *        LlamaOpenCLEngine.processPrompt -- see that class's javadoc.
 *
 * 2. KERNEL ARGUMENTS ARE SET ONE AT A TIME (clSetKernelArg per index)
 *    INSTEAD OF ONE COMBINED PARAMETER ARRAY. This is simply how OpenCL's
 *    API is shaped (cuLaunchKernel takes one {@code void**} array; OpenCL has
 *    no equivalent, only per-index clSetKernelArg calls before
 *    clEnqueueNDRangeKernel) -- not a design choice made here.
 *
 * 3. NO DYNAMIC SHARED-MEMORY BYTE COUNT AT LAUNCH TIME. CUDA's
 *    cuLaunchKernel takes a shared-memory-size argument; every kernel
 *    here that needs local memory instead declares a FIXED-SIZE __local
 *    array sized to the largest work-group this file ever dispatches for
 *    it -- see KernelSource's class javadoc. Net effect: this file's
 *    launch1D/launch2D take one fewer parameter than the CUDA original's.
 *
 * 4. QUEUE, NOT STREAM/CONTEXT-CURRENT. There is no OpenCL equivalent of
 *    cuCtxSetCurrent (a context is not implicitly "current" per-thread
 *    the way a CUDA context is) -- every enqueue call here names its
 *    queue explicitly (ctx.queue), so that call is simply absent.
 *    dispatchLock is still held around each forward_layer/prefill_layer
 *    call for the same reason the CUDA port holds it: the in-order queue
 *    and the KV-cache/scratch buffers it's building on are shared
 *    per-context state; the lock keeps a single sequence's kernel
 *    sequence from interleaving with another thread's.
 *
 * SYNC DISCIPLINE: identical rationale to the CUDA original. Kernels
 * enqueued on the same in-order command queue execute in issued order
 * without host intervention; the ICD guarantees that. clEnqueueReadBuffer
 * called with blocking=CL_TRUE is the actual synchronization point
 * wherever the HOST needs a result (RMSNorm partial-sum readback, final
 * logits readback) -- it both waits for every earlier-enqueued command on
 * this queue to finish AND performs the transfer. A single clFinish is
 * kept at the end of forward_layer/prefill_layer, not required for
 * correctness, kept for predictable per-token/per-batch timing if the
 * caller profiles calls to these methods (mirrors the CUDA port's
 * end-of-call cuCtxSynchronize).
 *
 * UNVERIFIED: no OpenCL platform/device was available while writing this
 * port -- see GpuContext's class javadoc for the same standing caveat.
 */
public final class LlamaLayer {

    private LlamaLayer() {
    }

    /** Which FFN activation this layer's weights expect. Same three options as the CUDA original. */
    public enum ActivationType {
        /** out = gate * sigmoid(gate) * up. Llama/Mistral/Qwen-family default. */
        SWIGLU,
        /**
         * out = gelu(gate). UNGATED -- "up" is still loaded by GpuWeights
         * (fromGguf always loads ffn_up.weight) but not read by this
         * activation. See the CUDA original's javadoc for the GGUF-
         * tensor-mapping caveat on true ungated models.
         */
        GELU,
        /** out = gelu(gate) * up. Same gated shape as SWIGLU, GeLU gate instead of SiLU. */
        GEGLU
    }

    /** Identical shape and fromGguf mapping to the CUDA original's Config -- architecture is a model-file fact, independent of which GPU API runs it. */
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

        public ActivationType activationType = ActivationType.SWIGLU;

        /** 0 disables prefill entirely -- see CUDA Config's javadoc. */
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

        /** Same GGUF metadata mapping as the CUDA original's Config.fromGguf -- see that method's javadoc for the full per-key rationale. */
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

    /** GPU-resident weights for ONE decoder layer. Same tensors, same GGUF names as the CUDA original -- only the field TYPE changed (MemorySegment cl_mem, not long CUdeviceptr). */
    public static final class GpuWeights implements AutoCloseable {
        final GpuContext ctx;
        final MemorySegment wq_q8_0, wk_q8_0, wv_q8_0;
        final MemorySegment wo_f32;
        final MemorySegment w_gate_q8_0, w_up_q8_0, w_down_q8_0;
        final MemorySegment attn_norm_gamma, ffn_norm_gamma;

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

    /** Per-sequence GPU state: KV cache, RoPE tables, decode-path scratch, PLUS prefill-path batch scratch. Same layout/sizing as the CUDA original -- only the field TYPE changed. */
    public static final class GpuState implements AutoCloseable {
        final GpuContext ctx;
        final int kvDim;
        final int maxBatchT;

        // ---- decode-path ----
        final MemorySegment k_cache_f32, v_cache_f32;
        final MemorySegment cos_table, sin_table;
        final MemorySegment xNorm, xNormQ8;
        final MemorySegment qSplit, kNew, vNew;
        final MemorySegment scores;
        final MemorySegment attnOut, attnProj;
        final MemorySegment ffnNorm, ffnNormQ8;
        final MemorySegment gate, up, swigluOut, swigluOutQ8;
        final MemorySegment ffnDownOut;
        final MemorySegment rmsPartials;

        // ---- prefill-path batch scratch (0-length/null-equivalent if maxBatchT == 0) ----
        final MemorySegment qBatch, kBatch, vBatch;
        final MemorySegment xNormBatch, xNormQ8Batch;
        final MemorySegment scoresBatch;
        final MemorySegment attnOutBatch, attnProjBatch;
        final MemorySegment ffnNormBatch, ffnNormQ8Batch;
        final MemorySegment gateBatch, upBatch, swigluOutBatch, swigluOutQ8Batch;
        final MemorySegment ffnDownOutBatch;
        final MemorySegment rmsPartialsBatch;
        final MemorySegment rmsRowValuesBatch;
        final MemorySegment positionsBatch;

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
            precomputeRope(cosHost, sinHost, cfg.max_seq, cfg.head_dim, cfg.rope_theta);
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
                this.qBatch = null; this.kBatch = null; this.vBatch = null;
                this.xNormBatch = null; this.xNormQ8Batch = null;
                this.scoresBatch = null;
                this.attnOutBatch = null; this.attnProjBatch = null;
                this.ffnNormBatch = null; this.ffnNormQ8Batch = null;
                this.gateBatch = null; this.upBatch = null; this.swigluOutBatch = null; this.swigluOutQ8Batch = null;
                this.ffnDownOutBatch = null;
                this.rmsPartialsBatch = null; this.rmsRowValuesBatch = null; this.positionsBatch = null;
            }
        }

        private static int q8_0Bytes(int len) {
            return (len / 32) * 34;
        }

        private static void precomputeRope(float[] cosOut, float[] sinOut, int maxSeq, int headDim, double base) {
            int halfDim = headDim / 2;
            for (int p = 0; p < maxSeq; p++) {
                for (int i = 0; i < halfDim; i++) {
                    double freq = 1.0 / Math.pow(base, (2.0 * i) / headDim);
                    double angle = p * freq;
                    cosOut[p * halfDim + i] = (float) Math.cos(angle);
                    sinOut[p * halfDim + i] = (float) Math.sin(angle);
                }
            }
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

    // ================= device alloc/upload/copy helpers (cl_mem via MemorySegment) =================

    public static MemorySegment allocFloats(GpuContext ctx, long count) throws Throwable {
        return allocBytes(ctx, count * ValueLayout.JAVA_FLOAT.byteSize());
    }

    public static MemorySegment allocBytes(GpuContext ctx, long byteCount) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
            MemorySegment mem = (MemorySegment) ctx.cl.clCreateBuffer.invoke(
                    ctx.context, OpenCLBindings.CL_MEM_READ_WRITE, byteCount, MemorySegment.NULL, errBuf);
            GpuContext.check(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer");
            return mem;
        }
    }

    public static MemorySegment uploadBytes(GpuContext ctx, byte[] data) throws Throwable {
        MemorySegment mem = allocBytes(ctx, data.length);
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate(Math.max(data.length, 1));
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_BYTE, 0, data.length);
            writeBlocking(ctx, mem, host, data.length);
        }
        return mem;
    }

    public static MemorySegment uploadInts(GpuContext ctx, int[] data) throws Throwable {
        MemorySegment mem = allocBytes(ctx, (long) data.length * ValueLayout.JAVA_INT.byteSize());
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) data.length * ValueLayout.JAVA_INT.byteSize();
            MemorySegment host = tmp.allocate(Math.max(byteSize, 1));
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);
            writeBlocking(ctx, mem, host, byteSize);
        }
        return mem;
    }

    public static MemorySegment uploadFloats(GpuContext ctx, float[] data) throws Throwable {
        MemorySegment mem = allocFloats(ctx, data.length);
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) data.length * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment host = tmp.allocate(Math.max(byteSize, 1));
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            writeBlocking(ctx, mem, host, byteSize);
        }
        return mem;
    }

    /** Uploads a freshly-computed int[] (e.g. per-row RoPE positions) into an ALREADY-ALLOCATED device buffer -- no new clCreateBuffer, reuses GpuState's persistent scratch. */
    private static void uploadIntsInto(GpuContext ctx, MemorySegment device, int[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) data.length * ValueLayout.JAVA_INT.byteSize();
            MemorySegment host = tmp.allocate(byteSize);
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_INT, 0, data.length);
            writeBlocking(ctx, device, host, byteSize);
        }
    }

    /** Uploads a freshly-computed float[] (e.g. per-row rms values) into an ALREADY-ALLOCATED device buffer. */
    private static void uploadFloatsInto(GpuContext ctx, MemorySegment device, float[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) data.length * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment host = tmp.allocate(byteSize);
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            writeBlocking(ctx, device, host, byteSize);
        }
    }

    /** Blocking readback of a float buffer -- public so callers outside this package (benchmarks, tests, other engine code) can pull a result off the device without reaching into GpuContext's raw OpenCL handles. Blocking -> also acts as a sync point against ctx.queue, same as every other blocking read in this file. */
    public static float[] downloadFloats(GpuContext ctx, MemorySegment mem, int count) throws Throwable {
        float[] out = new float[count];
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) count * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment host = tmp.allocate(Math.max(byteSize, 1));
            readBlocking(ctx, mem, host, byteSize);
            MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, out, 0, count);
        }
        return out;
    }

    /** Blocks until every command already enqueued on ctx.queue has completed. Not needed anywhere in forward_layer/prefill_layer (see class javadoc), but useful to callers timing a single kernel dispatch in isolation -- e.g. ActivationBenchmark brackets a kernel call with this before stopping its clock, otherwise it would only be timing host-side enqueue overhead, not actual GPU execution. */
    public static void finish(GpuContext ctx) throws Throwable {
        GpuContext.check((int) ctx.cl.clFinish.invoke(ctx.queue), "clFinish");
    }

    public static void freeQuietly(GpuContext ctx, MemorySegment mem) {
        try {
            if (mem != null) {
                ctx.cl.clReleaseMemObject.invoke(mem);
            }
        } catch (Throwable t) {
            // best-effort cleanup
        }
    }

    private static void writeBlocking(GpuContext ctx, MemorySegment mem, MemorySegment hostSrc, long byteSize) throws Throwable {
        GpuContext.check((int) ctx.cl.clEnqueueWriteBuffer.invoke(
                ctx.queue, mem, OpenCLBindings.CL_TRUE, 0L, byteSize, hostSrc, 0, MemorySegment.NULL, MemorySegment.NULL),
                "clEnqueueWriteBuffer");
    }

    private static void readBlocking(GpuContext ctx, MemorySegment mem, MemorySegment hostDst, long byteSize) throws Throwable {
        GpuContext.check((int) ctx.cl.clEnqueueReadBuffer.invoke(
                ctx.queue, mem, OpenCLBindings.CL_TRUE, 0L, byteSize, hostDst, 0, MemorySegment.NULL, MemorySegment.NULL),
                "clEnqueueReadBuffer");
    }

    /** Device-to-device copy with independent byte offsets on each side -- see class javadoc's writeIntoCache note for why this replaces the CUDA original's host round trip. */
    private static void copyDeviceToDevice(GpuContext ctx, MemorySegment src, MemorySegment dst,
            long srcOffsetBytes, long dstOffsetBytes, long byteSize) throws Throwable {
        GpuContext.check((int) ctx.cl.clEnqueueCopyBuffer.invoke(
                ctx.queue, src, dst, srcOffsetBytes, dstOffsetBytes, byteSize, 0, MemorySegment.NULL, MemorySegment.NULL),
                "clEnqueueCopyBuffer");
    }

    // =====================================================================
    // THE DECODE-PATH DECODER LAYER (single token, M=1)
    // =====================================================================

    /** Runs one decoder layer for the CURRENT token, in place on x. x: [dim] floats device buffer, in/out. */
    public static void forward_layer(MemorySegment x, GpuWeights w, GpuState s, Config cfg, GpuContext ctx) throws Throwable {
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
            GpuContext.check((int) ctx.cl.clFinish.invoke(ctx.queue), "clFinish(end of forward_layer)");
        }

        s.pos++;
    }

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

    /** Runs the full model for one token: embedding -> N decoder layers -> final norm -> LM head, greedy argmax. For non-greedy sampling use generate_logits + Sampler instead. */
    public static int generate_token(
            MemorySegment tokenEmbeddingDevice,
            GpuWeights[] layers,
            MemorySegment finalNormGammaDevice,
            MemorySegment lmHeadDevice,
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

    /** Same forward pass as generate_token, but returns raw logits for Sampler-based decoding instead of collapsing to argmax. */
    public static float[] generate_logits(
            MemorySegment tokenEmbeddingDevice,
            GpuWeights[] layers,
            MemorySegment finalNormGammaDevice,
            MemorySegment lmHeadDevice,
            GpuState[] states,
            Config cfg,
            GpuContext ctx,
            int vocabSize) throws Throwable {

        MemorySegment x = tokenEmbeddingDevice;

        for (int l = 0; l < layers.length; l++) {
            forward_layer(x, layers[l], states[l], cfg, ctx);
        }

        float[] logits = finalLogits(x, finalNormGammaDevice, lmHeadDevice, cfg, ctx, vocabSize);

        for (GpuState st : states) {
            st.pos++;
        }
        return logits;
    }

    /**
     * Final norm + LM head + host readback on an ARBITRARY already-computed
     * [dim]-sized row buffer -- does NOT run the decoder layers. xRow must
     * be a buffer of exactly cfg.dim floats starting at element 0 (unlike
     * the CUDA original, this port cannot accept an offset INTO a larger
     * buffer here -- see class javadoc; callers needing one row out of a
     * multi-row batch buffer must extract it first, e.g. via
     * clEnqueueCopyBuffer, which is what LlamaOpenCLEngine does).
     */
    public static float[] finalLogits(MemorySegment xRow, MemorySegment finalNormGammaDevice, MemorySegment lmHeadDevice,
            Config cfg, GpuContext ctx, int vocabSize) throws Throwable {
        MemorySegment normed = allocFloats(ctx, cfg.dim);
        MemorySegment logitsDevice = allocFloats(ctx, vocabSize);
        int maxGroups = (cfg.dim + KernelSource.RMSNORM_WORKGROUP_SIZE - 1) / KernelSource.RMSNORM_WORKGROUP_SIZE;
        MemorySegment partials = allocFloats(ctx, maxGroups);

        try {
            synchronized (ctx.dispatchLock) {
                rmsNorm(ctx, xRow, finalNormGammaDevice, normed, cfg.dim, cfg.norm_eps, partials);
                f32Gemv(ctx, normed, lmHeadDevice, logitsDevice, cfg.dim, vocabSize);
            }

            float[] logits = new float[vocabSize];
            try (Arena tmp = Arena.ofConfined()) {
                long byteSize = (long) vocabSize * ValueLayout.JAVA_FLOAT.byteSize();
                MemorySegment logitsHost = tmp.allocate(byteSize);
                readBlocking(ctx, logitsDevice, logitsHost, byteSize);
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
    // THE BATCHED PREFILL PATH: T tokens at once, one layer.
    // =====================================================================

    /**
     * Runs one decoder layer over a BATCH of T prompt positions at once.
     * Same single-call-only, startPos=0-only contract as the CUDA
     * original -- see its javadoc for the full rationale (this port
     * changes none of that algorithmic behavior, only the GPU-API calls
     * underneath it).
     */
    public static void prefill_layer(MemorySegment xBatch, GpuWeights w, GpuState s, Config cfg, GpuContext ctx,
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
            uploadIntsInto(ctx, s.positionsBatch, positions);

            // === 1. Attention block ===
            rmsNormRows(ctx, xBatch, w.attn_norm_gamma, s.xNormBatch, dim, cfg.norm_eps, T, s.rmsPartialsBatch, s.rmsRowValuesBatch);
            quantizeActivationQ8_0(ctx, s.xNormBatch, s.xNormQ8Batch, T * dim);

            q8_0GemmTiled(ctx, s.xNormQ8Batch, w.wq_q8_0, s.qBatch, T, qRowStride, dim);
            q8_0GemmTiled(ctx, s.xNormQ8Batch, w.wk_q8_0, s.kBatch, T, kvDim, dim);
            q8_0GemmTiled(ctx, s.xNormQ8Batch, w.wv_q8_0, s.vBatch, T, kvDim, dim);

            ropeApplyPairwiseRows(ctx, s.qBatch, s.cos_table, s.sin_table, numHeads, headDim, s.positionsBatch, T);
            ropeApplyPairwiseRows(ctx, s.kBatch, s.cos_table, s.sin_table, kvHeads, headDim, s.positionsBatch, T);

            // Bulk cache write -- device-to-device, T*kvDim elements. See
            // class javadoc: this is a genuine improvement over the CUDA
            // original's host round trip, not just a translation.
            writeIntoCache(ctx, s.kBatch, s.k_cache_f32, (long) startPos * kvDim, T * kvDim);
            writeIntoCache(ctx, s.vBatch, s.v_cache_f32, (long) startPos * kvDim, T * kvDim);

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

            GpuContext.check((int) ctx.cl.clFinish.invoke(ctx.queue), "clFinish(end of prefill_layer)");
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
    // No clFinish inside any of these -- see class javadoc for why that's
    // correct on an in-order queue.
    // =====================================================================

    // ---- decode-path dispatch ----

    static void rmsNorm(GpuContext ctx, MemorySegment x, MemorySegment gamma, MemorySegment out,
            int features, double eps, MemorySegment partials) throws Throwable {
        int wgSize = KernelSource.RMSNORM_WORKGROUP_SIZE;
        int numGroups = (features + wgSize - 1) / wgSize;

        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kRmsnormPartialSumsq, 0, x);
            setArg(tmp, ctx, ctx.kRmsnormPartialSumsq, 1, partials);
            setArgI(tmp, ctx, ctx.kRmsnormPartialSumsq, 2, features);
            launch1D(ctx, ctx.kRmsnormPartialSumsq, (long) numGroups * wgSize, wgSize);
        }

        float[] partialHost = new float[numGroups];
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) numGroups * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment host = tmp.allocate(byteSize);
            readBlocking(ctx, partials, host, byteSize); // blocking -> implicit sync point
            MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, partialHost, 0, numGroups);
        }
        double sumSq = 0.0;
        for (float p : partialHost) {
            sumSq += p;
        }
        float rms = (float) (1.0 / Math.sqrt(sumSq / features + eps));

        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kRmsnormApply, 0, x);
            setArg(tmp, ctx, ctx.kRmsnormApply, 1, gamma);
            setArg(tmp, ctx, ctx.kRmsnormApply, 2, out);
            setArgF(tmp, ctx, ctx.kRmsnormApply, 3, rms);
            setArgI(tmp, ctx, ctx.kRmsnormApply, 4, features);
            launch1D(ctx, ctx.kRmsnormApply, features, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    static void quantizeActivationQ8_0(GpuContext ctx, MemorySegment x, MemorySegment outQ8, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kQuantizeActivationQ8_0, 0, x);
            setArg(tmp, ctx, ctx.kQuantizeActivationQ8_0, 1, outQ8);
            setArgI(tmp, ctx, ctx.kQuantizeActivationQ8_0, 2, len);
            launch1D(ctx, ctx.kQuantizeActivationQ8_0, len, KernelSource.QUANTIZE_BLOCK_SIZE);
        }
    }

    static void q8_0GemvSplit(GpuContext ctx, MemorySegment xQ8, MemorySegment wQ8, MemorySegment out,
            int heads, int headDim, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kQ8_0GemvSplit, 0, xQ8);
            setArg(tmp, ctx, ctx.kQ8_0GemvSplit, 1, wQ8);
            setArg(tmp, ctx, ctx.kQ8_0GemvSplit, 2, out);
            setArgI(tmp, ctx, ctx.kQ8_0GemvSplit, 3, heads);
            setArgI(tmp, ctx, ctx.kQ8_0GemvSplit, 4, headDim);
            setArgI(tmp, ctx, ctx.kQ8_0GemvSplit, 5, K);
            launch1D(ctx, ctx.kQ8_0GemvSplit, (long) heads * (headDim / 2), KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    static void q8_0GemvPlain(GpuContext ctx, MemorySegment xQ8, MemorySegment wQ8, MemorySegment out,
            int N, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kQ8_0GemvPlain, 0, xQ8);
            setArg(tmp, ctx, ctx.kQ8_0GemvPlain, 1, wQ8);
            setArg(tmp, ctx, ctx.kQ8_0GemvPlain, 2, out);
            setArgI(tmp, ctx, ctx.kQ8_0GemvPlain, 3, N);
            setArgI(tmp, ctx, ctx.kQ8_0GemvPlain, 4, K);
            launch1D(ctx, ctx.kQ8_0GemvPlain, N, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    static void ropeApplySplit(GpuContext ctx, MemorySegment buf, MemorySegment cosTable, MemorySegment sinTable,
            int heads, int headDim, int cosSinOffset) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kRopeApplySplit, 0, buf);
            setArg(tmp, ctx, ctx.kRopeApplySplit, 1, cosTable);
            setArg(tmp, ctx, ctx.kRopeApplySplit, 2, sinTable);
            setArgI(tmp, ctx, ctx.kRopeApplySplit, 3, heads);
            setArgI(tmp, ctx, ctx.kRopeApplySplit, 4, headDim);
            setArgI(tmp, ctx, ctx.kRopeApplySplit, 5, cosSinOffset);
            launch1D(ctx, ctx.kRopeApplySplit, (long) heads * (headDim / 2), KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    static void attnScores(GpuContext ctx, MemorySegment qAllHeads, MemorySegment kCache, MemorySegment scores,
            int qHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive, float rsqrtD) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kAttnScores, 0, qAllHeads);
            setArg(tmp, ctx, ctx.kAttnScores, 1, kCache);
            setArg(tmp, ctx, ctx.kAttnScores, 2, scores);
            setArgI(tmp, ctx, ctx.kAttnScores, 3, qHeadOff);
            setArgI(tmp, ctx, ctx.kAttnScores, 4, headDim);
            setArgI(tmp, ctx, ctx.kAttnScores, 5, kvDim);
            setArgI(tmp, ctx, ctx.kAttnScores, 6, kvHeadOff);
            setArgI(tmp, ctx, ctx.kAttnScores, 7, posInclusive);
            setArgF(tmp, ctx, ctx.kAttnScores, 8, rsqrtD);
            launch1D(ctx, ctx.kAttnScores, posInclusive, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    static void softmaxInplace(GpuContext ctx, MemorySegment scores, int len) throws Throwable {
        int localSize = nextPow2(Math.max(len, 1));
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kSoftmaxInplace, 0, scores);
            setArgI(tmp, ctx, ctx.kSoftmaxInplace, 1, len);
            launch1D(ctx, ctx.kSoftmaxInplace, localSize, localSize);
        }
    }

    private static int nextPow2(int v) {
        int p = 1;
        while (p < v) {
            p <<= 1;
        }
        return Math.min(p, 256);
    }

    static void attnWeightedSum(GpuContext ctx, MemorySegment scores, MemorySegment vCache, MemorySegment attnOutAllHeads,
            int outHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kAttnWeightedSum, 0, scores);
            setArg(tmp, ctx, ctx.kAttnWeightedSum, 1, vCache);
            setArg(tmp, ctx, ctx.kAttnWeightedSum, 2, attnOutAllHeads);
            setArgI(tmp, ctx, ctx.kAttnWeightedSum, 3, outHeadOff);
            setArgI(tmp, ctx, ctx.kAttnWeightedSum, 4, headDim);
            setArgI(tmp, ctx, ctx.kAttnWeightedSum, 5, kvDim);
            setArgI(tmp, ctx, ctx.kAttnWeightedSum, 6, kvHeadOff);
            setArgI(tmp, ctx, ctx.kAttnWeightedSum, 7, posInclusive);
            launch1D(ctx, ctx.kAttnWeightedSum, headDim, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    /** Package-private (not private) so ActivationBenchmark, in this same package, can dispatch it directly for correctness/perf testing without this becoming public API. */
    static void swigluActivate(GpuContext ctx, MemorySegment gate, MemorySegment up, MemorySegment out, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kSwigluActivate, 0, gate);
            setArg(tmp, ctx, ctx.kSwigluActivate, 1, up);
            setArg(tmp, ctx, ctx.kSwigluActivate, 2, out);
            setArgI(tmp, ctx, ctx.kSwigluActivate, 3, len);
            launch1D(ctx, ctx.kSwigluActivate, len, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    /** Package-private -- see swigluActivate's note above. */
    static void gegluActivate(GpuContext ctx, MemorySegment gate, MemorySegment up, MemorySegment out, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kGegluActivate, 0, gate);
            setArg(tmp, ctx, ctx.kGegluActivate, 1, up);
            setArg(tmp, ctx, ctx.kGegluActivate, 2, out);
            setArgI(tmp, ctx, ctx.kGegluActivate, 3, len);
            launch1D(ctx, ctx.kGegluActivate, len, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    /** Package-private -- see swigluActivate's note above. */
    static void geluActivate(GpuContext ctx, MemorySegment gate, MemorySegment out, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kGeluActivate, 0, gate);
            setArg(tmp, ctx, ctx.kGeluActivate, 1, out);
            setArgI(tmp, ctx, ctx.kGeluActivate, 2, len);
            launch1D(ctx, ctx.kGeluActivate, len, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    private static void residualAdd(GpuContext ctx, MemorySegment x, MemorySegment y, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kResidualAdd, 0, x);
            setArg(tmp, ctx, ctx.kResidualAdd, 1, y);
            setArgI(tmp, ctx, ctx.kResidualAdd, 2, len);
            launch1D(ctx, ctx.kResidualAdd, len, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    static void f32Gemv(GpuContext ctx, MemorySegment a, MemorySegment B, MemorySegment out, int K, int N) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kF32Gemv, 0, a);
            setArg(tmp, ctx, ctx.kF32Gemv, 1, B);
            setArg(tmp, ctx, ctx.kF32Gemv, 2, out);
            setArgI(tmp, ctx, ctx.kF32Gemv, 3, K);
            setArgI(tmp, ctx, ctx.kF32Gemv, 4, N);
            launch1D(ctx, ctx.kF32Gemv, N, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    /** Device-to-device KV-cache write at a growing per-token/per-batch element offset -- see class javadoc. */
    private static void writeIntoCache(GpuContext ctx, MemorySegment src, MemorySegment cache, long elementOffset, int len) throws Throwable {
        long dstOffsetBytes = elementOffset * ValueLayout.JAVA_FLOAT.byteSize();
        long byteSize = (long) len * ValueLayout.JAVA_FLOAT.byteSize();
        copyDeviceToDevice(ctx, src, cache, 0L, dstOffsetBytes, byteSize);
    }

    // ---- batched prefill dispatch ----

    static void rmsNormRows(GpuContext ctx, MemorySegment x, MemorySegment gamma, MemorySegment out,
            int features, double eps, int T, MemorySegment partialsBatch, MemorySegment rmsRowValuesBatch) throws Throwable {
        int wgSize = KernelSource.RMSNORM_WORKGROUP_SIZE;
        int numGroups = (features + wgSize - 1) / wgSize;

        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kRmsnormPartialSumsqRows, 0, x);
            setArg(tmp, ctx, ctx.kRmsnormPartialSumsqRows, 1, partialsBatch);
            setArgI(tmp, ctx, ctx.kRmsnormPartialSumsqRows, 2, features);
            setArgI(tmp, ctx, ctx.kRmsnormPartialSumsqRows, 3, numGroups);
            launch2D(ctx, ctx.kRmsnormPartialSumsqRows, T, (long) numGroups * wgSize, wgSize);
        }

        float[] partialHost = new float[T * numGroups];
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) T * numGroups * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment host = tmp.allocate(byteSize);
            readBlocking(ctx, partialsBatch, host, byteSize); // blocking -> implicit sync point
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
            setArg(tmp, ctx, ctx.kRmsnormApplyRows, 0, x);
            setArg(tmp, ctx, ctx.kRmsnormApplyRows, 1, gamma);
            setArg(tmp, ctx, ctx.kRmsnormApplyRows, 2, out);
            setArg(tmp, ctx, ctx.kRmsnormApplyRows, 3, rmsRowValuesBatch);
            setArgI(tmp, ctx, ctx.kRmsnormApplyRows, 4, features);
            launch2D(ctx, ctx.kRmsnormApplyRows, T, features, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    static void q8_0GemmTiled(GpuContext ctx, MemorySegment xQ8, MemorySegment wQ8, MemorySegment out, int T, int N, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kQ8_0GemmTiled, 0, xQ8);
            setArg(tmp, ctx, ctx.kQ8_0GemmTiled, 1, wQ8);
            setArg(tmp, ctx, ctx.kQ8_0GemmTiled, 2, out);
            setArgI(tmp, ctx, ctx.kQ8_0GemmTiled, 3, T);
            setArgI(tmp, ctx, ctx.kQ8_0GemmTiled, 4, N);
            setArgI(tmp, ctx, ctx.kQ8_0GemmTiled, 5, K);
            launch2D(ctx, ctx.kQ8_0GemmTiled, T, N, KernelSource.GEMM_TILE_N);
        }
    }

    static void f32GemmTiled(GpuContext ctx, MemorySegment a, MemorySegment B, MemorySegment out, int T, int K, int N) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kF32GemmTiled, 0, a);
            setArg(tmp, ctx, ctx.kF32GemmTiled, 1, B);
            setArg(tmp, ctx, ctx.kF32GemmTiled, 2, out);
            setArgI(tmp, ctx, ctx.kF32GemmTiled, 3, T);
            setArgI(tmp, ctx, ctx.kF32GemmTiled, 4, K);
            setArgI(tmp, ctx, ctx.kF32GemmTiled, 5, N);
            launch2D(ctx, ctx.kF32GemmTiled, T, N, KernelSource.GEMM_TILE_N);
        }
    }

    static void ropeApplyPairwiseRows(GpuContext ctx, MemorySegment buf, MemorySegment cosTable, MemorySegment sinTable,
            int heads, int headDim, MemorySegment positionsBatch, int T) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kRopeApplyPairwiseRows, 0, buf);
            setArg(tmp, ctx, ctx.kRopeApplyPairwiseRows, 1, cosTable);
            setArg(tmp, ctx, ctx.kRopeApplyPairwiseRows, 2, sinTable);
            setArgI(tmp, ctx, ctx.kRopeApplyPairwiseRows, 3, heads);
            setArgI(tmp, ctx, ctx.kRopeApplyPairwiseRows, 4, headDim);
            setArg(tmp, ctx, ctx.kRopeApplyPairwiseRows, 5, positionsBatch);
            launch2D(ctx, ctx.kRopeApplyPairwiseRows, T, (long) heads * (headDim / 2), KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    static void attnScoresCausalBatched(GpuContext ctx, MemorySegment qAll, MemorySegment kAll, MemorySegment scoresBatch,
            int qRowStride, int kRowStride, int qHeadOff, int kHeadOff, int headDim, int T, float rsqrtD) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kAttnScoresCausalBatched, 0, qAll);
            setArg(tmp, ctx, ctx.kAttnScoresCausalBatched, 1, kAll);
            setArg(tmp, ctx, ctx.kAttnScoresCausalBatched, 2, scoresBatch);
            setArgI(tmp, ctx, ctx.kAttnScoresCausalBatched, 3, qRowStride);
            setArgI(tmp, ctx, ctx.kAttnScoresCausalBatched, 4, kRowStride);
            setArgI(tmp, ctx, ctx.kAttnScoresCausalBatched, 5, qHeadOff);
            setArgI(tmp, ctx, ctx.kAttnScoresCausalBatched, 6, kHeadOff);
            setArgI(tmp, ctx, ctx.kAttnScoresCausalBatched, 7, headDim);
            setArgI(tmp, ctx, ctx.kAttnScoresCausalBatched, 8, T);
            setArgF(tmp, ctx, ctx.kAttnScoresCausalBatched, 9, rsqrtD);
            launch2D(ctx, ctx.kAttnScoresCausalBatched, T, T, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    static void softmaxInplaceRows(GpuContext ctx, MemorySegment scoresBatch, int T) throws Throwable {
        int localSize = nextPow2(Math.max(T, 1));
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kSoftmaxInplaceRows, 0, scoresBatch);
            setArgI(tmp, ctx, ctx.kSoftmaxInplaceRows, 1, T);
            launch1D(ctx, ctx.kSoftmaxInplaceRows, (long) T * localSize, localSize);
        }
    }

    static void attnWeightedSumCausalBatched(GpuContext ctx, MemorySegment scoresBatch, MemorySegment vAll, MemorySegment attnOutBatch,
            int vRowStride, int outRowStride, int vHeadOff, int outHeadOff, int headDim, int T) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArg(tmp, ctx, ctx.kAttnWeightedSumCausalBatched, 0, scoresBatch);
            setArg(tmp, ctx, ctx.kAttnWeightedSumCausalBatched, 1, vAll);
            setArg(tmp, ctx, ctx.kAttnWeightedSumCausalBatched, 2, attnOutBatch);
            setArgI(tmp, ctx, ctx.kAttnWeightedSumCausalBatched, 3, vRowStride);
            setArgI(tmp, ctx, ctx.kAttnWeightedSumCausalBatched, 4, outRowStride);
            setArgI(tmp, ctx, ctx.kAttnWeightedSumCausalBatched, 5, vHeadOff);
            setArgI(tmp, ctx, ctx.kAttnWeightedSumCausalBatched, 6, outHeadOff);
            setArgI(tmp, ctx, ctx.kAttnWeightedSumCausalBatched, 7, headDim);
            setArgI(tmp, ctx, ctx.kAttnWeightedSumCausalBatched, 8, T);
            launch2D(ctx, ctx.kAttnWeightedSumCausalBatched, T, headDim, KernelSource.DEFAULT_BLOCK_SIZE);
        }
    }

    // ================= FFM plumbing: clSetKernelArg + clEnqueueNDRangeKernel =================

    private static void setArg(Arena a, GpuContext ctx, MemorySegment kernel, int index, MemorySegment mem) throws Throwable {
        MemorySegment slot = a.allocate(ValueLayout.ADDRESS);
        slot.set(ValueLayout.ADDRESS, 0, mem);
        GpuContext.check((int) ctx.cl.clSetKernelArg.invoke(kernel, index, ValueLayout.ADDRESS.byteSize(), slot),
                "clSetKernelArg(mem, " + index + ")");
    }

    private static void setArgI(Arena a, GpuContext ctx, MemorySegment kernel, int index, int v) throws Throwable {
        MemorySegment slot = a.allocate(ValueLayout.JAVA_INT);
        slot.set(ValueLayout.JAVA_INT, 0, v);
        GpuContext.check((int) ctx.cl.clSetKernelArg.invoke(kernel, index, ValueLayout.JAVA_INT.byteSize(), slot),
                "clSetKernelArg(int, " + index + ")");
    }

    private static void setArgF(Arena a, GpuContext ctx, MemorySegment kernel, int index, float v) throws Throwable {
        MemorySegment slot = a.allocate(ValueLayout.JAVA_FLOAT);
        slot.set(ValueLayout.JAVA_FLOAT, 0, v);
        GpuContext.check((int) ctx.cl.clSetKernelArg.invoke(kernel, index, ValueLayout.JAVA_FLOAT.byteSize(), slot),
                "clSetKernelArg(float, " + index + ")");
    }

    /** 1D launch: rounds workItems up to a whole number of work-groups. No clFinish -- see class javadoc. */
    private static void launch1D(GpuContext ctx, MemorySegment kernel, long workItems, int localSize) throws Throwable {
        long global = ((workItems + localSize - 1) / localSize) * localSize;
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment globalSize = tmp.allocate(ValueLayout.JAVA_LONG);
            globalSize.set(ValueLayout.JAVA_LONG, 0, global);
            MemorySegment localSizeSeg = tmp.allocate(ValueLayout.JAVA_LONG);
            localSizeSeg.set(ValueLayout.JAVA_LONG, 0, localSize);
            GpuContext.check((int) ctx.cl.clEnqueueNDRangeKernel.invoke(
                    ctx.queue, kernel, 1, MemorySegment.NULL, globalSize, localSizeSeg, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueNDRangeKernel(1D)");
        }
    }

    /** 2D launch: dim1 (rows) exact, dim0 (cols) rounded up to a whole number of work-groups of size localSizeX. No clFinish -- see class javadoc. */
    private static void launch2D(GpuContext ctx, MemorySegment kernel, int rows, long cols, int localSizeX) throws Throwable {
        long globalX = ((cols + localSizeX - 1) / localSizeX) * localSizeX;
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment globalSize = tmp.allocate(ValueLayout.JAVA_LONG, 2);
            globalSize.setAtIndex(ValueLayout.JAVA_LONG, 0, globalX);
            globalSize.setAtIndex(ValueLayout.JAVA_LONG, 1, rows);
            MemorySegment localSizeSeg = tmp.allocate(ValueLayout.JAVA_LONG, 2);
            localSizeSeg.setAtIndex(ValueLayout.JAVA_LONG, 0, localSizeX);
            localSizeSeg.setAtIndex(ValueLayout.JAVA_LONG, 1, 1);
            GpuContext.check((int) ctx.cl.clEnqueueNDRangeKernel.invoke(
                    ctx.queue, kernel, 2, MemorySegment.NULL, globalSize, localSizeSeg, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueNDRangeKernel(2D)");
        }
    }
}