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
package com.github.gbenroscience.gpu.opencl.llm;

import com.github.gbenroscience.gpu.opencl.OpenClBindings;
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.NoSuchElementException;

/**
 *
 * @author GBEMIRO GPU-resident Llama-style decoder layer, mirroring
 * LlamaLayerInt8.forward_layer_q8's exact six-stage structure -- same stages,
 * same order, each dispatched as GPU kernels (see KernelSource) instead of SIMD
 * CPU calls:
 *
 * 1a. RMSNorm(x) -> rmsNorm 1b. Quantize activation to Q8_0 blocks ->
 * quantizeActivationQ8_0 1c. Q/K/V projections (Q8_0 GEMV) -> q8_0GemvSplit 1d.
 * RoPE on Q and K -> ropeApplySplit 1e. Write K/V into the cache ->
 * writeIntoCache 1f. Multi-head attention (THE MHA LOOP) -> attnScores +
 * softmaxInplace + attnWeightedSum, per head 1g. O-projection (plain FP32 GEMV)
 * -> f32Gemv 1h. Residual add -> residualAdd 2a. RMSNorm(x) -> rmsNorm 2b.
 * Gate/Up projections (Q8_0 GEMV) -> q8_0GemvPlain 2c. SwiGLU activation ->
 * swigluActivate 2d. Down projection (Q8_0 GEMV) -> q8_0GemvPlain 2e. Residual
 * add -> residualAdd
 *
 * GQA (grouped-query attention): query heads are mapped to KV heads in groups
 * of (num_heads / kv_heads) -- the standard llama.cpp convention, consistent
 * with Config.kv_heads's stated purpose ("set &lt; num_heads for Llama2-70B" in
 * the CPU reference). num_heads must be an exact multiple of kv_heads.
 *
 * K/V CACHE PRECISION: stored FP32 on-device here, not GGUF Q8_0 blocks like
 * the CPU reference's cache. This is a deliberate departure -- the CPU version
 * re-dequantizes cached K/V from Q8_0 on every single attention read (every
 * head, every step), which is fine for a scalar loop but would mean re-decoding
 * the same FP16 block scales redundantly thousands of times per token on the
 * GPU. A device-resident FP32 cache dequantizes each K/V vector exactly ONCE,
 * at write time (see writeIntoCache), trading roughly 4x more cache memory for
 * that redundant work. For max_seq=4096, kv_heads=32, head_dim=128 that's
 * 4096*4096*4 bytes = 64MB per cache (K and V each) -- easily affordable on any
 * GPU capable of holding the model weights themselves.
 *
 * UNVERIFIED: no OpenCL driver, no GPU, and critically no reference model to
 * run token-for-token comparison against were available while writing this.
 * Every kernel is a traced port of concrete scalar reference logic (see
 * KernelSource's per-kernel javadoc for exactly which CPU method each one
 * ports), but "traced carefully" is not the same bar as "verified against real
 * output" -- run this against a small known-good model and compare per-layer
 * activations against the CPU path before trusting it for real inference.
 */
public final class LlamaLayer {

    private LlamaLayer() {
    }

    /**
     * Mirrors LlamaLayerInt8.Config's fields exactly.
     */
    public static final class Config {

        public int dim = 4096;
        public int hidden_dim = 11008;
        public int num_heads = 32;
        public int kv_heads = 32;
        public int head_dim = 128;
        public int max_seq = 4096;
        public double norm_eps = 1e-6;
        public double rope_theta = 10000.0;

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
        }
    }

    /**
     * GPU-resident weights for ONE decoder layer. Q8_0 tensors are uploaded to
     * the device AS-IS (raw GGUF block bytes) -- dequantized on-device inside
     * the GEMV kernels, never on the host. wo stays FP32, matching
     * LlamaLayerInt8's own choice not to quantize it ("it's small").
     */
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

        /**
         * Convenience factory reading straight from a loaded GGUF file -- same
         * tensor-name lookups ModelLoader.loadLlamaWeights already uses, but
         * uploads Q8_0 tensors via loadQ8_0() (raw block bytes) rather than
         * loadQ8_0AsFloat() (CPU-side dequantized floats): dequantization
         * happens once, on-device, inside the GEMV kernels instead of once on
         * the host at load time.
         *
         * @param ctx
         * @param gguf
         * @param layerPrefix
         * @return
         * @throws Throwable
         */
        public static GpuWeights fromGguf(GpuContext ctx, GGUFLoader.GGUFFile gguf, String layerPrefix) throws Throwable {
            return new GpuWeights(ctx,
                    find(gguf, layerPrefix + "attn_q.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "attn_k.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "attn_v.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "attn_output.weight").loadFloat(),
                    find(gguf, layerPrefix + "ffn_gate.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "ffn_up.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "ffn_down.weight").loadQ8_0(),
                    find(gguf, layerPrefix + "attn_norm.weight").loadFloat(),
                    find(gguf, layerPrefix + "ffn_norm.weight").loadFloat());
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
            try {
                ctx.cl.clReleaseMemObject.invoke(wq_q8_0);
                ctx.cl.clReleaseMemObject.invoke(wk_q8_0);
                ctx.cl.clReleaseMemObject.invoke(wv_q8_0);
                ctx.cl.clReleaseMemObject.invoke(wo_f32);
                ctx.cl.clReleaseMemObject.invoke(w_gate_q8_0);
                ctx.cl.clReleaseMemObject.invoke(w_up_q8_0);
                ctx.cl.clReleaseMemObject.invoke(w_down_q8_0);
                ctx.cl.clReleaseMemObject.invoke(attn_norm_gamma);
                ctx.cl.clReleaseMemObject.invoke(ffn_norm_gamma);
            } catch (Throwable t) {
                // best-effort cleanup
            }
        }
    }

    /**
     * Per-sequence GPU state: KV cache (FP32, see class javadoc for why) and
     * RoPE tables, both device-resident for the whole generation loop, plus
     * persistent scratch buffers reused every forward_layer call -- zero
     * per-token device allocation once a GpuState is constructed.
     */
    public static final class GpuState implements AutoCloseable {

        final GpuContext ctx;
        final int kvDim;

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

        public int pos = 0;

        public GpuState(GpuContext ctx, Config cfg) throws Throwable {
            cfg.validate();
            this.ctx = ctx;
            this.kvDim = cfg.kv_heads * cfg.head_dim;

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
            int maxPartials = (maxLen + KernelSource.RMSNORM_WORKGROUP_SIZE - 1) / KernelSource.RMSNORM_WORKGROUP_SIZE;
            this.rmsPartials = allocFloats(ctx, maxPartials);
        }

        private static int q8_0Bytes(int len) {
            return (len / 32) * 34;
        }

        /**
         * Same formula as KernelsFloat's precompute_rope_f32: theta_i =
         * base^(-2i/head_dim).
         */
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

        private static MemorySegment allocFloats(GpuContext ctx, long count) throws Throwable {
            return allocBytes(ctx, count * ValueLayout.JAVA_FLOAT.byteSize());
        }

        private static MemorySegment allocBytes(GpuContext ctx, long byteCount) throws Throwable {
            try (Arena tmp = Arena.ofConfined()) {
                MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
                MemorySegment dev = (MemorySegment) ctx.cl.clCreateBuffer.invoke(ctx.context,
                        OpenClBindings.CL_MEM_READ_WRITE, byteCount, MemorySegment.NULL, errBuf);
                checkStatus(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(GpuState)");
                return dev;
            }
        }

        private static MemorySegment uploadFloats(GpuContext ctx, float[] data) throws Throwable {
            try (Arena tmp = Arena.ofConfined()) {
                MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_FLOAT.byteSize());
                MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
                MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
                MemorySegment dev = (MemorySegment) ctx.cl.clCreateBuffer.invoke(ctx.context,
                        OpenClBindings.CL_MEM_READ_ONLY, host.byteSize(), MemorySegment.NULL, errBuf);
                checkStatus(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(rope tables)");
                checkStatus((int) ctx.cl.clEnqueueWriteBuffer.invoke(ctx.queue, dev, OpenClBindings.CL_TRUE,
                        0L, host.byteSize(), host, 0, MemorySegment.NULL, MemorySegment.NULL),
                        "clEnqueueWriteBuffer(rope tables)");
                return dev;
            }
        }

        @Override
        public void close() {
            try {
                ctx.cl.clReleaseMemObject.invoke(k_cache_f32);
                ctx.cl.clReleaseMemObject.invoke(v_cache_f32);
                ctx.cl.clReleaseMemObject.invoke(cos_table);
                ctx.cl.clReleaseMemObject.invoke(sin_table);
                ctx.cl.clReleaseMemObject.invoke(xNorm);
                ctx.cl.clReleaseMemObject.invoke(xNormQ8);
                ctx.cl.clReleaseMemObject.invoke(qSplit);
                ctx.cl.clReleaseMemObject.invoke(kNew);
                ctx.cl.clReleaseMemObject.invoke(vNew);
                ctx.cl.clReleaseMemObject.invoke(scores);
                ctx.cl.clReleaseMemObject.invoke(attnOut);
                ctx.cl.clReleaseMemObject.invoke(attnProj);
                ctx.cl.clReleaseMemObject.invoke(ffnNorm);
                ctx.cl.clReleaseMemObject.invoke(ffnNormQ8);
                ctx.cl.clReleaseMemObject.invoke(gate);
                ctx.cl.clReleaseMemObject.invoke(up);
                ctx.cl.clReleaseMemObject.invoke(swigluOut);
                ctx.cl.clReleaseMemObject.invoke(swigluOutQ8);
                ctx.cl.clReleaseMemObject.invoke(ffnDownOut);
                ctx.cl.clReleaseMemObject.invoke(rmsPartials);
            } catch (Throwable t) {
                // best-effort cleanup
            }
        }
    }

    private static MemorySegment uploadBytes(GpuContext ctx, byte[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate(data.length);
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_BYTE, 0, data.length);
            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
            MemorySegment dev = (MemorySegment) ctx.cl.clCreateBuffer.invoke(ctx.context,
                    OpenClBindings.CL_MEM_READ_ONLY, host.byteSize(), MemorySegment.NULL, errBuf);
            checkStatus(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(weight bytes)");
            checkStatus((int) ctx.cl.clEnqueueWriteBuffer.invoke(ctx.queue, dev, OpenClBindings.CL_TRUE,
                    0L, host.byteSize(), host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueWriteBuffer(weight bytes)");
            return dev;
        }
    }

    /**
     *
     * @param ctx
     * @param data
     * @return
     * @throws Throwable
     */
    private static MemorySegment uploadFloats(GpuContext ctx, float[] data) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) data.length * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
            MemorySegment dev = (MemorySegment) ctx.cl.clCreateBuffer.invoke(ctx.context,
                    OpenClBindings.CL_MEM_READ_ONLY, host.byteSize(), MemorySegment.NULL, errBuf);
            checkStatus(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(weight floats)");
            checkStatus((int) ctx.cl.clEnqueueWriteBuffer.invoke(ctx.queue, dev, OpenClBindings.CL_TRUE,
                    0L, host.byteSize(), host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueWriteBuffer(weight floats)");
            return dev;
        }
    }

    // =====================================================================
    // THE DECODER LAYER
    // =====================================================================
    /**
     * Runs one decoder layer for the CURRENT token, in place on x. x: device
     * buffer, [dim] floats, in/out -- same contract as LlamaLayerInt8's
     * `double[] x`.
     *
     * @param x
     * @param w
     * @param s
     * @param cfg
     * @param ctx
     * @throws Throwable
     */
    public static void forward_layer(MemorySegment x, GpuWeights w, GpuState s, Config cfg, GpuContext ctx) throws Throwable {
        final int dim = cfg.dim;
        final int hidden = cfg.hidden_dim;
        final int headDim = cfg.head_dim;
        final int numHeads = cfg.num_heads;
        final int kvHeads = cfg.kv_heads;
        final int kvDim = s.kvDim;
        final int halfDim = headDim / 2;
        final int groupSize = numHeads / kvHeads; // GQA grouping, see class javadoc
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

            // Cache write: small host round-trip (kvDim floats) rather than
            // adding a clCreateSubBuffer/clEnqueueCopyBuffer binding this
            // project doesn't otherwise need -- see class javadoc.
            writeIntoCache(ctx, s.kNew, s.k_cache_f32, (long) s.pos * kvDim, kvDim);
            writeIntoCache(ctx, s.vNew, s.v_cache_f32, (long) s.pos * kvDim, kvDim);

            // --- THE MULTI-HEAD ATTENTION LOOP ---
            // One query head at a time: score against every cached position
            // up to and including this one (GQA-mapped to its KV head
            // group), softmax those scores, then weighted-sum against the
            // cached V vectors. Three kernel dispatches per head.
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

            q8_0GemvPlain(ctx, s.ffnNormQ8, w.w_gate_q8_0, s.gate, hidden, dim);
            q8_0GemvPlain(ctx, s.ffnNormQ8, w.w_up_q8_0, s.up, hidden, dim);
            swigluActivate(ctx, s.gate, s.up, s.swigluOut, hidden);

            quantizeActivationQ8_0(ctx, s.swigluOut, s.swigluOutQ8, hidden);
            q8_0GemvPlain(ctx, s.swigluOutQ8, w.w_down_q8_0, s.ffnDownOut, dim, hidden);
            residualAdd(ctx, x, s.ffnDownOut, dim);
        }

        s.pos++;
    }

    /**
     * Runs the full model for one token: embedding -> N decoder layers -> final
     * norm -> LM head. Mirrors LlamaLayer.generate_token's structure; the LM
     * head projection uses f32Gemv (matching matmul_down's FP32 usage in the
     * CPU reference for that step) -- quantizing the LM head too is a
     * reasonable future optimization, not attempted here since vocab-sized
     * (~32000-wide) output is comparatively cheap next to the per-layer GEMVs
     * regardless of precision.
     *
     * @param tokenEmbeddingDevice
     * @param layers
     * @param finalNormGammaDevice
     * @param lmHeadDevice
     * @param states
     * @param cfg
     * @param ctx
     * @param vocabSize
     * @return
     * @throws Throwable
     */
    public static int generate_token(
            MemorySegment tokenEmbeddingDevice, // [dim], mutated in place by forward_layer
            GpuWeights[] layers,
            MemorySegment finalNormGammaDevice, // [dim]
            MemorySegment lmHeadDevice, // [dim, vocab_size], FP32
            GpuState[] states,
            Config cfg,
            GpuContext ctx,
            int vocabSize) throws Throwable {

        MemorySegment x = tokenEmbeddingDevice;

        for (int l = 0; l < layers.length; l++) {
            forward_layer(x, layers[l], states[l], cfg, ctx);
        }

        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment errBuf = tmp.allocate(ValueLayout.JAVA_INT);
            MemorySegment normed = (MemorySegment) ctx.cl.clCreateBuffer.invoke(ctx.context,
                    OpenClBindings.CL_MEM_READ_WRITE, (long) cfg.dim * ValueLayout.JAVA_FLOAT.byteSize(),
                    MemorySegment.NULL, errBuf);
            checkStatus(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(final norm)");

            MemorySegment logitsDevice = (MemorySegment) ctx.cl.clCreateBuffer.invoke(ctx.context,
                    OpenClBindings.CL_MEM_READ_WRITE, (long) vocabSize * ValueLayout.JAVA_FLOAT.byteSize(),
                    MemorySegment.NULL, errBuf);
            checkStatus(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(logits)");

            try {
                MemorySegment partials = (MemorySegment) ctx.cl.clCreateBuffer.invoke(ctx.context,
                        OpenClBindings.CL_MEM_READ_WRITE,
                        (long) ((cfg.dim + KernelSource.RMSNORM_WORKGROUP_SIZE - 1) / KernelSource.RMSNORM_WORKGROUP_SIZE)
                        * ValueLayout.JAVA_FLOAT.byteSize(),
                        MemorySegment.NULL, errBuf);
                checkStatus(errBuf.get(ValueLayout.JAVA_INT, 0), "clCreateBuffer(final norm partials)");
                try {
                    synchronized (ctx.dispatchLock) {
                        rmsNorm(ctx, x, finalNormGammaDevice, normed, cfg.dim, cfg.norm_eps, partials);
                        f32Gemv(ctx, normed, lmHeadDevice, logitsDevice, cfg.dim, vocabSize);
                    }

                    float[] logits = new float[vocabSize];
                    MemorySegment logitsHost = tmp.allocate((long) vocabSize * ValueLayout.JAVA_FLOAT.byteSize());
                    checkStatus((int) ctx.cl.clEnqueueReadBuffer.invoke(ctx.queue, logitsDevice, OpenClBindings.CL_TRUE,
                            0L, logitsHost.byteSize(), logitsHost, 0, MemorySegment.NULL, MemorySegment.NULL),
                            "clEnqueueReadBuffer(logits)");
                    MemorySegment.copy(logitsHost, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);

                    int maxIdx = 0;
                    float maxVal = logits[0];
                    for (int i = 1; i < vocabSize; i++) {
                        if (logits[i] > maxVal) {
                            maxVal = logits[i];
                            maxIdx = i;
                        }
                    }

                    for (GpuState st : states) {
                        st.pos++;
                    }

                    return maxIdx;
                } finally {
                    ctx.cl.clReleaseMemObject.invoke(partials);
                }
            } finally {
                ctx.cl.clReleaseMemObject.invoke(normed);
                ctx.cl.clReleaseMemObject.invoke(logitsDevice);
            }
        }
    }

    // =====================================================================
    // Kernel dispatch helpers -- one per KernelSource kernel. Each sets
    // args then enqueues an NDRange; all assume the caller already holds
    // ctx.dispatchLock (forward_layer/generate_token do).
    // =====================================================================
    private static void rmsNorm(GpuContext ctx, MemorySegment x, MemorySegment gamma, MemorySegment out,
            int features, double eps, MemorySegment partials) throws Throwable {
        int wgSize = KernelSource.RMSNORM_WORKGROUP_SIZE;
        int numGroups = (features + wgSize - 1) / wgSize;

        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kRmsnormPartialSumsq, 0, x);
            setArgPtr(ctx, tmp, ctx.kRmsnormPartialSumsq, 1, partials);
            setArgLocal(ctx, tmp, ctx.kRmsnormPartialSumsq, 2, (long) wgSize * ValueLayout.JAVA_FLOAT.byteSize());
            setArgInt(ctx, tmp, ctx.kRmsnormPartialSumsq, 3, features);
            enqueueNDRange(ctx, ctx.kRmsnormPartialSumsq, (long) numGroups * wgSize, wgSize);
        }

        float[] partialHost = new float[numGroups];
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) numGroups * ValueLayout.JAVA_FLOAT.byteSize());
            checkStatus((int) ctx.cl.clEnqueueReadBuffer.invoke(ctx.queue, partials, OpenClBindings.CL_TRUE,
                    0L, host.byteSize(), host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueReadBuffer(rmsnorm partials)");
            MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, partialHost, 0, numGroups);
        }
        double sumSq = 0.0;
        for (float p : partialHost) {
            sumSq += p;
        }
        float rms = (float) (1.0 / Math.sqrt(sumSq / features + eps));

        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kRmsnormApply, 0, x);
            setArgPtr(ctx, tmp, ctx.kRmsnormApply, 1, gamma);
            setArgPtr(ctx, tmp, ctx.kRmsnormApply, 2, out);
            setArgFloat(ctx, tmp, ctx.kRmsnormApply, 3, rms);
            setArgInt(ctx, tmp, ctx.kRmsnormApply, 4, features);
            enqueueNDRange(ctx, ctx.kRmsnormApply, features, 0);
        }
    }

    private static void quantizeActivationQ8_0(GpuContext ctx, MemorySegment x, MemorySegment outQ8, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kQuantizeActivationQ8_0, 0, x);
            setArgPtr(ctx, tmp, ctx.kQuantizeActivationQ8_0, 1, outQ8);
            setArgLocal(ctx, tmp, ctx.kQuantizeActivationQ8_0, 2, 32L * ValueLayout.JAVA_FLOAT.byteSize());
            setArgInt(ctx, tmp, ctx.kQuantizeActivationQ8_0, 3, len);
            enqueueNDRange(ctx, ctx.kQuantizeActivationQ8_0, len, 32);
        }
    }

    private static void q8_0GemvSplit(GpuContext ctx, MemorySegment xQ8, MemorySegment wQ8, MemorySegment out,
            int heads, int headDim, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kQ8_0GemvSplit, 0, xQ8);
            setArgPtr(ctx, tmp, ctx.kQ8_0GemvSplit, 1, wQ8);
            setArgPtr(ctx, tmp, ctx.kQ8_0GemvSplit, 2, out);
            setArgInt(ctx, tmp, ctx.kQ8_0GemvSplit, 3, heads);
            setArgInt(ctx, tmp, ctx.kQ8_0GemvSplit, 4, headDim);
            setArgInt(ctx, tmp, ctx.kQ8_0GemvSplit, 5, K);
            enqueueNDRange(ctx, ctx.kQ8_0GemvSplit, (long) heads * (headDim / 2), 0);
        }
    }

    private static void q8_0GemvPlain(GpuContext ctx, MemorySegment xQ8, MemorySegment wQ8, MemorySegment out,
            int N, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kQ8_0GemvPlain, 0, xQ8);
            setArgPtr(ctx, tmp, ctx.kQ8_0GemvPlain, 1, wQ8);
            setArgPtr(ctx, tmp, ctx.kQ8_0GemvPlain, 2, out);
            setArgInt(ctx, tmp, ctx.kQ8_0GemvPlain, 3, N);
            setArgInt(ctx, tmp, ctx.kQ8_0GemvPlain, 4, K);
            enqueueNDRange(ctx, ctx.kQ8_0GemvPlain, N, 0);
        }
    }

    private static void ropeApplySplit(GpuContext ctx, MemorySegment buf, MemorySegment cosTable, MemorySegment sinTable,
            int heads, int headDim, int cosSinOffset) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kRopeApplySplit, 0, buf);
            setArgPtr(ctx, tmp, ctx.kRopeApplySplit, 1, cosTable);
            setArgPtr(ctx, tmp, ctx.kRopeApplySplit, 2, sinTable);
            setArgInt(ctx, tmp, ctx.kRopeApplySplit, 3, heads);
            setArgInt(ctx, tmp, ctx.kRopeApplySplit, 4, headDim);
            setArgInt(ctx, tmp, ctx.kRopeApplySplit, 5, cosSinOffset);
            enqueueNDRange(ctx, ctx.kRopeApplySplit, (long) heads * (headDim / 2), 0);
        }
    }

    private static void attnScores(GpuContext ctx, MemorySegment qAllHeads, MemorySegment kCache, MemorySegment scores,
            int qHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive, float rsqrtD) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kAttnScores, 0, qAllHeads);
            setArgPtr(ctx, tmp, ctx.kAttnScores, 1, kCache);
            setArgPtr(ctx, tmp, ctx.kAttnScores, 2, scores);
            setArgInt(ctx, tmp, ctx.kAttnScores, 3, qHeadOff);
            setArgInt(ctx, tmp, ctx.kAttnScores, 4, headDim);
            setArgInt(ctx, tmp, ctx.kAttnScores, 5, kvDim);
            setArgInt(ctx, tmp, ctx.kAttnScores, 6, kvHeadOff);
            setArgInt(ctx, tmp, ctx.kAttnScores, 7, posInclusive);
            setArgFloat(ctx, tmp, ctx.kAttnScores, 8, rsqrtD);
            enqueueNDRange(ctx, ctx.kAttnScores, posInclusive, 0);
        }
    }

    private static void softmaxInplace(GpuContext ctx, MemorySegment scores, int len) throws Throwable {
        int localSize = nextPow2(Math.max(len, 1));
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kSoftmaxInplace, 0, scores);
            setArgLocal(ctx, tmp, ctx.kSoftmaxInplace, 1, (long) localSize * ValueLayout.JAVA_FLOAT.byteSize());
            setArgInt(ctx, tmp, ctx.kSoftmaxInplace, 2, len);
            // Single work-group dispatch: global size == local size (see kernel javadoc).
            enqueueNDRangeExact(ctx, ctx.kSoftmaxInplace, localSize, localSize);
        }
    }

    private static int nextPow2(int v) {
        int p = 1;
        while (p < v) {
            p <<= 1;
        }
        return Math.min(p, 256);
    }

    private static void attnWeightedSum(GpuContext ctx, MemorySegment scores, MemorySegment vCache, MemorySegment attnOutAllHeads,
            int outHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kAttnWeightedSum, 0, scores);
            setArgPtr(ctx, tmp, ctx.kAttnWeightedSum, 1, vCache);
            setArgPtr(ctx, tmp, ctx.kAttnWeightedSum, 2, attnOutAllHeads);
            setArgInt(ctx, tmp, ctx.kAttnWeightedSum, 3, outHeadOff);
            setArgInt(ctx, tmp, ctx.kAttnWeightedSum, 4, headDim);
            setArgInt(ctx, tmp, ctx.kAttnWeightedSum, 5, kvDim);
            setArgInt(ctx, tmp, ctx.kAttnWeightedSum, 6, kvHeadOff);
            setArgInt(ctx, tmp, ctx.kAttnWeightedSum, 7, posInclusive);
            enqueueNDRange(ctx, ctx.kAttnWeightedSum, headDim, 0);
        }
    }

    private static void swigluActivate(GpuContext ctx, MemorySegment gate, MemorySegment up, MemorySegment out, int hidden) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kSwigluActivate, 0, gate);
            setArgPtr(ctx, tmp, ctx.kSwigluActivate, 1, up);
            setArgPtr(ctx, tmp, ctx.kSwigluActivate, 2, out);
            setArgInt(ctx, tmp, ctx.kSwigluActivate, 3, hidden);
            enqueueNDRange(ctx, ctx.kSwigluActivate, hidden, 0);
        }
    }

    private static void residualAdd(GpuContext ctx, MemorySegment x, MemorySegment y, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kResidualAdd, 0, x);
            setArgPtr(ctx, tmp, ctx.kResidualAdd, 1, y);
            setArgInt(ctx, tmp, ctx.kResidualAdd, 2, len);
            enqueueNDRange(ctx, ctx.kResidualAdd, len, 0);
        }
    }

    private static void f32Gemv(GpuContext ctx, MemorySegment a, MemorySegment B, MemorySegment out, int K, int N) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            setArgPtr(ctx, tmp, ctx.kF32Gemv, 0, a);
            setArgPtr(ctx, tmp, ctx.kF32Gemv, 1, B);
            setArgPtr(ctx, tmp, ctx.kF32Gemv, 2, out);
            setArgInt(ctx, tmp, ctx.kF32Gemv, 3, K);
            setArgInt(ctx, tmp, ctx.kF32Gemv, 4, N);
            enqueueNDRange(ctx, ctx.kF32Gemv, N, 0);
        }
    }

    /**
     * Host round-trip cache write -- see class javadoc for why.
     */
    private static void writeIntoCache(GpuContext ctx, MemorySegment src, MemorySegment cache, long elementOffset, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) len * ValueLayout.JAVA_FLOAT.byteSize());
            checkStatus((int) ctx.cl.clEnqueueReadBuffer.invoke(ctx.queue, src, OpenClBindings.CL_TRUE,
                    0L, host.byteSize(), host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueReadBuffer(cache write, read new K/V)");
            checkStatus((int) ctx.cl.clEnqueueWriteBuffer.invoke(ctx.queue, cache, OpenClBindings.CL_TRUE,
                    elementOffset * ValueLayout.JAVA_FLOAT.byteSize(), host.byteSize(), host, 0,
                    MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueWriteBuffer(cache write, at offset)");
        }
    }

    // ================= FFM plumbing =================
    private static void setArgPtr(GpuContext ctx, Arena tmp, MemorySegment kernel, int index, MemorySegment value) throws Throwable {
        MemorySegment holder = tmp.allocate(ValueLayout.ADDRESS);
        holder.set(ValueLayout.ADDRESS, 0, value);
        checkStatus((int) ctx.cl.clSetKernelArg.invoke(kernel, index, ValueLayout.ADDRESS.byteSize(), holder),
                "clSetKernelArg[" + index + "]");
    }

    private static void setArgInt(GpuContext ctx, Arena tmp, MemorySegment kernel, int index, int value) throws Throwable {
        MemorySegment holder = tmp.allocate(ValueLayout.JAVA_INT);
        holder.set(ValueLayout.JAVA_INT, 0, value);
        checkStatus((int) ctx.cl.clSetKernelArg.invoke(kernel, index, ValueLayout.JAVA_INT.byteSize(), holder),
                "clSetKernelArg[" + index + "]");
    }

    private static void setArgFloat(GpuContext ctx, Arena tmp, MemorySegment kernel, int index, float value) throws Throwable {
        MemorySegment holder = tmp.allocate(ValueLayout.JAVA_FLOAT);
        holder.set(ValueLayout.JAVA_FLOAT, 0, value);
        checkStatus((int) ctx.cl.clSetKernelArg.invoke(kernel, index, ValueLayout.JAVA_FLOAT.byteSize(), holder),
                "clSetKernelArg[" + index + "]");
    }

    /**
     * __local array argument: no host-side value, just a device-local
     * allocation of byteSize bytes.
     */
    private static void setArgLocal(GpuContext ctx, Arena tmp, MemorySegment kernel, int index, long byteSize) throws Throwable {
        checkStatus((int) ctx.cl.clSetKernelArg.invoke(kernel, index, byteSize, MemorySegment.NULL),
                "clSetKernelArg[local," + index + "]");
    }

    private static void enqueueNDRange(GpuContext ctx, MemorySegment kernel, long workItems, int localSizeHint) throws Throwable {
        long global = localSizeHint > 0 ? ((workItems + localSizeHint - 1) / localSizeHint) * localSizeHint : workItems;
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment globalSize = tmp.allocate(ValueLayout.JAVA_LONG);
            globalSize.set(ValueLayout.JAVA_LONG, 0, global);
            MemorySegment localSize = MemorySegment.NULL;
            if (localSizeHint > 0) {
                localSize = tmp.allocate(ValueLayout.JAVA_LONG);
                localSize.set(ValueLayout.JAVA_LONG, 0, (long) localSizeHint);
            }
            checkStatus((int) ctx.cl.clEnqueueNDRangeKernel.invoke(ctx.queue, kernel,
                    1, MemorySegment.NULL, globalSize, localSize, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueNDRangeKernel");
            checkStatus((int) ctx.cl.clFinish.invoke(ctx.queue), "clFinish");
        }
    }

    /**
     * Exact global==local single-work-group dispatch, used by softmaxInplace.
     */
    private static void enqueueNDRangeExact(GpuContext ctx, MemorySegment kernel, long globalSize, long localSize) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment g = tmp.allocate(ValueLayout.JAVA_LONG);
            g.set(ValueLayout.JAVA_LONG, 0, globalSize);
            MemorySegment l = tmp.allocate(ValueLayout.JAVA_LONG);
            l.set(ValueLayout.JAVA_LONG, 0, localSize);
            checkStatus((int) ctx.cl.clEnqueueNDRangeKernel.invoke(ctx.queue, kernel,
                    1, MemorySegment.NULL, g, l, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueNDRangeKernel(exact)");
            checkStatus((int) ctx.cl.clFinish.invoke(ctx.queue), "clFinish");
        }
    }

    private static void checkStatus(int status, String call) {
        if (status != OpenClBindings.CL_SUCCESS) {
            throw new IllegalStateException("OpenCL error in " + call + ": code " + status);
        }
    }
}
