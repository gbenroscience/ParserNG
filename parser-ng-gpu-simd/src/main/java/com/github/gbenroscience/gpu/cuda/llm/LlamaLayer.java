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
 

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.NoSuchElementException;

/**
 * @author GBEMIRO
 * CUDA counterpart of {@code com.github.gbenroscience.gpu.llm.LlamaLayer}
 * -- identical six-stage forward_layer structure, identical GQA grouping,
 * identical device-resident FP32 K/V cache rationale (see the OpenCL
 * version's class javadoc, which applies unchanged). Only the dispatch
 * mechanics differ, and only because CUDA's driver API shape forces it:
 *
 * - Device buffers are CUdeviceptr values -- plain {@code long} handles,
 *   not MemorySegments -- matching CudaCompositeExpression's convention
 *   for the interpreter kernels. wq_q8_0/xNorm/k_cache_f32/etc. are all
 *   `long` here where the OpenCL version used `MemorySegment` (cl_mem).
 *
 * - Kernel arguments are a hand-built {@code void** kernelParams} array
  (one small buffer per argument, an array of pointers to those
  buffers) rather than OpenCL's one-arg-at-a-time clSetKernelArg --
  same construction pattern as CudaCompositeExpression.buildKernelParams,
  just per-kernel here since there are 13 distinct signatures instead
  of one shared interpreter signature.

- The three kernels that took an OpenCL `__local` scratch parameter
  (quantize_activation_q8_0_blocks, rmsnorm_partial_sumsq,
  softmax_inplace) have ONE FEWER kernel argument here -- shared memory
  is supplied via cuLaunchKernel's sharedMemBytes argument instead (see
  KernelSource's javadoc). Get the byte count wrong here and the
  kernel reads/writes past its shared allocation; each dispatch helper
  below computes it right next to the launch that needs it.

- Explicit block/grid dimensions are required for every launch (CUDA
  has no OpenCL-style "let the driver pick a local size" default) --
  DEFAULT_BLOCK_SIZE (256) is used for every simple bound-checked 1D
  kernel; QUANTIZE_BLOCK_SIZE (32) and RMSNORM_WORKGROUP_SIZE (256) are
  used where the kernel algorithm itself fixes the block size.

- cuCtxSetCurrent(ctx.context) is called once per forward_layer (the
  whole call executes under ctx.dispatchLock, same as the OpenCL
  version's synchronized block), then every individual launch is
  followed by cuCtxSynchronize -- matching the OpenCL version's
  per-kernel clFinish exactly: correctness/debuggability over the
  throughput a stream-pipelined version could get, consistent with
  this codebase's stated "correctness over cleverness" priority for
  code that hasn't run against real hardware yet.

UNVERIFIED, same caveat as everything else in this port: no CUDA
driver, GPU, NVRTC toolchain, or reference model were available while
writing this. Every dispatch helper is a traced port of the OpenCL
version's equivalent (see LlamaLayer's per-method structure, which
this mirrors 1:1) -- treat as an untested first draft; diff per-layer
activations against a working CPU or OpenCL path before trusting it.
 */
public final class LlamaLayer {

    private LlamaLayer() {
    }

    /** Mirrors LlamaLayer.Config / LlamaLayerInt8.Config exactly -- unchanged, backend-independent. */
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
     * GPU-resident weights for ONE decoder layer. Q8_0 tensors uploaded
     * AS-IS (raw GGUF block bytes), dequantized on-device inside the GEMV
     * kernels -- same contract as GpuLlamaLayer.GpuWeights, just CUdeviceptr
     * (`long`) handles instead of cl_mem MemorySegments.
     */
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

        /** Same GGUF tensor-name lookups as LlamaLayer.GpuWeights.fromGguf; loadQ8_0() gives raw block bytes, dequant happens on-device. */
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
     * Per-sequence GPU state: KV cache (FP32) and RoPE tables, plus
     * persistent scratch buffers reused every forward_layer call -- zero
     * per-token device allocation once constructed. Same layout as
     * GpuLlamaLayer.GpuState, `long` CUdeviceptr handles instead of cl_mem.
     */
    public static final class GpuState implements AutoCloseable {
        final GpuContext ctx;
        final int kvDim;

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

        /** Same formula as LlamaLayer.GpuState.precomputeRope: theta_i = base^(-2i/head_dim). */
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
        }
    }

    // ================= device alloc/upload helpers (long CUdeviceptr) =================

    private static long allocFloats(GpuContext ctx, long count) throws Throwable {
        return allocBytes(ctx, count * ValueLayout.JAVA_FLOAT.byteSize());
    }

    private static long allocBytes(GpuContext ctx, long byteCount) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment ptrBuf = tmp.allocate(ValueLayout.JAVA_LONG);
            GpuContext.check((int) ctx.cu.cuMemAlloc.invoke(ptrBuf, byteCount), "cuMemAlloc");
            return ptrBuf.get(ValueLayout.JAVA_LONG, 0);
        }
    }

    private static long uploadBytes(GpuContext ctx, byte[] data) throws Throwable {
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

    private static long uploadFloats(GpuContext ctx, float[] data) throws Throwable {
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

    private static void freeQuietly(GpuContext ctx, long devicePtr) {
        try {
            if (devicePtr != 0L) {
                ctx.cu.cuMemFree.invoke(devicePtr);
            }
        } catch (Throwable t) {
            // best-effort cleanup
        }
    }

    // =====================================================================
    // THE DECODER LAYER
    // =====================================================================

    /**
     * Runs one decoder layer for the CURRENT token, in place on x.
     * x: device pointer, [dim] floats, in/out -- same contract as
     * GpuLlamaLayer.forward_layer's MemorySegment x, just a CUdeviceptr
     * `long` here.
     */
    public static void forward_layer(long x, GpuWeights w, GpuState s, Config cfg, GpuContext ctx) throws Throwable {
        final int dim = cfg.dim;
        final int hidden = cfg.hidden_dim;
        final int headDim = cfg.head_dim;
        final int numHeads = cfg.num_heads;
        final int kvHeads = cfg.kv_heads;
        final int kvDim = s.kvDim;
        final int halfDim = headDim / 2;
        final int groupSize = numHeads / kvHeads; // GQA grouping
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

            // Cache write: small host round-trip (kvDim floats), same
            // choice LlamaLayer's writeIntoCache makes rather than
            // adding a cuMemcpyDtoD binding this project doesn't
            // otherwise need.
            writeIntoCache(ctx, s.kNew, s.k_cache_f32, (long) s.pos * kvDim, kvDim);
            writeIntoCache(ctx, s.vNew, s.v_cache_f32, (long) s.pos * kvDim, kvDim);

            // --- THE MULTI-HEAD ATTENTION LOOP ---
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
     * Runs the full model for one token: embedding -> N decoder layers ->
     * final norm -> LM head. Mirrors GpuLlamaLayer.generate_token exactly;
     * the LM head projection uses f32Gemv, same as the OpenCL version.
     */
    public static int generate_token(
            long tokenEmbeddingDevice, // [dim], mutated in place by forward_layer
            GpuWeights[] layers,
            long finalNormGammaDevice, // [dim]
            long lmHeadDevice,         // [dim, vocab_size], FP32
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
        int maxPartials = (cfg.dim + KernelSource.RMSNORM_WORKGROUP_SIZE - 1) / KernelSource.RMSNORM_WORKGROUP_SIZE;
        long partials = allocFloats(ctx, maxPartials);

        try {
            synchronized (ctx.dispatchLock) {
                GpuContext.check((int) ctx.cu.cuCtxSetCurrent.invoke(ctx.context), "cuCtxSetCurrent");
                rmsNorm(ctx, x, finalNormGammaDevice, normed, cfg.dim, cfg.norm_eps, partials);
                f32Gemv(ctx, normed, lmHeadDevice, logitsDevice, cfg.dim, vocabSize);
            }

            try (Arena tmp = Arena.ofConfined()) {
                MemorySegment logitsHost = tmp.allocate((long) vocabSize * ValueLayout.JAVA_FLOAT.byteSize());
                GpuContext.check((int) ctx.cu.cuMemcpyDtoH.invoke(logitsHost, logitsDevice, logitsHost.byteSize()),
                        "cuMemcpyDtoH(logits)");

                float[] logits = new float[vocabSize];
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
            }
        } finally {
            freeQuietly(ctx, normed);
            freeQuietly(ctx, logitsDevice);
            freeQuietly(ctx, partials);
        }
    }

    // =====================================================================
    // Kernel dispatch helpers -- one per KernelSource kernel. Each
    // builds a kernelParams array in argument order matching the kernel's
    // CUDA signature exactly, computes grid/block dims (and sharedMemBytes
    // for the three reduction kernels), launches, then synchronizes --
    // mirroring LlamaLayer's dispatch helpers 1:1. All assume the
    // caller already holds ctx.dispatchLock and has set the current
    // context (forward_layer/generate_token do both).
    // =====================================================================

    private static void rmsNorm(GpuContext ctx, long x, long gamma, long out,
            int features, double eps, long partials) throws Throwable {
        int wgSize = KernelSource.RMSNORM_WORKGROUP_SIZE;
        int numGroups = (features + wgSize - 1) / wgSize;

        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, x), argL(tmp, partials), argI(tmp, features));
            launch1D(ctx, ctx.kRmsnormPartialSumsq, (long) numGroups * wgSize, wgSize,
                    wgSize * (int) ValueLayout.JAVA_FLOAT.byteSize(), params);
        }

        float[] partialHost = new float[numGroups];
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) numGroups * ValueLayout.JAVA_FLOAT.byteSize());
            GpuContext.check((int) ctx.cu.cuMemcpyDtoH.invoke(host, partials, host.byteSize()),
                    "cuMemcpyDtoH(rmsnorm partials)");
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

    private static void quantizeActivationQ8_0(GpuContext ctx, long x, long outQ8, int len) throws Throwable {
        int blockSize = KernelSource.QUANTIZE_BLOCK_SIZE;
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, x), argL(tmp, outQ8), argI(tmp, len));
            launch1D(ctx, ctx.kQuantizeActivationQ8_0, len, blockSize,
                    blockSize * (int) ValueLayout.JAVA_FLOAT.byteSize(), params);
        }
    }

    private static void q8_0GemvSplit(GpuContext ctx, long xQ8, long wQ8, long out,
            int heads, int headDim, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, xQ8), argL(tmp, wQ8), argL(tmp, out),
                    argI(tmp, heads), argI(tmp, headDim), argI(tmp, K));
            launch1D(ctx, ctx.kQ8_0GemvSplit, (long) heads * (headDim / 2), KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    private static void q8_0GemvPlain(GpuContext ctx, long xQ8, long wQ8, long out,
            int N, int K) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, xQ8), argL(tmp, wQ8), argL(tmp, out), argI(tmp, N), argI(tmp, K));
            launch1D(ctx, ctx.kQ8_0GemvPlain, N, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    private static void ropeApplySplit(GpuContext ctx, long buf, long cosTable, long sinTable,
            int heads, int headDim, int cosSinOffset) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, buf), argL(tmp, cosTable), argL(tmp, sinTable),
                    argI(tmp, heads), argI(tmp, headDim), argI(tmp, cosSinOffset));
            launch1D(ctx, ctx.kRopeApplySplit, (long) heads * (headDim / 2), KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    private static void attnScores(GpuContext ctx, long qAllHeads, long kCache, long scores,
            int qHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive, float rsqrtD) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, qAllHeads), argL(tmp, kCache), argL(tmp, scores),
                    argI(tmp, qHeadOff), argI(tmp, headDim), argI(tmp, kvDim),
                    argI(tmp, kvHeadOff), argI(tmp, posInclusive), argF(tmp, rsqrtD));
            launch1D(ctx, ctx.kAttnScores, posInclusive, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    private static void softmaxInplace(GpuContext ctx, long scores, int len) throws Throwable {
        int localSize = nextPow2(Math.max(len, 1));
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, scores), argI(tmp, len));
            // Exactly one block: workItems == blockSize == localSize forces gridDim == 1,
            // matching the kernel's single-block reduction requirement.
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

    private static void attnWeightedSum(GpuContext ctx, long scores, long vCache, long attnOutAllHeads,
            int outHeadOff, int headDim, int kvDim, int kvHeadOff, int posInclusive) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp,
                    argL(tmp, scores), argL(tmp, vCache), argL(tmp, attnOutAllHeads),
                    argI(tmp, outHeadOff), argI(tmp, headDim), argI(tmp, kvDim),
                    argI(tmp, kvHeadOff), argI(tmp, posInclusive));
            launch1D(ctx, ctx.kAttnWeightedSum, headDim, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    private static void swigluActivate(GpuContext ctx, long gate, long up, long out, int hidden) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, gate), argL(tmp, up), argL(tmp, out), argI(tmp, hidden));
            launch1D(ctx, ctx.kSwigluActivate, hidden, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    private static void residualAdd(GpuContext ctx, long x, long y, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, x), argL(tmp, y), argI(tmp, len));
            launch1D(ctx, ctx.kResidualAdd, len, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    private static void f32Gemv(GpuContext ctx, long a, long B, long out, int K, int N) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment params = kernelParams(tmp, argL(tmp, a), argL(tmp, B), argL(tmp, out), argI(tmp, K), argI(tmp, N));
            launch1D(ctx, ctx.kF32Gemv, N, KernelSource.DEFAULT_BLOCK_SIZE, 0, params);
        }
    }

    /** Host round-trip cache write -- same rationale as GpuLlamaLayer.writeIntoCache. Offset is folded directly into the CUdeviceptr value (valid pointer arithmetic within one cuMemAlloc allocation). */
    private static void writeIntoCache(GpuContext ctx, long src, long cache, long elementOffset, int len) throws Throwable {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) len * ValueLayout.JAVA_FLOAT.byteSize());
            GpuContext.check((int) ctx.cu.cuMemcpyDtoH.invoke(host, src, host.byteSize()),
                    "cuMemcpyDtoH(cache write, read new K/V)");
            long dstOffset = cache + elementOffset * ValueLayout.JAVA_FLOAT.byteSize();
            GpuContext.check((int) ctx.cu.cuMemcpyHtoD.invoke(dstOffset, host, host.byteSize()),
                    "cuMemcpyHtoD(cache write, at offset)");
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

    /** Builds the {@code void** kernelParams} array cuLaunchKernel expects -- one pointer per argument, in kernel-signature order. */
    private static MemorySegment kernelParams(Arena a, MemorySegment... argPtrs) {
        MemorySegment arr = a.allocate(ValueLayout.ADDRESS, argPtrs.length);
        for (int i = 0; i < argPtrs.length; i++) {
            arr.setAtIndex(ValueLayout.ADDRESS, i, argPtrs[i]);
        }
        return arr;
    }

    /** Rounds workItems up to a whole number of blocks, launches on the default stream, then synchronizes (mirrors the OpenCL version's per-kernel clFinish). */
    private static void launch1D(GpuContext ctx, MemorySegment function, long workItems, int blockSize,
            int sharedMemBytes, MemorySegment kernelParams) throws Throwable {
        int gridDim = (int) ((workItems + blockSize - 1) / blockSize);
        GpuContext.check((int) ctx.cu.cuLaunchKernel.invoke(
                function,
                gridDim, 1, 1,
                blockSize, 1, 1,
                sharedMemBytes,
                MemorySegment.NULL, // default stream
                kernelParams,
                MemorySegment.NULL),
                "cuLaunchKernel");
        GpuContext.check((int) ctx.cu.cuCtxSynchronize.invoke(), "cuCtxSynchronize");
    }
}