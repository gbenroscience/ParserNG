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

/**
 *
 * @author GBEMIRO 
 * OpenCL C kernels for GPU-side Llama-style decoder inference, operating on
 * GGUF's native Q8_0 block format (2-byte FP16 scale + 32 INT8 values per
 * block -- see GGUFLoader.calculateTensorBytes case 8, and
 * KernelsInt8.matmul_q8_0_1xN_split_opt, whose exact per-block dequant +
 * dot-product algorithm this kernel ports).
 *
 * DELIBERATELY NOT what LlamaLayerInt8.matmul_q8_1xN expects (flat byte[]
 * weights + one float scale per OUTPUT COLUMN, not per 32-element block) --
 * that format has no GGUFLoader path to produce it from a real model file.
 * This targets the format GGUFLoader.loadQ8_0() actually emits.
 *
 * SCOPE, STATED PLAINLY:
 *  - Single-token decode path only (M=1 GEMV, not batched prefill GEMM).
 *    Prefill/batched prompt processing would need a real GEMM kernel
 *    (tiled, shared-memory blocked) -- out of scope here, and a
 *    meaningfully larger undertaking than what's below.
 *  - Attention is UNFUSED: three kernel launches per forward pass
 *    (scores, softmax, weighted-sum), each looping over heads internally.
 *    Not a fused flash-attention-style kernel -- correctness over cleverness
 *    given this is unverified against real hardware. A fused kernel is a
 *    legitimate future optimization, not attempted here.
 *  - RMSNorm is two-phase: a device kernel produces one partial sum-of-
 *    squares per work-group, the tiny partials array (dim/256-ish elements)
 *    is summed on the HOST, then a second device kernel applies the
 *    normalization. This avoids a single-pass cross-workgroup reduction
 *    (a well-known source of subtle bugs) at the cost of one extra small
 *    host<->device round trip per RMSNorm call -- negligible next to the
 *    GEMV cost it sits beside.
 *  - No kernel here has been run. Ported carefully from your scalar/SIMD
 *    reference implementations, but treat as unverified.
 */
public final class KernelSource {

    private KernelSource() {
    }

    public static final String KERNEL_QUANTIZE_I8 = "quantize_i8";
    public static final String KERNEL_QUANTIZE_ACTIVATION_Q8_0 = "quantize_activation_q8_0_blocks";
    public static final String KERNEL_Q8_0_GEMV_SPLIT = "q8_0_gemv_split";
    public static final String KERNEL_Q8_0_GEMV_PLAIN = "q8_0_gemv_plain";
    public static final String KERNEL_ROPE_APPLY_SPLIT = "rope_apply_split";
    public static final String KERNEL_RMSNORM_PARTIAL_SUMSQ = "rmsnorm_partial_sumsq";
    public static final String KERNEL_RMSNORM_APPLY = "rmsnorm_apply";
    public static final String KERNEL_ATTN_SCORES = "attn_scores";
    public static final String KERNEL_SOFTMAX_INPLACE = "softmax_inplace";
    public static final String KERNEL_ATTN_WEIGHTED_SUM = "attn_weighted_sum";
    public static final String KERNEL_SWIGLU_ACTIVATE = "swiglu_activate";
    public static final String KERNEL_RESIDUAL_ADD = "residual_add";
    public static final String KERNEL_F32_GEMV = "f32_gemv";

    /** Work-group size used by the RMSNorm partial-sum kernel's local reduction. Must match host dispatch. */
    public static final int RMSNORM_WORKGROUP_SIZE = 256;

    public static final String OPENCL_SOURCE = """
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable

        #define Q8_0_GROUP_SIZE 32
        #define Q8_0_BLOCK_SIZE 34

        /*
         * Decodes a GGML fp16 scale (as stored in the first 2 bytes of every
         * Q8_0 block) to float. Matches KernelsInt8.f16ToF32 exactly --
         * OpenCL's built-in vload_half/half conversions assume IEEE binary16
         * laid out the same way GGML stores it, so this can use the native
         * conversion rather than a hand-rolled bit-twiddle port.
         */
        inline float decode_fp16(uchar lo, uchar hi) {
            ushort bits = (ushort)lo | ((ushort)hi << 8);
            return vload_half(0, (const __global half*)&bits);
        }

        /*
         * quantize_i8: FP32 -> INT8 with a single caller-supplied scale.
         * Direct port of KernelsInt8.quantize_i8's round-and-clamp formula.
         * One work-item per element.
         *
         * NOTE: this produces PLAIN INT8 with one shared scale for the
         * whole tensor -- it is NOT the format q8_0_gemv_split expects for
         * its activation input (which needs per-32-block scales, GGUF Q8_0
         * layout). Use quantize_activation_q8_0_blocks below for that.
         * Kept here as a correct, standalone, generically useful primitive.
         */
        __kernel void quantize_i8(
            __global const float* x,
            __global char* x_q8,
            const float invScale)
        {
            const int i = get_global_id(0);
            float val = x[i] * invScale;
            int q = (int) rint(val);
            q = max(-127, min(127, q));
            x_q8[i] = (char) q;
        }

        /*
         * quantize_activation_q8_0_blocks: FP32 -> GGUF Q8_0 block format
         * (2-byte fp16 scale + 32 INT8 values per 32-element block), with a
         * fresh per-block absmax scale computed on-device. This is the
         * ACTIVATION-side counterpart to the weight quantization GGUF files
         * already ship with -- it does not correspond to any method in the
         * uploaded reference files (llama.cpp/ggml calls the equivalent
         * operation quantize_row_q8_0; it wasn't among what was shared
         * here), so it's implemented from the well-defined Q8_0 scheme
         * itself, using OpenCL's native vstore_half for the FP16 scale
         * encode (mirroring vload_half used for decode elsewhere in this
         * file) rather than a hand-rolled bit-manipulation encoder.
         *
         * MUST be launched with local_work_size = 32 (one work-item per
         * element in a block) and global_work_size = numBlocks * 32.
         */
        __kernel void quantize_activation_q8_0_blocks(
            __global const float* x,
            __global uchar* x_q8_0,
            __local float* scratch,
            const int len)
        {
            const int block = get_group_id(0);
            const int lane = get_local_id(0); // 0..31
            const int blockOff = block * Q8_0_GROUP_SIZE;
            const int outOff = block * Q8_0_BLOCK_SIZE;

            float v = x[blockOff + lane];
            scratch[lane] = fabs(v);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int stride = 16; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    scratch[lane] = fmax(scratch[lane], scratch[lane + stride]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float absmax = scratch[0];
            float scale = absmax / 127.0f;
            float invScale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            int q = (int) rint(v * invScale);
            q = max(-127, min(127, q));
            x_q8_0[outOff + 2 + lane] = (uchar)(char) q;

            if (lane == 0) {
                vstore_half(scale, 0, (__global half*)&x_q8_0[outOff]);
            }
        }

        /*
         * q8_0_gemv_split: GGML Q8_0 block-format GEMV, split even/odd
         * output layout -- exact port of
         * KernelsInt8.matmul_q8_0_1xN_split_opt's per-block algorithm.
         *
         * x_q8_0:  quantized activation, Q8_0 block format, K elements
         * W_q8_0:  quantized weight matrix, Q8_0 block format,
         *          [num_heads * head_dim, K] row-major, each row its own
         *          block sequence (same layout matmul_q8_0_1xN_split_opt
         *          assumes via B_stride = numBlocks * Q8_0_BLOCK_SIZE)
         * out_f32_split: output, head_dim elements per head, laid out
         *          [even-indexed outputs][odd-indexed outputs] per head --
         *          same split layout the CPU kernel produces, so
         *          rope_apply_split below (also split-layout-native) needs
         *          no extra shuffle step in between.
         *
         * One work-item per (head, half-dim-index) pair -- i.e. one
         * work-item computes BOTH the even and odd output for one head's
         * pair of adjacent dimensions, mirroring the CPU loop's inner body
         * exactly (which computes n0/n1 = 2*i, 2*i+1 together per iteration).
         */
        __kernel void q8_0_gemv_split(
            __global const uchar* x_q8_0,
            __global const uchar* W_q8_0,
            __global float* out_f32_split,
            const int qHeads,
            const int head_dim,
            const int K)
        {
            const int halfDim = head_dim >> 1;
            const int gid = get_global_id(0); // 0 .. qHeads*halfDim - 1
            const int h = gid / halfDim;
            const int i = gid % halfDim;
            if (h >= qHeads) return;

            const int numBlocks = K / Q8_0_GROUP_SIZE;
            const int B_stride = numBlocks * Q8_0_BLOCK_SIZE;

            const int headOutOff = h * head_dim;
            const int evenOutOff = headOutOff;
            const int oddOutOff = headOutOff + halfDim;

            const int n0 = h * head_dim + (i << 1);
            const int n1 = n0 + 1;

            float acc0 = 0.0f;
            float acc1 = 0.0f;

            for (int b = 0; b < numBlocks; b++) {
                const int xBlockOff = b * Q8_0_BLOCK_SIZE;
                const int w0BlockOff = n0 * B_stride + b * Q8_0_BLOCK_SIZE;
                const int w1BlockOff = n1 * B_stride + b * Q8_0_BLOCK_SIZE;

                float xScale = decode_fp16(x_q8_0[xBlockOff], x_q8_0[xBlockOff + 1]);
                float w0Scale = decode_fp16(W_q8_0[w0BlockOff], W_q8_0[w0BlockOff + 1]);
                float w1Scale = decode_fp16(W_q8_0[w1BlockOff], W_q8_0[w1BlockOff + 1]);

                float scale0 = xScale * w0Scale;
                float scale1 = xScale * w1Scale;

                int iacc0 = 0, iacc1 = 0;
                for (int j = 0; j < Q8_0_GROUP_SIZE; j++) {
                    char xv = (char) x_q8_0[xBlockOff + 2 + j];
                    char w0v = (char) W_q8_0[w0BlockOff + 2 + j];
                    char w1v = (char) W_q8_0[w1BlockOff + 2 + j];
                    iacc0 += (int) xv * (int) w0v;
                    iacc1 += (int) xv * (int) w1v;
                }

                acc0 += iacc0 * scale0;
                acc1 += iacc1 * scale1;
            }

            out_f32_split[evenOutOff + i] = acc0;
            out_f32_split[oddOutOff + i] = acc1;
        }

        /*
         * q8_0_gemv_plain: same Q8_0 block-format GEMV algorithm as
         * q8_0_gemv_split, but natural contiguous output order --
         * out_f32[n] for n in [0,N), no even/odd pairing. The split
         * kernel's pairing is specifically a RoPE optimization for Q/K
         * projections (adjacent dims get rotated together); it does NOT
         * apply to the FFN gate/up/down projections, which need plain
         * contiguous output. One work-item per output column n.
         */
        __kernel void q8_0_gemv_plain(
            __global const uchar* x_q8_0,
            __global const uchar* W_q8_0,
            __global float* out_f32,
            const int N,
            const int K)
        {
            const int n = get_global_id(0);
            if (n >= N) return;

            const int numBlocks = K / Q8_0_GROUP_SIZE;
            const int B_stride = numBlocks * Q8_0_BLOCK_SIZE;
            const int wRowOff = n * B_stride;

            float acc = 0.0f;
            for (int b = 0; b < numBlocks; b++) {
                const int xBlockOff = b * Q8_0_BLOCK_SIZE;
                const int wBlockOff = wRowOff + b * Q8_0_BLOCK_SIZE;

                float xScale = decode_fp16(x_q8_0[xBlockOff], x_q8_0[xBlockOff + 1]);
                float wScale = decode_fp16(W_q8_0[wBlockOff], W_q8_0[wBlockOff + 1]);
                float scale = xScale * wScale;

                int iacc = 0;
                for (int j = 0; j < Q8_0_GROUP_SIZE; j++) {
                    char xv = (char) x_q8_0[xBlockOff + 2 + j];
                    char wv = (char) W_q8_0[wBlockOff + 2 + j];
                    iacc += (int) xv * (int) wv;
                }
                acc += iacc * scale;
            }
            out_f32[n] = acc;
        }

        /*
         * rope_apply_split: pairwise rotation on split-layout Q/K buffers.
         * Exact port of KernelsFloat.rope_inplace_split_ws's formula:
         *   x0' = x0*cos - x1*sin
         *   x1' = x0*sin + x1*cos
         * One work-item per (head, half-dim-index) pair, run once for Q
         * (qBuf) and once for K (kBuf) -- two separate dispatches from the
         * host, same kernel, different buffer/headCount args.
         */
        __kernel void rope_apply_split(
            __global float* buf,
            __global const float* cos_table,
            __global const float* sin_table,
            const int heads,
            const int head_dim,
            const int cosSinOffset)
        {
            const int halfDim = head_dim >> 1;
            const int gid = get_global_id(0);
            const int h = gid / halfDim;
            const int i = gid % halfDim;
            if (h >= heads) return;

            const int headOff = h * head_dim;
            const int evenOff = headOff;
            const int oddOff = headOff + halfDim;

            float c = cos_table[cosSinOffset + i];
            float s = sin_table[cosSinOffset + i];
            float x0 = buf[evenOff + i];
            float x1 = buf[oddOff + i];

            buf[evenOff + i] = x0 * c - x1 * s;
            buf[oddOff + i] = x0 * s + x1 * c;
        }

        /*
         * rmsnorm_partial_sumsq: phase 1 of 2 -- one partial sum-of-squares
         * per work-group via local-memory tree reduction. Host sums the
         * (small) partials array and computes rms = 1/sqrt(mean+eps) before
         * calling rmsnorm_apply. Matches rms_norm_fast's sum(x^2) exactly;
         * only the reduction STRATEGY differs from the CPU's linear
         * accumulation (order-of-addition floating point differences are
         * expected and immaterial here).
         */
        __kernel void rmsnorm_partial_sumsq(
            __global const float* x,
            __global float* partials,
            __local float* scratch,
            const int features)
        {
            const int gid = get_global_id(0);
            const int lid = get_local_id(0);
            const int groupId = get_group_id(0);
            const int localSize = get_local_size(0);

            float v = (gid < features) ? x[gid] : 0.0f;
            scratch[lid] = v * v;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    scratch[lid] += scratch[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                partials[groupId] = scratch[0];
            }
        }

        /*
         * rmsnorm_apply: phase 2 of 2 -- out = x * rms * gamma, where rms
         * was computed on the host from phase 1's partials. Exact port of
         * rms_norm_fast's final elementwise step.
         */
        __kernel void rmsnorm_apply(
            __global const float* x,
            __global const float* gamma,
            __global float* out,
            const float rms,
            const int features)
        {
            const int i = get_global_id(0);
            if (i >= features) return;
            out[i] = x[i] * rms * gamma[i];
        }

        /*
         * attn_scores: for one head h and all cache positions j in
         * [0, pos], score[j] = dot(Q_h, dequant(K_cache_j_h)) * rsqrt_d.
         * Takes the FULL q_all_heads buffer plus a qHeadOff offset rather
         * than a pre-sliced per-head buffer -- OpenCL 1.2 has no cheap way
         * to pass "buffer + offset" without a clCreateSubBuffer binding
         * this project doesn't otherwise need, so the kernel indexes the
         * offset itself instead (same pattern rope_apply_split already
         * uses via cosSinOffset).
         * K cache is stored FP32 here (dequantized once at cache-write
         * time on the host side of GpuLlamaLayer -- see its javadoc for
         * why the CPU reference's re-dequant-every-read approach was not
         * mirrored 1:1: it would mean K/V get dequantized once per query
         * position PER HEAD PER STEP on the GPU, which is needlessly
         * repeated work a device-resident FP32 cache avoids entirely).
         * One work-item per cache position j.
         */
        __kernel void attn_scores(
            __global const float* q_all_heads, // [numHeads * head_dim] -- full buffer, all heads
            __global const float* k_cache_f32, // [max_seq, kv_dim] dequantized
            __global float* scores,            // [pos+1]
            const int qHeadOff,
            const int head_dim,
            const int kv_dim,
            const int kv_head_off,
            const int posInclusive,
            const float rsqrt_d)
        {
            const int j = get_global_id(0);
            if (j >= posInclusive) return;

            const int kOff = j * kv_dim + kv_head_off;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_all_heads[qHeadOff + d] * k_cache_f32[kOff + d];
            }
            scores[j] = dot * rsqrt_d;
        }

        /*
         * softmax_inplace: single work-group, numerically stable 3-pass
         * softmax over `len` elements -- exact algorithm match to
         * softmax_row_f32 (max-subtract, exp+sum, normalize). Must be
         * launched with local size >= len is NOT required; this loops
         * internally so any local size works, but launching with ONE
         * work-group (global size == local size) is required since it
         * uses local-memory reduction for max and sum.
         */
        __kernel void softmax_inplace(
            __global float* scores,
            __local float* scratch,
            const int len)
        {
            const int lid = get_local_id(0);
            const int localSize = get_local_size(0);

            // Pass 1: max
            float localMax = -FLT_MAX;
            for (int i = lid; i < len; i += localSize) {
                localMax = fmax(localMax, scores[i]);
            }
            scratch[lid] = localMax;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float maxVal = scratch[0];
            barrier(CLK_LOCAL_MEM_FENCE);

            // Pass 2: exp(x - max), store back, accumulate sum
            float localSum = 0.0f;
            for (int i = lid; i < len; i += localSize) {
                float e = exp(scores[i] - maxVal);
                scores[i] = e;
                localSum += e;
            }
            scratch[lid] = localSum;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    scratch[lid] += scratch[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float sumVal = scratch[0];
            barrier(CLK_LOCAL_MEM_FENCE);

            // Pass 3: normalize
            float invSum = 1.0f / sumVal;
            for (int i = lid; i < len; i += localSize) {
                scores[i] *= invSum;
            }
        }

        /*
         * attn_weighted_sum: attn_out_all_heads[outHeadOff+d] =
         * sum_j scores[j] * V_cache[j,h,d]. One work-item per output
         * dimension d (head_dim work-items). Writes directly into the full
         * concatenated multi-head output buffer at outHeadOff, so no
         * host-side per-head gather step is needed before the O-projection
         * GEMV reads it. Same device-resident-FP32-V-cache rationale as
         * attn_scores.
         */
        __kernel void attn_weighted_sum(
            __global const float* scores,      // [pos+1]
            __global const float* v_cache_f32, // [max_seq, kv_dim] dequantized
            __global float* attn_out_all_heads,// [numHeads * head_dim] -- full buffer, all heads
            const int outHeadOff,
            const int head_dim,
            const int kv_dim,
            const int kv_head_off,
            const int posInclusive)
        {
            const int d = get_global_id(0);
            if (d >= head_dim) return;

            float acc = 0.0f;
            for (int j = 0; j < posInclusive; j++) {
                acc += scores[j] * v_cache_f32[j * kv_dim + kv_head_off + d];
            }
            attn_out_all_heads[outHeadOff + d] = acc;
        }

        /*
         * swiglu_activate: out[h] = gate[h] * sigmoid(gate[h]) * up[h].
         * Exact port of the SiLU formula in matmul_swiglu_q8, including its
         * -88 exp-argument clamp for numerical stability at very negative
         * gate values (exp(88) is close to FLT_MAX; anything more negative
         * would underflow sigmoid to exactly 0 anyway, so the clamp costs
         * nothing in practice and avoids a potential exp() overflow path
         * on some GPU math libraries for extreme inputs).
         */
        __kernel void swiglu_activate(
            __global const float* gate,
            __global const float* up,
            __global float* out,
            const int hidden)
        {
            const int h = get_global_id(0);
            if (h >= hidden) return;

            float g = gate[h];
            float sigmoid = 1.0f / (1.0f + exp(-fmax(g, -88.0f)));
            out[h] = g * sigmoid * up[h];
        }

        /*
         * residual_add: x[i] += y[i]. Used for both the attention and FFN
         * residual connections (steps 1c/2d in forward_layer_q8).
         */
        __kernel void residual_add(
            __global float* x,
            __global const float* y,
            const int len)
        {
            const int i = get_global_id(0);
            if (i >= len) return;
            x[i] += y[i];
        }

        /*
         * f32_gemv: plain FP32 GEMV, out[n] = sum_k a[k] * B[k*N+n].
         * Used for the O-projection (wo), which LlamaLayerInt8 keeps FP32
         * ("it's small") rather than quantizing -- exact port of
         * matmul_bias_f32's no-bias case.
         */
        __kernel void f32_gemv(
            __global const float* a,
            __global const float* B,
            __global float* out,
            const int K,
            const int N)
        {
            const int n = get_global_id(0);
            if (n >= N) return;

            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += a[k] * B[k * N + n];
            }
            out[n] = acc;
        }
        """;
}