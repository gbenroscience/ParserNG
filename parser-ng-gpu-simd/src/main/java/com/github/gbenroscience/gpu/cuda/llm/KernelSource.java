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

/**
 *
 * @author GBEMIRO 
 * CUDA C counterpart of {@code com.github.gbenroscience.gpu.llm.LlmKernelSource}
-- the same 13 Llama-decoder kernels, ported from OpenCL C to CUDA C for
NVRTC compilation, following the exact pattern already established by
KernelSource (interpreter kernels): one combined source string,
`extern "C" __global__` entry points so cuModuleGetFunction can find them
by their un-mangled names, compiled once via NvrtcBindings and loaded via
CudaBindings.cuModuleLoadData -- see LlmCudaContext.

SCOPE: identical to LlmKernelSource -- single-token decode path only
(M=1 GEMV per projection, not batched prefill), attention unfused
(three launches per forward pass), RMSNorm two-phase (device partial
sums, host reduction, device apply). See LlmKernelSource's javadoc for
the full rationale; it applies unchanged here, CUDA is just the backend.

STRUCTURAL DIFFERENCES FROM THE OpenCL VERSION, all forced by CUDA's
shape rather than chosen:

- OpenCL's `__local float* scratch` kernel PARAMETER (caller-sized local
  memory passed via clSetKernelArg with a NULL host pointer) has no
  direct CUDA equivalent as a parameter. CUDA's analogous mechanism is
  dynamic shared memory: an `extern __shared__ float NAME[];` declared
  at file scope right above the kernel that uses it, sized by the
  `sharedMemBytes` argument to cuLaunchKernel rather than by a kernel
  argument. Every kernel that took a `__local` scratch parameter in the
  OpenCL version therefore has ONE FEWER parameter here -- see each
  kernel's javadoc below for the exact signature change, and
  LlmCudaContext's dispatch helpers for the corresponding sharedMemBytes
  value each launch must pass.

- GGUF Q8_0's FP16 block scale is decoded/encoded with CUDA's built-in
  __half type (cuda_fp16.h) via __ushort_as_half/__half_as_ushort bit
  reinterpretation, rather than OpenCL's vload_half/vstore_half. Same
  IEEE-binary16 assumption, same result -- just the native CUDA
  equivalent of the same intrinsic.

- `get_global_id(0)` becomes `blockIdx.x * blockDim.x + threadIdx.x`;
  `get_local_id(0)`/`get_group_id(0)`/`get_local_size(0)` become
  `threadIdx.x`/`blockIdx.x`/`blockDim.x`; `barrier(CLK_LOCAL_MEM_FENCE)`
  becomes `__syncthreads()`. Purely mechanical renames, same semantics.

UNVERIFIED, same as the source this was ported from: no CUDA-capable
GPU or NVRTC toolchain available in the environment this was written
in. Ported carefully, kernel algorithm and control flow unchanged from
LlmKernelSource -- but treat as an untested first draft until run
against real per-layer activations.
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

    /** Block size used for the RMSNorm partial-sum kernel's local reduction. Must match host dispatch. */
    public static final int RMSNORM_WORKGROUP_SIZE = 256;

    /** Fixed block size for quantize_activation_q8_0_blocks -- one thread per element in a 32-wide Q8_0 group. */
    public static final int QUANTIZE_BLOCK_SIZE = 32;

    /** Default block size for the simple bound-checked 1D kernels (GEMVs, RoPE, elementwise ops). */
    public static final int DEFAULT_BLOCK_SIZE = 256;

    public static final String CUDA_SOURCE = """
        #include <cuda_fp16.h>

        #define Q8_0_GROUP_SIZE 32
        #define Q8_0_BLOCK_SIZE 34

        // NVRTC has no <cfloat>/<float.h> in its restricted device-code
        // standard library -- inline the IEEE-754 single-precision max
        // directly rather than depending on a header that may not resolve.
        #define DEVICE_FLT_MAX 3.402823466e+38f

        /*
         * Decodes a GGML fp16 scale (2 bytes, as stored in every Q8_0
         * block) to float via CUDA's native __half type -- bit-reinterpret
         * the two bytes as a __half, then widen. Matches the OpenCL
         * version's vload_half-based decode_fp16 exactly (both assume the
         * same IEEE binary16 layout GGML stores).
         */
        __device__ __forceinline__ float decode_fp16(unsigned char lo, unsigned char hi) {
            unsigned short bits = (unsigned short) lo | ((unsigned short) hi << 8);
            return __half2float(__ushort_as_half(bits));
        }

        /*
         * quantize_i8: FP32 -> INT8 with a single caller-supplied scale.
         * Direct port -- see LlmKernelSource's quantize_i8 javadoc for the
         * "not the format q8_0_gemv_split needs" caveat, which applies
         * unchanged here. NOT bounds-checked (matches the OpenCL original
         * exactly) -- the caller must launch EXACTLY `len` threads
         * (gridDim*blockDim == len), or use quantize_activation_q8_0_blocks
         * below instead, which is what this codebase's dispatch path
         * actually calls.
         */
        extern "C" __global__ void quantize_i8(
            const float* x,
            signed char* x_q8,
            const float invScale)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            float val = x[i] * invScale;
            int q = (int) rintf(val);
            q = max(-127, min(127, q));
            x_q8[i] = (signed char) q;
        }

        /*
         * quantize_activation_q8_0_blocks: FP32 -> GGUF Q8_0 block format,
         * fresh per-block absmax scale computed on-device. Same algorithm
         * as LlmKernelSource's OpenCL version; the `__local float* scratch`
         * parameter is gone -- this uses dynamic shared memory instead
         * (sh_quant below), sized via cuLaunchKernel's sharedMemBytes
         * argument (QUANTIZE_BLOCK_SIZE * sizeof(float) = 128 bytes).
         *
         * MUST be launched with blockDim.x = 32 (one thread per element in
         * a block) and gridDim.x = numBlocks (len / 32).
         */
        extern __shared__ float sh_quant[];
        extern "C" __global__ void quantize_activation_q8_0_blocks(
            const float* x,
            unsigned char* x_q8_0,
            const int len)
        {
            const int block = blockIdx.x;
            const int lane = threadIdx.x; // 0..31
            const int blockOff = block * Q8_0_GROUP_SIZE;
            const int outOff = block * Q8_0_BLOCK_SIZE;

            float v = x[blockOff + lane];
            sh_quant[lane] = fabsf(v);
            __syncthreads();

            for (int stride = 16; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    sh_quant[lane] = fmaxf(sh_quant[lane], sh_quant[lane + stride]);
                }
                __syncthreads();
            }
            float absmax = sh_quant[0];
            float scale = absmax / 127.0f;
            float invScale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            int q = (int) rintf(v * invScale);
            q = max(-127, min(127, q));
            x_q8_0[outOff + 2 + lane] = (unsigned char)(signed char) q;

            if (lane == 0) {
                unsigned short bits = __half_as_ushort(__float2half(scale));
                x_q8_0[outOff] = (unsigned char) (bits & 0xFF);
                x_q8_0[outOff + 1] = (unsigned char) ((bits >> 8) & 0xFF);
            }
        }

        /*
         * q8_0_gemv_split: GGML Q8_0 block-format GEMV, split even/odd
         * output layout -- exact port of the OpenCL version's per-block
         * algorithm (which itself ports KernelsInt8.matmul_q8_0_1xN_split_opt).
         * See LlmKernelSource's javadoc for the split-layout rationale
         * (RoPE optimization for Q/K). Unchanged from OpenCL besides the
         * index-computation intrinsics.
         */
        extern "C" __global__ void q8_0_gemv_split(
            const unsigned char* x_q8_0,
            const unsigned char* W_q8_0,
            float* out_f32_split,
            const int qHeads,
            const int head_dim,
            const int K)
        {
            const int halfDim = head_dim >> 1;
            const int gid = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. qHeads*halfDim - 1
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
                    signed char xv = (signed char) x_q8_0[xBlockOff + 2 + j];
                    signed char w0v = (signed char) W_q8_0[w0BlockOff + 2 + j];
                    signed char w1v = (signed char) W_q8_0[w1BlockOff + 2 + j];
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
         * q8_0_gemv_plain: same Q8_0 GEMV algorithm, natural contiguous
         * output order -- used for the FFN gate/up/down projections
         * (RoPE's even/odd pairing doesn't apply there). One thread per
         * output column n.
         */
        extern "C" __global__ void q8_0_gemv_plain(
            const unsigned char* x_q8_0,
            const unsigned char* W_q8_0,
            float* out_f32,
            const int N,
            const int K)
        {
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
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
                    signed char xv = (signed char) x_q8_0[xBlockOff + 2 + j];
                    signed char wv = (signed char) W_q8_0[wBlockOff + 2 + j];
                    iacc += (int) xv * (int) wv;
                }
                acc += iacc * scale;
            }
            out_f32[n] = acc;
        }

        /*
         * rope_apply_split: pairwise rotation on split-layout Q/K buffers.
         *   x0' = x0*cos - x1*sin
         *   x1' = x0*sin + x1*cos
         * One thread per (head, half-dim-index) pair -- run once for Q,
         * once for K, same kernel, different buffer/headCount args.
         */
        extern "C" __global__ void rope_apply_split(
            float* buf,
            const float* cos_table,
            const float* sin_table,
            const int heads,
            const int head_dim,
            const int cosSinOffset)
        {
            const int halfDim = head_dim >> 1;
            const int gid = blockIdx.x * blockDim.x + threadIdx.x;
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
         * per block via shared-memory tree reduction. Host sums the
         * (small) partials array and computes rms = 1/sqrt(mean+eps)
         * before calling rmsnorm_apply. The `__local float* scratch`
         * parameter is gone -- replaced by dynamic shared memory
         * (sh_rms), sized via sharedMemBytes = blockDim.x * sizeof(float)
         * (RMSNORM_WORKGROUP_SIZE * 4 for the standard dispatch).
         */
        extern __shared__ float sh_rms[];
        extern "C" __global__ void rmsnorm_partial_sumsq(
            const float* x,
            float* partials,
            const int features)
        {
            const int gid = blockIdx.x * blockDim.x + threadIdx.x;
            const int lid = threadIdx.x;
            const int groupId = blockIdx.x;
            const int localSize = blockDim.x;

            float v = (gid < features) ? x[gid] : 0.0f;
            sh_rms[lid] = v * v;
            __syncthreads();

            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_rms[lid] += sh_rms[lid + stride];
                }
                __syncthreads();
            }

            if (lid == 0) {
                partials[groupId] = sh_rms[0];
            }
        }

        /*
         * rmsnorm_apply: phase 2 of 2 -- out = x * rms * gamma, rms
         * computed on the host from phase 1's partials.
         */
        extern "C" __global__ void rmsnorm_apply(
            const float* x,
            const float* gamma,
            float* out,
            const float rms,
            const int features)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= features) return;
            out[i] = x[i] * rms * gamma[i];
        }

        /*
         * attn_scores: for one head h and all cache positions j in
         * [0, pos], score[j] = dot(Q_h, K_cache_j_h) * rsqrt_d. Takes the
         * FULL q_all_heads buffer plus a qHeadOff offset (same pattern
         * rope_apply_split uses via cosSinOffset) rather than a pre-sliced
         * per-head pointer, so no extra device-pointer-arithmetic launch
         * variant is needed per head.
         * K cache is FP32 here (dequantized once at cache-write time on
         * the host side -- see LlmCudaLlamaLayer's class javadoc for why
         * the CPU/OpenCL re-dequant-every-read approach isn't mirrored).
         * One thread per cache position j.
         */
        extern "C" __global__ void attn_scores(
            const float* q_all_heads, // [numHeads * head_dim] -- full buffer, all heads
            const float* k_cache_f32, // [max_seq, kv_dim] dequantized
            float* scores,            // [pos+1]
            const int qHeadOff,
            const int head_dim,
            const int kv_dim,
            const int kv_head_off,
            const int posInclusive,
            const float rsqrt_d)
        {
            const int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (j >= posInclusive) return;

            const int kOff = j * kv_dim + kv_head_off;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_all_heads[qHeadOff + d] * k_cache_f32[kOff + d];
            }
            scores[j] = dot * rsqrt_d;
        }

        /*
         * softmax_inplace: single-block, numerically stable 3-pass softmax
         * over `len` elements (max-subtract, exp+sum, normalize). MUST be
         * launched as exactly ONE block (gridDim.x == 1) since it uses
         * shared-memory reduction for both max and sum; blockDim.x can be
         * any size (loops internally), sharedMemBytes must be
         * blockDim.x * sizeof(float). The `__local float* scratch`
         * parameter is gone -- replaced by dynamic shared memory
         * (sh_softmax) for the same reason as the other reduction kernels.
         */
        extern __shared__ float sh_softmax[];
        extern "C" __global__ void softmax_inplace(
            float* scores,
            const int len)
        {
            const int lid = threadIdx.x;
            const int localSize = blockDim.x;

            // Pass 1: max
            float localMax = -DEVICE_FLT_MAX;
            for (int i = lid; i < len; i += localSize) {
                localMax = fmaxf(localMax, scores[i]);
            }
            sh_softmax[lid] = localMax;
            __syncthreads();
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_softmax[lid] = fmaxf(sh_softmax[lid], sh_softmax[lid + stride]);
                }
                __syncthreads();
            }
            float maxVal = sh_softmax[0];
            __syncthreads();

            // Pass 2: exp(x - max), store back, accumulate sum
            float localSum = 0.0f;
            for (int i = lid; i < len; i += localSize) {
                float e = expf(scores[i] - maxVal);
                scores[i] = e;
                localSum += e;
            }
            sh_softmax[lid] = localSum;
            __syncthreads();
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_softmax[lid] += sh_softmax[lid + stride];
                }
                __syncthreads();
            }
            float sumVal = sh_softmax[0];
            __syncthreads();

            // Pass 3: normalize
            float invSum = 1.0f / sumVal;
            for (int i = lid; i < len; i += localSize) {
                scores[i] *= invSum;
            }
        }

        /*
         * attn_weighted_sum: attn_out_all_heads[outHeadOff+d] =
         * sum_j scores[j] * V_cache[j,h,d]. One thread per output
         * dimension d. Writes directly into the full concatenated
         * multi-head output buffer at outHeadOff, same as the OpenCL
         * version, so no host-side per-head gather is needed before the
         * O-projection GEMV reads it.
         */
        extern "C" __global__ void attn_weighted_sum(
            const float* scores,      // [pos+1]
            const float* v_cache_f32, // [max_seq, kv_dim] dequantized
            float* attn_out_all_heads,// [numHeads * head_dim] -- full buffer, all heads
            const int outHeadOff,
            const int head_dim,
            const int kv_dim,
            const int kv_head_off,
            const int posInclusive)
        {
            const int d = blockIdx.x * blockDim.x + threadIdx.x;
            if (d >= head_dim) return;

            float acc = 0.0f;
            for (int j = 0; j < posInclusive; j++) {
                acc += scores[j] * v_cache_f32[j * kv_dim + kv_head_off + d];
            }
            attn_out_all_heads[outHeadOff + d] = acc;
        }

        /*
         * swiglu_activate: out[h] = gate[h] * sigmoid(gate[h]) * up[h].
         * Includes the same -88 exp-argument clamp as the OpenCL version
         * for numerical stability at very negative gate values.
         */
        extern "C" __global__ void swiglu_activate(
            const float* gate,
            const float* up,
            float* out,
            const int hidden)
        {
            const int h = blockIdx.x * blockDim.x + threadIdx.x;
            if (h >= hidden) return;

            float g = gate[h];
            float sigmoid = 1.0f / (1.0f + expf(-fmaxf(g, -88.0f)));
            out[h] = g * sigmoid * up[h];
        }

        /*
         * residual_add: x[i] += y[i]. Used for both the attention and FFN
         * residual connections.
         */
        extern "C" __global__ void residual_add(
            float* x,
            const float* y,
            const int len)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= len) return;
            x[i] += y[i];
        }

        /*
         * f32_gemv: plain FP32 GEMV, out[n] = sum_k a[k] * B[k*N+n]. Used
         * for the O-projection (wo), kept FP32 rather than quantized (same
         * choice the CPU/OpenCL reference makes -- "it's small").
         */
        extern "C" __global__ void f32_gemv(
            const float* a,
            const float* B,
            float* out,
            const int K,
            const int N)
        {
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            if (n >= N) return;

            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += a[k] * B[k * N + n];
            }
            out[n] = acc;
        }
        """;
}