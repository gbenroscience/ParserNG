package com.github.gbenroscience.gpu.llm.opencl;

/**
 * OpenCL C kernel source for the Llama-style decoder -- OpenCL counterpart
 * of {@code com.github.gbenroscience.gpu.llm.cuda.KernelSource}. Same 23
 * kernels, same algorithms, translated line-for-line where the two
 * languages agree and re-derived where they don't. Read this class's
 * javadoc alongside the CUDA original's -- the algorithmic commentary
 * there (what "batched" buys you, what this is NOT: not cuBLAS/clBLAS-
 * tier tiling, attention still three unfused launches per head) applies
 * unchanged here and is not repeated per kernel.
 *
 * TRANSLATION NOTES (CUDA -> OpenCL C):
 *   - {@code __global__ void} -> {@code __kernel void}; {@code extern "C"} dropped
 *     (OpenCL C has no name mangling to begin with).
 *   - Buffer parameters (raw pointers in CUDA) get an explicit
 *     {@code __global} address-space qualifier -- OpenCL requires this,
 *     CUDA does not.
 *   - {@code blockIdx.x}/{@code blockIdx.y} -> {@code get_group_id(0)}/{@code get_group_id(1)};
 *     {@code threadIdx.x} -> {@code get_local_id(0)}; {@code blockDim.x} -> {@code get_local_size(0)}.
 *     Kept as this literal group/local-id mapping (rather than collapsing
 *     simple 1D kernels to {@code get_global_id(0)}, which would be
 *     numerically equivalent) so every kernel body stays structurally
 *     diffable against its CUDA counterpart.
 *   - {@code __shared__}/{@code extern __shared__} -> {@code __local}. CUDA's dynamic
 *     shared-memory-size-at-launch mechanism (the 5th cuLaunchKernel
 *     argument) has NO direct OpenCL equivalent for a plain {@code __kernel}
 *     parameter list without also threading a {@code __local} pointer
 *     parameter through every call site. Sidestepped here by declaring
 *     each shared buffer as a FIXED-SIZE {@code __local} array sized to the
 *     largest work-group this codebase ever dispatches for that kernel
 *     (32 for the Q8_0 quantize block, 256 for every RMSNorm/softmax
 *     reduction -- see QUANTIZE_BLOCK_SIZE/RMSNORM_WORKGROUP_SIZE/
 *     DEFAULT_BLOCK_SIZE below) -- a reduction only ever touches indices
 *     {@code [0, get_local_size(0))}, so a fixed array that's >= the actual
 *     work-group size is correct regardless of how few of its slots a
 *     smaller launch (e.g. softmax over a short prefix) actually uses.
 *     This also means the host-side dispatch code needs no dynamic
 *     local-memory clSetKernelArg call at all -- one fewer moving part
 *     than the CUDA port's sharedMemBytes plumbing.
 *   - {@code __syncthreads()} -> {@code barrier(CLK_LOCAL_MEM_FENCE)}.
 *   - {@code expf/fmaxf/fabsf/rintf/erff} -> OpenCL's type-overloaded builtins
 *     {@code exp/fmax/fabs/rint/erf} (no f-suffix; OpenCL resolves by argument type).
 *   - {@code __half2float(__ushort_as_half(...))} / {@code __half_as_ushort(__float2half(...))}:
 *     CUDA gets these from {@code <cuda_fp16.h>}. OpenCL's core spec has no
 *     required equivalent that works without the optional cl_khr_fp16
 *     extension (not guaranteed present on every device -- some older/
 *     embedded-profile GPUs lack it). Reimplemented here as manual
 *     IEEE-754 binary16<->binary32 bit conversion (decode_fp16 /
 *     encode_fp16_bits below) so Q8_0's per-block fp16 scale factor
 *     works on every OpenCL 1.2+ device unconditionally. encode_fp16_bits
 *     rounds by truncation+round-bit (round-half-up, not strict
 *     round-to-nearest-even) -- immaterial here since it only encodes a
 *     quantization scale factor, not a value whose bit-exactness matters.
 *   - No pointer arithmetic anywhere in this file takes a {@code __global}
 *     buffer parameter and offsets its BASE address from the host side
 *     (impossible in OpenCL -- cl_mem is an opaque handle, not an
 *     address). Every kernel that needs a sub-region already receives an
 *     explicit int offset parameter (qHeadOff, kvHeadOff, outHeadOff,
 *     etc.) and indexes with it internally -- this was already true of
 *     every kernel in the CUDA original, so no kernel signature changed
 *     shape to accommodate the port. The two places the CUDA HOST code
 *     itself did raw pointer-plus-byte-offset arithmetic (extracting one
 *     row out of a [T,dim] prefill batch buffer, and writing the KV cache
 *     at a growing per-token offset) are handled in GpuContext/LlamaLayer
 *     via clEnqueueCopyBuffer's byte-offset parameters instead -- see
 *     LlamaLayer's class javadoc.
 */
public final class KernelSource {

    private KernelSource() {
    }

    // ---- decode-path kernels (unchanged names from the CUDA original) ----
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

    // ---- FFN activation alternatives ----
    public static final String KERNEL_GELU_ACTIVATE = "gelu_activate";
    public static final String KERNEL_GEGLU_ACTIVATE = "geglu_activate";

    // ---- batched prefill kernels ----
    public static final String KERNEL_Q8_0_GEMM_TILED = "q8_0_gemm_tiled";
    public static final String KERNEL_F32_GEMM_TILED = "f32_gemm_tiled";
    public static final String KERNEL_RMSNORM_PARTIAL_SUMSQ_ROWS = "rmsnorm_partial_sumsq_rows";
    public static final String KERNEL_RMSNORM_APPLY_ROWS = "rmsnorm_apply_rows";
    public static final String KERNEL_ROPE_APPLY_PAIRWISE_ROWS = "rope_apply_pairwise_rows";
    public static final String KERNEL_ATTN_SCORES_CAUSAL_BATCHED = "attn_scores_causal_batched";
    public static final String KERNEL_SOFTMAX_INPLACE_ROWS = "softmax_inplace_rows";
    public static final String KERNEL_ATTN_WEIGHTED_SUM_CAUSAL_BATCHED = "attn_weighted_sum_causal_batched";

    /** Work-group size used by both RMSNorm partial-sum kernels' local reduction, AND the fixed __local array size those kernels declare. Must match host dispatch. */
    public static final int RMSNORM_WORKGROUP_SIZE = 256;

    /** Fixed work-group size for quantize_activation_q8_0_blocks -- one work-item per element in a 32-wide Q8_0 group. Also the fixed __local array size that kernel declares. */
    public static final int QUANTIZE_BLOCK_SIZE = 32;

    /** Default work-group size for the simple bound-checked 1D kernels (GEMVs, RoPE, elementwise ops); also the upper cap nextPow2() uses for softmax's local reduction size, and therefore the __local array size softmax_inplace/softmax_inplace_rows declare. */
    public static final int DEFAULT_BLOCK_SIZE = 256;

    /** Default column-tile work-group size for the batched GEMM kernels (local_work_size[0]; get_group_id(1) iterates rows). */
    public static final int GEMM_TILE_N = 128;

    public static final String CL_SOURCE = """
        #define Q8_0_GROUP_SIZE 32
        #define Q8_0_BLOCK_SIZE 34
        #define DEVICE_FLT_MAX 3.402823466e+38f

        // ---- manual IEEE-754 binary16 <-> binary32 conversion (see class javadoc) ----

        inline float decode_fp16(uchar lo, uchar hi) {
            uint bits = (uint) lo | ((uint) hi << 8);
            uint sign = (bits >> 15) & 0x1u;
            uint exp = (bits >> 10) & 0x1Fu;
            uint mant = bits & 0x3FFu;
            uint f;
            if (exp == 0u) {
                if (mant == 0u) {
                    f = sign << 31;
                } else {
                    uint e = 1u;
                    while ((mant & 0x400u) == 0u) {
                        mant <<= 1;
                        e--;
                    }
                    mant &= 0x3FFu;
                    uint fexp = e - 15u + 127u;
                    f = (sign << 31) | (fexp << 23) | (mant << 13);
                }
            } else if (exp == 0x1Fu) {
                f = (sign << 31) | (0xFFu << 23) | (mant << 13);
            } else {
                uint fexp = exp - 15u + 127u;
                f = (sign << 31) | (fexp << 23) | (mant << 13);
            }
            return as_float(f);
        }

        /** Round-half-up float->half bit pattern -- see class javadoc for why exact ties-to-even isn't needed here. */
        inline ushort encode_fp16_bits(float value) {
            uint x = as_uint(value);
            uint sign = (x >> 16) & 0x8000u;
            uint mantissa = x & 0x007FFFFFu;
            int exponent = (int) ((x >> 23) & 0xFFu) - 127;

            if (((x >> 23) & 0xFFu) == 0xFFu) {
                return (ushort) (sign | 0x7C00u | (mantissa != 0u ? 0x200u : 0u));
            }
            if (exponent > 15) {
                return (ushort) (sign | 0x7C00u);
            }
            if (exponent < -14) {
                if (exponent < -24) {
                    return (ushort) sign;
                }
                mantissa |= 0x00800000u;
                int shift = -14 - exponent; // 1..10
                uint halfMant = mantissa >> (13 + shift);
                uint roundBit = (mantissa >> (12 + shift)) & 1u;
                halfMant += roundBit;
                return (ushort) (sign | halfMant);
            }
            uint halfExp = (uint) (exponent + 15);
            uint halfMant = mantissa >> 13;
            uint roundBit = (mantissa >> 12) & 1u;
            uint rounded = halfMant + roundBit;
            if (rounded == 0x400u) {
                rounded = 0u;
                halfExp += 1u;
            }
            return (ushort) (sign | (halfExp << 10) | rounded);
        }

        inline float gpu_sigmoid_f(float x) {
            return 1.0f / (1.0f + exp(-x));
        }

        // Exact-erf GeLU -- see CUDA original's javadoc: matches the
        // textbook formula, NOT necessarily a specific model's training-
        // time tanh-approximation. Diff against reference activations
        // before trusting GeLU-FFN outputs.
        inline float gpu_gelu_f(float x) {
            return 0.5f * x * (1.0f + erf(x * 0.70710678f));
        }

        // =====================================================================
        // ===================== DECODE-PATH KERNELS ==========================
        // =====================================================================

        __kernel void quantize_i8(
            __global const float* x,
            __global char* x_q8,
            const float invScale)
        {
            const int i = get_group_id(0) * get_local_size(0) + get_local_id(0);
            float val = x[i] * invScale;
            int q = (int) rint(val);
            q = max(-127, min(127, q));
            x_q8[i] = (char) q;
        }

        __kernel void quantize_activation_q8_0_blocks(
            __global const float* x,
            __global uchar* x_q8_0,
            const int len)
        {
            __local float sh_quant[32];
            const int block = get_group_id(0);
            const int lane = get_local_id(0); // 0..31
            const int blockOff = block * Q8_0_GROUP_SIZE;
            const int outOff = block * Q8_0_BLOCK_SIZE;

            float v = x[blockOff + lane];
            sh_quant[lane] = fabs(v);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int stride = 16; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    sh_quant[lane] = fmax(sh_quant[lane], sh_quant[lane + stride]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float absmax = sh_quant[0];
            float scale = absmax / 127.0f;
            float invScale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            int q = (int) rint(v * invScale);
            q = max(-127, min(127, q));
            x_q8_0[outOff + 2 + lane] = (uchar) (char) q;

            if (lane == 0) {
                ushort bits = encode_fp16_bits(scale);
                x_q8_0[outOff] = (uchar) (bits & 0xFF);
                x_q8_0[outOff + 1] = (uchar) ((bits >> 8) & 0xFF);
            }
        }

        __kernel void q8_0_gemv_split(
            __global const uchar* x_q8_0,
            __global const uchar* W_q8_0,
            __global float* out_f32_split,
            const int qHeads,
            const int head_dim,
            const int K)
        {
            const int halfDim = head_dim >> 1;
            const int gid = get_group_id(0) * get_local_size(0) + get_local_id(0);
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

        __kernel void q8_0_gemv_plain(
            __global const uchar* x_q8_0,
            __global const uchar* W_q8_0,
            __global float* out_f32,
            const int N,
            const int K)
        {
            const int n = get_group_id(0) * get_local_size(0) + get_local_id(0);
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

        __kernel void rope_apply_split(
            __global float* buf,
            __global const float* cos_table,
            __global const float* sin_table,
            const int heads,
            const int head_dim,
            const int cosSinOffset)
        {
            const int halfDim = head_dim >> 1;
            const int gid = get_group_id(0) * get_local_size(0) + get_local_id(0);
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

        __kernel void rmsnorm_partial_sumsq(
            __global const float* x,
            __global float* partials,
            const int features)
        {
            __local float sh_rms[256];
            const int gid = get_group_id(0) * get_local_size(0) + get_local_id(0);
            const int lid = get_local_id(0);
            const int groupId = get_group_id(0);
            const int localSize = get_local_size(0);

            float v = (gid < features) ? x[gid] : 0.0f;
            sh_rms[lid] = v * v;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_rms[lid] += sh_rms[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                partials[groupId] = sh_rms[0];
            }
        }

        __kernel void rmsnorm_apply(
            __global const float* x,
            __global const float* gamma,
            __global float* out,
            const float rms,
            const int features)
        {
            const int i = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (i >= features) return;
            out[i] = x[i] * rms * gamma[i];
        }

        __kernel void attn_scores(
            __global const float* q_all_heads,
            __global const float* k_cache_f32,
            __global float* scores,
            const int qHeadOff,
            const int head_dim,
            const int kv_dim,
            const int kv_head_off,
            const int posInclusive,
            const float rsqrt_d)
        {
            const int j = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (j >= posInclusive) return;

            const int kOff = j * kv_dim + kv_head_off;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_all_heads[qHeadOff + d] * k_cache_f32[kOff + d];
            }
            scores[j] = dot * rsqrt_d;
        }

        __kernel void softmax_inplace(
            __global float* scores,
            const int len)
        {
            __local float sh_softmax[256];
            const int lid = get_local_id(0);
            const int localSize = get_local_size(0);

            float localMax = -DEVICE_FLT_MAX;
            for (int i = lid; i < len; i += localSize) {
                localMax = fmax(localMax, scores[i]);
            }
            sh_softmax[lid] = localMax;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_softmax[lid] = fmax(sh_softmax[lid], sh_softmax[lid + stride]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float maxVal = sh_softmax[0];
            barrier(CLK_LOCAL_MEM_FENCE);

            float localSum = 0.0f;
            for (int i = lid; i < len; i += localSize) {
                float e = exp(scores[i] - maxVal);
                scores[i] = e;
                localSum += e;
            }
            sh_softmax[lid] = localSum;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_softmax[lid] += sh_softmax[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float sumVal = sh_softmax[0];
            barrier(CLK_LOCAL_MEM_FENCE);

            float invSum = 1.0f / sumVal;
            for (int i = lid; i < len; i += localSize) {
                scores[i] *= invSum;
            }
        }

        __kernel void attn_weighted_sum(
            __global const float* scores,
            __global const float* v_cache_f32,
            __global float* attn_out_all_heads,
            const int outHeadOff,
            const int head_dim,
            const int kv_dim,
            const int kv_head_off,
            const int posInclusive)
        {
            const int d = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (d >= head_dim) return;

            float acc = 0.0f;
            for (int j = 0; j < posInclusive; j++) {
                acc += scores[j] * v_cache_f32[j * kv_dim + kv_head_off + d];
            }
            attn_out_all_heads[outHeadOff + d] = acc;
        }

        __kernel void swiglu_activate(
            __global const float* gate,
            __global const float* up,
            __global float* out,
            const int hidden)
        {
            const int h = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (h >= hidden) return;

            float g = gate[h];
            float sigmoid = 1.0f / (1.0f + exp(-fmax(g, -88.0f)));
            out[h] = g * sigmoid * up[h];
        }

        __kernel void residual_add(
            __global float* x,
            __global const float* y,
            const int len)
        {
            const int i = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (i >= len) return;
            x[i] += y[i];
        }

        __kernel void f32_gemv(
            __global const float* a,
            __global const float* B,
            __global float* out,
            const int K,
            const int N)
        {
            const int n = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (n >= N) return;

            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += a[k] * B[k * N + n];
            }
            out[n] = acc;
        }

        // =====================================================================
        // ===================== FFN ACTIVATION ALTERNATIVES =================
        // =====================================================================

        __kernel void gelu_activate(
            __global const float* gate,
            __global float* out,
            const int hidden)
        {
            const int h = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (h >= hidden) return;
            out[h] = gpu_gelu_f(gate[h]);
        }

        __kernel void geglu_activate(
            __global const float* gate,
            __global const float* up,
            __global float* out,
            const int hidden)
        {
            const int h = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (h >= hidden) return;
            out[h] = gpu_gelu_f(gate[h]) * up[h];
        }

        // =====================================================================
        // ===================== BATCHED PREFILL KERNELS =====================
        // =====================================================================

        __kernel void q8_0_gemm_tiled(
            __global const uchar* X_q8_0, // [T, numBlocks*34]
            __global const uchar* W_q8_0, // [N, numBlocks*34]
            __global float* out,          // [T, N]
            const int T,
            const int N,
            const int K)
        {
            const int t = get_group_id(1);
            const int n = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (t >= T || n >= N) return;

            const int numBlocks = K / Q8_0_GROUP_SIZE;
            const int rowStride = numBlocks * Q8_0_BLOCK_SIZE;
            const int xRowOff = t * rowStride;
            const int wRowOff = n * rowStride;

            float acc = 0.0f;
            for (int b = 0; b < numBlocks; b++) {
                const int xBlockOff = xRowOff + b * Q8_0_BLOCK_SIZE;
                const int wBlockOff = wRowOff + b * Q8_0_BLOCK_SIZE;

                float xScale = decode_fp16(X_q8_0[xBlockOff], X_q8_0[xBlockOff + 1]);
                float wScale = decode_fp16(W_q8_0[wBlockOff], W_q8_0[wBlockOff + 1]);
                float scale = xScale * wScale;

                int iacc = 0;
                for (int j = 0; j < Q8_0_GROUP_SIZE; j++) {
                    char xv = (char) X_q8_0[xBlockOff + 2 + j];
                    char wv = (char) W_q8_0[wBlockOff + 2 + j];
                    iacc += (int) xv * (int) wv;
                }
                acc += iacc * scale;
            }
            out[t * N + n] = acc;
        }

        __kernel void f32_gemm_tiled(
            __global const float* A, // [T, K]
            __global const float* B, // [K, N]
            __global float* out,     // [T, N]
            const int T,
            const int K,
            const int N)
        {
            const int t = get_group_id(1);
            const int n = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (t >= T || n >= N) return;

            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[t * K + k] * B[k * N + n];
            }
            out[t * N + n] = acc;
        }

        __kernel void rmsnorm_partial_sumsq_rows(
            __global const float* x,       // [T, features]
            __global float* partials,      // [T, numGroups]
            const int features,
            const int numGroups)
        {
            __local float sh_rms_rows[256];
            const int row = get_group_id(1);
            const int groupId = get_group_id(0);
            const int lid = get_local_id(0);
            const int localSize = get_local_size(0);
            const int idx = groupId * localSize + lid;

            float v = (idx < features) ? x[row * features + idx] : 0.0f;
            sh_rms_rows[lid] = v * v;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_rms_rows[lid] += sh_rms_rows[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                partials[row * numGroups + groupId] = sh_rms_rows[0];
            }
        }

        __kernel void rmsnorm_apply_rows(
            __global const float* x,      // [T, features]
            __global const float* gamma,  // [features]
            __global float* out,          // [T, features]
            __global const float* rms,    // [T]
            const int features)
        {
            const int row = get_group_id(1);
            const int i = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (i >= features) return;
            out[row * features + i] = x[row * features + i] * rms[row] * gamma[i];
        }

        __kernel void rope_apply_pairwise_rows(
            __global float* buf,             // [T, heads*head_dim], PLAIN contiguous layout
            __global const float* cos_table, // [max_seq, halfDim]
            __global const float* sin_table,
            const int heads,
            const int head_dim,
            __global const int* positions)   // [T] absolute sequence position per row
        {
            const int halfDim = head_dim >> 1;
            const int row = get_group_id(1);
            const int gid = get_group_id(0) * get_local_size(0) + get_local_id(0);
            const int h = gid / halfDim;
            const int i = gid % halfDim;
            if (h >= heads) return;

            const int pos = positions[row];
            const int rowOff = row * heads * head_dim;
            const int pairOff = rowOff + h * head_dim + (i << 1);

            float c = cos_table[pos * halfDim + i];
            float s = sin_table[pos * halfDim + i];
            float x0 = buf[pairOff];
            float x1 = buf[pairOff + 1];

            buf[pairOff] = x0 * c - x1 * s;
            buf[pairOff + 1] = x0 * s + x1 * c;
        }

        __kernel void attn_scores_causal_batched(
            __global const float* q_all,   // [T, qRowStride]
            __global const float* k_all,   // [T, kRowStride]
            __global float* scores,        // [T, T]
            const int qRowStride,
            const int kRowStride,
            const int qHeadOff,
            const int kHeadOff,
            const int head_dim,
            const int T,
            const float rsqrt_d)
        {
            const int t = get_group_id(1);
            const int j = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (j >= T) return;

            if (j > t) {
                scores[t * T + j] = -DEVICE_FLT_MAX;
                return;
            }

            const int qOff = t * qRowStride + qHeadOff;
            const int kOff = j * kRowStride + kHeadOff;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_all[qOff + d] * k_all[kOff + d];
            }
            scores[t * T + j] = dot * rsqrt_d;
        }

        __kernel void softmax_inplace_rows(
            __global float* scores, // [T, T]
            const int T)
        {
            __local float sh_softmax_rows[256];
            const int row = get_group_id(0);
            const int lid = get_local_id(0);
            const int localSize = get_local_size(0);
            __global float* rowPtr = scores + row * T;

            float localMax = -DEVICE_FLT_MAX;
            for (int i = lid; i < T; i += localSize) {
                localMax = fmax(localMax, rowPtr[i]);
            }
            sh_softmax_rows[lid] = localMax;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_softmax_rows[lid] = fmax(sh_softmax_rows[lid], sh_softmax_rows[lid + stride]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float maxVal = sh_softmax_rows[0];
            barrier(CLK_LOCAL_MEM_FENCE);

            float localSum = 0.0f;
            for (int i = lid; i < T; i += localSize) {
                float e = exp(rowPtr[i] - maxVal);
                rowPtr[i] = e;
                localSum += e;
            }
            sh_softmax_rows[lid] = localSum;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_softmax_rows[lid] += sh_softmax_rows[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float sumVal = sh_softmax_rows[0];
            barrier(CLK_LOCAL_MEM_FENCE);

            float invSum = 1.0f / sumVal;
            for (int i = lid; i < T; i += localSize) {
                rowPtr[i] *= invSum;
            }
        }

        __kernel void attn_weighted_sum_causal_batched(
            __global const float* scores,  // [T, T]
            __global const float* v_all,   // [T, vRowStride]
            __global float* attn_out,      // [T, outRowStride]
            const int vRowStride,
            const int outRowStride,
            const int vHeadOff,
            const int outHeadOff,
            const int head_dim,
            const int T)
        {
            const int t = get_group_id(1);
            const int d = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (d >= head_dim) return;

            __global const float* rowScores = scores + t * T;
            float acc = 0.0f;
            for (int j = 0; j <= t; j++) {
                acc += rowScores[j] * v_all[j * vRowStride + vHeadOff + d];
            }
            attn_out[t * outRowStride + outHeadOff + d] = acc;
        }
        """;
}