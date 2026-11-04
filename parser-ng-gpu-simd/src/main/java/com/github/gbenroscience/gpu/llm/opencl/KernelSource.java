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
    public static final String KERNEL_ROPE_APPLY_PAIRS = "rope_apply_pairs";
    public static final String KERNEL_RMSNORM_PARTIAL_SUMSQ = "rmsnorm_partial_sumsq";
    public static final String KERNEL_RMSNORM_APPLY = "rmsnorm_apply";
    public static final String KERNEL_ATTN_SCORES = "attn_scores";
    public static final String KERNEL_SOFTMAX_INPLACE = "softmax_inplace";
    public static final String KERNEL_ATTN_WEIGHTED_SUM = "attn_weighted_sum";
    public static final String KERNEL_ATTN_DECODE_FUSED = "attn_decode_fused";
    public static final String KERNEL_SWIGLU_ACTIVATE = "swiglu_activate";
    public static final String KERNEL_RESIDUAL_ADD = "residual_add";
    public static final String KERNEL_F32_GEMV = "f32_gemv";
    public static final String KERNEL_Q8_0_GEMV_PLAIN_LEGACY = "q8_0_gemv_plain_legacy";
    public static final String KERNEL_F32_GEMV_LEGACY = "f32_gemv_legacy";
    public static final String KERNEL_RMSNORM_FUSED = "rmsnorm_fused";
    public static final String KERNEL_RMSNORM_FUSED_ROWS = "rmsnorm_fused_rows";
    public static final String KERNEL_ROPE_ROWS_SEQ = "rope_apply_pairwise_rows_seq";
    public static final String KERNEL_Q8_0_GEMV_VLOADF = "q8_0_gemv_vloadf";
    public static final String KERNEL_Q8_0_GEMV_U32 = "q8_0_gemv_u32";
    public static final String KERNEL_Q8_0_GEMM_ROWS = "q8_0_gemm_rows";
    public static final String KERNEL_F32_GEMM_ROWS = "f32_gemm_rows";

    /** Tokens processed per work-group pass by the *_gemm_rows kernels (must equal GEMM_TCH in the kernel source). */
    public static final int GEMM_TOKEN_CHUNK = 16;
    public static final String KERNEL_BW_U8 = "bw_read_u8";
    public static final String KERNEL_BW_U32 = "bw_read_u32";
    public static final String KERNEL_BW_U128 = "bw_read_u128";
    public static final String KERNEL_BW_V16_OFF2 = "bw_read_v16_off2";

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

    /** Work-group size of the GEMV kernels (one work-group per output row). Must equal GEMV_WG in the kernel source. */
    public static final int GEMV_WORKGROUP = 64;

    /** Largest head_dim handled by attn_decode_fused (must equal ATTN_MAX_HEAD_DIM in the kernel source). */
    public static final int ATTN_FUSED_MAX_HEAD_DIM = 256;

    /** Default column-tile work-group size for the batched GEMM kernels (local_work_size[0]; get_group_id(1) iterates rows). */
    public static final int GEMM_TILE_N = 128;

    public static final String CL_SOURCE = """
        #define Q8_0_GROUP_SIZE 32
        #define Q8_0_BLOCK_SIZE 34
        #define GEMV_WG 64

        // exact int8 x int8 dot of 16 lanes
        inline int dot_i8x16(char16 a, char16 b) {
            int16 p = convert_int16(a) * convert_int16(b);
            int8 q = p.lo + p.hi;
            int4 r = q.lo + q.hi;
            int2 t = r.lo + r.hi;
            return t.x + t.y;
        }

        // tree reduction of one float per work-item across the GEMV_WG work-group
        inline float wg_sum(__local float* partial, float v) {
            const int lid = get_local_id(0);
            partial[lid] = v;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = GEMV_WG / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    partial[lid] += partial[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            return partial[0];
        }
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
            // IEEE binary16, round-to-nearest-EVEN. Bit-exact with Float.floatToFloat16 /
            // CUDA __float2half / Metal half() for all 2^32 float patterns (checked exhaustively
            // on the Java twin, core/Fp16.java). NB: 'half' is a reserved type name in OpenCL C.
            uint x = as_uint(value);
            uint sign = (x >> 16) & 0x8000u;
            uint expField = (x >> 23) & 0xFFu;
            uint mant = x & 0x007FFFFFu;
            if (expField == 0xFFu) {
                return (ushort) (sign | 0x7C00u | (mant != 0u ? 0x200u : 0u));
            }
            int e = (int) expField - 127 + 15;
            if (e >= 31) {
                return (ushort) (sign | 0x7C00u);
            }
            if (e <= 0) {
                if (e < -10) {
                    return (ushort) sign;
                }
                mant |= 0x00800000u;
                uint shift = (uint) (14 - e);
                uint hs = mant >> shift;
                uint rem = mant & ((1u << shift) - 1u);
                uint halfway = 1u << (shift - 1u);
                if (rem > halfway || (rem == halfway && (hs & 1u) != 0u)) {
                    hs += 1u;
                }
                return (ushort) (sign | hs);
            }
            uint hn = ((uint) e << 10) | (mant >> 13);
            uint rem2 = mant & 0x1FFFu;
            if (rem2 > 0x1000u || (rem2 == 0x1000u && (hn & 1u) != 0u)) {
                hn += 1u;
            }
            return (ushort) (sign | hn);
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

        // One work-group per output row. The GEMV_WG work-items stride over the row's 34-byte
        // blocks, so neighbouring work-items read neighbouring bytes (coalesced) instead of
        // rows K/32*34 bytes apart; partial sums are combined by a local-memory tree reduction.
        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void q8_0_gemv_plain(
            __global const uchar* x_q8_0,
            __global const uchar* W_q8_0,
            __global float* out_f32,
            const int N,
            const int K)
        {
            __local float partial[GEMV_WG];
            const int n = get_group_id(0);
            if (n >= N) return; // uniform across the work-group, so the barriers below stay legal

            const int numBlocks = K / Q8_0_GROUP_SIZE;
            __global const uchar* wRow = W_q8_0 + (size_t) n * (size_t) numBlocks * Q8_0_BLOCK_SIZE;

            float acc = 0.0f;
            for (int b = get_local_id(0); b < numBlocks; b += GEMV_WG) {
                __global const uchar* xb = x_q8_0 + (size_t) b * Q8_0_BLOCK_SIZE;
                __global const uchar* wb = wRow + (size_t) b * Q8_0_BLOCK_SIZE;
                const float scale = decode_fp16(xb[0], xb[1]) * decode_fp16(wb[0], wb[1]);
                __global const char* xq = (__global const char*) (xb + 2);
                __global const char* wq = (__global const char*) (wb + 2);
                const int iacc = dot_i8x16(vload16(0, xq), vload16(0, wq))
                               + dot_i8x16(vload16(1, xq), vload16(1, wq));
                acc += iacc * scale;
            }
            const float total = wg_sum(partial, acc);
            if (get_local_id(0) == 0) out_f32[n] = total;
        }

        __kernel void rope_apply_pairs(
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
            const int evenOff = headOff + (i << 1);
            const int oddOff = evenOff + 1;

            float c = cos_table[cosSinOffset + i];
            float s = sin_table[cosSinOffset + i];
            float x0 = buf[evenOff];
            float x1 = buf[oddOff];

            buf[evenOff] = x0 * c - x1 * s;
            buf[oddOff] = x0 * s + x1 * c;
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

        // NOTE (fixed): B is [N, K] row-major -- N=out_features rows, K=in_features
        // cols, matching GGUF's native Linear-weight layout (PyTorch dumps
        // [out_features, in_features] as-is, no transpose on conversion).
        // This now matches q8_0_gemv_plain's convention. The earlier version
        // read B as [K, N] (B[k*N+n]) -- silently transposed for wo (where
        // K==N==dim, so no shape check could ever catch it) and for the LM
        // head (K=dim, N=vocab, still in-bounds, still wrong). See
        // LlamaLayer's/this kernel's git history -- or ask the person who
        // introduced this fix -- for the full derivation.
        // One work-group per output row; float4 loads when the row is 16-byte aligned (K % 4 == 0).
        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void f32_gemv(
            __global const float* a,
            __global const float* B,
            __global float* out,
            const int K,
            const int N)
        {
            __local float partial[GEMV_WG];
            const int n = get_group_id(0);
            if (n >= N) return; // uniform across the work-group

            __global const float* row = B + (size_t) n * (size_t) K;
            float acc = 0.0f;
            if ((K & 3) == 0) {
                __global const float4* a4 = (__global const float4*) a;
                __global const float4* r4 = (__global const float4*) row;
                float4 acc4 = (float4)(0.0f);
                for (int k = get_local_id(0); k < (K >> 2); k += GEMV_WG) {
                    acc4 += a4[k] * r4[k];
                }
                acc = (acc4.x + acc4.y) + (acc4.z + acc4.w);
            } else {
                for (int k = get_local_id(0); k < K; k += GEMV_WG) {
                    acc += a[k] * row[k];
                }
            }
            const float total = wg_sum(partial, acc);
            if (get_local_id(0) == 0) out[n] = total;
        }

        // ================= candidate Q8_0 GEMV loops and row-wise GEMMs (all bit-compatible with the integer dot) =================
        // The int8*int8 products of one 32-element block are summed as FLOATS here: every product and every partial sum is
        // an integer of magnitude < 2^24 (at most 32*127*127 = 516128), so float arithmetic is exact and the result is
        // identical to the integer dot used by q8_0_gemv_plain; only the instruction mix differs (hardware dp4).
        #define GEMM_TCH 16

        inline float dp4_i8(uint a, uint b) {
            return dot(convert_float4(as_char4(a)), convert_float4(as_char4(b)));
        }

        // One block, loaded with vload16 from the 2-byte-aligned quants (the same memory access as q8_0_gemv_plain).
        inline float blk_dot_vloadf(__global const char* xq, __global const char* wq) {
            const char16 xa = vload16(0, xq);
            const char16 xb = vload16(1, xq);
            const char16 wa = vload16(0, wq);
            const char16 wb = vload16(1, wq);
            return (dot(convert_float4(xa.s0123), convert_float4(wa.s0123)) + dot(convert_float4(xa.s4567), convert_float4(wa.s4567)))
                 + (dot(convert_float4(xa.s89ab), convert_float4(wa.s89ab)) + dot(convert_float4(xa.scdef), convert_float4(wa.scdef)))
                 + ((dot(convert_float4(xb.s0123), convert_float4(wb.s0123)) + dot(convert_float4(xb.s4567), convert_float4(wb.s4567)))
                 + (dot(convert_float4(xb.s89ab), convert_float4(wb.s89ab)) + dot(convert_float4(xb.scdef), convert_float4(wb.scdef))));
        }

        // Two consecutive Q8_0 blocks form a 68-byte unit that is 4-byte aligned (68 = 17 words) whenever the row has an
        // even number of blocks. Word 0 = scale0 + quants 0,1; words 0..8 hold block 0 (quants straddle two words, shifted
        // by 16 bits); word 8's high half is scale1; words 9..16 hold block 1's 32 quants, aligned.
        // wu = the 17 words of the weight unit (already loaded), xp = the activation unit in global memory.
        inline float2 pair_dot_w(__global const uint* xp, const uint* wu) {
            uint xu[17];
            #pragma unroll
            for (int i = 0; i < 17; i++) {
                xu[i] = xp[i];
            }
            const float xs0 = decode_fp16((uchar) (xu[0] & 0xFFu), (uchar) ((xu[0] >> 8) & 0xFFu));
            const float ws0 = decode_fp16((uchar) (wu[0] & 0xFFu), (uchar) ((wu[0] >> 8) & 0xFFu));
            const float xs1 = decode_fp16((uchar) ((xu[8] >> 16) & 0xFFu), (uchar) ((xu[8] >> 24) & 0xFFu));
            const float ws1 = decode_fp16((uchar) ((wu[8] >> 16) & 0xFFu), (uchar) ((wu[8] >> 24) & 0xFFu));
            float d0 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                d0 += dp4_i8((xu[i] >> 16) | (xu[i + 1] << 16), (wu[i] >> 16) | (wu[i + 1] << 16));
            }
            float d1 = 0.0f;
            #pragma unroll
            for (int i = 9; i < 17; i++) {
                d1 += dp4_i8(xu[i], wu[i]);
            }
            return (float2)(d0 * (xs0 * ws0), d1 * (xs1 * ws1));
        }

        // Same memory access as q8_0_gemv_plain, but the block dot uses float dp4.
        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void q8_0_gemv_vloadf(
            __global const uchar* x_q8_0,
            __global const uchar* W_q8_0,
            __global float* out_f32,
            const int N,
            const int K)
        {
            __local float partial[GEMV_WG];
            const int n = get_group_id(0);
            if (n >= N) return;
            const int numBlocks = K / Q8_0_GROUP_SIZE;
            __global const uchar* wRow = W_q8_0 + (size_t) n * (size_t) numBlocks * Q8_0_BLOCK_SIZE;
            float acc = 0.0f;
            for (int b = get_local_id(0); b < numBlocks; b += GEMV_WG) {
                __global const uchar* xb = x_q8_0 + (size_t) b * Q8_0_BLOCK_SIZE;
                __global const uchar* wb = wRow + (size_t) b * Q8_0_BLOCK_SIZE;
                const float scale = decode_fp16(xb[0], xb[1]) * decode_fp16(wb[0], wb[1]);
                acc += blk_dot_vloadf((__global const char*) (xb + 2), (__global const char*) (wb + 2)) * scale;
            }
            const float total = wg_sum(partial, acc);
            if (get_local_id(0) == 0) out_f32[n] = total;
        }

        // Aligned 4-byte loads over 68-byte block pairs. Requires an even number of blocks per row (K % 64 == 0).
        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void q8_0_gemv_u32(
            __global const uchar* x_q8_0,
            __global const uchar* W_q8_0,
            __global float* out_f32,
            const int N,
            const int K)
        {
            __local float partial[GEMV_WG];
            const int n = get_group_id(0);
            if (n >= N) return;
            const int numPairs = K / (2 * Q8_0_GROUP_SIZE);
            __global const uchar* wRow = W_q8_0 + (size_t) n * (size_t) numPairs * 68;
            float acc = 0.0f;
            for (int p = get_local_id(0); p < numPairs; p += GEMV_WG) {
                __global const uint* wp = (__global const uint*) (wRow + (size_t) p * 68);
                uint wu[17];
                #pragma unroll
                for (int i = 0; i < 17; i++) {
                    wu[i] = wp[i];
                }
                const float2 c = pair_dot_w((__global const uint*) (x_q8_0 + (size_t) p * 68), wu);
                acc += c.x;
                acc += c.y;
            }
            const float total = wg_sum(partial, acc);
            if (get_local_id(0) == 0) out_f32[n] = total;
        }

        // Prefill GEMM: one work-group per output row n and per chunk of GEMM_TCH tokens. The weight row is read ONCE
        // (aligned words) and applied to every token of the chunk, instead of being re-read for each token.
        // Output layout [T, N]. Requires an even number of blocks per row.
        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void q8_0_gemm_rows(
            __global const uchar* X_q8_0,
            __global const uchar* W_q8_0,
            __global float* out,
            const int T,
            const int N,
            const int K)
        {
            __local float partial[GEMM_TCH * (GEMV_WG + 1)];
            const int n = get_group_id(0);
            const int t0 = get_group_id(1) * GEMM_TCH;
            const int lid = get_local_id(0);
            if (n >= N) return;
            const int tn = min(GEMM_TCH, T - t0);
            const int numPairs = K / (2 * Q8_0_GROUP_SIZE);
            const size_t rowStride = (size_t) numPairs * 68;
            __global const uchar* wRow = W_q8_0 + (size_t) n * rowStride;
            float acc[GEMM_TCH];
            #pragma unroll
            for (int t = 0; t < GEMM_TCH; t++) {
                acc[t] = 0.0f;
            }
            for (int p = lid; p < numPairs; p += GEMV_WG) {
                __global const uint* wp = (__global const uint*) (wRow + (size_t) p * 68);
                uint wu[17];
                #pragma unroll
                for (int i = 0; i < 17; i++) {
                    wu[i] = wp[i];
                }
                #pragma unroll
                for (int t = 0; t < GEMM_TCH; t++) {
                    if (t < tn) {
                        const float2 c = pair_dot_w((__global const uint*) (X_q8_0 + (size_t) (t0 + t) * rowStride + (size_t) p * 68), wu);
                        acc[t] += c.x;
                        acc[t] += c.y;
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < GEMM_TCH; t++) {
                partial[t * (GEMV_WG + 1) + lid] = acc[t];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < tn) {
                float sum = 0.0f;
                for (int l = 0; l < GEMV_WG; l++) {
                    sum += partial[lid * (GEMV_WG + 1) + l];
                }
                out[(size_t) (t0 + lid) * N + n] = sum;
            }
        }

        // F32 prefill GEMM with the same structure: A [T, K], B [N, K] (row-major weights), out [T, N]. Requires K % 4 == 0.
        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void f32_gemm_rows(
            __global const float* A,
            __global const float* B,
            __global float* out,
            const int T,
            const int K,
            const int N)
        {
            __local float partial[GEMM_TCH * (GEMV_WG + 1)];
            const int n = get_group_id(0);
            const int t0 = get_group_id(1) * GEMM_TCH;
            const int lid = get_local_id(0);
            if (n >= N) return;
            const int tn = min(GEMM_TCH, T - t0);
            const int K4 = K >> 2;
            __global const float4* w4 = (__global const float4*) (B + (size_t) n * (size_t) K);
            float acc[GEMM_TCH];
            #pragma unroll
            for (int t = 0; t < GEMM_TCH; t++) {
                acc[t] = 0.0f;
            }
            for (int k = lid; k < K4; k += GEMV_WG) {
                const float4 w = w4[k];
                #pragma unroll
                for (int t = 0; t < GEMM_TCH; t++) {
                    if (t < tn) {
                        acc[t] += dot(((__global const float4*) (A + (size_t) (t0 + t) * (size_t) K))[k], w);
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < GEMM_TCH; t++) {
                partial[t * (GEMV_WG + 1) + lid] = acc[t];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < tn) {
                float sum = 0.0f;
                for (int l = 0; l < GEMV_WG; l++) {
                    sum += partial[lid * (GEMV_WG + 1) + l];
                }
                out[(size_t) (t0 + lid) * N + n] = sum;
            }
        }

        // Batched (prefill) RMSNorm in ONE launch with no host round trip: one work-group per row. The legacy prefill path
        // computed per-group partial sums, read them back (blocking), summed them on the host and uploaded the per-row
        // factors again: 2 blocking transfers per norm, ~9 ms of idle GPU each on the Intel driver this was measured on.
        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void rmsnorm_fused_rows(
            __global const float* x,       // [T, features]
            __global const float* gamma,   // [features]
            __global float* out,           // [T, features]
            const int features,
            const float eps)
        {
            __local float partial[GEMV_WG];
            __global const float* xr = x + (size_t) get_group_id(0) * (size_t) features;
            __global float* outr = out + (size_t) get_group_id(0) * (size_t) features;
            float ss = 0.0f;
            for (int i = get_local_id(0); i < features; i += GEMV_WG) {
                const float v = xr[i];
                ss += v * v;
            }
            const float total = wg_sum(partial, ss);
            const float inv = 1.0f / sqrt(total / (float) features + eps);
            for (int i = get_local_id(0); i < features; i += GEMV_WG) {
                outr[i] = xr[i] * inv * gamma[i];
            }
        }

        // rope_apply_pairwise_rows with positions computed in the kernel (row r is at absolute position startPos + r),
        // so no positions array has to be uploaded (the legacy path uploaded the same array once per layer, blocking).
        __kernel void rope_apply_pairwise_rows_seq(
            __global float* buf,             // [T, heads*head_dim], plain contiguous layout
            __global const float* cos_table, // [max_seq, halfDim]
            __global const float* sin_table,
            const int heads,
            const int head_dim,
            const int startPos)
        {
            const int halfDim = head_dim >> 1;
            const int row = get_group_id(1);
            const int gid = get_group_id(0) * get_local_size(0) + get_local_id(0);
            const int h = gid / halfDim;
            const int i = gid % halfDim;
            if (h >= heads) return;

            const int pos = startPos + row;
            const int pairOff = row * heads * head_dim + h * head_dim + (i << 1);
            const float c = cos_table[pos * halfDim + i];
            const float s = sin_table[pos * halfDim + i];
            const float x0 = buf[pairOff];
            const float x1 = buf[pairOff + 1];
            buf[pairOff] = x0 * c - x1 * s;
            buf[pairOff + 1] = x0 * s + x1 * c;
        }

        // ---- original generic implementations (one work-item per output row); kept as selectable fallbacks (-Dllm.gemv=legacy)
        __kernel void q8_0_gemv_plain_legacy(
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

        __kernel void f32_gemv_legacy(
            __global const float* a,
            __global const float* B,
            __global float* out,
            const int K,
            const int N)
        {
            const int n = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (n >= N) return;

            const int rowOff = n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += a[k] * B[rowOff + k];
            }
            out[n] = acc;
        }

        // RMSNorm in ONE launch with no host round trip: a single work-group computes sum(x^2) by
        // tree reduction, then scales. (The legacy path needs a partial-sums kernel, a blocking
        // read-back, a host sum, and a second kernel.) In-place safe: all reads of x precede the barrier.
        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void rmsnorm_fused(
            __global const float* x,
            __global const float* gamma,
            __global float* out,
            const int features,
            const float eps)
        {
            __local float partial[GEMV_WG];
            float ss = 0.0f;
            for (int i = get_local_id(0); i < features; i += GEMV_WG) {
                const float v = x[i];
                ss += v * v;
            }
            const float total = wg_sum(partial, ss);
            const float inv = 1.0f / sqrt(total / (float) features + eps);
            for (int i = get_local_id(0); i < features; i += GEMV_WG) {
                out[i] = x[i] * inv * gamma[i];
            }
        }

        // ---- memory-bandwidth probes (OpenCLBenchmark --bandwidth): grid-stride coalesced reads at different load widths.
        // bw_read_v16_off2 reads 16-byte chunks starting 2 bytes past a 4-byte boundary, i.e. the alignment of a
        // Q8_0 block's quants (34-byte blocks), to show what misaligned vector loads cost on the device.
        __kernel void bw_read_u8(__global const uchar* p, __global uint* out, const int n) {
            uint acc = 0;
            for (int i = get_global_id(0); i < n; i += get_global_size(0)) acc += p[i];
            out[get_global_id(0)] = acc;
        }
        __kernel void bw_read_u32(__global const uint* p, __global uint* out, const int n) {
            uint acc = 0;
            for (int i = get_global_id(0); i < n; i += get_global_size(0)) acc += p[i];
            out[get_global_id(0)] = acc;
        }
        __kernel void bw_read_u128(__global const uint4* p, __global uint* out, const int n) {
            uint4 acc = (uint4)(0u);
            for (int i = get_global_id(0); i < n; i += get_global_size(0)) acc += p[i];
            out[get_global_id(0)] = acc.x + acc.y + acc.z + acc.w;
        }
        __kernel void bw_read_v16_off2(__global const uchar* p, __global uint* out, const int n) {
            __global const char* q = (__global const char*) (p + 2);
            int acc = 0;
            for (int i = get_global_id(0); i < n; i += get_global_size(0)) {
                char16 v = vload16(i, q);
                int8 t = convert_int8(v.lo) + convert_int8(v.hi);
                int4 u = t.lo + t.hi;
                acc += u.x + u.y + u.z + u.w;
            }
            out[get_global_id(0)] = (uint) acc;
        }

        #define ATTN_WG GEMV_WG
        #define ATTN_TILE 256
        #define ATTN_MAX_HEAD_DIM 256

        // work-group max / sum that are safe to call repeatedly on the same scratch array
        inline float wg_max_sync(__local float* red, float v) {
            const int lid = get_local_id(0);
            red[lid] = v;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = ATTN_WG / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    red[lid] = fmax(red[lid], red[lid + stride]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const float r = red[0];
            barrier(CLK_LOCAL_MEM_FENCE);
            return r;
        }

        inline float wg_sum_sync(__local float* red, float v) {
            const float r = wg_sum(red, v);
            barrier(CLK_LOCAL_MEM_FENCE);
            return r;
        }

        // Decode-step attention for ALL query heads in one launch: one work-group per head.
        // The cached context is consumed in tiles of ATTN_TILE positions with an online softmax
        // (running max m, normaliser l, per-dimension accumulators), so no score buffer is needed and
        // any context length works. Grouped-query: head h reads KV head h / (numHeads / kvHeads).
        // Requires headDim % 4 == 0 and headDim <= ATTN_MAX_HEAD_DIM (the host falls back otherwise).
        __kernel __attribute__((reqd_work_group_size(ATTN_WG, 1, 1)))
        void attn_decode_fused(
            __global const float* q,
            __global const float* k_cache,
            __global const float* v_cache,
            __global float* out,
            const int numHeads,
            const int kvHeads,
            const int headDim,
            const int nCtx,
            const float scale)
        {
            __local float qs[ATTN_MAX_HEAD_DIM];
            __local float tile[ATTN_TILE];
            __local float red[ATTN_WG];

            const int h = get_group_id(0);
            const int lid = get_local_id(0);
            const int kvDim = kvHeads * headDim;
            const int kvOff = (h / (numHeads / kvHeads)) * headDim;

            for (int d = lid; d < headDim; d += ATTN_WG) {
                qs[d] = q[h * headDim + d];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float m = -FLT_MAX;
            float l = 0.0f;
            float acc[ATTN_MAX_HEAD_DIM / ATTN_WG];
            for (int i = 0; i < ATTN_MAX_HEAD_DIM / ATTN_WG; i++) {
                acc[i] = 0.0f;
            }

            for (int t0 = 0; t0 < nCtx; t0 += ATTN_TILE) {
                const int tn = min(ATTN_TILE, nCtx - t0);

                float tmax = -FLT_MAX;
                for (int j = lid; j < tn; j += ATTN_WG) {
                    __global const float4* kj = (__global const float4*) (k_cache + (size_t) (t0 + j) * kvDim + kvOff);
                    float4 a4 = (float4)(0.0f);
                    for (int d4 = 0; d4 < (headDim >> 2); d4++) {
                        a4 += vload4(d4, qs) * kj[d4];
                    }
                    const float sc = ((a4.x + a4.y) + (a4.z + a4.w)) * scale;
                    tile[j] = sc;
                    tmax = fmax(tmax, sc);
                }
                const float mNew = fmax(m, wg_max_sync(red, tmax));
                const float corr = exp(m - mNew);

                float psum = 0.0f;
                for (int j = lid; j < tn; j += ATTN_WG) {
                    const float p = exp(tile[j] - mNew);
                    tile[j] = p;
                    psum += p;
                }
                l = l * corr + wg_sum_sync(red, psum); // the barrier inside also publishes tile[]

                for (int i = 0; i < ATTN_MAX_HEAD_DIM / ATTN_WG; i++) {
                    const int d = lid + i * ATTN_WG;
                    if (d < headDim) {
                        float a = acc[i] * corr;
                        for (int j = 0; j < tn; j++) {
                            a += tile[j] * v_cache[(size_t) (t0 + j) * kvDim + kvOff + d];
                        }
                        acc[i] = a;
                    }
                }
                m = mNew;
                barrier(CLK_LOCAL_MEM_FENCE); // tile[] is rewritten by the next tile
            }

            for (int i = 0; i < ATTN_MAX_HEAD_DIM / ATTN_WG; i++) {
                const int d = lid + i * ATTN_WG;
                if (d < headDim) {
                    out[h * headDim + d] = acc[i] / l;
                }
            }
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

        // NOTE (fixed): B is [N, K] row-major, matching GGUF's native
        // Linear-weight layout -- same fix and rationale as f32_gemv above.
        __kernel void f32_gemm_tiled(
            __global const float* A, // [T, K]
            __global const float* B, // [N, K]
            __global float* out,     // [T, N]
            const int T,
            const int K,
            const int N)
        {
            const int t = get_group_id(1);
            const int n = get_group_id(0) * get_local_size(0) + get_local_id(0);
            if (t >= T || n >= N) return;

            const int bRowOff = n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[t * K + k] * B[bRowOff + k];
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