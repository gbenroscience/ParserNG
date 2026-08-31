package com.github.gbenroscience.gpu.llm.metal;

/**
 * Metal Shading Language counterpart of
 * {@code com.github.gbenroscience.gpu.llm.cuda.KernelSource} -- same 21
 * kernels (13 decode-path + 2 FFN activation alternatives + 8 batched
 * prefill), same names, same algorithms, same tolerances. Only the
 * dispatch-model plumbing differs; see the per-kernel notes below for the
 * mechanical translation rules applied uniformly:
 *
 *   - {@code blockIdx.x*blockDim.x+threadIdx.x} (1D CUDA global id)
 *     becomes {@code gid.x} from {@code [[thread_position_in_grid]]},
 *     because every dispatch in this port uses
 *     {@code dispatchThreads:threadsPerThreadgroup:} (see
 *     {@link MetalBindings#dispatchThreads}), which gives MSL kernels a
 *     GLOBAL thread id directly -- no manual
 *     {@code blockIdx*blockDim+threadIdx} reconstruction needed, and no
 *     grid-rounding bounds-check edge case either (Metal handles the
 *     ragged final threadgroup itself for this dispatch selector).
 *   - CUDA's 2D {@code blockIdx.y} (row) / {@code blockIdx.x*blockDim.x+threadIdx.x}
 *     (column) becomes {@code gid.y} / {@code gid.x} directly, same reasoning.
 *   - {@code extern __shared__ float buf[]} (CUDA's dynamically-sized
 *     shared memory) becomes a {@code threadgroup float*} parameter at
 *     {@code [[threadgroup(0)]]}, sized host-side via
 *     {@link MetalBindings#setThreadgroupMemoryLength}.
 *   - {@code __syncthreads()} becomes {@code threadgroup_barrier(mem_flags::mem_threadgroup)}.
 *   - CUDA kernel scalar arguments (plain {@code int}/{@code float} params)
 *     become {@code constant T&} parameters, each at its own
 *     {@code [[buffer(N)]]} index, set host-side via
 *     {@link MetalBindings#setBytes} -- MSL has no varargs kernel-params
 *     array the way {@code cuLaunchKernel} does, so every kernel below
 *     gets an explicit, sequential buffer-index scheme matching its CUDA
 *     signature's argument order 1:1 (documented per dispatch call in
 *     {@link LlamaLayer}).
 *   - {@code __half}/{@code __float2half}/{@code __half2float} (CUDA's
 *     {@code cuda_fp16.h}) become MSL's native {@code half} type --
 *     {@code as_type<half>(bits)} / {@code as_type<ushort>(half_value)}
 *     reinterpret exactly the way {@code __ushort_as_half}/{@code __half_as_ushort}
 *     did.
 *   - {@code erff}/{@code expf}/{@code fmaxf}/{@code fabsf}/{@code rintf}
 *     become MSL's {@code erf}/{@code exp}/{@code fmax}/{@code fabs}/{@code rint}
 *     (MSL's {@code metal_math} overloads on {@code float} directly, no
 *     {@code f}-suffixed variant needed).
 *
 * See the CUDA {@code KernelSource}'s class javadoc for the algorithmic
 * rationale each kernel implements (Q8_0 block layout, why attention is
 * three unfused dispatches, what "batched" buys prefill, and the explicit
 * "not cuBLAS-tier" caveat on the GEMM kernels) -- all of that applies
 * completely unchanged here; only the C-vs-MSL syntax differs.
 *
 * UNVERIFIED, same standing caveat as every kernel file in this codebase:
 * no Metal GPU/toolchain was available while writing this. Traced
 * carefully against the CUDA source kernel-by-kernel and against Apple's
 * published MSL specification, but not compiled or run.
 */
public final class MetalKernelSource {

    private MetalKernelSource() {
    }

    // ---- decode-path kernels (unchanged names from the CUDA/OpenCL ports) ----
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

    /** Threadgroup size used for both RMSNorm partial-sum kernels' local reduction. Must match host dispatch. */
    public static final int RMSNORM_WORKGROUP_SIZE = 256;

    /** Fixed threadgroup size for quantize_activation_q8_0_blocks -- one thread per element in a 32-wide Q8_0 group. */
    public static final int QUANTIZE_BLOCK_SIZE = 32;

    /** Default threadgroup size for the simple bound-checked 1D kernels (GEMVs, RoPE, elementwise ops). */
    public static final int DEFAULT_BLOCK_SIZE = 256;

    /** Default column-tile threadgroup width for the batched GEMM kernels (threadgroup.x; grid.y iterates rows). */
    public static final int GEMM_TILE_N = 128;

    public static final String METAL_SOURCE = """
        #include <metal_stdlib>
        using namespace metal;

        #define Q8_0_GROUP_SIZE 32
        #define Q8_0_BLOCK_SIZE 34
        #define DEVICE_FLT_MAX 3.402823466e+38f

        inline float decode_fp16(uchar lo, uchar hi) {
            ushort bits = (ushort) lo | ((ushort) hi << 8);
            return float(as_type<half>(bits));
        }

        inline float gpu_sigmoid_f(float x) {
            return 1.0f / (1.0f + exp(-x));
        }

        // Exact-erf GeLU -- same convention as the CUDA/OpenCL ports' gpu_gelu_f:
        // NOT a specific model's training-time tanh/sigmoid approximation.
        // Diff against reference activations before trusting GeLU-FFN output.
        inline float gpu_gelu_f(float x) {
            return 0.5f * x * (1.0f + erf(x * 0.70710678f));
        }

        // =====================================================================
        // ===================== DECODE-PATH KERNELS ==========================
        // =====================================================================

        kernel void quantize_i8(
            device const float* x            [[buffer(0)]],
            device char* x_q8                [[buffer(1)]],
            constant float& invScale         [[buffer(2)]],
            uint gid                         [[thread_position_in_grid]])
        {
            float val = x[gid] * invScale;
            int q = (int) rint(val);
            q = max(-127, min(127, q));
            x_q8[gid] = (char) q;
        }

        kernel void quantize_activation_q8_0_blocks(
            device const float* x            [[buffer(0)]],
            device uchar* x_q8_0             [[buffer(1)]],
            constant int& len                [[buffer(2)]],
            threadgroup float* sh            [[threadgroup(0)]],
            uint3 tgpig                      [[threadgroup_position_in_grid]],
            uint3 tid                        [[thread_position_in_threadgroup]])
        {
            const uint block = tgpig.x;
            const uint lane = tid.x; // 0..31
            const uint blockOff = block * Q8_0_GROUP_SIZE;
            const uint outOff = block * Q8_0_BLOCK_SIZE;

            float v = x[blockOff + lane];
            sh[lane] = fabs(v);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = 16; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    sh[lane] = fmax(sh[lane], sh[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float absmax = sh[0];
            float scale = absmax / 127.0f;
            float invScale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            int q = (int) rint(v * invScale);
            q = max(-127, min(127, q));
            x_q8_0[outOff + 2 + lane] = (uchar)(char) q;

            if (lane == 0) {
                half scaleH = half(scale);
                ushort bits = as_type<ushort>(scaleH);
                x_q8_0[outOff] = (uchar) (bits & 0xFF);
                x_q8_0[outOff + 1] = (uchar) ((bits >> 8) & 0xFF);
            }
        }

        kernel void q8_0_gemv_split(
            device const uchar* x_q8_0       [[buffer(0)]],
            device const uchar* W_q8_0       [[buffer(1)]],
            device float* out_f32_split      [[buffer(2)]],
            constant int& qHeads             [[buffer(3)]],
            constant int& head_dim           [[buffer(4)]],
            constant int& K                  [[buffer(5)]],
            uint gid                         [[thread_position_in_grid]])
        {
            const int halfDim = head_dim >> 1;
            const int h = (int) gid / halfDim;
            const int i = (int) gid % halfDim;
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

        kernel void q8_0_gemv_plain(
            device const uchar* x_q8_0       [[buffer(0)]],
            device const uchar* W_q8_0       [[buffer(1)]],
            device float* out_f32            [[buffer(2)]],
            constant int& N                  [[buffer(3)]],
            constant int& K                  [[buffer(4)]],
            uint gid                         [[thread_position_in_grid]])
        {
            const int n = (int) gid;
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

        kernel void rope_apply_split(
            device float* buf                [[buffer(0)]],
            device const float* cos_table    [[buffer(1)]],
            device const float* sin_table    [[buffer(2)]],
            constant int& heads              [[buffer(3)]],
            constant int& head_dim           [[buffer(4)]],
            constant int& cosSinOffset       [[buffer(5)]],
            uint gid                         [[thread_position_in_grid]])
        {
            const int halfDim = head_dim >> 1;
            const int h = (int) gid / halfDim;
            const int i = (int) gid % halfDim;
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

        kernel void rmsnorm_partial_sumsq(
            device const float* x            [[buffer(0)]],
            device float* partials           [[buffer(1)]],
            constant int& features           [[buffer(2)]],
            threadgroup float* sh            [[threadgroup(0)]],
            uint3 tgpig                      [[threadgroup_position_in_grid]],
            uint3 tid                        [[thread_position_in_threadgroup]],
            uint3 tgSize                     [[threads_per_threadgroup]])
        {
            const uint gid = tgpig.x * tgSize.x + tid.x;
            const uint lid = tid.x;
            const uint groupId = tgpig.x;
            const uint localSize = tgSize.x;

            float v = (gid < (uint) features) ? x[gid] : 0.0f;
            sh[lid] = v * v;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh[lid] += sh[lid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                partials[groupId] = sh[0];
            }
        }

        kernel void rmsnorm_apply(
            device const float* x            [[buffer(0)]],
            device const float* gamma        [[buffer(1)]],
            device float* out                [[buffer(2)]],
            constant float& rms              [[buffer(3)]],
            constant int& features           [[buffer(4)]],
            uint gid                         [[thread_position_in_grid]])
        {
            if ((int) gid >= features) return;
            out[gid] = x[gid] * rms * gamma[gid];
        }

        kernel void attn_scores(
            device const float* q_all_heads  [[buffer(0)]],
            device const float* k_cache_f32  [[buffer(1)]],
            device float* scores             [[buffer(2)]],
            constant int& qHeadOff           [[buffer(3)]],
            constant int& head_dim           [[buffer(4)]],
            constant int& kv_dim             [[buffer(5)]],
            constant int& kv_head_off        [[buffer(6)]],
            constant int& posInclusive       [[buffer(7)]],
            constant float& rsqrt_d          [[buffer(8)]],
            uint gid                         [[thread_position_in_grid]])
        {
            const int j = (int) gid;
            if (j >= posInclusive) return;

            const int kOff = j * kv_dim + kv_head_off;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_all_heads[qHeadOff + d] * k_cache_f32[kOff + d];
            }
            scores[j] = dot * rsqrt_d;
        }

        kernel void softmax_inplace(
            device float* scores             [[buffer(0)]],
            constant int& len                [[buffer(1)]],
            threadgroup float* sh            [[threadgroup(0)]],
            uint3 tid                        [[thread_position_in_threadgroup]],
            uint3 tgSize                     [[threads_per_threadgroup]])
        {
            const uint lid = tid.x;
            const uint localSize = tgSize.x;

            float localMax = -DEVICE_FLT_MAX;
            for (uint i = lid; i < (uint) len; i += localSize) {
                localMax = fmax(localMax, scores[i]);
            }
            sh[lid] = localMax;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh[lid] = fmax(sh[lid], sh[lid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float maxVal = sh[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float localSum = 0.0f;
            for (uint i = lid; i < (uint) len; i += localSize) {
                float e = exp(scores[i] - maxVal);
                scores[i] = e;
                localSum += e;
            }
            sh[lid] = localSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh[lid] += sh[lid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float sumVal = sh[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float invSum = 1.0f / sumVal;
            for (uint i = lid; i < (uint) len; i += localSize) {
                scores[i] *= invSum;
            }
        }

        kernel void attn_weighted_sum(
            device const float* scores       [[buffer(0)]],
            device const float* v_cache_f32  [[buffer(1)]],
            device float* attn_out_all_heads [[buffer(2)]],
            constant int& outHeadOff         [[buffer(3)]],
            constant int& head_dim           [[buffer(4)]],
            constant int& kv_dim             [[buffer(5)]],
            constant int& kv_head_off        [[buffer(6)]],
            constant int& posInclusive       [[buffer(7)]],
            uint gid                         [[thread_position_in_grid]])
        {
            const int d = (int) gid;
            if (d >= head_dim) return;

            float acc = 0.0f;
            for (int j = 0; j < posInclusive; j++) {
                acc += scores[j] * v_cache_f32[j * kv_dim + kv_head_off + d];
            }
            attn_out_all_heads[outHeadOff + d] = acc;
        }

        kernel void swiglu_activate(
            device const float* gate         [[buffer(0)]],
            device const float* up           [[buffer(1)]],
            device float* out                [[buffer(2)]],
            constant int& hidden             [[buffer(3)]],
            uint gid                         [[thread_position_in_grid]])
        {
            if ((int) gid >= hidden) return;
            float g = gate[gid];
            float sigmoid = 1.0f / (1.0f + exp(-fmax(g, -88.0f)));
            out[gid] = g * sigmoid * up[gid];
        }

        kernel void residual_add(
            device float* x                  [[buffer(0)]],
            device const float* y            [[buffer(1)]],
            constant int& len                [[buffer(2)]],
            uint gid                         [[thread_position_in_grid]])
        {
            if ((int) gid >= len) return;
            x[gid] += y[gid];
        }

        // B is [N, K] row-major -- N=out_features rows, K=in_features cols,
        // matching GGUF's native Linear-weight layout (same fix/rationale
        // as the CUDA port's f32_gemv).
        kernel void f32_gemv(
            device const float* a            [[buffer(0)]],
            device const float* B            [[buffer(1)]],
            device float* out                [[buffer(2)]],
            constant int& K                  [[buffer(3)]],
            constant int& N                  [[buffer(4)]],
            uint gid                         [[thread_position_in_grid]])
        {
            const int n = (int) gid;
            if (n >= N) return;

            const int rowOff = n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += a[k] * B[rowOff + k];
            }
            out[n] = acc;
        }

        // =====================================================================
        // ===================== FFN ACTIVATION ALTERNATIVES ==================
        // =====================================================================

        kernel void gelu_activate(
            device const float* gate         [[buffer(0)]],
            device float* out                [[buffer(1)]],
            constant int& hidden             [[buffer(2)]],
            uint gid                         [[thread_position_in_grid]])
        {
            if ((int) gid >= hidden) return;
            out[gid] = gpu_gelu_f(gate[gid]);
        }

        kernel void geglu_activate(
            device const float* gate         [[buffer(0)]],
            device const float* up           [[buffer(1)]],
            device float* out                [[buffer(2)]],
            constant int& hidden             [[buffer(3)]],
            uint gid                         [[thread_position_in_grid]])
        {
            if ((int) gid >= hidden) return;
            out[gid] = gpu_gelu_f(gate[gid]) * up[gid];
        }

        // =====================================================================
        // ===================== BATCHED PREFILL KERNELS ======================
        // =====================================================================

        kernel void q8_0_gemm_tiled(
            device const uchar* X_q8_0       [[buffer(0)]], // [T, numBlocks*34]
            device const uchar* W_q8_0       [[buffer(1)]], // [N, numBlocks*34]
            device float* out                [[buffer(2)]], // [T, N]
            constant int& T                  [[buffer(3)]],
            constant int& N                  [[buffer(4)]],
            constant int& K                  [[buffer(5)]],
            uint2 gid                        [[thread_position_in_grid]]) // gid.x = n, gid.y = t
        {
            const int t = (int) gid.y;
            const int n = (int) gid.x;
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

        // B is [N, K] row-major, matching GGUF's native Linear-weight layout.
        kernel void f32_gemm_tiled(
            device const float* A            [[buffer(0)]], // [T, K]
            device const float* B            [[buffer(1)]], // [N, K]
            device float* out                [[buffer(2)]], // [T, N]
            constant int& T                  [[buffer(3)]],
            constant int& K                  [[buffer(4)]],
            constant int& N                  [[buffer(5)]],
            uint2 gid                        [[thread_position_in_grid]])
        {
            const int t = (int) gid.y;
            const int n = (int) gid.x;
            if (t >= T || n >= N) return;

            const int bRowOff = n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[t * K + k] * B[bRowOff + k];
            }
            out[t * N + n] = acc;
        }

        kernel void rmsnorm_partial_sumsq_rows(
            device const float* x            [[buffer(0)]], // [T, features]
            device float* partials           [[buffer(1)]], // [T, numGroups]
            constant int& features           [[buffer(2)]],
            constant int& numGroups          [[buffer(3)]],
            threadgroup float* sh            [[threadgroup(0)]],
            uint3 tgpig                      [[threadgroup_position_in_grid]], // .x = groupId, .y = row
            uint3 tid                        [[thread_position_in_threadgroup]],
            uint3 tgSize                     [[threads_per_threadgroup]])
        {
            const uint row = tgpig.y;
            const uint groupId = tgpig.x;
            const uint lid = tid.x;
            const uint localSize = tgSize.x;
            const uint idx = groupId * localSize + lid;

            float v = (idx < (uint) features) ? x[row * (uint) features + idx] : 0.0f;
            sh[lid] = v * v;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh[lid] += sh[lid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                partials[row * (uint) numGroups + groupId] = sh[0];
            }
        }

        kernel void rmsnorm_apply_rows(
            device const float* x            [[buffer(0)]], // [T, features]
            device const float* gamma        [[buffer(1)]], // [features]
            device float* out                [[buffer(2)]], // [T, features]
            device const float* rms          [[buffer(3)]], // [T]
            constant int& features           [[buffer(4)]],
            uint2 gid                        [[thread_position_in_grid]]) // gid.x = i, gid.y = row
        {
            const int row = (int) gid.y;
            const int i = (int) gid.x;
            if (i >= features) return;
            out[row * features + i] = x[row * features + i] * rms[row] * gamma[i];
        }

        // Batched RoPE for the ADJACENT-PAIR layout q8_0_gemm_tiled naturally
        // produces -- see the CUDA KernelSource's javadoc for why this differs
        // deliberately from rope_apply_split's even/odd-SPLIT layout.
        kernel void rope_apply_pairwise_rows(
            device float* buf                [[buffer(0)]], // [T, heads*head_dim]
            device const float* cos_table    [[buffer(1)]], // [max_seq, halfDim]
            device const float* sin_table    [[buffer(2)]],
            constant int& heads              [[buffer(3)]],
            constant int& head_dim           [[buffer(4)]],
            device const int* positions      [[buffer(5)]], // [T]
            uint2 gid                        [[thread_position_in_grid]]) // gid.x -> (h,i), gid.y = row
        {
            const int halfDim = head_dim >> 1;
            const int row = (int) gid.y;
            const int g = (int) gid.x;
            const int h = g / halfDim;
            const int i = g % halfDim;
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

        kernel void attn_scores_causal_batched(
            device const float* q_all        [[buffer(0)]], // [T, qRowStride]
            device const float* k_all        [[buffer(1)]], // [T, kRowStride]
            device float* scores             [[buffer(2)]], // [T, T]
            constant int& qRowStride         [[buffer(3)]],
            constant int& kRowStride         [[buffer(4)]],
            constant int& qHeadOff           [[buffer(5)]],
            constant int& kHeadOff           [[buffer(6)]],
            constant int& head_dim           [[buffer(7)]],
            constant int& T                  [[buffer(8)]],
            constant float& rsqrt_d          [[buffer(9)]],
            uint2 gid                        [[thread_position_in_grid]]) // gid.x = j, gid.y = t
        {
            const int t = (int) gid.y;
            const int j = (int) gid.x;
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

        kernel void softmax_inplace_rows(
            device float* scores             [[buffer(0)]], // [T, T]
            constant int& T                  [[buffer(1)]],
            threadgroup float* sh            [[threadgroup(0)]],
            uint3 tgpig                      [[threadgroup_position_in_grid]], // .x = row (one threadgroup per row)
            uint3 tid                        [[thread_position_in_threadgroup]],
            uint3 tgSize                     [[threads_per_threadgroup]])
        {
            const uint row = tgpig.x;
            const uint lid = tid.x;
            const uint localSize = tgSize.x;
            device float* rowPtr = scores + row * (uint) T;

            float localMax = -DEVICE_FLT_MAX;
            for (uint i = lid; i < (uint) T; i += localSize) {
                localMax = fmax(localMax, rowPtr[i]);
            }
            sh[lid] = localMax;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh[lid] = fmax(sh[lid], sh[lid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float maxVal = sh[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float localSum = 0.0f;
            for (uint i = lid; i < (uint) T; i += localSize) {
                float e = exp(rowPtr[i] - maxVal);
                rowPtr[i] = e;
                localSum += e;
            }
            sh[lid] = localSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh[lid] += sh[lid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float sumVal = sh[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float invSum = 1.0f / sumVal;
            for (uint i = lid; i < (uint) T; i += localSize) {
                rowPtr[i] *= invSum;
            }
        }

        kernel void attn_weighted_sum_causal_batched(
            device const float* scores       [[buffer(0)]], // [T, T]
            device const float* v_all        [[buffer(1)]], // [T, vRowStride]
            device float* attn_out           [[buffer(2)]], // [T, outRowStride]
            constant int& vRowStride         [[buffer(3)]],
            constant int& outRowStride       [[buffer(4)]],
            constant int& vHeadOff           [[buffer(5)]],
            constant int& outHeadOff         [[buffer(6)]],
            constant int& head_dim           [[buffer(7)]],
            constant int& T                  [[buffer(8)]],
            uint2 gid                        [[thread_position_in_grid]]) // gid.x = d, gid.y = t
        {
            const int t = (int) gid.y;
            const int d = (int) gid.x;
            if (d >= head_dim) return;

            device const float* rowScores = scores + t * T;
            float acc = 0.0f;
            for (int j = 0; j <= t; j++) {
                acc += rowScores[j] * v_all[j * vRowStride + vHeadOff + d];
            }
            attn_out[t * outRowStride + outHeadOff + d] = acc;
        }
        """;
}