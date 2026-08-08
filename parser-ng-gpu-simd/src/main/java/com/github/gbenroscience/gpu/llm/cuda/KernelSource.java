package com.github.gbenroscience.gpu.llm.cuda;

/**
 * CUDA kernel source for the Llama-style decoder -- v2. Extends the
 * decode-only kernel set (see the per-kernel javadocs below for the 13
 * kernels carried over unchanged) with:
 *
 *   1. GeLU / GeGLU FFN activations (gelu_activate, geglu_activate),
 *      alongside the existing swiglu_activate -- selectable per model via
 *      GpuLlamaLayerCuda.Config.activationType.
 *   2. A real BATCHED PREFILL path: q8_0_gemm_tiled / f32_gemm_tiled
 *      (T rows at once instead of T sequential GEMV launches),
 *      rmsnorm_partial_sumsq_rows / rmsnorm_apply_rows (per-row
 *      normalization, T rows batched), rope_apply_pairwise_rows (per-row
 *      position), and a causal-masked batched attention triple
 *      (attn_scores_causal_batched / softmax_inplace_rows /
 *      attn_weighted_sum_causal_batched).
 *
 * WHAT "BATCHED" BUYS YOU, CONCRETELY: the decode-path kernels
 * (q8_0_gemv_split etc.) parallelize over N output columns for ONE row.
 * Processing a T-token prompt by calling them T times means T sequential
 * kernel launches, each only exposing N-wide parallelism, plus T-1 extra
 * round trips through the launch/sync machinery. The _tiled/_rows/_batched
 * kernels below instead expose T*N (or T*T for attention) parallelism to
 * the GPU in ONE launch -- the actual mechanism that makes prefill fast.
 *
 * WHAT THIS IS NOT: cuBLAS-grade GEMM. q8_0_gemm_tiled and f32_gemm_tiled
 * use a straightforward one-thread-per-output-element scheme (2D grid:
 * blockIdx.y = row t, blockIdx.x/threadIdx.x = column n) with no
 * multi-level shared-memory blocking, no register tiling, no tensor-core
 * path. Every thread computing a given (t, n) re-reads the same x-row
 * bytes as every other thread with the same t -- on real hardware this is
 * a broadcast read (all threads in a warp/block requesting the same
 * global address collapse to one transaction), so it is not naive-bad,
 * but it is not cuBLAS-competitive either. A tiled/blocked GEMM with
 * shared-memory-resident weight and activation tiles is the natural next
 * optimization pass and is NOT attempted here -- flagged explicitly
 * rather than silently left out.
 *
 * Attention here is batched but still UNFUSED (three kernel launches:
 * scores, softmax, weighted-sum) and still one dispatch triple PER HEAD
 * (grid.y = query row t within that launch) -- not a flash-attention-style
 * single fused kernel. Causal masking is done by writing -DEVICE_FLT_MAX
 * for j > t in attn_scores_causal_batched and letting softmax's exp()
 * naturally zero those out, rather than a separate mask kernel or
 * skipping the j > t work at the launch-config level.
 *
 * UNVERIFIED, same standing caveat as every kernel file in this
 * codebase: no CUDA GPU, driver, or NVRTC toolchain were available while
 * writing this. Carefully traced against the decode kernels' already-
 * established algorithms and extended by the same patterns (bounds
 * checks, shared-memory tree reductions, Q8_0 block decode) -- but
 * treat every new kernel here as unrun until diffed against known-good
 * per-layer activations, same bar as the rest of this file set.
 */
public final class KernelSource {

    private KernelSource() {
    }

    // ---- decode-path kernels (unchanged names/signatures from v1) ----
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

    // ---- FFN activation alternatives (new) ----
    public static final String KERNEL_GELU_ACTIVATE = "gelu_activate";
    public static final String KERNEL_GEGLU_ACTIVATE = "geglu_activate";

    // ---- batched prefill kernels (new) ----
    public static final String KERNEL_Q8_0_GEMM_TILED = "q8_0_gemm_tiled";
    public static final String KERNEL_F32_GEMM_TILED = "f32_gemm_tiled";
    public static final String KERNEL_RMSNORM_PARTIAL_SUMSQ_ROWS = "rmsnorm_partial_sumsq_rows";
    public static final String KERNEL_RMSNORM_APPLY_ROWS = "rmsnorm_apply_rows";
    public static final String KERNEL_ROPE_APPLY_PAIRWISE_ROWS = "rope_apply_pairwise_rows";
    public static final String KERNEL_ATTN_SCORES_CAUSAL_BATCHED = "attn_scores_causal_batched";
    public static final String KERNEL_SOFTMAX_INPLACE_ROWS = "softmax_inplace_rows";
    public static final String KERNEL_ATTN_WEIGHTED_SUM_CAUSAL_BATCHED = "attn_weighted_sum_causal_batched";

    /** Block size used for both RMSNorm partial-sum kernels' local reduction. Must match host dispatch. */
    public static final int RMSNORM_WORKGROUP_SIZE = 256;

    /** Fixed block size for quantize_activation_q8_0_blocks -- one thread per element in a 32-wide Q8_0 group. */
    public static final int QUANTIZE_BLOCK_SIZE = 32;

    /** Default block size for the simple bound-checked 1D kernels (GEMVs, RoPE, elementwise ops). */
    public static final int DEFAULT_BLOCK_SIZE = 256;

    /** Default column-tile block size for the batched GEMM kernels (blockDim.x; blockIdx.y iterates rows). */
    public static final int GEMM_TILE_N = 128;

    public static final String CUDA_SOURCE = """
        #include <cuda_fp16.h>

        #define Q8_0_GROUP_SIZE 32
        #define Q8_0_BLOCK_SIZE 34

        // NVRTC's device-code standard library doesn't reliably expose
        // <cfloat> -- inline the IEEE-754 single-precision max directly.
        #define DEVICE_FLT_MAX 3.402823466e+38f

        __device__ __forceinline__ float decode_fp16(unsigned char lo, unsigned char hi) {
            unsigned short bits = (unsigned short) lo | ((unsigned short) hi << 8);
            return __half2float(__ushort_as_half(bits));
        }

        __device__ __forceinline__ float gpu_sigmoid_f(float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        // Exact-erf GeLU, matching the textbook formula this codebase's
        // interpreter kernels (CudaKernelSource.gpu_gelu_f) already use --
        // NOT whatever a specific model's training-time approximation
        // used; diff against reference activations before trusting GeLU-
        // FFN outputs, same caveat CudaExpressionBridge already carries
        // for its own GELU/GEGLU opcodes.
        __device__ __forceinline__ float gpu_gelu_f(float x) {
            return 0.5f * x * (1.0f + erff(x * 0.70710678f));
        }

        // =====================================================================
        // ===================== DECODE-PATH KERNELS (v1, unchanged) =========
        // =====================================================================

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

        extern "C" __global__ void q8_0_gemv_split(
            const unsigned char* x_q8_0,
            const unsigned char* W_q8_0,
            float* out_f32_split,
            const int qHeads,
            const int head_dim,
            const int K)
        {
            const int halfDim = head_dim >> 1;
            const int gid = blockIdx.x * blockDim.x + threadIdx.x;
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

        extern "C" __global__ void attn_scores(
            const float* q_all_heads,
            const float* k_cache_f32,
            float* scores,
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

        extern __shared__ float sh_softmax[];
        extern "C" __global__ void softmax_inplace(
            float* scores,
            const int len)
        {
            const int lid = threadIdx.x;
            const int localSize = blockDim.x;

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

            float invSum = 1.0f / sumVal;
            for (int i = lid; i < len; i += localSize) {
                scores[i] *= invSum;
            }
        }

        extern "C" __global__ void attn_weighted_sum(
            const float* scores,
            const float* v_cache_f32,
            float* attn_out_all_heads,
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

        extern "C" __global__ void residual_add(
            float* x,
            const float* y,
            const int len)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= len) return;
            x[i] += y[i];
        }

        // NOTE (fixed): B is [N, K] row-major -- N=out_features rows,
        // K=in_features cols, matching GGUF's native Linear-weight layout.
        // The earlier version read B as [K, N] (B[k*N+n]) -- silently
        // transposed for wo (K==N==dim, no shape check could catch it) and
        // for the LM head (K=dim, N=vocab, still in-bounds, still wrong).
        extern "C" __global__ void f32_gemv(
            const float* a,
            const float* B,
            float* out,
            const int K,
            const int N)
        {
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            if (n >= N) return;

            const int rowOff = n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += a[k] * B[rowOff + k];
            }
            out[n] = acc;
        }

        // =====================================================================
        // ===================== FFN ACTIVATION ALTERNATIVES (new) ===========
        // =====================================================================

        /*
         * gelu_activate: out[h] = gelu(gate[h]) -- for UNGATED GeLU FFNs
         * (down(gelu(up_proj(x))), single projection, no separate gate/up
         * split). Use when Config.activationType == GELU.
         */
        extern "C" __global__ void gelu_activate(
            const float* gate,
            float* out,
            const int hidden)
        {
            const int h = blockIdx.x * blockDim.x + threadIdx.x;
            if (h >= hidden) return;
            out[h] = gpu_gelu_f(gate[h]);
        }

        /*
         * geglu_activate: out[h] = gelu(gate[h]) * up[h] -- GATED GeLU,
         * same shape as swiglu_activate but with GeLU instead of the
         * SiLU/sigmoid gate. Use when Config.activationType == GEGLU.
         */
        extern "C" __global__ void geglu_activate(
            const float* gate,
            const float* up,
            float* out,
            const int hidden)
        {
            const int h = blockIdx.x * blockDim.x + threadIdx.x;
            if (h >= hidden) return;
            out[h] = gpu_gelu_f(gate[h]) * up[h];
        }

        // =====================================================================
        // ===================== BATCHED PREFILL KERNELS (new) ===============
        // =====================================================================

        extern "C" __global__ void q8_0_gemm_tiled(
            const unsigned char* X_q8_0, // [T, numBlocks*34]
            const unsigned char* W_q8_0, // [N, numBlocks*34]
            float* out,                  // [T, N]
            const int T,
            const int N,
            const int K)
        {
            const int t = blockIdx.y;
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
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
                    signed char xv = (signed char) X_q8_0[xBlockOff + 2 + j];
                    signed char wv = (signed char) W_q8_0[wBlockOff + 2 + j];
                    iacc += (int) xv * (int) wv;
                }
                acc += iacc * scale;
            }
            out[t * N + n] = acc;
        }

        // NOTE (fixed): B is [N, K] row-major, matching GGUF's native
        // Linear-weight layout -- same fix and rationale as f32_gemv above.
        extern "C" __global__ void f32_gemm_tiled(
            const float* A, // [T, K]
            const float* B, // [N, K]
            float* out,     // [T, N]
            const int T,
            const int K,
            const int N)
        {
            const int t = blockIdx.y;
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            if (t >= T || n >= N) return;

            const int bRowOff = n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[t * K + k] * B[bRowOff + k];
            }
            out[t * N + n] = acc;
        }

        extern __shared__ float sh_rms_rows[];
        extern "C" __global__ void rmsnorm_partial_sumsq_rows(
            const float* x,       // [T, features]
            float* partials,      // [T, numGroups]
            const int features,
            const int numGroups)
        {
            const int row = blockIdx.y;
            const int groupId = blockIdx.x;
            const int lid = threadIdx.x;
            const int localSize = blockDim.x;
            const int idx = groupId * localSize + lid;

            float v = (idx < features) ? x[row * features + idx] : 0.0f;
            sh_rms_rows[lid] = v * v;
            __syncthreads();

            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_rms_rows[lid] += sh_rms_rows[lid + stride];
                }
                __syncthreads();
            }

            if (lid == 0) {
                partials[row * numGroups + groupId] = sh_rms_rows[0];
            }
        }

        extern "C" __global__ void rmsnorm_apply_rows(
            const float* x,      // [T, features]
            const float* gamma,  // [features]
            float* out,          // [T, features]
            const float* rms,    // [T]
            const int features)
        {
            const int row = blockIdx.y;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= features) return;
            out[row * features + i] = x[row * features + i] * rms[row] * gamma[i];
        }

        /*
         * rope_apply_pairwise_rows: batched RoPE for the ADJACENT-PAIR
         * layout q8_0_gemm_tiled naturally produces (plain contiguous
         * output: dims [2i, 2i+1] sit next to each other within each
         * head's head_dim span), NOT the even/odd-SPLIT layout the
         * decode-path's q8_0_gemv_split emits. This is a deliberate
         * layout difference from rope_apply_split, not an inconsistency:
         * the split layout was specifically an artifact of
         * q8_0_gemv_split's own output pairing (a RoPE-adjacent
         * optimization tied to that ONE kernel); q8_0_gemm_tiled has no
         * such pairing, so its RoPE pass rotates the natural adjacent
         * pairs directly instead. Each row t has its OWN absolute
         * sequence position (positions[t] = startPos + t during
         * prefill), unlike the decode path's single shared position.
         */
        extern "C" __global__ void rope_apply_pairwise_rows(
            float* buf,             // [T, heads*head_dim], PLAIN contiguous layout
            const float* cos_table, // [max_seq, halfDim]
            const float* sin_table,
            const int heads,
            const int head_dim,
            const int* positions)   // [T] absolute sequence position per row
        {
            const int halfDim = head_dim >> 1;
            const int row = blockIdx.y;
            const int gid = blockIdx.x * blockDim.x + threadIdx.x;
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

        extern "C" __global__ void attn_scores_causal_batched(
            const float* q_all,   // [T, qRowStride]
            const float* k_all,   // [T, kRowStride]
            float* scores,        // [T, T]
            const int qRowStride,
            const int kRowStride,
            const int qHeadOff,
            const int kHeadOff,
            const int head_dim,
            const int T,
            const float rsqrt_d)
        {
            const int t = blockIdx.y;
            const int j = blockIdx.x * blockDim.x + threadIdx.x;
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

        extern __shared__ float sh_softmax_rows[];
        extern "C" __global__ void softmax_inplace_rows(
            float* scores, // [T, T]
            const int T)
        {
            const int row = blockIdx.x;
            const int lid = threadIdx.x;
            const int localSize = blockDim.x;
            float* rowPtr = scores + row * T;

            float localMax = -DEVICE_FLT_MAX;
            for (int i = lid; i < T; i += localSize) {
                localMax = fmaxf(localMax, rowPtr[i]);
            }
            sh_softmax_rows[lid] = localMax;
            __syncthreads();
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_softmax_rows[lid] = fmaxf(sh_softmax_rows[lid], sh_softmax_rows[lid + stride]);
                }
                __syncthreads();
            }
            float maxVal = sh_softmax_rows[0];
            __syncthreads();

            float localSum = 0.0f;
            for (int i = lid; i < T; i += localSize) {
                float e = expf(rowPtr[i] - maxVal);
                rowPtr[i] = e;
                localSum += e;
            }
            sh_softmax_rows[lid] = localSum;
            __syncthreads();
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    sh_softmax_rows[lid] += sh_softmax_rows[lid + stride];
                }
                __syncthreads();
            }
            float sumVal = sh_softmax_rows[0];
            __syncthreads();

            float invSum = 1.0f / sumVal;
            for (int i = lid; i < T; i += localSize) {
                rowPtr[i] *= invSum;
            }
        }

        extern "C" __global__ void attn_weighted_sum_causal_batched(
            const float* scores,  // [T, T]
            const float* v_all,   // [T, vRowStride]
            float* attn_out,      // [T, outRowStride]
            const int vRowStride,
            const int outRowStride,
            const int vHeadOff,
            const int outHeadOff,
            const int head_dim,
            const int T)
        {
            const int t = blockIdx.y;
            const int d = blockIdx.x * blockDim.x + threadIdx.x;
            if (d >= head_dim) return;

            const float* rowScores = scores + t * T;
            float acc = 0.0f;
            for (int j = 0; j <= t; j++) {
                acc += rowScores[j] * v_all[j * vRowStride + vHeadOff + d];
            }
            attn_out[t * outRowStride + outHeadOff + d] = acc;
        }
        """;
}