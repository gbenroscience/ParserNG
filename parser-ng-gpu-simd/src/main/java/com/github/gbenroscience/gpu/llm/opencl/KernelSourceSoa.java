package com.github.gbenroscience.gpu.llm.opencl;

/**
 * OpenCL C for the coalesced structure-of-arrays (SoA) Q8_0 layout. Appended to {@link KernelSource#CL_SOURCE} when the
 * program is built (it reuses {@code dp4_i8}, {@code decode_fp16}, {@code encode_fp16_bits} and {@code GEMV_WG} from
 * there); if the device compiler rejects it the context rebuilds without it and the engine stays on the AoS layout.
 *
 * <h2>Why a second layout</h2>
 * GGUF stores Q8_0 as 34-byte blocks (2-byte fp16 scale + 32 int8 quants). The kernels that read that array-of-structs
 * layout cannot use aligned vector loads (the quants of block b start at byte 34*b+2) and neighbouring work-items sit
 * 68 bytes apart, so one load instruction touches many cache lines. Measured on an Intel HD 520 (see LLM-AUDIT.md),
 * the best AoS kernel reached 8.5 GB/s while an F32 kernel with unit-stride float4 loads reached 17.8 GB/s on the
 * same memory. The SoA layout separates the two streams so every load is a unit-stride, 16-byte-aligned uint4:
 *
 * <pre>
 *   weight tensor [N rows x K cols]  (one device buffer)
 *       bytes [0, N*K)                      int8 quants, row n at n*K            (16-byte aligned: K % 32 == 0)
 *       bytes [N*K, N*K + N*(K/32)*2)       fp16 scales (ushort), row n at n*(K/32)
 *
 *   activation vector / matrix of len elements (T rows flattened, len = T*K)
 *       bytes [0, len)                      int8 quants
 *       bytes [len, len + (len/32)*4)       FLOAT scales, already rounded through fp16 so that
 *                                           xScale * wScale is bit-identical to the AoS kernels' product
 * </pre>
 *
 * <h2>Kernels</h2>
 * <ul>
 *   <li>{@code quantize_activation_q8_0_soa}: same arithmetic as {@code quantize_activation_q8_0_blocks}
 *       (fp16-RNE scale, rint, clamp +-127), different stores.</li>
 *   <li>{@code q8_0_gemv_soa_l<LPR>}: {@code LPR} lanes cooperate on each output row and a 64-lane work-group covers
 *       {@code 64/LPR} rows. One work-group-wide reduction serves several rows, instead of one 64-wide tree per row
 *       (which, for K=2048, is one block of work per lane). {@code LPR} trades memory-level parallelism against
 *       reduction cost and is chosen on the real device by the autotuner.</li>
 *   <li>{@code q8_0_gemm_soa}: prefill. Same lane mapping with {@code LPR = 16}, 16 tokens per work-group so each
 *       weight block is read once per 16-token chunk.</li>
 * </ul>
 * Every int8 product and partial sum of one 32-element block is an integer below 2^24, so the float dp4 arithmetic is
 * exact; results differ from the AoS kernels only in the order of the per-block float accumulation.
 */
final class KernelSourceSoa {

    private KernelSourceSoa() {
    }

    /** Lanes-per-row variants of the decode GEMV. Each must divide the 64-lane work-group. */
    static final int[] GEMV_LPR = {8, 16, 32, 64};

    /** Lanes per row of the prefill GEMM: 4 rows x 16 lanes per 64-lane work-group. */
    static final int GEMM_LPR = 16;

    static final String KERNEL_QUANTIZE = "quantize_activation_q8_0_soa";
    static final String KERNEL_GEMM = "q8_0_gemm_soa";

    static String gemvName(int lpr) {
        return "q8_0_gemv_soa_l" + lpr;
    }

    private static final String COMMON = """

        // ======================= SoA Q8_0 (see KernelSourceSoa) =======================

        // exact: sum of 16 int8*int8 products (|sum| <= 16*127*127, every partial an integer < 2^24)
        inline float soa_dp16(uint4 a, uint4 b) {
            return (dp4_i8(a.x, b.x) + dp4_i8(a.y, b.y)) + (dp4_i8(a.z, b.z) + dp4_i8(a.w, b.w));
        }

        inline float soa_scale(ushort s) {
            return decode_fp16((uchar) (s & 0xFFu), (uchar) (s >> 8));
        }

        __kernel void quantize_activation_q8_0_soa(
            __global const float* x,
            __global uchar* out,
            const int len)
        {
            __local float sh_quant[32];
            const int block = get_group_id(0);
            const int lane = get_local_id(0); // 0..31

            float v = x[block * Q8_0_GROUP_SIZE + lane];
            sh_quant[lane] = fabs(v);
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = 16; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    sh_quant[lane] = fmax(sh_quant[lane], sh_quant[lane + stride]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const float absmax = sh_quant[0];
            const float scale = absmax / 127.0f;
            const float invScale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            int q = (int) rint(v * invScale);
            q = max(-127, min(127, q));
            out[block * Q8_0_GROUP_SIZE + lane] = (uchar) (char) q;

            if (lane == 0) {
                const ushort bits = encode_fp16_bits(scale);
                ((__global float*) (out + len))[block] = decode_fp16((uchar) (bits & 0xFFu), (uchar) ((bits >> 8) & 0xFFu));
            }
        }
        """;

    private static final String GEMV_TEMPLATE = """

        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void q8_0_gemv_soa_l@LPR@(
            __global const uchar* X,
            __global const uchar* W,
            __global float* out,
            const int N,
            const int K)
        {
            __local float part[GEMV_WG];
            const int nb = K >> 5;
            const int lid = (int) get_local_id(0);
            const int sub = lid & (@LPR@ - 1);
            const int n = (int) get_group_id(0) * (GEMV_WG / @LPR@) + lid / @LPR@;
            float acc = 0.0f;
            if (n < N) {
                __global const uint4* wq = (__global const uint4*) (W + (size_t) n * (size_t) K);
                __global const ushort* ws = ((__global const ushort*) (W + (size_t) N * (size_t) K)) + (size_t) n * (size_t) nb;
                __global const uint4* xq = (__global const uint4*) X;
                __global const float* xs = (__global const float*) (X + K);
                #pragma unroll 4
                for (int b = sub; b < nb; b += @LPR@) {
                    const uint4 w0 = wq[2 * b];
                    const uint4 w1 = wq[2 * b + 1];
                    const uint4 x0 = xq[2 * b];
                    const uint4 x1 = xq[2 * b + 1];
                    const float d = soa_dp16(w0, x0) + soa_dp16(w1, x1);
                    acc += d * (xs[b] * soa_scale(ws[b]));
                }
            }
            part[lid] = acc;
            barrier(CLK_LOCAL_MEM_FENCE);
        @REDUCE@
        }
        """;

    /** One barrier, then the first lane of each row adds its few partials (cheap for small LPR). */
    private static final String REDUCE_SEQUENTIAL = """
            if (sub == 0 && n < N) {
                float s = 0.0f;
                for (int j = 0; j < @LPR@; j++) {
                    s += part[lid + j];
                }
                out[n] = s;
            }""";

    /** log2(LPR) barriers; every lane reaches every barrier (the trip count is a compile-time constant). */
    private static final String REDUCE_TREE = """
            for (int stride = @LPR@ / 2; stride > 0; stride >>= 1) {
                if (sub < stride) {
                    part[lid] += part[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sub == 0 && n < N) {
                out[n] = part[lid];
            }""";

    private static final String GEMM_TEMPLATE = """

        #define SOA_GEMM_LPR @GEMM_LPR@
        #define SOA_GEMM_RPW (GEMV_WG / SOA_GEMM_LPR)

        // Prefill: X = SoA activations of T*K elements, W = SoA weights [N, K], out [T, N].
        // work-group (gx, gy) = rows gx*RPW .. +RPW-1 for tokens gy*GEMM_TCH .. +GEMM_TCH-1.
        __kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
        void q8_0_gemm_soa(
            __global const uchar* X,
            __global const uchar* W,
            __global float* out,
            const int T,
            const int N,
            const int K)
        {
            __local float part[GEMM_TCH * GEMV_WG];
            const int nb = K >> 5;
            const int lid = (int) get_local_id(0);
            const int sub = lid & (SOA_GEMM_LPR - 1);
            const int rowInWg = lid / SOA_GEMM_LPR;
            const int n = (int) get_group_id(0) * SOA_GEMM_RPW + rowInWg;
            const int t0 = (int) get_group_id(1) * GEMM_TCH;
            const int tn = min(GEMM_TCH, T - t0);
            const size_t xLen = (size_t) T * (size_t) K;

            float acc[GEMM_TCH];
            #pragma unroll
            for (int t = 0; t < GEMM_TCH; t++) {
                acc[t] = 0.0f;
            }
            if (n < N) {
                __global const uint4* wq = (__global const uint4*) (W + (size_t) n * (size_t) K);
                __global const ushort* ws = ((__global const ushort*) (W + (size_t) N * (size_t) K)) + (size_t) n * (size_t) nb;
                __global const float* xs = (__global const float*) (X + xLen);
                for (int b = sub; b < nb; b += SOA_GEMM_LPR) {
                    const uint4 w0 = wq[2 * b];
                    const uint4 w1 = wq[2 * b + 1];
                    const float wsc = soa_scale(ws[b]);
                    #pragma unroll
                    for (int t = 0; t < GEMM_TCH; t++) {
                        if (t < tn) {
                            __global const uint4* xq = (__global const uint4*) (X + (size_t) (t0 + t) * (size_t) K);
                            const float d = soa_dp16(w0, xq[2 * b]) + soa_dp16(w1, xq[2 * b + 1]);
                            acc[t] += d * (xs[(size_t) (t0 + t) * (size_t) nb + b] * wsc);
                        }
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < GEMM_TCH; t++) {
                part[t * GEMV_WG + lid] = acc[t];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int idx = lid; idx < SOA_GEMM_RPW * GEMM_TCH; idx += GEMV_WG) {
                const int t = idx / SOA_GEMM_RPW;
                const int r = idx % SOA_GEMM_RPW;
                const int nn = (int) get_group_id(0) * SOA_GEMM_RPW + r;
                if (t < tn && nn < N) {
                    float s = 0.0f;
                    for (int j = 0; j < SOA_GEMM_LPR; j++) {
                        s += part[t * GEMV_WG + r * SOA_GEMM_LPR + j];
                    }
                    out[(size_t) (t0 + t) * (size_t) N + nn] = s;
                }
            }
        }
        """;

    /** The complete SoA source (append to KernelSource.CL_SOURCE). */
    static final String SOURCE = build();

    private static String build() {
        StringBuilder sb = new StringBuilder(COMMON);
        for (int lpr : GEMV_LPR) {
            String reduce = lpr <= 16 ? REDUCE_SEQUENTIAL : REDUCE_TREE;
            sb.append(GEMV_TEMPLATE.replace("@REDUCE@", reduce).replace("@LPR@", Integer.toString(lpr)));
        }
        sb.append(GEMM_TEMPLATE.replace("@GEMM_LPR@", Integer.toString(GEMM_LPR)));
        return sb.toString();
    }
}
