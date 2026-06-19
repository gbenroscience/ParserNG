package com.github.gbenroscience.simd.turbo.tools.matrix;

/**
 * High-Performance Pure Java Quantization and Low-Level Integer Kernels.
 * Designed to maximize HotSpot C2 auto-vectorization (SLP) and Instruction-Level Parallelism (ILP)
 * without using raw Vector API dependencies.
 * * @author GBEMIRO
 */
public final class KernelsInt8 {

    // GGML Q8_0 block format: 2 bytes FP16 scale + 32 bytes INT8
    public static final int Q8_0_GROUP_SIZE = 32;
    public static final int Q8_0_BLOCK_SIZE = 2 + Q8_0_GROUP_SIZE; // 34 bytes

    /**
     * Quantize FP32 -> INT8 with per-tensor scale q = clamp(round(x / scale), -127, 127)
     */
    public static void quantize_i8(float[] x, byte[] x_q8, int len, float scale) {
        quantize_f32_to_i8(x, 0, len, x_q8, 0, scale);
    }

    /**
     * Dequantize INT8 -> FP32 with per-tensor scale
     */
    public static void dequantize_i8(byte[] x_q8, float[] x_f32, int len, float scale) {
        dequant_i8_to_f32(x_q8, 0, len, x_f32, 0, scale);
    }

    /**
     * Find absmax for dynamic quantization (with offset support)
     * Employs 4 independent accumulators to break hardware instruction dependency chains.
     */
    public static float absmax(float[] arr, int off, int len) {
        float m0 = 0.0f;
        float m1 = 0.0f;
        float m2 = 0.0f;
        float m3 = 0.0f;
        
        int i = 0;
        for (; i <= len - 4; i += 4) {
            m0 = Math.max(m0, Math.abs(arr[off + i]));
            m1 = Math.max(m1, Math.abs(arr[off + i + 1]));
            m2 = Math.max(m2, Math.abs(arr[off + i + 2]));
            m3 = Math.max(m3, Math.abs(arr[off + i + 3]));
        }
        
        float max = Math.max(Math.max(m0, m1), Math.max(m2, m3));
        
        // Clean remaining elements
        for (; i < len; i++) {
            max = Math.max(max, Math.abs(arr[off + i]));
        }
        return max;
    }

    /**
     * Find absmax for dynamic quantization
     */
    public static float absmax(float[] x, int len) {
        return absmax(x, 0, len);
    }

    /**
     * Dequantize single Q8_0 block to FP32. Used for reading KV cache during attention.
     * Fully unrolled structure allows C2 to instantly compile loop increments to direct register strides.
     */
    public static void dequantize_q8_0_block(byte[] q8_0_block, float[] out_f32) {
        short scaleBits = (short) ((q8_0_block[0] & 0xFF) | ((q8_0_block[1] & 0xFF) << 8));
        float scale = f16ToF32(scaleBits);

        // Fully unroll the 32 elements to saturate continuous execution slots
        for (int i = 0; i < Q8_0_GROUP_SIZE; i += 8) {
            out_f32[i]     = q8_0_block[2 + i]     * scale;
            out_f32[i + 1] = q8_0_block[2 + i + 1] * scale;
            out_f32[i + 2] = q8_0_block[2 + i + 2] * scale;
            out_f32[i + 3] = q8_0_block[2 + i + 3] * scale;
            out_f32[i + 4] = q8_0_block[2 + i + 4] * scale;
            out_f32[i + 5] = q8_0_block[2 + i + 5] * scale;
            out_f32[i + 6] = q8_0_block[2 + i + 6] * scale;
            out_f32[i + 7] = q8_0_block[2 + i + 7] * scale;
        }
    }

    /**
     * Convert FP16 bits to FP32. Matches ggml_fp16_to_fp32.
     */
    private static float f16ToF32(short h) {
        int bits = h & 0xFFFF;
        int sign = (bits & 0x8000) << 16;
        int exp = (bits >>> 10) & 0x1F;
        int mant = bits & 0x3FF;

        if (exp == 0) {
            if (mant == 0) {
                return Float.intBitsToFloat(sign);
            }
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
        } else if (exp == 31) {
            return Float.intBitsToFloat(sign | 0x7F800000 | (mant << 13));
        }
        exp += 127 - 15;
        return Float.intBitsToFloat(sign | (exp << 23) | (mant << 13));
    }

    /**
     * INT8 GEMV Q8_0 -> FP32 split layout 
     * Output: For each head, writes [x0,x2,x4...][x1,x3,x5...] instead of interleaved
     */
    public static void matmul_q8_0_1xN_split_opt(
            byte[] x_q8_0, byte[] W_q8_0, float[] out_f32_split,
            int qHeads, int head_dim, int K) {

        if ((head_dim & 1) != 0) {
            throw new IllegalArgumentException("head_dim must be even");
        }
        if (K % Q8_0_GROUP_SIZE != 0) {
            throw new IllegalArgumentException("K must be multiple of 32");
        }

        final int halfDim = head_dim >>> 1;
        final int numBlocks = K / Q8_0_GROUP_SIZE;
        final int B_stride = numBlocks * Q8_0_BLOCK_SIZE;

        for (int h = 0; h < qHeads; h++) {
            int headOutOff = h * head_dim;
            int evenOutOff = headOutOff;
            int oddOutOff = headOutOff + halfDim;

            for (int i = 0; i < halfDim; i++) {
                int n0 = h * head_dim + (i << 1);
                int n1 = n0 + 1;

                float acc0 = 0.0f;
                float acc1 = 0.0f;

                final int w0StrideBase = n0 * B_stride;
                final int w1StrideBase = n1 * B_stride;

                for (int b = 0; b < numBlocks; b++) {
                    int xBlockOff = b * Q8_0_BLOCK_SIZE;
                    int w0BlockOff = w0StrideBase + xBlockOff;
                    int w1BlockOff = w1StrideBase + xBlockOff;

                    short xScaleBits = (short) ((x_q8_0[xBlockOff] & 0xFF) | ((x_q8_0[xBlockOff + 1] & 0xFF) << 8));
                    float xScale = f16ToF32(xScaleBits);

                    short w0ScaleBits = (short) ((W_q8_0[w0BlockOff] & 0xFF) | ((W_q8_0[w0BlockOff + 1] & 0xFF) << 8));
                    short w1ScaleBits = (short) ((W_q8_0[w1BlockOff] & 0xFF) | ((W_q8_0[w1BlockOff + 1] & 0xFF) << 8));

                    float scale0 = xScale * f16ToF32(w0ScaleBits);
                    float scale1 = xScale * f16ToF32(w1ScaleBits);

                    // Unroll the internal 32-byte quantization block to allow pure pipeline parallelization
                    int iacc0_0 = 0, iacc0_1 = 0, iacc0_2 = 0, iacc0_3 = 0;
                    int iacc1_0 = 0, iacc1_1 = 0, iacc1_2 = 0, iacc1_3 = 0;

                    final int xBase = xBlockOff + 2;
                    final int w0Base = w0BlockOff + 2;
                    final int w1Base = w1BlockOff + 2;

                    // Step size of 8 allows full AVX2 execution block parity matching via JIT auto-vectorization
                    for (int j = 0; j < 32; j += 8) {
                        iacc0_0 += x_q8_0[xBase + j]     * W_q8_0[w0Base + j];
                        iacc0_1 += x_q8_0[xBase + j + 1] * W_q8_0[w0Base + j + 1];
                        iacc0_2 += x_q8_0[xBase + j + 2] * W_q8_0[w0Base + j + 2];
                        iacc0_3 += x_q8_0[xBase + j + 3] * W_q8_0[w0Base + j + 3];
                        iacc0_0 += x_q8_0[xBase + j + 4] * W_q8_0[w0Base + j + 4];
                        iacc0_1 += x_q8_0[xBase + j + 5] * W_q8_0[w0Base + j + 5];
                        iacc0_2 += x_q8_0[xBase + j + 6] * W_q8_0[w0Base + j + 6];
                        iacc0_3 += x_q8_0[xBase + j + 7] * W_q8_0[w0Base + j + 7];

                        iacc1_0 += x_q8_0[xBase + j]     * W_q8_0[w1Base + j];
                        iacc1_1 += x_q8_0[xBase + j + 1] * W_q8_0[w1Base + j + 1];
                        iacc1_2 += x_q8_0[xBase + j + 2] * W_q8_0[w1Base + j + 2];
                        iacc1_3 += x_q8_0[xBase + j + 3] * W_q8_0[w1Base + j + 3];
                        iacc1_0 += x_q8_0[xBase + j + 4] * W_q8_0[w1Base + j + 4];
                        iacc1_1 += x_q8_0[xBase + j + 5] * W_q8_0[w1Base + j + 5];
                        iacc1_2 += x_q8_0[xBase + j + 6] * W_q8_0[w1Base + j + 6];
                        iacc1_3 += x_q8_0[xBase + j + 7] * W_q8_0[w1Base + j + 7];
                    }

                    acc0 += (iacc0_0 + iacc0_1 + iacc0_2 + iacc0_3) * scale0;
                    acc1 += (iacc1_0 + iacc1_1 + iacc1_2 + iacc1_3) * scale1;
                }

                out_f32_split[evenOutOff + i] = acc0;
                out_f32_split[oddOutOff + i]  = acc1;
            }
        }
    }

    /**
     * Scalar fallback comparison layout logic
     */
    public static void quantize_f32_to_i8_alt(float[] src, int srcOff, int len, byte[] dst, int dstOff, float scale) {
        quantize_f32_to_i8(src, srcOff, len, dst, dstOff, scale);
    }

    /**
     * Vectorless Batched Q8 Symmetric Quantization
     */
    public static void quantize_f32_to_i8(float[] src, int srcOff, int len, byte[] dst, int dstOff, float scale) {
        if (scale == 0.0f) {
            java.util.Arrays.fill(dst, dstOff, dstOff + len, (byte) 0);
            return;
        }
        final float invScale = 1.0f / scale;
        int i = 0;

        // Unroll loop blocks to expose execution dependencies cleanly to SLP pass
        for (; i <= len - 4; i += 4) {
            float v0 = src[srcOff + i]     * invScale;
            float v1 = src[srcOff + i + 1] * invScale;
            float v2 = src[srcOff + i + 2] * invScale;
            float v3 = src[srcOff + i + 3] * invScale;

            dst[dstOff + i]     = (byte) Math.max(-127, Math.min(127, Math.round(v0)));
            dst[dstOff + i + 1] = (byte) Math.max(-127, Math.min(127, Math.round(v1)));
            dst[dstOff + i + 2] = (byte) Math.max(-127, Math.min(127, Math.round(v2)));
            dst[dstOff + i + 3] = (byte) Math.max(-127, Math.min(127, Math.round(v3)));
        }

        // Tail Remainder Processing
        for (; i < len; i++) {
            float v = src[srcOff + i] * invScale;
            dst[dstOff + i] = (byte) Math.max(-127, Math.min(127, Math.round(v)));
        }
    }

    /**
     * Pure Scalar Baseline Dequantization
     */
    public static void dequant_i8_to_f32_scalar(byte[] src, int srcOff, int len, float[] dst, int dstOff, float scale) {
        for (int i = 0; i < len; i++) {
            dst[dstOff + i] = src[srcOff + i] * scale;
        }
    }

    /**
     * High-Performance Stride-1 Dequantization Loop
     * Widens bytes directly to floats. C2 easily maps this pattern to SIMD register promotions.
     */
    public static void dequant_i8_to_f32(byte[] src, int srcOff, int len, float[] dst, int dstOff, float scale) {
        int i = 0;
        
        // Processing in unrolled blocks of 8 maximizes execution port utilization on x86/ARM cores
        for (; i <= len - 8; i += 8) {
            dst[dstOff + i]     = src[srcOff + i]     * scale;
            dst[dstOff + i + 1] = src[srcOff + i + 1] * scale;
            dst[dstOff + i + 2] = src[srcOff + i + 2] * scale;
            dst[dstOff + i + 3] = src[srcOff + i + 3] * scale;
            dst[dstOff + i + 4] = src[srcOff + i + 4] * scale;
            dst[dstOff + i + 5] = src[srcOff + i + 5] * scale;
            dst[dstOff + i + 6] = src[srcOff + i + 6] * scale;
            dst[dstOff + i + 7] = src[srcOff + i + 7] * scale;
        }

        // Tail Remainder Loop
        for (; i < len; i++) {
            dst[dstOff + i] = src[srcOff + i] * scale;
        }
    }
}