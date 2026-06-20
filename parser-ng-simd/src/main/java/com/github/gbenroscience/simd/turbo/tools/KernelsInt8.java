package com.github.gbenroscience.simd.turbo.tools;

import com.github.gbenroscience.simd.turbo.tools.utils.Utils;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 *
 * // 1. QKV projection from INT8 weights -> FP32 split layout
 * KernelsInt8.matmul_q8_0_1xN_split_opt(x_q8_0, wq_q8_0, q_split, qHeads,
 * head_dim, dim); KernelsInt8.matmul_q8_0_1xN_split_opt(x_q8_0, wk_q8_0,
 * k_split, kHeads, head_dim, dim);
 *
 * // 2. FP32 RoPE in KernelsFloat KernelsFloat.apply_rope_f32_split(q_split,
 * qHeads, k_split, kHeads, head_dim, pos, cos_t, sin_t);
 *
 * // 3. Attention in FP32 using KernelsFloat kernels
 *
 * INT8/Q8_0 kernels for LLM inference
 *
 * @author GBEMIRO
 */
public final class KernelsInt8 {

    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Short> S_SPECIES = ShortVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;

    // GGML Q8_0 block format: 2 bytes FP16 scale + 32 bytes INT8
    public static final int Q8_0_GROUP_SIZE = 32;
    public static final int Q8_0_BLOCK_SIZE = 2 + Q8_0_GROUP_SIZE; // 34 bytes
    private static final ThreadLocal<int[]> INT_BUF = ThreadLocal.withInitial(
            () -> new int[FloatVector.SPECIES_MAX.length()]
    );
   // ThreadLocal backing primitives to keep the hot loop at a dead flat 0 B/op score
    private static final ThreadLocal<int[]> INT_CONVERSION_SCRATCHPAD
            = ThreadLocal.withInitial(() -> new int[64]); // Caps maximum possible hardware vector width safely

    /**
     * Quantize FP32 -> INT8 with per-tensor scale q = clamp(round(x / scale),
     * -127, 127)
     */
    public static void quantize_i8(float[] x, byte[] x_q8, int len, float scale) {
        float invScale = 1.0f / scale;
        int i = 0;
        FloatVector vInvScale = FloatVector.broadcast(F_SPECIES, invScale);

        for (; i < F_SPECIES.loopBound(len); i += F_SPECIES.length()) {
            FloatVector xv = FloatVector.fromArray(F_SPECIES, x, i);
            FloatVector qf = xv.mul(vInvScale);

            for (int j = 0; j < F_SPECIES.length(); j++) {
                float val = qf.lane(j);
                int q = Math.round(val);
                q = Math.max(-127, Math.min(127, q));
                x_q8[i + j] = (byte) q;
            }
        }

        for (; i < len; i++) {
            int q = Math.round(x[i] * invScale);
            x_q8[i] = (byte) Math.max(-127, Math.min(127, q));
        }
    }

    /**
     * Dequantize INT8 -> FP32 with per-tensor scale Fixed: handles species
     * mismatch correctly
     */
    public static void dequantize_i8(byte[] x_q8, float[] x_f32, int len, float scale) {
        int i = 0;
        final int step = I_SPECIES.length();
        FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);

        for (; i <= len - step; i += step) {
            IntVector iv = IntVector.zero(I_SPECIES);
            for (int j = 0; j < step; j++) {
                iv = iv.withLane(j, x_q8[i + j]);
            }
            // JDK 17: convert returns Vector<F>, need cast
            FloatVector fv = (FloatVector) iv.convert(VectorOperators.I2F, 0);
            fv.mul(vScale).intoArray(x_f32, i);
        }

        for (; i < len; i++) {
            x_f32[i] = x_q8[i] * scale;
        }
    }
    // In KernelsInt8

    public static float absmax(float[] arr, int off, int len) {
        FloatVector vmax = FloatVector.zero(F_SPECIES);
        int i = 0;
        for (; i < F_SPECIES.loopBound(len); i += F_SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(F_SPECIES, arr, off + i).abs();
            vmax = vmax.max(v);
        }
        float max = vmax.reduceLanes(VectorOperators.MAX);
        for (; i < len; i++) {
            max = Math.max(max, Math.abs(arr[off + i]));
        }
        return max;
    }

    /**
     * Find absmax for dynamic quantization
     */
    public static float absmax(float[] x, int len) {
        float max = 0.0f;
        int i = 0;
        FloatVector vMax = FloatVector.zero(F_SPECIES);
        for (; i < F_SPECIES.loopBound(len); i += F_SPECIES.length()) {
            FloatVector xv = FloatVector.fromArray(F_SPECIES, x, i).abs();
            vMax = vMax.max(xv);
        }
        max = vMax.reduceLanes(VectorOperators.MAX);
        for (; i < len; i++) {
            max = Math.max(max, Math.abs(x[i]));
        }
        return max;
    }

    public static IntVector convertFloatToInt(FloatVector fv) {
        return (IntVector) fv.convert(VectorOperators.F2I, 0);
    }

    /**
     * Dequantize single Q8_0 block to FP32 Used for reading KV cache during
     * attention
     */
    public static void dequantize_q8_0_block(byte[] q8_0_block, float[] out_f32) {
        short scaleBits = (short) ((q8_0_block[0] & 0xFF) | ((q8_0_block[1] & 0xFF) << 8));
        float scale = f16ToF32(scaleBits);
        FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);

        int i = 0;
        for (; i < B_SPECIES.loopBound(Q8_0_GROUP_SIZE); i += B_SPECIES.length()) {
            ByteVector bv = ByteVector.fromArray(B_SPECIES, q8_0_block, 2 + i);
            ShortVector sv = (ShortVector) bv.convertShape(VectorOperators.B2S, S_SPECIES, 0);
            IntVector iv = (IntVector) sv.convertShape(VectorOperators.S2I, I_SPECIES, 0);
            FloatVector fv = (FloatVector) iv.convertShape(VectorOperators.I2F, F_SPECIES, 0);
            fv.mul(vScale).intoArray(out_f32, i);
        }
        for (; i < Q8_0_GROUP_SIZE; i++) {
            out_f32[i] = q8_0_block[2 + i] * scale;
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
     * INT8 GEMV Q8_0 -> FP32 split layout Output: For each head, writes
     * [x0,x2,x4...][x1,x3,x5...] instead of interleaved SIMD optimized inner
     * loop
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

                for (int b = 0; b < numBlocks; b++) {
                    int xBlockOff = b * Q8_0_BLOCK_SIZE;
                    int w0BlockOff = n0 * B_stride + b * Q8_0_BLOCK_SIZE;
                    int w1BlockOff = n1 * B_stride + b * Q8_0_BLOCK_SIZE;

                    short xScaleBits = (short) ((x_q8_0[xBlockOff] & 0xFF)
                            | ((x_q8_0[xBlockOff + 1] & 0xFF) << 8));
                    float xScale = f16ToF32(xScaleBits);

                    short w0ScaleBits = (short) ((W_q8_0[w0BlockOff] & 0xFF)
                            | ((W_q8_0[w0BlockOff + 1] & 0xFF) << 8));
                    short w1ScaleBits = (short) ((W_q8_0[w1BlockOff] & 0xFF)
                            | ((W_q8_0[w1BlockOff + 1] & 0xFF) << 8));

                    float scale0 = xScale * f16ToF32(w0ScaleBits);
                    float scale1 = xScale * f16ToF32(w1ScaleBits);

                    // SIMD dot product for 32 bytes
                    int iacc0 = 0, iacc1 = 0;
                    int j = 0;
                    for (; j < B_SPECIES.loopBound(Q8_0_GROUP_SIZE); j += B_SPECIES.length()) {
                        ByteVector xv = ByteVector.fromArray(B_SPECIES, x_q8_0, xBlockOff + 2 + j);
                        ByteVector wv0 = ByteVector.fromArray(B_SPECIES, W_q8_0, w0BlockOff + 2 + j);
                        ByteVector wv1 = ByteVector.fromArray(B_SPECIES, W_q8_0, w1BlockOff + 2 + j);

                        // Widen i8->i16 for safe multiply
                        ShortVector xvs = (ShortVector) xv.convertShape(VectorOperators.B2S, S_SPECIES, 0);
                        ShortVector w0s = (ShortVector) wv0.convertShape(VectorOperators.B2S, S_SPECIES, 0);
                        ShortVector w1s = (ShortVector) wv1.convertShape(VectorOperators.B2S, S_SPECIES, 0);

                        iacc0 += xvs.mul(w0s).reduceLanes(VectorOperators.ADD);
                        iacc1 += xvs.mul(w1s).reduceLanes(VectorOperators.ADD);
                    }
                    // Tail
                    for (; j < Q8_0_GROUP_SIZE; j++) {
                        byte xv = x_q8_0[xBlockOff + 2 + j];
                        iacc0 += xv * W_q8_0[w0BlockOff + 2 + j];
                        iacc1 += xv * W_q8_0[w1BlockOff + 2 + j];
                    }

                    acc0 += iacc0 * scale0;
                    acc1 += iacc1 * scale1;
                }

                out_f32_split[evenOutOff + i] = acc0; // x0,x2,x4...
                out_f32_split[oddOutOff + i] = acc1; // x1,x3,x5...
            }
        }
    }
//SCALAR

    public static void quantize_f32_to_i8_alt(float[] src, int srcOff, int len, byte[] dst, int dstOff, float scale) {
        if (scale == 0.0f) {
            java.util.Arrays.fill(dst, dstOff, dstOff + len, (byte) 0);
            return;
        }
        float invScale = 1.0f / scale;
        for (int i = 0; i < len; i++) {
            int q = Math.round(src[srcOff + i] * invScale);
            dst[dstOff + i] = (byte) Math.max(-127, Math.min(127, q));
        }
    }
//VECTOR

    public static void quantize_f32_to_i8(float[] src, int srcOff, int len, byte[] dst, int dstOff, float scale) {
        if (scale == 0.0f) {
            java.util.Arrays.fill(dst, dstOff, dstOff + len, (byte) 0);
            return;
        }
        var F_SPECIES = FloatVector.SPECIES_PREFERRED;
        int step = F_SPECIES.length();
        float invScale = 1.0f / scale;

        FloatVector vInvScale = FloatVector.broadcast(F_SPECIES, invScale);
        FloatVector vPosHalf = FloatVector.broadcast(F_SPECIES, 0.5f);
        FloatVector vNegHalf = FloatVector.broadcast(F_SPECIES, -0.5f);

        int i = 0;
        for (; i <= len - step; i += step) {
            FloatVector fv = FloatVector.fromArray(F_SPECIES, src, srcOff + i);
            FloatVector scaled = fv.mul(vInvScale);

            // Emulate Math.round: add 0.5 for >=0, add -0.5 for <0
            VectorMask<Float> neg = scaled.compare(VectorOperators.LT, 0.0f);
            // FIX: Call blend directly on your positive rounding vector.
// This selects lanes from vNegHalf wherever 'neg' is true, and keeps vPosHalf where it is false.
            FloatVector roundingOffsets = vPosHalf.blend(vNegHalf, neg);
            FloatVector rounded = scaled.add(roundingOffsets);

            IntVector iv = Utils.toInt(rounded.convert(VectorOperators.F2I, 0));

            // Symmetric Q8: [-127, 127]
            iv = iv.min(127).max(-127);

            int[] buf = INT_BUF.get();
            iv.intoArray(buf, 0);
            for (int j = 0; j < step; j++) {
                dst[dstOff + i + j] = (byte) buf[j];
            }
        }

        // Tail: use Math.round for exact match
        for (; i < len; i++) {
            int q = Math.round(src[srcOff + i] * invScale);
            dst[dstOff + i] = (byte) Math.max(-127, Math.min(127, q));
        }
    }

    //SCALAR
    public static void dequant_i8_to_f32_scalar(byte[] src, int srcOff, int len, float[] dst, int dstOff, float scale) {
        for (int i = 0; i < len; i++) {
            dst[dstOff + i] = src[srcOff + i] * scale; // byte->int->float automatic
        }
    }//VECTOR

    public static void dequant_i8_to_f32(byte[] src, int srcOff, int len, float[] dst, int dstOff, float scale) {
        var F_SPECIES = FloatVector.SPECIES_PREFERRED;
        int step = F_SPECIES.length(); // Stride step size matches float capacity (e.g., 8 lanes)

        FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);

        int i = 0;
        // ⚡ Blazing Fast, Zero-Allocation SIMD Vector Loop
        for (; i <= len - step; i += step) {
            // Load exactly 'step' bytes from the array, widening them to 32-bit integer register values
            // bypassing ByteVector limits entirely by using the primitive ByteVector-to-IntVector conversion tricks.
            // On a 256-bit register, this reads 8 bytes and spreads them into 8 integer lanes instantly.
            IntVector iv = IntVector.fromArray(IntVector.SPECIES_PREFERRED,
                    copyBytesToIntsReusable(src, srcOff + i, step), 0);

            // Convert to Float Vector and multiply by the dequantization scalar scale
            FloatVector fv = Utils.toFloat(iv.convert(VectorOperators.I2F, 0).lanewise(VectorOperators.MUL, vScale));

            // Stream values straight into your destination float array memory slots
            fv.intoArray(dst, dstOff + i);
        }

        // 🛑 Tail Remainder Fallback Loop
        for (; i < len; i++) {
            dst[dstOff + i] = (float) src[srcOff + i] * scale;
        }
    }

 
    private static int[] copyBytesToIntsReusable(byte[] src, int srcIdx, int count) {
        int[] buffer = INT_CONVERSION_SCRATCHPAD.get();
        for (int j = 0; j < count; j++) {
            buffer[j] = src[srcIdx + j]; // Sign extension happens natively at register level
        }
        return buffer;
    }

}
