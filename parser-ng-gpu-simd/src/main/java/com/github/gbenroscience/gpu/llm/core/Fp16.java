package com.github.gbenroscience.gpu.llm.core;

/**
 * IEEE-754 binary16 conversion with round-to-nearest-even. This is the exact
 * bit algorithm the OpenCL kernel uses (OpenCL 1.2 has no portable half
 * conversion without cl_khr_fp16), written in plain integer arithmetic so it
 * can be transcribed line for line into OpenCL C and checked here against
 * {@link Float#floatToFloat16(float)} (which CUDA's __float2half and Metal's
 * half() also implement).
 */
public final class Fp16 {

    private Fp16() {
    }

    public static int floatToHalfBits(float f) {
        int x = Float.floatToRawIntBits(f);
        int sign = (x >>> 16) & 0x8000;
        int exp = (x >>> 23) & 0xFF;
        int mant = x & 0x7FFFFF;

        if (exp == 0xFF) { // Inf / NaN
            return sign | 0x7C00 | (mant != 0 ? 0x200 : 0);
        }
        int e = exp - 127 + 15;
        if (e >= 31) { // overflow -> Inf
            return sign | 0x7C00;
        }
        if (e <= 0) { // half subnormal or zero
            if (e < -10) {
                return sign; // below half the smallest subnormal
            }
            mant |= 0x800000;
            int shift = 14 - e; // 14..24
            int half = mant >> shift;
            int rem = mant & ((1 << shift) - 1);
            int halfway = 1 << (shift - 1);
            if (rem > halfway || (rem == halfway && (half & 1) != 0)) {
                half++; // a carry into the exponent field is the correct result
            }
            return sign | half;
        }
        int half = (e << 10) | (mant >> 13);
        int rem = mant & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (half & 1) != 0)) {
            half++; // may carry into the exponent, up to Inf: correct
        }
        return sign | half;
    }

    public static float halfBitsToFloat(int bits) {
        return Float.float16ToFloat((short) bits);
    }

    /** The value a Q8_0 scale takes after being stored as fp16. */
    public static float roundTrip(float f) {
        return halfBitsToFloat(floatToHalfBits(f));
    }
}
