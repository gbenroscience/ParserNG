package com.github.gbenroscience.gpu.llm.core;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;
import org.junit.jupiter.api.Test;

class Fp16Test {

    private static void check(float f) {
        int mine = Fp16.floatToHalfBits(f);
        int ref = Float.floatToFloat16(f) & 0xFFFF;
        if (Float.isNaN(f)) {
            assertTrue((mine & 0x7C00) == 0x7C00 && (mine & 0x3FF) != 0, "NaN must stay NaN");
        } else {
            assertEquals(ref, mine, "float " + f + " bits 0x" + Integer.toHexString(Float.floatToRawIntBits(f)));
        }
    }

    @Test
    void edgeCases() {
        float[] v = {0f, -0f, Float.MIN_VALUE, 5.9604645e-8f, 2.9802322e-8f /*2^-25 tie->0*/, 2.9802326e-8f,
            6.1035156e-5f /*min normal*/, 6.0975552e-5f, 65504f, 65519.996f, 65520f, 65536f, 1e-8f, 1f, 1.0004883f,
            1.00048828125f /*tie*/, 1.00146484375f /*tie odd->even*/, Float.MAX_VALUE, Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY, Float.NaN, -1.5f, 3.14159265f, 0.1f};
        for (float f : v) {
            check(f);
            check(-f);
        }
    }

    @Test
    void randomBitPatterns() {
        Random r = new Random(1);
        for (int i = 0; i < 5_000_000; i++) {
            check(Float.intBitsToFloat(r.nextInt()));
        }
    }

    @Test
    void stridedSweepOfEntireFloatSpace() {
        for (long b = 0; b < (1L << 32); b += 17) {
            check(Float.intBitsToFloat((int) b));
        }
    }

    @Test
    void roundTripIsIdempotent() {
        Random r = new Random(2);
        for (int i = 0; i < 100000; i++) {
            float f = (float) (r.nextGaussian() * 100);
            float once = Fp16.roundTrip(f);
            assertEquals(once, Fp16.roundTrip(once));
        }
    }
}
