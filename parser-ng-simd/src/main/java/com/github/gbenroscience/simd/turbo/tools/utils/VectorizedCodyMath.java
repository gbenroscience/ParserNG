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
package com.github.gbenroscience.simd.turbo.tools.utils;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.math.CodyMath;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorMask;

public final class VectorizedCodyMath {

    // Automatically picks the widest instruction set available (AVX-2, AVX-512, or ARM Neon)
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private static final double THRESHOLD_LOW = 0.46875;
    private static final double THRESHOLD_HIGH = 4.0;
    private static final double MAX_X = 26.543; // erfc underflows to 0.0 past this

    /**
     * SIMD-vectorized bulk evaluation of erf(x) over flat arrays.
     * @param src
     * @param dest
     */
    public static void erfBulk(double[] src, double[] dest) {
        int i = 0;
        int upperBound = SPECIES.loopBound(src.length);

        // Main SIMD Loop
        for (; i < upperBound; i += SPECIES.length()) {
            DoubleVector vX = DoubleVector.fromArray(SPECIES, src, i);
            DoubleVector vAbsX = vX.abs();

            // 1. Evaluate Interval Low: 0 <= |x| <= 0.46875
            VectorMask<Double> mLow = vAbsX.compare(VectorOperators.LE, THRESHOLD_LOW);
            DoubleVector vResLow = evaluateLowVector(vX, vAbsX, mLow);

            // 2. Evaluate Interval Medium: 0.46875 < |x| <= 4.0
            VectorMask<Double> mMed = vAbsX.compare(VectorOperators.GT, THRESHOLD_LOW)
                    .and(vAbsX.compare(VectorOperators.LE, THRESHOLD_HIGH));
            DoubleVector vResMed = evaluateMediumVector(vX, vAbsX, mMed);

            // 3. Evaluate Interval Large: |x| > 4.0
            VectorMask<Double> mHigh = vAbsX.compare(VectorOperators.GT, THRESHOLD_HIGH);
            DoubleVector vResHigh = evaluateLargeVector(vX, vAbsX, mHigh);

            // SIMD Blend: Merge lane results based on interval masks
            DoubleVector vRes = vResLow.blend(vResMed, mMed).blend(vResHigh, mHigh);

            vRes.intoArray(dest, i);
        }

        // Tail Cleanup Loop (handles remaining elements if array length isn't a multiple of vector width)
        for (; i < src.length; i++) {
            dest[i] = scalarErf(src[i]);
        }
    }

    private static DoubleVector evaluateLowVector(DoubleVector vX, DoubleVector vAbsX, VectorMask<Double> mask) {
        // Safe cross-version check for empty lane configurations
        if (!mask.anyTrue()) {
            return DoubleVector.broadcast(SPECIES, 0.0);
        }

        DoubleVector xSq = vAbsX.mul(vAbsX);

        DoubleVector p = xSq.mul(0.260194122534674).add(30.59022585250011)
                .mul(xSq).add(573.9507736045833)
                .mul(xSq).add(2801.752391065013)
                .mul(xSq).add(3204.677458505002);

        DoubleVector q = xSq.mul(1.0).add(159.0884090976454)
                .mul(xSq).add(1422.080683811422)
                .mul(xSq).add(4423.613442045816)
                .mul(xSq).add(3204.677458506958);

        return vX.mul(p.div(q));
    }

    public static DoubleVector evaluateMediumVector(DoubleVector vX, DoubleVector vAbsX, VectorMask<Double> mask) {
        if (!mask.anyTrue()) {
            return DoubleVector.broadcast(SPECIES, 0.0);
        }

        DoubleVector p = vAbsX.mul(0.05895007051304052).add(3.829044036164835)
                .mul(vAbsX).add(43.18793405786769)
                .mul(vAbsX).add(157.3068460515812)
                .mul(vAbsX).add(214.6001532057584)
                .mul(vAbsX).add(106.1664295623469)
                .mul(vAbsX).add(14.5302830141757);

        DoubleVector q = vAbsX.mul(1.0).add(17.3459738020609)
                .mul(vAbsX).add(127.8650310266291)
                .mul(vAbsX).add(314.141639017465)
                .mul(vAbsX).add(309.7712496731671)
                .mul(vAbsX).add(122.2982420286843)
                .mul(vAbsX).add(14.53028300724806);

        // --- Emulate Math.floor(x * 16.0) / 16.0 using IEEE 754 Bit Alignment ---
        DoubleVector y = vAbsX.mul(16.0);
        DoubleVector magic = DoubleVector.broadcast(SPECIES, 4503599627370496.0); // 2^52

        // Nearest integer rounding via hardware mantissa shifting
        DoubleVector roundY = y.add(magic).sub(magic);

        // If it rounded up instead of down, subtract 1.0 to get the floor value
        VectorMask<Double> roundedUp = roundY.compare(VectorOperators.GT, y);
        DoubleVector correction = DoubleVector.broadcast(SPECIES, 0.0)
                .blend(DoubleVector.broadcast(SPECIES, 1.0), roundedUp);

        DoubleVector floorX = roundY.sub(correction).div(16.0);
        // ------------------------------------------------------------------------

        DoubleVector exp1 = floorX.neg().lanewise(VectorOperators.EXP);
        DoubleVector exp2 = vAbsX.mul(vAbsX).sub(floorX).neg().lanewise(VectorOperators.EXP);

        DoubleVector erfcVal = exp1.mul(exp2).mul(p.div(q));
        DoubleVector intermediate = DoubleVector.broadcast(SPECIES, 1.0).sub(erfcVal);

        // Emulate COPYSIGN
        VectorMask<Double> isNegative = vX.compare(VectorOperators.LT, 0.0);
        return intermediate.blend(intermediate.neg(), isNegative);
    }

    public static DoubleVector evaluateLargeVector(DoubleVector vX, DoubleVector vAbsX, VectorMask<Double> mask) {
        if (!mask.anyTrue()) {
            return DoubleVector.broadcast(SPECIES, 0.0);
        }

        VectorMask<Double> mOverflow = vAbsX.compare(VectorOperators.GT, MAX_X);
        DoubleVector xSqInv = DoubleVector.broadcast(SPECIES, 1.0).div(vAbsX.mul(vAbsX));

        DoubleVector p = xSqInv.mul(0.0736713390314174).add(0.745582913335534)
                .mul(xSqInv).add(1.541607549923533)
                .mul(xSqInv).add(0.910514304077531)
                .mul(xSqInv).add(0.1350648667813508);

        DoubleVector q = xSqInv.mul(0.01282616526077372).add(0.2447524102450513)
                .mul(xSqInv).add(1.137319453745207)
                .mul(xSqInv).add(1.84085398366255)
                .mul(xSqInv).add(0.9785461628585647)
                .mul(xSqInv).add(0.1350648965456337);

        DoubleVector expTerm = vAbsX.mul(vAbsX).neg().lanewise(VectorOperators.EXP);
        DoubleVector erfcVal = expTerm.div(vAbsX).mul(p.div(q));

        // Force underflow lanes cleanly to 0.0
        erfcVal = erfcVal.blend(DoubleVector.broadcast(SPECIES, 0.0), mOverflow);

        DoubleVector intermediate = DoubleVector.broadcast(SPECIES, 1.0).sub(erfcVal);

        // Emulate COPYSIGN
        VectorMask<Double> isNegative = vX.compare(VectorOperators.LT, 0.0);
        return intermediate.blend(intermediate.neg(), isNegative);
    }

    private static double scalarErf(double x) {
        return CodyMath.erf(x);
    }
}
