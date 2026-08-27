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
package com.github.gbenroscience.math;

/**
 * Single-precision Cody/fdlibm-style implementations of erf(x) and erfc(x).
 *
 * <p>This class is intended for FP32 numerical workloads where maintaining
 * the computation in float is important, such as SIMD and ML evaluation.</p>
 *
 * <p>The implementation uses piecewise rational approximations rather than
 * evaluating erfc(x) as {@code 1 - erf(x)} throughout the domain.</p>
 *
 * @author GBEMIRO
 */
public final class CodyMathFloat {

    private CodyMathFloat() {
    }

    private static final float ONE  = 1.0f;
    private static final float HALF = 0.5f;
    private static final float TWO  = 2.0f;

    /*
     * erf(x) = x + x * P(x^2) / Q(x^2)
     *
     * |x| < 0.84375
     */
    private static final float EFX =
            1.28379166e-1f;

    private static final float EFX8 =
            1.02703333f;

    private static final float PP0 =
            1.28379166e-1f;

    private static final float PP1 =
           -3.25042105e-1f;

    private static final float PP2 =
           -2.84817493e-2f;

    private static final float PP3 =
           -5.77027025e-3f;

    private static final float PP4 =
           -2.37630167e-5f;

    private static final float QQ1 =
            3.97917211e-1f;

    private static final float QQ2 =
            6.50222525e-2f;

    private static final float QQ3 =
            5.08130649e-3f;

    private static final float QQ4 =
            1.32494737e-4f;

    private static final float QQ5 =
           -3.96022822e-6f;

    /*
     * erf(x) = ERX + P(s)/Q(s)
     *
     * 0.84375 <= |x| < 1.25
     *
     * s = |x| - 1
     */
    private static final float ERX =
            8.45062912e-1f;

    private static final float PA0 =
           -2.36211856e-3f;

    private static final float PA1 =
            4.14856106e-1f;

    private static final float PA2 =
           -3.72207874e-1f;

    private static final float PA3 =
            3.18346620e-1f;

    private static final float PA4 =
           -1.10894695e-1f;

    private static final float PA5 =
            3.54783051e-2f;

    private static final float PA6 =
           -2.16637560e-3f;

    private static final float QA1 =
            1.06420882e-1f;

    private static final float QA2 =
            5.40397942e-1f;

    private static final float QA3 =
            7.18286559e-2f;

    private static final float QA4 =
            1.26171216e-1f;

    private static final float QA5 =
            1.36370841e-2f;

    private static final float QA6 =
            1.19844997e-2f;

    /*
     * erfc(x) = exp(-x*x - 0.5625 + R(z)/S(z)) / x
     *
     * 1.25 <= x < 1/0.35
     *
     * z = 1/x^2
     */
    private static final float RA0 =
           -9.86494422e-3f;

    private static final float RA1 =
           -6.93858564e-1f;

    private static final float RA2 =
           -1.05586262e1f;

    private static final float RA3 =
           -6.23753319e1f;

    private static final float RA4 =
           -1.62396667e2f;

    private static final float RA5 =
           -1.84605087e2f;

    private static final float RA6 =
           -8.12874374e1f;

    private static final float RA7 =
           -9.81432915f;

    private static final float SA1 =
            1.96512718e1f;

    private static final float SA2 =
            1.37657745e2f;

    private static final float SA3 =
            4.34565857e2f;

    private static final float SA4 =
            6.45387268e2f;

    private static final float SA5 =
            4.29008118e2f;

    private static final float SA6 =
            1.08635010e2f;

    private static final float SA7 =
            6.57024956f;

    private static final float SA8 =
           -6.04244120e-2f;

    /*
     * erfc(x), x >= 1/0.35
     */
    private static final float RB0 =
           -9.86494292e-3f;

    private static final float RB1 =
           -7.99283266e-1f;

    private static final float RB2 =
           -1.77579551e1f;

    private static final float RB3 =
           -1.60636383e2f;

    private static final float RB4 =
           -6.37566467e2f;

    private static final float RB5 =
           -1.02509509e3f;

    private static final float RB6 =
           -4.83519165e2f;

    private static final float SB1 =
            3.03380604e1f;

    private static final float SB2 =
            3.25792511e2f;

    private static final float SB3 =
            1.53672961e3f;

    private static final float SB4 =
            3.19985815e3f;

    private static final float SB5 =
            2.55305054e3f;

    private static final float SB6 =
            4.74528503e2f;

    /*
     * ---------------------------------------------------------------------
     * erf(float)
     * ---------------------------------------------------------------------
     */

    /**
     * Computes erf(x) in single precision.
     *
     * @param x argument
     * @return erf(x)
     */
    public static float erf(float x) {

        if (Float.isNaN(x)) {
            return Float.NaN;
        }

        if (Float.isInfinite(x)) {
            return Math.copySign(ONE, x);
        }

        final float ax = Math.abs(x);

        /*
         * |x| < 0.84375
         */
        if (ax < 0.84375f) {

            /*
             * Tiny x:
             *
             * erf(x) ~= 2/sqrt(pi) * x
             *
             * Below 2^-28 the correction cannot affect a float result.
             */
            if (ax < 0x1.0p-28f) {
                return x + EFX * x;
            }

            final float z = x * x;

            final float r =
                    PP0 + z * (
                    PP1 + z * (
                    PP2 + z * (
                    PP3 + z * PP4)));

            final float s =
                    ONE + z * (
                    QQ1 + z * (
                    QQ2 + z * (
                    QQ3 + z * (
                    QQ4 + z * QQ5))));

            return x + x * (r / s);
        }

        /*
         * 0.84375 <= |x| < 1.25
         */
        if (ax < 1.25f) {

            final float s = ax - ONE;

            final float p =
                    PA0 + s * (
                    PA1 + s * (
                    PA2 + s * (
                    PA3 + s * (
                    PA4 + s * (
                    PA5 + s * PA6)))));

            final float q =
                    ONE + s * (
                    QA1 + s * (
                    QA2 + s * (
                    QA3 + s * (
                    QA4 + s * (
                    QA5 + s * QA6)))));

            final float result = ERX + p / q;

            return x >= 0.0f ? result : -result;
        }

        /*
         * At x >= 6, erf(x) rounds to +/-1.0f.
         */
        if (ax >= 6.0f) {
            return Math.copySign(ONE, x);
        }

        /*
         * 1.25 <= |x| < 6
         */
        final float e = erfcPositive(ax);

        return x >= 0.0f
                ? ONE - e
                : e - ONE;
    }

    /*
     * ---------------------------------------------------------------------
     * erfc(float)
     * ---------------------------------------------------------------------
     */

    /**
     * Computes erfc(x) in single precision.
     *
     * <p>The implementation avoids {@code 1 - erf(x)} where cancellation
     * would cause unnecessary loss of precision.</p>
     *
     * @param x argument
     * @return erfc(x)
     */
    public static float erfc(float x) {

        if (Float.isNaN(x)) {
            return Float.NaN;
        }

        if (Float.isInfinite(x)) {
            return x > 0.0f ? 0.0f : TWO;
        }

        final float ax = Math.abs(x);

        /*
         * |x| < 0.84375
         */
        if (ax < 0.84375f) {

            /*
             * For sufficiently small x:
             *
             * erfc(x) = 1 - x + O(x^3)
             */
            if (ax < 0x1.0p-28f) {
                return ONE - x;
            }

            final float z = x * x;

            final float r =
                    PP0 + z * (
                    PP1 + z * (
                    PP2 + z * (
                    PP3 + z * PP4)));

            final float s =
                    ONE + z * (
                    QQ1 + z * (
                    QQ2 + z * (
                    QQ3 + z * (
                    QQ4 + z * QQ5))));

            final float y = r / s;

            /*
             * For small x, direct subtraction is sufficiently accurate.
             */
            if (x < 0.25f) {
                return ONE - (x + x * y);
            }

            /*
             * Cancellation-resistant formulation.
             */
            final float r2 = x * y;
            final float r3 = r2 + (x - HALF);

            return HALF - r3;
        }

        /*
         * 0.84375 <= |x| < 1.25
         */
        if (ax < 1.25f) {

            final float s = ax - ONE;

            final float p =
                    PA0 + s * (
                    PA1 + s * (
                    PA2 + s * (
                    PA3 + s * (
                    PA4 + s * (
                    PA5 + s * PA6)))));

            final float q =
                    ONE + s * (
                    QA1 + s * (
                    QA2 + s * (
                    QA3 + s * (
                    QA4 + s * (
                    QA5 + s * QA6)))));

            final float correction = p / q;

            if (x >= 0.0f) {
                return (ONE - ERX) - correction;
            }

            return ONE + (ERX + correction);
        }

        /*
         * Positive x >= 6:
         *
         * erfc(6) is already far below float precision.
         */
        if (x >= 6.0f) {
            return 0.0f;
        }

        /*
         * Negative x <= -6:
         *
         * erfc(x) = 2 - erfc(-x)
         *
         * erfc(-x) is below float precision.
         */
        if (x <= -6.0f) {
            return TWO;
        }

        /*
         * 1.25 <= |x| < 6
         */
        final float e = erfcPositive(ax);

        return x >= 0.0f
                ? e
                : TWO - e;
    }

    /*
     * ---------------------------------------------------------------------
     * Positive erfc tail.
     * ---------------------------------------------------------------------
     */

    private static float erfcPositive(float x) {

        final float z = ONE / (x * x);

        final float r;
        final float s;

        if (x < (1.0f / 0.35f)) {

            r =
                    RA0 + z * (
                    RA1 + z * (
                    RA2 + z * (
                    RA3 + z * (
                    RA4 + z * (
                    RA5 + z * (
                    RA6 + z * RA7))))));

            s =
                    ONE + z * (
                    SA1 + z * (
                    SA2 + z * (
                    SA3 + z * (
                    SA4 + z * (
                    SA5 + z * (
                    SA6 + z * (
                    SA7 + z * SA8)))))));
        } else {

            r =
                    RB0 + z * (
                    RB1 + z * (
                    RB2 + z * (
                    RB3 + z * (
                    RB4 + z * (
                    RB5 + z * RB6)))));

            s =
                    ONE + z * (
                    SB1 + z * (
                    SB2 + z * (
                    SB3 + z * (
                    SB4 + z * (
                    SB5 + z * SB6)))));
        }

        /*
         * Float equivalent of the fdlibm high-word split.
         *
         * Clearing the low 16 bits gives a useful high part while retaining
         * enough precision for the FP32 calculation.
         */
        final int bits = Float.floatToRawIntBits(x);

        final float zHigh =
                Float.intBitsToFloat(bits & 0xffff0000);

        /*
         * Evaluate:
         *
         * exp(-x*x - 0.5625 + r/s)
         *
         * in split form to reduce rounding error.
         */
        final float correction =
                (zHigh - x) * (zHigh + x)
                + r / s;

        final float exponential =
                (float) (
                        Math.exp(
                                -(double) zHigh * zHigh
                                - 0.5625
                                + correction));

        return exponential / x;
    }
}