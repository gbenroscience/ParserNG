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
 * High-accuracy implementations of the Gaussian error function:
 *
 * <pre>
 *     erf(x)  = 2 / sqrt(pi) * integral(0,x) exp(-t^2) dt
 *     erfc(x) = 1 - erf(x)
 * </pre>
 *
 * <p>
 * This implementation is based on the fdlibm/Cody rational approximations.
 * It uses different minimax/rational approximations over different regions
 * of the real line in order to preserve double-precision accuracy.
 * </p>
 *
 * <p>
 * Important numerical properties:
 * </p>
 *
 * <ul>
 *     <li>Does not compute erfc(x) as {@code 1 - erf(x)} except where
 *         that operation is numerically safe.</li>
 *     <li>Handles NaN and both infinities.</li>
 *     <li>Preserves the sign of zero for erf.</li>
 *     <li>Handles subnormal arguments.</li>
 *     <li>Uses cancellation-resistant formulas around x = 0.25.</li>
 *     <li>Uses rational approximations for the tails rather than a
 *         naive asymptotic expansion.</li>
 * </ul>
 *
 * <p>
 * The underlying fdlibm algorithm is documented to achieve approximately
 * one-ulp accuracy for erf in its small-argument region, with the rational
 * approximations for the remaining regions having errors on the order of
 * 2^-59 to 2^-62 in the approximation being evaluated.
 * </p>
 *
 * @author GBEMIRO
 */
public final class CodyMath {

    private CodyMath() {
        // Utility class.
    }

    /*
     * -----------------------------------------------------------------------
     * Constants
     * -----------------------------------------------------------------------
     */

    private static final double ONE = 1.0;
    private static final double HALF = 0.5;
    private static final double TWO = 2.0;

    /*
     * Used for the large-x saturation region.
     *
     * Keeping this non-zero is intentional. It mirrors the fdlibm strategy
     * for returning a value just inside the mathematical limit rather than
     * blindly returning exactly +/-1 in every large-x erf case.
     */
    private static final double TINY = 1.0e-300;

    /*
     * c = (float)0.84506291151
     *
     * This is deliberately the value used by the Cody/fdlibm approximation,
     * rather than erf(1.0) rounded to a double.
     */
    private static final double ERX =
            8.45062911510467529297e-01;

    /*
     * -----------------------------------------------------------------------
     * erf approximation on [0, 0.84375]
     *
     * erf(x) = x + x * P(x^2) / Q(x^2)
     * -----------------------------------------------------------------------
     */

    private static final double EFX =
            1.28379167095512586316e-01;

    private static final double EFX8 =
            1.02703333676410069053e+00;

    private static final double PP0 =
            1.28379167095512558561e-01;

    private static final double PP1 =
           -3.25042107247001499370e-01;

    private static final double PP2 =
           -2.84817495755985104766e-02;

    private static final double PP3 =
           -5.77027029648944159157e-03;

    private static final double PP4 =
           -2.37630166566501626084e-05;

    private static final double QQ1 =
            3.97917223959155352819e-01;

    private static final double QQ2 =
            6.50222499887672944485e-02;

    private static final double QQ3 =
            5.08130628187576562776e-03;

    private static final double QQ4 =
            1.32494738004321644526e-04;

    private static final double QQ5 =
           -3.96022827877536812320e-06;

    /*
     * -----------------------------------------------------------------------
     * erf approximation on [0.84375, 1.25]
     *
     * s = |x| - 1
     * erf(x) = sign(x) * (ERX + P(s)/Q(s))
     * -----------------------------------------------------------------------
     */

    private static final double PA0 =
           -2.36211856075265944077e-03;

    private static final double PA1 =
            4.14856118683748331666e-01;

    private static final double PA2 =
           -3.72207876035701323847e-01;

    private static final double PA3 =
            3.18346619901161753674e-01;

    private static final double PA4 =
           -1.10894694282396677476e-01;

    private static final double PA5 =
            3.54783043256182359371e-02;

    private static final double PA6 =
           -2.16637559486879084300e-03;

    private static final double QA1 =
            1.06420880400844228286e-01;

    private static final double QA2 =
            5.40397917702171048937e-01;

    private static final double QA3 =
            7.18286544141962662868e-02;

    private static final double QA4 =
            1.26171219808761642112e-01;

    private static final double QA5 =
            1.36370839120290507362e-02;

    private static final double QA6 =
            1.19844998467991074170e-02;

    /*
     * -----------------------------------------------------------------------
     * erfc approximation on [1.25, 1/0.35]
     *
     * z = 1/x^2
     *
     * erfc(x) = exp(-x^2 - 0.5625 + R(z)/S(z)) / x
     * -----------------------------------------------------------------------
     */

    private static final double RA0 =
           -9.86494403484714822705e-03;

    private static final double RA1 =
           -6.93858572707181764372e-01;

    private static final double RA2 =
           -1.05586262253232909814e+01;

    private static final double RA3 =
           -6.23753324503260060396e+01;

    private static final double RA4 =
           -1.62396669462573470355e+02;

    private static final double RA5 =
           -1.84605092906711035994e+02;

    private static final double RA6 =
           -8.12874355063065934246e+01;

    private static final double RA7 =
           -9.81432934416914548592e+00;

    private static final double SA1 =
            1.96512716674392571292e+01;

    private static final double SA2 =
            1.37657754143519042600e+02;

    private static final double SA3 =
            4.34565877475229228821e+02;

    private static final double SA4 =
            6.45387271733267880336e+02;

    private static final double SA5 =
            4.29008140027567833386e+02;

    private static final double SA6 =
            1.08635005541779435134e+02;

    private static final double SA7 =
            6.57024977031928170135e+00;

    private static final double SA8 =
           -6.04244152148580987438e-02;

    /*
     * -----------------------------------------------------------------------
     * erfc approximation on [1/0.35, 28]
     * -----------------------------------------------------------------------
     */

    private static final double RB0 =
           -9.86494292470009928597e-03;

    private static final double RB1 =
           -7.99283237680523006574e-01;

    private static final double RB2 =
           -1.77579549177547519889e+01;

    private static final double RB3 =
           -1.60636384855821916062e+02;

    private static final double RB4 =
           -6.37566443368389627722e+02;

    private static final double RB5 =
           -1.02509513161107724954e+03;

    private static final double RB6 =
           -4.83519191608651397019e+02;

    private static final double SB1 =
            3.03380607434824582924e+01;

    private static final double SB2 =
            3.25792512996573918826e+02;

    private static final double SB3 =
            1.53672958608443695994e+03;

    private static final double SB4 =
            3.19985821950859553908e+03;

    private static final double SB5 =
            2.55305040643316442583e+03;

    private static final double SB6 =
            4.74528541206955367215e+02;

    private static final double SB7 =
           -2.24409524465858183362e+01;

    /*
     * -----------------------------------------------------------------------
     * Public API
     * -----------------------------------------------------------------------
     */

    /**
     * Computes the Gaussian error function erf(x).
     *
     * @param x argument
     * @return erf(x)
     */
    public static double erf(double x) {

        final double ax = Math.abs(x);

        /*
         * NaN.
         *
         * Math.abs(NaN) is NaN, and this test also handles infinities below
         * without relying on comparisons involving NaN.
         */
        if (Double.isNaN(x)) {
            return Double.NaN;
        }

        /*
         * +/- infinity.
         */
        if (Double.isInfinite(x)) {
            return Math.copySign(ONE, x);
        }

        /*
         * |x| < 0.84375
         */
        if (ax < 0.84375) {

            /*
             * For extremely small x:
             *
             * erf(x) ~= 2/sqrt(pi) * x
             *
             * The special formulation avoids unnecessary underflow and
             * preserves the input value's sign.
             */
            if (ax < 0x1.0p-28) {
                if (ax < Double.MIN_NORMAL) {
                    return 0.125 * (8.0 * x + EFX8 * x);
                }

                return x + EFX * x;
            }

            final double z = x * x;

            final double r =
                    PP0 + z * (
                    PP1 + z * (
                    PP2 + z * (
                    PP3 + z * PP4)));

            final double s =
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
        if (ax < 1.25) {

            final double s = ax - ONE;

            final double p =
                    PA0 + s * (
                    PA1 + s * (
                    PA2 + s * (
                    PA3 + s * (
                    PA4 + s * (
                    PA5 + s * PA6)))));

            final double q =
                    ONE + s * (
                    QA1 + s * (
                    QA2 + s * (
                    QA3 + s * (
                    QA4 + s * (
                    QA5 + s * QA6)))));

            final double result = ERX + p / q;

            return x >= 0.0 ? result : -result;
        }

        /*
         * |x| >= 6.
         *
         * At this point erf is so close to +/-1 that evaluating the
         * rational approximation is unnecessary.
         */
        if (ax >= 6.0) {
            return x >= 0.0
                    ? ONE - TINY
                    : TINY - ONE;
        }

        /*
         * Remaining interval:
         *
         * 1.25 <= |x| < 6
         *
         * Compute erfc(|x|) using the dedicated tail approximation and
         * recover erf without ever evaluating exp(-x*x) directly in a
         * badly scaled polynomial.
         */
        final double e = erfcPositive(ax);

        return x >= 0.0
                ? ONE - e
                : e - ONE;
    }

    /**
     * Computes the complementary error function erfc(x).
     *
     * <p>
     * Unlike a naive implementation, this method does not generally compute
     * {@code 1.0 - erf(x)}. That would cause catastrophic cancellation for
     * positive x when erf(x) is close to one.
     * </p>
     *
     * @param x argument
     * @return erfc(x)
     */
    public static double erfc(double x) {

        if (Double.isNaN(x)) {
            return Double.NaN;
        }

        if (Double.isInfinite(x)) {
            return x > 0.0 ? 0.0 : TWO;
        }

        final double ax = Math.abs(x);

        /*
         * |x| < 0.84375
         */
        if (ax < 0.84375) {

            /*
             * erfc(x) = 1 - x for tiny x, because the remaining
             * correction is below double precision.
             */
            if (ax < 0x1.0p-56) {
                return ONE - x;
            }

            final double z = x * x;

            final double r =
                    PP0 + z * (
                    PP1 + z * (
                    PP2 + z * (
                    PP3 + z * PP4)));

            final double s =
                    ONE + z * (
                    QQ1 + z * (
                    QQ2 + z * (
                    QQ3 + z * (
                    QQ4 + z * QQ5))));

            final double y = r / s;

            /*
             * For x < 0.25 use 1 - (x + x*y).
             *
             * For x >= 0.25 use a cancellation-resistant rearrangement.
             */
            if (x < 0.25) {
                return ONE - (x + x * y);
            }

            final double r2 = x * y;
            final double r3 = r2 + (x - HALF);

            return HALF - r3;
        }

        /*
         * 0.84375 <= |x| < 1.25
         */
        if (ax < 1.25) {

            final double s = ax - ONE;

            final double p =
                    PA0 + s * (
                    PA1 + s * (
                    PA2 + s * (
                    PA3 + s * (
                    PA4 + s * (
                    PA5 + s * PA6)))));

            final double q =
                    ONE + s * (
                    QA1 + s * (
                    QA2 + s * (
                    QA3 + s * (
                    QA4 + s * (
                    QA5 + s * QA6)))));

            final double correction = p / q;

            if (x >= 0.0) {
                return (ONE - ERX) - correction;
            }

            return ONE + (ERX + correction);
        }

        /*
         * For |x| >= 28, erfc is beyond the useful range of a normal
         * double result.
         */
        if (ax >= 28.0) {
            return x > 0.0
                    ? 0.0
                    : TWO - TINY;
        }

        /*
         * Negative large arguments:
         *
         * erfc(-x) = 2 - erfc(x)
         *
         * Once x <= -6 the correction to 2 is below the practical
         * significance required here.
         */
        if (x < 0.0 && ax >= 6.0) {
            return TWO - TINY;
        }

        final double e = erfcPositive(ax);

        return x >= 0.0
                ? e
                : TWO - e;
    }

    /*
     * -----------------------------------------------------------------------
     * Positive erfc tail.
     *
     * Valid for x >= 1.25 and x < 28.
     * -----------------------------------------------------------------------
     */

    private static double erfcPositive(double x) {

        final double z = ONE / (x * x);

        final double r;
        final double s;

        if (x < (1.0 / 0.35)) {

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
         * Split x into a high part with the low 32 bits cleared.
         *
         * This is an important part of the fdlibm evaluation. It reduces
         * rounding error in -x*x and permits the correction term to be
         * evaluated separately.
         */
        final double zHigh = highPart(x);

        /*
         * x^2 = zHigh^2 + (x-zHigh)(x+zHigh)
         *
         * Therefore:
         *
         * exp(-x^2 - 0.5625 + r/s)
         *
         * can be evaluated as two exponentials with better numerical
         * behavior.
         */
        final double correction =
                (zHigh - x) * (zHigh + x) + r / s;

        final double exponential =
                Math.exp(-zHigh * zHigh - 0.5625)
                * Math.exp(correction);

        return exponential / x;
    }

    /*
     * Returns x with the low 32 bits of its IEEE-754 representation cleared.
     *
     * This is the Java equivalent of the fdlibm:
     *
     *     z = x;
     *     __LO(z) = 0;
     *
     * The split is used only in the tail calculation to improve the
     * evaluation of -x*x.
     */
    private static double highPart(double x) {

        final long bits = Double.doubleToRawLongBits(x);

        final long high =
                bits & 0xffffffff00000000L;

        return Double.longBitsToDouble(high);
    }

    /*
     * -----------------------------------------------------------------------
     * Simple verification harness.
     * -----------------------------------------------------------------------
     */

    public static void main(String[] args) {

        final double[] values = {
                -10.0,
                -6.0,
                -3.0,
                -2.0,
                -1.0,
                -0.5,
                -0.25,
                -0.0,
                0.0,
                0.25,
                0.5,
                1.0,
                2.0,
                3.0,
                6.0,
                10.0
        };

        for (double x : values) {
            System.out.printf(
                    "x=% .6f  erf=% .17g  erfc=% .17g%n",
                    x,
                    erf(x),
                    erfc(x));
        }

        System.out.println();
        System.out.println("Special values:");
        System.out.println("erf(+INF)  = " + erf(Double.POSITIVE_INFINITY));
        System.out.println("erf(-INF)  = " + erf(Double.NEGATIVE_INFINITY));
        System.out.println("erfc(+INF) = " + erfc(Double.POSITIVE_INFINITY));
        System.out.println("erfc(-INF) = " + erfc(Double.NEGATIVE_INFINITY));
        System.out.println("erf(NaN)   = " + erf(Double.NaN));
        System.out.println("erfc(NaN)  = " + erfc(Double.NaN));
        System.out.println("erf(-0.0)  = " + erf(-0.0));
        System.out.println("erfc(-0.0) = " + erfc(-0.0));
    }
}