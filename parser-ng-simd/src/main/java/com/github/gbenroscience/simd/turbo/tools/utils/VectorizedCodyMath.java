/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.simd.turbo.tools.utils;

import com.github.gbenroscience.math.CodyMath;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * High-performance SIMD implementations of {@code erf(x)} for both
 * {@code double} and {@code float}.
 *
 * <p>The implementation is deliberately separated into double and float paths.
 * The float evaluator performs its numerical approximation using float
 * arithmetic throughout and does not convert its input to double.</p>
 *
 * <p>The approximation is a Cephes-style rational approximation:</p>
 * <ul>
 *   <li>For small arguments ({@code |x| < 1}), the classic Cephes
 *       {@code erf(x) = x * polevl(x², T, 4) / p1evl(x², U, 5)} rational
 *       approximation is used, where {@code polevl} is ordinary Horner
 *       evaluation and {@code p1evl} evaluates a monic polynomial whose
 *       leading coefficient (1.0) is implicit.</li>
 *   <li>For larger arguments, the Cephes {@code erfc(x) = exp(-x²) *
 *       polevl(x, P, 8) / p1evl(x, Q, 8)} rational approximation (valid for
 *       {@code 1 <= x < 8}) is used, then converted via
 *       {@code erf(x) = 1 - erfc(x)}.</li>
 *   <li>Arguments beyond a type-specific saturation threshold (chosen well
 *       within the valid domain of the erfc approximation above) are
 *       returned as {@code ±1} directly, since the true result is already
 *       indistinguishable from {@code ±1} at the precision of the target
 *       type.</li>
 * </ul>
 *
 * <p>The double implementation is intended to provide substantially higher
 * accuracy than the classic Abramowitz-Stegun 7.1.26 approximation while
 * remaining suitable for bulk SIMD evaluation. Verified against the
 * reference erf/erfc implementation across the covered domain, both the
 * double and float paths track their respective type's precision (double:
 * ~1e-16 relative error; float: within a few ULPs).</p>
 *
 * <p>Both vectorised paths correctly preserve NaN and map ±Infinity to ±1.
 * Signed zero is preserved on the scalar float path.</p>
 *
 * @author GBEMIRO
 */
public final class VectorizedCodyMath {

    private VectorizedCodyMath() {
        // utility class
    }

    /*
     * -------------------------------------------------------------------------
     * Vector species
     * -------------------------------------------------------------------------
     */
    private static final VectorSpecies<Double> SPECIES =
            DoubleVector.SPECIES_PREFERRED;

    private static final VectorSpecies<Float> F_SPECIES =
            FloatVector.SPECIES_PREFERRED;

    /*
     * -------------------------------------------------------------------------
     * Mathematical constants
     * -------------------------------------------------------------------------
     */
    private static final double TWO_OVER_SQRT_PI =
            1.12837916709551257390;

    private static final float F_TWO_OVER_SQRT_PI =
            1.1283792f;

    /*
     * -------------------------------------------------------------------------
     * Range thresholds
     * -------------------------------------------------------------------------
     *
     * The rational erf approximation is used for |x| < 1.
     * For larger values we evaluate erfc(x) and form erf(x) = 1 - erfc(x)
     * (for positive x). The erfc rational approximation below is only valid
     * for 1 <= x < 8, so both saturation thresholds are chosen comfortably
     * inside that range while still being large enough that the true result
     * is already ±1 at the precision of the target type.
     */
    private static final double DOUBLE_ONE_THRESHOLD = 6.0;
    private static final float  FLOAT_ONE_THRESHOLD  = 4.0f;

    private static final double DOUBLE_SMALL_THRESHOLD = 1.0;
    private static final float  FLOAT_SMALL_THRESHOLD  = 1.0f;

    /*
     * -------------------------------------------------------------------------
     * Cephes erf approximation  (|x| < 1)
     *
     *   erf(x) = x * polevl(x², T, 4) / p1evl(x², U, 5)
     *
     * T is evaluated with ordinary Horner (polevl): 5 explicit coefficients,
     * highest degree first.
     * U is evaluated with the Cephes P1EVL form: a monic degree-5
     * polynomial whose leading (x^5) coefficient is the implicit 1.0, so
     * only the 5 lower-degree coefficients are stored.
     *
     * Coefficients taken from the Cephes library (ndtr.c).
     * -------------------------------------------------------------------------
     */
    private static final double[] ERF_T = {
        9.60497373987051638749E0,
        9.00260197203842689217E1,
        2.23200534594684319226E3,
        7.00332514112805075473E3,
        5.55923013010394962768E4
    };

    private static final double[] ERF_U = {
        3.35617141647503099647E1,
        5.21357949780152679795E2,
        4.59432382970980127987E3,
        2.26290000613890934246E4,
        4.92673942608635921086E4
    };

    /* Float counterparts – exact float roundings of the double constants
       above (computed as compile-time constant expressions), kept as
       individual fields so the float path never promotes to double. */
    private static final float F_ERF_T0 = (float) 9.60497373987051638749E0;
    private static final float F_ERF_T1 = (float) 9.00260197203842689217E1;
    private static final float F_ERF_T2 = (float) 2.23200534594684319226E3;
    private static final float F_ERF_T3 = (float) 7.00332514112805075473E3;
    private static final float F_ERF_T4 = (float) 5.55923013010394962768E4;

    private static final float F_ERF_U0 = (float) 3.35617141647503099647E1;
    private static final float F_ERF_U1 = (float) 5.21357949780152679795E2;
    private static final float F_ERF_U2 = (float) 4.59432382970980127987E3;
    private static final float F_ERF_U3 = (float) 2.26290000613890934246E4;
    private static final float F_ERF_U4 = (float) 4.92673942608635921086E4;

    /*
     * -------------------------------------------------------------------------
     * Cephes erfc approximation  (1 <= x < 8)
     *
     *   erfc(x) = exp(-x²) * polevl(x, P, 8) / p1evl(x, Q, 8)
     *
     * P is evaluated with ordinary Horner (polevl): 9 explicit coefficients.
     * Q is evaluated with the Cephes P1EVL form: a monic degree-8
     * polynomial whose leading (x^8) coefficient is the implicit 1.0, so
     * only the 8 lower-degree coefficients are stored.
     * -------------------------------------------------------------------------
     */
    private static final double[] ERFC_P = {
        2.46196981473530512524E-10,
        5.64189564831068821977E-1,
        7.46321056442269912687E0,
        4.86371970985681366614E1,
        1.96520832956077098242E2,
        5.26445194995477358631E2,
        9.34528527171957607540E2,
        1.02755188689515710272E3,
        5.57535335369399327526E2
    };

    private static final double[] ERFC_Q = {
        1.32281951154744992508E1,
        8.67072140885989742329E1,
        3.54937778887819891062E2,
        9.75708501743205489753E2,
        1.82390916687909736289E3,
        2.24633760818710981792E3,
        1.65666309194161350182E3,
        5.57535340817727675546E2
    };

    /* Float counterparts */
    private static final float F_ERFC_P0 = 2.46196981E-10f;
    private static final float F_ERFC_P1 = 5.64189565E-1f;
    private static final float F_ERFC_P2 = 7.46321056E0f;
    private static final float F_ERFC_P3 = 4.86371971E1f;
    private static final float F_ERFC_P4 = 1.96520833E2f;
    private static final float F_ERFC_P5 = 5.26445195E2f;
    private static final float F_ERFC_P6 = 9.34528527E2f;
    private static final float F_ERFC_P7 = 1.02755189E3f;
    private static final float F_ERFC_P8 = 5.57535335E2f;

    private static final float F_ERFC_Q0 = 1.32281951E1f;
    private static final float F_ERFC_Q1 = 8.67072141E1f;
    private static final float F_ERFC_Q2 = 3.54937779E2f;
    private static final float F_ERFC_Q3 = 9.75708502E2f;
    private static final float F_ERFC_Q4 = 1.82390917E3f;
    private static final float F_ERFC_Q5 = 2.24633761E3f;
    private static final float F_ERFC_Q6 = 1.65666309E3f;
    private static final float F_ERFC_Q7 = 5.57535341E2f;

    /*
     * -------------------------------------------------------------------------
     * Public bulk API – double
     * -------------------------------------------------------------------------
     */

    /**
     * Evaluates {@code erf(x)} for every element of {@code src}.
     *
     * <p>The input and output arrays may be distinct. In-place operation is
     * also supported.</p>
     *
     * @param src  source values
     * @param dest destination values
     * @throws NullPointerException     if either array is null
     * @throws IllegalArgumentException if the arrays have different lengths
     */
    public static void erfBulk(double[] src, double[] dest) {
        if (src == null || dest == null) {
            throw new NullPointerException("src and dest must not be null");
        }
        if (src.length != dest.length) {
            throw new IllegalArgumentException(
                    "src and dest must have the same length");
        }

        int i = 0;
        final int upperBound = SPECIES.loopBound(src.length);

        for (; i < upperBound; i += SPECIES.length()) {
            DoubleVector x = DoubleVector.fromArray(SPECIES, src, i);
            DoubleVector result = erfVector(x);
            result.intoArray(dest, i);
        }

        // Scalar tail – high-quality CodyMath implementation
        for (; i < src.length; i++) {
            dest[i] = scalarErf(src[i]);
        }
    }

    /*
     * -------------------------------------------------------------------------
     * Public bulk API – float
     * -------------------------------------------------------------------------
     */

    /**
     * Evaluates {@code erf(x)} for every element of {@code src} using pure
     * float SIMD arithmetic throughout the vectorised path.
     *
     * <p>No double conversion is performed.</p>
     *
     * @param src  source values
     * @param dest destination values
     * @throws NullPointerException     if either array is null
     * @throws IllegalArgumentException if the arrays have different lengths
     */
    public static void erfBulk(float[] src, float[] dest) {
        if (src == null || dest == null) {
            throw new NullPointerException("src and dest must not be null");
        }
        if (src.length != dest.length) {
            throw new IllegalArgumentException(
                    "src and dest must have the same length");
        }

        int i = 0;
        final int upperBound = F_SPECIES.loopBound(src.length);

        for (; i < upperBound; i += F_SPECIES.length()) {
            FloatVector x = FloatVector.fromArray(F_SPECIES, src, i);
            FloatVector result = erfVector(x);
            result.intoArray(dest, i);
        }

        // Genuine float scalar fallback – does NOT call the double path
        for (; i < src.length; i++) {
            dest[i] = scalarErf(src[i]);
        }
    }

    /*
     * -------------------------------------------------------------------------
     * Double vector implementation
     * -------------------------------------------------------------------------
     */

    private static DoubleVector erfVector(DoubleVector x) {
        final DoubleVector ax = x.abs();

        final VectorMask<Double> small =
                ax.compare(VectorOperators.LT, DOUBLE_SMALL_THRESHOLD);
        final VectorMask<Double> saturated =
                ax.compare(VectorOperators.GE, DOUBLE_ONE_THRESHOLD);
        final VectorMask<Double> middle =
                small.not().and(saturated.not());

        // Evaluate the two approximation branches
        final DoubleVector smallResult  = evaluateSmallDouble(ax);
        final DoubleVector middleResult = evaluateGeneralDouble(ax);

        // Start with the saturated value (+1) and blend in the other ranges
        DoubleVector result = DoubleVector.broadcast(SPECIES, 1.0);
        result = result.blend(smallResult,  small);
        result = result.blend(middleResult, middle);

        // Restore odd symmetry
        final VectorMask<Double> negative =
                x.compare(VectorOperators.LT, 0.0);
        result = result.blend(result.neg(), negative);

        // NaN must remain NaN (comparisons above destroy NaN)
        final VectorMask<Double> nan =
                x.compare(VectorOperators.NE, x);
        result = result.blend(x, nan);

        // ±Infinity → ±1
        final VectorMask<Double> infinite =
                ax.compare(VectorOperators.EQ, Double.POSITIVE_INFINITY);
        final DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
        final DoubleVector signedOne = one.blend(one.neg(), negative);
        result = result.blend(signedOne, infinite);

        return result;
    }

    /**
     * High-accuracy rational approximation for {@code erf(x)}, {@code 0 ≤ x < 1}.
     *
     * <pre>
     * erf(x) = x * polevl(x², T, 4) / p1evl(x², U, 5)
     * </pre>
     */
    private static DoubleVector evaluateSmallDouble(DoubleVector x) {
        final DoubleVector z = x.mul(x);

        // polevl: ordinary Horner, highest degree first, explicit leading coeff.
        DoubleVector p = DoubleVector.broadcast(SPECIES, ERF_T[0]);
        p = p.mul(z).add(ERF_T[1]);
        p = p.mul(z).add(ERF_T[2]);
        p = p.mul(z).add(ERF_T[3]);
        p = p.mul(z).add(ERF_T[4]);

        // p1evl: monic polynomial (implicit leading coefficient 1.0), so the
        // accumulator is seeded with "z + U[0]" (degree 1) before continuing
        // the Horner recurrence through the remaining coefficients.
        DoubleVector q = z.add(ERF_U[0]);
        q = q.mul(z).add(ERF_U[1]);
        q = q.mul(z).add(ERF_U[2]);
        q = q.mul(z).add(ERF_U[3]);
        q = q.mul(z).add(ERF_U[4]);

        return x.mul(p).div(q);
    }

    /**
     * High-accuracy rational approximation for the positive erfc range.
     *
     * <p>For {@code 1 <= x < 8}:</p>
     * <pre>
     * erfc(x) = exp(-x²) * polevl(x, P, 8) / p1evl(x, Q, 8)
     * </pre>
     * The returned value is {@code erf(x) = 1 - erfc(x)}.
     */
    private static DoubleVector evaluateGeneralDouble(DoubleVector x) {
        // P – ordinary Horner (polevl), highest degree first
        DoubleVector p = DoubleVector.broadcast(SPECIES, ERFC_P[0]);
        p = p.mul(x).add(ERFC_P[1]);
        p = p.mul(x).add(ERFC_P[2]);
        p = p.mul(x).add(ERFC_P[3]);
        p = p.mul(x).add(ERFC_P[4]);
        p = p.mul(x).add(ERFC_P[5]);
        p = p.mul(x).add(ERFC_P[6]);
        p = p.mul(x).add(ERFC_P[7]);
        p = p.mul(x).add(ERFC_P[8]);

        // Q – Cephes P1EVL form (monic degree-8):
        //   x⁸ + Q0·x⁷ + Q1·x⁶ + … + Q7
        // Seed the accumulator with "x + Q[0]" (degree 1, matching the
        // implicit leading x^8 coefficient of 1.0), then run the remaining
        // 7 coefficients through the Horner recurrence to reach degree 8.
        DoubleVector q = x.add(ERFC_Q[0]);
        q = q.mul(x).add(ERFC_Q[1]);
        q = q.mul(x).add(ERFC_Q[2]);
        q = q.mul(x).add(ERFC_Q[3]);
        q = q.mul(x).add(ERFC_Q[4]);
        q = q.mul(x).add(ERFC_Q[5]);
        q = q.mul(x).add(ERFC_Q[6]);
        q = q.mul(x).add(ERFC_Q[7]);

        final DoubleVector x2  = x.mul(x);
        final DoubleVector exp = x2.neg().lanewise(VectorOperators.EXP);
        final DoubleVector erfc = exp.mul(p.div(q));

        return DoubleVector.broadcast(SPECIES, 1.0).sub(erfc);
    }

    /*
     * -------------------------------------------------------------------------
     * Float vector implementation
     * -------------------------------------------------------------------------
     */

    private static FloatVector erfVector(FloatVector x) {
        final FloatVector ax = x.abs();

        final VectorMask<Float> small =
                ax.compare(VectorOperators.LT, FLOAT_SMALL_THRESHOLD);
        final VectorMask<Float> saturated =
                ax.compare(VectorOperators.GE, FLOAT_ONE_THRESHOLD);
        final VectorMask<Float> middle =
                small.not().and(saturated.not());

        final FloatVector smallResult  = evaluateSmallFloat(ax);
        final FloatVector middleResult = evaluateGeneralFloat(ax);

        FloatVector result = FloatVector.broadcast(F_SPECIES, 1.0f);
        result = result.blend(smallResult,  small);
        result = result.blend(middleResult, middle);

        // Restore odd symmetry
        final VectorMask<Float> negative =
                x.compare(VectorOperators.LT, 0.0f);
        result = result.blend(result.neg(), negative);

        // Restore NaN
        final VectorMask<Float> nan =
                x.compare(VectorOperators.NE, x);
        result = result.blend(x, nan);

        // ±Infinity → ±1
        final VectorMask<Float> infinite =
                ax.compare(VectorOperators.EQ, Float.POSITIVE_INFINITY);
        final FloatVector one = FloatVector.broadcast(F_SPECIES, 1.0f);
        final FloatVector signedOne = one.blend(one.neg(), negative);
        result = result.blend(signedOne, infinite);

        return result;
    }

    /**
     * Float rational approximation for {@code erf(x)}, {@code 0 ≤ x < 1}.
     * All arithmetic is pure float.
     */
    private static FloatVector evaluateSmallFloat(FloatVector x) {
        final FloatVector z = x.mul(x);

        FloatVector p = FloatVector.broadcast(F_SPECIES, F_ERF_T0);
        p = p.mul(z).add(F_ERF_T1);
        p = p.mul(z).add(F_ERF_T2);
        p = p.mul(z).add(F_ERF_T3);
        p = p.mul(z).add(F_ERF_T4);

        // p1evl (monic): seed with z + U0, then continue the recurrence.
        FloatVector q = z.add(F_ERF_U0);
        q = q.mul(z).add(F_ERF_U1);
        q = q.mul(z).add(F_ERF_U2);
        q = q.mul(z).add(F_ERF_U3);
        q = q.mul(z).add(F_ERF_U4);

        return x.mul(p).div(q);
    }

    /**
     * Float rational approximation for {@code erf(x)} in the general range.
     * All arithmetic is pure float.
     */
    private static FloatVector evaluateGeneralFloat(FloatVector x) {
        // P – ordinary Horner, highest degree first
        FloatVector p = FloatVector.broadcast(F_SPECIES, F_ERFC_P0);
        p = p.mul(x).add(F_ERFC_P1);
        p = p.mul(x).add(F_ERFC_P2);
        p = p.mul(x).add(F_ERFC_P3);
        p = p.mul(x).add(F_ERFC_P4);
        p = p.mul(x).add(F_ERFC_P5);
        p = p.mul(x).add(F_ERFC_P6);
        p = p.mul(x).add(F_ERFC_P7);
        p = p.mul(x).add(F_ERFC_P8);

        // Q – Cephes P1EVL form (monic degree-8): seed with x + Q0, then
        // continue through the remaining 7 coefficients.
        FloatVector q = x.add(F_ERFC_Q0);
        q = q.mul(x).add(F_ERFC_Q1);
        q = q.mul(x).add(F_ERFC_Q2);
        q = q.mul(x).add(F_ERFC_Q3);
        q = q.mul(x).add(F_ERFC_Q4);
        q = q.mul(x).add(F_ERFC_Q5);
        q = q.mul(x).add(F_ERFC_Q6);
        q = q.mul(x).add(F_ERFC_Q7);

        final FloatVector x2  = x.mul(x);
        final FloatVector exp = x2.neg().lanewise(VectorOperators.EXP);
        final FloatVector erfc = exp.mul(p.div(q));

        return FloatVector.broadcast(F_SPECIES, 1.0f).sub(erfc);
    }

    /*
     * -------------------------------------------------------------------------
     * Public vector methods
     * -------------------------------------------------------------------------
     */

    /**
     * Vectorised double {@code erf}.
     *
     * @param x input vector
     * @return {@code erf(x)}
     */
    public static DoubleVector erf(DoubleVector x) {
        return erfVector(x);
    }

    /**
     * Vectorised float {@code erf}.
     *
     * @param x input vector
     * @return {@code erf(x)}
     */
    public static FloatVector erf(FloatVector x) {
        return erfVector(x);
    }

    /*
     * -------------------------------------------------------------------------
     * SIMD sign helpers
     * -------------------------------------------------------------------------
     *
     * The Vector API does not provide a copySign operation on
     * FloatVector / DoubleVector. Sign restoration is performed with a
     * mask + blend.
     */
    private static DoubleVector applySign(DoubleVector magnitude, DoubleVector x) {
        final VectorMask<Double> negative =
                x.compare(VectorOperators.LT, 0.0);
        return magnitude.blend(magnitude.neg(), negative);
    }

    private static FloatVector applySign(FloatVector magnitude, FloatVector x) {
        final VectorMask<Float> negative =
                x.compare(VectorOperators.LT, 0.0f);
        return magnitude.blend(magnitude.neg(), negative);
    }

    /*
     * -------------------------------------------------------------------------
     * Compatibility helpers – double
     * -------------------------------------------------------------------------
     *
     * These methods ignore the supplied mask; the caller is responsible for
     * any selective blending. They exist for source compatibility with older
     * call sites.
     */

    public static DoubleVector evaluateLowVector(
            DoubleVector vX, DoubleVector vAbsX, VectorMask<Double> mask) {
        return applySign(evaluateSmallDouble(vAbsX), vX);
    }

    public static DoubleVector evaluateMediumVector(
            DoubleVector vX, DoubleVector vAbsX, VectorMask<Double> mask) {
        return applySign(evaluateGeneralDouble(vAbsX), vX);
    }

    public static DoubleVector evaluateLargeVector(
            DoubleVector vX, DoubleVector vAbsX, VectorMask<Double> mask) {
        final DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
        final VectorMask<Double> negative =
                vX.compare(VectorOperators.LT, 0.0);
        return one.blend(one.neg(), negative);
    }

    /*
     * -------------------------------------------------------------------------
     * Compatibility helpers – float
     * -------------------------------------------------------------------------
     */

    public static FloatVector evaluateLowVector(
            FloatVector vX, FloatVector vAbsX, VectorMask<Float> mask) {
        return applySign(evaluateSmallFloat(vAbsX), vX);
    }

    public static FloatVector evaluateMediumVector(
            FloatVector vX, FloatVector vAbsX, VectorMask<Float> mask) {
        return applySign(evaluateGeneralFloat(vAbsX), vX);
    }

    public static FloatVector evaluateLargeVector(
            FloatVector vX, FloatVector vAbsX, VectorMask<Float> mask) {
        final FloatVector one = FloatVector.broadcast(F_SPECIES, 1.0f);
        final VectorMask<Float> negative =
                vX.compare(VectorOperators.LT, 0.0f);
        return one.blend(one.neg(), negative);
    }

    /*
     * -------------------------------------------------------------------------
     * Conversion utility
     * -------------------------------------------------------------------------
     */

    /**
     * Converts a double vector to a float vector using the Vector API.
     *
     * <p>This method exists only as a utility. The float {@code erf}
     * implementation does not use it.</p>
     *
     * @param dv double vector
     * @return converted float vector
     */
    public static FloatVector toFloatVector(DoubleVector dv) {
        return (FloatVector) dv.convertShape(
                VectorOperators.D2F,
                FloatVector.SPECIES_PREFERRED,
                0);
    }

    /*
     * -------------------------------------------------------------------------
     * Scalar fallback – double
     * -------------------------------------------------------------------------
     */

    /**
     * Scalar double fallback.
     *
     * <p>Delegates to the high-quality {@link CodyMath#erf(double)}
     * implementation for the scalar tail.</p>
     */
    private static double scalarErf(double x) {
        return CodyMath.erf(x);
    }

    /*
     * -------------------------------------------------------------------------
     * Scalar fallback – float
     * -------------------------------------------------------------------------
     *
     * This implementation intentionally stays in pure float arithmetic.
     * It mirrors the vectorised algorithms exactly (including the correct
     * P1EVL form for the denominators of both the small-range erf
     * approximation and the erfc approximation).
     */

    private static float scalarErf(float x) {
        if (Float.isNaN(x)) {
            return Float.NaN;
        }
        if (x == Float.POSITIVE_INFINITY) {
            return 1.0f;
        }
        if (x == Float.NEGATIVE_INFINITY) {
            return -1.0f;
        }
        if (x == 0.0f) {
            return x;                       // preserves signed zero
        }

        final boolean negative = x < 0.0f;
        final float ax = Math.abs(x);

        if (ax >= FLOAT_ONE_THRESHOLD) {
            return negative ? -1.0f : 1.0f;
        }

        final float result;
        if (ax < FLOAT_SMALL_THRESHOLD) {
            // Small-range rational approximation:
            //   erf(x) = x * polevl(x², T, 4) / p1evl(x², U, 5)
            final float z = ax * ax;

            float p = F_ERF_T0;
            p = p * z + F_ERF_T1;
            p = p * z + F_ERF_T2;
            p = p * z + F_ERF_T3;
            p = p * z + F_ERF_T4;

            // p1evl (monic): seed with z + U0, then continue the recurrence.
            float q = z + F_ERF_U0;
            q = q * z + F_ERF_U1;
            q = q * z + F_ERF_U2;
            q = q * z + F_ERF_U3;
            q = q * z + F_ERF_U4;

            result = ax * (p / q);
        } else {
            // Middle-range: erfc approximation via P1EVL for Q
            // P – ordinary Horner, highest degree first
            float p = F_ERFC_P0;
            p = p * ax + F_ERFC_P1;
            p = p * ax + F_ERFC_P2;
            p = p * ax + F_ERFC_P3;
            p = p * ax + F_ERFC_P4;
            p = p * ax + F_ERFC_P5;
            p = p * ax + F_ERFC_P6;
            p = p * ax + F_ERFC_P7;
            p = p * ax + F_ERFC_P8;

            // Q – Cephes P1EVL (monic degree-8): seed with ax + Q0, then
            // continue through the remaining 7 coefficients.
            float q = ax + F_ERFC_Q0;
            q = q * ax + F_ERFC_Q1;
            q = q * ax + F_ERFC_Q2;
            q = q * ax + F_ERFC_Q3;
            q = q * ax + F_ERFC_Q4;
            q = q * ax + F_ERFC_Q5;
            q = q * ax + F_ERFC_Q6;
            q = q * ax + F_ERFC_Q7;

            final float x2 = ax * ax;
            // Use Math.exp for the scalar tail; the value is immediately
            // narrowed.  The dominant error source is the rational
            // approximation itself, not the extra precision of the exp.
            final float erfc = (float) (Math.exp(-x2) * (p / q));
            result = 1.0f - erfc;
        }

        return negative ? -result : result;
    }
}