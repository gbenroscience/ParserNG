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
package com.github.gbenroscience.simdext.turbo.tools.utils;

/**
 *
 * @author GBEMIRO
 */ 

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Fast approximate vectorized square root ("fast inverse square root" style:
 * bit-hack initial guess + Newton-Raphson refinement, mapped onto
 * {@code jdk.incubator.vector}).
 *
 * <p><b>This is NOT a drop-in replacement for {@code VectorOperators.SQRT}.</b>
 * {@code VectorOperators.SQRT} compiles to the hardware SQRTPS/SQRTPD
 * instruction: correctly-rounded, IEEE-754-exact to 0.5 ULP, for every
 * input including NaN/negative/infinite. This class trades that exactness
 * for throughput on hardware where the divide/sqrt unit is not fully
 * pipelined. Whether that trade is actually a win is hardware- and
 * precision-dependent - benchmark against {@code VectorOperators.SQRT} on
 * your actual target hardware before adopting this anywhere. In particular:
 * the {@code double} path below needs roughly twice the Newton-Raphson
 * iterations of the {@code float} path to reach comparable relative
 * precision (each iteration only doubles the number of correct bits, and
 * double's 52-bit mantissa needs more doublings than float's 23-bit one to
 * catch up from the same ~4-5-correct-bit initial guess), while hardware
 * SQRTPD is only modestly slower than SQRTPS on most current
 * microarchitectures. Do the arithmetic for your own iteration count before
 * assuming the double path wins the way the float path might.
 *
 * <p>This class is intentionally standalone and opt-in - it is not wired
 * into any {@code OP_SQRT} command path. Do not substitute it for
 * {@code VectorOperators.SQRT} anywhere that is compared bit-for-bit
 * against a reference implementation (e.g. a Gandiva/Math.sqrt correctness
 * harness) without an explicit, documented tolerance for the difference.
 *
 * <p>Special values are handled to match {@link Math#sqrt(double)}
 * semantics - NaN in {@literal ->} NaN out, negative {@literal ->} NaN,
 * +Infinity {@literal ->} +Infinity, {@code +0.0}/{@code -0.0} preserved -
 * rather than left to fall out of the bit-hack, which does not handle any
 * of these correctly on its own.
 */
public final class FastVectorMath {

    private FastVectorMath() {
    }

    // ==================================================================
    // float
    // ==================================================================

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> F_ISPECIES = VectorSpecies.of(int.class, F_SPECIES.vectorShape());

    /** Lomont's refined magic constant (better than Quake's original 0x5f3759df). */
    private static final int FLOAT_MAGIC = 0x5f375a86;

    private static final FloatVector F_ONE = FloatVector.broadcast(F_SPECIES, 1.0f);
    private static final FloatVector F_HALF = FloatVector.broadcast(F_SPECIES, 0.5f);
    private static final FloatVector F_THREE_HALVES = FloatVector.broadcast(F_SPECIES, 1.5f);
    private static final FloatVector F_NAN = FloatVector.broadcast(F_SPECIES, Float.NaN);
    private static final FloatVector F_POS_INF = FloatVector.broadcast(F_SPECIES, Float.POSITIVE_INFINITY);

    /** Default iteration count for {@link #fastSqrt(FloatVector)}: ~1 ULP relative error. */
    public static final int FLOAT_DEFAULT_ITERATIONS = 2;

    /**
     * Approximate vectorized {@code sqrt(x)} for {@code float} lanes, using
     * {@link #FLOAT_DEFAULT_ITERATIONS} Newton-Raphson iterations
     * (~1 ULP relative error on finite non-negative inputs). NaN, negative,
     * and infinite inputs are special-cased to match
     * {@link Math#sqrt(double)} semantics rather than fed through the
     * bit-hack, which does not handle them correctly.
     */
    public static FloatVector fastSqrt(FloatVector x) {
        return fastSqrt(x, FLOAT_DEFAULT_ITERATIONS);
    }

    /**
     * Same as {@link #fastSqrt(FloatVector)} but with an explicit
     * Newton-Raphson iteration count, for callers who want to trade
     * accuracy against speed. {@code iterations = 0} returns the raw
     * bit-hack initial guess (~5% relative error) with no refinement.
     *
     * @throws IllegalArgumentException if iterations is negative
     */
    public static FloatVector fastSqrt(FloatVector x, int iterations) {
        if (iterations < 0) {
            throw new IllegalArgumentException("iterations must be >= 0, got " + iterations);
        }

        VectorMask<Float> nan = x.compare(VectorOperators.NE, x); // NaN != NaN is the only case this is true
        VectorMask<Float> negative = x.compare(VectorOperators.LT, 0.0f);
        VectorMask<Float> zero = x.compare(VectorOperators.EQ, 0.0f);
        VectorMask<Float> posInf = x.compare(VectorOperators.EQ, Float.POSITIVE_INFINITY);
        VectorMask<Float> special = nan.or(negative).or(zero).or(posInf);

        // Route every special-case lane through a safe, arbitrary positive
        // finite stand-in (1.0f) before the bit-hack, so NaN/negative/Inf/
        // zero bit patterns never reach the integer reinterpret below. The
        // real result for those lanes is patched back in via blend() at
        // the end - the bit-hack's output for them is simply discarded.
        FloatVector safeX = x.blend(F_ONE, special);

        IntVector xBits = safeX.reinterpretAsInts();
        IntVector magicConst = IntVector.broadcast(F_ISPECIES, FLOAT_MAGIC);
        IntVector iBits = magicConst.sub(xBits.lanewise(VectorOperators.ASHR, 1));
        FloatVector y = iBits.reinterpretAsFloats();

        FloatVector halfX = safeX.mul(F_HALF);
        for (int i = 0; i < iterations; i++) {
            FloatVector ySq = y.mul(y);
            // y *= 1.5 - halfX * ySq, via a single FMA: halfX * (-ySq) + 1.5
            y = y.mul(halfX.fma(ySq.neg(), F_THREE_HALVES));
        }

        FloatVector result = safeX.mul(y);

        result = result.blend(x, zero);      // sqrt(+0.0)=+0.0, sqrt(-0.0)=-0.0: sign preserved from x
        result = result.blend(F_NAN, nan);
        result = result.blend(F_NAN, negative);
        result = result.blend(F_POS_INF, posInf);
        return result;
    }

    // ==================================================================
    // double
    // ==================================================================

    private static final VectorSpecies<Double> D_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Long> D_LSPECIES = VectorSpecies.of(long.class, D_SPECIES.vectorShape());

    /** Double-precision analog of the float magic constant. */
    private static final long DOUBLE_MAGIC = 0x5fe6ec85e7de30daL;

    private static final DoubleVector D_ONE = DoubleVector.broadcast(D_SPECIES, 1.0);
    private static final DoubleVector D_HALF = DoubleVector.broadcast(D_SPECIES, 0.5);
    private static final DoubleVector D_THREE_HALVES = DoubleVector.broadcast(D_SPECIES, 1.5);
    private static final DoubleVector D_NAN = DoubleVector.broadcast(D_SPECIES, Double.NaN);
    private static final DoubleVector D_POS_INF = DoubleVector.broadcast(D_SPECIES, Double.POSITIVE_INFINITY);

    /**
     * Default iteration count for {@link #fastSqrt(DoubleVector)}. 4
     * iterations lands around 1e-10 relative error, not full double
     * precision (~1e-16) - see the class Javadoc on why chasing full
     * precision here is unlikely to be worth it versus
     * {@code VectorOperators.SQRT}. Raise via
     * {@link #fastSqrt(DoubleVector, int)} if you specifically need more
     * accuracy and have benchmarked that it's still a net win on your
     * hardware.
     */
    public static final int DOUBLE_DEFAULT_ITERATIONS = 4;

    /**
     * Approximate vectorized {@code sqrt(x)} for {@code double} lanes,
     * using {@link #DOUBLE_DEFAULT_ITERATIONS} Newton-Raphson iterations.
     * NaN, negative, and infinite inputs are special-cased to match
     * {@link Math#sqrt(double)} semantics rather than fed through the
     * bit-hack, which does not handle them correctly.
     *
     * <p>Benchmark this against {@code VectorOperators.SQRT} before using
     * it - see the class Javadoc; unlike the {@code float} path, this one
     * is not a safe default assumption of a win.
     */
    public static DoubleVector fastSqrt(DoubleVector x) {
        return fastSqrt(x, DOUBLE_DEFAULT_ITERATIONS);
    }

    /**
     * Same as {@link #fastSqrt(DoubleVector)} but with an explicit
     * Newton-Raphson iteration count. {@code iterations = 0} returns the
     * raw bit-hack initial guess (~3.4% relative error) with no
     * refinement.
     *
     * @throws IllegalArgumentException if iterations is negative
     */
    public static DoubleVector fastSqrt(DoubleVector x, int iterations) {
        if (iterations < 0) {
            throw new IllegalArgumentException("iterations must be >= 0, got " + iterations);
        }

        VectorMask<Double> nan = x.compare(VectorOperators.NE, x);
        VectorMask<Double> negative = x.compare(VectorOperators.LT, 0.0);
        VectorMask<Double> zero = x.compare(VectorOperators.EQ, 0.0);
        VectorMask<Double> posInf = x.compare(VectorOperators.EQ, Double.POSITIVE_INFINITY);
        VectorMask<Double> special = nan.or(negative).or(zero).or(posInf);

        DoubleVector safeX = x.blend(D_ONE, special);

        LongVector xBits = safeX.reinterpretAsLongs();
        LongVector magicConst = LongVector.broadcast(D_LSPECIES, DOUBLE_MAGIC);
        LongVector iBits = magicConst.sub(xBits.lanewise(VectorOperators.ASHR, 1));
        DoubleVector y = iBits.reinterpretAsDoubles();

        DoubleVector halfX = safeX.mul(D_HALF);
        for (int i = 0; i < iterations; i++) {
            DoubleVector ySq = y.mul(y);
            y = y.mul(halfX.fma(ySq.neg(), D_THREE_HALVES));
        }

        DoubleVector result = safeX.mul(y);

        result = result.blend(x, zero);
        result = result.blend(D_NAN, nan);
        result = result.blend(D_NAN, negative);
        result = result.blend(D_POS_INF, posInf);
        return result;
    }
}