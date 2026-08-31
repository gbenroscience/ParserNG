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

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.simd.turbo.tools.utils.VectorizedCodyMath;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 *
 * @author GBEMIRO
 */
public final class VectorMathF {

    private VectorMathF() {
    }

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    public static int VECTOR_THRESHOLD = 256;

    // Angle conversions
    private static final float DEG_TO_RAD = (float) (Math.PI / 180.0);
    private static final float RAD_TO_DEG = (float) (180.0 / Math.PI);
    private static final float GRAD_TO_RAD = (float) (Math.PI / 200.0);
    private static final float RAD_TO_GRAD = (float) (200.0 / Math.PI);

    private static final FloatVector V_DEG_TO_RAD = FloatVector.broadcast(F_SPECIES, DEG_TO_RAD);
    private static final FloatVector V_RAD_TO_DEG = FloatVector.broadcast(F_SPECIES, RAD_TO_DEG);
    private static final FloatVector V_GRAD_TO_RAD = FloatVector.broadcast(F_SPECIES, GRAD_TO_RAD);
    private static final FloatVector V_RAD_TO_GRAD = FloatVector.broadcast(F_SPECIES, RAD_TO_GRAD);

    // Core constants
    private static final FloatVector V_ONE = FloatVector.broadcast(F_SPECIES, 1.0f);
    private static final FloatVector V_NEG_ONE = FloatVector.broadcast(F_SPECIES, -1.0f);
    private static final FloatVector V_HALF = FloatVector.broadcast(F_SPECIES, 0.5f);
    private static final FloatVector V_HALF_PI = FloatVector.broadcast(F_SPECIES, (float) (Math.PI / 2.0));
    private static final FloatVector V_NEG_HALF_PI = FloatVector.broadcast(F_SPECIES, (float) (-Math.PI / 2.0));
    private static final FloatVector V_NAN = FloatVector.broadcast(F_SPECIES, Float.NaN);
    private static final FloatVector ZERO = FloatVector.broadcast(F_SPECIES, 0.0f);

    private static final float THRESHOLD_LOW = 0.46875f;
    private static final float THRESHOLD_HIGH = 4.0f;

    // ========================================================================
    // NO-LAMBDA DIRECT OPERATIONS
    // ========================================================================
    // Radian
    public static void sin(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.SIN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.sin(s[base + i]);
        }
    }

    public static void cos(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.COS)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.cos(s[base + i]);
        }
    }

    public static void tan(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.TAN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.tan(s[base + i]);
        }
    }

    // Degree
    public static void sinDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_DEG_TO_RAD)
                    .lanewise(VectorOperators.SIN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.sin(Math.toRadians(s[base + i]));
        }
    }

    public static void cosDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_DEG_TO_RAD)
                    .lanewise(VectorOperators.COS)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.cos(Math.toRadians(s[base + i]));
        }
    }

    public static void tanDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_DEG_TO_RAD)
                    .lanewise(VectorOperators.TAN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.tan(Math.toRadians(s[base + i]));
        }
    }

    // Grad
    public static void sinGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_GRAD_TO_RAD)
                    .lanewise(VectorOperators.SIN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.sin(s[base + i] * GRAD_TO_RAD);
        }
    }

    public static void cosGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_GRAD_TO_RAD)
                    .lanewise(VectorOperators.COS)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.cos(s[base + i] * GRAD_TO_RAD);
        }
    }

    public static void tanGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_GRAD_TO_RAD)
                    .lanewise(VectorOperators.TAN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.tan(s[base + i] * GRAD_TO_RAD);
        }
    }

    // ===================== Reciprocal Trigonometric =====================
    // Radian
    public static void sec(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.COS))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) (1.0 / Math.cos(s[base + i]));
        }
    }

    public static void csc(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.SIN))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) (1.0 / Math.sin(s[base + i]));
        }
    }

    public static void cot(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(F_SPECIES, s, base + i);
            v.lanewise(VectorOperators.COS)
                    .div(v.lanewise(VectorOperators.SIN))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) (1.0 / Math.tan(s[base + i]));
        }
    }

    // Degree
    public static void secDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_DEG_TO_RAD)
                    .lanewise(VectorOperators.COS))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) (1.0 / Math.cos(Math.toRadians(s[base + i])));
        }
    }

    public static void cscDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_DEG_TO_RAD)
                    .lanewise(VectorOperators.SIN))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) (1.0 / Math.sin(Math.toRadians(s[base + i])));
        }
    }

    public static void cotDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_DEG_TO_RAD);
            v.lanewise(VectorOperators.COS)
                    .div(v.lanewise(VectorOperators.SIN))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) (1.0 / Math.tan(Math.toRadians(s[base + i])));
        }
    }

    // Grad
    public static void secGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_GRAD_TO_RAD)
                    .lanewise(VectorOperators.COS))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) (1.0 / Math.cos(s[base + i] * GRAD_TO_RAD));
        }
    }

    public static void cscGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_GRAD_TO_RAD)
                    .lanewise(VectorOperators.SIN))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) (1.0 / Math.sin(s[base + i] * GRAD_TO_RAD));
        }
    }

    public static void cotGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(F_SPECIES, s, base + i)
                    .mul(V_GRAD_TO_RAD);
            v.lanewise(VectorOperators.COS)
                    .div(v.lanewise(VectorOperators.SIN))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) (1.0 / Math.tan(s[base + i] * GRAD_TO_RAD));
        }
    }

    // ===================== Inverse Trigonometric =====================
    // Radian
    public static void asin(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.ASIN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.asin(s[base + i]);
        }
    }

    public static void acos(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.ACOS)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.acos(s[base + i]);
        }
    }

    public static void atan(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.ATAN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.atan(s[base + i]);
        }
    }

    // Degree
    public static void asinDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.ASIN)
                    .mul(V_RAD_TO_DEG)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.toDegrees(Math.asin(s[base + i]));
        }
    }

    public static void acosDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.ACOS)
                    .mul(V_RAD_TO_DEG)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.toDegrees(Math.acos(s[base + i]));
        }
    }

    public static void atanDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.ATAN)
                    .mul(V_RAD_TO_DEG)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.toDegrees(Math.atan(s[base + i]));
        }
    }

    // Grad
    public static void asinGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.ASIN)
                    .mul(V_RAD_TO_GRAD)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.asin(s[base + i]) * RAD_TO_GRAD;
        }
    }

    public static void acosGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.ACOS)
                    .mul(V_RAD_TO_GRAD)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.acos(s[base + i]) * RAD_TO_GRAD;
        }
    }

    public static void atanGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.ATAN)
                    .mul(V_RAD_TO_GRAD)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.atan(s[base + i]) * RAD_TO_GRAD;
        }
    }

    // ===================== Inverse Reciprocal Trigonometric =====================
    // Radian
    public static void acsc(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .lanewise(VectorOperators.ASIN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.asin(1.0 / s[base + i]);
        }
    }

    public static void asec(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .lanewise(VectorOperators.ACOS)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.acos(1.0 / s[base + i]);
        }
    }

    public static void acot(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .lanewise(VectorOperators.ATAN)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.atan(1.0 / s[base + i]);
        }
    }

    // Degree
    public static void acscDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .lanewise(VectorOperators.ASIN)
                    .mul(V_RAD_TO_DEG)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.toDegrees(Math.asin(1.0 / s[base + i]));
        }
    }

    public static void asecDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .lanewise(VectorOperators.ACOS)
                    .mul(V_RAD_TO_DEG)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.toDegrees(Math.acos(1.0 / s[base + i]));
        }
    }

    public static void acotDeg(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .lanewise(VectorOperators.ATAN)
                    .mul(V_RAD_TO_DEG)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.toDegrees(Math.atan(1.0 / s[base + i]));
        }
    }

    // Grad
    public static void acscGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .lanewise(VectorOperators.ASIN)
                    .mul(V_RAD_TO_GRAD)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.asin(1.0 / s[base + i]) * RAD_TO_GRAD;
        }
    }

    public static void asecGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .lanewise(VectorOperators.ACOS)
                    .mul(V_RAD_TO_GRAD)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.acos(1.0 / s[base + i]) * RAD_TO_GRAD;
        }
    }

    public static void acotGrad(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            V_ONE.div(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .lanewise(VectorOperators.ATAN)
                    .mul(V_RAD_TO_GRAD)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.atan(1.0 / s[base + i]) * RAD_TO_GRAD;
        }
    }

    // ===================== Hyperbolic =====================
    public static void sinh(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.SINH)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.sinh(s[base + i]);
        }
    }

    public static void cosh(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.COSH)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.cosh(s[base + i]);
        }
    }

    public static void tanh(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.TANH)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.tanh(s[base + i]);
        }
    }

    // ===================== Inverse Hyperbolic =====================
    public static void asinh(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            vectorAsinhImpl(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.log(s[base + i] + Math.sqrt(s[base + i] * s[base + i] + 1.0));
        }
    }

    public static void acosh(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            vectorAcoshImpl(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            double x = s[base + i];
            s[base + i] = (float) (x < 1.0 ? Double.NaN : Math.log(x + Math.sqrt(x * x - 1.0)));
        }
    }

    public static void atanh(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            vectorAtanhImpl(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            double x = s[base + i];
            s[base + i] = (float) (0.5 * Math.log((1.0 + x) / (1.0 - x)));
        }
    }

    public static void asech(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            vectorAsechImpl(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            double x = s[base + i];
            s[base + i] = (float) ((x <= 0.0 || x > 1.0) ? Double.NaN : Math.log((1.0 / x) + Math.sqrt((1.0 / (x * x)) - 1.0)));
        }
    }

    public static void acsch(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            vectorAcschImpl(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            double x = s[base + i];
            s[base + i] = (float) (x == 0.0 ? Double.NaN : Math.log((1.0 / x) + Math.sqrt((1.0 / (x * x)) + 1.0)));
        }
    }

    public static void acoth(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            vectorAcothImpl(FloatVector.fromArray(F_SPECIES, s, base + i))
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            double x = s[base + i];
            s[base + i] = (float) ((float) Math.abs(x) <= 1.0 ? Double.NaN : 0.5 * Math.log((1.0 + (1.0 / x)) / (1.0 - (1.0 / x))));
        }
    }

    public static void sqrt(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.SQRT)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.sqrt(s[base + i]);
        }
    }

    public static void cbrt(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.CBRT)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.cbrt(s[base + i]);
        }
    }

    // ===================== Exponential and Logarithmic =====================
    public static void exp(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.EXP)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.exp(s[base + i]);
        }
    }

    public static void ln(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.LOG)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.log(s[base + i]);
        }
    }

    public static void log10(int base, int n, float[] s) {
        int i = 0;
        int limit = F_SPECIES.loopBound(n);
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + i)
                    .lanewise(VectorOperators.LOG10)
                    .intoArray(s, base + i);
        }
        for (; i < n; i++) {
            s[base + i] = (float) Math.log10(s[base + i]);
        }
    }

    private static boolean isExponentUniform(float[] scratch, int offset, int n) {
        if (n <= 1) {
            return true;
        }

        final float first = scratch[offset];
        if (Float.isNaN(first)) {
            // All must be NaN
            final int vl = F_SPECIES.length();
            int i = 0;
            int bound = F_SPECIES.loopBound(n);
            for (; i < bound; i += vl) {
                FloatVector v = FloatVector.fromArray(F_SPECIES, scratch, offset + i);
                if (v.compare(VectorOperators.EQ, v).anyTrue()) {
                    return false;
                }
            }
            int remaining = n - i;
            if (remaining > 0) {
                var mask = F_SPECIES.indexInRange(0, remaining);
                FloatVector v = FloatVector.fromArray(F_SPECIES, scratch, offset + i, mask);
                if (v.compare(VectorOperators.EQ, v, mask).anyTrue()) {
                    return false;
                }
            }
            return true;
        }

        final FloatVector target = FloatVector.broadcast(F_SPECIES, first);
        final int vl = F_SPECIES.length();
        int i = 0;
        int bound = F_SPECIES.loopBound(n);

        for (; i < bound; i += vl) {
            FloatVector v = FloatVector.fromArray(F_SPECIES, scratch, offset + i);
            if (v.compare(VectorOperators.NE, target).anyTrue()) {
                return false;
            }
        }

        int remaining = n - i;
        if (remaining > 0) {
            var mask = F_SPECIES.indexInRange(0, remaining);
            FloatVector v = FloatVector.fromArray(F_SPECIES, scratch, offset + i, mask);
            if (v.compare(VectorOperators.NE, target, mask).anyTrue()) {
                return false;
            }
        }
        return true;
    }

    public static void evaluateVariableExponent(float[] base, int bOffset, float[] exp, int eOffset,
            float[] dest, int dOffset, int n) {
        if (n <= 0) {
            return;
        }

        int i = 0;
        final int limit = F_SPECIES.loopBound(n);

        // === 1. Core Vector Loop: exp(y * ln(x)) ===
        for (; i < limit; i += F_SPECIES.length()) {
            FloatVector vBase = FloatVector.fromArray(F_SPECIES, base, bOffset + i);
            FloatVector vExp = FloatVector.fromArray(F_SPECIES, exp, eOffset + i);

            // Execute algebraic transcendental transformation
            FloatVector log = vBase.lanewise(VectorOperators.LOG);
            FloatVector scaled = log.mul(vExp);
            scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
        }

        // === 2. Masked Tail Pass ===
        int remaining = n - i;
        if (remaining > 0) {
            var mask = F_SPECIES.indexInRange(0, remaining);
            FloatVector vBase = FloatVector.fromArray(F_SPECIES, base, bOffset + i, mask);
            FloatVector vExp = FloatVector.fromArray(F_SPECIES, exp, eOffset + i, mask);

            // Apply masks to intermediate operators to maintain lane isolation
            FloatVector log = vBase.lanewise(VectorOperators.LOG, mask);
            FloatVector scaled = log.mul(vExp, mask);
            FloatVector res = scaled.lanewise(VectorOperators.EXP, mask);

            res.intoArray(dest, dOffset + i, mask);
        }
    }

    public static void executePowerBlended(float[] scratch, int baseOffset, int expOffset, int n) {
        if (n <= 0) {
            return;
        }

        if (isExponentUniform(scratch, expOffset, n)) {
            float uniformExp = scratch[expOffset];

            if (uniformExp == 0.5) {
               VectorTranscendentals.evaluateNative(scratch, baseOffset, scratch, baseOffset, n, VectorOperators.SQRT);
                return;
            }
            if (uniformExp == 2.0) {
                computeSquare(scratch, baseOffset, scratch, baseOffset, n);
                return;
            }
            if (uniformExp == 3.0) {
                computeCube(scratch, baseOffset, scratch, baseOffset, n);
                return;
            }
            if (uniformExp == 4.0) {
                computeFourthPower(scratch, baseOffset, scratch, baseOffset, n);
                return;
            }

            // Isolated fallback for uniform constants
            evaluateUniformExponent(scratch, baseOffset, uniformExp, scratch, baseOffset, n);
        } else {
            // Isolated fallback for variable exponents
            evaluateVariableExponent(scratch, baseOffset, scratch, expOffset, scratch, baseOffset, n);
        }
    }

// ==========================================
// Isolated Fast-Path Micro-Methods (EA Safe)
// ==========================================
    private static void computeSquare(float[] src, int srcOff, float[] dest, int destOff, int n) {
        int k = 0;
        final int limit = F_SPECIES.loopBound(n);
        final int vl = F_SPECIES.length();

        for (; k < limit; k += vl) {
            FloatVector v = FloatVector.fromArray(F_SPECIES, src, srcOff + k);
            v.mul(v).intoArray(dest, destOff + k);
        }

        int remaining = n - k;
        if (remaining > 0) {
            var mask = F_SPECIES.indexInRange(0, remaining);
            FloatVector v = FloatVector.fromArray(F_SPECIES, src, srcOff + k, mask);
            v.mul(v).intoArray(dest, destOff + k, mask);
        }
    }

    private static void computeCube(float[] src, int srcOff, float[] dest, int destOff, int n) {
        int k = 0;
        final int limit = F_SPECIES.loopBound(n);
        final int vl = F_SPECIES.length();

        for (; k < limit; k += vl) {
            FloatVector v = FloatVector.fromArray(F_SPECIES, src, srcOff + k);
            v.mul(v).mul(v).intoArray(dest, destOff + k);
        }

        int remaining = n - k;
        if (remaining > 0) {
            var mask = F_SPECIES.indexInRange(0, remaining);
            FloatVector v = FloatVector.fromArray(F_SPECIES, src, srcOff + k, mask);
            v.mul(v).mul(v).intoArray(dest, destOff + k, mask);
        }
    }

    private static void computeFourthPower(float[] src, int srcOff, float[] dest, int destOff, int n) {
        int k = 0;
        final int limit = F_SPECIES.loopBound(n);
        final int vl = F_SPECIES.length();

        for (; k < limit; k += vl) {
            FloatVector v = FloatVector.fromArray(F_SPECIES, src, srcOff + k);
            FloatVector sq = v.mul(v);
            sq.mul(sq).intoArray(dest, destOff + k);
        }

        int remaining = n - k;
        if (remaining > 0) {
            var mask = F_SPECIES.indexInRange(0, remaining);
            FloatVector v = FloatVector.fromArray(F_SPECIES, src, srcOff + k, mask);
            FloatVector sq = v.mul(v);
            sq.mul(sq).intoArray(dest, destOff + k, mask);
        }
    }

    public static void evaluateUniformExponent(float[] base, int bOffset, float exp,
            float[] dest, int dOffset, int n) {
        if (n <= 0) {
            return;
        }

        if (exp == 1.0) {
            if (base != dest || bOffset != dOffset) {
                System.arraycopy(base, bOffset, dest, dOffset, n);
            }
            return;
        }
        if (exp == 2.0) {
            computeSquare(base, bOffset, dest, dOffset, n);
            return;
        }
        if (exp == 3.0) {
            computeCube(base, bOffset, dest, dOffset, n);
            return;
        }
        if (exp == 4.0) {
            computeFourthPower(base, bOffset, dest, dOffset, n);
            return;
        }

        if (exp == 0.5) {
            VectorTranscendentals.evaluateNative(base, bOffset, dest, dOffset, n, VectorOperators.SQRT);
            return;
        }

        // Delegate the highly complex log/exp routines to a separate compilation target
        evaluateComplexUniformExponent(base, bOffset, exp, dest, dOffset, n);
    }

    private static void evaluateComplexUniformExponent(float[] base, int bOffset, float exp,
            float[] dest, int dOffset, int n) {
        final int vl = F_SPECIES.length();
        final int limit = F_SPECIES.loopBound(n);
        int i = 0;

        if (exp == 0.0) {
            for (; i < limit; i += vl) {
                V_ONE.intoArray(dest, dOffset + i);
            }
        } else if (exp == -1.0) {
            for (; i < limit; i += vl) {
                FloatVector v = FloatVector.fromArray(F_SPECIES, base, bOffset + i);
                V_ONE.div(v).intoArray(dest, dOffset + i);
            }
        } else {
            final FloatVector vExp = FloatVector.broadcast(F_SPECIES, exp);
            if (exp % 1.0 == 0.0) {
                if (exp % 2.0 != 0.0) {
                    // Scenario 1: Odd Integer (FIXED: targetIdx bug resolved)
                    for (; i < limit; i += vl) {
                        FloatVector v = FloatVector.fromArray(F_SPECIES, base, bOffset + i);
                        var isNegativeMask = v.compare(VectorOperators.LT, 0.0f);
                        FloatVector log = v.abs().lanewise(VectorOperators.LOG);
                        FloatVector scaled = log.mul(vExp);
                        FloatVector resAbs = scaled.lanewise(VectorOperators.EXP);
                        resAbs.blend(resAbs.neg(), isNegativeMask).intoArray(dest, dOffset + i);
                    }
                } else {
                    // Scenario 2: Even Integer
                    for (; i < limit; i += vl) {
                        FloatVector v = FloatVector.fromArray(F_SPECIES, base, bOffset + i);
                        FloatVector log = v.abs().lanewise(VectorOperators.LOG);
                        FloatVector scaled = log.mul(vExp);
                        scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                    }
                }
            } else {
                // Scenario 3: Non-Integer
                for (; i < limit; i += vl) {
                    FloatVector v = FloatVector.fromArray(F_SPECIES, base, bOffset + i);
                    FloatVector log = v.lanewise(VectorOperators.LOG);
                    FloatVector scaled = log.mul(vExp);
                    scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                }
            }
        }

        // Clean Scalar Tail Pass
        for (; i < n; i++) {
            final double b = base[bOffset + i];
            dest[dOffset + i] = (float) ((exp == 0.0) ? 1.0 : (exp == -1.0) ? 1.0 / b : Math.pow(b, exp));
        }
    }

    // ========================================================================
    // Specialized Mathematical Transcendentals
    // ========================================================================
    /**
     * High-performance vectorized exp() using magic-number rounding +
     * 6th-degree minimax polynomial via FMA + fast bit manipulation for 2^k.
     */
    static FloatVector fastVectorExp(FloatVector x) {
        // Float overflows to +Inf above ~88.72 and underflows to 0 below ~-87.33;
        // the old double-range clamp (-745.13/709.78) let values through that
        // blow up float's exponent field long before reaching the bit trick below.
        x = x.lanewise(VectorOperators.MAX, -87.33f).lanewise(VectorOperators.MIN, 88.72f);

        FloatVector invLn2 = FloatVector.broadcast(F_SPECIES, 1.4426950408889634074f);
        FloatVector ln2Hi = FloatVector.broadcast(F_SPECIES, -0.6931471805599453f);
        FloatVector ln2Lo = FloatVector.broadcast(F_SPECIES, -2.8235290563031574E-13f);

        // Float mantissa is 23 bits (not double's 52), so the magic rounding
        // constant is 2^23, not 2^52.
        FloatVector magic = FloatVector.broadcast(F_SPECIES, 8388608.0f); // 2^23
        FloatVector k = x.mul(invLn2).add(magic).sub(magic);
        FloatVector r = x.add(k.mul(ln2Hi)).add(k.mul(ln2Lo));

        FloatVector p = r.mul(0.001398199650f).add(0.0088632903f);
        p = r.lanewise(VectorOperators.FMA, p, FloatVector.broadcast(F_SPECIES, 0.04166666666f));
        p = r.lanewise(VectorOperators.FMA, p, FloatVector.broadcast(F_SPECIES, 0.16666666666f));
        p = r.lanewise(VectorOperators.FMA, p, FloatVector.broadcast(F_SPECIES, 0.5f));
        p = r.lanewise(VectorOperators.FMA, p, V_ONE);
        p = r.lanewise(VectorOperators.FMA, p, V_ONE);

        // Build 2^k directly as a float via int bit-cast: float exponent bias is
        // 127 and the exponent field starts at bit 23. IntVector and FloatVector
        // share the same lane width under F_SPECIES, so this converts 1:1 with no
        // lane-splitting -- unlike the old FloatVector->LongVector->DoubleVector
        // ->FloatVector path, where each widening `convert(..., 0)` step only
        // populated half of the lanes.
        IntVector kInt = (IntVector) k.convert(VectorOperators.F2I, 0);
        IntVector exponent = kInt.add(127).lanewise(VectorOperators.LSHL, 23);
        FloatVector twoK = (FloatVector) exponent.convert(VectorOperators.REINTERPRET_I2F, 0);

        return p.mul(twoK);
    }

    static FloatVector vectorizedErf(FloatVector x) {
        return VectorizedCodyMath.erf(x);
    }

    // ===================== Stirling's Factorial Approximation =====================
    public static void stirling(int base, int n, float[] s) {
        int vl = F_SPECIES.length();
        int bound = F_SPECIES.loopBound(n);
        FloatVector pi2 = FloatVector.broadcast(F_SPECIES, (float) (2.0 * Math.PI));
        FloatVector nanVec = FloatVector.broadcast(F_SPECIES, Float.NaN);
        int i = 0;

        for (; i < bound; i += vl) {
            FloatVector v = FloatVector.fromArray(F_SPECIES, s, base + i);
            FloatVector lnN = v.lanewise(VectorOperators.LOG);
            FloatVector term1 = v.mul(lnN).sub(v);
            FloatVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5f);
            FloatVector term3 = V_ONE.div(v.mul(12.0f));
            FloatVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

            var invalidMask = v.compare(VectorOperators.LE, 0.0f);
            result.blend(nanVec, invalidMask).intoArray(s, base + i);
        }

        int remaining = n - i;
        if (remaining > 0) {
            var mask = F_SPECIES.indexInRange(0, remaining);
            FloatVector v = FloatVector.fromArray(F_SPECIES, s, base + i, mask);
            FloatVector lnN = v.lanewise(VectorOperators.LOG);
            FloatVector term1 = v.mul(lnN).sub(v);
            FloatVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5f);
            FloatVector term3 = V_ONE.div(v.mul(12.0f));
            FloatVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

            var invalidMask = v.compare(VectorOperators.LE, 0.0f);
            result.blend(nanVec, invalidMask).intoArray(s, base + i, mask);
        }
    }
// Inside VectorMath class
// Inside VectorMath class

    public static void swiglu2(int lOff, int rOff, int destOff, int n, float[] s) {
        int limit = F_SPECIES.loopBound(n);
        int k = 0;
        final FloatVector ONE = FloatVector.broadcast(F_SPECIES, 1.0f);

        for (; k < limit; k += F_SPECIES.length()) {
            FloatVector x = FloatVector.fromArray(F_SPECIES, s, lOff + k);
            FloatVector y = FloatVector.fromArray(F_SPECIES, s, rOff + k);
            FloatVector expNegX = fastVectorExp(x.neg());

            // Math: x * y / (exp(-x) + 1)
            x.mul(y).div(expNegX.add(ONE)).intoArray(s, destOff + k);
        }
        for (; k < n; k++) {
            s[destOff + k] = (float) Maths.swiglu(s[lOff + k], s[rOff + k]);
        }
    }

    public static void geglu2(int lOff, int rOff, int destOff, int n, float[] s) {
        int limit = F_SPECIES.loopBound(n);
        int k = 0;
        final FloatVector HALF = FloatVector.broadcast(F_SPECIES, 0.5f);
        final FloatVector ONE = FloatVector.broadcast(F_SPECIES, 1.0f);
        final FloatVector INV_SQRT_2 = FloatVector.broadcast(F_SPECIES, 0.7071067811865476f);

        for (; k < limit; k += F_SPECIES.length()) {
            FloatVector x = FloatVector.fromArray(F_SPECIES, s, lOff + k);
            FloatVector y = FloatVector.fromArray(F_SPECIES, s, rOff + k);

            // Math: x * (y * 0.5 * (erf(y * 0.707) + 1))
            FloatVector erfVal = vectorizedErf(y.mul(INV_SQRT_2));
            FloatVector geluY = y.mul(HALF).mul(erfVal.add(ONE));

            x.mul(geluY).intoArray(s, destOff + k);
        }
        for (; k < n; k++) {
            s[destOff + k] = (float) Maths.geglu(s[lOff + k], s[rOff + k]);
        }
    }

    public static void swiglu(int base, int n, float[] s) {
        int limit = F_SPECIES.loopBound(n);
        int k = 0;
        final FloatVector ONE = FloatVector.broadcast(F_SPECIES, 1.0f);

        for (; k < limit; k += F_SPECIES.length()) {
            FloatVector x = FloatVector.fromArray(F_SPECIES, s, base + k);
            FloatVector expNegX = fastVectorExp(x.neg());
            x.div(expNegX.add(ONE)).intoArray(s, base + k);
        }
        for (; k < n; k++) {
            s[base + k] = (float) Maths.swiglu(s[base + k]);
        }
    }

    public static void gelu(int base, int n, float[] s) {
        int limit = F_SPECIES.loopBound(n);
        int k = 0;
        final FloatVector HALF = FloatVector.broadcast(F_SPECIES, 0.5f);
        final FloatVector ONE = FloatVector.broadcast(F_SPECIES, 1.0f);
        final FloatVector INV_SQRT_2 = FloatVector.broadcast(F_SPECIES, 0.7071067811865476f);

        for (; k < limit; k += F_SPECIES.length()) {
            FloatVector x = FloatVector.fromArray(F_SPECIES, s, base + k);
            x.mul(HALF).mul(vectorizedErf(x.mul(INV_SQRT_2)).add(ONE)).intoArray(s, base + k);
        }
        for (; k < n; k++) {
            s[base + k] = (float) Maths.gelu(s[base + k]);
        }
    }

    public static void geluFast(int base, int n, float[] s) {
        int limit = F_SPECIES.loopBound(n);
        int k = 0;
        final FloatVector HALF = FloatVector.broadcast(F_SPECIES, 0.5f);
        final FloatVector ONE = FloatVector.broadcast(F_SPECIES, 1.0f);
        final FloatVector TWO = FloatVector.broadcast(F_SPECIES, 2.0f);
        final FloatVector SQRT_2_OVER_PI = FloatVector.broadcast(F_SPECIES, 0.7978845608028654f);
        final FloatVector COEF = FloatVector.broadcast(F_SPECIES, 0.044715f);

        for (; k < limit; k += F_SPECIES.length()) {
            FloatVector x = FloatVector.fromArray(F_SPECIES, s, base + k);
            FloatVector x3 = x.mul(x).mul(x);
            FloatVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);
            FloatVector exp2z = fastVectorExp(z.mul(TWO));
            FloatVector tanhZ = exp2z.sub(ONE).div(exp2z.add(ONE));
            x.mul(HALF).mul(tanhZ.add(ONE)).intoArray(s, base + k);
        }
        for (; k < n; k++) {
            s[base + k] = (float) Maths.fastGelu(s[base + k]);
        }
    }

// Based on your switch case, unary GEGLU passes 'x' through SIMD but runs geglu() on the tail.
    public static void gegluUnary(int base, int n, float[] s) {
        int limit = F_SPECIES.loopBound(n);
        int k = 0;
        // Your original code did `result = x`, so SIMD does nothing to the array here.
        // If that was intentional, we just advance k. Otherwise, add vector math here.
        k = limit;
        for (; k < n; k++) {
            s[base + k] = (float) Maths.geglu(s[base + k]);
        }
    }

    public static void erf(int base, int n, float[] s) {
        int limit = F_SPECIES.loopBound(n);
        int k = 0;
        for (; k < limit; k += F_SPECIES.length()) {
            FloatVector x = FloatVector.fromArray(F_SPECIES, s, base + k);
            vectorizedErf(x).intoArray(s, base + k);
        }
        for (; k < n; k++) {
            s[base + k] = (float) Maths.erf(s[base + k]);
        }
    }

    // Add to VectorMath
    public static void abs(int base, int n, float[] s) {
        int limit = F_SPECIES.loopBound(n);
        int k = 0;
        for (; k < limit; k += F_SPECIES.length()) {
            FloatVector.fromArray(F_SPECIES, s, base + k)
                    .lanewise(VectorOperators.ABS)
                    .intoArray(s, base + k);
        }
        for (; k < n; k++) {
            s[base + k] = (float) Math.abs(s[base + k]);
        }
    }

    // ===================== Conditional Branching =====================
    public static void if3(int base, int tileN, float[] s, int block) {
        final int cond = base + block;
        final int trueVal = base + 2 * block;
        final int falseVal = base + 3 * block;
        final int res = base;

        int vl = F_SPECIES.length();
        int bound = F_SPECIES.loopBound(tileN);
        int i = 0;

        for (; i < bound; i += vl) {
            FloatVector vc = FloatVector.fromArray(F_SPECIES, s, cond + i);
            FloatVector vt = FloatVector.fromArray(F_SPECIES, s, trueVal + i);
            FloatVector vf = FloatVector.fromArray(F_SPECIES, s, falseVal + i);
            VectorMask<Float> mask = vc.compare(VectorOperators.NE, 0.0f).and(vc.compare(VectorOperators.EQ, vc));
            vf.blend(vt, mask).intoArray(s, res + i);
        }

        int remaining = tileN - i;
        if (remaining > 0) {
            var maskTail = F_SPECIES.indexInRange(0, remaining);
            FloatVector vc = FloatVector.fromArray(F_SPECIES, s, cond + i, maskTail);
            FloatVector vt = FloatVector.fromArray(F_SPECIES, s, trueVal + i, maskTail);
            FloatVector vf = FloatVector.fromArray(F_SPECIES, s, falseVal + i, maskTail);
            VectorMask<Float> mask = vc.compare(VectorOperators.NE, 0.0f).and(vc.compare(VectorOperators.EQ, vc));
            vf.blend(vt, mask).intoArray(s, res + i, maskTail);
        }
    }

    // ========================================================================
    // Vectorized Inverse Hyperbolic Implementations
    // ========================================================================
    private static FloatVector vectorAsinhImpl(FloatVector x) {
        return x.add(x.mul(x).add(V_ONE).lanewise(VectorOperators.SQRT))
                .lanewise(VectorOperators.LOG);
    }

    private static FloatVector vectorAcoshImpl(FloatVector x) {
        VectorMask<Float> valid = x.compare(VectorOperators.GE, V_ONE);
        FloatVector result = x.add(x.mul(x).sub(V_ONE).lanewise(VectorOperators.SQRT))
                .lanewise(VectorOperators.LOG);
        return result.blend(V_NAN, valid.not());
    }

    private static FloatVector vectorAtanhImpl(FloatVector x) {
        VectorMask<Float> valid = x.abs().compare(VectorOperators.LT, V_ONE);
        FloatVector result = V_ONE.add(x).div(V_ONE.sub(x))
                .lanewise(VectorOperators.LOG)
                .mul(V_HALF);
        return result.blend(V_NAN, valid.not());
    }

    private static FloatVector vectorAsechImpl(FloatVector x) {
        VectorMask<Float> valid = x.compare(VectorOperators.GT, 0.0f)
                .and(x.compare(VectorOperators.LE, V_ONE));
        FloatVector result = V_ONE.div(x).add(V_ONE.div(x.mul(x)).sub(V_ONE).lanewise(VectorOperators.SQRT))
                .lanewise(VectorOperators.LOG);
        return result.blend(V_NAN, valid.not());
    }

    private static FloatVector vectorAcschImpl(FloatVector x) {
        VectorMask<Float> valid = x.compare(VectorOperators.NE, 0.0f);
        FloatVector result = V_ONE.div(x).add(V_ONE.div(x.mul(x)).add(V_ONE).lanewise(VectorOperators.SQRT))
                .lanewise(VectorOperators.LOG);
        return result.blend(V_NAN, valid.not());
    }

    private static FloatVector vectorAcothImpl(FloatVector x) {
        VectorMask<Float> valid = x.abs().compare(VectorOperators.GT, V_ONE);
        FloatVector result = V_ONE.add(V_ONE.div(x)).div(V_ONE.sub(V_ONE.div(x)))
                .lanewise(VectorOperators.LOG)
                .mul(V_HALF);
        return result.blend(V_NAN, valid.not());
    }

}
