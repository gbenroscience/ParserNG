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
import com.github.gbenroscience.simdext.turbo.tools.junk.SIMDEngineF64;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 *
 * @author GBEMIRO
 */
  public final class VectorMath {

        private VectorMath() {
        }

        private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
        public static int VECTOR_THRESHOLD = 256;

        // Angle conversions
        private static final double DEG_TO_RAD = Math.PI / 180.0;
        private static final double RAD_TO_DEG = 180.0 / Math.PI;
        private static final double GRAD_TO_RAD = Math.PI / 200.0;
        private static final double RAD_TO_GRAD = 200.0 / Math.PI;

        private static final DoubleVector V_DEG_TO_RAD = DoubleVector.broadcast(SPECIES, DEG_TO_RAD);
        private static final DoubleVector V_RAD_TO_DEG = DoubleVector.broadcast(SPECIES, RAD_TO_DEG);
        private static final DoubleVector V_GRAD_TO_RAD = DoubleVector.broadcast(SPECIES, GRAD_TO_RAD);
        private static final DoubleVector V_RAD_TO_GRAD = DoubleVector.broadcast(SPECIES, RAD_TO_GRAD);

        // Core constants
        private static final DoubleVector V_ONE = DoubleVector.broadcast(SPECIES, 1.0);
        private static final DoubleVector V_NEG_ONE = DoubleVector.broadcast(SPECIES, -1.0);
        private static final DoubleVector V_HALF = DoubleVector.broadcast(SPECIES, 0.5);
        private static final DoubleVector V_HALF_PI = DoubleVector.broadcast(SPECIES, Math.PI / 2.0);
        private static final DoubleVector V_NEG_HALF_PI = DoubleVector.broadcast(SPECIES, -Math.PI / 2.0);
        private static final DoubleVector V_NAN = DoubleVector.broadcast(SPECIES, Double.NaN);
        private static final DoubleVector ZERO = DoubleVector.broadcast(SPECIES, 0.0);

        private static final double THRESHOLD_LOW = 0.46875;
        private static final double THRESHOLD_HIGH = 4.0;

        // ========================================================================
        // NO-LAMBDA DIRECT OPERATIONS
        // ========================================================================
        // Radian
        public static void sin(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sin(s[base + i]);
            }
        }

        public static void cos(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cos(s[base + i]);
            }
        }

        public static void tan(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tan(s[base + i]);
            }
        }

        // Degree
        public static void sinDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sin(Math.toRadians(s[base + i]));
            }
        }

        public static void cosDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cos(Math.toRadians(s[base + i]));
            }
        }

        public static void tanDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tan(Math.toRadians(s[base + i]));
            }
        }

        // Grad
        public static void sinGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sin(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void cosGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cos(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void tanGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tan(s[base + i] * GRAD_TO_RAD);
            }
        }

        // ===================== Reciprocal Trigonometric =====================
        // Radian
        public static void sec(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.cos(s[base + i]);
            }
        }

        public static void csc(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.sin(s[base + i]);
            }
        }

        public static void cot(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.tan(s[base + i]);
            }
        }

        // Degree
        public static void secDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.cos(Math.toRadians(s[base + i]));
            }
        }

        public static void cscDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.sin(Math.toRadians(s[base + i]));
            }
        }

        public static void cotDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.tan(Math.toRadians(s[base + i]));
            }
        }

        // Grad
        public static void secGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.cos(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void cscGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.sin(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void cotGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.tan(s[base + i] * GRAD_TO_RAD);
            }
        }

        // ===================== Inverse Trigonometric =====================
        // Radian
        public static void asin(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(s[base + i]);
            }
        }

        public static void acos(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(s[base + i]);
            }
        }

        public static void atan(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(s[base + i]);
            }
        }

        // Degree
        public static void asinDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.asin(s[base + i]));
            }
        }

        public static void acosDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.acos(s[base + i]));
            }
        }

        public static void atanDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.atan(s[base + i]));
            }
        }

        // Grad
        public static void asinGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void acosGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void atanGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(s[base + i]) * RAD_TO_GRAD;
            }
        }

        // ===================== Inverse Reciprocal Trigonometric =====================
        // Radian
        public static void acsc(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(1.0 / s[base + i]);
            }
        }

        public static void asec(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(1.0 / s[base + i]);
            }
        }

        public static void acot(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(1.0 / s[base + i]);
            }
        }

        // Degree
        public static void acscDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.asin(1.0 / s[base + i]));
            }
        }

        public static void asecDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.acos(1.0 / s[base + i]));
            }
        }

        public static void acotDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.atan(1.0 / s[base + i]));
            }
        }

        // Grad
        public static void acscGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void asecGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void acotGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        // ===================== Hyperbolic =====================
        public static void sinh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SINH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sinh(s[base + i]);
            }
        }

        public static void cosh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.COSH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cosh(s[base + i]);
            }
        }

        public static void tanh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.TANH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tanh(s[base + i]);
            }
        }

        // ===================== Inverse Hyperbolic =====================
        public static void asinh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAsinhImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.log(s[base + i] + Math.sqrt(s[base + i] * s[base + i] + 1.0));
            }
        }

        public static void acosh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcoshImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = x < 1.0 ? Double.NaN : Math.log(x + Math.sqrt(x * x - 1.0));
            }
        }

        public static void atanh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAtanhImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = 0.5 * Math.log((1.0 + x) / (1.0 - x));
            }
        }

        public static void asech(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAsechImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = (x <= 0.0 || x > 1.0) ? Double.NaN : Math.log((1.0 / x) + Math.sqrt((1.0 / (x * x)) - 1.0));
            }
        }

        public static void acsch(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcschImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = x == 0.0 ? Double.NaN : Math.log((1.0 / x) + Math.sqrt((1.0 / (x * x)) + 1.0));
            }
        }

        public static void acoth(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcothImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = Math.abs(x) <= 1.0 ? Double.NaN : 0.5 * Math.log((1.0 + (1.0 / x)) / (1.0 - (1.0 / x)));
            }
        }

        public static void sqrt(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SQRT)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sqrt(s[base + i]);
            }
        }

        public static void cbrt(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.CBRT)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cbrt(s[base + i]);
            }
        }

        // ===================== Exponential and Logarithmic =====================
        public static void exp(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.EXP)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.exp(s[base + i]);
            }
        }

        public static void ln(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.LOG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.log(s[base + i]);
            }
        }

        public static void log10(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.LOG10)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.log10(s[base + i]);
            }
        }

        private static boolean isExponentUniform(double[] scratch, int offset, int n) {
            if (n <= 1) {
                return true;
            }

            final double first = scratch[offset];
            if (Double.isNaN(first)) {
                // All must be NaN
                final int vl = SPECIES.length();
                int i = 0;
                int bound = SPECIES.loopBound(n);
                for (; i < bound; i += vl) {
                    DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i);
                    if (v.compare(VectorOperators.EQ, v).anyTrue()) {
                        return false;
                    }
                }
                int remaining = n - i;
                if (remaining > 0) {
                    var mask = SPECIES.indexInRange(0, remaining);
                    DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i, mask);
                    if (v.compare(VectorOperators.EQ, v, mask).anyTrue()) {
                        return false;
                    }
                }
                return true;
            }

            final DoubleVector target = DoubleVector.broadcast(SPECIES, first);
            final int vl = SPECIES.length();
            int i = 0;
            int bound = SPECIES.loopBound(n);

            for (; i < bound; i += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i);
                if (v.compare(VectorOperators.NE, target).anyTrue()) {
                    return false;
                }
            }

            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i, mask);
                if (v.compare(VectorOperators.NE, target, mask).anyTrue()) {
                    return false;
                }
            }
            return true;
        }

        public static void evaluateVariableExponent(double[] base, int bOffset, double[] exp, int eOffset,
                double[] dest, int dOffset, int n) {
            if (n <= 0) {
                return;
            }

            int i = 0;
            final int limit = SPECIES.loopBound(n);

            // === 1. Core Vector Loop: exp(y * ln(x)) ===
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector vBase = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                DoubleVector vExp = DoubleVector.fromArray(SPECIES, exp, eOffset + i);

                // Execute algebraic transcendental transformation
                DoubleVector log = vBase.lanewise(VectorOperators.LOG);
                DoubleVector scaled = log.mul(vExp);
                scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
            }

            // === 2. Masked Tail Pass ===
            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector vBase = DoubleVector.fromArray(SPECIES, base, bOffset + i, mask);
                DoubleVector vExp = DoubleVector.fromArray(SPECIES, exp, eOffset + i, mask);

                // Apply masks to intermediate operators to maintain lane isolation
                DoubleVector log = vBase.lanewise(VectorOperators.LOG, mask);
                DoubleVector scaled = log.mul(vExp, mask);
                DoubleVector res = scaled.lanewise(VectorOperators.EXP, mask);

                res.intoArray(dest, dOffset + i, mask);
            }
        }

        public static void executePowerBlended(double[] scratch, int baseOffset, int expOffset, int n) {
            if (n <= 0) {
                return;
            }

            if (isExponentUniform(scratch, expOffset, n)) {
                double uniformExp = scratch[expOffset];

                if (uniformExp == 0.5) {
                    SIMDEngineF64.VectorTranscendentals.evaluateNative(scratch, baseOffset, scratch, baseOffset, n, VectorOperators.SQRT);
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
        private static void computeSquare(double[] src, int srcOff, double[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES.loopBound(n);
            final int vl = SPECIES.length();

            for (; k < limit; k += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k);
                v.mul(v).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k, mask);
                v.mul(v).intoArray(dest, destOff + k, mask);
            }
        }

        private static void computeCube(double[] src, int srcOff, double[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES.loopBound(n);
            final int vl = SPECIES.length();

            for (; k < limit; k += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k);
                v.mul(v).mul(v).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k, mask);
                v.mul(v).mul(v).intoArray(dest, destOff + k, mask);
            }
        }

        private static void computeFourthPower(double[] src, int srcOff, double[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES.loopBound(n);
            final int vl = SPECIES.length();

            for (; k < limit; k += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k);
                DoubleVector sq = v.mul(v);
                sq.mul(sq).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k, mask);
                DoubleVector sq = v.mul(v);
                sq.mul(sq).intoArray(dest, destOff + k, mask);
            }
        }

        public static void evaluateUniformExponent(double[] base, int bOffset, double exp,
                double[] dest, int dOffset, int n) {
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
                SIMDEngineF64.VectorTranscendentals.evaluateNative(base, bOffset, dest, dOffset, n, VectorOperators.SQRT);
                return;
            }

            // Delegate the highly complex log/exp routines to a separate compilation target
            evaluateComplexUniformExponent(base, bOffset, exp, dest, dOffset, n);
        }

        private static void evaluateComplexUniformExponent(double[] base, int bOffset, double exp,
                double[] dest, int dOffset, int n) {
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);
            int i = 0;

            if (exp == 0.0) {
                for (; i < limit; i += vl) {
                    V_ONE.intoArray(dest, dOffset + i);
                }
            } else if (exp == -1.0) {
                for (; i < limit; i += vl) {
                    DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                    V_ONE.div(v).intoArray(dest, dOffset + i);
                }
            } else {
                final DoubleVector vExp = DoubleVector.broadcast(SPECIES, exp);
                if (exp % 1.0 == 0.0) {
                    if (exp % 2.0 != 0.0) {
                        // Scenario 1: Odd Integer (FIXED: targetIdx bug resolved)
                        for (; i < limit; i += vl) {
                            DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                            var isNegativeMask = v.compare(VectorOperators.LT, 0.0);
                            DoubleVector log = v.abs().lanewise(VectorOperators.LOG);
                            DoubleVector scaled = log.mul(vExp);
                            DoubleVector resAbs = scaled.lanewise(VectorOperators.EXP);
                            resAbs.blend(resAbs.neg(), isNegativeMask).intoArray(dest, dOffset + i);
                        }
                    } else {
                        // Scenario 2: Even Integer
                        for (; i < limit; i += vl) {
                            DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                            DoubleVector log = v.abs().lanewise(VectorOperators.LOG);
                            DoubleVector scaled = log.mul(vExp);
                            scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                        }
                    }
                } else {
                    // Scenario 3: Non-Integer
                    for (; i < limit; i += vl) {
                        DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                        DoubleVector log = v.lanewise(VectorOperators.LOG);
                        DoubleVector scaled = log.mul(vExp);
                        scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                    }
                }
            }

            // Clean Scalar Tail Pass
            for (; i < n; i++) {
                final double b = base[bOffset + i];
                dest[dOffset + i] = (exp == 0.0) ? 1.0 : (exp == -1.0) ? 1.0 / b : Math.pow(b, exp);
            }
        }

        // ========================================================================
        // Specialized Mathematical Transcendentals
        // ========================================================================
        /**
         * High-performance vectorized exp() using magic-number rounding +
         * 6th-degree minimax polynomial via FMA + fast bit manipulation for
         * 2^k.
         */
        static DoubleVector fastVectorExp(DoubleVector x) {
            x = x.lanewise(VectorOperators.MAX, -745.13).lanewise(VectorOperators.MIN, 709.78);

            DoubleVector invLn2 = DoubleVector.broadcast(SPECIES, 1.4426950408889634074);
            DoubleVector ln2Hi = DoubleVector.broadcast(SPECIES, -0.6931471805599453);
            DoubleVector ln2Lo = DoubleVector.broadcast(SPECIES, -2.8235290563031574E-13);

            DoubleVector magic = DoubleVector.broadcast(SPECIES, 4503599627370496.0); // 2^52
            DoubleVector k = x.mul(invLn2).add(magic).sub(magic);
            DoubleVector r = x.add(k.mul(ln2Hi)).add(k.mul(ln2Lo));

            DoubleVector p = r.mul(0.001398199650).add(0.0088632903);
            p = r.lanewise(VectorOperators.FMA, p, DoubleVector.broadcast(SPECIES, 0.04166666666));
            p = r.lanewise(VectorOperators.FMA, p, DoubleVector.broadcast(SPECIES, 0.16666666666));
            p = r.lanewise(VectorOperators.FMA, p, DoubleVector.broadcast(SPECIES, 0.5));
            p = r.lanewise(VectorOperators.FMA, p, V_ONE);
            p = r.lanewise(VectorOperators.FMA, p, V_ONE);

            LongVector kLong = (LongVector) k.convert(VectorOperators.D2L, 0);
            LongVector exponent = kLong.add(1023).lanewise(VectorOperators.LSHL, 52);
            DoubleVector twoK = (DoubleVector) exponent.convert(VectorOperators.REINTERPRET_L2D, 0);

            return p.mul(twoK);
        }

        static DoubleVector vectorizedErf(DoubleVector x) {
            return VectorizedCodyMath.erf(x);
        }

        // ===================== Stirling's Factorial Approximation =====================
        public static void stirling(int base, int n, double[] s) {
            int vl = SPECIES.length();
            int bound = SPECIES.loopBound(n);
            DoubleVector pi2 = DoubleVector.broadcast(SPECIES, 2.0 * Math.PI);
            DoubleVector nanVec = DoubleVector.broadcast(SPECIES, Double.NaN);
            int i = 0;

            for (; i < bound; i += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                DoubleVector lnN = v.lanewise(VectorOperators.LOG);
                DoubleVector term1 = v.mul(lnN).sub(v);
                DoubleVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5);
                DoubleVector term3 = V_ONE.div(v.mul(12.0));
                DoubleVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

                var invalidMask = v.compare(VectorOperators.LE, 0.0);
                result.blend(nanVec, invalidMask).intoArray(s, base + i);
            }

            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i, mask);
                DoubleVector lnN = v.lanewise(VectorOperators.LOG);
                DoubleVector term1 = v.mul(lnN).sub(v);
                DoubleVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5);
                DoubleVector term3 = V_ONE.div(v.mul(12.0));
                DoubleVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

                var invalidMask = v.compare(VectorOperators.LE, 0.0);
                result.blend(nanVec, invalidMask).intoArray(s, base + i, mask);
            }
        }
// Inside VectorMath class
// Inside VectorMath class

        public static void swiglu2(int lOff, int rOff, int destOff, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, lOff + k);
                DoubleVector y = DoubleVector.fromArray(SPECIES, s, rOff + k);
                DoubleVector expNegX = fastVectorExp(x.neg());

                // Math: x * y / (exp(-x) + 1)
                x.mul(y).div(expNegX.add(ONE)).intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = Maths.swiglu(s[lOff + k], s[rOff + k]);
            }
        }

        public static void geglu2(int lOff, int rOff, int destOff, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
            final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, lOff + k);
                DoubleVector y = DoubleVector.fromArray(SPECIES, s, rOff + k);

                // Math: x * (y * 0.5 * (erf(y * 0.707) + 1))
                DoubleVector erfVal = vectorizedErf(y.mul(INV_SQRT_2));
                DoubleVector geluY = y.mul(HALF).mul(erfVal.add(ONE));

                x.mul(geluY).intoArray(s, destOff + k);
            }
            for (; k < n; k++) {
                s[destOff + k] = Maths.geglu(s[lOff + k], s[rOff + k]);
            }
        }

        public static void swiglu(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + k);
                DoubleVector expNegX = fastVectorExp(x.neg());
                x.div(expNegX.add(ONE)).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Maths.swiglu(s[base + k]);
            }
        }
   
        public static void gelu(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
            final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + k);
                x.mul(HALF).mul(vectorizedErf(x.mul(INV_SQRT_2)).add(ONE)).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Maths.gelu(s[base + k]);
            }
        }

        public static void geluFast(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
            final DoubleVector TWO = DoubleVector.broadcast(SPECIES, 2.0);
            final DoubleVector SQRT_2_OVER_PI = DoubleVector.broadcast(SPECIES, 0.7978845608028654);
            final DoubleVector COEF = DoubleVector.broadcast(SPECIES, 0.044715);

            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + k);
                DoubleVector x3 = x.mul(x).mul(x);
                DoubleVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);
                DoubleVector exp2z = fastVectorExp(z.mul(TWO));
                DoubleVector tanhZ = exp2z.sub(ONE).div(exp2z.add(ONE));
                x.mul(HALF).mul(tanhZ.add(ONE)).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Maths.fastGelu(s[base + k]);
            }
        }

// Based on your switch case, unary GEGLU passes 'x' through SIMD but runs geglu() on the tail.
        public static void gegluUnary(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            // Your original code did `result = x`, so SIMD does nothing to the array here.
            // If that was intentional, we just advance k. Otherwise, add vector math here.
            k = limit;
            for (; k < n; k++) {
                s[base + k] = Maths.geglu(s[base + k]);
            }
        }

        public static void erf(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + k);
                vectorizedErf(x).intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Maths.erf(s[base + k]);
            }
        }

        // Add to VectorMath
        public static void abs(int base, int n, double[] s) {
            int limit = SPECIES.loopBound(n);
            int k = 0;
            for (; k < limit; k += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + k)
                        .lanewise(VectorOperators.ABS)
                        .intoArray(s, base + k);
            }
            for (; k < n; k++) {
                s[base + k] = Math.abs(s[base + k]);
            }
        }

        // ===================== Conditional Branching =====================
        public static void if3(int base, int tileN, double[] s, int block) {
            final int cond = base + block;
            final int trueVal = base + 2 * block;
            final int falseVal = base + 3 * block;
            final int res = base;

            int vl = SPECIES.length();
            int bound = SPECIES.loopBound(tileN);
            int i = 0;

            for (; i < bound; i += vl) {
                DoubleVector vc = DoubleVector.fromArray(SPECIES, s, cond + i);
                DoubleVector vt = DoubleVector.fromArray(SPECIES, s, trueVal + i);
                DoubleVector vf = DoubleVector.fromArray(SPECIES, s, falseVal + i);
                VectorMask<Double> mask = vc.compare(VectorOperators.NE, 0.0).and(vc.compare(VectorOperators.EQ, vc));
                vf.blend(vt, mask).intoArray(s, res + i);
            }

            int remaining = tileN - i;
            if (remaining > 0) {
                var maskTail = SPECIES.indexInRange(0, remaining);
                DoubleVector vc = DoubleVector.fromArray(SPECIES, s, cond + i, maskTail);
                DoubleVector vt = DoubleVector.fromArray(SPECIES, s, trueVal + i, maskTail);
                DoubleVector vf = DoubleVector.fromArray(SPECIES, s, falseVal + i, maskTail);
                VectorMask<Double> mask = vc.compare(VectorOperators.NE, 0.0).and(vc.compare(VectorOperators.EQ, vc));
                vf.blend(vt, mask).intoArray(s, res + i, maskTail);
            }
        }

        // ========================================================================
        // Vectorized Inverse Hyperbolic Implementations
        // ========================================================================
        private static DoubleVector vectorAsinhImpl(DoubleVector x) {
            return x.add(x.mul(x).add(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
        }

        private static DoubleVector vectorAcoshImpl(DoubleVector x) {
            VectorMask<Double> valid = x.compare(VectorOperators.GE, V_ONE);
            DoubleVector result = x.add(x.mul(x).sub(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAtanhImpl(DoubleVector x) {
            VectorMask<Double> valid = x.abs().compare(VectorOperators.LT, V_ONE);
            DoubleVector result = V_ONE.add(x).div(V_ONE.sub(x))
                    .lanewise(VectorOperators.LOG)
                    .mul(V_HALF);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAsechImpl(DoubleVector x) {
            VectorMask<Double> valid = x.compare(VectorOperators.GT, 0.0)
                    .and(x.compare(VectorOperators.LE, V_ONE));
            DoubleVector result = V_ONE.div(x).add(V_ONE.div(x.mul(x)).sub(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAcschImpl(DoubleVector x) {
            VectorMask<Double> valid = x.compare(VectorOperators.NE, 0.0);
            DoubleVector result = V_ONE.div(x).add(V_ONE.div(x.mul(x)).add(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAcothImpl(DoubleVector x) {
            VectorMask<Double> valid = x.abs().compare(VectorOperators.GT, V_ONE);
            DoubleVector result = V_ONE.add(V_ONE.div(x)).div(V_ONE.sub(V_ONE.div(x)))
                    .lanewise(VectorOperators.LOG)
                    .mul(V_HALF);
            return result.blend(V_NAN, valid.not());
        }

        // ========================================================================
        // Fused Load-Unary Variants (added for LoadUnaryMathCommand fusion)
        // ========================================================================
        // Each method below is the exact same computation as its in-place
        // counterpart above, but reads from an arbitrary (src, srcBase) and
        // writes to an arbitrary (dest, destBase) instead of operating on `s`
        // in place. This lets SIMDCommandF64 read straight out of
        // flatVariables/_2DVariables for a bare-variable unary call, skipping
        // the load->scratch->read round trip - the same idea as
        // LoadLoadAddCommand, generalized to one operand.
        //
        // Deliberately additive: nothing above this line was modified, so the
        // existing in-place methods (and anything that depends on their exact
        // behavior) are untouched. Each formula below was copied verbatim from
        // its in-place counterpart - same vector op sequence, same scalar
        // fallback - so it should be bit-identical, but please still run it
        // through your correctness harness before trusting it in production;
        // I can't run that harness from here.

        public static void sinFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.sin(src[srcBase + i]);
            }
        }

        public static void cosFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.COS)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.cos(src[srcBase + i]);
            }
        }

        public static void tanFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.tan(src[srcBase + i]);
            }
        }

        public static void sinDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.sin(Math.toRadians(src[srcBase + i]));
            }
        }

        public static void cosDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.COS)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.cos(Math.toRadians(src[srcBase + i]));
            }
        }

        public static void tanDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.tan(Math.toRadians(src[srcBase + i]));
            }
        }

        public static void sinGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.sin(src[srcBase + i] * GRAD_TO_RAD);
            }
        }

        public static void cosGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.COS)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.cos(src[srcBase + i] * GRAD_TO_RAD);
            }
        }

        public static void tanGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.tan(src[srcBase + i] * GRAD_TO_RAD);
            }
        }

        public static void secDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.COS))
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = 1.0 / Math.cos(Math.toRadians(src[srcBase + i]));
            }
        }

        public static void cscDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = 1.0 / Math.sin(Math.toRadians(src[srcBase + i]));
            }
        }

        public static void cotDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_DEG_TO_RAD);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = 1.0 / Math.tan(Math.toRadians(src[srcBase + i]));
            }
        }

        public static void secGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.COS))
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = 1.0 / Math.cos(src[srcBase + i] * GRAD_TO_RAD);
            }
        }

        public static void cscGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = 1.0 / Math.sin(src[srcBase + i] * GRAD_TO_RAD);
            }
        }

        public static void cotGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .mul(V_GRAD_TO_RAD);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = 1.0 / Math.tan(src[srcBase + i] * GRAD_TO_RAD);
            }
        }

        public static void asinFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.asin(src[srcBase + i]);
            }
        }

        public static void acosFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.acos(src[srcBase + i]);
            }
        }

        public static void atanFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.atan(src[srcBase + i]);
            }
        }

        public static void asinDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.toDegrees(Math.asin(src[srcBase + i]));
            }
        }

        public static void acosDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.toDegrees(Math.acos(src[srcBase + i]));
            }
        }

        public static void atanDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.toDegrees(Math.atan(src[srcBase + i]));
            }
        }

        public static void asinGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.asin(src[srcBase + i]) * RAD_TO_GRAD;
            }
        }

        public static void acosGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.acos(src[srcBase + i]) * RAD_TO_GRAD;
            }
        }

        public static void atanGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.atan(src[srcBase + i]) * RAD_TO_GRAD;
            }
        }

        public static void acscFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.asin(1.0 / src[srcBase + i]);
            }
        }

        public static void asecFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.acos(1.0 / src[srcBase + i]);
            }
        }

        public static void acotFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.atan(1.0 / src[srcBase + i]);
            }
        }

        public static void acscDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.toDegrees(Math.asin(1.0 / src[srcBase + i]));
            }
        }

        public static void asecDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.toDegrees(Math.acos(1.0 / src[srcBase + i]));
            }
        }

        public static void acotDegFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.toDegrees(Math.atan(1.0 / src[srcBase + i]));
            }
        }

        public static void acscGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.asin(1.0 / src[srcBase + i]) * RAD_TO_GRAD;
            }
        }

        public static void asecGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.acos(1.0 / src[srcBase + i]) * RAD_TO_GRAD;
            }
        }

        public static void acotGradFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.atan(1.0 / src[srcBase + i]) * RAD_TO_GRAD;
            }
        }

        public static void sinhFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.SINH)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.sinh(src[srcBase + i]);
            }
        }

        public static void coshFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.COSH)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.cosh(src[srcBase + i]);
            }
        }

        public static void tanhFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.TANH)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.tanh(src[srcBase + i]);
            }
        }

        public static void asinhFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAsinhImpl(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                double x = src[srcBase + i];
                dest[destBase + i] = Math.log(x + Math.sqrt(x * x + 1.0));
            }
        }

        public static void acoshFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcoshImpl(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                double x = src[srcBase + i];
                dest[destBase + i] = x < 1.0 ? Double.NaN : Math.log(x + Math.sqrt(x * x - 1.0));
            }
        }

        public static void atanhFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAtanhImpl(DoubleVector.fromArray(SPECIES, src, srcBase + i))
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                double x = src[srcBase + i];
                dest[destBase + i] = 0.5 * Math.log((1.0 + x) / (1.0 - x));
            }
        }

        public static void sqrtFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.SQRT)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.sqrt(src[srcBase + i]);
            }
        }

        public static void cbrtFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.CBRT)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.cbrt(src[srcBase + i]);
            }
        }

        public static void expFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.EXP)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.exp(src[srcBase + i]);
            }
        }

        public static void lnFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.LOG)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.log(src[srcBase + i]);
            }
        }

        public static void log10Fused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.LOG10)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.log10(src[srcBase + i]);
            }
        }

        public static void absFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, src, srcBase + i)
                        .lanewise(VectorOperators.ABS)
                        .intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Math.abs(src[srcBase + i]);
            }
        }

        public static void erfFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, src, srcBase + i);
                vectorizedErf(x).intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Maths.erf(src[srcBase + i]);
            }
        }

        public static void swigluFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, src, srcBase + i);
                DoubleVector expNegX = fastVectorExp(x.neg());
                x.div(expNegX.add(ONE)).intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Maths.swiglu(src[srcBase + i]);
            }
        }

        public static void geluFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
            final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, src, srcBase + i);
                x.mul(HALF).mul(vectorizedErf(x.mul(INV_SQRT_2)).add(ONE)).intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Maths.gelu(src[srcBase + i]);
            }
        }

        public static void geluFastFused(double[] src, int srcBase, double[] dest, int destBase, int n) {
            int i = 0, limit = SPECIES.loopBound(n);
            final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
            final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
            final DoubleVector TWO = DoubleVector.broadcast(SPECIES, 2.0);
            final DoubleVector SQRT_2_OVER_PI = DoubleVector.broadcast(SPECIES, 0.7978845608028654);
            final DoubleVector COEF = DoubleVector.broadcast(SPECIES, 0.044715);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, src, srcBase + i);
                DoubleVector x3 = x.mul(x).mul(x);
                DoubleVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);
                DoubleVector exp2z = fastVectorExp(z.mul(TWO));
                DoubleVector tanhZ = exp2z.sub(ONE).div(exp2z.add(ONE));
                x.mul(HALF).mul(tanhZ.add(ONE)).intoArray(dest, destBase + i);
            }
            for (; i < n; i++) {
                dest[destBase + i] = Maths.fastGelu(src[srcBase + i]);
            }
        }


    }