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

import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.F_SPECIES;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 *
 * @author GBEMIRO
 */
public class VectorTranscendentals {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

        public static void evaluateNative(double[] src, int srcOffset, double[] dest, int destOffset, int n, VectorOperators.Unary op) {
            int vl = SPECIES.length();
            int limit = SPECIES.loopBound(n);
            int i = 0;

            // Vector Loop
            for (; i < limit; i += vl) {
                DoubleVector va  = DoubleVector.fromArray(SPECIES, src, srcOffset + i);
                va.lanewise(op).intoArray(dest, destOffset + i);
            }

            // Clean Masked Tail
            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector va  = DoubleVector.fromArray(SPECIES, src, srcOffset + i, mask);
                va.lanewise(op).intoArray(dest, destOffset + i, mask);
            }
        }
        
          public static void evaluateNative(float[] src, int srcOffset, float[] dest, int destOffset, int n, VectorOperators.Unary op) {
            int vl = F_SPECIES.length();
            int limit = F_SPECIES.loopBound(n);
            int i = 0;

            // Vector Loop
            for (; i < limit; i += vl) {
                FloatVector va  = FloatVector.fromArray(F_SPECIES, src, srcOffset + i);
                va.lanewise(op).intoArray(dest, destOffset + i);
            }

            // Clean Masked Tail
            int remaining = n - i;
            if (remaining > 0) {
                var mask = F_SPECIES.indexInRange(0, remaining);
                FloatVector va  = FloatVector.fromArray(F_SPECIES, src, srcOffset + i, mask);
                va.lanewise(op).intoArray(dest, destOffset + i, mask);
            }
        }
}
