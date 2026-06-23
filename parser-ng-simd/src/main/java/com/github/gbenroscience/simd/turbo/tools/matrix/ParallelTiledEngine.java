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
package com.github.gbenroscience.simd.turbo.tools.matrix;

/**
 *
 * @author GBEMIRO
 */import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorMask;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Phaser;

/**
 * High-performance, allocation-free GELU activation engine using Java Vector API.
 */
public final class ParallelTiledEngine {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VF_LEN = SPECIES.length();
    private static final int UNROLL_FACTOR = 2;

    private static final float GELU_C1 = 0.79788456f;
    private static final float GELU_C2 = 0.044715f;
    
    private static final float TANH_A = 0.1056f;
    private static final float TANH_B = 0.4324f;

    // Hoisted broadcast vectors for flawless JIT register pinning
    private static final FloatVector V_ONE     = FloatVector.broadcast(SPECIES, 1.0f);
    private static final FloatVector V_NEG_ONE = FloatVector.broadcast(SPECIES, -1.0f);
    private static final FloatVector V_HALF    = FloatVector.broadcast(SPECIES, 0.5f);
    private static final FloatVector V_C1      = FloatVector.broadcast(SPECIES, GELU_C1);
    private static final FloatVector V_C2      = FloatVector.broadcast(SPECIES, GELU_C2);
    private static final FloatVector V_TANH_A  = FloatVector.broadcast(SPECIES, TANH_A);
    private static final FloatVector V_TANH_B  = FloatVector.broadcast(SPECIES, TANH_B);

    private ParallelTiledEngine() {}

    public static void geluInPlaceParallel(ExecutorService pool, int cores, float[] data, int offset, int len) {
        if (len < 64_000) {                     // Lower threshold for small arrays
            executeSingleThreaded(data, offset, len);
            return;
        }

        int chunkSize = (len + cores - 1) / cores;
        // Align to cache line (64 bytes = 16 floats) to eliminate false sharing
        chunkSize = ((chunkSize + 15) / 16) * 16;

        final Phaser phaser = new Phaser(1);

        for (int c = 0; c < cores; c++) {
            int start = c * chunkSize;
            if (start >= len) break;

            int end = Math.min(start + chunkSize, len);
            int taskLen = end - start;

            phaser.register();
            pool.submit(() -> {
                try {
                    processChunk(data, offset + start, taskLen);
                } finally {
                    phaser.arriveAndDeregister();
                }
            });
        }

        phaser.arriveAndAwaitAdvance();
    }

    private static void processChunk(float[] data, int offset, int len) {
        int i = 0;
        int unrollBound = (len / (UNROLL_FACTOR * VF_LEN)) * (UNROLL_FACTOR * VF_LEN);

        // 2× Unrolled Vector Loop
        for (; i < unrollBound; i += UNROLL_FACTOR * VF_LEN) {
            FloatVector v0 = FloatVector.fromArray(SPECIES, data, offset + i);
            FloatVector v1 = FloatVector.fromArray(SPECIES, data, offset + i + VF_LEN);

            v0 = geluVector(v0);
            v1 = geluVector(v1);

            v0.intoArray(data, offset + i);
            v1.intoArray(data, offset + i + VF_LEN);
        }

        // Main Vector Loop
        int vectorBound = SPECIES.loopBound(len);
        for (; i < vectorBound; i += VF_LEN) {
            FloatVector v = FloatVector.fromArray(SPECIES, data, offset + i);
            v = geluVector(v);
            v.intoArray(data, offset + i);
        }

        // Scalar Tail - Mirrored mathematically with fastTanh for bitwise determinism
        for (; i < len; i++) {
            float x = data[offset + i];
            float inner = GELU_C1 * (x + GELU_C2 * x * x * x);
            
            // Scalar evaluation of the minimax rational approximation
            float x2 = inner * inner;
            float num = inner + x2 * inner * TANH_A;
            float den = 1.0f + x2 * TANH_B;
            float t = num / den;
            
            if (inner > 3.0f) t = 1.0f;
            else if (inner < -3.0f) t = -1.0f;

            data[offset + i] = 0.5f * x * (1.0f + t);
        }
    }

    private static FloatVector geluVector(FloatVector x) {
        FloatVector inner = x.add(x.mul(x).mul(x).mul(V_C2)).mul(V_C1);
        FloatVector tanh = fastTanh(inner);
        return x.mul(V_HALF).mul(V_ONE.add(tanh));
    }

    private static FloatVector fastTanh(FloatVector x) {
        VectorMask<Float> high = x.compare(VectorOperators.GT, 3.0f);
        VectorMask<Float> low  = x.compare(VectorOperators.LT, -3.0f);

        FloatVector x2 = x.mul(x);
        FloatVector num = x.add(x2.mul(x).mul(V_TANH_A));
        FloatVector den = V_ONE.add(x2.mul(V_TANH_B));

        FloatVector result = num.div(den);
        return result.blend(V_ONE, high).blend(V_NEG_ONE, low);
    }

    public static void executeSingleThreaded(float[] data, int offset, int len) {
        processChunk(data, offset, len);
    }

    public static void executeGeluScalar(float[] data, int offset, int len) {
        for (int i = 0; i < len; i++) {
            float x = data[offset + i];
            float inner = GELU_C1 * (x + GELU_C2 * x * x * x);
            
            float x2 = inner * inner;
            float num = inner + x2 * inner * TANH_A;
            float den = 1.0f + x2 * TANH_B;
            float t = num / den;
            
            if (inner > 3.0f) t = 1.0f;
            else if (inner < -3.0f) t = -1.0f;

            data[offset + i] = 0.5f * x * (1.0f + t);
        }
    }
}