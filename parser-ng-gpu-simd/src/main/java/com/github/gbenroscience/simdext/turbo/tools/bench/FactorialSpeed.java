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
package com.github.gbenroscience.simdext.turbo.tools.bench;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.simd.turbo.tools.utils.VectorMath;
import com.github.gbenroscience.simdext.turbo.tools.utils.FactorialVectorF;
import com.github.gbenroscience.simdext.turbo.tools.utils.VectorMathF;
import java.util.Random;

/**
 *
 * @author GBEMIRO ns/element for VectorMath / VectorMathF vs a scalar baseline
 * (Maths.fact, the original algorithm).
 */
public class FactorialSpeed {

    static volatile double sinkD;
    static volatile float sinkF;

    interface FF {

        void run(float[] s, float[] d, int n);
    }

    interface DD {

        void run(double[] s, double[] d, int n);
    }

    static double timeF(FF f, float[] s, float[] d, int n, int reps) {
        for (int w = 0; w < 200; w++) {
            f.run(s, d, n);
        }
        long best = Long.MAX_VALUE;
        for (int r = 0; r < 7; r++) {
            long t0 = System.nanoTime();
            for (int k = 0; k < reps; k++) {
                f.run(s, d, n);
            }
            best = Math.min(best, System.nanoTime() - t0);
        }
        sinkF = d[n / 2];
        return best / (double) reps / n;
    }

    static double timeD(DD f, double[] s, double[] d, int n, int reps) {
        for (int w = 0; w < 200; w++) {
            f.run(s, d, n);
        }
        long best = Long.MAX_VALUE;
        for (int r = 0; r < 7; r++) {
            long t0 = System.nanoTime();
            for (int k = 0; k < reps; k++) {
                f.run(s, d, n);
            }
            best = Math.min(best, System.nanoTime() - t0);
        }
        sinkD = d[n / 2];
        return best / (double) reps / n;
    }

    public static void main(String[] a) {
        int N = (1 << 14) + 7;                       // not a lane multiple: the masked tail is part of the measurement
        Random r = new Random(42);
        System.out.println("double lanes = " + jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length()
                + ", float lanes = " + jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length() + ", n = " + N);

        double[][] dd = new double[3][N];
        for (int i = 0; i < N; i++) {
            dd[0][i] = r.nextInt(171);
            dd[1][i] = r.nextDouble() * 20;
            dd[2][i] = r.nextDouble() * 170.6;
        }
        double[] o = new double[N];
        DD sc = (s, q, n) -> {
            for (int i = 0; i < n; i++) {
                q[i] = Maths.fact(s[i]);
            }
        };
        DD ve = (s, q, n) -> VectorMath.fact(s, 0, q, 0, n);
        String[] dn = {"integers 0..170", "non-integers [0,20)", "non-integers [0,170.6]"};
        System.out.println("\nDOUBLE VectorMath.fact (ns/element)");
        System.out.printf("  %-26s %8s %9s %9s%n", "workload", "scalar", "vector", "speedup");
        for (int k = 0; k < 3; k++) {
            int reps = k == 0 ? 400 : 40;
            double ts = timeD(sc, dd[k], o, N, reps), tv = timeD(ve, dd[k], o, N, reps);
            System.out.printf("  %-26s %8.2f %9.2f %8.1fx%n", dn[k], ts, tv, ts / tv);
        }

        float[][] data = new float[4][N];
        for (int i = 0; i < N; i++) {
            data[0][i] = r.nextInt(35);
            data[1][i] = (i & 1) == 0 ? r.nextInt(35) : r.nextFloat() * 34f;
            data[2][i] = r.nextFloat() * 10f;
            data[3][i] = r.nextFloat() * 34f;
        }
        float[] d = new float[N];
        FF scalar = (s, q, n) -> {
            for (int i = 0; i < n; i++) {
                q[i] = (float) Maths.fact((double) s[i]);
            }
        };
        FF accurate = (s, q, n) -> VectorMathF.fact(s, 0, q, 0, n);
        FF fast = (s, q, n) -> FactorialVectorF.factFast(s, 0, q, 0, n);
        String[] names = {"integers 0..34", "mixed 50% int / 50% frac", "non-integers [0,10)", "non-integers [0,34]"};
        System.out.println("\nFLOAT (ns/element)");
        System.out.printf("  %-26s %8s %10s %9s | %9s %9s%n", "workload", "scalar", "fact(acc)", "factFast", "acc spd", "fast spd");
        for (int k = 0; k < 4; k++) {
            int reps = k == 0 ? 400 : 40;
            double ts = timeF(scalar, data[k], d, N, reps), ta = timeF(accurate, data[k], d, N, reps), tf = timeF(fast, data[k], d, N, reps);
            System.out.printf("  %-26s %8.2f %10.2f %9.2f | %8.1fx %8.1fx%n", names[k], ts, ta, tf, ts / ta, ts / tf);
        }
    }
}
