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

import com.github.gbenroscience.simd.turbo.tools.utils.FactorialVector;
import com.github.gbenroscience.simd.turbo.tools.utils.VectorMath;
import com.github.gbenroscience.simdext.turbo.tools.utils.FactorialVectorF;
import com.github.gbenroscience.simdext.turbo.tools.utils.VectorMathF;
import java.util.Arrays;
import java.util.Random;
/**
 *
 * @author GBEMIRO
 */ 

/** Correctness + accuracy checks for VectorMath (double) and VectorMathF (float). Needs only Ref.java. */
public class FactorialVerify {
    interface DFn { void run(double[] s, int sb, double[] d, int db, int n); }
    interface FFn { void run(float[] s, int sb, float[] d, int db, int n); }

    static final boolean SIMD = FactorialVector.isSimd() && FactorialVectorF.isSimd();
    static final double TOL_D = SIMD ? 5e-14 : 2e-13;          // scalar fallback is the original ~9e-14 algorithm

    static boolean sameD(double got, double want, double relTol) {
        if (Double.isNaN(want)) return Double.isNaN(got);
        if (got == want) return true;
        if (Double.isInfinite(want) || Double.isInfinite(got) || Double.isNaN(got)) return false;
        return Math.abs(got / want - 1) <= relTol;
    }

    static double ulpF(float got, float want) {
        if (Float.isNaN(want)) return Float.isNaN(got) ? 0 : Double.POSITIVE_INFINITY;
        if (got == want) return 0;
        if (Float.isInfinite(want) || Float.isInfinite(got) || Float.isNaN(got)) return Double.POSITIVE_INFINITY;
        return Math.abs((double) got - want) / Math.ulp(want);
    }

    // ---------------- double ----------------
    static final double[] POOL_D = {0, 1, 2, 5, 22, 23, 100, 170, 171, 170.5, 170.6, 170.7, -1, -0.5, -170, Double.NaN,
            Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1e-300, 1e300, -1e300, 0.5, 3.7, 123.25, 20.5, 169.9, 24.01, 23.99};

    static double[] randomD(Random r, int len) {
        double[] s = new double[len];
        for (int i = 0; i < len; i++) s[i] = r.nextInt(3) == 0 ? POOL_D[r.nextInt(POOL_D.length)] : r.nextDouble() * 170.6;
        return s;
    }

    static int tailsD(String name, DFn f, Random r) {
        int bad = 0;
        for (int n = 0; n <= 80; n++) {
            double[] s = randomD(r, n);                         // exact length: an out-of-bounds masked read would throw
            double[] d = new double[n + 8];
            Arrays.fill(d, 777.0);
            f.run(s, 0, d, 0, n);
            for (int i = 0; i < n; i++) if (!sameD(d[i], Ref.expectedD(s[i]), TOL_D)) { if (++bad < 4) System.out.println("    " + name + " n=" + n + " i=" + i + " x=" + s[i] + " got=" + d[i] + " want=" + Ref.expectedD(s[i])); }
            for (int i = n; i < d.length; i++) if (d[i] != 777.0) { bad++; System.out.println("    " + name + " WROTE PAST END n=" + n); break; }
        }
        return bad;
    }

    static int inPlaceD(Random r) {
        int bad = 0;
        for (int base : new int[]{0, 3, 11}) for (int n = 0; n <= 70; n++) {
            double[] s = randomD(r, base + n + 9), orig = s.clone(), want = s.clone();
            VectorMath.fact(orig, base, want, base, n);          // out-of-place
            VectorMath.fact(base, n, s);                         // in place
            for (int i = 0; i < s.length; i++) {
                double expect = (i >= base && i < base + n) ? want[i] : orig[i];
                if (Double.doubleToLongBits(s[i]) != Double.doubleToLongBits(expect)) { if (++bad < 4) System.out.println("    D in-place base=" + base + " n=" + n + " i=" + i); break; }
            }
        }
        return bad;
    }

    // ---------------- float ----------------
    static final float[] POOL_F = {0, 1, 2, 5, 22, 33, 34, 35, 36, 37, 100, 34.04f, 34.040096f, 34.0401f, -1, -0.5f, -34f,
            Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 1e-30f, Float.MIN_VALUE, -0f, 1e9f, -1e9f, 3e30f,
            -3e30f, 8388608f, -8388609f, 0.5f, 3.7f, 12.25f, 33.9f, 20.5f};

    static float[] randomF(Random r, int len) {
        float[] s = new float[len];
        for (int i = 0; i < len; i++) s[i] = r.nextInt(3) == 0 ? POOL_F[r.nextInt(POOL_F.length)] : r.nextFloat() * 34.04f;
        return s;
    }

    static int tailsF(String name, FFn f, double maxUlp, Random r) {
        int bad = 0;
        for (int n = 0; n <= 80; n++) {
            float[] s = randomF(r, n);
            float[] d = new float[n + 8];
            Arrays.fill(d, 777f);
            f.run(s, 0, d, 0, n);
            for (int i = 0; i < n; i++) if (ulpF(d[i], Ref.expectedF(s[i])) > maxUlp) { if (++bad < 4) System.out.println("    " + name + " n=" + n + " i=" + i + " x=" + s[i] + " got=" + d[i] + " want=" + Ref.expectedF(s[i])); }
            for (int i = n; i < d.length; i++) if (d[i] != 777f) { bad++; System.out.println("    " + name + " WROTE PAST END n=" + n); break; }
        }
        return bad;
    }

    static int inPlaceF(Random r) {
        int bad = 0;
        for (int base : new int[]{0, 3, 11}) for (int n = 0; n <= 70; n++) {
            float[] s = randomF(r, base + n + 9), orig = s.clone(), want = s.clone();
            VectorMathF.fact(orig, base, want, base, n);
            VectorMathF.fact(base, n, s);
            for (int i = 0; i < s.length; i++) {
                float expect = (i >= base && i < base + n) ? want[i] : orig[i];
                if (Float.floatToIntBits(s[i]) != Float.floatToIntBits(expect)) { if (++bad < 4) System.out.println("    F in-place base=" + base + " n=" + n + " i=" + i); break; }
            }
        }
        return bad;
    }

    public static void main(String[] a) {
        Random rnd = new Random(5);
        System.out.println("SIMD kernels active: VectorMath=" + FactorialVector.isSimd() + ", VectorMathF=" + FactorialVectorF.isSimd() + "   double lanes = " + jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length()
                + ", float lanes = " + jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length());
        int fail = 0, f;

        System.out.println("\n== masked tails (n = 0..80, exact-length src, guarded dest) vs BigDecimal reference ==");
        f = tailsD("double fact", VectorMath::fact, rnd);              fail += f; System.out.println("  VectorMath.fact        (double)        : " + f + " failures");
        f = tailsF("float fact", VectorMathF::fact, SIMD ? 0.0 : 1.0, rnd);         fail += f; System.out.println("  VectorMathF.fact       (float)         : " + f + " failures");
        f = tailsF("float factFast", FactorialVectorF::factFast, 1.0, rnd); fail += f; System.out.println("  VectorMathF.factFast   (float, <=1 ulp): " + f + " failures");

        System.out.println("\n== in-place fact(base, n, s) vs out-of-place (bit-identical; outside [base,base+n) untouched) ==");
        f = inPlaceD(rnd); fail += f; System.out.println("  VectorMath.fact(int,int,double[])  : " + f + " failures");
        f = inPlaceF(rnd); fail += f; System.out.println("  VectorMathF.fact(int,int,float[])  : " + f + " failures");

        // ---- double accuracy
        Ref.Pts pd = Ref.doublePoints();
        int m = pd.xd.length;
        double[] out = new double[m];
        VectorMath.fact(pd.xd, 0, out, 0, m);
        double[] mx = new double[3];
        for (int i = 0; i < m; i++) {
            int b = pd.xd[i] < 20 ? 0 : pd.xd[i] < 160 ? 1 : 2;
            mx[b] = Math.max(mx[b], Math.abs(out[i] / pd.refd[i] - 1));
        }
        System.out.println("\n== double accuracy: max relative error vs 60-digit reference (" + m + " non-integer points) ==");
        String[] nd = {"[0,20)", "[20,160)", "[160,170.6]"};
        for (int b = 0; b < 3; b++) { System.out.printf("  %-12s %.2e%n", nd[b], mx[b]); if (mx[b] > TOL_D) fail++; }
        double[] ints = new double[171], io = new double[171];
        for (int i = 0; i <= 170; i++) ints[i] = i;
        VectorMath.fact(ints, 0, io, 0, 171);
        int ib = 0;
        for (int i = 0; i <= 170; i++) if (io[i] != Ref.expectedD(i)) ib++;
        System.out.println("  integers 0..170: " + ib + " not correctly rounded (of 171)");
        fail += ib;

        // ---- float accuracy
        Ref.Pts pf = Ref.floatPoints();
        int mf = pf.xf.length;
        float[][] fo = new float[2][mf];
        VectorMathF.fact(pf.xf, 0, fo[0], 0, mf);
        FactorialVectorF.factFast(pf.xf, 0, fo[1], 0, mf);
        String[] nf = {"fact     (accurate, widened)", "factFast (native float)  "};
        System.out.println("\n== float accuracy vs correctly rounded reference (" + mf + " non-integer points) ==");
        for (int v = 0; v < 2; v++) {
            double mxu = 0; int exact = 0;
            for (int i = 0; i < mf; i++) { double u = ulpF(fo[v][i], pf.reff[i]); if (u == 0) exact++; mxu = Math.max(mxu, u); }
            System.out.printf("  %s  bit-exact %6.2f%%   max %.3f ulp%n", nf[v], 100.0 * exact / mf, mxu);
            if (mxu > (v == 0 && SIMD ? 0.0 : 1.0)) fail++;
        }
        int sb = 0;
        float[] sp = POOL_F.clone(), so = new float[sp.length], so2 = new float[sp.length];
        VectorMathF.fact(sp, 0, so, 0, sp.length);
        FactorialVectorF.factFast(sp, 0, so2, 0, sp.length);
        for (int i = 0; i < sp.length; i++) {
            float want = Ref.expectedF(sp[i]);
            if (ulpF(so[i], want) > (SIMD ? 0.0 : 1.0) || ulpF(so2[i], want) > 1.0) { sb++; System.out.println("    special MISMATCH x=" + sp[i] + " want=" + want + " fact=" + so[i] + " factFast=" + so2[i]); }
        }
        System.out.println("  special values (inf/NaN/negatives/overflow edge/-0/subnormal): " + sb + " mismatches of " + sp.length);
        fail += sb;
        int sbd = 0;
        double[] spd = POOL_D.clone(), sod = new double[spd.length];
        VectorMath.fact(spd, 0, sod, 0, spd.length);
        for (int i = 0; i < spd.length; i++) if (!sameD(sod[i], Ref.expectedD(spd[i]), TOL_D)) { sbd++; System.out.println("    special MISMATCH x=" + spd[i] + " want=" + Ref.expectedD(spd[i]) + " got=" + sod[i]); }
        System.out.println("  double special values: " + sbd + " mismatches of " + spd.length);
        fail += sbd;

        System.out.println("\n" + (fail == 0 ? "ALL CHECKS PASSED" : "FAILED CHECKS: " + fail));
        if (fail != 0) System.exit(1);
    }
}