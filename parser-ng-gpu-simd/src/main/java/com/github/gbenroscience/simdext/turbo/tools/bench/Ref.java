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

 
/**
 *

 */ 

/**
 * High-precision reference for Gamma(p + 1) = p!  (60 significant digits, java.math only).
 * Same idea as the production code, with the precision turned up: shift the argument to y >= 60 with the
 * exact recurrence, apply Stirling with 12 Bernoulli terms (error ~1e-38), divide the shift back out.
 * Used by Bench / BenchF / Test2 to generate their test points, so no external data files are needed.
 */import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Random;

/** 
 * @author GBEMIRO
 * High-precision reference for Gamma(p + 1) = p!  (60 significant digits, java.math only).
 * Same idea as the production code, with the precision turned up: shift the argument to y >= 60 with the
 * exact recurrence, apply Stirling with 12 Bernoulli terms (error ~1e-38), divide the shift back out.
 * Used by Bench / BenchF / Test2 to generate their test points, so no external data files are needed.
 */
public final class Ref {
    private static final MathContext MC = new MathContext(60);
    private static final BigDecimal ONE = BigDecimal.ONE, TWO = BigDecimal.valueOf(2), HALF = new BigDecimal("0.5");
    private static final BigDecimal EPS = new BigDecimal("1e-66");
    private static final BigDecimal PI = new BigDecimal(
            "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899");
    private static final BigDecimal LN2 = atanh(ONE.divide(BigDecimal.valueOf(3), MC)).multiply(TWO, MC);
    private static final BigDecimal HALF_LN_2PI = ln(TWO.multiply(PI, MC)).divide(TWO, MC);
    // B_2k = num/den, k = 1..12
    private static final long[][] BERN = {{1, 6}, {-1, 30}, {1, 42}, {-1, 30}, {5, 66}, {-691, 2730}, {7, 6},
            {-3617, 510}, {43867, 798}, {-174611, 330}, {854513, 138}, {-236364091, 2730}};

    private static BigDecimal atanh(BigDecimal s) {                    // s + s^3/3 + s^5/5 + ...
        BigDecimal s2 = s.multiply(s, MC), term = s, sum = s;
        for (int k = 1; k < 1000; k++) {
            term = term.multiply(s2, MC);
            BigDecimal t = term.divide(BigDecimal.valueOf(2L * k + 1), MC);
            sum = sum.add(t, MC);
            if (t.abs().compareTo(EPS) < 0) break;
        }
        return sum;
    }

    static BigDecimal ln(BigDecimal x) {
        int e = 0;
        BigDecimal m = x, hi = new BigDecimal("1.5"), lo = new BigDecimal("0.75");
        while (m.compareTo(hi) >= 0) { m = m.divide(TWO, MC); e++; }
        while (m.compareTo(lo) < 0) { m = m.multiply(TWO, MC); e--; }
        BigDecimal s = m.subtract(ONE).divide(m.add(ONE), MC);
        BigDecimal r = atanh(s).multiply(TWO, MC);
        return e == 0 ? r : r.add(LN2.multiply(BigDecimal.valueOf(e), MC), MC);   // LN2 is null while initializing LN2 itself
    }

    static BigDecimal exp(BigDecimal x) {
        BigInteger k = x.divide(LN2, MC).setScale(0, RoundingMode.HALF_EVEN).toBigInteger();
        BigDecimal r = x.subtract(LN2.multiply(new BigDecimal(k), MC), MC);
        BigDecimal sum = ONE, term = ONE;
        for (int n = 1; n < 400; n++) {
            term = term.multiply(r, MC).divide(BigDecimal.valueOf(n), MC);
            sum = sum.add(term, MC);
            if (term.abs().compareTo(EPS) < 0) break;
        }
        int ki = k.intValueExact();
        BigDecimal pow2 = new BigDecimal(BigInteger.ONE.shiftLeft(Math.abs(ki)));
        return ki >= 0 ? sum.multiply(pow2, MC) : sum.divide(pow2, MC);
    }

    private static BigDecimal lnGamma(BigDecimal y) {                  // y >= 60
        BigDecimal res = y.subtract(HALF).multiply(ln(y), MC).subtract(y, MC).add(HALF_LN_2PI, MC);
        BigDecimal y2 = y.multiply(y, MC), pw = y;                     // pw = y^(2k-1)
        for (int k = 1; k <= BERN.length; k++) {
            BigDecimal b = BigDecimal.valueOf(BERN[k - 1][0]).divide(BigDecimal.valueOf(BERN[k - 1][1]), MC);
            BigDecimal d = BigDecimal.valueOf(2L * k * (2L * k - 1));
            res = res.add(b.divide(d.multiply(pw, MC), MC), MC);
            pw = pw.multiply(y2, MC);
        }
        return res;
    }

    /** p! = Gamma(p + 1) for p >= 0 (p taken exactly as the given double). */
    public static BigDecimal factorial(double p) {
        BigDecimal z = new BigDecimal(p).add(ONE);
        int m = Math.max(0, (int) Math.ceil(60 - z.doubleValue()));
        BigDecimal prod = ONE;
        for (int i = 0; i < m; i++) prod = prod.multiply(z.add(BigDecimal.valueOf(i)), MC);
        return exp(lnGamma(z.add(BigDecimal.valueOf(m)))).divide(prod, MC);
    }

    // ---------------------------------------------------------------------------------------------
    //  Deterministic test sets (what the old ref.txt / reff.txt contained)
    // ---------------------------------------------------------------------------------------------
    public static final class Pts {
        public double[] xd, refd;      // double test set
        public float[] xf, reff;       // float test set (reff = correctly rounded float of p!)
    }

    private static final BigDecimal FLT_OVERFLOW =
            new BigDecimal(BigInteger.ONE.shiftLeft(128).subtract(BigInteger.ONE.shiftLeft(103)));

    /** Non-integer doubles in [0,20), [20,160), [160,170.6] plus boundary points; length is a multiple of 16. */
    public static Pts doublePoints() {
        Random r = new Random(1);
        double[][] spec = {{0, 20, 2500}, {20, 160, 2500}, {160, 170.6, 2500}};
        double[] extra = {0.5, 1.5, 170.5, 170.6, 160.0000001, 161.5, 165.5, 0.0001, 23.99, 24.01};
        int n = extra.length;
        for (double[] s : spec) n += (int) s[2];
        n -= n % 16;
        double[] x = new double[n];
        int k = 0;
        for (double[] s : spec) for (int i = 0; i < (int) s[2] && k < n; i++) x[k++] = s[0] + r.nextDouble() * (s[1] - s[0]);
        for (int i = 0; i < extra.length && k < n; i++) x[k++] = extra[i];
        while (k < n) x[k++] = 0.75;
        Pts p = new Pts();
        p.xd = x;
        p.refd = new double[n];
        for (int i = 0; i < n; i++) p.refd[i] = factorial(x[i]).doubleValue();
        return p;
    }

    /** Non-integer floats in [0,1), [1,10), [10,34) plus the overflow boundary; length is a multiple of 16. */
    public static Pts floatPoints() {
        Random r = new Random(7);
        double[][] spec = {{0, 1, 1500}, {1, 10, 2000}, {10, 34, 3000}, {34, 34.06, 500}};
        float[] extra = {1e-30f, 1e-8f, 1e-5f, 0.5f, 9.5f, 10.5f, 23.99f, 33.99f, 34.01f, 34.03f, 34.04f, 34.05f, 34.1f,
                35.5f, 34.0399f, 34.040096f, 34.0401f};
        int n = extra.length;
        for (double[] s : spec) n += (int) s[2];
        n -= n % 16;
        float[] x = new float[n];
        int k = 0;
        for (double[] s : spec) for (int i = 0; i < (int) s[2] && k < n; i++) x[k++] = (float) (s[0] + r.nextDouble() * (s[1] - s[0]));
        for (int i = 0; i < extra.length && k < n; i++) x[k++] = extra[i];
        while (k < n) x[k++] = 1.5f;
        Pts p = new Pts();
        p.xf = x;
        p.reff = new float[n];
        for (int i = 0; i < n; i++) {
            BigDecimal v = factorial(x[i]);
            p.reff[i] = v.compareTo(FLT_OVERFLOW) >= 0 ? Float.POSITIVE_INFINITY : (float) v.doubleValue();
        }
        return p;
    }


    // ---------------------------------------------------------------------------------------------
    //  Expected results with the documented semantics of VectorMath / VectorMathF
    // ---------------------------------------------------------------------------------------------

    private static BigDecimal exactIntFactorial(int n) {
        BigInteger f = BigInteger.ONE;
        for (int i = 2; i <= n; i++) f = f.multiply(BigInteger.valueOf(i));
        return new BigDecimal(f);
    }

    /** VectorMath.fact semantics: p > 170.6 -> +inf; negative integer -> -inf; negative non-integer / NaN -> NaN. */
    public static double expectedD(double p) {
        if (Double.isNaN(p)) return Double.NaN;
        if (p < 0) return p == Math.floor(p) ? Double.NEGATIVE_INFINITY : Double.NaN;
        if (p > 170.6) return Double.POSITIVE_INFINITY;
        return (p == Math.rint(p) ? exactIntFactorial((int) p) : factorial(p)).doubleValue();
    }

    /** VectorMathF.fact semantics: correctly rounded p!, +inf once p! rounds past FLT_MAX. */
    public static float expectedF(float p) {
        if (Float.isNaN(p)) return Float.NaN;
        if (p < 0) return p == Math.floor(p) ? Float.NEGATIVE_INFINITY : Float.NaN;
        if (p > 36f) return Float.POSITIVE_INFINITY;
        BigDecimal v = p == Math.rint(p) ? exactIntFactorial((int) p) : factorial(p);
        return v.compareTo(FLT_OVERFLOW) >= 0 ? Float.POSITIVE_INFINITY : (float) v.doubleValue();
    }

    public static void main(String[] a) {                              // sanity check
        System.out.println("0.5!  = " + factorial(0.5).round(new MathContext(30)) + "   (expect 0.886226925452758013649083741671)");
        System.out.println("170.5! = " + factorial(170.5).round(new MathContext(30)));
        long t0 = System.nanoTime();
        doublePoints(); floatPoints();
        System.out.printf("generated both reference sets in %.1f s%n", (System.nanoTime() - t0) / 1e9);
    }
}