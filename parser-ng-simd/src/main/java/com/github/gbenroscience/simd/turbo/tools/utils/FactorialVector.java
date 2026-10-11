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
package com.github.gbenroscience.simd.turbo.tools.utils;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.math.BigInteger;
import java.util.Objects; 

/**
 * Vectorised double-precision factorial, p! = Gamma(p + 1).
 *
 * @author GBEMIRO
 * <ul>
 *   <li>Self-contained: depends on nothing but the JDK (no other project class).</li>
 *   <li>Architecture-agnostic: uses only {@code DoubleVector.SPECIES_PREFERRED}, so it adapts to SSE, AVX2,
 *       AVX-512, NEON, SVE... Nothing is hard-wired to a vector width or instruction set, and it avoids
 *       vector conversions that only exist natively on some CPUs (no D2L / L2D). On platforms where the Vector
 *       API would run as a slow Java emulation (x86 without AVX2+FMA, unknown architectures) it automatically
 *       uses a scalar loop instead; see {@link #isSimd()} and the {@code -Dvectormath.simd} override.</li>
 *   <li>Semantics: integers 0..170 come from a correctly rounded table; non-integers use the "indirect
 *       Stirling" method (shift up with the exact recurrence, Stirling in double-double, divide out);
 *       p &gt; 170.6 -&gt; +Infinity; negative integer -&gt; -Infinity; negative non-integer / NaN -&gt; NaN.</li>
 *   <li>Accuracy: max relative error ~2e-14 for non-integers on the SIMD path (~1e-13 on the scalar fallback).</li>
 * </ul>
 * Requires {@code --add-modules jdk.incubator.vector}.
 */
public final class FactorialVector {
    private FactorialVector() { }

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int LANES_PREFERRED = SPECIES.length();
    /** SIMD kernels are used only where the Vector API is genuinely intrinsified (see {@link #detectSimd()}). */
    private static final boolean SIMD = detectSimd();

    /** 0!..170! correctly rounded (exact BigInteger arithmetic, so no accumulated multiplication error). */
    private static final double[] FACT_TABLE = new double[171];
    static {
        BigInteger f = BigInteger.ONE;
        FACT_TABLE[0] = 1.0;
        for (int i = 1; i <= 170; i++) {
            f = f.multiply(BigInteger.valueOf(i));
            FACT_TABLE[i] = f.doubleValue();
        }
    }

    private static final double MAGIC = 6755399441055744.0;                 // 1.5 * 2^52: round-to-nearest trick
    private static final long MAGIC_BITS = Double.doubleToRawLongBits(MAGIC);
    private static final double LN2_HI = 6.93147180369123816490e-01;
    private static final double LN2_LO = 1.90821492927058770002e-10;
    private static final double LOG2E = 1.4426950408889634;
    private static final double SQRT2 = 1.4142135623730951;
    private static final double HALF_LN_2PI = 0.9189385332046728;           // nearest double to 0.5*ln(2*pi)
    private static final double HALF_LN_2PI_LO = -3.8782941580672414e-17;    // exact - HALF_LN_2PI
    private static final double E2 = 1.0 / 2, E3 = 1.0 / 6, E4 = 1.0 / 24, E5 = 1.0 / 120, E6 = 1.0 / 720,
            E7 = 1.0 / 5040, E8 = 1.0 / 40320, E9 = 1.0 / 362880, E10 = 1.0 / 3628800, E11 = 1.0 / 39916800,
            E12 = 1.0 / 479001600, E13 = 1.0 / 6227020800.0;

    private static DoubleVector bc(double v) { return DoubleVector.broadcast(SPECIES, v); }

    // ------------------------------------------------------------------------------------------------
    //  Capability check: the Vector API is only fast where the CPU/JIT have real intrinsics for the lane
    //  operations used here. Elsewhere (x86 without AVX2/FMA, unknown architectures) it silently degrades to a
    //  per-lane Java emulation that can be 100x slower than scalar code, so those platforms use the scalar path.
    //  Override with -Dvectormath.simd=true|false.
    // ------------------------------------------------------------------------------------------------

    private static boolean detectSimd() {
        String forced = System.getProperty("vectormath.simd");
        if (forced != null) return Boolean.parseBoolean(forced);
        if (LANES_PREFERRED < 2) return false;
        String arch = System.getProperty("os.arch", "").toLowerCase(java.util.Locale.ROOT);
        try {
            if (arch.equals("aarch64") || arch.equals("arm64")) return true;        // NEON / SVE: fma + integer lanes are intrinsified
            if (arch.equals("amd64") || arch.equals("x86_64")) {
                return Integer.parseInt(vmOption("UseAVX")) >= 2 && Boolean.parseBoolean(vmOption("UseFMA"));
            }
        } catch (Throwable ignored) {
            // management API unavailable (e.g. trimmed runtime image): be conservative
        }
        return false;
    }

    private static String vmOption(String name) throws Exception {
        Class<?> c = Class.forName("com.sun.management.HotSpotDiagnosticMXBean");
        Object mx = java.lang.management.ManagementFactory.getPlatformMXBean(
                c.asSubclass(java.lang.management.PlatformManagedObject.class));
        Object opt = c.getMethod("getVMOption", String.class).invoke(mx, name);
        return (String) opt.getClass().getMethod("getValue").invoke(opt);
    }

    /** True when the SIMD kernels are active; false when this platform uses the scalar fallback. */
    public static boolean isSimd() { return SIMD; }

    // ------------------------------------------------------------------------------------------------
    //  Public API
    // ------------------------------------------------------------------------------------------------

    /** dest[destBase + i] = src[srcBase + i]! for i in [0, n). {@code src} and {@code dest} may be the same array. */
    public static void fact(double[] src, int srcBase, double[] dest, int destBase, int n) {
        Objects.checkFromIndexSize(srcBase, n, src.length);
        Objects.checkFromIndexSize(destBase, n, dest.length);
        if (!SIMD) {
            for (int i = 0; i < n; i++) dest[destBase + i] = factScalar(src[srcBase + i]);
            return;
        }
        final int L = SPECIES.length();
        final double[] xa = new double[L], tbl = new double[L];
        int i = 0, limit = SPECIES.loopBound(n);
        for (; i < limit; i += L) {
            DoubleVector x = DoubleVector.fromArray(SPECIES, src, srcBase + i);
            if (allTable(x)) {                                       // hot path: straight table lookups
                for (int l = 0; l < L; l++) dest[destBase + i + l] = FACT_TABLE[(int) src[srcBase + i + l]];
                continue;
            }
            factVec(x, xa, tbl).intoArray(dest, destBase + i);
        }
        if (i < n) {                                                 // masked tail: same kernel, no scalar loop
            VectorMask<Double> tm = SPECIES.indexInRange(i, n);
            DoubleVector x = DoubleVector.fromArray(SPECIES, src, srcBase + i, tm);   // masked-off lanes = 0
            factVec(x, xa, tbl).intoArray(dest, destBase + i, tm);
        }
    }

    /** In place: s[base + i] = s[base + i]! for i in [0, n). */
    public static void fact(int base, int n, double[] s) {
        fact(s, base, s, base, n);          // every lane is read before it is overwritten, so aliasing is safe
    }

    // ------------------------------------------------------------------------------------------------
    //  Kernels
    // ------------------------------------------------------------------------------------------------

    private static boolean allTable(DoubleVector x) {
        return x.eq(x.add(MAGIC).sub(MAGIC)).and(x.compare(VectorOperators.GE, 0.0))
                .and(x.compare(VectorOperators.LE, 170.0)).allTrue();
    }

    private static DoubleVector factVec(DoubleVector x, double[] xa, double[] tbl) {
        DoubleVector r = x.add(MAGIC).sub(MAGIC);                    // round to nearest integer
        VectorMask<Double> isInt = x.eq(r);
        VectorMask<Double> ge0 = x.compare(VectorOperators.GE, 0.0);
        VectorMask<Double> tableM = isInt.and(ge0).and(x.compare(VectorOperators.LE, 170.0));
        DoubleVector res = bc(Double.NaN);                           // negative non-integers, NaN
        VectorMask<Double> gammaM = ge0.and(x.compare(VectorOperators.LE, 170.6)).and(tableM.not());
        if (gammaM.anyTrue()) {
            DoubleVector xs = x.blend(0.5, gammaM.not());            // keep dead lanes harmless
            res = res.blend(gammaPlus1(xs), gammaM);
        }
        if (tableM.anyTrue()) {
            x.intoArray(xa, 0);
            for (int l = 0; l < xa.length; l++) {
                double v = xa[l];
                tbl[l] = (v >= 0 && v <= 170 && v == (int) v) ? FACT_TABLE[(int) v] : 0.0;
            }
            res = res.blend(DoubleVector.fromArray(SPECIES, tbl, 0), tableM);
        }
        res = res.blend(Double.NEGATIVE_INFINITY, isInt.and(x.compare(VectorOperators.LT, 0.0)));
        return res.blend(Double.POSITIVE_INFINITY, x.compare(VectorOperators.GT, 170.6));
    }

    /** exp(xhi + xlo) where |xlo| << ulp(xhi); the low part is folded in after the (exact) range reduction. */
    private static DoubleVector vexp(DoubleVector xhi, DoubleVector xlo) {
        DoubleVector t = xhi.mul(LOG2E).add(MAGIC);                  // = MAGIC + round(x / ln2)
        DoubleVector k = t.sub(MAGIC);
        LongVector kl = t.reinterpretAsLongs().sub(MAGIC_BITS);      // k as integer lanes, no conversion op
        DoubleVector r = k.mul(-LN2_HI).add(xhi).add(k.mul(-LN2_LO).add(xlo));
        DoubleVector p = bc(E13);
        p = p.fma(r, bc(E12)); p = p.fma(r, bc(E11)); p = p.fma(r, bc(E10)); p = p.fma(r, bc(E9));
        p = p.fma(r, bc(E8));  p = p.fma(r, bc(E7));  p = p.fma(r, bc(E6));  p = p.fma(r, bc(E5));
        p = p.fma(r, bc(E4));  p = p.fma(r, bc(E3));  p = p.fma(r, bc(E2));
        p = p.fma(r, bc(1.0)); p = p.fma(r, bc(1.0));
        // scale by 2^k as 2^(k-1) * 2 so k = 1024 (results near DBL_MAX) cannot overflow the exponent field
        DoubleVector scale = kl.add(1022L).lanewise(VectorOperators.LSHL, 52).reinterpretAsDoubles();
        return p.mul(scale).mul(2.0);
    }

    /** Gamma(p + 1) for p in [0, 170.6]; the indirect Stirling method. */
    private static DoubleVector gammaPlus1(DoubleVector p) {
        final DoubleVector ONE = bc(1.0);
        DoubleVector z = p.add(1.0);
        // integer shift m >= 0 so that y = p + m + 1 >= 24
        DoubleVector m = bc(24.5).sub(z).add(MAGIC).sub(MAGIC).lanewise(VectorOperators.MAX, 0.0);
        int mmax = (int) m.reduceLanes(VectorOperators.MAX);

        // prod = (p+1)(p+2)...(p+m): each factor is p + integer (one rounding), two independent chains for ILP
        DoubleVector p0 = ONE, p1 = ONE;
        for (int i = 0; i < mmax; i += 2) {
            p0 = p0.mul(p.add(i + 1.0), m.compare(VectorOperators.GT, (double) i));
            p1 = p1.mul(p.add(i + 2.0), m.compare(VectorOperators.GT, (double) (i + 1)));
        }
        DoubleVector prod = p0.mul(p1);

        // y = p + (m + 1) as a double-double (TwoSum) so the input is never rounded away
        DoubleVector yi = m.add(1.0);
        DoubleVector y = p.add(yi);
        DoubleVector bb = y.sub(p);
        DoubleVector ylo = p.sub(y.sub(bb)).add(yi.sub(bb));

        DoubleVector iy = ONE.div(y);
        DoubleVector iy2 = iy.mul(iy);
        // Stirling correction: 1/(12y) - 1/(360y^3) + 1/(1260y^5) - 1/(1680y^7) + 1/(1188y^9)
        DoubleVector c = bc(1.0 / 1188);
        c = c.fma(iy2, bc(-1.0 / 1680)); c = c.fma(iy2, bc(1.0 / 1260));
        c = c.fma(iy2, bc(-1.0 / 360));  c = c.fma(iy2, bc(1.0 / 12));
        c = c.mul(iy);

        // ln y = H + T, with H = e*LN2_HI exact and T = e*LN2_LO + ln(m)
        LongVector bits = y.reinterpretAsLongs();
        // exponent as double without a long->double conversion: add the magic bits, reinterpret, subtract magic
        DoubleVector e = bits.lanewise(VectorOperators.LSHR, 52).sub(1023L).add(MAGIC_BITS)
                .reinterpretAsDoubles().sub(MAGIC);
        DoubleVector mm = bits.and(0x000FFFFFFFFFFFFFL).or(0x3FF0000000000000L).reinterpretAsDoubles();
        VectorMask<Double> big = mm.compare(VectorOperators.GT, SQRT2);
        mm = mm.mul(0.5, big);
        e = e.add(1.0, big);
        DoubleVector s = mm.sub(1.0).div(mm.add(1.0));
        DoubleVector s2 = s.mul(s);
        DoubleVector q = bc(1.0 / 21);
        q = q.fma(s2, bc(1.0 / 19)); q = q.fma(s2, bc(1.0 / 17)); q = q.fma(s2, bc(1.0 / 15));
        q = q.fma(s2, bc(1.0 / 13)); q = q.fma(s2, bc(1.0 / 11)); q = q.fma(s2, bc(1.0 / 9));
        q = q.fma(s2, bc(1.0 / 7));  q = q.fma(s2, bc(1.0 / 5));  q = q.fma(s2, bc(1.0 / 3));
        q = q.fma(s2, bc(1.0));
        DoubleVector lnH = e.mul(LN2_HI);
        DoubleVector lnT = e.fma(bc(LN2_LO), s.mul(q).mul(2.0));

        // A = (y - 0.5)(H + T) - y + 0.5 ln(2 pi) + c as a double-double (ahi + alo).
        // w = y - 0.5 is exact; fma recovers the rounding error of each product.
        DoubleVector w = y.sub(0.5);
        DoubleVector ph = w.mul(lnH);
        DoubleVector pl = w.fma(lnH, ph.neg());
        DoubleVector qh = w.mul(lnT);
        DoubleVector ql = w.fma(lnT, qh.neg());
        DoubleVector t1 = ph.sub(y);
        DoubleVector e1 = ph.sub(t1).sub(y);
        DoubleVector t2 = t1.add(qh);
        DoubleVector e2 = qh.sub(t2.sub(t1));
        DoubleVector t3 = t2.add(HALF_LN_2PI);
        DoubleVector e3 = bc(HALF_LN_2PI).sub(t3.sub(t2));
        // lnGamma(yhi + ylo) = lnGamma(yhi) + ylo * psi(yhi), psi(y) ~ ln(y)
        DoubleVector alo = pl.add(ql).add(e1.add(e2).add(e3)).add(c.add(HALF_LN_2PI_LO))
                .add(ylo.mul(lnH.add(lnT)));
        return vexp(t3, alo).div(prod);
    }

    // ------------------------------------------------------------------------------------------------
    //  Scalar fallback (platforms without usable SIMD)
    // ------------------------------------------------------------------------------------------------

    private static final double SQRT_2PI = Math.sqrt(2 * Math.PI);

    private static double factScalar(double p) {
        if (p >= 0 && p <= 170 && p == (int) p) return FACT_TABLE[(int) p];
        if (p != p) return p;
        if (p < 0) return (p == Math.floor(p)) ? Double.NEGATIVE_INFINITY : Double.NaN;
        if (p > 170.6) return Double.POSITIVE_INFINITY;
        double n = Math.floor(p);
        double k = p - n;
        double d;                    // Stirling is applied at d, then the shift is divided out
        int limit;
        if (n > 160) { d = p; limit = 0; } else { d = 160 + k; limit = (int) (160 - n); }
        double invD = 1.0 / d;
        double series = 1.0 + invD * (1.0 / 12.0 + invD * (1.0 / 288.0 + invD * (-139.0 / 51840.0
                + invD * (-571.0 / 2488320.0 + invD * (163879.0 / 209018880.0)))));
        double fact1 = Math.pow(d / Math.E, d) * SQRT_2PI * Math.sqrt(d) * series;
        double prod = 1.0;
        for (int j = 1; j <= limit; j++) prod *= (p + j);
        return fact1 / prod;
    }
}