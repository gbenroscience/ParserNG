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

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

import java.math.BigInteger;
import java.util.Objects;
/**
 * Vectorised single-precision factorial, p! = Gamma(p + 1).
 *
 * @author GBEMIRO
 * <ul>
 *   <li>Self-contained: depends on nothing but the JDK (not on VectorMath or any other project class).</li>
 *   <li>Architecture-agnostic: species come from the platform's preferred shape (SSE, AVX2, AVX-512, NEON,
 *       SVE...). It uses no vector conversions that only exist natively on some CPUs (no D2L/L2D/F2I/I2F). On
 *       platforms where the Vector API would run as a slow Java emulation (x86 without AVX2+FMA, unknown
 *       architectures) it automatically uses a scalar loop; see {@link #isSimd()} and {@code -Dvectormath.simd}.</li>
 *   <li>Range: a float overflows at 35!, so integers 0..34 come from a correctly rounded table, and
 *       p &gt; 34.0400963 (the exact float overflow point) gives +Infinity. Negative integer -&gt; -Infinity,
 *       negative non-integer / NaN -&gt; NaN.</li>
 *   <li>{@link #fact(float[], int, float[], int, int)}: float in, double-precision kernel, correctly rounded float out
 *       (0 ulp error in testing). This is the recommended method.</li>
 *   <li>{@link #factFast(float[], int, float[], int, int)}: all 32-bit lanes with float-float arithmetic;
 *       faithfully rounded (max 1 ulp, ~94% correctly rounded). Not faster than {@code fact} on the hardware tested.</li>
 * </ul>
 * Requires {@code --add-modules jdk.incubator.vector}.
 */
public final class FactorialVectorF {
    private FactorialVectorF() { }

    /** SIMD kernels are used only where the Vector API is genuinely intrinsified (see {@link #detectSimd()}). */
    private static final boolean SIMD = detectSimd();

    /** 0!..34! correctly rounded to float (exact BigInteger arithmetic). */
    private static final float[] FACT_F = new float[35];
    static {
        BigInteger f = BigInteger.ONE;
        FACT_F[0] = 1.0f;
        for (int i = 1; i < FACT_F.length; i++) {
            f = f.multiply(BigInteger.valueOf(i));
            FACT_F[i] = f.floatValue();                              // BigInteger.floatValue() rounds to nearest-even
        }
    }

    // Species for the accurate (widening) kernel: DS has the same number of lanes as FW, FW has half the bits.
    // Only standard power-of-two shapes are used; null means "no usable SIMD here" and selects the scalar loop.
    private static final VectorSpecies<Double> DS;
    private static final VectorSpecies<Float> FW;
    static {
        VectorSpecies<Double> ds = null;
        VectorSpecies<Float> fw = null;
        try {
            int bits = Integer.highestOneBit(DoubleVector.SPECIES_PREFERRED.vectorBitSize());
            if (bits > 512) bits = 512;
            if (SIMD && bits >= 128) {
                ds = VectorSpecies.of(double.class, VectorShape.forBitSize(bits));
                fw = VectorSpecies.of(float.class, VectorShape.forBitSize(bits / 2));
            }
        } catch (RuntimeException ex) {
            ds = null;
            fw = null;
        }
        DS = ds;
        FW = fw;
    }

    // ------------------------------------------------------------------------------------------------
    //  Capability check: the Vector API is only fast where the CPU/JIT have real intrinsics for the lane
    //  operations used here. Elsewhere (x86 without AVX2/FMA, unknown architectures) it silently degrades to a
    //  per-lane Java emulation that can be 100x slower than scalar code, so those platforms use the scalar path.
    //  Override with -Dvectormath.simd=true|false.
    // ------------------------------------------------------------------------------------------------

    private static boolean detectSimd() {
        String forced = System.getProperty("vectormath.simd");
        if (forced != null) return Boolean.parseBoolean(forced);
        if (FloatVector.SPECIES_PREFERRED.length() < 2) return false;
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

    // ================================================================================================
    //  Public API
    // ================================================================================================

    /**
     * Accurate float factorial: dest[destBase + i] = src[srcBase + i]! for i in [0, n), correctly rounded.
     * {@code src} and {@code dest} may be the same array.
     */
    public static void fact(float[] src, int srcBase, float[] dest, int destBase, int n) {
        Objects.checkFromIndexSize(srcBase, n, src.length);
        Objects.checkFromIndexSize(destBase, n, dest.length);
        if (DS == null) {
            for (int i = 0; i < n; i++) dest[destBase + i] = factScalar(src[srcBase + i]);
            return;
        }
        final int L = DS.length();
        final double[] xa = new double[L], tbl = new double[L];
        int i = 0, limit = FW.loopBound(n);
        for (; i < limit; i += L) {
            DoubleVector x = (DoubleVector) FloatVector.fromArray(FW, src, srcBase + i)
                    .convertShape(VectorOperators.F2D, DS, 0);
            if (allTableW(x)) {                                      // hot path: straight float table lookups
                for (int l = 0; l < L; l++) dest[destBase + i + l] = FACT_F[(int) src[srcBase + i + l]];
                continue;
            }
            ((FloatVector) factVecW(x, xa, tbl).convertShape(VectorOperators.D2F, FW, 0))
                    .intoArray(dest, destBase + i);
        }
        if (i < n) {                                                 // masked tail: same kernel, no scalar loop
            VectorMask<Float> tm = FW.indexInRange(i, n);
            DoubleVector x = (DoubleVector) FloatVector.fromArray(FW, src, srcBase + i, tm)
                    .convertShape(VectorOperators.F2D, DS, 0);
            ((FloatVector) factVecW(x, xa, tbl).convertShape(VectorOperators.D2F, FW, 0))
                    .intoArray(dest, destBase + i, tm);
        }
    }

    /** In place: s[base + i] = s[base + i]! for i in [0, n), correctly rounded. */
    public static void fact(int base, int n, float[] s) {
        fact(s, base, s, base, n);          // every lane is read before it is overwritten, so aliasing is safe
    }

    /** Native-float variant (pure 32-bit lanes, float-float arithmetic): faithfully rounded, max 1 ulp. */
    public static void factFast(float[] src, int srcBase, float[] dest, int destBase, int n) {
        Objects.checkFromIndexSize(srcBase, n, src.length);
        Objects.checkFromIndexSize(destBase, n, dest.length);
        if (!SIMD || FN.length() < 2) {
            for (int i = 0; i < n; i++) dest[destBase + i] = factScalar(src[srcBase + i]);
            return;
        }
        final int L = FN.length();
        final float[] xa = new float[L], tbl = new float[L];
        int i = 0, limit = FN.loopBound(n);
        for (; i < limit; i += L) {
            FloatVector x = FloatVector.fromArray(FN, src, srcBase + i);
            if (allTableN(x)) {
                for (int l = 0; l < L; l++) dest[destBase + i + l] = FACT_F[(int) src[srcBase + i + l]];
                continue;
            }
            factVecN(x, xa, tbl).intoArray(dest, destBase + i);
        }
        if (i < n) {
            VectorMask<Float> tm = FN.indexInRange(i, n);
            FloatVector x = FloatVector.fromArray(FN, src, srcBase + i, tm);
            factVecN(x, xa, tbl).intoArray(dest, destBase + i, tm);
        }
    }

    // ================================================================================================
    //  Accurate kernel: widen to double, indirect Stirling with a shift to y >= 10, narrow back
    // ================================================================================================

    private static final double MAGIC = 6755399441055744.0;              // 1.5 * 2^52
    private static final long MAGIC_BITS = Double.doubleToRawLongBits(MAGIC);
    private static final double LN2_HI = 6.93147180369123816490e-01;
    private static final double LN2_LO = 1.90821492927058770002e-10;
    private static final double LOG2E = 1.4426950408889634;
    private static final double SQRT2 = 1.4142135623730951;
    private static final double HALF_LN_2PI = 0.9189385332046728;
    private static final double E2 = 1.0 / 2, E3 = 1.0 / 6, E4 = 1.0 / 24, E5 = 1.0 / 120, E6 = 1.0 / 720,
            E7 = 1.0 / 5040, E8 = 1.0 / 40320, E9 = 1.0 / 362880, E10 = 1.0 / 3628800, E11 = 1.0 / 39916800,
            E12 = 1.0 / 479001600, E13 = 1.0 / 6227020800.0;

    private static DoubleVector bc(double v) { return DoubleVector.broadcast(DS, v); }

    private static boolean allTableW(DoubleVector x) {
        return x.eq(x.add(MAGIC).sub(MAGIC)).and(x.compare(VectorOperators.GE, 0.0))
                .and(x.compare(VectorOperators.LE, 34.0)).allTrue();
    }

    private static DoubleVector factVecW(DoubleVector x, double[] xa, double[] tbl) {
        DoubleVector r = x.add(MAGIC).sub(MAGIC);
        VectorMask<Double> isInt = x.eq(r);
        VectorMask<Double> ge0 = x.compare(VectorOperators.GE, 0.0);
        VectorMask<Double> tableM = isInt.and(ge0).and(x.compare(VectorOperators.LE, 34.0));
        DoubleVector res = bc(Double.NaN);                           // negative non-integers, NaN
        VectorMask<Double> gammaM = ge0.and(x.compare(VectorOperators.LE, 36.0)).and(tableM.not());
        if (gammaM.anyTrue()) {
            DoubleVector xs = x.blend(0.5, gammaM.not());            // keep dead lanes harmless
            res = res.blend(gammaPlus1W(xs), gammaM);                // > FLT_MAX narrows to +inf below
        }
        if (tableM.anyTrue()) {
            x.intoArray(xa, 0);
            for (int l = 0; l < xa.length; l++) {
                double v = xa[l];
                tbl[l] = (v >= 0 && v <= 34 && v == (int) v) ? (double) FACT_F[(int) v] : 0.0;
            }
            res = res.blend(DoubleVector.fromArray(DS, tbl, 0), tableM);
        }
        res = res.blend(Double.NEGATIVE_INFINITY, isInt.and(x.compare(VectorOperators.LT, 0.0)));
        return res.blend(Double.POSITIVE_INFINITY, x.compare(VectorOperators.GT, 36.0));
    }

    private static DoubleVector vexp(DoubleVector x) {
        DoubleVector t = x.mul(LOG2E).add(MAGIC);                    // = MAGIC + round(x / ln2)
        DoubleVector k = t.sub(MAGIC);
        LongVector kl = t.reinterpretAsLongs().sub(MAGIC_BITS);      // k as integer lanes, no conversion op
        DoubleVector r = k.mul(-LN2_HI).add(x).add(k.mul(-LN2_LO));
        DoubleVector p = bc(E13);
        p = p.fma(r, bc(E12)); p = p.fma(r, bc(E11)); p = p.fma(r, bc(E10)); p = p.fma(r, bc(E9));
        p = p.fma(r, bc(E8));  p = p.fma(r, bc(E7));  p = p.fma(r, bc(E6));  p = p.fma(r, bc(E5));
        p = p.fma(r, bc(E4));  p = p.fma(r, bc(E3));  p = p.fma(r, bc(E2));
        p = p.fma(r, bc(1.0)); p = p.fma(r, bc(1.0));
        DoubleVector scale = kl.add(1022L).lanewise(VectorOperators.LSHL, 52).reinterpretAsDoubles();
        return p.mul(scale).mul(2.0);
    }

    private static DoubleVector vlog(DoubleVector y) {
        LongVector bits = y.reinterpretAsLongs();
        DoubleVector e = bits.lanewise(VectorOperators.LSHR, 52).sub(1023L).add(MAGIC_BITS)
                .reinterpretAsDoubles().sub(MAGIC);
        DoubleVector m = bits.and(0x000FFFFFFFFFFFFFL).or(0x3FF0000000000000L).reinterpretAsDoubles();
        VectorMask<Double> big = m.compare(VectorOperators.GT, SQRT2);
        m = m.mul(0.5, big);
        e = e.add(1.0, big);
        DoubleVector s = m.sub(1.0).div(m.add(1.0));
        DoubleVector s2 = s.mul(s);
        DoubleVector q = bc(1.0 / 15);
        q = q.fma(s2, bc(1.0 / 13)); q = q.fma(s2, bc(1.0 / 11)); q = q.fma(s2, bc(1.0 / 9));
        q = q.fma(s2, bc(1.0 / 7));  q = q.fma(s2, bc(1.0 / 5));  q = q.fma(s2, bc(1.0 / 3));
        q = q.fma(s2, bc(1.0));
        return e.fma(bc(LN2_HI), e.mul(LN2_LO).add(s.mul(q).mul(2.0)));
    }

    /** Gamma(p + 1) for p in [0, 40], p exactly representable (it came from a float). */
    private static DoubleVector gammaPlus1W(DoubleVector p) {
        final DoubleVector ONE = bc(1.0);
        DoubleVector z = p.add(1.0);                                 // exact: p has <= 24 significant bits
        DoubleVector m = bc(10.5).sub(z).add(MAGIC).sub(MAGIC).lanewise(VectorOperators.MAX, 0.0);
        int mmax = (int) m.reduceLanes(VectorOperators.MAX);         // <= 10
        DoubleVector p0 = ONE, p1 = ONE;
        for (int i = 0; i < mmax; i += 2) {
            p0 = p0.mul(p.add(i + 1.0), m.compare(VectorOperators.GT, (double) i));
            p1 = p1.mul(p.add(i + 2.0), m.compare(VectorOperators.GT, (double) (i + 1)));
        }
        DoubleVector prod = p0.mul(p1);
        DoubleVector y = z.add(m);                                   // y in [10, 41], exact
        DoubleVector iy = ONE.div(y);
        DoubleVector iy2 = iy.mul(iy);
        DoubleVector c = bc(1.0 / 1188);
        c = c.fma(iy2, bc(-1.0 / 1680)); c = c.fma(iy2, bc(1.0 / 1260));
        c = c.fma(iy2, bc(-1.0 / 360));  c = c.fma(iy2, bc(1.0 / 12));
        c = c.mul(iy);
        DoubleVector a = y.sub(0.5).fma(vlog(y), y.neg()).add(c.add(HALF_LN_2PI));
        return vexp(a).div(prod);
    }

    // ================================================================================================
    //  Native-float kernel (float-float arithmetic): all lanes are 32-bit
    // ================================================================================================

    private static final VectorSpecies<Float> FN = FloatVector.SPECIES_PREFERRED;
    private static final float MAGIC_F = 12582912.0f;                  // 1.5 * 2^23
    private static final int MAGIC_BITS_F = Float.floatToRawIntBits(MAGIC_F);
    private static final float LN2_HI_F = 0.693359375f;                // 9 significant bits: k*LN2_HI exact
    private static final float LN2_LO_F = -2.1219444170128554e-4f;     // ln2 = LN2_HI_F + LN2_LO_F
    private static final float LOG2E_F = 1.4426950216293335f;
    private static final float SQRT2_F = 1.4142135381698608f;
    private static final float HALF_LN_2PI_HI_F = 0.9189385175704956f;
    private static final float HALF_LN_2PI_LO_F = 1.563417661998301e-08f;
    private static final float XMAX_F = 34.040096282958984f;           // largest float p with p! < 2^128 - 2^103
    private static final float TWO23_F = 8388608f;

    private static FloatVector fb(float x) { return FloatVector.broadcast(FN, x); }

    /** Rounding error of s = a + b (TwoSum, valid for any ordering of |a|, |b|). */
    private static FloatVector tsErr(FloatVector a, FloatVector b, FloatVector s) {
        FloatVector bb = s.sub(a);
        return a.sub(s.sub(bb)).add(b.sub(bb));
    }

    /** Gamma(p + 1) for float lanes p in [0, XMAX_F]. */
    private static FloatVector gammaNative(FloatVector p) {
        final FloatVector ONE = fb(1f);
        FloatVector z = p.add(1f);
        // integer shift m >= 0 with y = p + m + 1 in [5, 6] (or p + 1 when p >= 4)
        FloatVector m = fb(5.5f).sub(z).add(MAGIC_F).sub(MAGIC_F).lanewise(VectorOperators.MAX, 0f);
        int mmax = (int) m.reduceLanes(VectorOperators.MAX);           // <= 4

        // P = (p+1)(p+2)...(p+m) as a float-float (Ph + Pl)
        FloatVector Ph = ONE, Pl = fb(0f);
        for (int i = 1; i <= mmax; i++) {
            VectorMask<Float> act = m.compare(VectorOperators.GE, (float) i);
            FloatVector ti = fb((float) i);
            FloatVector th = p.add(ti);
            FloatVector tl = tsErr(p, ti, th);
            FloatVector nh = Ph.mul(th);
            FloatVector ne = Ph.fma(th, nh.neg());                      // exact error of Ph*th
            FloatVector nl = Ph.fma(tl, Pl.fma(th, ne));
            Ph = Ph.blend(nh, act);
            Pl = Pl.blend(nl, act);
        }

        // y = yh + yl (TwoSum) so the input is never rounded away
        FloatVector yi = m.add(1f);
        FloatVector yh = p.add(yi);
        FloatVector yl = tsErr(p, yi, yh);

        // Stirling correction c = 1/(12y) - 1/(360y^3) + 1/(1260y^5) - 1/(1680y^7) + 1/(1188y^9)
        FloatVector iy = ONE.div(yh);
        FloatVector iy2 = iy.mul(iy);
        FloatVector c = fb(1f / 1188);
        c = c.fma(iy2, fb(-1f / 1680)); c = c.fma(iy2, fb(1f / 1260));
        c = c.fma(iy2, fb(-1f / 360));  c = c.fma(iy2, fb(1f / 12));
        c = c.mul(iy);

        // ln(yh) = LH + U + tail:  LH = e*LN2_HI (exact), U = 2*sh (exact), tail small (<~5e-3)
        IntVector bits = yh.reinterpretAsInts();
        FloatVector e = bits.lanewise(VectorOperators.LSHR, 23).sub(127).add(MAGIC_BITS_F)
                .reinterpretAsFloats().sub(MAGIC_F);                    // exponent as float, no int->float convert
        FloatVector mm = bits.and(0x007FFFFF).or(0x3F800000).reinterpretAsFloats();
        VectorMask<Float> big = mm.compare(VectorOperators.GT, SQRT2_F);
        mm = mm.mul(0.5f, big);
        e = e.add(1f, big);
        FloatVector num = mm.sub(1f);                                   // exact
        FloatVector dh = mm.add(1f);
        FloatVector dl = tsErr(mm, ONE, dh);
        FloatVector sh = num.div(dh);                                   // s = (m-1)/(m+1) = sh + sl
        FloatVector rem = sh.neg().fma(dl, sh.neg().fma(dh, num));
        FloatVector sl = rem.div(dh);
        FloatVector s2 = sh.mul(sh);
        FloatVector q = fb(1f / 11);
        q = q.fma(s2, fb(1f / 9)); q = q.fma(s2, fb(1f / 7));
        q = q.fma(s2, fb(1f / 5)); q = q.fma(s2, fb(1f / 3));
        FloatVector tail = e.fma(fb(LN2_LO_F), sh.mul(s2).mul(2f).fma(q, sl.mul(2f)));
        FloatVector LH = e.mul(LN2_HI_F);
        FloatVector U = sh.mul(2f);

        // A = (yh-0.5)(LH+U+tail) - yh + 0.5 ln(2pi) + c + yl*psi(yh), as ahi + alo
        FloatVector wh = yh.sub(0.5f);                                  // exact
        FloatVector ph = wh.mul(LH),   pl = wh.fma(LH, ph.neg());
        FloatVector uh = wh.mul(U),    ul = wh.fma(U, uh.neg());
        FloatVector th = wh.mul(tail), tl = wh.fma(tail, th.neg());
        FloatVector nyh = yh.neg();
        FloatVector s1 = ph.add(nyh);          FloatVector e1 = tsErr(ph, nyh, s1);
        FloatVector s2s = s1.add(uh);          FloatVector e2 = tsErr(s1, uh, s2s);
        FloatVector s3 = s2s.add(th);          FloatVector e3 = tsErr(s2s, th, s3);
        FloatVector kh = fb(HALF_LN_2PI_HI_F);
        FloatVector s4 = s3.add(kh);           FloatVector e4 = tsErr(s3, kh, s4);
        FloatVector psi = LH.add(U).add(tail).sub(iy.mul(0.5f));
        FloatVector alo = pl.add(ul).add(tl).add(e1.add(e2)).add(e3.add(e4))
                .add(c.add(fb(HALF_LN_2PI_LO_F)).add(yl.mul(psi)));
        FloatVector ahi = s4;

        // exp(ahi + alo) = 2^k * exp(r), r = ahi + alo - k ln2 kept as rh + rl
        FloatVector t = ahi.mul(LOG2E_F).add(MAGIC_F);                  // = MAGIC + round(a / ln2)
        FloatVector k = t.sub(MAGIC_F);
        IntVector ki = t.reinterpretAsInts().sub(MAGIC_BITS_F);         // k as integer lanes, no conversion op
        FloatVector rr = k.neg().fma(fb(LN2_HI_F), ahi);                // exact
        FloatVector kLh = k.mul(LN2_LO_F);
        FloatVector kLl = k.fma(fb(LN2_LO_F), kLh.neg());
        FloatVector rh0 = rr.sub(kLh);
        FloatVector lo2 = tsErr(rr, kLh.neg(), rh0).add(alo.sub(kLl));
        FloatVector rh = rh0.add(lo2);
        FloatVector rl = lo2.sub(rh.sub(rh0));

        FloatVector g = fb(1f / 40320);
        g = g.fma(rh, fb(1f / 5040)); g = g.fma(rh, fb(1f / 720)); g = g.fma(rh, fb(1f / 120));
        g = g.fma(rh, fb(1f / 24));   g = g.fma(rh, fb(1f / 6));   g = g.fma(rh, fb(0.5f));
        FloatVector tt = rh.mul(rh).fma(g, rh);                         // exp(rh) = 1 + tt
        FloatVector eh = tt.add(1f);
        FloatVector el = tt.sub(eh.sub(1f)).add(rl.mul(eh));            // exp(rh+rl) = eh + el

        // result = (eh + el) / (Ph + Pl) * 2^k, one reciprocal and a residual correction
        FloatVector rcp = ONE.div(Ph);
        FloatVector qv = eh.mul(rcp);
        FloatVector resid = qv.neg().fma(Ph, eh).sub(qv.mul(Pl)).add(el);
        FloatVector res = resid.fma(rcp, qv);
        FloatVector scale = ki.add(126).lanewise(VectorOperators.LSHL, 23).reinterpretAsFloats();   // 2^(k-1)
        return res.mul(scale).mul(2f);
    }

    private static boolean allTableN(FloatVector x) {
        return x.eq(x.add(MAGIC_F).sub(MAGIC_F)).and(x.compare(VectorOperators.GE, 0f))
                .and(x.compare(VectorOperators.LE, 34f)).allTrue();
    }

    private static FloatVector factVecN(FloatVector x, float[] xa, float[] tbl) {
        FloatVector r = x.add(MAGIC_F).sub(MAGIC_F);
        // every float with |x| >= 2^23 is an integer (guards the rint trick against rounding)
        VectorMask<Float> isInt = x.eq(r).or(x.abs().compare(VectorOperators.GE, TWO23_F));
        VectorMask<Float> ge0 = x.compare(VectorOperators.GE, 0f);
        VectorMask<Float> tableM = isInt.and(ge0).and(x.compare(VectorOperators.LE, 34f));
        FloatVector res = fb(Float.NaN);
        VectorMask<Float> gammaM = ge0.and(x.compare(VectorOperators.LE, XMAX_F)).and(tableM.not());
        if (gammaM.anyTrue()) {
            FloatVector xs = x.blend(0.5f, gammaM.not());
            res = res.blend(gammaNative(xs), gammaM);
        }
        if (tableM.anyTrue()) {
            x.intoArray(xa, 0);
            for (int l = 0; l < xa.length; l++) {
                float v = xa[l];
                tbl[l] = (v >= 0 && v <= 34 && v == (int) v) ? FACT_F[(int) v] : 0f;
            }
            res = res.blend(FloatVector.fromArray(FN, tbl, 0), tableM);
        }
        res = res.blend(Float.NEGATIVE_INFINITY, isInt.and(x.compare(VectorOperators.LT, 0f)));
        return res.blend(Float.POSITIVE_INFINITY, x.compare(VectorOperators.GT, XMAX_F));
    }

    // ================================================================================================
    //  Scalar fallback (platforms without usable SIMD): indirect Stirling in double, rounded to float
    // ================================================================================================

    private static final double SQRT_2PI = Math.sqrt(2 * Math.PI);

    private static float factScalar(float x) {
        if (x >= 0 && x <= 34 && x == (int) x) return FACT_F[(int) x];
        if (x != x) return x;
        if (x < 0) return (x == Math.floor(x)) ? Float.NEGATIVE_INFINITY : Float.NaN;
        if (x > XMAX_F) return Float.POSITIVE_INFINITY;
        double p = x;
        double n = Math.floor(p);
        double k = p - n;
        double d = 160 + k;                                          // p <= 34.05, so the shift is always needed
        int limit = (int) (160 - n);
        double invD = 1.0 / d;
        double series = 1.0 + invD * (1.0 / 12.0 + invD * (1.0 / 288.0 + invD * (-139.0 / 51840.0
                + invD * (-571.0 / 2488320.0 + invD * (163879.0 / 209018880.0)))));
        double fact1 = Math.pow(d / Math.E, d) * SQRT_2PI * Math.sqrt(d) * series;
        double prod = 1.0;
        for (int j = 1; j <= limit; j++) prod *= (p + j);
        return (float) (fact1 / prod);
    }
}