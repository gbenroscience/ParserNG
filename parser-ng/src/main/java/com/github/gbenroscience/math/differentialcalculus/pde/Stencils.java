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
package com.github.gbenroscience.math.differentialcalculus.pde;

/**
 * Finite-difference weight generation (Fornberg's algorithm) and the
 * one-dimensional stencils used by the discretization.
 */
final class Stencils {

    private Stencils() {
    }

    /** One-dimensional stencil: integer node offsets and UNIT-SPACING weights. */
    static final class Stencil1D {

        final int[] offsets;
        final double[] weights;

        Stencil1D(int[] offsets, double[] weights) {
            this.offsets = offsets;
            this.weights = weights;
        }

        int reach() {
            int r = 0;
            for (int o : offsets) {
                r = Math.max(r, Math.abs(o));
            }
            return r;
        }
    }

    /**
     * Fornberg (1988) weights of the {@code m}-th derivative at {@code z=0}
     * from function values at the given (arbitrary, distinct) offsets.
     */
    static double[] weights(int m, double[] x) {
        int n = x.length;
        if (n <= m) {
            throw new PDEConfigurationException("Need at least " + (m + 1) + " points for a derivative of order " + m);
        }
        double[][] c = new double[n][m + 1];
        double c1 = 1.0;
        double c4 = x[0];
        c[0][0] = 1.0;
        for (int i = 1; i < n; i++) {
            int mn = Math.min(i, m);
            double c2 = 1.0;
            double c5 = c4;
            c4 = x[i];
            for (int j = 0; j < i; j++) {
                double c3 = x[i] - x[j];
                c2 *= c3;
                if (j == i - 1) {
                    for (int k = mn; k >= 1; k--) {
                        c[i][k] = c1 * (k * c[i - 1][k - 1] - c5 * c[i - 1][k]) / c2;
                    }
                    c[i][0] = -c1 * c5 * c[i - 1][0] / c2;
                }
                for (int k = mn; k >= 1; k--) {
                    c[j][k] = (c4 * c[j][k] - k * c[j][k - 1]) / c3;
                }
                c[j][0] = c4 * c[j][0] / c3;
            }
            c1 = c2;
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = c[i][m];
        }
        return out;
    }

    /**
     * Builds the interior stencil of the {@code order}-th derivative.
     *
     * @param order derivative order (&gt;= 1)
     * @param accuracy formal order of accuracy
     * @param scheme scheme for FIRST derivatives; higher derivatives are always central
     */
    static Stencil1D interior(int order, int accuracy, DerivativeScheme scheme) {
        int[] off;
        if (order == 1 && scheme == DerivativeScheme.BACKWARD) {
            off = new int[accuracy + 1];
            for (int q = 0; q <= accuracy; q++) {
                off[q] = -accuracy + q;
            }
        } else if (order == 1 && scheme == DerivativeScheme.FORWARD) {
            off = new int[accuracy + 1];
            for (int q = 0; q <= accuracy; q++) {
                off[q] = q;
            }
        } else {
            if (accuracy % 2 != 0) {
                throw new PDEConfigurationException("Central stencils need an even spatialOrder (2, 4 or 6), got " + accuracy
                        + ". Odd orders are only available through the BACKWARD/FORWARD first-derivative schemes");
            }
            int r = (order + accuracy - 2 + (order % 2)) / 2;
            off = new int[2 * r + 1];
            for (int q = -r; q <= r; q++) {
                off[q + r] = q;
            }
        }
        double[] xs = new double[off.length];
        for (int q = 0; q < off.length; q++) {
            xs[q] = off[q];
        }
        double[] w = weights(order, xs);
        int cnt = 0;
        for (double v : w) {
            if (Math.abs(v) > 1e-14) {
                cnt++;
            }
        }
        int[] o2 = new int[cnt];
        double[] w2 = new double[cnt];
        int p = 0;
        for (int q = 0; q < off.length; q++) {
            if (Math.abs(w[q]) > 1e-14) {
                o2[p] = off[q];
                w2[p] = w[q];
                p++;
            }
        }
        return new Stencil1D(o2, w2);
    }

    /** One-sided first-derivative weights at node 0 using offsets {@code 0..p} (inward direction). */
    static double[] boundaryFirstDerivative(int p) {
        double[] xs = new double[p + 1];
        for (int q = 0; q <= p; q++) {
            xs[q] = q;
        }
        return weights(1, xs);
    }

    /**
     * Lagrange weights that extrapolate a polynomial through the nodes
     * {@code 0..m} to the position {@code -j} (j = 1, 2, ...).
     */
    static double[] extrapolation(int j, int m) {
        double[] w = new double[m + 1];
        double xs = -j;
        for (int k = 0; k <= m; k++) {
            double v = 1.0;
            for (int l = 0; l <= m; l++) {
                if (l != k) {
                    v *= (xs - l) / (double) (k - l);
                }
            }
            w[k] = v;
        }
        return w;
    }
}
