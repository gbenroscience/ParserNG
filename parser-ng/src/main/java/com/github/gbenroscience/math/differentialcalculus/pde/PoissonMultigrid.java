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

import java.util.Arrays;

/**
 * Matrix-free geometric multigrid for the cell-centred Poisson equation
 * {@code laplace(phi) = rhs} on a uniform 2-D or 3-D grid, with a
 * Neumann, Dirichlet ({@code phi = 0} on the face) or periodic condition on
 * each face. This is the pressure solver of the incompressible flow code, but
 * it is a general tool: it is homogeneous in the boundary data, which for a
 * non-zero Dirichlet/Neumann value is moved into the right-hand side by the
 * caller.
 * <p>
 * Components: 5/7-point operator rediscretised on every level, cell-average
 * restriction, (bi/tri)linear prolongation, V or W cycles, smoothers weighted
 * Jacobi / Gauss-Seidel / red-black Gauss-Seidel / SOR, and an iterated
 * smoother as the coarsest-level solve. A problem with no Dirichlet face is
 * singular; its right-hand side is made compatible (zero mean) and the
 * solution is normalised to zero mean.
 * <p>
 * Memory: about {@code 4 * 8 * N * (1 + 1/2^d + ...)} bytes (phi, rhs,
 * residual, scratch on every level), allocated once. Work per cycle:
 * {@code O(N)}. {@link #solve()} allocates nothing.
 */
public final class PoissonMultigrid {

    /** Condition on one face of the (homogeneous) problem. */
    public enum BC {
        /** d(phi)/dn = 0. */
        NEUMANN,
        /** phi = 0 on the face (the ghost value is minus the interior value). */
        DIRICHLET,
        /** phi wraps to the opposite face; must be set on both faces of the axis. */
        PERIODIC
    }

    /** Relaxation scheme. */
    public enum Smoother {
        WEIGHTED_JACOBI, GAUSS_SEIDEL, RED_BLACK_GS, SOR
    }

    /** Cycle shape. */
    public enum Cycle {
        V(1), W(2);

        final int gamma;

        Cycle(int g) {
            this.gamma = g;
        }
    }

    /** Statistics of the last {@link #solve()}. */
    public static final class Stats {

        /** Cycles performed. */
        public int cycles;
        /** Max-norm of the residual before the first cycle. */
        public double initialResidual;
        /** Max-norm of the residual after the last cycle. */
        public double residual;
        /** Geometric-mean residual reduction per cycle (asymptotic convergence factor estimate). */
        public double convergenceFactor;
        /** True if the tolerance was met. */
        public boolean converged;
    }

    // ---- configuration ----
    private Smoother smoother = Smoother.RED_BLACK_GS;
    private Cycle cycle = Cycle.V;
    private int preSweeps = 2;
    private int postSweeps = 2;
    private int coarseSweeps = 60;
    private double omega = 1.15;
    private double jacobiWeight = 2.0 / 3.0;
    private int maxCycles = 50;
    private double rtol = 1e-10;
    private double atol = 1e-14;

    private final int dim;
    private final int levels;
    private final Box[] box;
    private final double[][] h;          // [level][axis]
    private final double[][] phi;
    private final double[][] rhs;
    private final double[][] res;
    private final double[][] tmp;
    private final BC[][] bc;             // [axis][0=min,1=max]
    private final boolean singular;
    private final Stats stats = new Stats();
    // conjugate-gradient work vectors of the coarsest level (allocated once; solve() stays allocation free)
    private final double[] cgP;
    private final double[] cgAp;
    private final double[] cgQ;

    /**
     * @param dim 2 or 3
     * @param n cells per axis (finest level)
     * @param h cell size per axis (finest level)
     * @param bc condition per axis and side ([axis][0 = min face, 1 = max face])
     */
    public PoissonMultigrid(int dim, int[] n, double[] h, BC[][] bc) {
        if (dim != 2 && dim != 3) {
            throw new PDEConfigurationException("PoissonMultigrid supports 2 or 3 dimensions, got " + dim);
        }
        this.dim = dim;
        this.bc = new BC[3][2];
        boolean dirichlet = false;
        for (int a = 0; a < dim; a++) {
            this.bc[a][0] = bc[a][0];
            this.bc[a][1] = bc[a][1];
            if ((bc[a][0] == BC.PERIODIC) != (bc[a][1] == BC.PERIODIC)) {
                throw new PDEConfigurationException("Periodic conditions must be given on both faces of axis " + a);
            }
            dirichlet |= bc[a][0] == BC.DIRICHLET || bc[a][1] == BC.DIRICHLET;
        }
        this.singular = !dirichlet;
        int lv = 1;
        int[] m = {n[0], n[1], dim == 3 ? n[2] : 1};
        while (true) {
            boolean ok = true;
            for (int a = 0; a < dim; a++) {
                ok &= m[a] % 2 == 0 && m[a] / 2 >= 2;
            }
            if (!ok) {
                break;
            }
            for (int a = 0; a < dim; a++) {
                m[a] /= 2;
            }
            lv++;
        }
        this.levels = lv;
        this.box = new Box[lv];
        this.h = new double[lv][3];
        this.phi = new double[lv][];
        this.rhs = new double[lv][];
        this.res = new double[lv][];
        this.tmp = new double[lv][];
        int[] c = {n[0], n[1], dim == 3 ? n[2] : 1};
        double[] hh = {h[0], h[1], dim == 3 ? h[2] : 1.0};
        for (int l = 0; l < lv; l++) {
            box[l] = new Box(dim, c[0], c[1], c[2]);
            this.h[l] = hh.clone();
            phi[l] = new double[box[l].size];
            rhs[l] = new double[box[l].size];
            res[l] = new double[box[l].size];
            tmp[l] = new double[box[l].size];
            for (int a = 0; a < dim; a++) {
                c[a] /= 2;
                hh[a] *= 2;
            }
        }
        int cs = box[lv - 1].size;
        this.cgP = new double[cs];
        this.cgAp = new double[cs];
        this.cgQ = new double[cs];
    }

    // ------------------------------------------------------------------ configuration
    public PoissonMultigrid smoother(Smoother s) {
        this.smoother = s;
        return this;
    }

    public PoissonMultigrid cycle(Cycle c) {
        this.cycle = c;
        return this;
    }

    /** Pre- and post-smoothing sweeps (default 2 and 2). */
    public PoissonMultigrid sweeps(int pre, int post) {
        this.preSweeps = pre;
        this.postSweeps = post;
        return this;
    }

    /** Smoothing sweeps used as the coarsest-level solve (default 60). */
    public PoissonMultigrid coarseSweeps(int s) {
        this.coarseSweeps = s;
        return this;
    }

    /** SOR relaxation factor (default 1.15 as a smoother; use 1.0 for Gauss-Seidel). */
    public PoissonMultigrid omega(double w) {
        this.omega = w;
        return this;
    }

    /** Stopping criteria: relative residual reduction, absolute residual, and cycle cap. */
    public PoissonMultigrid tolerance(double rel, double abs, int maxCycles) {
        this.rtol = rel;
        this.atol = abs;
        this.maxCycles = maxCycles;
        return this;
    }

    // ------------------------------------------------------------------ access
    /** @return number of grid levels (1 = no coarsening possible). */
    public int levels() {
        return levels;
    }

    /** @return the finest-level solution / initial guess array (ghost-padded, see {@link #index}). */
    public double[] phi() {
        return phi[0];
    }

    /** @return the finest-level right-hand side array. */
    public double[] rhs() {
        return rhs[0];
    }

    /** @return flat index of finest-level cell (i,j,k); ghost cells use -1 and n. */
    public int index(int i, int j, int k) {
        return box[0].idx(i, j, k);
    }

    /** @return true if the problem has no Dirichlet face (the solution is defined up to a constant). */
    public boolean isSingular() {
        return singular;
    }

    public Stats stats() {
        return stats;
    }

    /** Fills the finest-level ghost layer of an array with this problem's face conditions. */
    void fillFineGhosts(double[] u) {
        fillGhosts(0, u);
    }

    Box fineBox() {
        return box[0];
    }

    // ------------------------------------------------------------------ boundary ghosts
    private void fillGhosts(int l, double[] u) {
        Box b = box[l];
        int n0 = b.n[0], n1 = b.n[1], n2 = b.n[2];
        final int s0 = 1, s1 = b.st[1], s2 = b.st[2];
        // axis 0
        for (int k = 0; k < n2; k++) {
            for (int j = 0; j < n1; j++) {
                int lo = b.idx(0, j, k);
                int hi = b.idx(n0 - 1, j, k);
                u[lo - s0] = ghost(bc[0][0], u[lo], u[hi]);
                u[hi + s0] = ghost(bc[0][1], u[hi], u[lo]);
            }
        }
        for (int k = 0; k < n2; k++) {
            for (int i = 0; i < n0; i++) {
                int lo = b.idx(i, 0, k);
                int hi = b.idx(i, n1 - 1, k);
                u[lo - s1] = ghost(bc[1][0], u[lo], u[hi]);
                u[hi + s1] = ghost(bc[1][1], u[hi], u[lo]);
            }
        }
        if (dim == 3) {
            for (int j = 0; j < n1; j++) {
                for (int i = 0; i < n0; i++) {
                    int lo = b.idx(i, j, 0);
                    int hi = b.idx(i, j, n2 - 1);
                    u[lo - s2] = ghost(bc[2][0], u[lo], u[hi]);
                    u[hi + s2] = ghost(bc[2][1], u[hi], u[lo]);
                }
            }
        }
    }

    private static double ghost(BC type, double inner, double opposite) {
        switch (type) {
            case NEUMANN:
                return inner;
            case DIRICHLET:
                return -inner;
            default:
                return opposite;
        }
    }

    // ------------------------------------------------------------------ operators
    private double residual(int l) {
        Box b = box[l];
        double[] u = phi[l];
        double[] f = rhs[l];
        double[] r = res[l];
        fillGhosts(l, u);
        final double c0 = 1.0 / (h[l][0] * h[l][0]);
        final double c1 = 1.0 / (h[l][1] * h[l][1]);
        final double c2 = dim == 3 ? 1.0 / (h[l][2] * h[l][2]) : 0.0;
        final int s1 = b.st[1];
        final int s2 = dim == 3 ? b.st[2] : 0;
        double max = 0;
        for (int k = 0; k < b.n[2]; k++) {
            for (int j = 0; j < b.n[1]; j++) {
                int base = b.idx(0, j, k);
                for (int i = 0; i < b.n[0]; i++) {
                    int q = base + i;
                    double lap = c0 * (u[q + 1] + u[q - 1] - 2 * u[q]) + c1 * (u[q + s1] + u[q - s1] - 2 * u[q])
                            + c2 * (u[q + s2] + u[q - s2] - 2 * u[q]);
                    double v = f[q] - lap;
                    r[q] = v;
                    double a = Math.abs(v);
                    if (a > max) {
                        max = a;
                    }
                }
            }
        }
        return max;
    }

    private void smooth(int l, int sweeps) {
        Box b = box[l];
        double[] u = phi[l];
        double[] f = rhs[l];
        final double c0 = 1.0 / (h[l][0] * h[l][0]);
        final double c1 = 1.0 / (h[l][1] * h[l][1]);
        final double c2 = dim == 3 ? 1.0 / (h[l][2] * h[l][2]) : 0.0;
        final double diag = 2 * (c0 + c1 + c2);
        final int s1 = b.st[1];
        final int s2 = dim == 3 ? b.st[2] : 0;
        for (int sw = 0; sw < sweeps; sw++) {
            switch (smoother) {
                case WEIGHTED_JACOBI: {
                    fillGhosts(l, u);
                    double[] t = tmp[l];
                    final double w = jacobiWeight;
                    for (int k = 0; k < b.n[2]; k++) {
                        for (int j = 0; j < b.n[1]; j++) {
                            int base = b.idx(0, j, k);
                            for (int i = 0; i < b.n[0]; i++) {
                                int q = base + i;
                                double gs = (c0 * (u[q + 1] + u[q - 1]) + c1 * (u[q + s1] + u[q - s1])
                                        + c2 * (u[q + s2] + u[q - s2]) - f[q]) / diag;
                                t[q] = (1 - w) * u[q] + w * gs;
                            }
                        }
                    }
                    for (int k = 0; k < b.n[2]; k++) {
                        for (int j = 0; j < b.n[1]; j++) {
                            int base = b.idx(0, j, k);
                            System.arraycopy(t, base, u, base, b.n[0]);
                        }
                    }
                    break;
                }
                case RED_BLACK_GS:
                case SOR: {
                    final double w = smoother == Smoother.SOR ? omega : 1.0;
                    for (int color = 0; color < 2; color++) {
                        fillGhosts(l, u);
                        for (int k = 0; k < b.n[2]; k++) {
                            for (int j = 0; j < b.n[1]; j++) {
                                int base = b.idx(0, j, k);
                                int i0 = ((j + k) & 1) == color ? 0 : 1;
                                for (int i = i0; i < b.n[0]; i += 2) {
                                    int q = base + i;
                                    double gs = (c0 * (u[q + 1] + u[q - 1]) + c1 * (u[q + s1] + u[q - s1])
                                            + c2 * (u[q + s2] + u[q - s2]) - f[q]) / diag;
                                    u[q] += w * (gs - u[q]);
                                }
                            }
                        }
                    }
                    break;
                }
                default: { // lexicographic Gauss-Seidel
                    fillGhosts(l, u);
                    for (int k = 0; k < b.n[2]; k++) {
                        for (int j = 0; j < b.n[1]; j++) {
                            int base = b.idx(0, j, k);
                            for (int i = 0; i < b.n[0]; i++) {
                                int q = base + i;
                                // ghosts are frozen during a sweep: refresh the face neighbours that change
                                double gs = (c0 * (u[q + 1] + u[q - 1]) + c1 * (u[q + s1] + u[q - s1])
                                        + c2 * (u[q + s2] + u[q - s2]) - f[q]) / diag;
                                u[q] = gs;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    /** Restricts the residual of level {@code l} onto the right-hand side of level {@code l+1} (cell averages). */
    private void restrict(int l) {
        Box fb = box[l];
        Box cb = box[l + 1];
        double[] r = res[l];
        double[] out = rhs[l + 1];
        final double w = dim == 3 ? 0.125 : 0.25;
        final int s1 = fb.st[1];
        final int s2 = dim == 3 ? fb.st[2] : 0;
        for (int K = 0; K < cb.n[2]; K++) {
            for (int J = 0; J < cb.n[1]; J++) {
                int cbase = cb.idx(0, J, K);
                for (int I = 0; I < cb.n[0]; I++) {
                    int q = fb.idx(2 * I, 2 * J, dim == 3 ? 2 * K : 0);
                    double s = r[q] + r[q + 1] + r[q + s1] + r[q + 1 + s1];
                    if (dim == 3) {
                        s += r[q + s2] + r[q + 1 + s2] + r[q + s1 + s2] + r[q + 1 + s1 + s2];
                    }
                    out[cbase + I] = w * s;
                }
            }
        }
    }

    /** Adds the interpolated coarse correction of level {@code l+1} to level {@code l}. */
    private void prolongAdd(int l) {
        Box fb = box[l];
        Box cb = box[l + 1];
        double[] e = phi[l + 1];
        double[] u = phi[l];
        fillGhosts(l + 1, e);
        final int cs1 = cb.st[1];
        final int cs2 = dim == 3 ? cb.st[2] : 0;
        for (int k = 0; k < fb.n[2]; k++) {
            int K = k >> 1;
            int dk = (k & 1) == 0 ? -1 : 1;
            for (int j = 0; j < fb.n[1]; j++) {
                int J = j >> 1;
                int dj = (j & 1) == 0 ? -1 : 1;
                int fbase = fb.idx(0, j, k);
                for (int i = 0; i < fb.n[0]; i++) {
                    int I = i >> 1;
                    int di = (i & 1) == 0 ? -1 : 1;
                    int p = cb.idx(I, J, dim == 3 ? K : 0);
                    double v;
                    if (dim == 2) {
                        v = 0.5625 * e[p] + 0.1875 * (e[p + di] + e[p + dj * cs1]) + 0.0625 * e[p + di + dj * cs1];
                    } else {
                        int a = di;
                        int bb = dj * cs1;
                        int c = dk * cs2;
                        v = 0.421875 * e[p]
                                + 0.140625 * (e[p + a] + e[p + bb] + e[p + c])
                                + 0.046875 * (e[p + a + bb] + e[p + a + c] + e[p + bb + c])
                                + 0.015625 * e[p + a + bb + c];
                    }
                    u[fbase + i] += v;
                }
            }
        }
    }

    /**
     * Coarsest-level solve by (matrix-free) conjugate gradients on -laplace. The coarsening stops as soon as an axis has an
     * odd cell count (e.g. 14 x 7, 18 x 9, 30 x 15), so the "coarsest" level can be the whole fine grid; a fixed number of
     * smoothing sweeps then cannot converge and the pressure solve silently returned an unconverged field. The operator is
     * symmetric (homogeneous ghost rules) and, for an all-Neumann/periodic problem, singular but consistent: CG handles both.
     */
    private void coarseSolve(int l) {
        Box b = box[l];
        double[] x = phi[l];
        double[] f = rhs[l];
        if (singular) {
            removeMean(f, b);
        }
        double[] q = cgQ;
        double[] p = cgP;
        double[] ap = cgAp;
        double[] r = res[l];
        residual(l);                       // r = f - laplace(x)
        double qq = 0;
        for (int k = 0; k < b.n[2]; k++) {
            for (int j = 0; j < b.n[1]; j++) {
                int base = b.idx(0, j, k);
                for (int i = 0; i < b.n[0]; i++) {
                    int c = base + i;
                    q[c] = -r[c];          // residual of the SPD system (-laplace) x = -f
                    p[c] = q[c];
                    qq += q[c] * q[c];
                }
            }
        }
        if (!(qq > 0)) {
            return;
        }
        final double stop = (levels > 1 ? 1e-4 : 1e-12) * Math.sqrt(qq);
        final int maxIt = Math.max(coarseSweeps, 10 * (b.n[0] + b.n[1] + b.n[2]));
        final double c0 = 1.0 / (h[l][0] * h[l][0]);
        final double c1 = 1.0 / (h[l][1] * h[l][1]);
        final double c2 = dim == 3 ? 1.0 / (h[l][2] * h[l][2]) : 0.0;
        final int s1 = b.st[1];
        final int s2 = dim == 3 ? b.st[2] : 0;
        for (int it = 0; it < maxIt && Math.sqrt(qq) > stop; it++) {
            fillGhosts(l, p);
            double pap = 0;
            for (int k = 0; k < b.n[2]; k++) {
                for (int j = 0; j < b.n[1]; j++) {
                    int base = b.idx(0, j, k);
                    for (int i = 0; i < b.n[0]; i++) {
                        int c = base + i;
                        double v = -(c0 * (p[c + 1] + p[c - 1] - 2 * p[c]) + c1 * (p[c + s1] + p[c - s1] - 2 * p[c])
                                + c2 * (p[c + s2] + p[c - s2] - 2 * p[c]));
                        ap[c] = v;
                        pap += p[c] * v;
                    }
                }
            }
            if (!(pap > 0)) {
                break;
            }
            double alpha = qq / pap;
            double qqNew = 0;
            for (int k = 0; k < b.n[2]; k++) {
                for (int j = 0; j < b.n[1]; j++) {
                    int base = b.idx(0, j, k);
                    for (int i = 0; i < b.n[0]; i++) {
                        int c = base + i;
                        x[c] += alpha * p[c];
                        q[c] -= alpha * ap[c];
                        qqNew += q[c] * q[c];
                    }
                }
            }
            double beta = qqNew / qq;
            qq = qqNew;
            for (int k = 0; k < b.n[2]; k++) {
                for (int j = 0; j < b.n[1]; j++) {
                    int base = b.idx(0, j, k);
                    for (int i = 0; i < b.n[0]; i++) {
                        int c = base + i;
                        p[c] = q[c] + beta * p[c];
                    }
                }
            }
        }
    }

    private void cycleAt(int l) {
        if (l == levels - 1) {
            coarseSolve(l);
            return;
        }
        smooth(l, preSweeps);
        residual(l);
        restrict(l);
        Arrays.fill(phi[l + 1], 0.0);
        for (int g = 0; g < cycle.gamma; g++) {
            cycleAt(l + 1);
        }
        prolongAdd(l);
        smooth(l, postSweeps);
    }

    private void removeMean(double[] a, Box b) {
        double s = 0;
        for (int k = 0; k < b.n[2]; k++) {
            for (int j = 0; j < b.n[1]; j++) {
                int base = b.idx(0, j, k);
                for (int i = 0; i < b.n[0]; i++) {
                    s += a[base + i];
                }
            }
        }
        s /= (double) b.n[0] * b.n[1] * b.n[2];
        for (int k = 0; k < b.n[2]; k++) {
            for (int j = 0; j < b.n[1]; j++) {
                int base = b.idx(0, j, k);
                for (int i = 0; i < b.n[0]; i++) {
                    a[base + i] -= s;
                }
            }
        }
    }

    /**
     * Runs cycles on the current {@link #phi()} (initial guess) and
     * {@link #rhs()} until the tolerance is met.
     *
     * @return the statistics of this solve (a reused object)
     */
    public Stats solve() {
        if (singular) {
            removeMean(rhs[0], box[0]);
        }
        double r0 = residual(0);
        stats.initialResidual = r0;
        stats.cycles = 0;
        double r = r0;
        double target = Math.max(atol, rtol * r0);
        while (r > target && stats.cycles < maxCycles) {
            cycleAt(0);
            r = residual(0);
            stats.cycles++;
        }
        if (singular) {
            removeMean(phi[0], box[0]);
        }
        fillGhosts(0, phi[0]);   // the mean removal above shifted phi: keep the ghost layer consistent
        stats.residual = r;
        stats.converged = r <= target;
        stats.convergenceFactor = stats.cycles > 0 && r0 > 0 ? Math.pow(Math.max(r, 1e-300) / r0, 1.0 / stats.cycles) : 0.0;
        return stats;
    }

    /** @return the finest-level residual max-norm of the current phi / rhs. */
    public double residualNorm() {
        return residual(0);
    }
}
