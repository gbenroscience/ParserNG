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
 * Steady-state (elliptic) solvers.
 * <ul>
 * <li>Point iterations (Jacobi, Gauss-Seidel, SOR, red-black SOR) for a single
 * unknown. Each node is relaxed with a point-Newton update
 * {@code u -= omega * R / (dR/du_i)}. The diagonal {@code dR/du_i} comes from
 * ParserNG's forward-mode automatic differentiation (exact, computed once for
 * linear PDEs and refreshed periodically for nonlinear ones), or from a
 * finite-difference probe if AD cannot differentiate the equation.</li>
 * <li>Newton-Krylov (matrix-free BiCGStab with the exact AD Jacobian) for
 * systems and strongly nonlinear problems.</li>
 * </ul>
 */
final class EllipticEngine {

    private final Model m;
    private final Boundaries bc;
    private final PDEOptions o;
    private final SolverDiagnostics diag;
    private final Grid grid;
    private final double[][] y;
    private final int size;
    private EllipticMethod method;
    private double omega;
    private double[] diagonal;
    private boolean exactDiagonal;
    private double[] ynBuf;
    private final double[] resPart;
    private final double[] updPart;
    private final SweepTask sweepTask = new SweepTask();
    private double[][] res;
    private double[][] sBuf, wBuf, vBuf, jvBuf;
    private NewtonKrylov nk;
    private double[] zVec;
    private boolean[] pdeFlag;

    EllipticEngine(Model m, Boundaries bc, double[][] y, SolverDiagnostics diag) {
        this.m = m;
        this.bc = bc;
        this.o = m.opt;
        this.diag = diag;
        this.grid = m.grid;
        this.y = y;
        this.size = grid.storageSize();
        this.resPart = new double[m.pool.size()];
        this.updPart = new double[m.pool.size()];
        resolveMethod();
        diag.type = PDEType.ELLIPTIC;
        diag.spatialOrder = o.spatialOrder;
        diag.threads = m.threads;
        diag.omega = omega;
    }

    private void resolveMethod() {
        EllipticMethod r = o.ellipticMethod;
        if (r == EllipticMethod.AUTO) {
            if (m.nc > 1) {
                r = EllipticMethod.NEWTON_KRYLOV;
            } else {
                r = m.redBlackOk ? EllipticMethod.RED_BLACK_SOR : EllipticMethod.SOR;
            }
        }
        if (r != EllipticMethod.NEWTON_KRYLOV && m.nc > 1) {
            throw new PDEConfigurationException("Point iterations (" + r + ") solve a single unknown; this system has "
                    + m.nc + ". Use ellipticMethod(NEWTON_KRYLOV)");
        }
        if (r == EllipticMethod.RED_BLACK_SOR && !m.redBlackOk) {
            throw new PDEConfigurationException("RED_BLACK_SOR needs a nearest-neighbour stencil with no mixed derivatives;"
                    + " use SOR or GAUSS_SEIDEL for this discretization");
        }
        method = r;
        double w = o.omega;
        if (Double.isNaN(w)) {
            if (r == EllipticMethod.SOR || r == EllipticMethod.RED_BLACK_SOR) {
                w = estimateOmega();
            } else {
                w = 1.0;
            }
        }
        if (!(w > 0 && w < 2.0) && r != EllipticMethod.JACOBI) {
            throw new PDEConfigurationException("Relaxation factor omega must lie in (0,2) for SOR/Gauss-Seidel, got " + w);
        }
        omega = w;
        diag.method = r + " / spatial order " + o.spatialOrder + (r == EllipticMethod.NEWTON_KRYLOV ? "" : ", omega=" + String.format("%.4f", w));
    }

    /** Optimal SOR factor for a Laplace-type operator on this grid (heuristic for other operators). */
    private double estimateOmega() {
        double num = 0;
        double den = 0;
        for (int a = 0; a < grid.dim(); a++) {
            double ih2 = 1.0 / (grid.h(a) * grid.h(a));
            num += Math.cos(Math.PI / (grid.n(a) - 1)) * ih2;
            den += ih2;
        }
        double rho = num / den;
        double w = 2.0 / (1.0 + Math.sqrt(Math.max(0.0, 1.0 - rho * rho)));
        return Math.min(1.99, Math.max(1.0, w));
    }

    /** Runs the solver on the current contents of {@code y} as the initial guess. */
    boolean solve() {
        long t0 = System.nanoTime();
        bc.apply(y, 0.0);
        boolean ok;
        if (method == EllipticMethod.NEWTON_KRYLOV) {
            ok = newtonKrylov();
        } else {
            ok = pointIteration();
        }
        diag.converged = ok;
        diag.wallNanos += System.nanoTime() - t0;
        return ok;
    }

    // ------------------------------------------------------------ point iterations
    private void computeDiagonal() {
        if (!m.ensureJacobian()) {
            exactDiagonal = false;
            return;
        }
        m.linearize(0.0, y);
        if (diagonal == null) {
            diagonal = new double[size];
        }
        int[] js = m.jSym[0];
        double[] cw = new double[js.length];
        for (int j = 0; j < js.length; j++) {
            if (js[j] < m.symTerm) {
                cw[j] = js[j] == Model.SYM_STATE + m.valVar[0] ? 1.0 : 0.0;
            } else {
                cw[j] = m.terms[js[j] - m.symTerm].centerWeight;
            }
        }
        for (int k = m.pdeLo[2]; k <= m.pdeHi[2]; k++) {
            for (int j = m.pdeLo[1]; j <= m.pdeHi[1]; j++) {
                int base = grid.index(0, j, k);
                for (int i = m.pdeLo[0]; i <= m.pdeHi[0]; i++) {
                    int idx = base + i;
                    double d = 0;
                    for (int q = 0; q < js.length; q++) {
                        d += m.jac[0][q][idx] * cw[q];
                    }
                    diagonal[idx] = d;
                }
            }
        }
        exactDiagonal = true;
    }

    private final class SweepTask implements WorkerPool.Task {

        int color;          // -1 = all nodes, 0/1 = red/black
        boolean jacobi;
        double[] src;
        double[] dst;

        @Override
        public void run(int worker, int workers) {
            sweep(worker, workers, color, jacobi, src, dst);
        }
    }

    private void sweep(int worker, int workers, int color, boolean jacobi, double[] src, double[] dst) {
        Model.Ctx c = m.ctx[worker];
        double[] sym = c.sym;
        sym[Model.SYM_T] = 0.0;
        int iLo = m.pdeLo[0], iHi = m.pdeHi[0], jLo = m.pdeLo[1], jHi = m.pdeHi[1], kLo = m.pdeLo[2], kHi = m.pdeHi[2];
        if (m.outerAxis == 2) {
            kLo = m.chunkLo(worker, workers);
            kHi = m.chunkHi(worker, workers);
        } else if (m.outerAxis == 1) {
            jLo = m.chunkLo(worker, workers);
            jHi = m.chunkHi(worker, workers);
        } else {
            iLo = m.chunkLo(worker, workers);
            iHi = m.chunkHi(worker, workers);
        }
        final double[] xc = m.coords[0];
        final double[] yc = m.coords[1];
        final double[] zc = m.coords[2];
        final int sv = Model.SYM_STATE + m.valVar[0];
        double maxRes = 0;
        double maxUpd = 0;
        final double[][] yy = y;
        final boolean ex = exactDiagonal;
        final double[] dg = diagonal;
        final double w = omega;
        for (int k = kLo; k <= kHi; k++) {
            sym[2] = zc[k];
            for (int j = jLo; j <= jHi; j++) {
                sym[1] = yc[j];
                int base = grid.index(0, j, k);
                int i0 = iLo;
                int step = 1;
                if (color >= 0) {
                    step = 2;
                    if (((iLo + j + k) & 1) != color) {
                        i0 = iLo + 1;
                    }
                }
                for (int i = i0; i <= iHi; i += step) {
                    int idx = base + i;
                    sym[0] = xc[i];
                    m.loadNode(sym, yy, idx);
                    double r = m.evalEq(c, 0);
                    double d;
                    if (ex) {
                        d = dg[idx];
                    } else {
                        double u = sym[sv];
                        double delta = 1e-6 * (1.0 + Math.abs(u));
                        sym[sv] = u + delta;
                        for (int q = 0; q < m.terms.length; q++) {
                            sym[m.symTerm + q] += delta * m.terms[q].centerWeight;
                        }
                        double r1 = m.evalEq(c, 0);
                        d = (r1 - r) / delta;
                    }
                    if (d == 0.0 || Double.isNaN(d) || Double.isInfinite(d)) {
                        throw new PDEException("Singular or non-finite diagonal in the elliptic iteration at node ("
                                + i + "," + j + "," + k + "): the discretized equation does not depend on the unknown there");
                    }
                    double upd = -w * r / d;
                    dst[idx] = src[idx] + upd;
                    double ar = Math.abs(r);
                    if (ar > maxRes || Double.isNaN(ar)) {
                        maxRes = ar;
                    }
                    double au = Math.abs(upd);
                    if (au > maxUpd || Double.isNaN(au)) {
                        maxUpd = au;
                    }
                }
            }
        }
        resPart[worker] = maxRes;
        updPart[worker] = maxUpd;
    }

    private double reduce(double[] a) {
        double mx = 0;
        for (int i = 0; i < m.pool.size(); i++) {
            if (a[i] > mx || Double.isNaN(a[i])) {
                mx = a[i];
            }
        }
        return mx;
    }

    private boolean pointIteration() {
        computeDiagonal();
        final boolean jacobi = method == EllipticMethod.JACOBI;
        final boolean rb = method == EllipticMethod.RED_BLACK_SOR;
        if (jacobi && ynBuf == null) {
            ynBuf = new double[size];
        }
        double res0 = Double.NaN;
        double resIt = Double.NaN;
        double upd = Double.NaN;
        boolean ok = false;
        int it = 0;
        long linearityRefresh = m.constantPartials ? Long.MAX_VALUE : 10;
        for (it = 1; it <= o.maxIterations; it++) {
            double resMax;
            double updMax;
            if (jacobi) {
                System.arraycopy(y[0], 0, ynBuf, 0, size);
                sweepTask.color = -1;
                sweepTask.jacobi = true;
                sweepTask.src = y[0];
                sweepTask.dst = ynBuf;
                m.pool.run(sweepTask);
                resMax = reduce(resPart);
                updMax = reduce(updPart);
                double[] swap = y[0];
                y[0] = ynBuf;
                ynBuf = swap;
                bc.apply(y, 0.0);
            } else if (rb) {
                sweepTask.jacobi = false;
                sweepTask.src = y[0];
                sweepTask.dst = y[0];
                sweepTask.color = 0;
                m.pool.run(sweepTask);
                double r0c = reduce(resPart);
                double u0c = reduce(updPart);
                bc.apply(y, 0.0);
                sweepTask.color = 1;
                m.pool.run(sweepTask);
                resMax = Math.max(r0c, reduce(resPart));
                updMax = Math.max(u0c, reduce(updPart));
                bc.apply(y, 0.0);
            } else {
                // lexicographic Gauss-Seidel / SOR: strictly sequential, in place
                sweep(0, 1, -1, false, y[0], y[0]);
                resMax = resPart[0];
                updMax = updPart[0];
                bc.apply(y, 0.0);
            }
            if (Double.isNaN(resMax) || Double.isInfinite(resMax)) {
                diag.warnings.add("Elliptic iteration diverged (non-finite residual) after " + it + " iterations");
                break;
            }
            if (it == 1) {
                res0 = resMax;
                diag.initialResidualNorm = res0;
            }
            resIt = resMax;
            upd = updMax;
            if (resIt <= o.absoluteTolerance + o.tolerance * res0) {
                ok = true;
                break;
            }
            if (!m.constantPartials && it % linearityRefresh == 0) {
                computeDiagonal();
            }
        }
        diag.iterations = Math.min(it, o.maxIterations);
        diag.updateNorm = upd;
        // exact residual of the final iterate
        bc.apply(y, 0.0);
        if (res == null) {
            res = new double[m.nc][size];
        }
        m.residual(y, res);
        double exact = 0;
        for (int k = m.pdeLo[2]; k <= m.pdeHi[2]; k++) {
            for (int j = m.pdeLo[1]; j <= m.pdeHi[1]; j++) {
                int base = grid.index(0, j, k);
                for (int i = m.pdeLo[0]; i <= m.pdeHi[0]; i++) {
                    double a = Math.abs(res[0][base + i]);
                    if (a > exact || Double.isNaN(a)) {
                        exact = a;
                    }
                }
            }
        }
        diag.residualNorm = exact;
        if (!Double.isNaN(res0) && res0 > 0 && resIt > 0) {
            diag.contractionFactor = Math.pow(resIt / res0, 1.0 / Math.max(1, diag.iterations - 1 + (diag.iterations == 1 ? 1 : 0)));
        }
        double rho = diag.contractionFactor;
        if (!Double.isNaN(rho) && rho < 1) {
            diag.errorEstimate = rho / (1.0 - rho) * upd;
        }
        double thr = o.absoluteTolerance + o.tolerance * (Double.isNaN(res0) ? 0 : res0);
        if (ok && exact > 10 * thr + 1e-300) {
            diag.warnings.add("Sweep residual met the tolerance but the exact final residual is " + exact + " (> 10x target)");
            ok = false;
        }
        if (!ok) {
            diag.warnings.add("Elliptic iteration did not converge: residual " + exact + " after " + diag.iterations
                    + " iterations (target " + thr + ")");
        }
        return ok;
    }

    // ------------------------------------------------------------ Newton-Krylov
    private void unpack(double[] z, double[][] a) {
        int n = grid.nodeCount();
        int[] ni = m.nodeIdx;
        for (int s = 0; s < m.nState; s++) {
            double[] as = a[s];
            int off = s * n;
            for (int c = 0; c < n; c++) {
                as[ni[c]] = z[off + c];
            }
        }
    }

    private boolean newtonKrylov() {
        int ns = m.nState;
        int n = grid.nodeCount();
        if (nk == null) {
            sBuf = new double[ns][size];
            wBuf = new double[ns][size];
            vBuf = new double[ns][size];
            jvBuf = new double[ns][size];
            res = new double[ns][size];
            zVec = new double[ns * n];
            pdeFlag = new boolean[n];
            int p = 0;
            for (int k = 0; k < grid.n(2); k++) {
                for (int j = 0; j < grid.n(1); j++) {
                    for (int i = 0; i < grid.n(0); i++) {
                        pdeFlag[p++] = m.isPdeNode(i, j, k);
                    }
                }
            }
            final EllipticEngine self = this;
            NewtonKrylov.Jacobian jac = m.ensureJacobian() ? new NewtonKrylov.Jacobian() {
                @Override
                public void setPoint(double[] z) {
                    self.unpack(z, self.sBuf);
                    self.bc.fillHalos(self.sBuf);
                    self.m.linearize(0.0, self.sBuf);
                }

                @Override
                public void apply(double[] v, double[] out) {
                    self.jacMatvec(v, out);
                }

                @Override
                public boolean diagonal(double[] d) {
                    return self.jacDiagonal(d);
                }
            } : null;
            nk = new NewtonKrylov(ns * n, new NewtonKrylov.Residual() {
                @Override
                public void eval(double[] z, double[] out) {
                    self.residualVector(z, out);
                }
            }, jac, o);
            nk.preconditioning = o.jacobiPreconditioner;
            diag.method += m.analyticActive ? ", exact AD Jacobian" : ", finite-difference Jacobian";
        }
        int[] ni = m.nodeIdx;
        for (int s = 0; s < ns; s++) {
            for (int c = 0; c < n; c++) {
                zVec[s * n + c] = y[s][ni[c]];
            }
        }
        long n0 = nk.newtonIterations;
        long k0 = nk.krylovIterations;
        boolean ok = nk.solve(zVec);
        diag.newtonIterations += nk.newtonIterations - n0;
        diag.krylovIterations += nk.krylovIterations - k0;
        diag.iterations = (int) (nk.newtonIterations - n0);
        diag.initialResidualNorm = nk.initialResidualNorm;
        unpack(zVec, y);
        bc.apply(y, 0.0);
        m.residual(y, res);
        double exact = 0;
        for (int e = 0; e < m.nc; e++) {
            for (int k = m.pdeLo[2]; k <= m.pdeHi[2]; k++) {
                for (int j = m.pdeLo[1]; j <= m.pdeHi[1]; j++) {
                    int base = grid.index(0, j, k);
                    for (int i = m.pdeLo[0]; i <= m.pdeHi[0]; i++) {
                        double a = Math.abs(res[e][base + i]);
                        if (a > exact || Double.isNaN(a)) {
                            exact = a;
                        }
                    }
                }
            }
        }
        diag.residualNorm = exact;
        if (!ok) {
            diag.warnings.add("Newton-Krylov did not converge (residual " + nk.lastResidualNorm + " after "
                    + (nk.newtonIterations - n0) + " Newton iterations)");
        }
        return ok;
    }

    private void residualVector(double[] z, double[] out) {
        int ns = m.nState;
        int n = grid.nodeCount();
        int[] ni = m.nodeIdx;
        unpack(z, sBuf);
        for (int s = 0; s < ns; s++) {
            System.arraycopy(sBuf[s], 0, wBuf[s], 0, size);
        }
        bc.applyConstraints(wBuf, 0.0);
        bc.fillHalos(sBuf);
        m.residual(sBuf, res);
        for (int s = 0; s < ns; s++) {
            int off = s * n;
            for (int c = 0; c < n; c++) {
                int idx = ni[c];
                out[off + c] = pdeFlag[c] ? res[s][idx] : z[off + c] - wBuf[s][idx];
            }
        }
    }

    /**
     * Diagonal of the Newton Jacobian from the stored automatic-differentiation partials: for a PDE row of
     * unknown {@code s} it is {@code sum_j dR/dsym_j * (centre weight of sym_j in unknown s)}; boundary rows
     * are identity rows.
     */
    private boolean jacDiagonal(double[] d) {
        int ns = m.nState;
        int n = grid.nodeCount();
        int[] ni = m.nodeIdx;
        for (int s = 0; s < ns; s++) {
            int[] js = m.jSym[s];
            for (int c = 0; c < n; c++) {
                int idx = ni[c];
                if (!pdeFlag[c]) {
                    d[s * n + c] = 1.0;
                    continue;
                }
                double acc = 0;
                for (int q = 0; q < js.length; q++) {
                    int sym = js[q];
                    double w;
                    if (sym < m.symTerm) {
                        w = sym == Model.SYM_STATE + s ? 1.0 : 0.0;
                    } else {
                        Model.Term tm = m.terms[sym - m.symTerm];
                        w = tm.srcVar == s ? tm.centerWeight : 0.0;
                    }
                    if (w != 0.0) {
                        acc += m.jac[s][q][idx] * w;
                    }
                }
                d[s * n + c] = acc;
            }
        }
        return true;
    }

    private void jacMatvec(double[] v, double[] out) {
        int ns = m.nState;
        int n = grid.nodeCount();
        int[] ni = m.nodeIdx;
        unpack(v, vBuf);
        for (int s = 0; s < ns; s++) {
            System.arraycopy(vBuf[s], 0, wBuf[s], 0, size);
        }
        bc.applyConstraintsHomogeneous(wBuf);
        bc.fillHalos(vBuf);
        m.jacApply(vBuf, jvBuf);
        for (int s = 0; s < ns; s++) {
            int off = s * n;
            for (int c = 0; c < n; c++) {
                int idx = ni[c];
                out[off + c] = pdeFlag[c] ? jvBuf[s][idx] : v[off + c] - wBuf[s][idx];
            }
        }
    }
}
