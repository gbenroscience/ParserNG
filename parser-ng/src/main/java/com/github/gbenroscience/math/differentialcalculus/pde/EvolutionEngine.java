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
 * Time integration of the semi-discrete (method-of-lines) system
 * {@code y' = F(t,y)} produced by {@link Model}. Every work array is
 * allocated up front; {@link #step(double)} allocates nothing.
 */
final class EvolutionEngine {

    private static final double[][] EMPTY = new double[0][];

    final Model m;
    final Boundaries bc;
    final PDEOptions o;
    final SolverDiagnostics diag;
    final Grid grid;
    final int ns;
    final int size;
    final double[][] y;
    double t;
    TimeIntegrator integ;

    // work arrays (allocated per integrator)
    private double[][] k1, k2, k3, k4, k5, k6, k7, tmp, ynew;
    private double[][] yOld, fOld, probeA, probeB, probeV;
    private double[][] sBuf, wBuf, kBuf, vBuf, jvBuf;

    // adaptive controller
    private double hCur = Double.NaN;
    private boolean fsalValid;
    // Verlet cache
    private boolean accValid;
    // implicit machinery
    private NewtonKrylov nk;
    private double[] zVec;
    private double[] yOldC;
    private boolean[] pdeFlag;
    private double implicitDt, implicitTheta, implicitT;
    private boolean jacobianExact;
    // stability
    private double rhoJ = Double.NaN;
    private double verifiedStableDt = 0;
    private long stepsAtLastCheck = 0;
    private double stateScale = 1.0;
    private double lastGrowth = Double.NaN;
    private long stepCounter = 0;

    private final CombineTask combine;
    private final ArrayAxpyTask axpyTask = new ArrayAxpyTask();
    private final ErrTask errTask;

    EvolutionEngine(Model m, Boundaries bc, double[][] y, SolverDiagnostics diag) {
        this.m = m;
        this.bc = bc;
        this.o = m.opt;
        this.diag = diag;
        this.grid = m.grid;
        this.ns = m.nState;
        this.size = grid.storageSize();
        this.y = y;
        this.combine = new CombineTask();
        this.errTask = new ErrTask(m.pool.size());
        this.integ = resolve(o.integrator);
        allocate();
    }

    // ------------------------------------------------------------ set-up
    private TimeIntegrator resolve(TimeIntegrator req) {
        TimeIntegrator r = req;
        if (r == TimeIntegrator.AUTO) {
            r = m.type == PDEType.HYPERBOLIC ? TimeIntegrator.RK4 : TimeIntegrator.CRANK_NICOLSON;
        }
        if (r == TimeIntegrator.VELOCITY_VERLET) {
            for (int c = 0; c < m.nc; c++) {
                if (m.timeOrder[c] != 2) {
                    throw new PDEConfigurationException("VELOCITY_VERLET needs every unknown to be second order in time; '"
                            + m.comps[c] + "' is first order");
                }
            }
            for (int e = 0; e < m.nc; e++) {
                for (int s : m.eqs[e].srcSym) {
                    for (int c = 0; c < m.nc; c++) {
                        if (s == Model.SYM_STATE + m.velVar[c]) {
                            throw new PDEConfigurationException("VELOCITY_VERLET requires the force to be independent of u_t "
                                    + "(damping present in '" + m.eqs[e].source + "'); use RK4 or CRANK_NICOLSON");
                        }
                    }
                }
            }
        }
        return r;
    }

    private double[][] arr() {
        double[][] a = new double[ns][size];
        return a;
    }

    private void allocate() {
        k1 = arr();
        switch (integ) {
            case EULER:
                break;
            case HEUN:
                k2 = arr();
                tmp = arr();
                break;
            case RK4:
                k2 = arr();
                k3 = arr();
                k4 = arr();
                tmp = arr();
                break;
            case DORMAND_PRINCE:
                k2 = arr();
                k3 = arr();
                k4 = arr();
                k5 = arr();
                k6 = arr();
                k7 = arr();
                tmp = arr();
                ynew = arr();
                break;
            case VELOCITY_VERLET:
                k2 = arr();
                break;
            default: // implicit
                tmp = arr();
                kBuf = arr();
                yOld = arr();
                fOld = arr();
                sBuf = arr();
                wBuf = arr();
                vBuf = arr();
                jvBuf = arr();
                int n = grid.nodeCount() * ns;
                zVec = new double[n];
                yOldC = new double[n];
                pdeFlag = new boolean[grid.nodeCount()];
                int p = 0;
                for (int k = 0; k < grid.n(2); k++) {
                    for (int j = 0; j < grid.n(1); j++) {
                        for (int i = 0; i < grid.n(0); i++) {
                            pdeFlag[p++] = m.isPdeNode(i, j, k);
                        }
                    }
                }
                final EvolutionEngine self = this;
                jacobianExact = m.ensureJacobian();
                NewtonKrylov.Jacobian jac = jacobianExact ? new NewtonKrylov.Jacobian() {
                    @Override
                    public void setPoint(double[] z) {
                        self.jacSetPoint(z);
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
                nk = new NewtonKrylov(n, new NewtonKrylov.Residual() {
                    @Override
                    public void eval(double[] z, double[] out) {
                        self.implicitResidual(z, out);
                    }
                }, jac, o);
                nk.preconditioning = o.jacobiPreconditioner;
        }
        if (integ.isExplicit() && integ != TimeIntegrator.DORMAND_PRINCE) {
            probeA = arr();
            probeB = arr();
            probeV = arr();
            if (tmp == null) {
                tmp = arr();
            }
        }
        if (integ == TimeIntegrator.EULER || integ == TimeIntegrator.VELOCITY_VERLET || integ == TimeIntegrator.HEUN) {
            if (tmp == null) {
                tmp = arr();
            }
        }
        diag.method = integ.name() + " / method of lines, spatial order " + o.spatialOrder
                + ", first-derivative scheme " + o.firstDerivativeScheme + (integ.isExplicit() ? "" : (jacobianExact ? ", exact AD Jacobian" : ", finite-difference Jacobian"));
        diag.type = m.type;
        diag.spatialOrder = o.spatialOrder;
        diag.threads = m.threads;
    }

    /** Called after the initial condition has been loaded into y. */
    void initialize(double t0) {
        t = t0;
        bc.apply(y, t);
        double mx = 1.0;
        for (int s = 0; s < ns; s++) {
            for (int i = 0; i < size; i++) {
                mx = Math.max(mx, Math.abs(y[s][i]));
            }
        }
        stateScale = mx;
        fsalValid = false;
        accValid = false;
        hCur = Double.NaN;
        verifiedStableDt = 0;
        stepCounter = 0;
        stepsAtLastCheck = 0;
        diag.steps = 0;
        diag.rejectedSteps = 0;
        diag.finalTime = t;
        diag.newtonIterations = 0;
        diag.krylovIterations = 0;
        if (nk != null) {
            nk.newtonIterations = 0;
            nk.krylovIterations = 0;
        }
        rhoJ = Double.NaN;
        if (o.stabilityPolicy != StabilityPolicy.OFF) {
            estimateJacobianRadius();
        }
    }

    void invalidateCaches() {
        fsalValid = false;
        accValid = false;
    }

    // ------------------------------------------------------------ tasks
    /** {@code dst = base + sum c_q * ks_q}. */
    private final class CombineTask implements WorkerPool.Task {

        double[][] dst;
        double[][] base;
        int nt;
        final double[] c = new double[8];
        final double[][][] ks = new double[8][][];

        @Override
        public void run(int w, int nw) {
            int lo = (int) ((long) size * w / nw);
            int hi = (int) ((long) size * (w + 1) / nw);
            for (int s = 0; s < ns; s++) {
                double[] d = dst[s];
                double[] b = base[s];
                int q = 0;
                if (nt == 0) {
                    if (d != b) {
                        System.arraycopy(b, lo, d, lo, hi - lo);
                    }
                    continue;
                }
                // first pass fuses the base with up to two terms
                if (nt >= 2) {
                    double c0 = c[0];
                    double c1 = c[1];
                    double[] a0 = ks[0][s];
                    double[] a1 = ks[1][s];
                    for (int i = lo; i < hi; i++) {
                        d[i] = b[i] + c0 * a0[i] + c1 * a1[i];
                    }
                    q = 2;
                } else {
                    double c0 = c[0];
                    double[] a0 = ks[0][s];
                    for (int i = lo; i < hi; i++) {
                        d[i] = b[i] + c0 * a0[i];
                    }
                    q = 1;
                }
                while (q + 1 < nt) {
                    double c0 = c[q];
                    double c1 = c[q + 1];
                    double[] a0 = ks[q][s];
                    double[] a1 = ks[q + 1][s];
                    for (int i = lo; i < hi; i++) {
                        d[i] += c0 * a0[i] + c1 * a1[i];
                    }
                    q += 2;
                }
                if (q < nt) {
                    double c0 = c[q];
                    double[] a0 = ks[q][s];
                    for (int i = lo; i < hi; i++) {
                        d[i] += c0 * a0[i];
                    }
                }
            }
        }
    }

    /** {@code d += c * s} for one pair of arrays. */
    private final class ArrayAxpyTask implements WorkerPool.Task {

        double[] d;
        double[] s;
        double c;

        @Override
        public void run(int w, int nw) {
            int lo = (int) ((long) size * w / nw);
            int hi = (int) ((long) size * (w + 1) / nw);
            double cc = c;
            double[] dd = d;
            double[] ss = s;
            for (int i = lo; i < hi; i++) {
                dd[i] += cc * ss[i];
            }
        }
    }

    /** Scaled squared-error reduction for the adaptive integrator. */
    private final class ErrTask implements WorkerPool.Task {

        final double[] part;
        double h;
        double rtol;
        double atol;
        final double[] e = new double[8];

        ErrTask(int workers) {
            part = new double[workers];
        }

        @Override
        public void run(int w, int nw) {
            int lo = (int) ((long) size * w / nw);
            int hi = (int) ((long) size * (w + 1) / nw);
            double sum = 0;
            for (int s = 0; s < ns; s++) {
                double[] a1 = k1[s], a3 = k3[s], a4 = k4[s], a5 = k5[s], a6 = k6[s], a7 = k7[s];
                double[] yo = y[s];
                double[] yn = ynew[s];
                for (int i = lo; i < hi; i++) {
                    double err = h * (e[0] * a1[i] + e[2] * a3[i] + e[3] * a4[i] + e[4] * a5[i] + e[5] * a6[i] + e[6] * a7[i]);
                    double sc = atol + rtol * Math.max(Math.abs(yo[i]), Math.abs(yn[i]));
                    double q = err / sc;
                    sum += q * q;
                }
            }
            part[w] = sum;
        }
    }

    private void comb(double[][] dst, double[][] base, double c0, double[][] a0) {
        combine.dst = dst;
        combine.base = base;
        combine.nt = 1;
        combine.c[0] = c0;
        combine.ks[0] = a0;
        m.pool.run(combine);
    }

    private void comb(double[][] dst, double[][] base, double c0, double[][] a0, double c1, double[][] a1) {
        combine.dst = dst;
        combine.base = base;
        combine.nt = 2;
        combine.c[0] = c0;
        combine.ks[0] = a0;
        combine.c[1] = c1;
        combine.ks[1] = a1;
        m.pool.run(combine);
    }

    private void comb(double[][] dst, double[][] base, int nt, double[] c, double[][][] ks) {
        combine.dst = dst;
        combine.base = base;
        combine.nt = nt;
        for (int q = 0; q < nt; q++) {
            combine.c[q] = c[q];
            combine.ks[q] = ks[q];
        }
        m.pool.run(combine);
    }

    private void axpy(double[] d, double[] s, double c) {
        axpyTask.d = d;
        axpyTask.s = s;
        axpyTask.c = c;
        m.pool.run(axpyTask);
    }

    // ------------------------------------------------------------ public stepping
    /** Advances exactly {@code dt}. */
    void step(double dt) {
        if (!(dt > 0) || Double.isInfinite(dt)) {
            throw new PDEConfigurationException("Time step must be positive and finite, got " + dt);
        }
        long t0 = System.nanoTime();
        if (integ == TimeIntegrator.DORMAND_PRINCE) {
            adaptiveAdvance(dt);
        } else if (integ.isExplicit()) {
            explicitWithPolicy(dt);
        } else {
            implicitStep(dt, 0);
            t += dt;
            countStep(dt);
        }
        diag.finalTime = t;
        diag.wallNanos += System.nanoTime() - t0;
        stepCounter++;
        if (o.blowupCheckSteps > 0 && stepCounter % o.blowupCheckSteps == 0) {
            checkDivergence();
        }
    }

    private void countStep(double dt) {
        diag.steps++;
        diag.lastDt = dt;
        diag.minDt = Double.isNaN(diag.minDt) ? dt : Math.min(diag.minDt, dt);
        diag.maxDt = Double.isNaN(diag.maxDt) ? dt : Math.max(diag.maxDt, dt);
    }

    private void explicitWithPolicy(double dt) {
        StabilityPolicy pol = o.stabilityPolicy;
        if (pol != StabilityPolicy.OFF) {
            if (o.stabilityRecheckSteps > 0 && diag.steps - stepsAtLastCheck >= o.stabilityRecheckSteps) {
                verifiedStableDt = 0;
                estimateJacobianRadius();
                stepsAtLastCheck = diag.steps;
            }
            if (dt > verifiedStableDt * (1 + 1e-12)) {
                lastGrowth = Double.NaN;
                if (unstableAt(dt)) {
                    String growth = Double.isNaN(lastGrowth) ? "" : (" the linearised amplification factor per step is "
                            + String.format("%.6g", lastGrowth) + " (> 1);");
                    String msg = "Time step dt=" + dt + " is unstable for " + integ + ":" + growth
                            + " dt * spectralRadius = " + String.format("%.4g", dt * rhoJ) + " exceeds the stability interval ~"
                            + realAxisBound() + ". Estimated stability limit dt <~ " + String.format("%.4g", maxStableDt())
                            + " (spectral radius of the semi-discrete Jacobian ~ " + String.format("%.4g", rhoJ)
                            + "). Reduce dt, or use an implicit integrator (CRANK_NICOLSON / BACKWARD_EULER).";
                    if (pol == StabilityPolicy.THROW) {
                        throw new UnstableTimeStepException(msg, dt, maxStableDt());
                    } else if (pol == StabilityPolicy.WARN) {
                        if (diag.warnings.size() < 20 && !diag.warnings.contains(msg)) {
                            diag.warnings.add(msg);
                        }
                    } else { // SUBSTEP
                        double sub = dt;
                        int guard = 0;
                        while (unstableAt(sub) && guard++ < 40) {
                            sub *= 0.5;
                        }
                        int n = (int) Math.ceil(dt / sub - 1e-12);
                        double h = dt / n;
                        for (int q = 0; q < n; q++) {
                            explicitStepOn(y, t, h, true);
                            t += h;
                            countStep(h);
                        }
                        return;
                    }
                } else {
                    verifiedStableDt = dt;
                }
            }
        }
        explicitStepOn(y, t, dt, true);
        t += dt;
        countStep(dt);
    }

    // ------------------------------------------------------------ explicit one-step maps
    /** One explicit step on {@code s} (which must already satisfy the boundary conditions at time {@code tt}). */
    private void explicitStepOn(double[][] s, double tt, double dt, boolean cache) {
        switch (integ) {
            case EULER:
                m.rhs(tt, s, k1);
                comb(s, s, dt, k1);
                bc.apply(s, tt + dt);
                break;
            case HEUN:
                m.rhs(tt, s, k1);
                comb(tmp, s, dt, k1);
                bc.apply(tmp, tt + dt);
                m.rhs(tt + dt, tmp, k2);
                comb(s, s, 0.5 * dt, k1, 0.5 * dt, k2);
                bc.apply(s, tt + dt);
                break;
            case RK4:
                m.rhs(tt, s, k1);
                comb(tmp, s, 0.5 * dt, k1);
                bc.apply(tmp, tt + 0.5 * dt);
                m.rhs(tt + 0.5 * dt, tmp, k2);
                comb(tmp, s, 0.5 * dt, k2);
                bc.apply(tmp, tt + 0.5 * dt);
                m.rhs(tt + 0.5 * dt, tmp, k3);
                comb(tmp, s, dt, k3);
                bc.apply(tmp, tt + dt);
                m.rhs(tt + dt, tmp, k4);
                combine.dst = s;
                combine.base = s;
                combine.nt = 4;
                combine.c[0] = dt / 6.0;
                combine.ks[0] = k1;
                combine.c[1] = dt / 3.0;
                combine.ks[1] = k2;
                combine.c[2] = dt / 3.0;
                combine.ks[2] = k3;
                combine.c[3] = dt / 6.0;
                combine.ks[3] = k4;
                m.pool.run(combine);
                bc.apply(s, tt + dt);
                break;
            case VELOCITY_VERLET:
                verletStep(s, tt, dt, cache);
                break;
            default:
                throw new IllegalStateException("not an explicit fixed-step integrator: " + integ);
        }
    }

    private void verletStep(double[][] s, double tt, double dt, boolean cache) {
        if (!cache || !accValid) {
            m.rhs(tt, s, k1);
        }
        for (int c = 0; c < m.nc; c++) {
            axpy(s[m.velVar[c]], k1[m.velVar[c]], 0.5 * dt);
            axpy(s[m.valVar[c]], s[m.velVar[c]], dt);
        }
        bc.apply(s, tt + dt);
        m.rhs(tt + dt, s, k2);
        for (int c = 0; c < m.nc; c++) {
            axpy(s[m.velVar[c]], k2[m.velVar[c]], 0.5 * dt);
        }
        bc.apply(s, tt + dt);
        // new acceleration becomes the cached "k1" for the next step
        double[][] swap = k1;
        k1 = k2;
        k2 = swap;
        accValid = cache;
    }

    /**
     * Instability test for an explicit step: (1) the step must lie within the
     * integrator's stability interval for the estimated Jacobian spectral
     * radius, and (2) the power-iterated amplification of the integrator's
     * one-step map must not show strong growth (this is what rejects Euler /
     * Heun on oscillatory problems, whose stability interval on the imaginary
     * axis is empty). Marginal growth is left to the divergence guard.
     */
    private boolean unstableAt(double dt) {
        double bound = realAxisBound();
        if (!Double.isNaN(rhoJ) && dt * rhoJ > bound * 1.02) {
            return true;
        }
        lastGrowth = probeGrowth(dt);
        return lastGrowth > 1.01 || Double.isNaN(lastGrowth);
    }

    // ------------------------------------------------------------ stability analysis
    private double realAxisBound() {
        switch (integ) {
            case EULER:
            case HEUN:
            case VELOCITY_VERLET:
                return 2.0;
            case RK4:
                return 2.78;
            case DORMAND_PRINCE:
                return 3.3;
            default:
                return Double.POSITIVE_INFINITY;
        }
    }

    double maxStableDt() {
        if (Double.isNaN(rhoJ) || rhoJ == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return realAxisBound() / rhoJ;
    }

    double spectralRadius() {
        return rhoJ;
    }

    /** CFL-type step suggestion: 80% of the RK4 real-axis limit, from the Jacobian spectral radius. */
    double suggestTimeStep() {
        if (Double.isNaN(rhoJ)) {
            estimateJacobianRadius();
        }
        return rhoJ > 0 ? 0.8 * 2.78 / rhoJ : 1e-3;
    }

    private static double norm(double[][] a, Model m, int ns) {
        return 0;
    }

    private void perturbRandom(double[][] v) {
        long seed = 88172645463325252L;
        for (int s = 0; s < ns; s++) {
            Arrays.fill(v[s], 0.0);
        }
        for (int k = m.pdeLo[2]; k <= m.pdeHi[2]; k++) {
            for (int j = m.pdeLo[1]; j <= m.pdeHi[1]; j++) {
                for (int i = m.pdeLo[0]; i <= m.pdeHi[0]; i++) {
                    int idx = grid.index(i, j, k);
                    for (int s = 0; s < ns; s++) {
                        seed ^= seed << 13;
                        seed ^= seed >>> 7;
                        seed ^= seed << 17;
                        v[s][idx] = ((seed >>> 11) * (1.0 / 9007199254740992.0)) - 0.5;
                    }
                }
            }
        }
    }

    private double pdeNorm(double[][] v) {
        double sum = 0;
        for (int k = m.pdeLo[2]; k <= m.pdeHi[2]; k++) {
            for (int j = m.pdeLo[1]; j <= m.pdeHi[1]; j++) {
                int base = grid.index(0, j, k);
                for (int i = m.pdeLo[0]; i <= m.pdeHi[0]; i++) {
                    for (int s = 0; s < ns; s++) {
                        double q = v[s][base + i];
                        sum += q * q;
                    }
                }
            }
        }
        return Math.sqrt(sum);
    }

    private void scalePde(double[][] v, double f) {
        for (int k = m.pdeLo[2]; k <= m.pdeHi[2]; k++) {
            for (int j = m.pdeLo[1]; j <= m.pdeHi[1]; j++) {
                int base = grid.index(0, j, k);
                for (int i = m.pdeLo[0]; i <= m.pdeHi[0]; i++) {
                    for (int s = 0; s < ns; s++) {
                        v[s][base + i] *= f;
                    }
                }
            }
        }
    }

    /**
     * Power iteration on the Jacobian of the semi-discrete system, using
     * differences of the actual right-hand side. Gives the spectral radius
     * that sets CFL-type limits.
     */
    void estimateJacobianRadius() {
        double[][] pa = tmp != null ? tmp : (tmp = arr());
        double[][] f0 = k1;
        double[][] f1 = k2 != null ? k2 : (k2 = arr());
        double[][] v = probeV != null ? probeV : (probeV = arr());
        // base F
        bc.apply(y, t);
        m.rhs(t, y, f0);
        perturbRandom(v);
        double nv = pdeNorm(v);
        if (nv == 0) {
            rhoJ = 0;
            return;
        }
        scalePde(v, 1.0 / nv);
        double yn = 0;
        for (int s = 0; s < ns; s++) {
            for (int i = 0; i < size; i++) {
                yn = Math.max(yn, Math.abs(y[s][i]));
            }
        }
        double eps = 1e-7 * (1.0 + yn);
        double rho = 0;
        double logSum = 0;
        int cnt = 0;
        for (int it = 0; it < 80; it++) {
            for (int s = 0; s < ns; s++) {
                double[] a = pa[s], b = y[s], c = v[s];
                for (int i = 0; i < size; i++) {
                    a[i] = b[i] + eps * c[i];
                }
            }
            bc.apply(pa, t);
            m.rhs(t, pa, f1);
            // w = (f1 - f0)/eps  -> store in v (PDE nodes)
            for (int k = m.pdeLo[2]; k <= m.pdeHi[2]; k++) {
                for (int j = m.pdeLo[1]; j <= m.pdeHi[1]; j++) {
                    int base = grid.index(0, j, k);
                    for (int i = m.pdeLo[0]; i <= m.pdeHi[0]; i++) {
                        for (int s = 0; s < ns; s++) {
                            v[s][base + i] = (f1[s][base + i] - f0[s][base + i]) / eps;
                        }
                    }
                }
            }
            double wn = pdeNorm(v);
            if (wn == 0 || Double.isNaN(wn) || Double.isInfinite(wn)) {
                rho = wn == 0 ? 0 : Double.POSITIVE_INFINITY;
                break;
            }
            rho = wn;
            if (it >= 40) {
                // J is generally non-normal (e.g. wave systems): the per-step norm ratio alternates and only its
                // geometric mean converges to the spectral radius
                logSum += Math.log(wn);
                cnt++;
            }
            scalePde(v, 1.0 / wn);
        }
        if (cnt > 0 && rho != 0 && !Double.isInfinite(rho)) {
            rho = Math.exp(logSum / cnt);
        }
        rhoJ = rho;
        bc.apply(y, t);
        invalidateCaches();
        diag.spectralRadius = rhoJ;
        diag.maxStableDt = maxStableDt();
    }

    /**
     * Estimates the per-step growth factor of the linearised map of the CHOSEN
     * integrator at step {@code dt} by power iteration on the map itself
     * (so it is exact for the integrator's real stability region, including
     * the imaginary-axis behaviour of oscillatory problems).
     */
    private double probeGrowth(double dt) {
        double[][] a = probeA;
        double[][] b = probeB;
        double[][] v = probeV;
        for (int s = 0; s < ns; s++) {
            System.arraycopy(y[s], 0, a[s], 0, size);
        }
        // the three buffers k1/k2/tmp are scratch for the probe; remember to invalidate caches afterwards
        accValid = false;
        perturbRandom(v);
        double nv = pdeNorm(v);
        if (nv == 0) {
            return 0;
        }
        scalePde(v, 1.0 / nv);
        double yn = 0;
        for (int s = 0; s < ns; s++) {
            for (int i = 0; i < size; i++) {
                yn = Math.max(yn, Math.abs(y[s][i]));
            }
        }
        double eps = 1e-6 * (1.0 + yn);
        double logSum = 0;
        int cnt = 0;
        int iters = 48;
        // advance the base once (A -> A1), then iterate on differences
        bc.apply(a, t);
        explicitStepOn(a, t, dt, false);
        for (int it = 0; it < iters; it++) {
            for (int s = 0; s < ns; s++) {
                double[] pb = b[s], pa = y[s], pv = v[s];
                for (int i = 0; i < size; i++) {
                    pb[i] = pa[i] + eps * pv[i];
                }
            }
            bc.apply(b, t);
            accValid = false;
            explicitStepOn(b, t, dt, false);
            // w = (B1 - A1)/eps on PDE nodes (A1 = a)
            for (int k = m.pdeLo[2]; k <= m.pdeHi[2]; k++) {
                for (int j = m.pdeLo[1]; j <= m.pdeHi[1]; j++) {
                    int base = grid.index(0, j, k);
                    for (int i = m.pdeLo[0]; i <= m.pdeHi[0]; i++) {
                        for (int s = 0; s < ns; s++) {
                            v[s][base + i] = (b[s][base + i] - a[s][base + i]) / eps;
                        }
                    }
                }
            }
            double wn = pdeNorm(v);
            if (Double.isNaN(wn) || Double.isInfinite(wn)) {
                invalidateCaches();
                return Double.POSITIVE_INFINITY;
            }
            if (wn == 0) {
                invalidateCaches();
                return 0;
            }
            if (it >= iters - 24) {
                logSum += Math.log(wn);
                cnt++;
            }
            scalePde(v, 1.0 / wn);
        }
        invalidateCaches();
        bc.apply(y, t);
        return Math.exp(logSum / cnt);
    }

    // ------------------------------------------------------------ divergence guard
    void checkDivergence() {
        double mx = 0;
        for (int s = 0; s < ns; s++) {
            double[] a = y[s];
            for (int k = 0; k < grid.n(2); k++) {
                for (int j = 0; j < grid.n(1); j++) {
                    int base = grid.index(0, j, k);
                    for (int i = 0; i < grid.n(0); i++) {
                        double v = a[base + i];
                        if (Double.isNaN(v) || Double.isInfinite(v)) {
                            throw new UnstableTimeStepException("Solution became non-finite at t=" + t
                                    + " (unknown " + m.stateNames[s] + "): the scheme is unstable or the problem is singular",
                                    diag.lastDt, maxStableDt());
                        }
                        mx = Math.max(mx, Math.abs(v));
                    }
                }
            }
        }
        if (mx > o.blowupFactor * stateScale) {
            throw new UnstableTimeStepException("Solution is diverging at t=" + t + ": max|u|=" + mx + " exceeds "
                    + o.blowupFactor + " x its initial scale (" + stateScale + "). Reduce dt or use an implicit integrator",
                    diag.lastDt, maxStableDt());
        }
    }

    // ------------------------------------------------------------ adaptive Dormand-Prince 5(4)
    private static final double A21 = 1.0 / 5.0,
            A31 = 3.0 / 40.0, A32 = 9.0 / 40.0,
            A41 = 44.0 / 45.0, A42 = -56.0 / 15.0, A43 = 32.0 / 9.0,
            A51 = 19372.0 / 6561.0, A52 = -25360.0 / 2187.0, A53 = 64448.0 / 6561.0, A54 = -212.0 / 729.0,
            A61 = 9017.0 / 3168.0, A62 = -355.0 / 33.0, A63 = 46732.0 / 5247.0, A64 = 49.0 / 176.0, A65 = -5103.0 / 18656.0,
            B1 = 35.0 / 384.0, B3 = 500.0 / 1113.0, B4 = 125.0 / 192.0, B5 = -2187.0 / 6784.0, B6 = 11.0 / 84.0,
            E1 = 71.0 / 57600.0, E3 = -71.0 / 16695.0, E4 = 71.0 / 1920.0, E5 = -17253.0 / 339200.0, E6 = 22.0 / 525.0, E7 = -1.0 / 40.0;

    private void adaptiveAdvance(double dtTotal) {
        double tEnd = t + dtTotal;
        if (Double.isNaN(hCur)) {
            hCur = Math.min(dtTotal, Math.min(o.dtMax, Double.isNaN(o.dt) ? dtTotal : o.dt));
        }
        int guard = 0;
        while (t < tEnd - 1e-13 * Math.max(1.0, Math.abs(tEnd))) {
            double h = Math.min(hCur, tEnd - t);
            h = Math.min(h, o.dtMax);
            if (h < o.dtMin) {
                throw new UnstableTimeStepException("Adaptive step fell below dtMin=" + o.dtMin + " at t=" + t
                        + "; the problem is too stiff for an explicit adaptive method (try CRANK_NICOLSON or BACKWARD_EULER)",
                        h, Double.NaN);
            }
            double err = dpTrial(h);
            diag.errorEstimate = err;
            if (err <= 1.0 || h <= o.dtMin * 1.0000001) {
                // accept: ynew -> y
                for (int s = 0; s < ns; s++) {
                    System.arraycopy(ynew[s], 0, y[s], 0, size);
                }
                t += h;
                // FSAL: k7 becomes k1
                double[][] sw = k1;
                k1 = k7;
                k7 = sw;
                fsalValid = true;
                countStep(h);
            } else {
                diag.rejectedSteps++;
                fsalValid = true; // k1 is still valid for the unchanged y
            }
            double fac = err == 0 ? 5.0 : Math.max(0.2, Math.min(5.0, 0.9 * Math.pow(err, -0.2)));
            hCur = h * fac;
            if (++guard > 10_000_000) {
                throw new PDEException("Adaptive integrator exceeded its step budget");
            }
        }
    }

    /** Computes one trial step into {@code ynew}, leaves k7 = F(t+h, ynew); returns the scaled error norm. */
    private double dpTrial(double h) {
        if (!fsalValid) {
            bc.apply(y, t);
            m.rhs(t, y, k1);
        }
        comb(tmp, y, h * A21, k1);
        bc.apply(tmp, t + h * 0.2);
        m.rhs(t + 0.2 * h, tmp, k2);
        comb(tmp, y, h * A31, k1, h * A32, k2);
        bc.apply(tmp, t + 0.3 * h);
        m.rhs(t + 0.3 * h, tmp, k3);
        comb(tmp, y, h * A41, k1, h * A42, k2);
        axpyAll(tmp, k3, h * A43);
        bc.apply(tmp, t + 0.8 * h);
        m.rhs(t + 0.8 * h, tmp, k4);
        comb(tmp, y, h * A51, k1, h * A52, k2);
        axpyAll(tmp, k3, h * A53);
        axpyAll(tmp, k4, h * A54);
        bc.apply(tmp, t + 8.0 / 9.0 * h);
        m.rhs(t + 8.0 / 9.0 * h, tmp, k5);
        comb(tmp, y, h * A61, k1, h * A62, k2);
        axpyAll(tmp, k3, h * A63);
        axpyAll(tmp, k4, h * A64);
        axpyAll(tmp, k5, h * A65);
        bc.apply(tmp, t + h);
        m.rhs(t + h, tmp, k6);
        combine.dst = ynew;
        combine.base = y;
        combine.nt = 5;
        combine.c[0] = h * B1;
        combine.ks[0] = k1;
        combine.c[1] = h * B3;
        combine.ks[1] = k3;
        combine.c[2] = h * B4;
        combine.ks[2] = k4;
        combine.c[3] = h * B5;
        combine.ks[3] = k5;
        combine.c[4] = h * B6;
        combine.ks[4] = k6;
        m.pool.run(combine);
        bc.apply(ynew, t + h);
        m.rhs(t + h, ynew, k7);
        errTask.h = h;
        errTask.rtol = o.rtol;
        errTask.atol = o.atol;
        errTask.e[0] = E1;
        errTask.e[2] = E3;
        errTask.e[3] = E4;
        errTask.e[4] = E5;
        errTask.e[5] = E6;
        errTask.e[6] = E7;
        m.pool.run(errTask);
        double sum = 0;
        for (double p : errTask.part) {
            sum += p;
        }
        double dof = Math.max(1.0, (double) m.pdeNodeCount() * ns);
        double e = Math.sqrt(sum / dof);
        return Double.isNaN(e) ? Double.POSITIVE_INFINITY : e;
    }

    private void axpyAll(double[][] dst, double[][] src, double c) {
        for (int s = 0; s < ns; s++) {
            axpy(dst[s], src[s], c);
        }
    }

    // ------------------------------------------------------------ implicit theta-method
    private void unpack(double[] z, double[][] a) {
        int n = grid.nodeCount();
        int[] ni = m.nodeIdx;
        for (int s = 0; s < ns; s++) {
            double[] as = a[s];
            int off = s * n;
            for (int c = 0; c < n; c++) {
                as[ni[c]] = z[off + c];
            }
        }
    }

    private void pack(double[][] a, double[] z) {
        int n = grid.nodeCount();
        int[] ni = m.nodeIdx;
        for (int s = 0; s < ns; s++) {
            double[] as = a[s];
            int off = s * n;
            for (int c = 0; c < n; c++) {
                z[off + c] = as[ni[c]];
            }
        }
    }

    /** R(z) of the theta-method step. */
    void implicitResidual(double[] z, double[] out) {
        int n = grid.nodeCount();
        int[] ni = m.nodeIdx;
        unpack(z, sBuf);
        for (int s = 0; s < ns; s++) {
            System.arraycopy(sBuf[s], 0, wBuf[s], 0, size);
        }
        bc.applyConstraints(wBuf, implicitT);
        bc.fillHalos(sBuf);
        m.rhs(implicitT, sBuf, kBuf);
        double th = implicitTheta;
        double dt = implicitDt;
        for (int s = 0; s < ns; s++) {
            double[] kb = kBuf[s];
            double[] fo = fOld[s];
            double[] wb = wBuf[s];
            int off = s * n;
            for (int c = 0; c < n; c++) {
                int idx = ni[c];
                if (pdeFlag[c]) {
                    out[off + c] = z[off + c] - yOldC[off + c] - dt * (th * kb[idx] + (1.0 - th) * fo[idx]);
                } else {
                    out[off + c] = z[off + c] - wb[idx];
                }
            }
        }
    }

    private void jacSetPoint(double[] z) {
        unpack(z, sBuf);
        bc.fillHalos(sBuf);
        m.linearize(implicitT, sBuf);
    }

    /** Diagonal of the step Jacobian {@code I - theta dt F'}: boundary rows are identity rows. */
    private boolean jacDiagonal(double[] d) {
        int n = grid.nodeCount();
        int[] ni = m.nodeIdx;
        double c = implicitTheta * implicitDt;
        for (int s = 0; s < ns; s++) {
            // the equation whose row this state variable carries (value of a 1st-order unknown, velocity of a 2nd-order one)
            int eq = -1;
            for (int e = 0; e < m.nc; e++) {
                int pc = m.eqs[e].principalComp;
                if ((m.timeOrder[pc] == 2 ? m.velVar[pc] : m.valVar[pc]) == s) {
                    eq = e;
                }
            }
            int[] js = eq >= 0 ? m.jSym[eq] : new int[0];
            for (int q0 = 0; q0 < n; q0++) {
                int idx = ni[q0];
                if (!pdeFlag[q0]) {
                    d[s * n + q0] = 1.0;
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
                        acc += m.jac[eq][q][idx] * w;
                    }
                }
                d[s * n + q0] = 1.0 - c * acc;
            }
        }
        return true;
    }

    /** Exact J*v of the step residual using the stored AD partials. */
    private void jacMatvec(double[] v, double[] out) {
        int n = grid.nodeCount();
        int[] ni = m.nodeIdx;
        unpack(v, vBuf);
        for (int s = 0; s < ns; s++) {
            System.arraycopy(vBuf[s], 0, wBuf[s], 0, size);
        }
        bc.applyConstraintsHomogeneous(wBuf);
        bc.fillHalos(vBuf);
        m.jacApply(vBuf, jvBuf);
        double c = implicitTheta * implicitDt;
        for (int s = 0; s < ns; s++) {
            double[] jv = jvBuf[s];
            double[] wb = wBuf[s];
            int off = s * n;
            for (int q = 0; q < n; q++) {
                int idx = ni[q];
                if (pdeFlag[q]) {
                    out[off + q] = v[off + q] - c * jv[idx];
                } else {
                    out[off + q] = v[off + q] - wb[idx];
                }
            }
        }
    }

    private void implicitStep(double dt, int depth) {
        double theta = integ == TimeIntegrator.BACKWARD_EULER ? 1.0 : 0.5;
        // save old state and old F
        for (int s = 0; s < ns; s++) {
            System.arraycopy(y[s], 0, yOld[s], 0, size);
        }
        pack(y, yOldC);
        System.arraycopy(yOldC, 0, zVec, 0, zVec.length);
        if (theta < 1.0) {
            bc.apply(y, t);
            m.rhs(t, y, fOld);
        }
        implicitDt = dt;
        implicitTheta = theta;
        implicitT = t + dt;
        // better initial guess: explicit predictor along F (zero for BE)
        long n0 = nk.newtonIterations;
        long k0 = nk.krylovIterations;
        boolean ok = nk.solve(zVec);
        diag.newtonIterations += nk.newtonIterations - n0;
        diag.krylovIterations += nk.krylovIterations - k0;
        diag.residualNorm = nk.lastResidualNorm;
        if (!ok) {
            if (depth < 4) {
                for (int s = 0; s < ns; s++) {
                    System.arraycopy(yOld[s], 0, y[s], 0, size);
                }
                // two half steps
                implicitStep(0.5 * dt, depth + 1);
                t += 0.5 * dt;
                implicitStep(0.5 * dt, depth + 1);
                t -= 0.5 * dt;
                return;
            }
            diag.converged = false;
            throw new NonConvergenceException("Implicit step failed to converge at t=" + t + " with dt=" + dt
                    + " (Newton residual " + nk.lastResidualNorm + "). Reduce dt or check the problem for singularities", diag);
        }
        unpack(zVec, y);
        bc.apply(y, t + dt);
    }

    // ------------------------------------------------------------ misc
    /** Re-synchronises the integrator after the caller changed the state array directly. */
    void stateChanged() {
        invalidateCaches();
        bc.apply(y, t);
    }
}
