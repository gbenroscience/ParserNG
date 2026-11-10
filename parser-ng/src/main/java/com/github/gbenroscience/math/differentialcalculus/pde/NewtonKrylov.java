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
 * Matrix-free Newton iteration with a BiCGStab linear solver. The Jacobian is
 * never formed: {@code J*v} is a forward difference of the residual. All
 * work vectors are allocated once in the constructor, so {@link #solve} does
 * not allocate.
 */
final class NewtonKrylov {

    /** Nonlinear residual {@code out = R(z)}. */
    interface Residual {

        void eval(double[] z, double[] out);
    }

    /** Optional exact Jacobian: {@code setPoint} linearises, {@code apply} multiplies. */
    interface Jacobian {

        void setPoint(double[] z);

        void apply(double[] v, double[] out);

        /**
         * Fills {@code d} with the diagonal of the Jacobian at the current point.
         *
         * @return false if the diagonal is not available (no preconditioning is then used)
         */
        boolean diagonal(double[] d);
    }

    private final int n;
    private final Residual f;
    private final Jacobian jacobian;
    private final PDEOptions opt;
    private final double[] r, dz, trial, rTrial, fz;
    // BiCGStab workspace
    private final double[] br, br0, bp, bv, bs, bt, bx, bPert, bOut, phat, shat, diagInv;
    private boolean usePrecond;
    /** Total BiCGStab iterations are in {@link #krylovIterations}; set to false to disable the Jacobi preconditioner. */
    boolean preconditioning = true;

    long newtonIterations;
    long krylovIterations;
    double lastResidualNorm;
    double initialResidualNorm;
    boolean linearFailure;

    NewtonKrylov(int n, Residual f, Jacobian jacobian, PDEOptions opt) {
        this.n = n;
        this.f = f;
        this.jacobian = jacobian;
        this.opt = opt;
        r = new double[n];
        dz = new double[n];
        trial = new double[n];
        rTrial = new double[n];
        fz = new double[n];
        br = new double[n];
        br0 = new double[n];
        bp = new double[n];
        bv = new double[n];
        bs = new double[n];
        bt = new double[n];
        bx = new double[n];
        bPert = new double[n];
        bOut = new double[n];
        phat = new double[n];
        shat = new double[n];
        diagInv = new double[n];
    }

    private static double norm2(double[] a, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += a[i] * a[i];
        }
        return Math.sqrt(s);
    }

    private static double dot(double[] a, double[] b, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /** @return true if converged. {@code z} is the initial guess and receives the solution. */
    boolean solve(double[] z) {
        linearFailure = false;
        f.eval(z, r);
        double rn = norm2(r, n);
        initialResidualNorm = rn;
        lastResidualNorm = rn;
        double target = opt.newtonTolerance * (1.0 + norm2(z, n));
        for (int it = 0; it < opt.newtonMaxIterations; it++) {
            // the target scales with the CURRENT iterate: a cold start from z = 0 must not collapse it to a bare tolerance
            target = opt.newtonTolerance * (1.0 + norm2(z, n));
            if (rn <= target) {
                return true;
            }
            newtonIterations++;
            // linearisation point: exact AD Jacobian if available, else finite differences about z
            usePrecond = false;
            if (jacobian != null) {
                jacobian.setPoint(z);
                if (preconditioning && jacobian.diagonal(diagInv)) {
                    usePrecond = true;
                    for (int i = 0; i < n; i++) {
                        double dd = diagInv[i];
                        diagInv[i] = (dd == 0.0 || Double.isNaN(dd) || Double.isInfinite(dd)) ? 1.0 : 1.0 / dd;
                    }
                }
            }
            System.arraycopy(r, 0, fz, 0, n);
            double zNorm = norm2(z, n);
            // solve J dz = -r
            for (int i = 0; i < n; i++) {
                bx[i] = 0;
                br[i] = -r[i];
            }
            bicgstab(z, zNorm, rn);
            for (int i = 0; i < n; i++) {
                dz[i] = bx[i];
            }
            // backtracking line search
            double alpha = 1.0;
            boolean accepted = false;
            double rnNew = rn;
            for (int ls = 0; ls < 12; ls++) {
                for (int i = 0; i < n; i++) {
                    trial[i] = z[i] + alpha * dz[i];
                }
                f.eval(trial, rTrial);
                rnNew = norm2(rTrial, n);
                if (rnNew <= (1.0 - 1e-4 * alpha) * rn || rnNew <= target) {
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if (!accepted) {
                lastResidualNorm = rn;
                return rn <= opt.newtonTolerance * (1.0 + norm2(z, n));
            }
            System.arraycopy(trial, 0, z, 0, n);
            System.arraycopy(rTrial, 0, r, 0, n);
            rn = rnNew;
            lastResidualNorm = rn;
        }
        return rn <= opt.newtonTolerance * (1.0 + norm2(z, n));
    }

    /** J*v by forward difference about the current linearisation point. */
    private void jv(double[] z, double zNorm, double[] v, double[] out) {
        if (jacobian != null) {
            jacobian.apply(v, out);
            return;
        }
        double vn = norm2(v, n);
        if (vn == 0) {
            for (int i = 0; i < n; i++) {
                out[i] = 0;
            }
            return;
        }
        double eps = 1.4901161193847656e-8 * (1.0 + zNorm) / vn;
        for (int i = 0; i < n; i++) {
            bPert[i] = z[i] + eps * v[i];
        }
        f.eval(bPert, bOut);
        double inv = 1.0 / eps;
        for (int i = 0; i < n; i++) {
            out[i] = (bOut[i] - fz[i]) * inv;
        }
    }

    /** Right-preconditioned BiCGStab for J x = b with x0 = 0 (b in br, result in bx); preconditioner = Jacobi if available. */
    private void bicgstab(double[] z, double zNorm, double rn) {
        double bnorm = norm2(br, n);
        if (bnorm == 0) {
            return;
        }
        double tol = Math.max(opt.krylovTolerance, 1e-14) * bnorm;
        System.arraycopy(br, 0, br0, 0, n);
        double rho = 1, alpha = 1, omega = 1;
        for (int i = 0; i < n; i++) {
            bv[i] = 0;
            bp[i] = 0;
        }
        final boolean pc = usePrecond;
        final double[] di = diagInv;
        for (int it = 0; it < opt.krylovMaxIterations; it++) {
            krylovIterations++;
            double rhoNew = dot(br0, br, n);
            if (Math.abs(rhoNew) < 1e-300) {
                linearFailure = true;
                return;
            }
            double beta = (rhoNew / rho) * (alpha / omega);
            for (int i = 0; i < n; i++) {
                bp[i] = br[i] + beta * (bp[i] - omega * bv[i]);
            }
            double[] pUse = bp;
            if (pc) {
                for (int i = 0; i < n; i++) {
                    phat[i] = bp[i] * di[i];
                }
                pUse = phat;
            }
            jv(z, zNorm, pUse, bv);
            double den = dot(br0, bv, n);
            if (Math.abs(den) < 1e-300) {
                linearFailure = true;
                return;
            }
            alpha = rhoNew / den;
            for (int i = 0; i < n; i++) {
                bs[i] = br[i] - alpha * bv[i];
            }
            if (norm2(bs, n) <= tol) {
                for (int i = 0; i < n; i++) {
                    bx[i] += alpha * pUse[i];
                }
                return;
            }
            double[] sUse = bs;
            if (pc) {
                for (int i = 0; i < n; i++) {
                    shat[i] = bs[i] * di[i];
                }
                sUse = shat;
            }
            jv(z, zNorm, sUse, bt);
            double tt = dot(bt, bt, n);
            if (tt < 1e-300) {
                linearFailure = true;
                return;
            }
            omega = dot(bt, bs, n) / tt;
            for (int i = 0; i < n; i++) {
                bx[i] += alpha * pUse[i] + omega * sUse[i];
                br[i] = bs[i] - omega * bt[i];
            }
            rho = rhoNew;
            if (norm2(br, n) <= tol) {
                return;
            }
            if (Math.abs(omega) < 1e-300) {
                linearFailure = true;
                return;
            }
        }
    }
}
