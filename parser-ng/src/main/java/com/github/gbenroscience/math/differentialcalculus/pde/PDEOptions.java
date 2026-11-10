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
 * Numerical configuration of a solve. Every field has a documented default,
 * so {@code new PDEOptions()} is a valid configuration. All setters return
 * {@code this}.
 */
public final class PDEOptions {

    TimeIntegrator integrator = TimeIntegrator.AUTO;
    EllipticMethod ellipticMethod = EllipticMethod.AUTO;
    int spatialOrder = 2;
    DerivativeScheme firstDerivativeScheme = DerivativeScheme.CENTRAL;
    double dt = Double.NaN;
    double t0 = 0.0;
    StabilityPolicy stabilityPolicy = StabilityPolicy.THROW;
    int stabilityRecheckSteps = 500;
    double blowupFactor = 1e8;
    int blowupCheckSteps = 10;
    double rtol = 1e-6;
    double atol = 1e-9;
    double dtMin = 1e-14;
    double dtMax = Double.POSITIVE_INFINITY;
    double tolerance = 1e-10;
    double absoluteTolerance = 1e-12;
    int maxIterations = 200000;
    double omega = Double.NaN;
    boolean failOnNonConvergence = false;
    int newtonMaxIterations = 25;
    double newtonTolerance = 1e-10;
    int krylovMaxIterations = 2000;
    double krylovTolerance = 1e-8;
    boolean jacobiPreconditioner = true;
    int threads = 0;                      // 0 = automatic
    int parallelThresholdNodes = 20000;   // below this, run single-threaded under AUTO

    /** Default configuration. */
    public PDEOptions() {
    }

    /** @return a new default configuration. */
    public static PDEOptions defaults() {
        return new PDEOptions();
    }

    /** Time integrator (default {@link TimeIntegrator#AUTO}). */
    public PDEOptions integrator(TimeIntegrator v) {
        this.integrator = v == null ? TimeIntegrator.AUTO : v;
        return this;
    }

    /** Time integrator by name, e.g. {@code "rk4"} (as in ParserNG's {@code diffeqn(..., rk4)}). */
    public PDEOptions integrator(String name) {
        this.integrator = TimeIntegrator.parse(name);
        return this;
    }

    /** Elliptic iteration (default {@link EllipticMethod#AUTO}). */
    public PDEOptions ellipticMethod(EllipticMethod v) {
        this.ellipticMethod = v == null ? EllipticMethod.AUTO : v;
        return this;
    }

    /** Elliptic iteration by name. */
    public PDEOptions ellipticMethod(String name) {
        this.ellipticMethod = EllipticMethod.parse(name);
        return this;
    }

    /** Formal spatial order of accuracy: 2 (default), 4 or 6 for central stencils; 1..6 for one-sided schemes. */
    public PDEOptions spatialOrder(int v) {
        if (v < 1 || v > 6) {
            throw new PDEConfigurationException("spatialOrder must be in 1..6, got " + v);
        }
        this.spatialOrder = v;
        return this;
    }

    /** Scheme for first spatial derivatives (default CENTRAL). Use BACKWARD/FORWARD for upwinding. */
    public PDEOptions firstDerivativeScheme(DerivativeScheme v) {
        this.firstDerivativeScheme = v == null ? DerivativeScheme.CENTRAL : v;
        return this;
    }

    /** Time step. If unset, a step based on a CFL-type estimate is chosen. */
    public PDEOptions dt(double v) {
        this.dt = v;
        return this;
    }

    /** Initial time (default 0). */
    public PDEOptions t0(double v) {
        this.t0 = v;
        return this;
    }

    /** Behaviour when an explicit step exceeds the estimated stability limit (default THROW). */
    public PDEOptions stabilityPolicy(StabilityPolicy v) {
        this.stabilityPolicy = v == null ? StabilityPolicy.THROW : v;
        return this;
    }

    /** Re-estimate the stability limit every this many steps (0 = only once). Default 500. */
    public PDEOptions stabilityRecheckSteps(int v) {
        this.stabilityRecheckSteps = Math.max(0, v);
        return this;
    }

    /** Divergence guard: fail if max|u| exceeds this factor times its initial scale. Default 1e8. */
    public PDEOptions blowupFactor(double v) {
        this.blowupFactor = v;
        return this;
    }

    /** Check for non-finite / runaway values every this many steps (default 10; 0 disables the scan). */
    public PDEOptions blowupCheckSteps(int v) {
        this.blowupCheckSteps = Math.max(0, v);
        return this;
    }

    /** Relative tolerance of the adaptive integrator (default 1e-6). */
    public PDEOptions rtol(double v) {
        this.rtol = v;
        return this;
    }

    /** Absolute tolerance of the adaptive integrator (default 1e-9). */
    public PDEOptions atol(double v) {
        this.atol = v;
        return this;
    }

    /** Smallest step the adaptive integrator may take (default 1e-14). */
    public PDEOptions dtMin(double v) {
        this.dtMin = v;
        return this;
    }

    /** Largest step the adaptive integrator may take (default unlimited). */
    public PDEOptions dtMax(double v) {
        this.dtMax = v;
        return this;
    }

    /** Relative residual reduction required by elliptic iterations (default 1e-10). */
    public PDEOptions tolerance(double v) {
        this.tolerance = v;
        return this;
    }

    /** Absolute residual (max-norm) below which an elliptic iteration stops (default 1e-12). */
    public PDEOptions absoluteTolerance(double v) {
        this.absoluteTolerance = v;
        return this;
    }

    /** Iteration cap of elliptic iterations (default 200000). */
    public PDEOptions maxIterations(int v) {
        this.maxIterations = v;
        return this;
    }

    /** Relaxation factor for SOR (default: estimated from the grid). */
    public PDEOptions omega(double v) {
        this.omega = v;
        return this;
    }

    /** If true, an elliptic solve that does not converge throws {@link NonConvergenceException}. Default false (result flags it). */
    public PDEOptions failOnNonConvergence(boolean v) {
        this.failOnNonConvergence = v;
        return this;
    }

    /** Newton iteration cap (default 25). */
    public PDEOptions newtonMaxIterations(int v) {
        this.newtonMaxIterations = v;
        return this;
    }

    /** Relative nonlinear residual tolerance of Newton iterations (default 1e-10). */
    public PDEOptions newtonTolerance(double v) {
        this.newtonTolerance = v;
        return this;
    }

    /** BiCGStab iteration cap per linear solve (default 2000). */
    public PDEOptions krylovMaxIterations(int v) {
        this.krylovMaxIterations = v;
        return this;
    }

    /** Relative tolerance of the linear solves (default 1e-8). */
    public PDEOptions krylovTolerance(double v) {
        this.krylovTolerance = v;
        return this;
    }

    /**
     * Jacobi (diagonal) preconditioning of the BiCGStab solves, using the exact diagonal of the automatic-differentiation
     * Jacobian (default true; ignored when the Jacobian is finite-difference). It equilibrates rows whose scales differ
     * by orders of magnitude (PDE rows ~ 1/h^2 versus boundary rows ~ 1).
     */
    public PDEOptions jacobiPreconditioner(boolean v) {
        this.jacobiPreconditioner = v;
        return this;
    }

    /** Worker threads: 1 = serial, n &gt; 1 = exactly n, 0 = automatic (default). */
    public PDEOptions threads(int v) {
        this.threads = Math.max(0, v);
        return this;
    }

    /** Under automatic threading, problems with fewer grid nodes than this run serially (default 20000). */
    public PDEOptions parallelThresholdNodes(int v) {
        this.parallelThresholdNodes = Math.max(0, v);
        return this;
    }
}
