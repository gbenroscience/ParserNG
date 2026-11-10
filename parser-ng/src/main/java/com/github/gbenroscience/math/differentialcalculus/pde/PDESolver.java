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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A configured, reusable PDE solver. Everything expensive happens in the
 * constructor and in {@link #initialize()}: expressions are parsed and
 * compiled once, work buffers are allocated once. After that,
 * {@link #step(double)} reuses those buffers and performs no allocation.
 * <p>
 * Real-time use:
 * <pre>{@code
 * PDESolver s = PDE.problem("diff(u,t) = 0.1*diff(u,x,2)")...solver();
 * s.initialize();
 * while (running) {
 *     s.step(dt);
 *     double[] u = s.state("u");          // live storage, no copy
 *     int k = s.grid().index(i, 0, 0);    // flat index of node i
 * }
 * s.close();
 * }</pre>
 * The array returned by {@link #state(String)} is the solver's own storage
 * (halo-padded, see {@link Grid}); read it, do not keep it across a call to
 * {@link #solveSteady()}. A solver is NOT thread-safe: drive it from one
 * thread (it uses its own worker threads internally if configured).
 */
public final class PDESolver implements AutoCloseable {

    private final PDEProblem problem;
    private final PDEOptions options;
    private final Model model;
    private final Boundaries bounds;
    private final SolverDiagnostics diag = new SolverDiagnostics();
    private final double[][] y;
    private final CompiledExpr[] icVal;
    private final CompiledExpr[] icVel;
    private EvolutionEngine evo;
    private EllipticEngine ell;
    private boolean initialized;
    private boolean closed;
    private double time;

    PDESolver(PDEProblem problem, PDEOptions options) {
        this.problem = problem;
        this.options = options == null ? new PDEOptions() : options;
        this.model = new Model(problem, this.options);
        this.bounds = new Boundaries(model);
        this.y = new double[model.nState][model.grid.storageSize()];
        this.icVal = new CompiledExpr[model.nc];
        this.icVel = new CompiledExpr[model.nc];
        compileInitialConditions();
        if (model.timeDependent) {
            evo = new EvolutionEngine(model, bounds, y, diag);
        } else {
            ell = new EllipticEngine(model, bounds, y, diag);
        }
        this.time = this.options.t0;
    }

    // ------------------------------------------------------------ introspection
    public PDEType type() {
        return model.type;
    }

    public Grid grid() {
        return model.grid;
    }

    /** @return names of the unknowns, in the order used by {@link PDEResult}. */
    public List<String> unknowns() {
        return Collections.unmodifiableList(Arrays.asList(model.comps));
    }

    public boolean isTimeDependent() {
        return model.timeDependent;
    }

    /** @return the integrator in use after {@code AUTO} has been resolved (null for steady problems). */
    public TimeIntegrator integrator() {
        return evo == null ? null : evo.integ;
    }

    public SolverDiagnostics diagnostics() {
        return diag;
    }

    /** @return the current time. */
    public double time() {
        return time;
    }

    /** @return number of worker threads in use. */
    public int threads() {
        return model.threads;
    }

    // ------------------------------------------------------------ conditions
    private void compileInitialConditions() {
        for (int c = 0; c < model.nc; c++) {
            String u = model.comps[c];
            String e = problem.getInitialConditions().get(u);
            if (e != null) {
                icVal[c] = compileIc(e, "initial condition of '" + u + "'");
            } else if (model.timeDependent) {
                throw new PDEConfigurationException("Missing initial condition for '" + u + "': call initial(\"" + u + "\", expr)");
            }
            String v = problem.getInitialVelocities().get(u);
            if (v != null) {
                if (model.timeOrder[c] != 2) {
                    throw new PDEConfigurationException("initialVelocity given for '" + u + "' which is not second order in time");
                }
                icVel[c] = compileIc(v, "initial velocity of '" + u + "'");
            }
        }
        for (String k : problem.getInitialConditions().keySet()) {
            if (Model.indexOf(model.comps, k) < 0) {
                throw new PDEConfigurationException("Initial condition given for '" + k + "' which is not an unknown " + Arrays.toString(model.comps));
            }
        }
    }

    private CompiledExpr compileIc(String text, String what) {
        CompiledExpr ex = new CompiledExpr(text);
        for (int i = 0; i < ex.names.length; i++) {
            String nm = ex.names[i];
            boolean known = false;
            for (int a = 0; a < model.dim; a++) {
                known |= model.axisNames[a].equals(nm);
            }
            Double pv = problem.getParameters().get(nm);
            if (pv != null) {
                ex.template[ex.slots[i]] = pv;
                known = true;
            }
            if (!known && !(nm.equals("pi") || nm.equals("PI") || nm.equals("\u03c0") || nm.equals("e") || nm.equals("\u03c6"))) {
                throw new PDEConfigurationException("Unknown symbol '" + nm + "' in " + what + " '" + text
                        + "'. Initial data may use the domain variables and declared parameters");
            }
        }
        return ex;
    }

    private void fillFromExpr(CompiledExpr ex, double[] dst) {
        Grid g = model.grid;
        double[] fr = ex.newFrame();
        int[] slot = {-1, -1, -1};
        for (int a = 0; a < model.dim; a++) {
            slot[a] = ex.slotOf(model.axisNames[a]);
        }
        for (int k = 0; k < g.n(2); k++) {
            if (slot[2] >= 0) {
                fr[slot[2]] = g.coord(2, k);
            }
            for (int j = 0; j < g.n(1); j++) {
                if (slot[1] >= 0) {
                    fr[slot[1]] = g.coord(1, j);
                }
                int base = g.index(0, j, k);
                for (int i = 0; i < g.n(0); i++) {
                    if (slot[0] >= 0) {
                        fr[slot[0]] = g.coord(0, i);
                    }
                    double v = ex.eval(fr);
                    if (Double.isNaN(v) || Double.isInfinite(v)) {
                        throw new PDEConfigurationException("Initial data '" + ex.text + "' is not finite at node ("
                                + i + "," + j + "," + k + ")");
                    }
                    dst[base + i] = v;
                }
            }
        }
    }

    // ------------------------------------------------------------ lifecycle
    /**
     * (Re)loads the initial conditions and resets time to {@code t0}. Must be
     * called before the first {@link #step}; may be called again to restart.
     */
    public PDESolver initialize() {
        ensureOpen();
        for (double[] a : y) {
            Arrays.fill(a, 0.0);
        }
        for (int c = 0; c < model.nc; c++) {
            if (icVal[c] != null) {
                fillFromExpr(icVal[c], y[model.valVar[c]]);
            }
            if (model.velVar[c] >= 0 && icVel[c] != null) {
                fillFromExpr(icVel[c], y[model.velVar[c]]);
            }
        }
        time = options.t0;
        diag.warnings.clear();
        diag.converged = true;
        // initial / boundary compatibility check
        double[][] before = new double[model.nState][];
        for (int s = 0; s < model.nState; s++) {
            before[s] = y[s].clone();
        }
        bounds.applyConstraints(y, time);
        double worst = 0;
        int ws = 0;
        Grid g = model.grid;
        for (int s = 0; s < model.nState; s++) {
            for (int k = 0; k < g.n(2); k++) {
                for (int j = 0; j < g.n(1); j++) {
                    int base = g.index(0, j, k);
                    for (int i = 0; i < g.n(0); i++) {
                        double d = Math.abs(y[s][base + i] - before[s][base + i]);
                        if (d > worst) {
                            worst = d;
                            ws = s;
                        }
                    }
                }
            }
        }
        boolean userData = false;
        for (int c = 0; c < model.nc; c++) {
            userData |= icVal[c] != null || icVel[c] != null;
        }
        if (userData && worst > 1e-8 * (1.0 + maxAbs(before[ws]))) {
            diag.warnings.add("Initial data for '" + model.stateNames[ws] + "' does not satisfy the boundary condition (max mismatch "
                    + worst + "); the boundary condition takes precedence on boundary nodes");
        }
        if (evo != null) {
            evo.initialize(time);
        } else {
            bounds.apply(y, 0.0);
        }
        initialized = true;
        return this;
    }

    private static double maxAbs(double[] a) {
        double m = 0;
        for (double v : a) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }

    private void ensureOpen() {
        if (closed) {
            throw new PDEException("Solver is closed");
        }
    }

    private void ensureTimeDependent() {
        ensureOpen();
        if (!model.timeDependent) {
            throw new PDEException("This is a steady-state problem: use solveSteady()");
        }
        if (!initialized) {
            throw new PDEException("Call initialize() before stepping");
        }
    }

    // ------------------------------------------------------------ stepping
    /** Advances the solution by exactly {@code dt}. Allocation-free after {@link #initialize()}. */
    public void step(double dt) {
        ensureTimeDependent();
        evo.step(dt);
        time = evo.t;
    }

    /**
     * Advances to {@code tEnd} using the configured time step (or a CFL-based
     * default, or one adaptive interval for {@link TimeIntegrator#DORMAND_PRINCE}).
     *
     * @return the time reached
     */
    public double advanceTo(double tEnd) {
        ensureTimeDependent();
        if (tEnd < time) {
            throw new PDEConfigurationException("tEnd=" + tEnd + " is before the current time " + time);
        }
        double dt = options.dt;
        if (Double.isNaN(dt)) {
            dt = evo.integ.isAdaptive() ? Math.max(tEnd - time, 1e-300) : evo.suggestTimeStep();
        }
        if (!(dt > 0)) {
            throw new PDEConfigurationException("Time step must be positive, got " + dt);
        }
        double eps = 1e-12 * Math.max(1.0, Math.abs(tEnd));
        while (time < tEnd - eps) {
            double h = Math.min(dt, tEnd - time);
            if (tEnd - time - h < 1e-9 * dt) {
                h = tEnd - time;
            }
            step(h);
        }
        return time;
    }

    /** Solves the steady problem starting from the initial data (or zero) as the initial guess. */
    public PDEResult solveSteady() {
        ensureOpen();
        if (model.timeDependent) {
            throw new PDEException("This problem is time dependent; use step()/advanceTo()");
        }
        if (!initialized) {
            initialize();
        }
        boolean ok = ell.solve();
        if (!ok && options.failOnNonConvergence) {
            throw new NonConvergenceException("Elliptic solve did not converge: " + diag, diag);
        }
        time = 0.0;
        return result();
    }

    // ------------------------------------------------------------ data access
    /**
     * @return the LIVE halo-padded storage of an unknown (zero copy). Index
     * with {@link Grid#index(int, int, int)}.
     */
    public double[] state(String unknown) {
        int c = Model.indexOf(model.comps, unknown);
        if (c < 0) {
            throw new PDEException("Unknown field '" + unknown + "'; available: " + Arrays.toString(model.comps));
        }
        return y[model.valVar[c]];
    }

    /** @return live storage of the velocity {@code u_t} of a second-order-in-time unknown. */
    public double[] velocity(String unknown) {
        int c = Model.indexOf(model.comps, unknown);
        if (c < 0 || model.velVar[c] < 0) {
            throw new PDEException("'" + unknown + "' has no velocity field (it is not second order in time)");
        }
        return y[model.velVar[c]];
    }

    /** Copies the real grid nodes of an unknown, x-fastest, into {@code out} (length &gt;= node count). */
    public void copyState(String unknown, double[] out) {
        double[] a = state(unknown);
        Grid g = model.grid;
        if (out.length < g.nodeCount()) {
            throw new PDEException("out is too small: need " + g.nodeCount());
        }
        int p = 0;
        for (int k = 0; k < g.n(2); k++) {
            for (int j = 0; j < g.n(1); j++) {
                int base = g.index(0, j, k);
                System.arraycopy(a, base, out, p, g.n(0));
                p += g.n(0);
            }
        }
    }

    /** @return an independent snapshot of the solution with diagnostics. */
    public PDEResult result() {
        Grid g = model.grid;
        double[][] f = new double[model.nc][g.nodeCount()];
        for (int c = 0; c < model.nc; c++) {
            copyState(model.comps[c], f[c]);
        }
        SolverDiagnostics d = snapshotDiagnostics();
        return new PDEResult(g, model.comps, f, time, d);
    }

    private SolverDiagnostics snapshotDiagnostics() {
        SolverDiagnostics d = new SolverDiagnostics();
        d.method = diag.method;
        d.type = diag.type;
        d.converged = diag.converged;
        d.steps = diag.steps;
        d.rejectedSteps = diag.rejectedSteps;
        d.iterations = diag.iterations;
        d.newtonIterations = diag.newtonIterations;
        d.krylovIterations = diag.krylovIterations;
        d.residualNorm = diag.residualNorm;
        d.initialResidualNorm = diag.initialResidualNorm;
        d.updateNorm = diag.updateNorm;
        d.errorEstimate = diag.errorEstimate;
        d.contractionFactor = diag.contractionFactor;
        d.spectralRadius = diag.spectralRadius;
        d.maxStableDt = diag.maxStableDt;
        d.lastDt = diag.lastDt;
        d.minDt = diag.minDt;
        d.maxDt = diag.maxDt;
        d.finalTime = time;
        d.omega = diag.omega;
        d.spatialOrder = diag.spatialOrder;
        d.threads = diag.threads;
        d.wallNanos = diag.wallNanos;
        d.warnings.addAll(diag.warnings);
        return d;
    }

    // ------------------------------------------------------------ stability helpers
    /** @return a CFL-type time step suggestion (80% of the RK4 stability limit). */
    public double suggestedTimeStep() {
        ensureTimeDependent();
        return evo.suggestTimeStep();
    }

    /** @return the estimated largest stable explicit step for the selected integrator (infinite for implicit ones). */
    public double maxStableTimeStep() {
        ensureTimeDependent();
        return evo.integ.isExplicit() ? evo.maxStableDt() : Double.POSITIVE_INFINITY;
    }

    /** @return the estimated spectral radius of the semi-discrete Jacobian. */
    public double spectralRadius() {
        ensureTimeDependent();
        return evo.spectralRadius();
    }

    // ------------------------------------------------------------ ODE interoperability
    /**
     * Exposes the semi-discrete system {@code y' = F(t,y)} in the calling
     * convention of ParserNG's existing ODE integrators, so that they (for
     * example the adaptive RK45 of {@code DifferentialEquations}) can drive
     * the PDE directly.
     */
    public MethodOfLinesSystem toOdeSystem() {
        ensureTimeDependent();
        return new MethodOfLinesSystem(model, bounds, y);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            model.close();
        }
    }
}
