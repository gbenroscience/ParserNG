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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Incompressible Navier-Stokes solver for 2-D and 3-D rectangular domains:
 * <pre>
 *   u_t + (u.grad)u = -grad(p)/rho + nu laplace(u) + f,      div(u) = 0
 * </pre>
 * <b>Method.</b> Staggered (MAC) grid: each velocity component lives on the
 * faces normal to it, the pressure at cell centres, so the pressure-velocity
 * coupling is tight and there are no checkerboard modes. Time stepping is the
 * incremental pressure-correction (projection) method:
 * <ol>
 * <li>predictor {@code u* = u^n + dt [AB2(-convection + diffusion + f) - grad p^n]}
 * (second-order Adams-Bashforth, forward Euler on the first step);</li>
 * <li>pressure Poisson equation {@code laplace(phi) = div(u*)/dt} solved by
 * {@link PoissonMultigrid};</li>
 * <li>correction {@code u^{n+1} = u* - dt grad(phi)}, {@code p^{n+1} = p^n + phi}.</li>
 * </ol>
 * The discrete divergence of the corrected velocity is zero up to the Poisson
 * tolerance (the gradient and divergence operators are exact adjoints on the
 * staggered grid). Convection is conservative and second-order central, with an
 * optional donor-cell blend ({@link Builder#upwindBlend}) for high cell Reynolds numbers.
 * <p>
 * <b>Cost.</b> Per step: {@code O(N)} for the momentum update plus a handful of
 * multigrid cycles, {@code O(N)} each. Memory: about {@code (3*(d) + 3)} velocity-sized
 * and the multigrid hierarchy ({@code ~4/3 (d=2)} or {@code 8/7 (d=3)} of the finest
 * level) arrays, all allocated at construction; {@link #step(double)} allocates nothing.
 * Execution is single-threaded in this release (the kernels are row-partitionable).
 * <p>
 * <b>Limitations.</b> Uniform Cartesian grid; the explicit diffusion and AB2
 * convection impose {@code dt <~ min(h/|u|, h^2/(4 nu d))}
 * ({@link #suggestTimeStep()}); the incremental scheme is formally second order in
 * velocity but only first order in the pressure near boundaries; no turbulence
 * model; no obstacles inside the domain.
 */
public final class IncompressibleFlow {

    /** Fluent configuration. */
    public static final class Builder {

        private int dim = 2;
        private final int[] n = {32, 32, 1};
        private final double[] size = {1, 1, 1};
        private final double[] origin = {0, 0, 0};
        private double nu = 0.01;
        private double rho = 1.0;
        private double refU = 1.0;
        private double refL = 1.0;
        private double gamma = 0.0;
        private double cfl = 0.5;
        private final double[] body = {0, 0, 0};
        private final FlowBoundary[][] bnd = new FlowBoundary[3][2];
        private final String[] ic = new String[3];
        private final Map<String, Double> params = new LinkedHashMap<String, Double>();
        private double poissonRtol = 1e-10;
        private double poissonAtol = 1e-12;
        private int poissonMaxCycles = 40;
        private PoissonMultigrid.Smoother smoother = PoissonMultigrid.Smoother.RED_BLACK_GS;
        private PoissonMultigrid.Cycle cycle = PoissonMultigrid.Cycle.V;

        /** Number of cells per direction: 2 values for a 2-D flow, 3 for 3-D. */
        public Builder cells(int... cells) {
            if (cells.length != 2 && cells.length != 3) {
                throw new PDEConfigurationException("cells(...) needs 2 or 3 values");
            }
            dim = cells.length;
            for (int a = 0; a < dim; a++) {
                if (cells[a] < 4) {
                    throw new PDEConfigurationException("Need at least 4 cells per direction, got " + cells[a]);
                }
                n[a] = cells[a];
            }
            return this;
        }

        /** Domain length per direction (default 1 each). */
        public Builder size(double... lengths) {
            for (int a = 0; a < lengths.length && a < 3; a++) {
                size[a] = lengths[a];
            }
            return this;
        }

        /** Coordinates of the lower corner (default 0,0,0). */
        public Builder origin(double... o) {
            for (int a = 0; a < o.length && a < 3; a++) {
                origin[a] = o[a];
            }
            return this;
        }

        /** Kinematic viscosity. */
        public Builder viscosity(double nu) {
            this.nu = nu;
            return this;
        }

        /** Sets the viscosity from a Reynolds number: {@code nu = U_ref L_ref / Re}. */
        public Builder reynolds(double re) {
            this.nu = refU * refL / re;
            return this;
        }

        /** Reference velocity used to define the Reynolds number (default 1). */
        public Builder referenceVelocity(double u) {
            this.refU = u;
            return this;
        }

        /** Reference length used to define the Reynolds number (default 1). */
        public Builder referenceLength(double l) {
            this.refL = l;
            return this;
        }

        /** Density; only scales {@link IncompressibleFlow#pressure(int, int, int)} to physical units. */
        public Builder density(double rho) {
            this.rho = rho;
            return this;
        }

        /** Body force per unit mass, one value per component. */
        public Builder bodyForce(double... f) {
            for (int a = 0; a < f.length && a < 3; a++) {
                body[a] = f[a];
            }
            return this;
        }

        /** Donor-cell blending of the convective flux: 0 = pure central (default), 1 = full upwind. */
        public Builder upwindBlend(double gamma) {
            this.gamma = Math.max(0.0, Math.min(1.0, gamma));
            return this;
        }

        /** Courant number used by {@link IncompressibleFlow#suggestTimeStep()} (default 0.5). */
        public Builder cfl(double cfl) {
            this.cfl = cfl;
            return this;
        }

        /** Condition on a face. Faces without one are no-slip walls. */
        public Builder boundary(Face face, FlowBoundary b) {
            bnd[face.axis()][face.isMax() ? 1 : 0] = b;
            return this;
        }

        /** Makes the axis ({@code "x"}, {@code "y"} or {@code "z"}) periodic. */
        public Builder periodic(String axis) {
            int a = "xyz".indexOf(axis.trim().toLowerCase());
            if (a < 0) {
                throw new PDEConfigurationException("Axis must be x, y or z, got '" + axis + "'");
            }
            bnd[a][0] = FlowBoundary.periodic();
            bnd[a][1] = FlowBoundary.periodic();
            return this;
        }

        /** Initial velocity: one expression per component of x, y, z (and declared parameters). */
        public Builder initialVelocity(String... expr) {
            for (int a = 0; a < expr.length && a < 3; a++) {
                ic[a] = expr[a];
            }
            return this;
        }

        /** Constant usable in velocity expressions. */
        public Builder param(String name, double value) {
            params.put(name, value);
            return this;
        }

        /** Pressure-Poisson stopping criteria: relative and absolute residual and the cycle cap. */
        public Builder poissonTolerance(double rel, double abs, int maxCycles) {
            this.poissonRtol = rel;
            this.poissonAtol = abs;
            this.poissonMaxCycles = maxCycles;
            return this;
        }

        /** Multigrid smoother and cycle for the pressure solve. */
        public Builder poissonSolver(PoissonMultigrid.Smoother s, PoissonMultigrid.Cycle c) {
            this.smoother = s;
            this.cycle = c;
            return this;
        }

        /** Validates and creates the solver. Call {@link IncompressibleFlow#initialize()} before stepping. */
        public IncompressibleFlow build() {
            return new IncompressibleFlow(this);
        }
    }

    /** @return a new configuration. */
    public static Builder builder() {
        return new Builder();
    }

    /** A velocity/boundary expression of x, y, z, t. */
    private static final class Expr {

        final CompiledExpr ex;
        final double[] frame;
        final int[] slot = {-1, -1, -1};
        int tSlot = -1;
        final boolean constant;
        double value;

        Expr(String text, Map<String, Double> params, String context) {
            this.ex = new CompiledExpr(text);
            this.frame = ex.newFrame();
            boolean dep = false;
            String[] axes = {"x", "y", "z"};
            for (int i = 0; i < ex.names.length; i++) {
                String nm = ex.names[i];
                boolean known = false;
                for (int a = 0; a < 3; a++) {
                    if (axes[a].equals(nm)) {
                        slot[a] = ex.slots[i];
                        dep = true;
                        known = true;
                    }
                }
                if (nm.equals("t")) {
                    tSlot = ex.slots[i];
                    dep = true;
                    known = true;
                }
                Double pv = params.get(nm);
                if (pv != null) {
                    ex.template[ex.slots[i]] = pv;
                    frame[ex.slots[i]] = pv;
                    known = true;
                }
                if (!known && !(nm.equals("pi") || nm.equals("PI") || nm.equals("e"))) {
                    throw new PDEConfigurationException("Unknown symbol '" + nm + "' in " + context + " '" + text
                            + "'. Use x, y, z, t and parameters declared with param(...)");
                }
            }
            this.constant = !dep;
            if (constant) {
                value = ex.eval(frame);
            }
        }

        double eval(double x, double y, double z, double t) {
            if (constant) {
                return value;
            }
            if (slot[0] >= 0) {
                frame[slot[0]] = x;
            }
            if (slot[1] >= 0) {
                frame[slot[1]] = y;
            }
            if (slot[2] >= 0) {
                frame[slot[2]] = z;
            }
            if (tSlot >= 0) {
                frame[tSlot] = t;
            }
            return ex.eval(frame);
        }
    }

    // ------------------------------------------------------------------ state
    private final int dim;
    private final int[] n = new int[3];
    private final double[] h = new double[3];
    private final double[] invh = new double[3];
    private final double[] invh2 = new double[3];
    private final double[] len = new double[3];
    private final double[] origin = new double[3];
    private final double nu;
    private final double rho;
    private final double refU;
    private final double refL;
    private final double gamma;
    private final double cflTarget;
    private final double[] body = new double[3];
    private final FlowBoundary.Type[][] btype = new FlowBoundary.Type[3][2];
    private final boolean[] per = new boolean[3];
    private final Expr[][][] bexpr = new Expr[3][2][3];   // [axis][side][component]
    private final Expr[] ic = new Expr[3];
    private final Box[] ub = new Box[3];
    private final Box pb;
    private double[][] u;
    private double[][] us;
    private double[][] rCur;
    private double[][] rOld;
    private final double[] p;
    private final PoissonMultigrid mg;
    private final int[][] mLo = new int[3][3];   // momentum (interior) face ranges per component
    private final int[][] mHi = new int[3][3];
    private final int[][] cLo = new int[3][3];   // correction ranges (interior + outflow faces)
    private final int[][] cHi = new int[3][3];
    private final int[] m = new int[3];
    private final int[][] other = new int[3][2];
    private double time;
    private long steps;
    private double lastDt;
    private double dtOld;
    private boolean haveOld;
    private boolean initialized;
    private double velocityResidual;
    private double poissonResidual;
    private int poissonCycles;
    private boolean poissonConverged = true;
    private double lastPoissonMeanRhs;

    private IncompressibleFlow(Builder b) {
        this.dim = b.dim;
        for (int a = 0; a < 3; a++) {
            n[a] = a < dim ? b.n[a] : 1;
            len[a] = b.size[a];
            origin[a] = b.origin[a];
            h[a] = a < dim ? len[a] / n[a] : 1.0;
            invh[a] = 1.0 / h[a];
            invh2[a] = invh[a] * invh[a];
            body[a] = b.body[a];
        }
        this.nu = b.nu;
        this.rho = b.rho;
        this.refU = b.refU;
        this.refL = b.refL;
        this.gamma = b.gamma;
        this.cflTarget = b.cfl;
        if (!(nu > 0)) {
            throw new PDEConfigurationException("Viscosity must be positive, got " + nu);
        }
        for (int a = 0; a < dim; a++) {
            FlowBoundary lo = b.bnd[a][0] == null ? FlowBoundary.noSlip() : b.bnd[a][0];
            FlowBoundary hi = b.bnd[a][1] == null ? FlowBoundary.noSlip() : b.bnd[a][1];
            boolean pl = lo.getType() == FlowBoundary.Type.PERIODIC;
            boolean ph = hi.getType() == FlowBoundary.Type.PERIODIC;
            if (pl != ph) {
                throw new PDEConfigurationException("Periodic boundaries must be set on both faces of axis " + "xyz".charAt(a));
            }
            per[a] = pl;
            btype[a][0] = lo.getType();
            btype[a][1] = hi.getType();
            FlowBoundary[] both = {lo, hi};
            for (int s = 0; s < 2; s++) {
                String[] v = both[s].getVelocity();
                if (both[s].getType() == FlowBoundary.Type.INFLOW && v.length < dim) {
                    throw new PDEConfigurationException("inflow() on " + Face.of(a, s == 1) + " needs " + dim + " velocity expressions");
                }
                for (int c = 0; c < dim; c++) {
                    if (v != null && c < v.length && v[c] != null && !v[c].trim().isEmpty()) {
                        bexpr[a][s][c] = new Expr(v[c], b.params, "boundary velocity on " + Face.of(a, s == 1));
                    }
                }
            }
        }
        for (int c = 0; c < dim; c++) {
            if (b.ic[c] != null && !b.ic[c].trim().isEmpty()) {
                ic[c] = new Expr(b.ic[c], b.params, "initial velocity");
            }
        }
        for (int a = 0; a < 3; a++) {
            int q = 0;
            for (int o = 0; o < 3; o++) {
                if (o != a) {
                    other[a][q++] = o;
                }
            }
        }
        // layouts
        for (int c = 0; c < dim; c++) {
            ub[c] = new Box(dim, n[0] + (c == 0 ? 1 : 0), n[1] + (c == 1 ? 1 : 0), n[2] + (c == 2 ? 1 : 0));
        }
        pb = new Box(dim, n[0], n[1], n[2]);
        u = new double[dim][];
        us = new double[dim][];
        rCur = new double[dim][];
        rOld = new double[dim][];
        for (int c = 0; c < dim; c++) {
            u[c] = new double[ub[c].size];
            us[c] = new double[ub[c].size];
            rCur[c] = new double[ub[c].size];
            rOld[c] = new double[ub[c].size];
        }
        p = new double[pb.size];
        // ranges
        for (int c = 0; c < dim; c++) {
            for (int a = 0; a < 3; a++) {
                if (a >= dim) {
                    mLo[c][a] = 0;
                    mHi[c][a] = 0;
                    cLo[c][a] = 0;
                    cHi[c][a] = 0;
                } else if (a == c) {
                    mLo[c][a] = per[a] ? 0 : 1;
                    mHi[c][a] = per[a] ? n[a] - 1 : n[a] - 1;
                    cLo[c][a] = per[a] ? 0 : (btype[a][0] == FlowBoundary.Type.OUTFLOW ? 0 : 1);
                    cHi[c][a] = per[a] ? n[a] - 1 : (btype[a][1] == FlowBoundary.Type.OUTFLOW ? n[a] : n[a] - 1);
                } else {
                    mLo[c][a] = 0;
                    mHi[c][a] = n[a] - 1;
                    cLo[c][a] = 0;
                    cHi[c][a] = n[a] - 1;
                }
            }
        }
        // pressure solver
        PoissonMultigrid.BC[][] pbc = new PoissonMultigrid.BC[3][2];
        for (int a = 0; a < dim; a++) {
            for (int s = 0; s < 2; s++) {
                pbc[a][s] = per[a] ? PoissonMultigrid.BC.PERIODIC
                        : (btype[a][s] == FlowBoundary.Type.OUTFLOW ? PoissonMultigrid.BC.DIRICHLET : PoissonMultigrid.BC.NEUMANN);
            }
        }
        this.mg = new PoissonMultigrid(dim, n, h, pbc).smoother(b.smoother).cycle(b.cycle)
                .tolerance(b.poissonRtol, b.poissonAtol, b.poissonMaxCycles);
    }

    // ------------------------------------------------------------------ geometry helpers
    /** @return 2 or 3. */
    public int dimension() {
        return dim;
    }

    /** @return cells per direction. */
    public int cells(int axis) {
        return n[axis];
    }

    /** @return cell size along an axis. */
    public double spacing(int axis) {
        return h[axis];
    }

    /** @return flat index into {@link #velocity(int)} of the face (i,j,k) of component {@code c} (i along c = face index). */
    public int velocityIndex(int c, int i, int j, int k) {
        return ub[c].idx(i, j, k);
    }

    /** @return flat index into {@link #pressureArray()} of cell (i,j,k). */
    public int pressureIndex(int i, int j, int k) {
        return pb.idx(i, j, k);
    }

    /** @return the LIVE face array of velocity component {@code c} (ghost-padded; index with {@link #velocityIndex}). */
    public double[] velocity(int c) {
        return u[c];
    }

    /** @return the LIVE kinematic-pressure array ({@code p/rho}); multiply by the density for physical pressure. */
    public double[] pressureArray() {
        return p;
    }

    /** @return physical pressure {@code rho * p} at a cell. */
    public double pressure(int i, int j, int k) {
        return rho * p[pb.idx(i, j, k)];
    }

    public double time() {
        return time;
    }

    public long steps() {
        return steps;
    }

    public double viscosity() {
        return nu;
    }

    /** @return the position of component {@code c}'s face (i,j,k) along axis {@code a}. */
    private double coordOf(int a, int idx, int c) {
        return origin[a] + (a == c ? idx * h[a] : (idx + 0.5) * h[a]);
    }

    // ------------------------------------------------------------------ boundary conditions
    private double wallValue(int a, int s, int c, double x, double y, double z, double t) {
        Expr e = bexpr[a][s][c];
        if (e == null) {
            return 0.0;
        }
        return e.eval(x, y, z, t);
    }

    /**
     * Applies the velocity boundary conditions to {@code v}: prescribed normal
     * velocities on wall/inflow faces (when {@code setNormal}), periodic wrap,
     * and the tangential ghost layer (wall: {@code 2 w - u_inner}).
     */
    private void applyBoundary(double[][] v, double t, boolean setNormal) {
        for (int c = 0; c < dim; c++) {
            Box bx = ub[c];
            double[] uc = v[c];
            for (int a = 0; a < dim; a++) {
                int o0 = other[a][0];
                int o1 = other[a][1];
                int cnt0 = o0 < dim ? n[o0] + (o0 == c ? 1 : 0) : 1;
                int cnt1 = o1 < dim ? n[o1] + (o1 == c ? 1 : 0) : 1;
                for (int s = 0; s < 2; s++) {
                    FlowBoundary.Type bt = per[a] ? FlowBoundary.Type.PERIODIC : btype[a][s];
                    for (int t1 = 0; t1 < cnt1; t1++) {
                        for (int t0 = 0; t0 < cnt0; t0++) {
                            m[o0] = t0;
                            m[o1] = t1;
                            if (a == c) {
                                if (bt == FlowBoundary.Type.PERIODIC) {
                                    if (s == 0) {
                                        m[a] = 0;
                                        int q0 = bx.idx(m[0], m[1], m[2]);
                                        m[a] = n[a];
                                        int qn = bx.idx(m[0], m[1], m[2]);
                                        uc[qn] = uc[q0];
                                        uc[q0 - bx.st[a]] = uc[qn - bx.st[a]];
                                        uc[qn + bx.st[a]] = uc[q0 + bx.st[a]];
                                    }
                                } else if (setNormal && bt != FlowBoundary.Type.OUTFLOW) {
                                    m[a] = s == 0 ? 0 : n[a];
                                    int q = bx.idx(m[0], m[1], m[2]);
                                    double val = 0.0;
                                    if (bt == FlowBoundary.Type.INFLOW) {
                                        val = bexpr[a][s][c] == null ? 0.0 : bexpr[a][s][c].eval(posX(0, a, s, c), posX(1, a, s, c), posX(2, a, s, c), t);
                                    }
                                    uc[q] = val;
                                }
                            } else {
                                // tangential component: ghost cell
                                int gi = s == 0 ? -1 : n[a];
                                int ii = s == 0 ? 0 : n[a] - 1;
                                m[a] = gi;
                                int qg = bx.idx(m[0], m[1], m[2]);
                                m[a] = ii;
                                int qi = bx.idx(m[0], m[1], m[2]);
                                double g;
                                switch (bt) {
                                    case PERIODIC:
                                        m[a] = s == 0 ? n[a] - 1 : 0;
                                        g = uc[bx.idx(m[0], m[1], m[2])];
                                        break;
                                    case SLIP:
                                    case OUTFLOW:
                                        g = uc[qi];
                                        break;
                                    default: { // NO_SLIP, INFLOW: ghost makes the face value equal the wall velocity
                                        double w = bexpr[a][s][c] == null ? 0.0
                                                : bexpr[a][s][c].eval(posX(0, a, s, c), posX(1, a, s, c), posX(2, a, s, c), t);
                                        g = 2 * w - uc[qi];
                                    }
                                }
                                uc[qg] = g;
                            }
                        }
                    }
                }
            }
        }
    }

    /** coordinate along axis {@code q} of the node currently held in {@link #m}, for component {@code c} at the wall (a, s). */
    private double posX(int q, int a, int s, int c) {
        if (q == a) {
            return origin[a] + (s == 0 ? 0.0 : len[a]);
        }
        if (q >= dim) {
            return origin[q];
        }
        return coordOf(q, m[q], c);
    }

    /** Zero-gradient extrapolation of the normal velocity on outflow faces (used for the predictor). */
    private void extrapolateOutflow(double[][] v) {
        for (int a = 0; a < dim; a++) {
            Box bx = ub[a];
            int o0 = other[a][0];
            int o1 = other[a][1];
            int cnt0 = o0 < dim ? n[o0] : 1;
            int cnt1 = o1 < dim ? n[o1] : 1;
            for (int s = 0; s < 2; s++) {
                if (per[a] || btype[a][s] != FlowBoundary.Type.OUTFLOW) {
                    continue;
                }
                for (int t1 = 0; t1 < cnt1; t1++) {
                    for (int t0 = 0; t0 < cnt0; t0++) {
                        m[o0] = t0;
                        m[o1] = t1;
                        m[a] = s == 0 ? 0 : n[a];
                        int q = bx.idx(m[0], m[1], m[2]);
                        v[a][q] = v[a][s == 0 ? q + bx.st[a] : q - bx.st[a]];
                    }
                }
            }
        }
    }

    private void fillPressureGhosts() {
        mg.fillFineGhosts(p);
    }

    // ------------------------------------------------------------------ kernels
    private double flux(double l, double r, double uf) {
        return uf * ((1.0 - gamma) * 0.5 * (l + r) + gamma * (uf > 0 ? l : r));
    }

    /** {@code R = -(u.grad)u + nu laplace(u) + f} at the interior faces of every component. */
    private void computeRhs(double[][] v, double[][] out) {
        for (int c = 0; c < dim; c++) {
            Box bc = ub[c];
            double[] uc = v[c];
            double[] r = out[c];
            final int sc = bc.st[c];
            for (int k = mLo[c][2]; k <= mHi[c][2]; k++) {
                for (int j = mLo[c][1]; j <= mHi[c][1]; j++) {
                    int baseC = bc.idx(0, j, k);
                    for (int i = mLo[c][0]; i <= mHi[c][0]; i++) {
                        int q = baseC + i;
                        double ucq = uc[q];
                        double ucp = uc[q + sc];
                        double ucm = uc[q - sc];
                        double conv = (flux(ucq, ucp, 0.5 * (ucq + ucp)) - flux(ucm, ucq, 0.5 * (ucm + ucq))) * invh[c];
                        double diff = (ucp - 2 * ucq + ucm) * invh2[c];
                        for (int d = 0; d < dim; d++) {
                            if (d == c) {
                                continue;
                            }
                            Box bd = ub[d];
                            double[] ud = v[d];
                            int qd = bd.idx(0, j, k) + i;
                            int sdd = bd.st[d];
                            int scd = bd.st[c];
                            int sCd = bc.st[d];
                            double ufp = 0.5 * (ud[qd + sdd - scd] + ud[qd + sdd]);
                            double ufm = 0.5 * (ud[qd - scd] + ud[qd]);
                            double up = uc[q + sCd];
                            double um = uc[q - sCd];
                            conv += (flux(ucq, up, ufp) - flux(um, ucq, ufm)) * invh[d];
                            diff += (up - 2 * ucq + um) * invh2[d];
                        }
                        r[q] = -conv + nu * diff + body[c];
                    }
                }
            }
        }
    }

    private double divergence(double[][] v, double[] out, double scale) {
        double max = 0;
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < n[1]; j++) {
                int base = pb.idx(0, j, k);
                for (int i = 0; i < n[0]; i++) {
                    double div = 0;
                    for (int c = 0; c < dim; c++) {
                        int qc = ub[c].idx(i, j, k);
                        div += (v[c][qc + ub[c].st[c]] - v[c][qc]) * invh[c];
                    }
                    if (out != null) {
                        out[base + i] = div * scale;
                    }
                    double a = Math.abs(div);
                    if (a > max) {
                        max = a;
                    }
                }
            }
        }
        return max;
    }

    // ------------------------------------------------------------------ lifecycle
    /** Evaluates the initial velocity, applies boundary conditions and projects it onto a divergence-free field. */
    public IncompressibleFlow initialize() {
        for (int c = 0; c < dim; c++) {
            Arrays.fill(u[c], 0.0);
            Arrays.fill(us[c], 0.0);
            Arrays.fill(rCur[c], 0.0);
            Arrays.fill(rOld[c], 0.0);
        }
        Arrays.fill(p, 0.0);
        time = 0;
        steps = 0;
        haveOld = false;
        for (int c = 0; c < dim; c++) {
            if (ic[c] == null) {
                continue;
            }
            Box bx = ub[c];
            for (int k = 0; k < bx.n[2]; k++) {
                for (int j = 0; j < bx.n[1]; j++) {
                    for (int i = 0; i < bx.n[0]; i++) {
                        double x = coordOf(0, i, c);
                        double y = coordOf(1, j, c);
                        double z = dim == 3 ? coordOf(2, k, c) : 0.0;
                        u[c][bx.idx(i, j, k)] = ic[c].eval(x, y, z, 0.0);
                    }
                }
            }
        }
        applyBoundary(u, 0.0, true);
        project(u, 1.0);
        applyBoundary(u, 0.0, true);
        initialized = true;
        velocityResidual = Double.NaN;
        return this;
    }

    /** Projects {@code v} onto the divergence-free space: solves laplace(phi) = div(v)/scale and subtracts scale*grad(phi). */
    private void project(double[][] v, double scale) {
        double[] rhs = mg.rhs();
        double[] phi = mg.phi();
        Arrays.fill(phi, 0.0);
        divergence(v, rhs, 1.0 / scale);
        PoissonMultigrid.Stats st = mg.solve();
        poissonResidual = st.residual;
        poissonCycles = st.cycles;
        poissonConverged = st.converged;
        for (int c = 0; c < dim; c++) {
            Box bx = ub[c];
            int sp = pb.st[c];
            for (int k = cLo[c][2]; k <= cHi[c][2]; k++) {
                for (int j = cLo[c][1]; j <= cHi[c][1]; j++) {
                    int base = bx.idx(0, j, k);
                    int pbase = pb.idx(0, j, k);
                    for (int i = cLo[c][0]; i <= cHi[c][0]; i++) {
                        v[c][base + i] -= scale * (phi[pbase + i] - phi[pbase + i - sp]) * invh[c];
                    }
                }
            }
        }
    }

    /** @return a time step satisfying the advective (CFL) and explicit-diffusion limits for the current velocity field. */
    public double suggestTimeStep() {
        double adv = 0;
        for (int c = 0; c < dim; c++) {
            double mx = 0;
            Box bx = ub[c];
            for (int k = 0; k < bx.n[2]; k++) {
                for (int j = 0; j < bx.n[1]; j++) {
                    int base = bx.idx(0, j, k);
                    for (int i = 0; i < bx.n[0]; i++) {
                        mx = Math.max(mx, Math.abs(u[c][base + i]));
                    }
                }
            }
            adv += mx * invh[c];
        }
        double sumInvH2 = 0;
        for (int a = 0; a < dim; a++) {
            sumInvH2 += invh2[a];
        }
        double dtDiff = 0.2 / (nu * sumInvH2);
        double dtAdv = adv > 0 ? cflTarget / adv : Double.POSITIVE_INFINITY;
        return Math.min(dtDiff, dtAdv);
    }

    private double suggestTimeStepSafe() {
        try {
            return suggestTimeStep();
        } catch (RuntimeException ex) {
            return Double.NaN;
        }
    }

    /** Advances one step of {@link #suggestTimeStep()}. */
    public double step() {
        double dt = suggestTimeStep();
        step(dt);
        return dt;
    }

    /** Advances exactly {@code dt}. Allocation free. */
    public void step(double dt) {
        if (!initialized) {
            throw new PDEException("Call initialize() before stepping");
        }
        if (!(dt > 0) || Double.isInfinite(dt)) {
            throw new PDEConfigurationException("Time step must be positive and finite, got " + dt);
        }
        final double tNew = time + dt;
        computeRhs(u, rCur);
        double a0 = 1.0;
        double a1 = 0.0;
        if (haveOld) {
            double r = dt / dtOld;
            // Variable-step AB2 extrapolates R_old with weight -r/2; its stability region shrinks quickly once r > 1 and the
            // coefficients blow up for r >> 1 (e.g. a step following a shortened last step of advanceTo). Fall back to a
            // first-order (Euler) step there. Constant steps (r = 1) are unchanged.
            if (r <= 1.25) {
                a0 = 1.0 + 0.5 * r;
                a1 = -0.5 * r;
            }
        }
        fillPressureGhosts();
        for (int c = 0; c < dim; c++) {
            Box bx = ub[c];
            System.arraycopy(u[c], 0, us[c], 0, bx.size);
            int sp = pb.st[c];
            for (int k = mLo[c][2]; k <= mHi[c][2]; k++) {
                for (int j = mLo[c][1]; j <= mHi[c][1]; j++) {
                    int base = bx.idx(0, j, k);
                    int pbase = pb.idx(0, j, k);
                    for (int i = mLo[c][0]; i <= mHi[c][0]; i++) {
                        int q = base + i;
                        double gradp = (p[pbase + i] - p[pbase + i - sp]) * invh[c];
                        us[c][q] = u[c][q] + dt * (a0 * rCur[c][q] + a1 * rOld[c][q] - gradp);
                    }
                }
            }
        }
        applyBoundary(us, tNew, true);
        extrapolateOutflow(us);
        applyBoundary(us, tNew, false);
        // pressure correction
        double[] rhs = mg.rhs();
        double[] phi = mg.phi();
        Arrays.fill(phi, 0.0);
        divergence(us, rhs, 1.0 / dt);
        PoissonMultigrid.Stats st = mg.solve();
        poissonResidual = st.residual;
        poissonCycles = st.cycles;
        poissonConverged = st.converged;
        double vres = 0;
        for (int c = 0; c < dim; c++) {
            Box bx = ub[c];
            int sp = pb.st[c];
            for (int k = cLo[c][2]; k <= cHi[c][2]; k++) {
                for (int j = cLo[c][1]; j <= cHi[c][1]; j++) {
                    int base = bx.idx(0, j, k);
                    int pbase = pb.idx(0, j, k);
                    for (int i = cLo[c][0]; i <= cHi[c][0]; i++) {
                        int q = base + i;
                        us[c][q] -= dt * (phi[pbase + i] - phi[pbase + i - sp]) * invh[c];
                    }
                }
            }
            for (int k = 0; k < bx.n[2]; k++) {
                for (int j = 0; j < bx.n[1]; j++) {
                    int base = bx.idx(0, j, k);
                    for (int i = 0; i < bx.n[0]; i++) {
                        double d = Math.abs(us[c][base + i] - u[c][base + i]);
                        if (d > vres || Double.isNaN(d)) {
                            vres = d;
                        }
                    }
                }
            }
        }
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < n[1]; j++) {
                int base = pb.idx(0, j, k);
                for (int i = 0; i < n[0]; i++) {
                    p[base + i] += phi[base + i];
                }
            }
        }
        velocityResidual = vres / dt;
        if (Double.isNaN(vres) || Double.isInfinite(vres) || vres > 1e6) {
            throw new UnstableTimeStepException("The flow diverged at t=" + tNew + " (dt=" + dt + "): the largest velocity change in one step is "
                    + vres + ". Use a smaller step (suggestTimeStep() gives the CFL/diffusion limit), or raise upwindBlend() for a "
                    + "high cell Reynolds number.", dt, suggestTimeStepSafe());
        }
        // roll the state: u <- us, R_old <- R_cur
        double[][] t1 = u;
        u = us;
        us = t1;
        double[][] t2 = rOld;
        rOld = rCur;
        rCur = t2;
        applyBoundary(u, tNew, true);
        dtOld = dt;
        lastDt = dt;
        haveOld = true;
        time = tNew;
        steps++;
    }

    /** Steps with a fixed {@code dt} (the last step shortened) until {@code tEnd}. */
    public void advanceTo(double tEnd, double dt) {
        double eps = 1e-12 * Math.max(1.0, Math.abs(tEnd));
        while (time < tEnd - eps) {
            double d = Math.min(dt, tEnd - time);
            if (tEnd - time - d < 1e-9 * dt) {
                d = tEnd - time;
            }
            step(d);
        }
    }

    /**
     * Steps until the velocity change per unit time falls below {@code tolerance}
     * (a steady state) or {@code tMax} is reached.
     *
     * @return true if a steady state was reached
     */
    public boolean runToSteadyState(double tolerance, double tMax, double dt) {
        while (time < tMax) {
            step(Math.min(dt, tMax - time));
            if (steps > 2 && velocityResidual < tolerance) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ diagnostics
    /** @return a snapshot of the quantities that tell whether the simulation is behaving. */
    public FlowDiagnostics diagnostics() {
        FlowDiagnostics d = new FlowDiagnostics();
        d.time = time;
        d.steps = steps;
        d.dt = lastDt;
        double[] none = null;
        d.divergenceMax = divergence(u, none, 1.0);
        double s2 = 0;
        double integral = 0;
        double vol = 1.0;
        for (int a = 0; a < dim; a++) {
            vol *= h[a];
        }
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < n[1]; j++) {
                for (int i = 0; i < n[0]; i++) {
                    double div = 0;
                    for (int c = 0; c < dim; c++) {
                        int qc = ub[c].idx(i, j, k);
                        div += (u[c][qc + ub[c].st[c]] - u[c][qc]) * invh[c];
                    }
                    s2 += div * div;
                    integral += div * vol;
                }
            }
        }
        d.divergenceL2 = Math.sqrt(s2 / ((double) n[0] * n[1] * n[2]));
        d.divergenceIntegral = integral;
        double flux = 0;
        for (int a = 0; a < dim; a++) {
            if (per[a]) {
                continue;
            }
            double area = 1.0;
            for (int b = 0; b < dim; b++) {
                if (b != a) {
                    area *= h[b];
                }
            }
            int o0 = other[a][0];
            int o1 = other[a][1];
            int c0 = o0 < dim ? n[o0] : 1;
            int c1 = o1 < dim ? n[o1] : 1;
            for (int t1 = 0; t1 < c1; t1++) {
                for (int t0 = 0; t0 < c0; t0++) {
                    m[o0] = t0;
                    m[o1] = t1;
                    m[a] = 0;
                    flux -= u[a][ub[a].idx(m[0], m[1], m[2])] * area;
                    m[a] = n[a];
                    flux += u[a][ub[a].idx(m[0], m[1], m[2])] * area;
                }
            }
        }
        d.massFluxError = flux;
        d.pressureResidual = poissonResidual;
        d.pressureCycles = poissonCycles;
        d.pressureConverged = poissonConverged;
        d.velocityResidual = velocityResidual;
        // cell-centred velocity, energy, vorticity
        double[][] cc = new double[dim][pb.size];
        double ke = 0;
        double maxSpeed = 0;
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < n[1]; j++) {
                for (int i = 0; i < n[0]; i++) {
                    double sp2 = 0;
                    for (int c = 0; c < dim; c++) {
                        int qc = ub[c].idx(i, j, k);
                        double v = 0.5 * (u[c][qc] + u[c][qc + ub[c].st[c]]);
                        cc[c][pb.idx(i, j, k)] = v;
                        sp2 += v * v;
                    }
                    ke += 0.5 * sp2 * vol;
                    maxSpeed = Math.max(maxSpeed, Math.sqrt(sp2));
                }
            }
        }
        d.kineticEnergy = ke;
        d.maxSpeed = maxSpeed;
        double maxVort = 0;
        double ens = 0;
        for (int k = 1; k < Math.max(2, n[2] - 1); k++) {
            for (int j = 1; j < n[1] - 1; j++) {
                for (int i = 1; i < n[0] - 1; i++) {
                    int q = pb.idx(i, j, dim == 3 ? k : 0);
                    double wx = 0;
                    double wy = 0;
                    double wz = (cc[1][q + 1] - cc[1][q - 1]) * 0.5 * invh[0] - (cc[0][q + pb.st[1]] - cc[0][q - pb.st[1]]) * 0.5 * invh[1];
                    if (dim == 3) {
                        wx = (cc[2][q + pb.st[1]] - cc[2][q - pb.st[1]]) * 0.5 * invh[1] - (cc[1][q + pb.st[2]] - cc[1][q - pb.st[2]]) * 0.5 * invh[2];
                        wy = (cc[0][q + pb.st[2]] - cc[0][q - pb.st[2]]) * 0.5 * invh[2] - (cc[2][q + 1] - cc[2][q - 1]) * 0.5 * invh[0];
                    }
                    double w2 = wx * wx + wy * wy + wz * wz;
                    maxVort = Math.max(maxVort, Math.sqrt(w2));
                    ens += 0.5 * w2 * vol;
                }
            }
        }
        d.maxVorticity = maxVort;
        d.enstrophy = ens;
        d.reynolds = refU * refL / nu;
        double hmin = h[0];
        double sumInvH2 = 0;
        double advSum = 0;
        for (int a = 0; a < dim; a++) {
            hmin = Math.min(hmin, h[a]);
            sumInvH2 += invh2[a];
            double mx = 0;
            Box bx = ub[a];
            for (int k = 0; k < bx.n[2]; k++) {
                for (int j = 0; j < bx.n[1]; j++) {
                    int base = bx.idx(0, j, k);
                    for (int i = 0; i < bx.n[0]; i++) {
                        mx = Math.max(mx, Math.abs(u[a][base + i]));
                    }
                }
            }
            advSum += mx * invh[a];
        }
        d.cellReynolds = maxSpeed * hmin / nu;
        d.cfl = lastDt * advSum;
        d.diffusionNumber = lastDt * nu * sumInvH2;
        return d;
    }

    /** @return an independent snapshot of the fields for post-processing. */
    public FlowResult result() {
        return new FlowResult(this);
    }

    // package-private access for FlowResult
    int[] gridCells() {
        return n;
    }

    double[] gridSpacing() {
        return h;
    }

    double[] gridOrigin() {
        return origin;
    }

    Box[] velocityBoxes() {
        return ub;
    }

    Box pressureBox() {
        return pb;
    }

    double[][] velocityArrays() {
        return u;
    }

    double[] pressureKinematic() {
        return p;
    }

    double density() {
        return rho;
    }

    PoissonMultigrid multigrid() {
        return mg;
    }

    @Override
    public String toString() {
        return "IncompressibleFlow{" + dim + "D, cells=" + n[0] + "x" + n[1] + (dim == 3 ? "x" + n[2] : "") + ", nu=" + nu
                + ", t=" + time + ", steps=" + steps + ", multigridLevels=" + mg.levels() + "}";
    }
}
