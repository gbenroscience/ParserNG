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
 * Cell-centred finite-volume solver for hyperbolic conservation laws
 * {@code U_t + div F(U) = S} on a uniform Cartesian grid in 1, 2 or 3 dimensions.
 * <p>
 * <b>Method.</b> Conservative update {@code dU/dt = -sum_d (F_{i+1/2} - F_{i-1/2}) / h_d}
 * from numerical fluxes at the faces; MUSCL reconstruction (limited linear
 * slopes, in primitive variables where the law provides them) of the left and
 * right face states; a choice of numerical flux ({@link NumericalFlux}); SSP
 * Runge-Kutta time stepping (forward Euler, SSP-RK2, SSP-RK3); two ghost layers
 * for periodic, outflow, reflecting-wall and fixed-inflow boundaries.
 * Faces whose limited reconstruction is unphysical fall back to the cell
 * values (counted in the diagnostics); a cell that becomes unphysical is
 * reported, never clipped silently.
 * <p>
 * <b>Conservation.</b> The scheme is conservative by construction;
 * {@link FVDiagnostics#getConservationDefect()} verifies it numerically for any
 * boundary condition by accounting for the boundary flux.
 * <p>
 * <b>Cost.</b> {@code O(N)} per stage, with {@code 3 + 1} state-sized arrays per
 * component plus primitive work arrays; no allocation while stepping. Single
 * threaded in this release. <b>Limitations.</b> Dimension-by-dimension fluxes
 * (no genuinely multidimensional Riemann solvers); at most second order in
 * space; positivity is protected by the face fallback but is not proven.
 */
public final class FiniteVolumeSolver {

    /** Time integrators. */
    public enum Time {
        EULER(1), SSP_RK2(2), SSP_RK3(3);

        final int stages;

        Time(int s) {
            stages = s;
        }
    }

    /** Condition on one face of the domain. */
    public static final class Boundary {

        /** Kind of boundary. */
        public enum Type {
            PERIODIC, OUTFLOW, REFLECTING, INFLOW
        }

        private final Type type;
        private final double[] state;

        private Boundary(Type type, double[] state) {
            this.type = type;
            this.state = state;
        }

        public static Boundary periodic() {
            return new Boundary(Type.PERIODIC, null);
        }

        /** Zero-gradient (transmissive) boundary. */
        public static Boundary outflow() {
            return new Boundary(Type.OUTFLOW, null);
        }

        /** Solid wall (the law must define {@link ConservationLaw#reflect}). */
        public static Boundary reflecting() {
            return new Boundary(Type.REFLECTING, null);
        }

        /** Fixed inflow with the given CONSERVED state. */
        public static Boundary inflow(double... conservedState) {
            return new Boundary(Type.INFLOW, conservedState.clone());
        }

        public Type getType() {
            return type;
        }
    }

    /** Creates the law at {@code build()} time, once every constant given to the builder is known. */
    public interface LawFactory {

        ConservationLaw create(Map<String, Double> params);
    }

    /** Fluent configuration. */
    public static final class Builder {

        private ConservationLaw law;
        private LawFactory lawFactory;
        private int[] cells;
        private final double[] origin = {0, 0, 0};
        private final double[] size = {1, 1, 1};
        private Limiter limiter = Limiter.MC;
        private NumericalFlux flux = NumericalFlux.HLL;
        private Time time = Time.SSP_RK3;
        private double cfl = 0.4;
        private final Boundary[][] bnd = new Boundary[3][2];
        private String[] icConserved;
        private String[] icPrimitive;
        private final Map<String, Double> params = new LinkedHashMap<String, Double>();

        public Builder law(ConservationLaw l) {
            this.law = l;
            this.lawFactory = null;
            return this;
        }

        /** Law built from the constants of this builder (used by {@code MathExpression.asFiniteVolume()}). */
        public Builder lawFactory(LawFactory f) {
            this.lawFactory = f;
            this.law = null;
            return this;
        }

        /** Cells per direction. */
        public Builder cells(int... n) {
            this.cells = n.clone();
            return this;
        }

        /** Lower corner of the domain. */
        public Builder origin(double... o) {
            for (int a = 0; a < o.length && a < 3; a++) {
                origin[a] = o[a];
            }
            return this;
        }

        /** Domain lengths. */
        public Builder size(double... s) {
            for (int a = 0; a < s.length && a < 3; a++) {
                size[a] = s[a];
            }
            return this;
        }

        public Builder limiter(Limiter l) {
            this.limiter = l;
            return this;
        }

        public Builder numericalFlux(NumericalFlux f) {
            this.flux = f;
            return this;
        }

        public Builder timeIntegrator(Time t) {
            this.time = t;
            return this;
        }

        /** Courant number (default 0.4; it applies to the SUM over directions of speed/h). */
        public Builder cfl(double c) {
            this.cfl = c;
            return this;
        }

        public Builder boundary(Face f, Boundary b) {
            bnd[f.axis()][f.isMax() ? 1 : 0] = b;
            return this;
        }

        /** Same boundary on both faces of every axis. */
        public Builder allBoundaries(Boundary b) {
            for (int a = 0; a < 3; a++) {
                bnd[a][0] = b;
                bnd[a][1] = b;
            }
            return this;
        }

        public Builder periodic(String axis) {
            int a = "xyz".indexOf(axis.trim().toLowerCase());
            if (a < 0) {
                throw new PDEConfigurationException("Axis must be x, y or z, got '" + axis + "'");
            }
            bnd[a][0] = Boundary.periodic();
            bnd[a][1] = Boundary.periodic();
            return this;
        }

        /** Initial data: one expression of x, y, z per CONSERVED component. */
        public Builder initial(String... expr) {
            this.icConserved = expr.clone();
            return this;
        }

        /** Initial data in PRIMITIVE variables (e.g. rho, u, p for Euler), converted with the law. */
        public Builder initialPrimitive(String... expr) {
            this.icPrimitive = expr.clone();
            return this;
        }

        public Builder param(String name, double value) {
            params.put(name, value);
            return this;
        }

        public FiniteVolumeSolver build() {
            return new FiniteVolumeSolver(this);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    // ------------------------------------------------------------------ state
    private static final int G = 2;
    private final ConservationLaw law;
    private final int dim;
    private final int nc;
    private final int[] n = new int[3];
    private final int[] st = new int[3];
    private final int g2;
    private final int size;
    private final double[] h = new double[3];
    private final double[] invh = new double[3];
    private final double[] origin = new double[3];
    private final double cellVolume;
    private final Limiter limiter;
    private final NumericalFlux flux;
    private final Time time;
    private final double cfl;
    private final Boundary[][] bnd = new Boundary[3][2];
    private final boolean[] periodic = new boolean[3];
    private final CompiledExpr[] ic;
    private final boolean icIsPrimitive;
    private final boolean primitive;
    private final double[][] u;
    private final double[][] a;
    private final double[][] r;
    private final double[][] w;
    private final double[][] slope;
    private final double[][] frow;
    private final double[] uL, uR, wL, wR, fl, fOut, tmp, spd, spd2, srcv;
    private double t;
    private long steps;
    private double lastDt;
    private double lastCfl;
    private long fallbacks;
    private double[] initialTotals;
    private final double[] boundaryFlux;
    private final double[] sourceIntegral;
    private double stageWeight;
    private boolean initialized;
    private final boolean hllcOk;

    private FiniteVolumeSolver(Builder b) {
        if (b.law == null && b.lawFactory == null) {
            throw new PDEConfigurationException("law(...) is required");
        }
        this.law = b.law != null ? b.law : b.lawFactory.create(new LinkedHashMap<String, Double>(b.params));
        this.dim = law.dimension();
        this.nc = law.components();
        if (b.cells == null || b.cells.length != dim) {
            throw new PDEConfigurationException("cells(...) needs " + dim + " values for this " + dim + "-D law");
        }
        for (int d = 0; d < 3; d++) {
            n[d] = d < dim ? b.cells[d] : 1;
            if (d < dim && n[d] < 4) {
                throw new PDEConfigurationException("Need at least 4 cells per direction, got " + n[d]);
            }
            origin[d] = b.origin[d];
            h[d] = d < dim ? b.size[d] / n[d] : 1.0;
            invh[d] = 1.0 / h[d];
        }
        this.g2 = dim == 3 ? G : 0;
        st[0] = 1;
        st[1] = n[0] + 2 * G;
        st[2] = (n[0] + 2 * G) * (n[1] + 2 * G);
        size = (n[0] + 2 * G) * (n[1] + (dim >= 2 ? 2 * G : 0)) * (n[2] + 2 * g2);
        // for 1-D the second axis has a single row: keep strides consistent with 'size'
        if (dim == 1) {
            st[1] = n[0] + 2 * G;
            st[2] = n[0] + 2 * G;
        } else if (dim == 2) {
            st[2] = (n[0] + 2 * G) * (n[1] + 2 * G);
        }
        double cv = 1.0;
        for (int d = 0; d < dim; d++) {
            cv *= h[d];
        }
        this.cellVolume = cv;
        this.limiter = b.limiter;
        this.flux = b.flux;
        this.time = b.time;
        this.cfl = b.cfl;
        if (flux == NumericalFlux.HLLC && !law.supportsHllc()) {
            throw new PDEConfigurationException("The HLLC flux is not available for this conservation law; use HLL, LLF or RUSANOV");
        }
        this.hllcOk = law.supportsHllc();
        for (int d = 0; d < dim; d++) {
            Boundary lo = b.bnd[d][0] == null ? Boundary.outflow() : b.bnd[d][0];
            Boundary hi = b.bnd[d][1] == null ? Boundary.outflow() : b.bnd[d][1];
            if ((lo.type == Boundary.Type.PERIODIC) != (hi.type == Boundary.Type.PERIODIC)) {
                throw new PDEConfigurationException("Periodic boundaries must be set on both faces of axis " + "xyz".charAt(d));
            }
            bnd[d][0] = lo;
            bnd[d][1] = hi;
            periodic[d] = lo.type == Boundary.Type.PERIODIC;
            for (Boundary x : new Boundary[]{lo, hi}) {
                if (x.type == Boundary.Type.INFLOW && x.state.length != nc) {
                    throw new PDEConfigurationException("inflow state needs " + nc + " conserved values, got " + x.state.length);
                }
            }
        }
        this.primitive = law.usesPrimitiveReconstruction();
        if (b.icConserved != null && b.icPrimitive != null) {
            throw new PDEConfigurationException("Give either initial(...) or initialPrimitive(...), not both");
        }
        String[] ics = b.icConserved != null ? b.icConserved : b.icPrimitive;
        this.icIsPrimitive = b.icPrimitive != null;
        if (ics == null) {
            throw new PDEConfigurationException("Initial data are required: initial(...) or initialPrimitive(...)");
        }
        if (ics.length != nc) {
            throw new PDEConfigurationException("Initial data need " + nc + " expressions, got " + ics.length);
        }
        this.ic = new CompiledExpr[nc];
        for (int q = 0; q < nc; q++) {
            ic[q] = new CompiledExpr(ics[q]);
            for (int i = 0; i < ic[q].names.length; i++) {
                String nm = ic[q].names[i];
                Double pv = b.params.get(nm);
                if (pv != null) {
                    ic[q].template[ic[q].slots[i]] = pv;
                } else if (!(nm.equals("x") || nm.equals("y") || nm.equals("z") || nm.equals("pi") || nm.equals("PI") || nm.equals("e"))) {
                    throw new PDEConfigurationException("Unknown symbol '" + nm + "' in initial data '" + ics[q] + "'");
                }
            }
        }
        u = new double[nc][size];
        a = new double[nc][size];
        r = new double[nc][size];
        w = primitive ? new double[nc][size] : u;
        int maxRow = 0;
        for (int d = 0; d < dim; d++) {
            maxRow = Math.max(maxRow, n[d] + 2 * G);
        }
        slope = new double[nc][maxRow];
        frow = new double[nc][maxRow];
        uL = new double[nc];
        uR = new double[nc];
        wL = new double[nc];
        wR = new double[nc];
        fl = new double[nc];
        fOut = new double[nc];
        tmp = new double[nc];
        spd = new double[2];
        spd2 = new double[2];
        srcv = new double[nc];
        boundaryFlux = new double[nc];
        sourceIntegral = new double[nc];
    }

    // ------------------------------------------------------------------ indexing
    /** @return flat index of cell (i,j,k) of a component array (ghost cells use -2..n+1). */
    public int index(int i, int j, int k) {
        return (i + G) + (j + (dim >= 2 ? G : 0)) * st[1] + (k + g2) * st[2];
    }

    /** @return the LIVE array of conserved component {@code q} (ghost-padded; index with {@link #index}). */
    public double[] state(int q) {
        return u[q];
    }

    public int cells(int axis) {
        return n[axis];
    }

    public double spacing(int axis) {
        return h[axis];
    }

    public double cellCenter(int axis, int i) {
        return origin[axis] + (i + 0.5) * h[axis];
    }

    public double time() {
        return t;
    }

    public long steps() {
        return steps;
    }

    public ConservationLaw law() {
        return law;
    }

    private int jr() {
        return dim >= 2 ? n[1] : 1;
    }

    // ------------------------------------------------------------------ initialisation
    /** Sets the initial data (cell-centre sampling) and resets time. */
    public FiniteVolumeSolver initialize() {
        for (int q = 0; q < nc; q++) {
            Arrays.fill(u[q], 0.0);
            Arrays.fill(a[q], 0.0);
            Arrays.fill(r[q], 0.0);
        }
        double[][] frames = new double[nc][];
        for (int q = 0; q < nc; q++) {
            frames[q] = ic[q].newFrame();
        }
        int[][] sl = new int[nc][3];
        String[] ax = {"x", "y", "z"};
        for (int q = 0; q < nc; q++) {
            for (int d = 0; d < 3; d++) {
                sl[q][d] = ic[q].slotOf(ax[d]);
            }
        }
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < jr(); j++) {
                for (int i = 0; i < n[0]; i++) {
                    double[] pos = {cellCenter(0, i), dim >= 2 ? cellCenter(1, j) : 0.0, dim == 3 ? cellCenter(2, k) : 0.0};
                    for (int q = 0; q < nc; q++) {
                        for (int d = 0; d < 3; d++) {
                            if (sl[q][d] >= 0) {
                                frames[q][sl[q][d]] = pos[d];
                            }
                        }
                        wL[q] = ic[q].eval(frames[q]);
                    }
                    int idx = index(i, j, k);
                    if (icIsPrimitive) {
                        law.toConserved(wL, uL);
                        for (int q = 0; q < nc; q++) {
                            u[q][idx] = uL[q];
                        }
                    } else {
                        for (int q = 0; q < nc; q++) {
                            u[q][idx] = wL[q];
                        }
                    }
                    for (int q = 0; q < nc; q++) {
                        uL[q] = u[q][idx];
                    }
                    if (!law.isPhysical(uL)) {
                        throw new PDEConfigurationException("Initial state is not physical at cell (" + i + "," + j + "," + k + ")");
                    }
                }
            }
        }
        t = 0;
        steps = 0;
        fallbacks = 0;
        Arrays.fill(boundaryFlux, 0.0);
        Arrays.fill(sourceIntegral, 0.0);
        initialTotals = totals(u);
        initialized = true;
        return this;
    }

    private double[] totals(double[][] arr) {
        double[] s = new double[nc];
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < jr(); j++) {
                int base = index(0, j, k);
                for (int i = 0; i < n[0]; i++) {
                    for (int q = 0; q < nc; q++) {
                        s[q] += arr[q][base + i];
                    }
                }
            }
        }
        for (int q = 0; q < nc; q++) {
            s[q] *= cellVolume;
        }
        return s;
    }

    // ------------------------------------------------------------------ boundaries
    private int stride(int d) {
        return st[d];
    }

    private void fillGhosts(double[][] arr) {
        for (int ax = 0; ax < dim; ax++) {
            // ranges of the other axes: earlier axes include their ghosts (so corners are filled), later ones do not
            final int iLo = (0 < ax) ? -G : 0;
            final int iHi = ax == 0 ? 0 : ((0 < ax) ? n[0] - 1 + G : n[0] - 1);
            final int jLo = (1 < ax && dim >= 2) ? -G : 0;
            final int jHi = ax == 1 ? 0 : ((1 < ax && dim >= 2) ? jr() - 1 + G : jr() - 1);
            final int kLo = (2 < ax && dim >= 3) ? -G : 0;
            final int kHi = ax == 2 ? 0 : n[2] - 1;
            int s = stride(ax);
            for (int k = kLo; k <= kHi; k++) {
                for (int j = jLo; j <= jHi; j++) {
                    for (int i = iLo; i <= iHi; i++) {
                        int base = index(i, j, k);          // cell with axis-ax index 0 (other indices as looped)
                        for (int side = 0; side < 2; side++) {
                            Boundary b = bnd[ax][side];
                            for (int g = 1; g <= G; g++) {
                                int ghost = side == 0 ? base - g * s : base + (n[ax] - 1 + g) * s;
                                int src;
                                switch (b.type) {
                                    case PERIODIC:
                                        src = side == 0 ? base + (n[ax] - g) * s : base + (g - 1) * s;
                                        for (int q = 0; q < nc; q++) {
                                            arr[q][ghost] = arr[q][src];
                                        }
                                        break;
                                    case OUTFLOW:
                                        src = side == 0 ? base : base + (n[ax] - 1) * s;
                                        for (int q = 0; q < nc; q++) {
                                            arr[q][ghost] = arr[q][src];
                                        }
                                        break;
                                    case REFLECTING:
                                        src = side == 0 ? base + (g - 1) * s : base + (n[ax] - g) * s;
                                        for (int q = 0; q < nc; q++) {
                                            tmp[q] = arr[q][src];
                                        }
                                        law.reflect(ax, tmp);
                                        for (int q = 0; q < nc; q++) {
                                            arr[q][ghost] = tmp[q];
                                        }
                                        break;
                                    default:
                                        for (int q = 0; q < nc; q++) {
                                            arr[q][ghost] = b.state[q];
                                        }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ right-hand side
    private void numericalFlux(int dir, double[] l, double[] rr, double[] out) {
        switch (flux) {
            case HLLC:
                law.hllc(dir, l, rr, out);
                return;
            case HLL: {
                law.waveSpeeds(dir, l, spd);
                law.waveSpeeds(dir, rr, spd2);
                double sL = Math.min(spd[0], spd2[0]);
                double sR = Math.max(spd[1], spd2[1]);
                if (sL >= 0) {
                    law.flux(dir, l, out);
                } else if (sR <= 0) {
                    law.flux(dir, rr, out);
                } else {
                    law.flux(dir, l, out);
                    law.flux(dir, rr, fl);
                    double inv = 1.0 / (sR - sL);
                    for (int q = 0; q < nc; q++) {
                        out[q] = (sR * out[q] - sL * fl[q] + sL * sR * (rr[q] - l[q])) * inv;
                    }
                }
                return;
            }
            default: { // LLF / RUSANOV
                law.waveSpeeds(dir, l, spd);
                law.waveSpeeds(dir, rr, spd2);
                double al = Math.max(Math.max(Math.abs(spd[0]), Math.abs(spd[1])), Math.max(Math.abs(spd2[0]), Math.abs(spd2[1])));
                law.flux(dir, l, out);
                law.flux(dir, rr, fl);
                for (int q = 0; q < nc; q++) {
                    out[q] = 0.5 * (out[q] + fl[q]) - 0.5 * al * (rr[q] - l[q]);
                }
            }
        }
    }

    /** Computes {@code res = L(arr)} and accumulates the weighted boundary flux. */
    private void rhs(double[][] arr, double[][] res, double tNow) {
        fillGhosts(arr);
        if (primitive) {
            // conserved -> primitive over the whole padded array (ghost corners were filled too)
            for (int idx = 0; idx < size; idx++) {
                for (int q = 0; q < nc; q++) {
                    uL[q] = arr[q][idx];
                }
                if (uL[0] == 0.0) {
                    for (int q = 0; q < nc; q++) {
                        w[q][idx] = 0.0;
                    }
                    continue;
                }
                law.toPrimitive(uL, wL);
                for (int q = 0; q < nc; q++) {
                    w[q][idx] = wL[q];
                }
            }
        }
        for (int q = 0; q < nc; q++) {
            Arrays.fill(res[q], 0.0);
        }
        final boolean muscl = limiter != Limiter.NONE;
        // reconstruction acts on the CURRENT stage array: the primitive work array when the law has primitives,
        // otherwise the conserved array being differentiated (not always u: stages 2 and 3 live in 'a')
        final double[][] wa = primitive ? w : arr;
        for (int d = 0; d < dim; d++) {
            int s = stride(d);
            int nd = n[d];
            final int iHi = d == 0 ? 0 : n[0] - 1;
            final int jHi = d == 1 ? 0 : jr() - 1;
            final int kHi = d == 2 ? 0 : n[2] - 1;
            double faceArea = 1.0;
            for (int e = 0; e < dim; e++) {
                if (e != d) {
                    faceArea *= h[e];
                }
            }
            for (int k = 0; k <= kHi; k++) {
                for (int j = 0; j <= jHi; j++) {
                    for (int i = 0; i <= iHi; i++) {
                        int base = index(i, j, k);
                        if (muscl) {
                            for (int c = -1; c <= nd; c++) {
                                for (int q = 0; q < nc; q++) {
                                    double[] wq = wa[q];
                                    int p = base + c * s;
                                    slope[q][c + 1] = limiter.slope(wq[p] - wq[p - s], wq[p + s] - wq[p]);
                                }
                            }
                        }
                        for (int f = 0; f <= nd; f++) {
                            int pl = base + (f - 1) * s;
                            int pr = base + f * s;
                            boolean ok = false;
                            if (muscl) {
                                for (int q = 0; q < nc; q++) {
                                    wL[q] = wa[q][pl] + 0.5 * slope[q][f];
                                    wR[q] = wa[q][pr] - 0.5 * slope[q][f + 1];
                                }
                                if (primitive) {
                                    law.toConserved(wL, uL);
                                    law.toConserved(wR, uR);
                                } else {
                                    for (int q = 0; q < nc; q++) {
                                        uL[q] = wL[q];
                                        uR[q] = wR[q];
                                    }
                                }
                                ok = law.isPhysical(uL) && law.isPhysical(uR);
                                if (!ok) {
                                    fallbacks++;
                                }
                            }
                            if (!ok) {
                                for (int q = 0; q < nc; q++) {
                                    uL[q] = arr[q][pl];
                                    uR[q] = arr[q][pr];
                                }
                            }
                            numericalFlux(d, uL, uR, fOut);
                            for (int q = 0; q < nc; q++) {
                                frow[q][f] = fOut[q];
                            }
                        }
                        for (int c = 0; c < nd; c++) {
                            int p = base + c * s;
                            for (int q = 0; q < nc; q++) {
                                res[q][p] -= (frow[q][c + 1] - frow[q][c]) * invh[d];
                            }
                        }
                        // boundary flux (outward positive) for the conservation defect
                        for (int q = 0; q < nc; q++) {
                            boundaryFlux[q] += stageWeight * lastDtForFlux * faceArea * (frow[q][nd] - frow[q][0]);
                        }
                    }
                }
            }
        }
        if (law.hasSource()) {
            for (int k = 0; k < n[2]; k++) {
                for (int j = 0; j < jr(); j++) {
                    for (int i = 0; i < n[0]; i++) {
                        int idx = index(i, j, k);
                        for (int q = 0; q < nc; q++) {
                            uL[q] = arr[q][idx];
                        }
                        law.source(uL, cellCenter(0, i), dim >= 2 ? cellCenter(1, j) : 0.0, dim == 3 ? cellCenter(2, k) : 0.0, tNow, srcv);
                        for (int q = 0; q < nc; q++) {
                            res[q][idx] += srcv[q];
                            sourceIntegral[q] += stageWeight * lastDtForFlux * cellVolume * srcv[q];
                        }
                    }
                }
            }
        }
    }

    private double lastDtForFlux;

    // ------------------------------------------------------------------ stepping
    /** @return the largest stable time step for the current state at the configured Courant number. */
    public double suggestTimeStep() {
        double sum = 0;
        for (int d = 0; d < dim; d++) {
            double mx = 0;
            for (int k = 0; k < n[2]; k++) {
                for (int j = 0; j < jr(); j++) {
                    int base = index(0, j, k);
                    for (int i = 0; i < n[0]; i++) {
                        for (int q = 0; q < nc; q++) {
                            uL[q] = u[q][base + i];
                        }
                        law.waveSpeeds(d, uL, spd);
                        mx = Math.max(mx, Math.max(Math.abs(spd[0]), Math.abs(spd[1])));
                    }
                }
            }
            sum += mx * invh[d];
        }
        return sum > 0 ? cfl / sum : Double.POSITIVE_INFINITY;
    }

    /** Advances one step of {@link #suggestTimeStep()}; returns the step taken. */
    public double step() {
        double dt = suggestTimeStep();
        if (Double.isInfinite(dt)) {
            throw new PDEException("All wave speeds are zero: pass an explicit time step to step(dt)");
        }
        step(dt);
        return dt;
    }

    /** Advances exactly {@code dt} (the caller is responsible for dt respecting the CFL condition). */
    public void step(double dt) {
        if (!initialized) {
            throw new PDEException("Call initialize() before stepping");
        }
        if (!(dt > 0) || Double.isInfinite(dt)) {
            throw new PDEConfigurationException("Time step must be positive and finite, got " + dt);
        }
        lastDtForFlux = dt;
        switch (time) {
            case EULER:
                stageWeight = 1.0;
                rhs(u, r, t);
                axpyInterior(u, u, dt, r);
                break;
            case SSP_RK2:
                stageWeight = 0.5;
                rhs(u, r, t);
                axpyInterior(a, u, dt, r);                       // a = u + dt L(u)
                stageWeight = 0.5;
                rhs(a, r, t + dt);
                for (int q = 0; q < nc; q++) {                   // u = 1/2 u + 1/2 (a + dt L(a))
                    for (int k = 0; k < n[2]; k++) {
                        for (int j = 0; j < jr(); j++) {
                            int base = index(0, j, k);
                            for (int i = 0; i < n[0]; i++) {
                                int p = base + i;
                                u[q][p] = 0.5 * u[q][p] + 0.5 * (a[q][p] + dt * r[q][p]);
                            }
                        }
                    }
                }
                break;
            default:
                stageWeight = 1.0 / 6.0;
                rhs(u, r, t);
                axpyInterior(a, u, dt, r);                       // a = u + dt L(u)
                stageWeight = 1.0 / 6.0;
                rhs(a, r, t + dt);
                for (int q = 0; q < nc; q++) {                   // a = 3/4 u + 1/4 (a + dt L(a))
                    for (int k = 0; k < n[2]; k++) {
                        for (int j = 0; j < jr(); j++) {
                            int base = index(0, j, k);
                            for (int i = 0; i < n[0]; i++) {
                                int p = base + i;
                                a[q][p] = 0.75 * u[q][p] + 0.25 * (a[q][p] + dt * r[q][p]);
                            }
                        }
                    }
                }
                stageWeight = 2.0 / 3.0;
                rhs(a, r, t + 0.5 * dt);
                for (int q = 0; q < nc; q++) {                   // u = 1/3 u + 2/3 (a + dt L(a))
                    for (int k = 0; k < n[2]; k++) {
                        for (int j = 0; j < jr(); j++) {
                            int base = index(0, j, k);
                            for (int i = 0; i < n[0]; i++) {
                                int p = base + i;
                                u[q][p] = u[q][p] / 3.0 + 2.0 / 3.0 * (a[q][p] + dt * r[q][p]);
                            }
                        }
                    }
                }
        }
        checkPhysical();
        t += dt;
        steps++;
        lastDt = dt;
        lastCfl = dt * cflSum();
    }

    private double cflSum() {
        double sum = 0;
        for (int d = 0; d < dim; d++) {
            double mx = 0;
            for (int k = 0; k < n[2]; k++) {
                for (int j = 0; j < jr(); j++) {
                    int base = index(0, j, k);
                    for (int i = 0; i < n[0]; i++) {
                        for (int q = 0; q < nc; q++) {
                            uL[q] = u[q][base + i];
                        }
                        law.waveSpeeds(d, uL, spd);
                        mx = Math.max(mx, Math.max(Math.abs(spd[0]), Math.abs(spd[1])));
                    }
                }
            }
            sum += mx * invh[d];
        }
        return sum;
    }

    private void axpyInterior(double[][] dst, double[][] src, double dt, double[][] rr) {
        for (int q = 0; q < nc; q++) {
            for (int k = 0; k < n[2]; k++) {
                for (int j = 0; j < jr(); j++) {
                    int base = index(0, j, k);
                    for (int i = 0; i < n[0]; i++) {
                        int p = base + i;
                        dst[q][p] = src[q][p] + dt * rr[q][p];
                    }
                }
            }
        }
    }

    private void checkPhysical() {
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < jr(); j++) {
                int base = index(0, j, k);
                for (int i = 0; i < n[0]; i++) {
                    for (int q = 0; q < nc; q++) {
                        uL[q] = u[q][base + i];
                    }
                    boolean bad = false;
                    for (int q = 0; q < nc; q++) {
                        bad |= Double.isNaN(uL[q]) || Double.isInfinite(uL[q]);
                    }
                    if (bad || !law.isPhysical(uL)) {
                        throw new UnstableTimeStepException("The solution became non-physical (negative density/pressure or non-finite) at cell ("
                                + i + "," + j + "," + k + "), t=" + t + ". Reduce the Courant number, use a more diffusive limiter (MINMOD) "
                                + "or flux (LLF), or first-order reconstruction.", lastDtForFlux, Double.NaN);
                    }
                }
            }
        }
    }

    /** Advances to {@code tEnd} using the stable step, shortening the last one. */
    public void advanceTo(double tEnd) {
        double eps = 1e-12 * Math.max(1.0, Math.abs(tEnd));
        while (t < tEnd - eps) {
            double dt = Math.min(suggestTimeStep(), tEnd - t);
            step(dt);
        }
    }

    // ------------------------------------------------------------------ diagnostics
    /** @return a snapshot (conserved totals, conservation defect, extrema, last CFL). */
    public FVDiagnostics diagnostics() {
        FVDiagnostics d = new FVDiagnostics();
        d.time = t;
        d.steps = steps;
        d.dt = lastDt;
        d.cfl = lastCfl;
        d.totals = totals(u);
        d.initialTotals = initialTotals == null ? new double[nc] : initialTotals.clone();
        d.boundaryFlux = boundaryFlux.clone();
        d.conservationDefect = new double[nc];
        for (int q = 0; q < nc; q++) {
            d.conservationDefect[q] = d.totals[q] - d.initialTotals[q] + boundaryFlux[q] - sourceIntegral[q];
        }
        d.min = new double[nc];
        d.max = new double[nc];
        d.minPrimitive = new double[nc];
        Arrays.fill(d.min, Double.POSITIVE_INFINITY);
        Arrays.fill(d.max, Double.NEGATIVE_INFINITY);
        Arrays.fill(d.minPrimitive, Double.POSITIVE_INFINITY);
        for (int k = 0; k < n[2]; k++) {
            for (int j = 0; j < jr(); j++) {
                int base = index(0, j, k);
                for (int i = 0; i < n[0]; i++) {
                    for (int q = 0; q < nc; q++) {
                        uL[q] = u[q][base + i];
                        d.min[q] = Math.min(d.min[q], uL[q]);
                        d.max[q] = Math.max(d.max[q], uL[q]);
                    }
                    if (primitive) {
                        law.toPrimitive(uL, wL);
                        for (int q = 0; q < nc; q++) {
                            d.minPrimitive[q] = Math.min(d.minPrimitive[q], wL[q]);
                        }
                    } else {
                        for (int q = 0; q < nc; q++) {
                            d.minPrimitive[q] = d.min[q];
                        }
                    }
                }
            }
        }
        d.reconstructionFallbacks = fallbacks;
        return d;
    }

    @Override
    public String toString() {
        return "FiniteVolumeSolver{" + dim + "D, " + nc + " components, cells=" + n[0] + "x" + n[1] + "x" + n[2]
                + ", " + flux + ", " + limiter + ", " + time + ", t=" + t + "}";
    }
}
