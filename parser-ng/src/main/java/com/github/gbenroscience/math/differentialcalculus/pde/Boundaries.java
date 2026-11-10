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

import com.github.gbenroscience.math.differentialcalculus.autodiff.SystemAutoDiffEvaluator;
import java.util.Arrays;

/**
 * Compiled boundary conditions and their application to the state arrays.
 * <p>
 * Design: boundary nodes (and the duplicated end node of a periodic axis) are
 * CONSTRAINED nodes. They are not advanced by the PDE; after every state
 * update they are recomputed algebraically from the boundary condition and the
 * neighbouring interior values:
 * <ul>
 * <li>Dirichlet: {@code u = g}</li>
 * <li>Neumann/Robin: {@code a*u + b*du/dn = g} with a one-sided difference of the
 * configured accuracy, solved for the boundary node value</li>
 * <li>Periodic: the upper end node copies the lower end node</li>
 * </ul>
 * Halo cells are then filled by polynomial extrapolation of degree
 * {@code spatialOrder+1} (non-periodic) or by wrapping (periodic), so stencils
 * never branch at the boundary.
 */
final class Boundaries {

    /** A boundary coefficient expression, evaluated at boundary nodes. */
    private static final class BCExpr {

        final CompiledExpr ex;
        final double[] frame;
        final int[] axisSlot = {-1, -1, -1};
        int tSlot = -1;
        final boolean constant;
        final boolean usesT;
        double value;
        final SystemAutoDiffEvaluator ad;
        final double[] adOut = new double[3];

        BCExpr(String text, Model m, String context) {
            this.ex = new CompiledExpr(text);
            this.frame = ex.newFrame();
            boolean dep = false;
            boolean ut = false;
            for (int i = 0; i < ex.names.length; i++) {
                String nm = ex.names[i];
                int slot = ex.slots[i];
                boolean known = false;
                for (int a = 0; a < m.dim; a++) {
                    if (m.axisNames[a].equals(nm)) {
                        axisSlot[a] = slot;
                        dep = true;
                        known = true;
                    }
                }
                if (nm.equals(PDEProblem.TIME)) {
                    tSlot = slot;
                    dep = true;
                    ut = true;
                    known = true;
                }
                Double pv = m.problem.getParameters().get(nm);
                if (pv != null) {
                    ex.template[slot] = pv;
                    frame[slot] = pv;
                    known = true;
                }
                if (!known && !isConstantName(nm)) {
                    throw new PDEConfigurationException("Unknown symbol '" + nm + "' in " + context + " '" + text
                            + "'. Boundary data may use the domain variables, t, and declared parameters");
                }
            }
            this.constant = !dep;
            this.usesT = ut;
            this.ad = ut ? ex.ad() : null;
            if (constant) {
                value = ex.eval(frame);
                if (Double.isNaN(value) || Double.isInfinite(value)) {
                    throw new PDEConfigurationException("Boundary expression '" + text + "' is not finite");
                }
            }
        }

        double eval(double x, double y, double z, double t) {
            if (constant) {
                return value;
            }
            if (axisSlot[0] >= 0) {
                frame[axisSlot[0]] = x;
            }
            if (axisSlot[1] >= 0) {
                frame[axisSlot[1]] = y;
            }
            if (axisSlot[2] >= 0) {
                frame[axisSlot[2]] = z;
            }
            if (tSlot >= 0) {
                frame[tSlot] = t;
            }
            return ex.eval(frame);
        }

        /** dg/dt: exact (forward-mode AD) when available, else central difference. */
        double evalDt(double x, double y, double z, double t) {
            if (!usesT) {
                return 0.0;
            }
            if (ad != null) {
                eval(x, y, z, t);
                ad.evaluateDerivatives(frame, tSlot, 1, adOut);
                return adOut[1];
            }
            double d = 1e-6 * (1.0 + Math.abs(t));
            return (eval(x, y, z, t + d) - eval(x, y, z, t - d)) / (2 * d);
        }
    }

    private static boolean isConstantName(String n) {
        return n.equals("pi") || n.equals("PI") || n.equals("\u03c0") || n.equals("e") || n.equals("\u03c6");
    }

    /** One (state variable, face) condition. */
    private static final class FaceBC {

        BoundaryCondition.Type type;
        BCExpr a;
        BCExpr b;
        BCExpr g;
        boolean velocity;
    }

    private final Model m;
    private final Grid grid;
    private final FaceBC[][] bc;           // [stateVar][face ordinal]
    private final double[] dWeights;       // one-sided first-derivative weights
    private final double[][] extrap;       // [layer-1][k]
    private final int extrapPts;
    private final boolean[] periodic = new boolean[3];
    private boolean homogeneous;

    Boundaries(Model m) {
        this.m = m;
        this.grid = m.grid;
        int p = Math.max(2, m.spatialOrder);
        this.dWeights = Stencils.boundaryFirstDerivative(p);
        this.extrapPts = Math.max(m.spatialOrder, 2) + 2;
        this.extrap = new double[grid.halo()][];
        for (int j = 1; j <= grid.halo(); j++) {
            extrap[j - 1] = Stencils.extrapolation(j, extrapPts - 1);
        }
        this.bc = new FaceBC[m.nState][6];
        for (int a = 0; a < m.dim; a++) {
            periodic[a] = m.periodic[a];
        }
        for (int c = 0; c < m.nc; c++) {
            for (int a = 0; a < m.dim; a++) {
                BoundaryCondition lo = m.bcOrNull(m.comps[c], Face.of(a, false));
                BoundaryCondition hi = m.bcOrNull(m.comps[c], Face.of(a, true));
                if (lo == null) {
                    throw new PDEConfigurationException("Missing boundary condition for '" + m.comps[c] + "' on " + Face.of(a, false)
                            + ". Every unknown needs a condition on every face (or periodic(\"" + m.axisNames[a] + "\"))");
                }
                if (hi == null) {
                    throw new PDEConfigurationException("Missing boundary condition for '" + m.comps[c] + "' on " + Face.of(a, true)
                            + ". Every unknown needs a condition on every face (or periodic(\"" + m.axisNames[a] + "\"))");
                }
                boolean pl = lo.getType() == BoundaryCondition.Type.PERIODIC;
                boolean ph = hi.getType() == BoundaryCondition.Type.PERIODIC;
                if (pl != ph) {
                    throw new PDEConfigurationException("Periodic boundary conditions must be declared on BOTH faces of axis '"
                            + m.axisNames[a] + "' (unknown '" + m.comps[c] + "')");
                }
                if (pl != periodic[a]) {
                    throw new PDEConfigurationException("Axis '" + m.axisNames[a] + "' is periodic for some unknowns but not for '"
                            + m.comps[c] + "'; periodicity must apply to all unknowns");
                }
                if (!pl) {
                    check(lo, m.comps[c], Face.of(a, false));
                    check(hi, m.comps[c], Face.of(a, true));
                }
                bc[m.valVar[c]][Face.of(a, false).ordinal()] = compile(lo, m.comps[c], Face.of(a, false), false);
                bc[m.valVar[c]][Face.of(a, true).ordinal()] = compile(hi, m.comps[c], Face.of(a, true), false);
                if (m.velVar[c] >= 0) {
                    bc[m.velVar[c]][Face.of(a, false).ordinal()] = compile(lo, m.comps[c], Face.of(a, false), true);
                    bc[m.velVar[c]][Face.of(a, true).ordinal()] = compile(hi, m.comps[c], Face.of(a, true), true);
                }
            }
        }
    }

    private void check(BoundaryCondition b, String unknown, Face f) {
        if (b.getType() == BoundaryCondition.Type.PERIODIC) {
            throw new PDEConfigurationException("Unexpected periodic condition for '" + unknown + "' on " + f);
        }
    }

    private FaceBC compile(BoundaryCondition b, String unknown, Face f, boolean velocity) {
        FaceBC fb = new FaceBC();
        fb.type = b.getType();
        fb.velocity = velocity;
        if (fb.type != BoundaryCondition.Type.PERIODIC) {
            String ctx = "boundary condition of '" + unknown + "' on " + f;
            fb.a = new BCExpr(b.getA(), m, ctx);
            fb.b = new BCExpr(b.getB(), m, ctx);
            fb.g = new BCExpr(b.getG(), m, ctx);
        }
        return fb;
    }

    // ------------------------------------------------------------ application
    /** Recomputes every constrained node of every state variable. */
    void applyConstraints(double[][] y, double t) {
        for (int s = 0; s < m.nState; s++) {
            applyConstraints(y[s], s, t);
        }
    }

    private void applyConstraints(double[] u, int s, double t) {
        for (int a = 0; a < m.dim; a++) {
            if (periodic[a]) {
                applyPeriodicEnd(u, a);
                continue;
            }
            applyFace(u, s, a, false, t);
            applyFace(u, s, a, true, t);
        }
    }

    private void applyPeriodicEnd(double[] u, int a) {
        int st = grid.stride(a);
        int far = (grid.n(a) - 1) * st;
        int iHi = a == 0 ? 0 : grid.n(0) - 1;
        int jHi = a == 1 ? 0 : grid.n(1) - 1;
        int kHi = a == 2 ? 0 : grid.n(2) - 1;
        for (int k = 0; k <= kHi; k++) {
            for (int j = 0; j <= jHi; j++) {
                int base = grid.index(0, j, k);
                for (int i = 0; i <= iHi; i++) {
                    int idx = base + i;
                    u[idx + far] = u[idx];
                }
            }
        }
    }

    private void applyFace(double[] u, int s, int a, boolean max, double t) {
        FaceBC fb = bc[s][Face.of(a, max).ordinal()];
        int fixed = max ? grid.n(a) - 1 : 0;
        int iLo = a == 0 ? fixed : 0, iHi = a == 0 ? fixed : grid.n(0) - 1;
        int jLo = a == 1 ? fixed : 0, jHi = a == 1 ? fixed : grid.n(1) - 1;
        int kLo = a == 2 ? fixed : 0, kHi = a == 2 ? fixed : grid.n(2) - 1;
        int stIn = max ? -grid.stride(a) : grid.stride(a);
        double h = grid.h(a);
        double[] d = dWeights;
        for (int k = kLo; k <= kHi; k++) {
            double z = grid.coord(2, k);
            for (int j = jLo; j <= jHi; j++) {
                double yy = grid.coord(1, j);
                int base = grid.index(0, j, k);
                for (int i = iLo; i <= iHi; i++) {
                    int idx = base + i;
                    double x = grid.coord(0, i);
                    double g = homogeneous ? 0.0 : (fb.velocity ? fb.g.evalDt(x, yy, z, t) : fb.g.eval(x, yy, z, t));
                    if (fb.type == BoundaryCondition.Type.DIRICHLET) {
                        u[idx] = g;
                        continue;
                    }
                    double av = fb.a.eval(x, yy, z, t);
                    double bv = fb.b.eval(x, yy, z, t);
                    double rest = 0;
                    for (int q = 1; q < d.length; q++) {
                        rest += d[q] * u[idx + q * stIn];
                    }
                    double den = av - bv * d[0] / h;
                    if (Math.abs(den) < 1e-14 * (Math.abs(av) + Math.abs(bv) / h + 1e-300)) {
                        throw new PDEException("Singular Robin/Neumann boundary relation on " + Face.of(a, max)
                                + " (a - b*d0/h = 0)");
                    }
                    u[idx] = (g + bv / h * rest) / den;
                }
            }
        }
    }

    /**
     * Applies the LINEAR (homogeneous, g = 0) part of the constraints; used for
     * Jacobian-vector products, where only the dependence on the interior
     * values matters.
     */
    void applyConstraintsHomogeneous(double[][] y) {
        homogeneous = true;
        try {
            applyConstraints(y, 0.0);
        } finally {
            homogeneous = false;
        }
    }

    /** Fills the halo cells of every state variable. */
    void fillHalos(double[][] y) {
        for (int s = 0; s < m.nState; s++) {
            fillHalos(y[s]);
        }
    }

    private void fillHalos(double[] u) {
        int H = grid.halo();
        for (int a = 0; a < m.dim; a++) {
            // axes processed earlier are swept over their padded extent so that corner halos get filled
            int iLo = (0 < a) ? -grid.pad(0) : 0, iHi = (0 < a) ? grid.n(0) - 1 + grid.pad(0) : grid.n(0) - 1;
            int jLo = (1 < a) ? -grid.pad(1) : 0, jHi = (1 < a) ? grid.n(1) - 1 + grid.pad(1) : grid.n(1) - 1;
            int kLo = (2 < a) ? -grid.pad(2) : 0, kHi = (2 < a) ? grid.n(2) - 1 + grid.pad(2) : grid.n(2) - 1;
            if (a == 0) {
                iHi = 0;
            } else if (a == 1) {
                jHi = 0;
            } else {
                kHi = 0;
            }
            int n = grid.n(a);
            int st = grid.stride(a);
            final boolean per = periodic[a];
            for (int k = kLo; k <= kHi; k++) {
                for (int j = jLo; j <= jHi; j++) {
                    int base = grid.index(0, j, k);
                    for (int i = iLo; i <= iHi; i++) {
                        int idx = base + i;       // node with axis-a index 0
                        int idxMax = idx + (n - 1) * st;
                        if (per) {
                            for (int q = 1; q <= H; q++) {
                                u[idx - q * st] = u[idxMax - q * st];
                                u[idxMax + q * st] = u[idx + q * st];
                            }
                        } else {
                            for (int q = 1; q <= H; q++) {
                                double[] w = extrap[q - 1];
                                double sMin = 0;
                                double sMax = 0;
                                for (int c = 0; c < w.length; c++) {
                                    sMin += w[c] * u[idx + c * st];
                                    sMax += w[c] * u[idxMax - c * st];
                                }
                                u[idx - q * st] = sMin;
                                u[idxMax + q * st] = sMax;
                            }
                        }
                    }
                }
            }
        }
    }

    /** Constraints followed by halos. */
    void apply(double[][] y, double t) {
        applyConstraints(y, t);
        fillHalos(y);
    }

    @Override
    public String toString() {
        return "Boundaries" + Arrays.toString(periodic);
    }
}
