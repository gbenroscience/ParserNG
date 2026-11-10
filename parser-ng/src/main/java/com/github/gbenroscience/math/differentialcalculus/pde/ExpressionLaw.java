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
import java.util.Map;

/**
 * A conservation law whose fluxes (and optional sources) are ParserNG
 * expressions, so the finite-volume solver shares the expression machinery of
 * the rest of the PDE engine: {@code Burgers: flux "u^2/2"}, {@code advection:
 * flux "a*u"}, and so on. Flux expressions may use the unknowns and constants only.
 * <p>
 * Wave speeds come from ParserNG's automatic differentiation: for a scalar law
 * the exact characteristic speed {@code dF/du}; for a system the Jacobian
 * {@code dF_i/du_j} is formed by AD and the speeds are bounded by
 * {@code +/- max_i sum_j |dF_i/du_j|} (a valid spectral-radius bound, so the
 * LLF / Rusanov / HLL fluxes remain stable, though HLL then reduces to a
 * Rusanov-like flux). If an expression is outside the differentiator's operator
 * set the derivative is taken by a centred difference.
 */
public final class ExpressionLaw implements ConservationLaw {

    private final int dim;
    private final int nc;
    private final String[] unknowns;
    private final CompiledExpr[][] flux;           // [dir][comp]
    private final int[][][] uSlot;                 // [dir][comp][unknown] frame slot or -1
    private final double[][][] frames;             // [dir][comp] -> frame
    private final CompiledExpr[] src;
    private final int[][] srcUSlot;
    private final int[][] srcAxisSlot;
    private final int[] srcTSlot;
    private final double[][] srcFrames;
    private final double[] jrow;
    private final double[] adOut = new double[3];
    private final double[] pos = new double[3];

    /**
     * @param unknowns names of the conserved unknowns
     * @param axes names of the space variables (length = dimension)
     * @param fluxes {@code fluxes[dir][comp]} expression of the flux of component {@code comp} along axis {@code dir}
     * @param sources source expression per component (null or entries null/empty for none); may use the axes and {@code t}
     * @param params constants usable in the expressions
     */
    public ExpressionLaw(String[] unknowns, String[] axes, String[][] fluxes, String[] sources, Map<String, Double> params) {
        this.unknowns = unknowns.clone();
        this.nc = unknowns.length;
        this.dim = axes.length;
        if (dim < 1 || dim > 3) {
            throw new PDEConfigurationException("Conservation laws need 1 to 3 space dimensions");
        }
        if (fluxes.length != dim) {
            throw new PDEConfigurationException("Need a flux expression row per space direction (" + dim + "), got " + fluxes.length);
        }
        this.flux = new CompiledExpr[dim][nc];
        this.uSlot = new int[dim][nc][nc];
        this.frames = new double[dim][nc][];
        for (int d = 0; d < dim; d++) {
            if (fluxes[d].length != nc) {
                throw new PDEConfigurationException("Direction " + d + " needs " + nc + " flux expressions, got " + fluxes[d].length);
            }
            for (int q = 0; q < nc; q++) {
                CompiledExpr ex = new CompiledExpr(fluxes[d][q]);
                for (int i = 0; i < ex.names.length; i++) {
                    String nm = ex.names[i];
                    int ui = indexOf(unknowns, nm);
                    Double pv = params == null ? null : params.get(nm);
                    if (ui >= 0) {
                        continue;
                    } else if (pv != null) {
                        ex.template[ex.slots[i]] = pv;
                    } else if (!isConst(nm)) {
                        throw new PDEConfigurationException("Unknown symbol '" + nm + "' in flux '" + fluxes[d][q]
                                + "'. A flux may use the unknowns " + java.util.Arrays.toString(unknowns) + " and declared constants only");
                    }
                }
                flux[d][q] = ex;
                frames[d][q] = ex.newFrame();
                for (int j = 0; j < nc; j++) {
                    uSlot[d][q][j] = ex.slotOf(unknowns[j]);
                }
            }
        }
        boolean anySource = false;
        if (sources != null) {
            for (String s : sources) {
                anySource |= s != null && !s.trim().isEmpty();
            }
        }
        if (anySource) {
            this.src = new CompiledExpr[nc];
            this.srcUSlot = new int[nc][nc];
            this.srcAxisSlot = new int[nc][3];
            this.srcTSlot = new int[nc];
            this.srcFrames = new double[nc][];
            for (int q = 0; q < nc; q++) {
                String s = q < sources.length && sources[q] != null && !sources[q].trim().isEmpty() ? sources[q] : "0";
                CompiledExpr ex = new CompiledExpr(s);
                java.util.Arrays.fill(srcAxisSlot[q], -1);
                srcTSlot[q] = -1;
                for (int i = 0; i < ex.names.length; i++) {
                    String nm = ex.names[i];
                    Double pv = params == null ? null : params.get(nm);
                    boolean known = indexOf(unknowns, nm) >= 0 || nm.equals("t") || indexOf(axes, nm) >= 0;
                    if (nm.equals("t")) {
                        srcTSlot[q] = ex.slots[i];
                    }
                    for (int a = 0; a < dim; a++) {
                        if (axes[a].equals(nm)) {
                            srcAxisSlot[q][a] = ex.slots[i];
                        }
                    }
                    if (pv != null) {
                        ex.template[ex.slots[i]] = pv;
                        known = true;
                    }
                    if (!known && !isConst(nm)) {
                        throw new PDEConfigurationException("Unknown symbol '" + nm + "' in source '" + s + "'");
                    }
                }
                src[q] = ex;
                srcFrames[q] = ex.newFrame();
                for (int j = 0; j < nc; j++) {
                    srcUSlot[q][j] = ex.slotOf(unknowns[j]);
                }
            }
        } else {
            this.src = null;
            this.srcUSlot = null;
            this.srcAxisSlot = null;
            this.srcTSlot = null;
            this.srcFrames = null;
        }
        this.jrow = new double[nc];
    }

    private static boolean isConst(String n) {
        return n.equals("pi") || n.equals("PI") || n.equals("e");
    }

    private static int indexOf(String[] a, String s) {
        for (int i = 0; i < a.length; i++) {
            if (a[i].equals(s)) {
                return i;
            }
        }
        return -1;
    }

    @Override
    public int components() {
        return nc;
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public String[] names() {
        return unknowns.clone();
    }

    private void load(int dir, int q, double[] u) {
        double[] fr = frames[dir][q];
        int[] sl = uSlot[dir][q];
        for (int j = 0; j < nc; j++) {
            if (sl[j] >= 0) {
                fr[sl[j]] = u[j];
            }
        }
    }

    @Override
    public void flux(int dir, double[] u, double[] f) {
        for (int q = 0; q < nc; q++) {
            load(dir, q, u);
            f[q] = flux[dir][q].eval(frames[dir][q]);
        }
    }

    /** dF_q/du_j at u (AD, or a centred difference when AD is unavailable). */
    private double partial(int dir, int q, int j, double[] u) {
        int slot = uSlot[dir][q][j];
        if (slot < 0) {
            return 0.0;
        }
        load(dir, q, u);
        SystemAutoDiffEvaluator ad = flux[dir][q].ad();
        double[] fr = frames[dir][q];
        if (ad != null) {
            ad.evaluateDerivatives(fr, slot, 1, adOut);
            return adOut[1];
        }
        double h = 1e-6 * (1.0 + Math.abs(u[j]));
        fr[slot] = u[j] + h;
        double fp = flux[dir][q].eval(fr);
        fr[slot] = u[j] - h;
        double fm = flux[dir][q].eval(fr);
        fr[slot] = u[j];
        return (fp - fm) / (2 * h);
    }

    @Override
    public void waveSpeeds(int dir, double[] u, double[] out) {
        if (nc == 1) {
            double a = partial(dir, 0, 0, u);
            out[0] = a;
            out[1] = a;
            return;
        }
        double r = 0;
        for (int q = 0; q < nc; q++) {
            double s = 0;
            for (int j = 0; j < nc; j++) {
                s += Math.abs(partial(dir, q, j, u));
            }
            r = Math.max(r, s);
        }
        out[0] = -r;
        out[1] = r;
    }

    @Override
    public boolean hasSource() {
        return src != null;
    }

    @Override
    public void source(double[] u, double x, double y, double z, double t, double[] out) {
        pos[0] = x;
        pos[1] = y;
        pos[2] = z;
        for (int q = 0; q < nc; q++) {
            double[] fr = srcFrames[q];
            for (int j = 0; j < nc; j++) {
                if (srcUSlot[q][j] >= 0) {
                    fr[srcUSlot[q][j]] = u[j];
                }
            }
            for (int a = 0; a < dim; a++) {
                if (srcAxisSlot[q][a] >= 0) {
                    fr[srcAxisSlot[q][a]] = pos[a];
                }
            }
            if (srcTSlot[q] >= 0) {
                fr[srcTSlot[q]] = t;
            }
            out[q] = src[q].eval(fr);
        }
    }
}
