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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Compiled, solver-independent form of a {@link PDEProblem}: unknowns, state
 * layout, discretization stencils, compiled equation kernels and the
 * node-level evaluation loops. Nothing here parses text after construction.
 * <p>
 * Symbol table used by the kernels (one {@code double[]} per thread, no maps):
 * <pre>
 * [0..2]                    coordinates x,y,z
 * [3]                       time
 * [SYM_STATE .. +nState)    state variables (values, then velocities)
 * [symTerm  .. +nTerms)     discretized spatial derivative terms
 * </pre>
 */
final class Model {

    static final int SYM_T = 3;
    static final int SYM_STATE = 4;
    private static final Set<String> BUILTIN_CONSTANTS = new HashSet<String>(Arrays.asList("pi", "PI", "\u03c0", "e", "\u03c6"));

    /** A discretized spatial derivative: {@code sum_m w[m] * field[idx + off[m]]}. */
    static final class Term {

        int srcVar;
        int[] order = new int[3];
        int[] off;
        double[] w;
        double centerWeight;
    }

    /** A compiled equation. */
    static final class Eq {

        String source;
        CompiledExpr expr;
        int principalComp = -1;      // component whose time derivative this equation defines (-1: steady)
        boolean explicit;            // expr is just the right-hand side F
        int pSlot = -1;              // frame slot of the principal derivative (general form)
        int[] srcSym;                // symbol index feeding each copied slot
        int[] dstSlot;               // frame slot receiving it
        boolean affine;              // F = c0 + sum a[j]*sym[aSym[j]] exactly (constant coefficients): no expression evaluation needed
        double c0;
        double[] aCoef;
        int[] aSym;
    }

    /** Per-thread scratch: symbol table and one frame per equation. */
    static final class Ctx {

        final double[] sym;
        final double[] dsym;
        final double[] adOut = new double[3];
        final double[][] frames;

        Ctx(int nSym, double[][] templates) {
            sym = new double[nSym];
            dsym = new double[nSym];
            frames = new double[templates.length][];
            for (int i = 0; i < templates.length; i++) {
                frames[i] = templates[i].clone();
            }
        }
    }

    final PDEProblem problem;
    final PDEOptions opt;
    final Grid grid;
    final int dim;
    final String[] axisNames;
    final String[] comps;
    final int nc;
    final boolean timeDependent;
    final int[] timeOrder;
    final PDEType type;
    final int nState;
    final int[] valVar;
    final int[] velVar;
    final String[] stateNames;
    final Term[] terms;
    final int symTerm;
    final int nSym;
    final Eq[] eqs;
    final Set<String> paramNames;
    final int spatialOrder;
    final int[] pdeLo = new int[3];
    final int[] pdeHi = new int[3];
    final boolean[] periodic = new boolean[3];
    final double[][] coords = new double[3][];
    final int outerAxis;
    final int[] nodeIdx;                // flat index of every real node, compact order
    final double[][] templates;
    final int threads;
    final WorkerPool pool;
    final Ctx[] ctx;
    final boolean hasMixed;
    final boolean redBlackOk;
    final boolean usesVelocity;
    // ---- analytic Jacobian (ParserNG forward-mode AD) ----
    final int[][] jSym = new int[16][];
    final int[][] jSlot = new int[16][];
    double[][][] jac;                   // [equation][dependency][padded node]
    boolean analyticAvailable;          // AD supported for every equation and equation form allows it
    boolean analyticActive;             // storage allocated
    boolean constantPartials;           // partials wrt state/terms do not depend on the state (linear PDE)
    // ---- fused row kernels for constant-coefficient linear equations ----
    boolean fused;                      // every equation is affine: use the tap-list row kernels
    int[][] fTapVar;                    // [eq][tap] source state variable
    int[][] fTapOff;                    // [eq][tap] flat offset
    double[][] fTapCoef;                // [eq][tap] coefficient (already includes 1/h^n stencil weights)

    Model(PDEProblem problem, PDEOptions opt) {
        this.problem = problem;
        this.opt = opt;
        this.dim = problem.getDimension();
        this.axisNames = new String[dim];
        for (int a = 0; a < dim; a++) {
            axisNames[a] = problem.getAxisName(a);
        }
        this.paramNames = problem.getParameters().keySet();

        // ---- 1. text analysis ------------------------------------------------
        EquationAnalyzer an = new EquationAnalyzer(axisNames, PDEProblem.TIME, problem.getDeclaredUnknowns());
        List<EquationAnalyzer.Parsed> parsed = new ArrayList<EquationAnalyzer.Parsed>();
        for (String eq : problem.getEquations()) {
            parsed.add(an.parse(eq));
        }
        if (an.unknowns.isEmpty()) {
            throw new PDEConfigurationException("No unknown function found: write derivatives as diff(u,x,2), or declare unknowns(...)");
        }
        this.comps = an.unknowns.toArray(new String[0]);
        this.nc = comps.length;
        Set<String> reserved = new LinkedHashSet<String>(Arrays.asList(axisNames));
        reserved.add(PDEProblem.TIME);
        reserved.addAll(paramNames);
        for (String c : comps) {
            if (reserved.contains(c)) {
                throw new PDEConfigurationException("Unknown '" + c + "' clashes with an independent variable or parameter name");
            }
            if (c.startsWith("pde")) {
                throw new PDEConfigurationException("Unknown names starting with 'pde' are reserved: '" + c + "'");
            }
        }
        if (parsed.size() != nc) {
            throw new PDEConfigurationException("The problem has " + nc + " unknown(s) " + Arrays.toString(comps)
                    + " but " + parsed.size() + " equation(s); the system must be square");
        }

        // ---- 2. time orders, spatial terms ----------------------------------
        this.timeOrder = new int[nc];
        List<EquationAnalyzer.DerivRef> spatialRefs = new ArrayList<EquationAnalyzer.DerivRef>();
        for (EquationAnalyzer.DerivRef r : an.byKey.values()) {
            int c = indexOf(comps, r.comp);
            if (r.tOrd > 0) {
                if (r.isSpatial()) {
                    throw new PDEConfigurationException("Mixed space-time derivative of '" + r.comp
                            + "' is not supported; write the equation without it");
                }
                timeOrder[c] = Math.max(timeOrder[c], r.tOrd);
                if (r.tOrd > 2) {
                    throw new PDEConfigurationException("Time derivatives above order 2 are not supported (found order "
                            + r.tOrd + " for '" + r.comp + "')");
                }
            } else {
                spatialRefs.add(r);
            }
        }
        boolean td = false;
        for (int c = 0; c < nc; c++) {
            td |= timeOrder[c] > 0;
        }
        this.timeDependent = td;
        if (td) {
            for (int c = 0; c < nc; c++) {
                if (timeOrder[c] == 0) {
                    throw new PDEConfigurationException("Unknown '" + comps[c] + "' has no time derivative while others do: "
                            + "algebraic constraints (DAEs) are not supported");
                }
            }
        }

        // ---- 3. state layout -------------------------------------------------
        this.valVar = new int[nc];
        this.velVar = new int[nc];
        List<String> sn = new ArrayList<String>();
        for (int c = 0; c < nc; c++) {
            valVar[c] = sn.size();
            sn.add(comps[c]);
        }
        boolean anyVel = false;
        for (int c = 0; c < nc; c++) {
            if (timeOrder[c] == 2) {
                velVar[c] = sn.size();
                sn.add(comps[c] + "_t");
                anyVel = true;
            } else {
                velVar[c] = -1;
            }
        }
        this.usesVelocity = anyVel;
        this.stateNames = sn.toArray(new String[0]);
        this.nState = stateNames.length;

        // ---- 4. stencils and halo -------------------------------------------
        this.spatialOrder = opt.spatialOrder;
        int reach = 1;
        boolean mixed = false;
        int nt = spatialRefs.size();
        this.terms = new Term[nt];
        Stencils.Stencil1D[][] st = new Stencils.Stencil1D[nt][3];
        for (int q = 0; q < nt; q++) {
            EquationAnalyzer.DerivRef r = spatialRefs.get(q);
            r.index = q;
            int active = 0;
            for (int a = 0; a < 3; a++) {
                if (r.sp[a] > 0) {
                    if (a >= dim) {
                        throw new PDEConfigurationException("Derivative with respect to an undeclared axis in '" + r.comp + "'");
                    }
                    active++;
                    st[q][a] = Stencils.interior(r.sp[a], spatialOrder, opt.firstDerivativeScheme);
                    reach = Math.max(reach, st[q][a].reach());
                }
            }
            mixed |= active > 1;
        }
        this.hasMixed = mixed;
        int minPts = Math.max(3, spatialOrder + 3);
        int[] n = new int[3];
        double[] mn = new double[3];
        double[] mx = new double[3];
        for (int a = 0; a < dim; a++) {
            n[a] = problem.getAxisPoints(a);
            mn[a] = problem.getAxisMin(a);
            mx[a] = problem.getAxisMax(a);
            if (n[a] < minPts) {
                throw new PDEConfigurationException("Axis '" + axisNames[a] + "' has " + n[a]
                        + " points; spatialOrder=" + spatialOrder + " needs at least " + minPts);
            }
        }
        this.grid = new Grid(dim, axisNames, mn, mx, n, reach);

        // ---- 5. flat terms ---------------------------------------------------
        for (int q = 0; q < nt; q++) {
            EquationAnalyzer.DerivRef r = spatialRefs.get(q);
            Term tm = new Term();
            tm.srcVar = valVar[indexOf(comps, r.comp)];
            tm.order = r.sp.clone();
            int cnt = 1;
            for (int a = 0; a < dim; a++) {
                if (r.sp[a] > 0) {
                    cnt *= st[q][a].offsets.length;
                }
            }
            int[] off = new int[cnt];
            double[] w = new double[cnt];
            off[0] = 0;
            w[0] = 1.0;
            int len = 1;
            for (int a = 0; a < dim; a++) {
                if (r.sp[a] == 0) {
                    continue;
                }
                Stencils.Stencil1D s1 = st[q][a];
                double scale = Math.pow(grid.h(a), -r.sp[a]);
                int[] no = new int[len * s1.offsets.length];
                double[] nw = new double[no.length];
                int p = 0;
                for (int u = 0; u < len; u++) {
                    for (int m = 0; m < s1.offsets.length; m++) {
                        no[p] = off[u] + s1.offsets[m] * grid.stride(a);
                        nw[p] = w[u] * s1.weights[m] * scale;
                        p++;
                    }
                }
                off = no;
                w = nw;
                len = p;
            }
            tm.off = off;
            tm.w = w;
            for (int m = 0; m < off.length; m++) {
                if (off[m] == 0) {
                    tm.centerWeight += w[m];
                }
            }
            terms[q] = tm;
        }

        // red-black ordering is valid only if every non-centre tap flips parity
        boolean rb = !mixed;
        for (int q = 0; q < nt && rb; q++) {
            for (int m = 0; m < terms[q].off.length; m++) {
                if (terms[q].off[m] == 0) {
                    continue;
                }
                int par = 0;
                int rem = terms[q].off[m];
                // decompose flat offset back into (di,dj,dk) parity: exactly one axis per tap for non-mixed terms
                for (int a = 0; a < 3; a++) {
                    if (terms[q].order[a] > 0 && a < dim) {
                        // offset is a multiple of stride(a) because only this axis participates
                        int di = rem / grid.stride(a);
                        par += Math.abs(di);
                        break;
                    }
                }
                if ((par & 1) == 0) {
                    rb = false;
                    break;
                }
            }
        }
        this.redBlackOk = rb;

        // ---- 6. PDE node ranges, coordinates ---------------------------------
        for (int a = 0; a < 3; a++) {
            coords[a] = new double[grid.n(a)];
            for (int i = 0; i < grid.n(a); i++) {
                coords[a][i] = grid.coord(a, i);
            }
            if (a >= dim) {
                pdeLo[a] = 0;
                pdeHi[a] = 0;
                continue;
            }
            Face lo = Face.of(a, false);
            BoundaryCondition bc = bcOrNull(comps[0], lo);
            periodic[a] = bc != null && bc.getType() == BoundaryCondition.Type.PERIODIC;
            pdeLo[a] = periodic[a] ? 0 : 1;
            pdeHi[a] = grid.n(a) - 2;
        }
        this.outerAxis = dim - 1;
        this.nodeIdx = new int[grid.nodeCount()];
        {
            int p = 0;
            for (int k = 0; k < grid.n(2); k++) {
                for (int j = 0; j < grid.n(1); j++) {
                    for (int i = 0; i < grid.n(0); i++) {
                        nodeIdx[p++] = grid.index(i, j, k);
                    }
                }
            }
        }

        // ---- 7. symbols and equations ---------------------------------------
        this.symTerm = SYM_STATE + nState;
        this.nSym = symTerm + nt;
        this.eqs = new Eq[nc];
        boolean[] principalTaken = new boolean[nc];
        for (int e = 0; e < nc; e++) {
            eqs[e] = compileEquation(an, parsed.get(e), principalTaken);
        }
        if (timeDependent) {
            for (int c = 0; c < nc; c++) {
                if (!principalTaken[c]) {
                    throw new PDEConfigurationException("No equation defines the time derivative of '" + comps[c]
                            + "'; each equation must contain exactly one highest-order time derivative of its own unknown");
                }
            }
        }
        this.templates = new double[nc][];
        for (int e = 0; e < nc; e++) {
            templates[e] = eqs[e].expr.template;
        }

        try {
            prepareJacobianStructure();
        } catch (RuntimeException ex) {
            // the automatic differentiator could not handle some equation: fall back to finite differences
            analyticAvailable = false;
            constantPartials = false;
            fused = false;
            for (Eq q : eqs) {
                q.affine = false;
            }
        }

        // ---- 8. classification ------------------------------------------------
        if (!timeDependent) {
            type = PDEType.ELLIPTIC;
        } else {
            boolean second = false;
            for (int c = 0; c < nc; c++) {
                second |= timeOrder[c] == 2;
            }
            boolean highSpace = false;
            for (Term t : terms) {
                for (int a = 0; a < 3; a++) {
                    highSpace |= t.order[a] >= 2;
                }
            }
            type = second ? PDEType.HYPERBOLIC : (highSpace || terms.length == 0 ? PDEType.PARABOLIC : PDEType.HYPERBOLIC);
        }

        // ---- 9. parallel infrastructure ----------------------------------------
        int th = opt.threads;
        if (th == 0) {
            th = grid.nodeCount() >= opt.parallelThresholdNodes ? Runtime.getRuntime().availableProcessors() : 1;
        }
        int outerCount = pdeHi[outerAxis] - pdeLo[outerAxis] + 1;
        th = Math.max(1, Math.min(th, Math.max(1, outerCount)));
        this.threads = th;
        this.pool = new WorkerPool(th);
        this.ctx = new Ctx[th];
        for (int i = 0; i < th; i++) {
            ctx[i] = new Ctx(nSym, templates);
        }
    }

    // ------------------------------------------------------------------ compile
    private Eq compileEquation(EquationAnalyzer an, EquationAnalyzer.Parsed p, boolean[] principalTaken) {
        Eq q = new Eq();
        q.source = p.source;
        String lhs = stripParens(p.lhs);
        String rhs = stripParens(p.rhs);
        String exprText;
        if (timeDependent) {
            // find the principal (highest-order) time derivative of this equation
            List<String> ph = new ArrayList<String>(an.placeholdersIn(p.lhs));
            for (String s : an.placeholdersIn(p.rhs)) {
                if (!ph.contains(s)) {
                    ph.add(s);
                }
            }
            String principal = null;
            for (String name : ph) {
                EquationAnalyzer.DerivRef r = an.byPlaceholder.get(name);
                if (r.tOrd == 0) {
                    continue;
                }
                int c = indexOf(comps, r.comp);
                if (r.tOrd == timeOrder[c]) {
                    if (principal != null) {
                        throw new PDEConfigurationException("Equation '" + p.source + "' contains the highest-order time derivative of more than one unknown;"
                                + " solve for one of them (explicit time stepping needs u_t = F)");
                    }
                    principal = name;
                    q.principalComp = c;
                } else if (!(r.tOrd == 1 && timeOrder[c] == 2)) {
                    throw new PDEConfigurationException("Unsupported time derivative of '" + r.comp + "' in '" + p.source + "'");
                }
            }
            if (principal == null) {
                throw new PDEConfigurationException("Equation '" + p.source + "' has no time derivative of an unknown: "
                        + "mixed steady/transient systems are not supported");
            }
            if (principalTaken[q.principalComp]) {
                throw new PDEConfigurationException("Two equations define the time derivative of '" + comps[q.principalComp] + "'");
            }
            principalTaken[q.principalComp] = true;
            if (lhs.equals(principal)) {
                q.explicit = true;
                exprText = rhs;
            } else if (rhs.equals(principal)) {
                q.explicit = true;
                exprText = lhs;
            } else {
                q.explicit = false;
                exprText = "(" + p.lhs + ")-(" + p.rhs + ")";
            }
            q.expr = new CompiledExpr(exprText);
            if (!q.explicit) {
                q.pSlot = q.expr.slotOf(principal);
                if (q.pSlot < 0) {
                    throw new PDEConfigurationException("Principal time derivative cancels out of '" + p.source + "'");
                }
            } else if (q.expr.slotOf(principal) >= 0) {
                throw new PDEConfigurationException("The time derivative appears on both sides of '" + p.source + "'");
            }
        } else {
            q.explicit = false;
            exprText = "(" + p.lhs + ")-(" + p.rhs + ")";
            q.expr = new CompiledExpr(exprText);
        }

        // map every variable of the compiled expression onto a symbol or constant
        int cnt = 0;
        int[] src = new int[q.expr.names.length];
        int[] dst = new int[q.expr.names.length];
        for (int i = 0; i < q.expr.names.length; i++) {
            String nm = q.expr.names[i];
            int slot = q.expr.slots[i];
            int sym = -1;
            for (int a = 0; a < dim; a++) {
                if (axisNames[a].equals(nm)) {
                    sym = a;
                }
            }
            if (nm.equals(PDEProblem.TIME)) {
                sym = SYM_T;
            }
            int c = indexOf(comps, nm);
            if (c >= 0) {
                sym = SYM_STATE + valVar[c];
            }
            EquationAnalyzer.DerivRef r = an.byPlaceholder.get(nm);
            if (r != null) {
                if (r.isSpatial()) {
                    sym = symTerm + r.index;
                } else {
                    int cc = indexOf(comps, r.comp);
                    if (cc >= 0 && r.tOrd == 1 && timeOrder[cc] == 2) {
                        sym = SYM_STATE + velVar[cc];
                    } else if (slot == q.pSlot || q.explicit) {
                        sym = -2; // principal, handled separately
                    } else {
                        sym = -2;
                    }
                }
            }
            if (sym == -1) {
                Double pv = problem.getParameters().get(nm);
                if (pv != null) {
                    q.expr.template[slot] = pv;
                    continue;
                }
                if (BUILTIN_CONSTANTS.contains(nm)) {
                    continue; // value already in template
                }
                throw new PDEConfigurationException("Unknown symbol '" + nm + "' in equation '" + p.source
                        + "'. Declare it with param(\"" + nm + "\", value), or add it to the unknowns / domain");
            }
            if (sym >= 0) {
                src[cnt] = sym;
                dst[cnt] = slot;
                cnt++;
            }
        }
        q.srcSym = Arrays.copyOf(src, cnt);
        q.dstSlot = Arrays.copyOf(dst, cnt);

        if (timeDependent && !q.explicit) {
            verifyLinearInPrincipal(q);
        }
        return q;
    }

    /**
     * Checks that a general-form equation is linear in, and not degenerate in,
     * its principal time derivative. Uses ParserNG's automatic differentiator
     * (exact first and second derivative with respect to the principal
     * derivative) when it supports the expression, and a three-point test
     * otherwise.
     */
    private void verifyLinearInPrincipal(Eq q) {
        double[] fr = q.expr.template.clone();
        SystemAutoDiffEvaluator ad = q.expr.ad();
        double[] d = new double[3];
        boolean checked = false;
        for (int trial = 0; trial < 4 && !checked; trial++) {
            for (int i = 0; i < q.srcSym.length; i++) {
                fr[q.dstSlot[i]] = 0.21 + 0.137 * ((i * 7 + trial * 3) % 11);
            }
            fr[q.pSlot] = 0.3;
            double a;
            double curv;
            double scale;
            if (ad != null) {
                ad.evaluateDerivatives(fr, q.pSlot, 2, d);
                if (Double.isNaN(d[0] + d[1] + d[2]) || Double.isInfinite(d[0] + d[1] + d[2])) {
                    continue;
                }
                a = d[1];
                curv = Math.abs(d[2]);
                scale = Math.abs(d[0]) + Math.abs(d[1]) + 1e-300;
            } else {
                fr[q.pSlot] = 0;
                double r0 = q.expr.eval(fr);
                fr[q.pSlot] = 1;
                double r1 = q.expr.eval(fr);
                fr[q.pSlot] = 2;
                double r2 = q.expr.eval(fr);
                if (Double.isNaN(r0 + r1 + r2) || Double.isInfinite(r0 + r1 + r2)) {
                    continue;
                }
                a = r1 - r0;
                curv = Math.abs((r2 - r1) - a);
                scale = Math.abs(r0) + Math.abs(r1) + Math.abs(r2) + 1e-300;
            }
            if (curv > 1e-8 * scale) {
                throw new PDEConfigurationException("Equation '" + q.source + "' is not linear in its highest-order time derivative;"
                        + " it cannot be solved for u_t");
            }
            if (Math.abs(a) <= 1e-12 * scale) {
                throw new PDEConfigurationException("Equation '" + q.source + "' has a zero coefficient on its time derivative");
            }
            checked = true;
        }
    }

    private static String stripParens(String s) {
        String t = s.trim();
        while (t.startsWith("(") && EquationAnalyzer.matchParen(t, 0) == t.length() - 1) {
            t = t.substring(1, t.length() - 1).trim();
        }
        return t;
    }

    static int indexOf(String[] arr, String s) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(s)) {
                return i;
            }
        }
        return -1;
    }

    BoundaryCondition bcOrNull(String unknown, Face f) {
        return problem.getBoundaryCondition(unknown, f);
    }

    // ------------------------------------------------------------------ kernels
    /** Fills the derivative and state symbols for node {@code idx}. */
    void loadNode(double[] sym, double[][] y, int idx) {
        for (int s = 0; s < nState; s++) {
            sym[SYM_STATE + s] = y[s][idx];
        }
        final Term[] ts = terms;
        for (int q = 0; q < ts.length; q++) {
            Term tm = ts[q];
            double[] src = y[tm.srcVar];
            int[] off = tm.off;
            double[] w = tm.w;
            double acc = 0;
            for (int m = 0; m < off.length; m++) {
                acc += w[m] * src[idx + off[m]];
            }
            sym[symTerm + q] = acc;
        }
    }

    /** Evaluates equation {@code e} using the already loaded symbols. */
    double evalEq(Ctx c, int e) {
        Eq q = eqs[e];
        double[] sym = c.sym;
        if (q.affine) {
            double acc = q.c0;
            final int[] as = q.aSym;
            final double[] ac = q.aCoef;
            for (int m = 0; m < as.length; m++) {
                acc += ac[m] * sym[as[m]];
            }
            return acc;
        }
        double[] fr = c.frames[e];
        int[] s = q.srcSym;
        int[] d = q.dstSlot;
        for (int m = 0; m < s.length; m++) {
            fr[d[m]] = sym[s[m]];
        }
        if (q.explicit || !timeDependent) {
            return q.expr.fn.applyScalar(fr);
        }
        fr[q.pSlot] = 0.0;
        double r0 = q.expr.fn.applyScalar(fr);
        fr[q.pSlot] = 1.0;
        double r1 = q.expr.fn.applyScalar(fr);
        return -r0 / (r1 - r0);
    }

    /** Worker-partitioned chunk of the outer axis: [lo, hi] inclusive, possibly empty. */
    int chunkLo(int worker, int workers) {
        int lo = pdeLo[outerAxis];
        long cnt = pdeHi[outerAxis] - lo + 1;
        return lo + (int) (cnt * worker / workers);
    }

    int chunkHi(int worker, int workers) {
        int lo = pdeLo[outerAxis];
        long cnt = pdeHi[outerAxis] - lo + 1;
        return lo + (int) (cnt * (worker + 1) / workers) - 1;
    }

    /** Right-hand side kernel of the semi-discrete system on one worker's slab. */
    void rhsSlab(int worker, int workers, double t, double[][] y, double[][] dy) {
        if (fused) {
            fusedRhsSlab(worker, workers, y, dy);
            return;
        }
        Ctx c = ctx[worker];
        double[] sym = c.sym;
        sym[SYM_T] = t;
        int iLo = pdeLo[0], iHi = pdeHi[0], jLo = pdeLo[1], jHi = pdeHi[1], kLo = pdeLo[2], kHi = pdeHi[2];
        if (outerAxis == 2) {
            kLo = chunkLo(worker, workers);
            kHi = chunkHi(worker, workers);
        } else if (outerAxis == 1) {
            jLo = chunkLo(worker, workers);
            jHi = chunkHi(worker, workers);
        } else {
            iLo = chunkLo(worker, workers);
            iHi = chunkHi(worker, workers);
        }
        final int sy = grid.stride(1);
        final int sz = grid.stride(2);
        final double[] xc = coords[0];
        final double[] yc = coords[1];
        final double[] zc = coords[2];
        for (int k = kLo; k <= kHi; k++) {
            sym[2] = zc[k];
            for (int j = jLo; j <= jHi; j++) {
                sym[1] = yc[j];
                int base = grid.index(0, j, k);
                for (int i = iLo; i <= iHi; i++) {
                    int idx = base + i;
                    sym[0] = xc[i];
                    loadNode(sym, y, idx);
                    for (int e = 0; e < nc; e++) {
                        int pc = eqs[e].principalComp;
                        double f = evalEq(c, e);
                        if (timeOrder[pc] == 2) {
                            dy[valVar[pc]][idx] = y[velVar[pc]][idx];
                            dy[velVar[pc]][idx] = f;
                        } else {
                            dy[valVar[pc]][idx] = f;
                        }
                    }
                }
            }
        }
    }

    /** Steady residual kernel: {@code res[e][idx] = R_e} at PDE nodes. */
    void residualSlab(int worker, int workers, double[][] y, double[][] res) {
        if (fused) {
            fusedResidualSlab(worker, workers, y, res);
            return;
        }
        Ctx c = ctx[worker];
        double[] sym = c.sym;
        sym[SYM_T] = 0.0;
        int iLo = pdeLo[0], iHi = pdeHi[0], jLo = pdeLo[1], jHi = pdeHi[1], kLo = pdeLo[2], kHi = pdeHi[2];
        if (outerAxis == 2) {
            kLo = chunkLo(worker, workers);
            kHi = chunkHi(worker, workers);
        } else if (outerAxis == 1) {
            jLo = chunkLo(worker, workers);
            jHi = chunkHi(worker, workers);
        } else {
            iLo = chunkLo(worker, workers);
            iHi = chunkHi(worker, workers);
        }
        final double[] xc = coords[0];
        final double[] yc = coords[1];
        final double[] zc = coords[2];
        for (int k = kLo; k <= kHi; k++) {
            sym[2] = zc[k];
            for (int j = jLo; j <= jHi; j++) {
                sym[1] = yc[j];
                int base = grid.index(0, j, k);
                for (int i = iLo; i <= iHi; i++) {
                    int idx = base + i;
                    sym[0] = xc[i];
                    loadNode(sym, y, idx);
                    for (int e = 0; e < nc; e++) {
                        res[e][idx] = evalEq(c, e);
                    }
                }
            }
        }
    }


    private void fusedRhsSlab(int worker, int workers, double[][] y, double[][] dy) {
        int iLo = pdeLo[0], iHi = pdeHi[0], jLo = pdeLo[1], jHi = pdeHi[1], kLo = pdeLo[2], kHi = pdeHi[2];
        if (outerAxis == 2) {
            kLo = chunkLo(worker, workers);
            kHi = chunkHi(worker, workers);
        } else if (outerAxis == 1) {
            jLo = chunkLo(worker, workers);
            jHi = chunkHi(worker, workers);
        } else {
            iLo = chunkLo(worker, workers);
            iHi = chunkHi(worker, workers);
        }
        if (iHi < iLo || jHi < jLo || kHi < kLo) {
            return;
        }
        for (int k = kLo; k <= kHi; k++) {
            for (int j = jLo; j <= jHi; j++) {
                int base = grid.index(0, j, k);
                for (int e = 0; e < nc; e++) {
                    int pc = eqs[e].principalComp;
                    if (timeOrder[pc] == 2) {
                        fusedRow(e, y, dy[velVar[pc]], base, iLo, iHi);
                        System.arraycopy(y[velVar[pc]], base + iLo, dy[valVar[pc]], base + iLo, iHi - iLo + 1);
                    } else {
                        fusedRow(e, y, dy[valVar[pc]], base, iLo, iHi);
                    }
                }
            }
        }
    }

    private void fusedResidualSlab(int worker, int workers, double[][] y, double[][] res) {
        int iLo = pdeLo[0], iHi = pdeHi[0], jLo = pdeLo[1], jHi = pdeHi[1], kLo = pdeLo[2], kHi = pdeHi[2];
        if (outerAxis == 2) {
            kLo = chunkLo(worker, workers);
            kHi = chunkHi(worker, workers);
        } else if (outerAxis == 1) {
            jLo = chunkLo(worker, workers);
            jHi = chunkHi(worker, workers);
        } else {
            iLo = chunkLo(worker, workers);
            iHi = chunkHi(worker, workers);
        }
        if (iHi < iLo || jHi < jLo || kHi < kLo) {
            return;
        }
        for (int k = kLo; k <= kHi; k++) {
            for (int j = jLo; j <= jHi; j++) {
                int base = grid.index(0, j, k);
                for (int e = 0; e < nc; e++) {
                    fusedRow(e, y, res[e], base, iLo, iHi);
                }
            }
        }
    }

    // ---- tasks dispatched to the pool (fields set before run => no allocation)
    final class RhsTask implements WorkerPool.Task {

        double t;
        double[][] y;
        double[][] dy;

        @Override
        public void run(int worker, int workers) {
            rhsSlab(worker, workers, t, y, dy);
        }
    }

    final class ResidualTask implements WorkerPool.Task {

        double[][] y;
        double[][] res;

        @Override
        public void run(int worker, int workers) {
            residualSlab(worker, workers, y, res);
        }
    }

    private final RhsTask rhsTask = new RhsTask();
    private final ResidualTask resTask = new ResidualTask();

    /** dy = F(t, y) at PDE nodes (y must already satisfy the boundary conditions and have valid halos). */
    void rhs(double t, double[][] y, double[][] dy) {
        rhsTask.t = t;
        rhsTask.y = y;
        rhsTask.dy = dy;
        pool.run(rhsTask);
    }

    /** res[e] = residual of equation e at PDE nodes (steady problems). */
    void residual(double[][] y, double[][] res) {
        resTask.y = y;
        resTask.res = res;
        pool.run(resTask);
    }

    /** @return 1 + the number of PDE (non-constrained) nodes. */
    int pdeNodeCount() {
        int c = 1;
        for (int a = 0; a < 3; a++) {
            c *= pdeHi[a] - pdeLo[a] + 1;
        }
        return c;
    }

    boolean isPdeNode(int i, int j, int k) {
        return i >= pdeLo[0] && i <= pdeHi[0] && j >= pdeLo[1] && j <= pdeHi[1] && k >= pdeLo[2] && k <= pdeHi[2];
    }


    // ------------------------------------------------------------------
    // Analytic Jacobian via ParserNG's forward-mode automatic differentiation
    // ------------------------------------------------------------------
    /**
     * Records, per equation, which state / derivative symbols it depends on and
     * whether AD can differentiate it. Also tests (once, exactly, by AD at
     * random symbol values) whether the partial derivatives are
     * state-independent, i.e. the PDE is linear in the unknowns.
     */
    private void prepareJacobianStructure() {
        boolean ok = true;
        for (int e = 0; e < nc; e++) {
            Eq q = eqs[e];
            int cnt = 0;
            int[] sy = new int[q.srcSym.length];
            int[] sl = new int[q.srcSym.length];
            for (int m = 0; m < q.srcSym.length; m++) {
                if (q.srcSym[m] >= SYM_STATE) {
                    sy[cnt] = q.srcSym[m];
                    sl[cnt] = q.dstSlot[m];
                    cnt++;
                }
            }
            jSym[e] = Arrays.copyOf(sy, cnt);
            jSlot[e] = Arrays.copyOf(sl, cnt);
            if (q.expr.ad() == null || (timeDependent && !q.explicit)) {
                ok = false;
            }
        }
        analyticAvailable = ok;
        if (!ok) {
            return;
        }
        // structural linearity test: partials identical at several random symbol vectors
        boolean constant = true;
        double[] out = new double[3];
        double[][] ref = new double[nc][];
        for (int trial = 0; trial < 3 && constant; trial++) {
            for (int e = 0; e < nc && constant; e++) {
                Eq q = eqs[e];
                double[] fr = q.expr.template.clone();
                for (int m = 0; m < q.srcSym.length; m++) {
                    // coordinates / time are held fixed (variable coefficients are fine); state and derivative symbols vary
                    fr[q.dstSlot[m]] = q.srcSym[m] < SYM_STATE ? 0.37 + 0.11 * q.srcSym[m]
                            : 0.31 + 0.173 * ((m * 5 + trial * 7 + e) % 13) - 0.4 * trial;
                }
                double[] cur = new double[jSym[e].length];
                for (int j = 0; j < jSym[e].length; j++) {
                    q.expr.ad().evaluateDerivatives(fr, jSlot[e][j], 1, out);
                    cur[j] = out[1];
                }
                if (trial == 0) {
                    ref[e] = cur;
                } else {
                    for (int j = 0; j < cur.length; j++) {
                        if (Math.abs(cur[j] - ref[e][j]) > 1e-9 * (Math.abs(ref[e][j]) + 1e-12)
                                || Double.isNaN(cur[j])) {
                            constant = false;
                        }
                    }
                }
            }
        }
        constantPartials = constant;
        detectAffineKernels();
    }

    /**
     * Specialised-kernel detection. If ParserNG's automatic differentiator
     * gives constant partial derivatives and the expression reproduces
     * {@code c0 + sum a_j s_j} at several independent random points (all
     * symbols, coordinates and time varied), the equation is replaced by that
     * dot product: no expression evaluation, no frame copy.
     */
    private void detectAffineKernels() {
        double[] out = new double[3];
        long seed = 0x2545F4914F6CDD1DL;
        for (int e = 0; e < nc; e++) {
            Eq q = eqs[e];
            if (q.expr.ad() == null || (timeDependent && !q.explicit)) {
                continue;
            }
            int nj = jSym[e].length;
            double[] fr = q.expr.template.clone();
            // base point
            for (int mm = 0; mm < q.srcSym.length; mm++) {
                seed = seed * 6364136223846793005L + 1442695040888963407L;
                fr[q.dstSlot[mm]] = 0.15 + 0.7 * ((seed >>> 11) * (1.0 / 9007199254740992.0));
            }
            double[] a = new double[nj];
            boolean ok = true;
            for (int j = 0; j < nj; j++) {
                q.expr.ad().evaluateDerivatives(fr, jSlot[e][j], 1, out);
                a[j] = out[1];
                ok &= !Double.isNaN(a[j]) && !Double.isInfinite(a[j]);
            }
            // c0 = F with every state / derivative symbol zero
            double[] z = fr.clone();
            for (int mm = 0; mm < q.srcSym.length; mm++) {
                if (q.srcSym[mm] >= SYM_STATE) {
                    z[q.dstSlot[mm]] = 0.0;
                }
            }
            double c0 = q.expr.eval(z);
            ok &= !Double.isNaN(c0) && !Double.isInfinite(c0);
            for (int trial = 0; trial < 5 && ok; trial++) {
                double[] pt = q.expr.template.clone();
                double lin = c0;
                for (int mm = 0; mm < q.srcSym.length; mm++) {
                    seed = seed * 6364136223846793005L + 1442695040888963407L;
                    double v = -0.9 + 1.8 * ((seed >>> 11) * (1.0 / 9007199254740992.0));
                    pt[q.dstSlot[mm]] = v;
                }
                for (int j = 0; j < nj; j++) {
                    lin += a[j] * pt[jSlot[e][j]];
                }
                double f = q.expr.eval(pt);
                if (Double.isNaN(f) || Math.abs(f - lin) > 1e-9 * (1.0 + Math.abs(f) + Math.abs(lin))) {
                    ok = false;
                }
            }
            if (ok) {
                q.affine = true;
                q.c0 = c0;
                q.aCoef = a;
                q.aSym = jSym[e].clone();
            }
        }
        boolean all = true;
        for (int e = 0; e < nc; e++) {
            all &= eqs[e].affine;
        }
        fused = all;
        if (all) {
            buildFusedTaps();
        }
    }

    /** Merges the stencils of an affine equation into one flat (variable, offset, coefficient) tap list. */
    private void buildFusedTaps() {
        fTapVar = new int[nc][];
        fTapOff = new int[nc][];
        fTapCoef = new double[nc][];
        for (int e = 0; e < nc; e++) {
            Eq q = eqs[e];
            java.util.LinkedHashMap<Long, Double> taps = new java.util.LinkedHashMap<Long, Double>();
            for (int j = 0; j < q.aSym.length; j++) {
                int sym = q.aSym[j];
                double a = q.aCoef[j];
                if (a == 0.0) {
                    continue;
                }
                if (sym < symTerm) {
                    addTap(taps, sym - SYM_STATE, 0, a);
                } else {
                    Term tm = terms[sym - symTerm];
                    for (int mm = 0; mm < tm.off.length; mm++) {
                        addTap(taps, tm.srcVar, tm.off[mm], a * tm.w[mm]);
                    }
                }
            }
            int n = taps.size();
            fTapVar[e] = new int[n];
            fTapOff[e] = new int[n];
            fTapCoef[e] = new double[n];
            int p = 0;
            for (java.util.Map.Entry<Long, Double> en : taps.entrySet()) {
                long key = en.getKey();
                fTapVar[e][p] = (int) (key >> 32);
                fTapOff[e][p] = (int) key;
                fTapCoef[e][p] = en.getValue();
                p++;
            }
        }
    }

    private static void addTap(java.util.Map<Long, Double> taps, int var, int off, double c) {
        long key = ((long) var << 32) | (off & 0xffffffffL);
        Double old = taps.get(key);
        taps.put(key, old == null ? c : old + c);
    }

    /**
     * Row kernel of a fused (constant-coefficient linear) equation:
     * {@code out[idx] = c0 + sum_t coef_t * src_t[idx + off_t]} for the nodes
     * {@code base+iLo .. base+iHi}. Plain contiguous loops: auto-vectorisable.
     */
    private void fusedRow(int e, double[][] y, double[] out, int base, int iLo, int iHi) {
        final int[] tv = fTapVar[e];
        final int[] to = fTapOff[e];
        final double[] tc = fTapCoef[e];
        final double c0 = eqs[e].c0;
        final int from = base + iLo;
        final int to1 = base + iHi + 1;
        for (int i = from; i < to1; i++) {
            out[i] = c0;
        }
        int t = 0;
        // two taps per pass halves the traffic on out[]
        for (; t + 1 < tv.length; t += 2) {
            final double[] s0 = y[tv[t]];
            final double[] s1 = y[tv[t + 1]];
            final double c1a = tc[t];
            final double c1b = tc[t + 1];
            final int o0 = to[t];
            final int o1 = to[t + 1];
            for (int i = from; i < to1; i++) {
                out[i] += c1a * s0[i + o0] + c1b * s1[i + o1];
            }
        }
        if (t < tv.length) {
            final double[] s0 = y[tv[t]];
            final double c1a = tc[t];
            final int o0 = to[t];
            for (int i = from; i < to1; i++) {
                out[i] += c1a * s0[i + o0];
            }
        }
    }

    /** Allocates the Jacobian partial-derivative storage if it fits the memory budget. */
    boolean ensureJacobian() {
        if (!analyticAvailable) {
            return false;
        }
        if (analyticActive) {
            return true;
        }
        long total = 0;
        for (int e = 0; e < nc; e++) {
            total += (long) jSym[e].length * grid.storageSize();
        }
        if (total > 48_000_000L) {
            analyticAvailable = false;
            return false;
        }
        jac = new double[nc][][];
        for (int e = 0; e < nc; e++) {
            jac[e] = new double[jSym[e].length][grid.storageSize()];
        }
        analyticActive = true;
        return true;
    }

    private void fillFrame(Ctx c, int e) {
        Eq q = eqs[e];
        double[] fr = c.frames[e];
        double[] sym = c.sym;
        int[] s = q.srcSym;
        int[] d = q.dstSlot;
        for (int m = 0; m < s.length; m++) {
            fr[d[m]] = sym[s[m]];
        }
    }

    /** Computes exact partial derivatives of every equation w.r.t. its state / derivative symbols at PDE nodes. */
    void linearizeSlab(int worker, int workers, double t, double[][] y) {
        Ctx c = ctx[worker];
        double[] sym = c.sym;
        sym[SYM_T] = t;
        int iLo = pdeLo[0], iHi = pdeHi[0], jLo = pdeLo[1], jHi = pdeHi[1], kLo = pdeLo[2], kHi = pdeHi[2];
        if (outerAxis == 2) {
            kLo = chunkLo(worker, workers);
            kHi = chunkHi(worker, workers);
        } else if (outerAxis == 1) {
            jLo = chunkLo(worker, workers);
            jHi = chunkHi(worker, workers);
        } else {
            iLo = chunkLo(worker, workers);
            iHi = chunkHi(worker, workers);
        }
        final double[] xc = coords[0];
        final double[] yc = coords[1];
        final double[] zc = coords[2];
        double[] out = c.adOut;
        for (int k = kLo; k <= kHi; k++) {
            sym[2] = zc[k];
            for (int j = jLo; j <= jHi; j++) {
                sym[1] = yc[j];
                int base = grid.index(0, j, k);
                for (int i = iLo; i <= iHi; i++) {
                    int idx = base + i;
                    sym[0] = xc[i];
                    loadNode(sym, y, idx);
                    for (int e = 0; e < nc; e++) {
                        fillFrame(c, e);
                        SystemAutoDiffEvaluator ad = eqs[e].expr.ad();
                        double[] fr = c.frames[e];
                        int[] sl = jSlot[e];
                        for (int m = 0; m < sl.length; m++) {
                            ad.evaluateDerivatives(fr, sl[m], 1, out);
                            jac[e][m][idx] = out[1];
                        }
                    }
                }
            }
        }
    }

    /** out = F'(y) * v at PDE nodes, using the stored partials (v needs valid halos). */
    void jacApplySlab(int worker, int workers, double[][] v, double[][] out) {
        Ctx c = ctx[worker];
        double[] ds = c.dsym;
        int iLo = pdeLo[0], iHi = pdeHi[0], jLo = pdeLo[1], jHi = pdeHi[1], kLo = pdeLo[2], kHi = pdeHi[2];
        if (outerAxis == 2) {
            kLo = chunkLo(worker, workers);
            kHi = chunkHi(worker, workers);
        } else if (outerAxis == 1) {
            jLo = chunkLo(worker, workers);
            jHi = chunkHi(worker, workers);
        } else {
            iLo = chunkLo(worker, workers);
            iHi = chunkHi(worker, workers);
        }
        for (int k = kLo; k <= kHi; k++) {
            for (int j = jLo; j <= jHi; j++) {
                int base = grid.index(0, j, k);
                for (int i = iLo; i <= iHi; i++) {
                    int idx = base + i;
                    loadNode(ds, v, idx);
                    for (int e = 0; e < nc; e++) {
                        int[] js = jSym[e];
                        double acc = 0;
                        for (int m = 0; m < js.length; m++) {
                            acc += jac[e][m][idx] * ds[js[m]];
                        }
                        if (timeDependent) {
                            int pc = eqs[e].principalComp;
                            if (timeOrder[pc] == 2) {
                                out[valVar[pc]][idx] = v[velVar[pc]][idx];
                                out[velVar[pc]][idx] = acc;
                            } else {
                                out[valVar[pc]][idx] = acc;
                            }
                        } else {
                            out[e][idx] = acc;
                        }
                    }
                }
            }
        }
    }

    final class LinearizeTask implements WorkerPool.Task {

        double t;
        double[][] y;

        @Override
        public void run(int worker, int workers) {
            linearizeSlab(worker, workers, t, y);
        }
    }

    final class JacApplyTask implements WorkerPool.Task {

        double[][] v;
        double[][] out;

        @Override
        public void run(int worker, int workers) {
            jacApplySlab(worker, workers, v, out);
        }
    }

    private final LinearizeTask linTask = new LinearizeTask();
    private final JacApplyTask jacTask = new JacApplyTask();

    /** Exact linearization of the right-hand side (or steady residual) about {@code y} (halos/BCs already applied). */
    void linearize(double t, double[][] y) {
        linTask.t = t;
        linTask.y = y;
        pool.run(linTask);
    }

    /** {@code out = F'(y_lin) v}; call {@link #linearize} first. */
    void jacApply(double[][] v, double[][] out) {
        jacTask.v = v;
        jacTask.out = out;
        pool.run(jacTask);
    }

    void close() {
        pool.close();
    }
}
