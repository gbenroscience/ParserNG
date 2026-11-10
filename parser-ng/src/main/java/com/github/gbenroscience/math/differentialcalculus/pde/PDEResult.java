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
import java.util.Collections;
import java.util.List;

/**
 * Snapshot of a solution on the grid, plus diagnostics. Field arrays hold the
 * real grid nodes only (no halo), x-fastest:
 * {@code values[i + n0*(j + n1*k)]}.
 */
public final class PDEResult {

    /** Analytical reference solution {@code f(x,y,z,t)} used by the error helpers. */
    @FunctionalInterface
    public interface Exact {

        double at(double x, double y, double z, double t);
    }

    private final Grid grid;
    private final List<String> names;
    private final double[][] fields;
    private final double time;
    private final SolverDiagnostics diagnostics;

    PDEResult(Grid grid, String[] names, double[][] fields, double time, SolverDiagnostics diagnostics) {
        this.grid = grid;
        this.names = Collections.unmodifiableList(Arrays.asList(names.clone()));
        this.fields = fields;
        this.time = time;
        this.diagnostics = diagnostics;
    }

    public Grid getGrid() {
        return grid;
    }

    /** @return names of the unknowns, in field order. */
    public List<String> getUnknowns() {
        return names;
    }

    /** @return the time of this snapshot (the pseudo-time is 0 for steady problems). */
    public double getTime() {
        return time;
    }

    public SolverDiagnostics getDiagnostics() {
        return diagnostics;
    }

    /** @return true when the run completed without any non-convergence. */
    public boolean isConverged() {
        return diagnostics.isConverged();
    }

    /** @return the (live) node values of an unknown. */
    public double[] values(String unknown) {
        return fields[indexOf(unknown)];
    }

    /** @return the value of an unknown at grid node {@code (i,j,k)}. */
    public double value(String unknown, int i, int j, int k) {
        return fields[indexOf(unknown)][i + grid.n(0) * (j + grid.n(1) * k)];
    }

    /** 1-D convenience for {@link #value(String, int, int, int)}. */
    public double value(String unknown, int i) {
        return value(unknown, i, 0, 0);
    }

    /** @return the multilinear interpolant of an unknown at the given point (clamped to the domain). */
    public double valueAt(String unknown, double x, double y, double z) {
        double[] f = fields[indexOf(unknown)];
        double[] p = {x, y, z};
        int[] i0 = new int[3];
        double[] w = new double[3];
        for (int a = 0; a < 3; a++) {
            if (a >= grid.dim()) {
                i0[a] = 0;
                w[a] = 0;
                continue;
            }
            double s = (p[a] - grid.min(a)) / grid.h(a);
            s = Math.max(0, Math.min(grid.n(a) - 1, s));
            int i = (int) Math.floor(s);
            if (i >= grid.n(a) - 1) {
                i = grid.n(a) - 2;
            }
            i0[a] = i;
            w[a] = s - i;
        }
        double sum = 0;
        for (int c = 0; c < 8; c++) {
            double wt = 1;
            int[] ii = new int[3];
            boolean skip = false;
            for (int a = 0; a < 3; a++) {
                int bit = (c >> a) & 1;
                if (a >= grid.dim()) {
                    if (bit == 1) {
                        skip = true;
                    }
                    ii[a] = 0;
                    continue;
                }
                wt *= bit == 1 ? w[a] : 1 - w[a];
                ii[a] = i0[a] + bit;
            }
            if (!skip && wt != 0) {
                sum += wt * f[ii[0] + grid.n(0) * (ii[1] + grid.n(1) * ii[2])];
            }
        }
        return sum;
    }

    /** @return max-norm of {@code u - exact} over all nodes. */
    public double maxError(String unknown, Exact exact) {
        double[] f = fields[indexOf(unknown)];
        double m = 0;
        for (int k = 0; k < grid.n(2); k++) {
            for (int j = 0; j < grid.n(1); j++) {
                for (int i = 0; i < grid.n(0); i++) {
                    double e = Math.abs(f[i + grid.n(0) * (j + grid.n(1) * k)]
                            - exact.at(grid.coord(0, i), grid.coord(1, j), grid.coord(2, k), time));
                    if (e > m || Double.isNaN(e)) {
                        m = e;
                    }
                }
            }
        }
        return m;
    }

    /** @return the discrete RMS norm of {@code u - exact}. */
    public double rmsError(String unknown, Exact exact) {
        double[] f = fields[indexOf(unknown)];
        double s = 0;
        for (int k = 0; k < grid.n(2); k++) {
            for (int j = 0; j < grid.n(1); j++) {
                for (int i = 0; i < grid.n(0); i++) {
                    double e = f[i + grid.n(0) * (j + grid.n(1) * k)]
                            - exact.at(grid.coord(0, i), grid.coord(1, j), grid.coord(2, k), time);
                    s += e * e;
                }
            }
        }
        return Math.sqrt(s / grid.nodeCount());
    }

    private int indexOf(String unknown) {
        int i = names.indexOf(unknown);
        if (i < 0) {
            throw new PDEException("Unknown field '" + unknown + "'; available: " + names);
        }
        return i;
    }

    @Override
    public String toString() {
        return "PDEResult{unknowns=" + names + ", t=" + time + ", " + grid + ", " + diagnostics + "}";
    }
}
