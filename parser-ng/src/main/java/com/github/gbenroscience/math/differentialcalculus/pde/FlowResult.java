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
 * An independent snapshot of a flow field for post-processing: velocity on the
 * staggered faces (with the boundary ghost layer, so interpolation up to a wall
 * passes through the wall value), cell-centred pressure, and derived fields.
 */
public final class FlowResult {

    private final int dim;
    private final int[] n;
    private final double[] h;
    private final double[] origin;
    private final Box[] ub;
    private final Box pb;
    private final double[][] u;
    private final double[] p;
    private final double rho;

    FlowResult(IncompressibleFlow f) {
        this.dim = f.dimension();
        this.n = f.gridCells().clone();
        this.h = f.gridSpacing().clone();
        this.origin = f.gridOrigin().clone();
        this.ub = f.velocityBoxes().clone();
        this.pb = f.pressureBox();
        this.rho = f.density();
        this.u = new double[dim][];
        for (int c = 0; c < dim; c++) {
            u[c] = f.velocityArrays()[c].clone();
        }
        this.p = f.pressureKinematic().clone();
    }

    public int dimension() {
        return dim;
    }

    public int cells(int axis) {
        return n[axis];
    }

    public double spacing(int axis) {
        return h[axis];
    }

    /** @return the velocity component {@code c} at a point, by linear interpolation of the staggered field. */
    public double velocityAt(int c, double x, double y, double z) {
        double[] pos = {x, y, z};
        int[] i0 = new int[3];
        double[] w = new double[3];
        for (int a = 0; a < dim; a++) {
            double s = (pos[a] - origin[a]) / h[a] - (a == c ? 0.0 : 0.5);
            int lo = (int) Math.floor(s);
            int min = -1;
            int max = (a == c ? n[a] : n[a] - 1);      // last index with a right neighbour inside the ghost layer
            if (lo < min) {
                lo = min;
            }
            if (lo > max) {
                lo = max;
            }
            i0[a] = lo;
            w[a] = s - lo;
        }
        double sum = 0;
        int corners = 1 << dim;
        for (int cn = 0; cn < corners; cn++) {
            double wt = 1;
            int[] ii = new int[3];
            for (int a = 0; a < 3; a++) {
                int bit = a < dim ? (cn >> a) & 1 : 0;
                if (a < dim) {
                    wt *= bit == 1 ? w[a] : 1 - w[a];
                }
                ii[a] = i0[a] + bit;
            }
            sum += wt * u[c][ub[c].idx(ii[0], ii[1], ii[2])];
        }
        return sum;
    }

    /** @return kinematic pressure {@code p/rho} of cell (i,j,k). */
    public double kinematicPressure(int i, int j, int k) {
        return p[pb.idx(i, j, k)];
    }

    /** @return physical pressure of cell (i,j,k). */
    public double pressure(int i, int j, int k) {
        return rho * p[pb.idx(i, j, k)];
    }

    /** @return the cell-centred velocity component {@code c} of cell (i,j,k). */
    public double cellVelocity(int c, int i, int j, int k) {
        int q = ub[c].idx(i, j, k);
        return 0.5 * (u[c][q] + u[c][q + ub[c].st[c]]);
    }

    /** @return the z-vorticity {@code dv/dx - du/dy} at an interior cell centre of a 2-D flow (central differences). */
    public double vorticity2D(int i, int j) {
        return (cellVelocity(1, i + 1, j, 0) - cellVelocity(1, i - 1, j, 0)) / (2 * h[0])
                - (cellVelocity(0, i, j + 1, 0) - cellVelocity(0, i, j - 1, 0)) / (2 * h[1]);
    }

    /**
     * @return the stream function psi (with {@code u = dpsi/dy, v = -dpsi/dx}, psi = 0 at the origin
     * corner) at the cell CORNERS of a 2-D flow, as an array {@code psi[i + (n0+1)*j]}; the
     * extremum of psi locates the primary vortex.
     */
    public double[] streamFunction2D() {
        int nx = n[0];
        int ny = n[1];
        double[] psi = new double[(nx + 1) * (ny + 1)];
        for (int j = 1; j <= ny; j++) {
            for (int i = 0; i <= nx; i++) {
                // psi(i,j) = psi(i,j-1) + u(i, j-1/2) dy
                psi[i + (nx + 1) * j] = psi[i + (nx + 1) * (j - 1)] + u[0][ub[0].idx(i, j - 1, 0)] * h[1];
            }
        }
        return psi;
    }

    @Override
    public String toString() {
        return "FlowResult{" + dim + "D, cells=" + Arrays.toString(Arrays.copyOf(n, dim)) + "}";
    }
}
