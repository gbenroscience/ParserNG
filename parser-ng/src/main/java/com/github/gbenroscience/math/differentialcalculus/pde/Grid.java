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
 * Uniform tensor-product grid of 1, 2 or 3 spatial dimensions, with a halo
 * (ghost layer) of width {@link #halo()} around every ACTIVE axis.
 * <p>
 * Node {@code (i,j,k)} sits at {@code (min[0]+i*h[0], min[1]+j*h[1], min[2]+k*h[2])}
 * with {@code i = 0..n[0]-1}; both end points of every axis are grid nodes.
 * Unused axes have exactly one node and no halo.
 * <p>
 * Fields are stored in ONE flat {@code double[]} per unknown, x-fastest:
 * <pre>index(i,j,k) = (i + hx) + nxp * ((j + hy) + nyp * (k + hz))</pre>
 * where {@code nxp = n[0] + 2*hx} etc. Use {@link #index(int, int, int)} instead
 * of re-deriving the formula.
 */
public final class Grid {

    private final int dim;
    private final String[] names;
    private final double[] min;
    private final double[] max;
    private final int[] n;
    private final double[] h;
    private final int halo;
    private final int[] pad;      // halo width per axis (0 for inactive axes)
    private final int[] padded;   // padded node count per axis
    private final int[] stride;
    private final int size;
    private final int nodes;

    Grid(int dim, String[] names, double[] min, double[] max, int[] n, int halo) {
        this.dim = dim;
        this.names = Arrays.copyOf(names, 3);
        this.min = Arrays.copyOf(min, 3);
        this.max = Arrays.copyOf(max, 3);
        this.n = new int[]{1, 1, 1};
        this.h = new double[3];
        this.halo = halo;
        this.pad = new int[3];
        this.padded = new int[3];
        for (int a = 0; a < 3; a++) {
            if (a < dim) {
                this.n[a] = n[a];
                this.h[a] = (max[a] - min[a]) / (n[a] - 1);
                this.pad[a] = halo;
            } else {
                this.n[a] = 1;
                this.min[a] = 0;
                this.max[a] = 0;
                this.h[a] = 1;
                this.pad[a] = 0;
            }
            this.padded[a] = this.n[a] + 2 * this.pad[a];
        }
        this.stride = new int[]{1, padded[0], padded[0] * padded[1]};
        this.size = padded[0] * padded[1] * padded[2];
        this.nodes = this.n[0] * this.n[1] * this.n[2];
    }

    /** @return number of active spatial dimensions (1, 2 or 3). */
    public int dim() {
        return dim;
    }

    /** @return name of axis {@code a} (e.g. "x"), or null for an inactive axis. */
    public String name(int a) {
        return a < dim ? names[a] : null;
    }

    /** @return number of nodes along axis {@code a} (1 for inactive axes). */
    public int n(int a) {
        return n[a];
    }

    /** @return lower coordinate of axis {@code a}. */
    public double min(int a) {
        return min[a];
    }

    /** @return upper coordinate of axis {@code a}. */
    public double max(int a) {
        return max[a];
    }

    /** @return node spacing along axis {@code a}. */
    public double h(int a) {
        return h[a];
    }

    /** @return coordinate of node {@code i} along axis {@code a}. */
    public double coord(int a, int i) {
        return min[a] + i * h[a];
    }

    /** @return halo width (identical for every active axis). */
    public int halo() {
        return halo;
    }

    /** @return halo width of axis {@code a} (0 for inactive axes). */
    public int pad(int a) {
        return pad[a];
    }

    /** @return flat-array stride of axis {@code a}. */
    public int stride(int a) {
        return stride[a];
    }

    /** @return length of the flat (halo-padded) storage array of one field. */
    public int storageSize() {
        return size;
    }

    /** @return number of real grid nodes, {@code n0*n1*n2}. */
    public int nodeCount() {
        return nodes;
    }

    /** @return flat storage index of node {@code (i,j,k)}; halo nodes use negative / overflowing indices. */
    public int index(int i, int j, int k) {
        return (i + pad[0]) + padded[0] * ((j + pad[1]) + padded[1] * (k + pad[2]));
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Grid[");
        for (int a = 0; a < dim; a++) {
            if (a > 0) {
                sb.append(" x ");
            }
            sb.append(names[a]).append(" in [").append(min[a]).append(',').append(max[a]).append("] n=").append(n[a]);
        }
        return sb.append(", halo=").append(halo).append(']').toString();
    }
}
