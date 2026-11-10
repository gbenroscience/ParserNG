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

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

/**
 * The semi-discrete system {@code y' = F(t, y)} of a time-dependent PDE
 * (method of lines), expressed in the calling convention of ParserNG's ODE
 * integrators: a {@link MethodHandle} of type {@code void f(double[] vars,
 * double[] out)} where {@code vars[0] = t} and {@code vars[1..size]} is the
 * state. The state holds the values at the PDE (non-constrained) grid nodes of
 * every state variable, state variable by state variable; boundary nodes are
 * recomputed from the boundary conditions inside every evaluation.
 * <p>
 * Use it to integrate a PDE with {@code DifferentialEquations.stepRK4}, the
 * adaptive RK45, and the other existing ODE solvers: pass
 * {@code tSlot() = 0}, {@code ySlotStart() = 1}, {@code systemSize() = size()}
 * and {@code frameSize()}. Note that those integrators allocate per step; the
 * engine's own integrators do not.
 */
public final class MethodOfLinesSystem {

    private final Model m;
    private final Boundaries bc;
    private final double[][] live;
    private final double[][] work;
    private final double[][] out;
    private final int[] pdeIdx;
    private final int np;
    private final MethodHandle handle;

    MethodOfLinesSystem(Model m, Boundaries bc, double[][] live) {
        this.m = m;
        this.bc = bc;
        this.live = live;
        int size = m.grid.storageSize();
        this.work = new double[m.nState][size];
        this.out = new double[m.nState][size];
        int cnt = 0;
        Grid g = m.grid;
        int[] tmp = new int[g.nodeCount()];
        for (int k = m.pdeLo[2]; k <= m.pdeHi[2]; k++) {
            for (int j = m.pdeLo[1]; j <= m.pdeHi[1]; j++) {
                for (int i = m.pdeLo[0]; i <= m.pdeHi[0]; i++) {
                    tmp[cnt++] = g.index(i, j, k);
                }
            }
        }
        this.np = cnt;
        this.pdeIdx = java.util.Arrays.copyOf(tmp, cnt);
        try {
            this.handle = MethodHandles.lookup().findVirtual(MethodOfLinesSystem.class, "eval",
                    MethodType.methodType(void.class, double[].class, double[].class)).bindTo(this);
        } catch (ReflectiveOperationException ex) {
            throw new PDEException("Cannot create method handle", ex);
        }
    }

    /** @return number of ODE state variables. */
    public int size() {
        return np * m.nState;
    }

    /** @return frame slot of t (0). */
    public int tSlot() {
        return 0;
    }

    /** @return frame slot where the state starts (1). */
    public int ySlotStart() {
        return 1;
    }

    /** @return {@code 1 + size()}. */
    public int frameSize() {
        return 1 + size();
    }

    /** @return the current PDE state as an ODE initial vector (a fresh array). */
    public double[] initialState() {
        double[] z = new double[size()];
        for (int s = 0; s < m.nState; s++) {
            for (int p = 0; p < np; p++) {
                z[s * np + p] = live[s][pdeIdx[p]];
            }
        }
        return z;
    }

    /** Evaluates {@code out = F(vars[0], vars[1..])}. */
    public void eval(double[] vars, double[] dydt) {
        double t = vars[0];
        for (int s = 0; s < m.nState; s++) {
            double[] w = work[s];
            java.util.Arrays.fill(w, 0.0);
            for (int p = 0; p < np; p++) {
                w[pdeIdx[p]] = vars[1 + s * np + p];
            }
        }
        bc.apply(work, t);
        m.rhs(t, work, out);
        for (int s = 0; s < m.nState; s++) {
            for (int p = 0; p < np; p++) {
                dydt[s * np + p] = out[s][pdeIdx[p]];
            }
        }
    }

    /** @return handle of type {@code void (double[] vars, double[] out)}. */
    public MethodHandle handle() {
        return handle;
    }

    /**
     * Writes an ODE state vector back into the PDE state arrays of the owning
     * solver (and applies the boundary conditions at time {@code t}).
     */
    public void load(double t, double[] z) {
        for (int s = 0; s < m.nState; s++) {
            for (int p = 0; p < np; p++) {
                live[s][pdeIdx[p]] = z[s * np + p];
            }
        }
        bc.apply(live, t);
    }

    /** @return the flat storage index of the p-th PDE node. */
    public int nodeIndex(int p) {
        return pdeIdx[p];
    }
}
