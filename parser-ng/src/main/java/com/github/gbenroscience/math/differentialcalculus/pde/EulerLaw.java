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

/**
 * Compressible Euler equations of an ideal gas with ratio of specific heats
 * {@code gamma}: conserved variables (density, momentum components, total
 * energy), primitive variables (density, velocity components, pressure):
 * <pre>
 *   rho_t + div(rho u) = 0,   (rho u)_t + div(rho u u + p I) = 0,   E_t + div((E + p) u) = 0,
 *   p = (gamma - 1) (E - rho |u|^2 / 2)
 * </pre>
 * Provides exact wave speeds, an HLLC flux (Toro, with Einfeldt/Roe wave-speed
 * estimates) and a reflecting wall.
 */
public final class EulerLaw implements ConservationLaw {

    private final int dim;
    private final double gamma;
    private final double gm1;
    private final String[] names;

    public EulerLaw(int dimension, double gamma) {
        if (dimension < 1 || dimension > 3) {
            throw new PDEConfigurationException("Euler dimension must be 1, 2 or 3, got " + dimension);
        }
        if (!(gamma > 1.0)) {
            throw new PDEConfigurationException("gamma must exceed 1, got " + gamma);
        }
        this.dim = dimension;
        this.gamma = gamma;
        this.gm1 = gamma - 1.0;
        String[] nm = {"rho", "mx", "my", "mz"};
        this.names = new String[dim + 2];
        names[0] = nm[0];
        for (int d = 0; d < dim; d++) {
            names[1 + d] = nm[1 + d];
        }
        names[dim + 1] = "E";
    }

    public double gamma() {
        return gamma;
    }

    @Override
    public int components() {
        return dim + 2;
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public String[] names() {
        return names.clone();
    }

    /** @return pressure of a conserved state. */
    public double pressure(double[] u) {
        double ke = 0;
        for (int d = 0; d < dim; d++) {
            ke += u[1 + d] * u[1 + d];
        }
        return gm1 * (u[dim + 1] - 0.5 * ke / u[0]);
    }

    @Override
    public void flux(int dir, double[] u, double[] f) {
        double rho = u[0];
        double vn = u[1 + dir] / rho;
        double p = pressure(u);
        f[0] = u[1 + dir];
        for (int d = 0; d < dim; d++) {
            f[1 + d] = u[1 + d] * vn;
        }
        f[1 + dir] += p;
        f[dim + 1] = (u[dim + 1] + p) * vn;
    }

    @Override
    public void waveSpeeds(int dir, double[] u, double[] out) {
        double rho = u[0];
        double vn = u[1 + dir] / rho;
        double p = pressure(u);
        double c = Math.sqrt(Math.max(0.0, gamma * p / rho));
        out[0] = vn - c;
        out[1] = vn + c;
    }

    @Override
    public boolean isPhysical(double[] u) {
        if (!(u[0] > 0)) {
            return false;
        }
        return pressure(u) > 0;
    }

    @Override
    public boolean usesPrimitiveReconstruction() {
        return true;
    }

    @Override
    public void toPrimitive(double[] u, double[] w) {
        double rho = u[0];
        w[0] = rho;
        double ke = 0;
        for (int d = 0; d < dim; d++) {
            double v = u[1 + d] / rho;
            w[1 + d] = v;
            ke += u[1 + d] * v;
        }
        w[dim + 1] = gm1 * (u[dim + 1] - 0.5 * ke);
    }

    @Override
    public void toConserved(double[] w, double[] u) {
        double rho = w[0];
        u[0] = rho;
        double ke = 0;
        for (int d = 0; d < dim; d++) {
            u[1 + d] = rho * w[1 + d];
            ke += rho * w[1 + d] * w[1 + d];
        }
        u[dim + 1] = w[dim + 1] / gm1 + 0.5 * ke;
    }

    @Override
    public boolean supportsHllc() {
        return true;
    }

    @Override
    public void hllc(int dir, double[] uL, double[] uR, double[] f) {
        double rL = uL[0];
        double rR = uR[0];
        double vL = uL[1 + dir] / rL;
        double vR = uR[1 + dir] / rR;
        double pL = pressure(uL);
        double pR = pressure(uR);
        double cL = Math.sqrt(Math.max(0.0, gamma * pL / rL));
        double cR = Math.sqrt(Math.max(0.0, gamma * pR / rR));
        // Roe averages for the Einfeldt estimates
        double sqL = Math.sqrt(rL);
        double sqR = Math.sqrt(rR);
        double wL = sqL / (sqL + sqR);
        double wR = 1.0 - wL;
        double vRoe = wL * vL + wR * vR;
        double hL = (uL[dim + 1] + pL) / rL;
        double hR = (uR[dim + 1] + pR) / rR;
        double hRoe = wL * hL + wR * hR;
        double q2 = 0;
        for (int d = 0; d < dim; d++) {
            double a = wL * (uL[1 + d] / rL) + wR * (uR[1 + d] / rR);
            q2 += a * a;
        }
        double cRoe = Math.sqrt(Math.max(1e-300, gm1 * (hRoe - 0.5 * q2)));
        double sL = Math.min(vL - cL, vRoe - cRoe);
        double sR = Math.max(vR + cR, vRoe + cRoe);
        if (sL >= 0) {
            flux(dir, uL, f);
            return;
        }
        if (sR <= 0) {
            flux(dir, uR, f);
            return;
        }
        double sStar = (pR - pL + rL * vL * (sL - vL) - rR * vR * (sR - vR)) / (rL * (sL - vL) - rR * (sR - vR));
        double[] uk;
        double rk;
        double vk;
        double pk;
        double sk;
        if (sStar >= 0) {
            uk = uL;
            rk = rL;
            vk = vL;
            pk = pL;
            sk = sL;
        } else {
            uk = uR;
            rk = rR;
            vk = vR;
            pk = pR;
            sk = sR;
        }
        // F = F_K + S_K (U*_K - U_K); U*_K computed component by component without temporaries
        double fac = rk * (sk - vk) / (sk - sStar);
        double ek = uk[dim + 1];
        double estar = fac * (ek / rk + (sStar - vk) * (sStar + pk / (rk * (sk - vk))));
        flux(dir, uk, f);
        f[0] += sk * (fac - uk[0]);
        for (int d = 0; d < dim; d++) {
            double ustar = d == dir ? fac * sStar : fac * (uk[1 + d] / rk);
            f[1 + d] += sk * (ustar - uk[1 + d]);
        }
        f[dim + 1] += sk * (estar - ek);
    }

    @Override
    public void reflect(int dir, double[] u) {
        u[1 + dir] = -u[1 + dir];
    }
}
