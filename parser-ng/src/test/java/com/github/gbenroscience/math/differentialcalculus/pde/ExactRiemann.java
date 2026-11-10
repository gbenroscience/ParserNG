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
 * Exact solution of the 1-D Euler Riemann problem for an ideal gas (Toro, "Riemann
 * Solvers and Numerical Methods for Fluid Dynamics", ch. 4): the reference the
 * shock-capturing tests are compared against.
 */
final class ExactRiemann {

    final double gamma;
    final double rL, uL, pL, rR, uR, pR;
    final double aL, aR;
    final double pStar, uStar;

    ExactRiemann(double gamma, double rL, double uL, double pL, double rR, double uR, double pR) {
        this.gamma = gamma;
        this.rL = rL;
        this.uL = uL;
        this.pL = pL;
        this.rR = rR;
        this.uR = uR;
        this.pR = pR;
        this.aL = Math.sqrt(gamma * pL / rL);
        this.aR = Math.sqrt(gamma * pR / rR);
        // Newton iteration for the star pressure from a two-rarefaction initial guess
        double g1 = (gamma - 1) / (2 * gamma);
        double p = Math.pow((aL + aR - 0.5 * (gamma - 1) * (uR - uL)) / (aL / Math.pow(pL, g1) + aR / Math.pow(pR, g1)), 1 / g1);
        for (int it = 0; it < 100; it++) {
            double f = fK(p, rL, pL, aL) + fK(p, rR, pR, aR) + (uR - uL);
            double df = dfK(p, rL, pL, aL) + dfK(p, rR, pR, aR);
            double pn = Math.max(1e-12, p - f / df);
            if (Math.abs(pn - p) < 1e-14 * (pn + p)) {
                p = pn;
                break;
            }
            p = pn;
        }
        this.pStar = p;
        this.uStar = 0.5 * (uL + uR) + 0.5 * (fK(p, rR, pR, aR) - fK(p, rL, pL, aL));
    }

    private double fK(double p, double rK, double pK, double aK) {
        if (p > pK) {
            double a = 2 / ((gamma + 1) * rK);
            double b = (gamma - 1) / (gamma + 1) * pK;
            return (p - pK) * Math.sqrt(a / (p + b));
        }
        return 2 * aK / (gamma - 1) * (Math.pow(p / pK, (gamma - 1) / (2 * gamma)) - 1);
    }

    private double dfK(double p, double rK, double pK, double aK) {
        if (p > pK) {
            double a = 2 / ((gamma + 1) * rK);
            double b = (gamma - 1) / (gamma + 1) * pK;
            return Math.sqrt(a / (b + p)) * (1 - 0.5 * (p - pK) / (b + p));
        }
        return 1 / (rK * aK) * Math.pow(p / pK, -(gamma + 1) / (2 * gamma));
    }

    /** @return {density, velocity, pressure} at similarity coordinate {@code s = (x - x0)/t}. */
    double[] sample(double s) {
        double g = gamma;
        double[] out = new double[3];
        if (s <= uStar) { // left of the contact
            if (pStar > pL) { // left shock
                double sL = uL - aL * Math.sqrt((g + 1) / (2 * g) * pStar / pL + (g - 1) / (2 * g));
                if (s <= sL) {
                    out[0] = rL;
                    out[1] = uL;
                    out[2] = pL;
                } else {
                    double q = pStar / pL;
                    out[0] = rL * (q + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * q + 1);
                    out[1] = uStar;
                    out[2] = pStar;
                }
            } else { // left rarefaction
                double shL = uL - aL;
                double aStarL = aL * Math.pow(pStar / pL, (g - 1) / (2 * g));
                double stL = uStar - aStarL;
                if (s <= shL) {
                    out[0] = rL;
                    out[1] = uL;
                    out[2] = pL;
                } else if (s > stL) {
                    out[0] = rL * Math.pow(pStar / pL, 1 / g);
                    out[1] = uStar;
                    out[2] = pStar;
                } else {
                    double c = 2 / (g + 1) * (aL + (g - 1) / 2 * (uL - s));
                    out[1] = 2 / (g + 1) * (aL + (g - 1) / 2 * uL + s);
                    out[0] = rL * Math.pow(c / aL, 2 / (g - 1));
                    out[2] = pL * Math.pow(c / aL, 2 * g / (g - 1));
                }
            }
        } else { // right of the contact
            if (pStar > pR) { // right shock
                double sR = uR + aR * Math.sqrt((g + 1) / (2 * g) * pStar / pR + (g - 1) / (2 * g));
                if (s >= sR) {
                    out[0] = rR;
                    out[1] = uR;
                    out[2] = pR;
                } else {
                    double q = pStar / pR;
                    out[0] = rR * (q + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * q + 1);
                    out[1] = uStar;
                    out[2] = pStar;
                }
            } else { // right rarefaction
                double shR = uR + aR;
                double aStarR = aR * Math.pow(pStar / pR, (g - 1) / (2 * g));
                double stR = uStar + aStarR;
                if (s >= shR) {
                    out[0] = rR;
                    out[1] = uR;
                    out[2] = pR;
                } else if (s < stR) {
                    out[0] = rR * Math.pow(pStar / pR, 1 / g);
                    out[1] = uStar;
                    out[2] = pStar;
                } else {
                    double c = 2 / (g + 1) * (aR + (g - 1) / 2 * (s - uR));
                    out[1] = 2 / (g + 1) * (-aR + (g - 1) / 2 * uR + s);
                    out[0] = rR * Math.pow(c / aR, 2 / (g - 1));
                    out[2] = pR * Math.pow(c / aR, 2 * g / (g - 1));
                }
            }
        }
        return out;
    }

    /** @return speed of the right-going shock (valid when pStar > pR). */
    double rightShockSpeed() {
        return uR + aR * Math.sqrt((gamma + 1) / (2 * gamma) * pStar / pR + (gamma - 1) / (2 * gamma));
    }
}
