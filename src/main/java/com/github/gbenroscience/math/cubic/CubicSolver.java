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
package com.github.gbenroscience.math.cubic;

import com.github.gbenroscience.parser.MathExpression;

/**
 *
 * @author GBEMIRO
 */
public class CubicSolver {
    private final double a, b, c, d;
    private final double[] roots = new double[3];
    private int realRootCount = 0;

    public CubicSolver(double a, double b, double c, double d) {
        this.a = a; this.b = b; this.c = c; this.d = d;
        solve();
    }

    private void solve() {
        if (Math.abs(a) < 1e-12) return; // Should be handled as quadratic

        // 1. Convert to Depressed Cubic: t^3 + pt + q = 0
        // Formula: x = t - b/(3a)
        double p = (3 * a * c - b * b) / (3 * a * a);
        double q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a);
        
        // 2. Calculate the Discriminant (Cardano's Discriminant)
        double discriminant = (q * q / 4.0) + (p * p * p / 27.0);

        if (discriminant > 0) {
            // Case 1: 1 Real Root, 2 Complex
            double sqrtD = Math.sqrt(discriminant);
            double u = Math.cbrt(-q / 2.0 + sqrtD);
            double v = Math.cbrt(-q / 2.0 - sqrtD);
            
            roots[0] = (u + v) - b / (3 * a);
            realRootCount = 1;
        } else if (discriminant == 0) {
            // Case 2: Real Roots (some multiple)
            double u = Math.cbrt(-q / 2.0);
            roots[0] = 2 * u - b / (3 * a);
            roots[1] = -u - b / (3 * a);
            realRootCount = 2; // (One is a double root)
        } else {
            // Case 3: 3 Distinct Real Roots (Casus Irreducibilis)
            // Use trigonometric approach to avoid complex radicals
            double r = Math.sqrt(-(p * p * p) / 27.0);
            double phi = Math.acos(-q / (2.0 * r));
            double s = 2.0 * Math.pow(r, 1.0/3.0);

            roots[0] = s * Math.cos(phi / 3.0) - b / (3 * a);
            roots[1] = s * Math.cos((phi + 2 * Math.PI) / 3.0) - b / (3 * a);
            roots[2] = s * Math.cos((phi + 4 * Math.PI) / 3.0) - b / (3 * a);
            realRootCount = 3;
        }
    }

    public void wrapInResult(MathExpression.EvalResult out) {
        double[] result = new double[realRootCount];
        System.arraycopy(roots, 0, result, 0, realRootCount);
        out.wrap(result); // Wraps as a Vector
    }
}