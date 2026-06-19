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
package com.github.gbenroscience.math.numericalmethods;

import com.github.gbenroscience.parser.Function;

/**
 *
 *
 * How to use these in your Platform When you want the derivative of a badly
 * behaved function at a point : Map x to u using toChebyshev(x). Compute the
 * derivative of the Chebyshev series at (standard algorithm). Multiply that
 * result by map.derivativeFactor(u). When you want the integral: Sample f(x)
 *
 *
 * at mapped nodes. Multiply by map.dx_du(u). Sum using Clenshaw-Curtis weights.
 * Pro-Tip for Precision In the SemiInfiniteMap, as * , *
 *
 * . In your code, you should clamp to something like 0.9999999999 if you hit a
 * NaN in the physical domain to prevent the platform from crashing on a hard
 * infinity. Would you like the Clenshaw-Curtis weight generator to ensure your
 * integration reaches full 16-digit precision?
 *
 *
 *
 * @author GBEMIRO
 */
public class MappedExpander {

    private final double[] sampledValues; // Pre-sampled at Extrema nodes
    private final double[] coefficients;
    private final DomainMap map;
    public static final int MAX_DEPTH = 25;
    private final int N;

    /**
     * Uses Chebyshev Extrema nodes: u_k = cos(k * PI / N) for k = 0...N. This
     * matches Clenshaw-Curtis weights perfectly.
     */
    public MappedExpander(Function function, DomainMap map, int N) {
        this.map = map;
        this.N = N;
        this.sampledValues = new double[N + 1];
        this.coefficients = new double[N + 1];

        // 1. Sample at Extrema nodes (including boundaries -1 and 1)
        for (int k = 0; k <= N; k++) {
            double u = Math.cos((k * Math.PI) / N);
            double x = map.toPhysical(u);

            function.updateArgs(x);
            double val = function.calc();

            // HARDENING: Avoid zeroing out singularities. 
            // If NaN/Inf, sample slightly inside the domain instead.
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                double eps = 1e-14 * (k < N / 2 ? 1 : -1);
                function.updateArgs(map.toPhysical(u + eps));
                val = function.calc();
            }
            sampledValues[k] = val;
        }

        // 2. Compute coefficients using DCT-I logic
        for (int j = 0; j <= N; j++) {
            double sum = 0.5 * (sampledValues[0] + (j % 2 == 0 ? 1.0 : -1.0) * sampledValues[N]);
            for (int k = 1; k < N; k++) {
                sum += sampledValues[k] * Math.cos((j * k * Math.PI) / N);
            }
            this.coefficients[j] = (j == 0 || j == N ? 1.0 / N : 2.0 / N) * sum;
        }
    }

    public double getTailError() {
        int n = coefficients.length;
        if (n < 8) {
            return Double.MAX_VALUE; // Not enough data to judge
        }
        // Hardened Estimate: Look at the last few coefficients.
        // In a converged series, these should be near machine epsilon (~1e-16).
        double tailSum = 0.0;
        int tailSize = Math.min(8, n / 4); // Check the last 8 or 25% of coefficients

        for (int i = n - tailSize; i < n; i++) {
            tailSum += Math.abs(coefficients[i]);
        }

        // Return the average magnitude of the tail.
        // If this is > tolerance, the adaptive logic will trigger a subdivision.
        return tailSum / tailSize;
    }

    public double evaluate(double x) {
        double u = map.toChebyshev(x);
        // Standard Clenshaw evaluation logic...
        return clenshaw(u);
    }

    public double integrate() {
        int n = coefficients.length;
        double totalIntegral = 0.0;

        // We integrate over the standard Chebyshev interval [-1, 1]
        // using the sampled nodes to account for the mapping "stretch"
        for (int k = 1; k <= n; k++) {
            double u = Math.cos(Math.PI * (2.0 * k - 1.0) / (2.0 * n));

            // 1. Evaluate the approximated function at u
            double f_val = evaluate(map.toPhysical(u));

            // 2. Multiply by the "Stretch Factor" (Jacobian)
            double weight = map.dx_du(u);

            // 3. Apply Clenshaw-Curtis weights (simplified here as a sum)
            // For higher precision, use your existing Kahan summation logic
            totalIntegral += f_val * weight * (Math.PI / n) * Math.sqrt(1 - u * u);
        }

        return totalIntegral;
    }

    public double integrateSeamless() {
        int n = coefficients.length;
        double sum = 0.0;
        double compensation = 0.0; // For Kahan Summation

        // Clenshaw-Curtis Quadrature approach
        // We iterate through the Chebyshev nodes
        for (int k = 1; k <= n; k++) {
            double u = Math.cos(Math.PI * (2.0 * k - 1.0) / (2.0 * n));

            // 1. Get the physical value and the stretch factor (Jacobian)
            double x = map.toPhysical(u);
            double stretch = map.dx_du(u);

            // 2. Evaluate the function at the mapped point
            double fx = evaluate(x);

            // 3. Compute the contribution: f(x) * dx/du * weight
            // The standard Chebyshev weight is (PI/n) * sqrt(1 - u^2)
            double weight = (Math.PI / n) * Math.sqrt(1.0 - u * u);
            double term = fx * stretch * weight;

            // 4. Kahan Summation to prevent floating-point drift
            double y = term - compensation;
            double t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }

        return sum;
    }

    /**
     * O(N) Integration: Direct sum using pre-sampled values and CC weights.
     */
    public double integrateFinal(double[] ccWeights) {
        if (ccWeights.length != sampledValues.length) {
            throw new IllegalArgumentException("Weight array size must match node count N+1");
        }

        double sum = 0.0;
        double compensation = 0.0; // Kahan Summation

        for (int k = 0; k <= N; k++) {
            double u = Math.cos((k * Math.PI) / N);

            // Direct access to sampledValues[k] eliminates the O(N) evaluate() call
            double term = sampledValues[k] * map.dx_du(u) * ccWeights[k];

            // Kahan Summation Logic
            double y = term - compensation;
            double t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    public double integrateAdaptive(Function f, DomainMap map, double tol, int depth) {
        MappedExpander expander = new MappedExpander(f, map, 256);
        double errorEstimate = expander.getTailError();

        if (errorEstimate > tol && depth < MAX_DEPTH) {
            DomainMap leftHalf = new SubDomainMap(map, -1.0, 0.0);
            DomainMap rightHalf = new SubDomainMap(map, 0.0, 1.0);

            return integrateAdaptive(f, leftHalf, tol / 2.0, depth + 1)
                    + integrateAdaptive(f, rightHalf, tol / 2.0, depth + 1);
        }

        // UPDATED REFERENCE
        return expander.integrateFinal(CCWeightGenerator.getCachedWeights());
    }

    public boolean isAliasing() {
        int n = coefficients.length;
        // Look at the last 10% of the coefficients
        int checkZone = Math.max(5, n / 10);
        double highFreqEnergy = 0;
        double totalEnergy = 0;

        for (int i = 0; i < n; i++) {
            double absC = Math.abs(coefficients[i]);
            totalEnergy += absC;
            if (i > n - checkZone) {
                highFreqEnergy += absC;
            }
        }

        // Safety Valve: If more than 5% of the "information" is in the 
        // highest frequencies, the function is oscillating too fast.
        return (highFreqEnergy / totalEnergy) > 0.05;
    }

    /**
     * Heuristic to detect singularities (logarithmic or power-law) at an
     * endpoint. point: the boundary (e.g., a) direction: 1.0 for right-side
     * (a+), -1.0 for left-side (b-)
     */
    private boolean isLogarithmicSingularity(Function f, double point, double direction) {
        double eps1 = 1e-7;
        double eps2 = 1e-8;
        double eps3 = 1e-9;

        f.updateArgs(point + direction * eps1);
        double v1 = Math.abs(f.calc());

        f.updateArgs(point + direction * eps2);
        double v2 = Math.abs(f.calc());

        f.updateArgs(point + direction * eps3);
        double v3 = Math.abs(f.calc());

        // 1. Check for immediate blow-up (Poles/NaNs)
        if (Double.isInfinite(v3) || Double.isNaN(v3)) {
            return true;
        }

        // 2. Log-Slope Test: For f(x) ~ x^-a, log(f(x)) is linear with log(x)
        // We check if the rate of growth is accelerating as we get closer.
        double ratio1 = v2 / v1;
        double ratio2 = v3 / v2;

        // If the function is growing by more than a certain factor (e.g., > 1.5x) 
        // over an order-of-magnitude step, it's "hard" for polynomials.
        return (ratio1 > 1.5 && ratio2 > 1.5);
    }

    /**
     * Detection heuristic: Add this logic to your platform to automatically
     * decide which DomainMap to use.
     *
     * @param f
     * @param point
     * @param direction
     * @return
     */
    private boolean isSingularAt(Function f, double point, double direction) {
        double epsilon = 1e-9;

        // Sample at two points near the boundary
        f.updateArgs(point + direction * epsilon);
        double y1 = f.calc();

        f.updateArgs(point + direction * 2 * epsilon);
        double y2 = f.calc();

        // 1. Check for infinite values (Poles)
        if (Double.isInfinite(y1) || Double.isNaN(y1)) {
            return true;
        }

        // 2. Check for "Explosive" growth (Logarithmic/Power singularities)
        // If the function grows by more than 10x over a tiny 1e-9 step,
        // a standard linear Chebyshev series will struggle to converge.
        double ratio = Math.abs(y1 / y2);
        return ratio > 10.0 || ratio < 0.1;
    }

    /**
     * Detection heuristic
     *
     * @param f
     * @param a
     * @param b
     * @return
     */
    public DomainMap autoSelectMap(Function f, double a, double b) {
        // Strategy 1: Infinite bounds
        if (Double.isInfinite(b)) {
            return new SemiInfiniteMap(1.0);
        }

        // Strategy 2: Singularities at endpoints
        if (isSingularAt(f, a, 1.0)) {
            return new LogarithmicMap(b - a, 10.0);
        }

        // Strategy 3: Default to stable linear map
        return new LinearMap(a, b);
    }

    private double clenshaw(double u) {
        double b2 = 0, b1 = 0, b0 = 0;
        for (int j = coefficients.length - 1; j >= 1; j--) {
            b0 = coefficients[j] + 2.0 * u * b1 - b2;
            b2 = b1;
            b1 = b0;
        }
        return coefficients[0] + u * b1 - b2;
    }

    // Interface for different "hardening" strategies
    public interface DomainMap {

        double toPhysical(double u);  // [-1, 1] -> [a, b] or [0, inf)

        double toChebyshev(double x); // physical -> [-1, 1]

        double derivativeFactor(double u); // du/dx chain rule factor

        double dx_du(double u);
    }

    /**
     * Algebraic map for the semi-infinite interval [0, infinity) Hardens
     * against functions with "tails".
     */
    public static class SemiInfiniteMap implements DomainMap {

        private final double L;

        public SemiInfiniteMap(double L) {
            this.L = L;
        }

        @Override
        public double toPhysical(double u) {
            return L * (1.0 + u) / (1.0 - u);
        }

        @Override
        public double toChebyshev(double x) {
            return (x - L) / (x + L);
        }

        public double dx_du(double u) {
            // Essential for INTEGRATION
            double denom = 1.0 - u;
            return (2.0 * L) / (denom * denom);
        }

        @Override
        public double derivativeFactor(double u) {
            // Essential for DIFFERENTIATION (du/dx)
            // Calculated as: (1-u)^2 / 2L
            double factor = 1.0 - u;
            return (factor * factor) / (2.0 * L);
        }
    }

    public static class LinearMap implements DomainMap {

        private final double a, b;
        private final double halfWidth;

        public LinearMap(double a, double b) {
            this.a = a;
            this.b = b;
            this.halfWidth = 0.5 * (b - a);
        }

        @Override
        public double toPhysical(double u) {
            return halfWidth * u + 0.5 * (a + b);
        }

        @Override
        public double toChebyshev(double x) {
            return (2.0 * x - (a + b)) / (b - a);
        }

        public double dx_du(double u) {
            // This is the Jacobian used for INTEGRATION
            return halfWidth;
        }

        @Override
        public double derivativeFactor(double u) {
            // This is du/dx, used for DIFFERENTIATION (Chain Rule)
            // Since x = (b-a)/2 * u + C, then dx = (b-a)/2 * du
            // Therefore du/dx = 2 / (b - a)
            return 1.0 / halfWidth;
        }
    }

    public static class LogarithmicMap implements DomainMap {

        private final double L; // Interval length
        private final double s; // Sensitivity (usually 10.0 to 20.0)

        public LogarithmicMap(double L, double s) {
            this.L = L;
            this.s = s;
        }

        @Override
        public double toPhysical(double u) {
            return L * Math.exp(s * (u - 1.0));
        }

        @Override
        public double toChebyshev(double x) {
            return 1.0 + (Math.log(x / L) / s);
        }

        public double dx_du(double u) {
            // Jacobian: dx/du = s * x
            return s * toPhysical(u);
        }

        @Override
        public double derivativeFactor(double u) {
            // du/dx = 1 / (s * x)
            double x = toPhysical(u);
            // Hardening: Prevent division by zero at the absolute boundary
            if (x < 1e-300) {
                return 1e300;
            }
            return 1.0 / (s * x);
        }
    }
/**
 * A DomainMap that clusters nodes tightly at BOTH the lower bound (a) 
 * and the upper bound (b). Uses a Hyperbolic Tangent transformation.
 */
public static class DoubleLogarithmicMap implements MappedExpander.DomainMap {
    private final double a, b, c, m, s, tanhS;

    /**
     * @param s Sensitivity. For a tanh map, a value between 3.0 and 5.0 is ideal.
     * s = 4.0 creates extreme clustering at the boundaries.
     */
    public DoubleLogarithmicMap(double a, double b, double s) {
        this.a = a;
        this.b = b;
        this.c = (a + b) / 2.0;          // Midpoint
        this.m = (b - a) / 2.0;          // Half-width
        this.s = s;
        this.tanhS = Math.tanh(s);       // Cache for performance
    }

    @Override
    public double toPhysical(double u) {
        // Map u [-1, 1] to x [a, b] using tanh
        return c + m * (Math.tanh(s * u) / tanhS);
    }

    @Override
    public double toChebyshev(double x) {
        // Inverse mapping: x to u
        double val = (x - c) / m * tanhS;
        
        // HARDENING: Clamp to prevent NaN from floating-point overshoot near boundaries
        val = Math.max(-0.999999999999999, Math.min(0.999999999999999, val));
        
        // arctanh(val) = 0.5 * ln((1 + val) / (1 - val))
        return Math.log((1.0 + val) / (1.0 - val)) / (2.0 * s); 
    }

    @Override
    public double dx_du(double u) {
        // The Jacobian: How much the space is stretched.
        // Derivative of tanh(su) is s * sech^2(su)
        double coshSU = Math.cosh(s * u);
        return (m * s) / (tanhS * coshSU * coshSU);
    }

    @Override
    public double derivativeFactor(double u) {
        return 1.0 / dx_du(u);
    }
}
    /**
     * Logarithmic map that clusters nodes near the upper bound B.
     */
    public static class ReversedLogarithmicMap implements MappedExpander.DomainMap {

        private final double a, b, L, s;

        public ReversedLogarithmicMap(double a, double b, double s) {
            this.a = a;
            this.b = b;
            this.L = b - a;
            this.s = s;
        }

        @Override
        public double toPhysical(double u) {
            // u = -1 (left) -> x = a
            // u = 1  (right) -> x = b (Singularity point)
            return b - L * Math.exp(s * (-u - 1.0));
        }

        @Override
        public double toChebyshev(double x) {
            return -1.0 - (Math.log((b - x) / L) / s);
        }

        @Override
        public double dx_du(double u) {
            // Jacobian for integration: clusters points where u -> 1
            return s * (b - toPhysical(u));
        }

        @Override
        public double derivativeFactor(double u) {
            double dist = b - toPhysical(u);
            return (dist < 1e-300) ? 1e300 : 1.0 / (s * dist);
        }
    }

    public static final class SubDomainMap implements DomainMap {

        private final DomainMap parent;
        private final double uStart, uEnd;

        public SubDomainMap(DomainMap parent, double uStart, double uEnd) {
            this.parent = parent;
            this.uStart = uStart;
            this.uEnd = uEnd;
        }

        @Override
        public double toPhysical(double u) {
            // Map u from [-1, 1] to [uStart, uEnd], then to parent physical space
            double uMapped = uStart + (u + 1.0) * 0.5 * (uEnd - uStart);
            return parent.toPhysical(uMapped);
        }

        @Override
        public double dx_du(double u) {
            double uMapped = uStart + (u + 1.0) * 0.5 * (uEnd - uStart);
            // Chain rule: dx/du = dx/duMapped * duMapped/du
            return parent.dx_du(uMapped) * 0.5 * (uEnd - uStart);
        }

        @Override
        public double toChebyshev(double x) {
            double uMapped = parent.toChebyshev(x);
            return 2.0 * (uMapped - uStart) / (uEnd - uStart) - 1.0;
        }

        @Override
        public double derivativeFactor(double u) {
            return 1.0 / dx_du(u);
        }
    }

    public static final class CCWeightGenerator {
        // Standard size for high-resolution segments. 
        // N=256 creates an array of length 257 (0 to 256).

        private static final int DEFAULT_N = 256;
        private static final double[] CACHED_WEIGHTS_256 = generateWeights(DEFAULT_N);

        /**
         * Public accessor for the pre-computed weights.
         */
        public static double[] getCachedWeights() {
            return CACHED_WEIGHTS_256;
        }

        /**
         * Generates Clenshaw-Curtis weights for N+1 nodes. Hardened for
         * 16-digit precision and symmetry.
         */
        public static double[] generateWeights(int N) {
            if (N % 2 != 0) {
                throw new IllegalArgumentException("N must be even for standard Clenshaw-Curtis symmetry.");
            }

            double[] weights = new double[N + 1];
            double[] moments = new double[N + 1];

            // 1. Initialize moments: Integral(T_k) = 2 / (1 - k^2)
            // Only even indices are non-zero.
            for (int k = 0; k <= N; k += 2) {
                moments[k] = 2.0 / (1.0 - (double) k * k);
            }

            // 2. Compute weights via Inverse DCT-I
            // We calculate half and use symmetry (weights[i] == weights[N-i])
            int half = N / 2;
            for (int i = 0; i <= half; i++) {
                double theta = (i * Math.PI) / N;

                // Initialization for k=0 and k=N endpoints of the DCT
                // Cos(0) = 1, Cos(i*PI) = (-1)^i
                double sum = 0.5 * (moments[0] + ((i % 2 == 0) ? 1.0 : -1.0) * moments[N]);

                for (int k = 2; k < N; k += 2) {
                    sum += moments[k] * Math.cos(k * theta);
                }

                double w = (2.0 / N) * sum;
                weights[i] = w;
                weights[N - i] = w;
            }

            // 3. Hardening: Theoretical Boundary Weights
            // For even N, the endpoints must be exactly 1 / (N^2 - 1)
            double boundary = 1.0 / (N * N - 1.0);
            weights[0] = boundary;
            weights[N] = boundary;

            return weights;
        }
    }

    public static void main(String[] args) {
        // 1. Define the badly behaved function: 1 / (1 + x^2)
        // This function has a "long tail" that never hits zero.
        Function slowDecay = new Function("@(x) 1 / (1 + x^2)");

        // 2. Create an Algebraic Map for [0, infinity)
        // We set L (Scale Factor) to 1.0. 
        // This means 50% of our Chebyshev nodes will be placed between x=0 and x=1,
        // and the other 50% will be spread from x=1 to x=infinity.
        MappedExpander.DomainMap infiniteMap = new MappedExpander.SemiInfiniteMap(1.0);

        // 3. Expand the function using 64 Chebyshev nodes
        // Even though the range is infinite, 64 nodes provide near machine precision.
        MappedExpander expander = new MappedExpander(slowDecay, infiniteMap, 64);

        // 4. Evaluate at various points
        double test1 = 0.5;
        double test2 = 10.0;
        double test3 = 1000.0; // Extremely far out in the "tail"

        System.out.println("Evaluation at x=" + test1 + ": " + expander.evaluate(test1));
        System.out.println("Evaluation at x=" + test2 + ": " + expander.evaluate(test2));
        System.out.println("Evaluation at x=" + test3 + ": " + expander.evaluate(test3));

        // Comparison: The exact value at x=1000 is 0.000000999999
        // Standard polynomials would oscillate wildly here; MappedExpander stays stable.
    }

}
