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
import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Uses Chebyshev Polynomials to expand, evaluate, differentiate and integrate
 * functions
 *
 * @author GBEMIRO
 */
public class FunctionExpander {

    private final double[] coefficients;
    private double lower, upper;
    Function function;

    public FunctionExpander(MethodHandle function, double lower, double upper, int degree) throws Throwable {
        this.lower = lower;
        this.upper = upper;
        this.coefficients = computeCoefficients(function, degree);
    }

    public FunctionExpander(MethodHandle function, double lower, double upper, double tolerance) throws Throwable {
        this.lower = lower;
        this.upper = upper;
        this.coefficients = computeAdaptiveCoefficients(function, tolerance);
    }

    // Fixed Degree (For manual control)
    public FunctionExpander(Function function, double lower, double upper, int degree) {
        this.function = function;
        this.lower = lower;
        this.upper = upper;
        this.coefficients = computeCoefficients(function, degree);
    }

// Adaptive Degree (The "Turbo" Auto-Pilot)
    public FunctionExpander(Function function, double lower, double upper, double tolerance) {
        this.lower = lower;
        this.upper = upper;
        this.coefficients = computeAdaptiveCoefficients(function, tolerance);
    }

    public void setLower(double lower) {
        this.lower = lower;
    }

    public double getLower() {
        return lower;
    }

    public void setUpper(double upper) {
        this.upper = upper;
    }

    public double getUpper() {
        return upper;
    }

    private double[] computeCoefficients(MethodHandle function, int n) throws Throwable {
        double[] c = new double[n];
        double[] fx = new double[n];

        // 1. Sample function at Chebyshev nodes
        for (int k = 1; k <= n; k++) {
            double node = Math.cos(Math.PI * (2.0 * k - 1.0) / (2.0 * n));
            // Map [-1, 1] to [lower, upper]
            double x = 0.5 * (node + 1) * (upper - lower) + lower;
            fx[k - 1] = (double) function.invokeExact(new double[]{x});
        }

        // 2. Compute coefficients using the orthogonality property
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 1; k <= n; k++) {
                sum += fx[k - 1] * Math.cos(Math.PI * j * (2.0 * k - 1.0) / (2.0 * n));
            }
            double factor = (j == 0) ? (1.0 / n) : (2.0 / n);
            c[j] = factor * sum;
        }
        return c;
    }

    /**
     * Dynamically computes coefficients using MethodHandle for Turbo
     * performance. Double the degree N until the last few coefficients are
     * below the tolerance.
     *
     * * @param function The Turbo-compiled MethodHandle (double[])Object
     * @param tolerance The target precision (e.g., 1e-12)
     * @return The optimally sized coefficient array
     * @throws Throwable if evaluation fails
     */
    private double[] computeAdaptiveCoefficients(MethodHandle function, double tolerance) throws Throwable {
        int n = 16;       // Initial degree
        int maxN = 1024;  // Safety cap to prevent OOM
        double[] c = null;

        while (n <= maxN) {
            // Use your existing MethodHandle version of computeCoefficients
            c = computeCoefficients(function, n);

            // Check the 'tail' of the coefficients. 
            // We sum the last three to avoid being fooled by zeros in symmetric functions.
            double tailError = Math.abs(c[n - 1]) + Math.abs(c[n - 2]) + Math.abs(c[n - 3]);

            if (tailError <= tolerance) {
                // Success! The function is well-approximated.
                return trimCoefficients(c, tolerance);
            }

            // Precision not met, double the nodes and retry
            n *= 2;
        }

        // Return the best-effort coefficients if maxN is reached
        return c;
    }

    /**
     * Strips insignificant coefficients to keep the resulting polynomial
     * expression as lean as possible.
     */
    private double[] trimCoefficients(double[] c, double tolerance) {
        int lastSignificant = c.length - 1;
        // Walk backward until we find a coefficient larger than our tolerance
        while (lastSignificant > 0 && Math.abs(c[lastSignificant]) < (tolerance / 10.0)) {
            lastSignificant--;
        }

        double[] trimmed = new double[lastSignificant + 1];
        System.arraycopy(c, 0, trimmed, 0, lastSignificant + 1);
        return trimmed;
    }

    /**
     * Computes Chebyshev coefficients using a standard functional interface.
     * Useful for non-compiled or interpreted functions.
     */
    private double[] computeCoefficients(Function function, int n) {
        double[] c = new double[n];
        double[] fx = new double[n];

        // 1. Sample function at Chebyshev nodes
        for (int k = 1; k <= n; k++) {
            // Compute the k-th zero of the n-th Chebyshev polynomial
            double node = Math.cos(Math.PI * (2.0 * k - 1.0) / (2.0 * n));

            // Map from standard interval [-1, 1] to [lower, upper]
            double x = 0.5 * (node + 1.0) * (upper - lower) + lower;

            function.updateArgs(x);
            // Evaluate function
            fx[k - 1] = function.calc();
        }

        // 2. Compute coefficients using Discrete Cosine Transform (DCT-II) logic
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 1; k <= n; k++) {
                sum += fx[k - 1] * Math.cos(Math.PI * j * (2.0 * k - 1.0) / (2.0 * n));
            }
            // Orthogonality normalization factor
            double factor = (j == 0) ? (1.0 / n) : (2.0 / n);
            c[j] = factor * sum;
        }
        return c;
    }

    /**
     * Dynamically computes coefficients to guarantee a specific error
     * tolerance.
     *
     * * @param function The math function (wrapped as DoubleUnaryOperator)
     * @param tolerance The desired precision (e.g., 1e-10)
     * @return The optimally sized coefficient array
     */
    private double[] computeAdaptiveCoefficients(Function function, double tolerance) {
        int n = 16;       // Start with a low degree
        int maxN = 1024;  // Hard limit to prevent OOM or infinite loops
        double[] c = null;

        while (n <= maxN) {
            // Compute coefficients for the current degree
            c = computeCoefficients(function, n);

            // We check the last 3 coefficients. 
            // Why 3? Because symmetric functions (like even/odd functions) 
            // might have every other coefficient be exactly zero. Checking 3 
            // guarantees we don't get tricked by a single zero.
            double tailError = Math.abs(c[n - 1]) + Math.abs(c[n - 2]) + Math.abs(c[n - 3]);

            if (tailError <= tolerance) {
                // The tail is practically zero. We have converged!
                // Optional: We can actually trim the array to remove the trailing zeros 
                // to make the buildPolynomial() string even shorter.
                return trimCoefficients(c, tolerance);
            }

            // If error is still too high, double the degree and try again
            n *= 2;
        }

        // If we exit the loop, we hit maxN without fully converging.
        // It's usually best to warn the user, or just return the best effort.
        System.err.println("Warning: FunctionExpander reached max degree " + maxN
                + " without fully converging. Tail error: "
                + (Math.abs(c[maxN - 1]) + Math.abs(c[maxN - 2]) + Math.abs(c[maxN - 3])));
        return c;
    }

    /**
     * Evaluate the approximated polynomial using Clenshaw's Algorithm. This is
     * O(N) and much more stable than evaluating a^n + b^n-1...
     */
    public double evaluate(double x) {
        // Map x to [-1, 1]
        double u = (2.0 * x - lower - upper) / (upper - lower);
        double b2 = 0, b1 = 0, b0 = 0;

        for (int j = coefficients.length - 1; j >= 1; j--) {
            b0 = coefficients[j] + 2.0 * u * b1 - b2;
            b2 = b1;
            b1 = b0;
        }
        return coefficients[0] + u * b1 - b2;
    }

    /**
     * Computes the derivative of the approximation at point x. This is an exact
     * derivative of the surrogate polynomial.
     */
    public double derivative(double x) {
        int n = coefficients.length;
        if (n < 2) {
            return 0.0;
        }

        double[] cDeriv = new double[n];

        // Backward recurrence to find derivative coefficients
        cDeriv[n - 1] = 0; // Highest degree derivative is 0
        if (n > 1) {
            cDeriv[n - 2] = 2 * (n - 1) * coefficients[n - 1];
        }

        for (int j = n - 3; j >= 0; j--) {
            cDeriv[j] = cDeriv[j + 2] + 2 * (j + 1) * coefficients[j + 1];
        }

        // Evaluate the derivative coefficients at point x using Clenshaw
        double u = (2.0 * x - lower - upper) / (upper - lower);
        double b2 = 0, b1 = 0, b0 = 0;

        for (int j = n - 1; j >= 1; j--) {
            b0 = cDeriv[j] + (2.0 * u * b1) - b2;
            b2 = b1;
            b1 = b0;
        }

        double derivInU = (coefficients.length == 0) ? 0 : (cDeriv[0] * 0.5 + u * b1 - b2);

        // Apply chain rule: d/dx = d/du * du/dx
        return derivInU * (2.0 / (upper - lower));
    }

    /**
     * Integrates the approximated polynomial analytically using Kahan
     * summation. This preserves the precision of the high-frequency
     * coefficients.
     */
    public double integrateApproximation() {
        if (coefficients == null || coefficients.length == 0) {
            return 0.0;
        }

        // Start with the c0 term (integral of constant T0 = 1 over [-1,1] is 2)
        double sum = 2.0 * coefficients[0];
        double compensation = 0.0;

        for (int n = 2; n < coefficients.length; n += 2) {
            // 1. Calculate the high-precision term
            double term = coefficients[n] * (2.0 / (1.0 - (double) n * n));

            // 2. Kahan logic: subtract the previous error from the current term
            double y = term - compensation;

            // 3. Add to the running sum. 't' is potentially less precise than we want.
            double t = sum + y;

            // 4. Calculate the 'low bits' that were lost during the addition.
            // This MUST be written exactly like this to work.
            compensation = (t - sum) - y;

            // 5. Update the sum with the result
            sum = t;
        }

        // Map from normalized [-1, 1] to the user's [lower, upper]
        return sum * (upper - lower) / 2.0;
    }

    public String buildPolynomial() {
        if (coefficients == null || coefficients.length == 0) {
            return "0";
        }

        // Interval mapping: u = (2*x - (a+b)) / (b-a)
        // We define this once to avoid redundant calcs in the nested structure
        double a = lower;
        double b = upper;
        String u = String.format("((2*x - (%f)) / %f)", (a + b), (b - a));

        // Clenshaw's recurrence: 
        // y_k = c_k + 2*u*y_{k+1} - y_{k+2}
        // We build the expression from the highest degree down to 0.
        int n = coefficients.length - 1;

        // We need to represent the recurrence as a single nested string.
        // For n=3: c0 + u*b1 - b2...
        // This is more efficiently handled by a recursive string builder
        return generateClenshawString(u);
    }

    private String generateClenshawString(String u) {
        int n = coefficients.length - 1;
        if (n == 0) {
            return String.valueOf(coefficients[0]);
        }

        // b_{n+1} and b_{n+2} start at 0
        String b_next = "0";
        String b_next_next = "0";
        String current_b = "";

        for (int j = n; j >= 1; j--) {
            // b_j = c_j + 2*u*b_{j+1} - b_{j+2}
            current_b = String.format("(%.17g + (2*%s*%s) - %s)",
                    coefficients[j], u, b_next, b_next_next);
            b_next_next = b_next;
            b_next = current_b;
        }

        // Final result: f(x) = c0 + u*b1 - b2
        return String.format("(%.17g + (%s * %s) - %s)", coefficients[0], u, b_next, b_next_next);
    }

    /**
     * Estimates the truncation error of the current approximation. For 16dp
     * accuracy, this should be < 1e-16.
     */
    public double getTailError() {
        if (coefficients == null || coefficients.length < 4) {
            return Double.MAX_VALUE;
        }

        int n = coefficients.length;

        // We sum the last few coefficients.
        // Why? In double precision, the noise floor is ~2e-16.
        // If the sum of the last 3 coefficients is larger than our target,
        // the series hasn't converged yet.
        return Math.abs(coefficients[n - 1])
                + Math.abs(coefficients[n - 2])
                + Math.abs(coefficients[n - 3]);
    }

    public static final class ChebyshevForest1 {

        private final List<FunctionExpander> segments = new ArrayList<>();
        private final double tolerance = 1e-10;

        private final int MAX_DEPTH = 25; // Prevents infinite recursion
        private final double MIN_DX = 1.0e-12;
        
        public void build(MethodHandle function, double a, double b) throws Throwable {
            buildRecursive(function, a, b, 0);
        }

        private void buildRecursive(MethodHandle function, double a, double b, int depth) throws Throwable {
            // Use a high degree to see if the interval is "smoothable"
            FunctionExpander attempt = new FunctionExpander(function, a, b, 256);

            // Terminate if:
            // 1. Accuracy is met
            // 2. We've reached max recursion depth
            // 3. The interval is too small for double precision math to handle (sub-atomic scale)
            if (attempt.getTailError() <= tolerance || depth >= MAX_DEPTH || (b - a) < MIN_DX) {
                segments.add(attempt);
            } else {
                double mid = a + (b - a) / 2.0;
                buildRecursive(function, a, mid, depth + 1);
                buildRecursive(function, mid, b, depth + 1);
            }
        }

        public void build(Function function, double a, double b) throws Throwable {
            buildRecursive(function, a, b, 0);
        }

        private void buildRecursive(Function function, double a, double b, int depth) throws Throwable {
            // Use a high degree to see if the interval is "smoothable"
            FunctionExpander attempt = new FunctionExpander(function, a, b, 256);
           
            // Terminate if:
            // 1. Accuracy is met
            // 2. We've reached max recursion depth
            // 3. The interval is too small for double precision math to handle (sub-atomic scale)
            if (attempt.getTailError() <= tolerance || depth >= MAX_DEPTH || (b - a) < MIN_DX) {      
                segments.add(attempt);
            } else {
                double mid = a + (b - a) / 2.0;
                buildRecursive(function, a, mid, depth + 1);    
                buildRecursive(function, mid, b, depth + 1);     
            }
        }

        /**
         * The global integral is simply the sum of the integrals of all
         * segments.
         */
        public double integrate() {
            double totalArea = 0;
            for (FunctionExpander segment : segments) {
                totalArea += segment.integrateApproximation();
            }
            return totalArea;
        }

        /**
         * To find the derivative at x, we find the specific segment containing
         * x.
         */
        public double derivative(double x) {
            FunctionExpander segment = findSegment(x);
            return (segment != null) ? segment.derivative(x) : Double.NaN;
        }

        public double evaluate(double x) {
            FunctionExpander segment = findSegment(x);
            return (segment != null) ? segment.evaluate(x) : Double.NaN;
        }

        /**
         *
         * Find the segment that contains x and call its evaluate(x) For
         * performance, use binary search on the segment boundaries Optimized
         * segment lookup using Binary Search (O(log N))
         */
        private FunctionExpander findSegment(double x) {
            int low = 0;
            int high = segments.size() - 1;

            while (low <= high) {
                int mid = (low + high) >>> 1;
                FunctionExpander s = segments.get(mid);
                if (x < s.getLower()) {
                    high = mid - 1;
                } else if (x > s.getUpper()) {
                    low = mid + 1;
                } else {
                    return s;
                }
            }
            return null;
        }
    }

    /*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
     */
    /**
     * High-precision Piecewise Chebyshev approximation engine. Optimized for
     * ParserNG Turbo engine.
     *
     * @author GBEMIRO
     */
    public static final class ChebyshevForest {

        private final List<FunctionExpander> segments = new ArrayList<>();
        private final double tolerance = 1e-10;
        private final int MAX_DEPTH = 25;
        private final double MIN_DX = 1.0e-12;

        public void build(MethodHandle function, double a, double b) throws Throwable {
            segments.clear();
            buildRecursive(function, a, b, 0);
            segments.sort((o1, o2) -> Double.compare(o1.getLower(), o2.getLower()));
        }

        private void buildRecursive(MethodHandle function, double a, double b, int depth) throws Throwable {
            FunctionExpander attempt = new FunctionExpander(function, a, b, 256);

            if (attempt.getTailError() <= tolerance || depth >= MAX_DEPTH || (b - a) < MIN_DX) {
                segments.add(attempt);
            } else {
                double mid = a + (b - a) / 2.0;
                buildRecursive(function, a, mid, depth + 1);
                buildRecursive(function, mid, b, depth + 1);
            }
        }

        public void build(Function function, double a, double b) throws Throwable {
            segments.clear();
            buildRecursive(function, a, b, 0);
            segments.sort((o1, o2) -> Double.compare(o1.getLower(), o2.getLower()));
        }

        private void buildRecursive(Function function, double a, double b, int depth) throws Throwable {
            FunctionExpander attempt = new FunctionExpander(function, a, b, 256);
          
            if (attempt.getTailError() <= tolerance || depth >= MAX_DEPTH || (b - a) < MIN_DX) {
                segments.add(attempt);
            } else {
                double mid = a + (b - a) / 2.0;
                buildRecursive(function, a, mid, depth + 1);
                buildRecursive(function, mid, b, depth + 1);
            }
        }

        /**
         * The global integral sum using Kahan Summation.
         */
        public double integrate() {
            double sum = 0.0;
            double compensation = 0.0;

            for (FunctionExpander segment : segments) {
                double area = segment.integrateApproximation();
                double y = area - compensation;
                double t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
            return sum;
        }

        public double derivative(double x) {
            FunctionExpander segment = findSegment(x);
            return (segment != null) ? segment.derivative(x) : Double.NaN;
        }

        public double evaluate(double x) {
            FunctionExpander segment = findSegment(x);
            return (segment != null) ? segment.evaluate(x) : Double.NaN;
        }

        private FunctionExpander findSegment(double x) {
            int low = 0;
            int high = segments.size() - 1;

            while (low <= high) {
                int mid = (low + high) >>> 1;
                FunctionExpander s = segments.get(mid);
                if (x < s.getLower()) {
                    high = mid - 1;
                } else if (x > s.getUpper()) {
                    low = mid + 1;
                } else {
                    return s;
                }
            }
            return null;
        }
    }

    public static void main(String[] args) {

        FunctionExpander fe = new FunctionExpander(new Function("@(x)sin(x)"), 0, 10, 1e-16);
        System.out.println("f(" + 4 + ")=" + fe.evaluate(4));
        System.out.println("f'(4)=" + fe.derivative(4));
        System.out.println("I(0,10)=" + fe.integrateApproximation());
        /*
        FunctionExpander.ChebyshevForest1 fec = new FunctionExpander.ChebyshevForest1();
        try {
            fec.build(new Function("@(x)x/(x-sin(x))"), 1, 600);
            System.out.println("com.github.gbenroscience.math.numericalmethods.FunctionExpander.main()");
            System.out.println("f(" + 4 + ")=" + fec.evaluate(4));
            System.out.println("intg()=" + fec.integrate());
            System.out.println("dfx(" + 4 + ")=" + fec.derivative(4));

        } catch (Throwable ex) {
            Logger.getLogger(FunctionExpander.class.getName()).log(Level.SEVERE, null, ex);
        }*/

        FunctionExpander.ChebyshevForest fec = new FunctionExpander.ChebyshevForest();
        try {
            fec.build(new Function("@(x)sin(x)"), 1, 200);
            System.out.println("com.github.gbenroscience.math.numericalmethods.FnExpander.main()");
            System.out.println("f(" + 4 + ")=" + fec.evaluate(4));
            System.out.println("intg()=" + fec.integrate());
            System.out.println("dfx(" + 4 + ")=" + fec.derivative(4));

        } catch (Throwable ex) {
            Logger.getLogger(FunctionExpander.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
}
