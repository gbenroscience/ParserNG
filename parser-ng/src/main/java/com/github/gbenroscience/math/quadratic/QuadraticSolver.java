package com.github.gbenroscience.math.quadratic;

import static java.lang.Math.*;

/**
 * Solves quadratic equations of the form ax² + bx + c = 0. Includes numerical
 * stability for real roots and correct complex conjugate pairs.
 *
 * * @author GBEMIRO
 */
public class QuadraticSolver {

    private double a;
    private double b;
    private double c;
    private boolean complex;

    /**
     * solutions[0], solutions[1]: Real/Imag parts of root 1 solutions[2],
     * solutions[3]: Real/Imag parts of root 2 solutions[4]: 0 for Real, 1 for
     * Complex
     */
    public final double[] solutions = new double[5];

    public QuadraticSolver(double a, double b, double c) {
        this.a = a;
        this.b = b;
        this.c = c;
        solution();
    }

    public void setA(double a) {
        this.a = a;
        solution();
    }

    public double getA() {
        return a;
    }

    public void setB(double b) {
        this.b = b;
        solution();
    }

    public double getB() {
        return b;
    }

    public void setC(double c) {
        this.c = c;
        solution();
    }

    public double getC() {
        return c;
    }

    public final boolean isComplex() {
        return complex;
    }

    /**
     * Returns the solutions as a formatted string.
     */
    public String solve() {
        if (!complex) {
            // Return both real roots
            return solutions[0] + ", " + solutions[2];
        } else {
            double real = solutions[0];
            double imag = abs(solutions[1]);
            return real + " + " + imag + " i,\n"
                    + real + " - " + imag + " i";
        }
    }

    /**
     * Returns the solutions as an array of strings.
     */
    public String[] soln() {
        if (complex) {
            double real = solutions[0];
            double imag = abs(solutions[1]);
            return new String[]{
                real + " + " + imag + " i",
                real + " - " + imag + " i"
            };
        } else {
            String r1 = Double.isNaN(solutions[0]) ? "No Solution" : String.valueOf(solutions[0]);
            String r2 = Double.isNaN(solutions[2]) ? "" : String.valueOf(solutions[2]);
            return new String[]{r1, r2};
        }
    }

    /**
     * Internal solver logic.
     */
    final void solution() {
        // Safety check for linear case: ax + c = 0
        if (abs(a) < 1e-18) {
            this.complex = false;
            if (abs(b) > 1e-18) {
                // b is now the coefficient of x, c is constant
                double root = -c / b;
                solutions[0] = root;
                solutions[1] = 0.0;
                solutions[2] = Double.NaN; // No second root
                solutions[3] = 0.0;
            } else {
                // Degenerate case: c = 0
                solutions[0] = Double.NaN;
                solutions[1] = 0.0;
                solutions[2] = Double.NaN;
                solutions[3] = 0.0;
            }
            solutions[4] = 0;
            return;
        }

        double discriminant = b * b - 4 * a * c;

        if (discriminant >= 0) {
            this.complex = false;
            double sqrtD = sqrt(discriminant);

            // Stable calculation of q
            double q = -0.5 * (b + copySign(sqrtD, b));

            // Root 1 and Root 2 interleaved [R1, I1, R2, I2]
            solutions[0] = q / a;
            solutions[1] = 0.0;
            solutions[2] = c / q;
            solutions[3] = 0.0;
            solutions[4] = 0;
        } else {
            this.complex = true;
            double realPart = -b / (2 * a);
            double imagPart = sqrt(-discriminant) / (2 * a);

            solutions[0] = realPart;
            solutions[1] = imagPart;
            solutions[2] = realPart;
            solutions[3] = -imagPart;
            solutions[4] = 1;
        }
    }

    @Override
    public String toString() {
        return a + "x² + " + b + "x + " + c + " = 0";
    }

    public static void main(String args[]) {
        // Test with the quadratic part of 5x^3 - 12x + 120 after finding the real root
        // The quadratic was roughly x^2 + 3.161x + 7.593 = 0
        QuadraticSolver test = new QuadraticSolver(1.0, 3.16107, 7.593);
        System.out.println(test.solve());
    }
}
