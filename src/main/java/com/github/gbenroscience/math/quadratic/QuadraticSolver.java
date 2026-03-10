package com.github.gbenroscience.math.quadratic;

import static java.lang.Math.*;

/**
 * Solves quadratic equations of the form ax² + bx + c = 0.
 * Includes numerical stability for real roots and correct complex conjugate pairs.
 * * @author GBEMIRO
 */
public class QuadraticSolver {

    private double a;
    private double b;
    private double c;
    private boolean complex;

    /**
     * solutions[0], solutions[1]: Real/Imag parts of root 1
     * solutions[2], solutions[3]: Real/Imag parts of root 2
     * solutions[4]: 0 for Real, 1 for Complex
     */
    public final double[] solutions = new double[5];
    public final double[] complexCoords1 = new double[2];
    public final double[] complexCoords2 = new double[2];

    public QuadraticSolver(double a, double b, double c) {
        this.a = a;
        this.b = b;
        this.c = c;
        solution();
    }

    public void setA(double a) { this.a = a; solution(); }
    public double getA() { return a; }
    public void setB(double b) { this.b = b; solution(); }
    public double getB() { return b; }
    public void setC(double c) { this.c = c; solution(); }
    public double getC() { return c; }

    public final boolean isComplex() {
        return complex;
    }

    /**
     * Returns the solutions as a formatted string.
     */
    public String solve() {
        if (!complex) {
            return solutions[0] + ", " + solutions[1];
        } else {
            // Using abs() on the imaginary part ensures the string 
            // format "real +/- imag i" is always preserved.
            double imagMagnitude = abs(solutions[1]);
            return solutions[0] + " + " + imagMagnitude + " i,\n"
                 + solutions[2] + " - " + imagMagnitude + " i";
        }
    }

    /**
     * Returns the solutions as an array of strings.
     */
    public String[] soln() {
        String[] result = new String[2];
        if (!complex) {
            result[0] = String.valueOf(solutions[0]);
            result[1] = String.valueOf(solutions[1]);
        } else {
            double imagMagnitude = abs(solutions[1]);
            result[0] = solutions[0] + " + " + imagMagnitude + " i";
            result[1] = solutions[2] + " - " + imagMagnitude + " i";
        }
        return result;
    }

    /**
     * Internal solver logic.
     */
    final void solution() {
        double discriminant = b * b - 4 * a * c;

        if (discriminant >= 0) {
            this.complex = false;
            double sqrtD = sqrt(discriminant);

            // Numerically stable quadratic formula to prevent precision loss
            double q = -0.5 * (b + copySign(sqrtD, b));

            if (q != 0) {
                solutions[0] = q / a;
                solutions[1] = c / q;
            } else {
                // Linear case: ax + c = 0
                solutions[0] = (a != 0) ? -c / a : Double.NaN;
                solutions[1] = Double.NaN;
            }
            
            complexCoords1[0] = solutions[0]; complexCoords1[1] = 0;
            complexCoords2[0] = solutions[1]; complexCoords2[1] = 0;
            solutions[4] = 0;
            
        } else {
            this.complex = true;
            double realPart = -b / (2 * a);
            // Magnitude of imaginary part
            double imagPart = sqrt(-discriminant) / (2 * a);

            // Store parts symmetrically
            solutions[0] = realPart;
            solutions[1] = imagPart;
            solutions[2] = realPart;
            solutions[3] = -imagPart; 

            complexCoords1[0] = realPart; complexCoords1[1] = imagPart;
            complexCoords2[0] = realPart; complexCoords2[1] = -imagPart;
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