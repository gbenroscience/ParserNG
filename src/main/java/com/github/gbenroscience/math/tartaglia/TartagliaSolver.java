package com.github.gbenroscience.math.tartaglia;

import com.github.gbenroscience.math.quadratic.QuadraticSolver;
import static java.lang.Math.*;

/**
 * Solves the depressed cubic equation: cx^3 + ax + b = 0
 * Uses Cardano's method and the Trigonometric identity for Casus Irreducibilis.
 * * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class TartagliaSolver {

    private double a; // coefficient of x
    private double b; // constant term
    private double c; // coefficient of x^3

    public TartagliaSolver(double c, double a, double b) {
        this.c = c;
        this.a = a;
        this.b = b;
        normalizeCoefficients();
    }

    /**
     * Divides through by c to get the form x^3 + ax + b = 0.
     * IEEE 754 doubles do not throw ArithmeticException on division by zero;
     * they return Infinity. We check explicitly instead.
     */
    private void normalizeCoefficients() {
        if (abs(c) < 1e-15) {
            throw new IllegalArgumentException("The coefficient of x^3 (c) must not be zero.");
        }
        this.a = a / c;
        this.b = b / c;
        this.c = 1.0;
    }

    public String solve() {
        // After normalization, we solve x^3 + ax + b = 0
        // Discriminant Delta = (b/2)^2 + (a/3)^3
        double discriminant = (pow(b, 2) / 4.0) + (pow(a, 3) / 27.0);

        // Case 1: One real root and two complex roots
        if (discriminant > 0) {
            double sqrtDisc = sqrt(discriminant);
            double u = cbrt(-b / 2.0 + sqrtDisc);
            double v = cbrt(-b / 2.0 - sqrtDisc);
            double x1 = u + v;

            // To find complex roots, we solve the depressed quadratic: x^2 + x1*x + (x1^2 + a) = 0
            // Note: QuadraticSolver must handle complex/real roots appropriately.
            QuadraticSolver solver = new QuadraticSolver(1.0, x1, pow(x1, 2) + a);
            return x1 + ",\n" + solver.solve();
        } 
        
        // Case 2: Three real roots (Casus Irreducibilis)
        else if (discriminant < 0) {
            double r = sqrt(-pow(a, 3) / 27.0);
            double phi = acos(-b / (2.0 * r));
            double s = 2.0 * pow(r, 1.0/3.0);
            
            double x1 = s * cos(phi / 3.0);
            double x2 = s * cos((phi + 2.0 * PI) / 3.0);
            double x3 = s * cos((phi + 4.0 * PI) / 3.0);
            
            return x1 + ",\n" + x2 + ",\n" + x3;
        } 
        
        // Case 3: Multiple roots (Discriminant is exactly zero)
        else {
            if (abs(a) < 1e-15 && abs(b) < 1e-15) return "0.0";
            
            double x1 = 3.0 * b / a;
            double x2 = -3.0 * b / (2.0 * a);
            return x1 + ",\n" + x2;
        }
    }

    // Getters and Setters
    public double getA() { return a; }
    public void setA(double a) { this.a = a; }
    public double getB() { return b; }
    public void setB(double b) { this.b = b; }
    public double getC() { return c; }
    public void setC(double c) { this.c = c; normalizeCoefficients(); }
}