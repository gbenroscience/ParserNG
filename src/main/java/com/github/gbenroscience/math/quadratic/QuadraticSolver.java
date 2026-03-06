package com.github.gbenroscience.math.quadratic;

import static java.lang.Math.*;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author GBEMIRO
 */
public class QuadraticSolver {

    private double a;
    private double b;
    private double c;

    private boolean complex;

    /**
     * Stores the roots of the equation. 
     * If the solutions are not complex, then
     * the 2 solutions are real numbers stored in index 0 and
     * index 1 of this array.
     * 
     * If the solutions are complex(check field {@link QuadraticSolver#complex}), then
     * index 0 contains the real part of the first complex solution and index 1 contains the complex part of the first complex solution.
     * Index 2 contains the real part of the second complex solution and index 3 contains the complex part of the second complex solution.
     * If the content of the 4th index(solutions[4]) is 0, then the equation has real roots.
     * If the content of the 4th index(solutions[4]) is 1,then it has complex roots.
     * 
     */
    public final double[] solutions = new double[5];
    /**
     * Store the real root of the equation in the first index of the array Store
     * the complex root of the equation in the second index of the array
     *
     */
    public final double[] complexCoords1 = new double[2];
    /**
     * Store the real root of the equation in the first index of the array Store
     * the complex root of the equation in the second index of the array
     *
     */
    public final double[] complexCoords2 = new double[2];

    /**
     *
     * @param a the co-efficient of the squared variable
     * @param b the co-efficient of the variable
     * @param c the constant term
     */
    public QuadraticSolver(double a, double b, double c) {
        this.a = a;
        this.b = b;
        this.c = c;
        solution();
    }

    /**
     *
     * @param a the co-efficient of the squared variable
     */
    public void setA(double a) {
        this.a = a;
    }

    /**
     *
     * @return the co-efficient of the squared variable
     */
    public double getA() {
        return a;
    }

    /**
     *
     * @param b the co-efficient of the variable
     */
    public void setB(double b) {
        this.b = b;
    }

    /**
     *
     * @return the co-efficient of the squared variable
     */
    public double getB() {
        return b;
    }

    /**
     *
     * @param c the constant term
     */
    public void setC(double c) {
        this.c = c;
    }

    /**
     *
     * @return the constant term
     */
    public double getC() {
        return c;
    }

    public final boolean isComplex() {
        return complex;
    }

    
    /**
     *
     * @return the solutions as a string of values.
     */
    public String solve() {
        String result = "";

        if (!complex) {
            result = String.valueOf(solutions[0]) + ",  " + String.valueOf(solutions[1]);
        } else{
            result = String.valueOf(solutions[0]) + "+" + String.valueOf(solutions[1]) + " i  ,\n"
                    + String.valueOf(solutions[2]) + "-" + String.valueOf(solutions[3]) + " i ";
        }
//2p^2-3p-4.09=0
        result = result.replace("+-", "-");
        result = result.replace("-+", "-");
        result = result.replace("--", "+");
        return result;
    }

    /**
     *
     * @return the solutions as an array of values
     */
    public String[] soln() {

        String result[] = new String[2];

        if (!complex) {
            result[0] = String.valueOf(solutions[0]);
            result[1] = String.valueOf(solutions[1]);
        } else  {
            result[0] = String.valueOf(solutions[0]) + "+" + String.valueOf(solutions[1]) + " i";
            result[1] =  String.valueOf(solutions[1]) + "-" +  String.valueOf(solutions[3])+ " i ";
        }
//2p^2-3p-4.09=0

        result[0] = result[0].replace("+-", "-");
        result[0] = result[0].replace("-+", "-");
        result[0] = result[0].replace("--", "+");

        result[1] = result[1].replace("+-", "-");
        result[1] = result[1].replace("-+", "-");
        result[1] = result[1].replace("--", "+");

        return result;
    }

/**
     * Solves the quadratic equation ax² + bx + c = 0.
     * Implements a numerically stable version of the quadratic formula
     * to prevent precision loss when 4ac is very small compared to b².
     */
    final void solution() {
        double discriminant = b * b - 4 * a * c;

        if (discriminant >= 0) {
            this.complex = false;
            double sqrtD = sqrt(discriminant);

            // "World-Class" Stable Quadratic Formula:
            // Prevents catastrophic cancellation if -b and sqrt(D) have similar magnitudes.
            // We calculate 'q' based on the sign of 'b' to ensure we are adding, not subtracting.
            double q = -0.5 * (b + copySign(sqrtD, b));

            // Avoid division by zero if both a and b are zero
            if (q != 0) {
                solutions[0] = q / a;
                solutions[1] = c / q;
            } else {
                // Handle linear or degenerate case (ax + c = 0)
                solutions[0] = (a != 0) ? -c / a : Double.NaN;
                solutions[1] = Double.NaN;
            }
            
            // Clear complex coordinate arrays
            complexCoords1[0] = solutions[0]; complexCoords1[1] = 0;
            complexCoords2[0] = solutions[1]; complexCoords2[1] = 0;
            solutions[4] = 0;//1 in the 5th index means the equation has real roots
            
        } else {
            this.complex = true;
            double realPart = -b / (2 * a);
            double imagPart = sqrt(-discriminant) / (2 * a);

            // Index 0/1: First complex root (x + yi)
            solutions[0] = realPart;
            solutions[1] = imagPart;

            // Index 2/3: Second complex root (x - yi) -> Conjugate
            solutions[2] = realPart;
            solutions[3] = -imagPart; // Fixed: Use negative for the conjugate

            // Fill helper arrays
            complexCoords1[0] = realPart; complexCoords1[1] = imagPart;
            complexCoords2[0] = realPart; complexCoords2[1] = -imagPart;
            solutions[4] = 1;//1 in the 5th index means the equation has complex roots
        }
    }

    @Override
    public String toString() {
        return a + "x²+" + b + "x+" + c + "=0";
    }

    public static void main(String args[]) {
        QuadraticSolver ohMy = new QuadraticSolver(3, -9, 6);
        System.out.println(ohMy.soln()[0] + ", " + ohMy.soln()[1]);
        System.out.println(ohMy);
    }

}//end class QuadraticSolver
