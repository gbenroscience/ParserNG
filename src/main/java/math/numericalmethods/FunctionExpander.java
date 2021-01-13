/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.numericalmethods;

import parser.Function;
import java.util.InputMismatchException;
import parser.MathExpression;
import parser.MathScanner;
import static parser.Operator.*;
import java.math.BigDecimal;
import java.util.ArrayList;

import math.matrix.expressParser.Matrix;
import math.matrix.expressParser.PrecisionMatrix;
import static java.lang.Math.*;
import math.differentialcalculus.Formula;

/**
 *
 * Objects of this class take a function as input and convert it into its
 * polynomial form.
 *
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class FunctionExpander {

    /**
     * The degree of the polynomial. It determines how accurately the polynomial
     * will describe the function.
     *
     * The unit step along x will then be (xUpper - xLower)/polySize
     *
     */
    private int degree;

    /**
     * The upper boundary value of x.
     */
    private double xUpper;
    /**
     * The lower boundary value of x.
     */
    private double xLower;

    /**
     * The function string.
     */
    private Function function;
    /**
     * The polynomial generated.
     */
    private String polynomial;
    /**
     * Uses the precision of double numbers to expand the function. This is
     * about 16 places of decimal.
     */
    public static final int DOUBLE_PRECISION = 1;
    /**
     * Uses the precision of double numbers to expand the function. This is
     * about 33 places of decimal.
     * <br><b color='red'>CAUTION!!!!</b><br>
     * This should be used only if the algorithm of the parser that expands the
     * function has this accuracy.
     */
    public static final int BIGDECIMAL_PRECISION = 2;

    /**
     *
     * Objects of this class will employ this constructor in creating the
     * polynomial of best fit for the input function between the given boundary
     * values of x. The degree parameter is the highest power of the polynomial
     * formed.
     *
     * Creates a new object of this class and initializes it with the following
     * attributes:
     *
     * @param xLower The lower boundary value of x.
     * @pparam xUpper The upper boundary value of x.
     * @param degree The degree of the polynomial. It determines how accurately
     * the polynomial will describe the function. The unit step along x will
     * then be (xUpper - xLower)/polySize
     * @param precision The precision mode to employ in expanding the Function.
     * @param function The function string.
     */
    public FunctionExpander(double xLower, double xUpper, int degree, int precision, Function function) {
        this.xLower = xLower;
        this.xUpper = xUpper;
        this.degree = degree;
        this.function = function;
        buildPolynomial(precision);
    }

    /**
     *
     * @param precision The precision mode to employ in expanding the Function.
     * @param expression An expression containing information about the function
     * whose polynomial expansion is to be deduced and the limits of expansion.
     * For example: function,2,4,20....means expand the function between
     * horizontal coordinates 2 and 3 as a polynomial up to degree 20.. Direct
     * examples would be: sin(x+1),3,4,20 cos(sinh(x-2/tan9x)),4,4.32,32 and so
     * on.
     *
     * F(x)=var x=3;3x; poly(F,4,4.32,32)
     *
     *
     */
    public FunctionExpander(String expression, int precision) {
        setFunction(expression, precision);
    }

    /**
     * @param precision The precision mode to employ in expanding the Function.
     * @param expression An expression containing information about the function
     * whose polynomial expansion is to be deduced and the limits of expansion.
     * For example: function,2,4,20....means expand the function between
     * horizontal coordinates 2 and 3 as a polynomial up to degree 20.. Direct
     * examples would be: sin(x+1),3,4,20 cos(sinh(x-2/tan9x)),4,4.32,32 and so
     * on.
     *
     * F(x)=var x=3;3x; poly(F,4,4.32,32)
     */
    public void setFunction(String expression, int precision) {
        parsePolynomialCommand(expression);
        buildPolynomial(precision);
    }

    /**
     * Changes the Function object dealt with by this class.
     *
     * @param function The new Function object
     */
    public void setFunction(Function function) {
        this.function = function;
    }

    public Function getFunction() {
        return function;
    }

    public int getDegree() {
        return degree;
    }

    public void setDegree(int degree) {
        this.degree = degree;
    }

    public void setxLower(double xLower) {
        this.xLower = xLower;
    }

    public double getxLower() {
        return xLower;
    }

    public void setxUpper(double xUpper) {
        this.xUpper = xUpper;
    }

    public double getxUpper() {
        return xUpper;
    }

    /**
     * @return the unit step along x.
     */
    private double getXStep() {
        double val = degree;
        return (xUpper - xLower) / val;
    }

    public void setPolynomial(String polynomial) {
        this.polynomial = polynomial;
    }

    public String getPolynomial() {
        return polynomial;
    }

    /**
     *
     * @return the coefficient matrix of the function's polynomial.
     */
    public Matrix getMatrix() {
        MathExpression fun = function.getMathExpression();
        fun.setDRG(1);
        double dx = getXStep();
        double arr[][] = new double[degree + 1][degree + 2];

        for (int rows = 0; rows < degree + 1; rows++) {
            for (int cols = 0; cols < degree + 2; cols++) {
                if (cols < degree + 1) {
                    arr[rows][cols] = pow(xLower + rows * dx, cols);
                }//end if
                else if (cols == degree + 1) {
                    fun.setValue(function.getIndependentVariables().get(0).getName(), String.valueOf((xLower + rows * dx)));
                    try {
                        arr[rows][cols] = Double.valueOf(fun.solve());
                    }//end try
                    catch (NumberFormatException numException) {

                    }//end catch
                }//end else if
            }//end cols
        }//end rows

        return new Matrix(arr);

    }//end method

    /**
     *
     * @return the coefficient matrix of the function's polynomial.
     */
    public PrecisionMatrix getPrecisionMatrix() {
        MathExpression fun = function.getMathExpression();
        fun.setDRG(1);
        double dx = getXStep();
        BigDecimal arr[][] = new BigDecimal[degree + 1][degree + 2];

        for (int rows = 0; rows < degree + 1; rows++) {
            for (int cols = 0; cols < degree + 2; cols++) {
                if (cols < degree + 1) {
                    arr[rows][cols] = BigDecimal.valueOf(pow(xLower + rows * dx, cols));
                }//end if
                else if (cols == degree + 1) {
                    fun.setValue(function.getIndependentVariables().get(0).getName(), String.valueOf((xLower + rows * dx)));
                    try {
                        arr[rows][cols] = new BigDecimal(fun.solve());
                    }//end try
                    catch (NumberFormatException numException) {

                    }//end catch
                }//end else if
            }//end cols
        }//end rows

        return new PrecisionMatrix(arr);

    }//end method

    /**
     * Method that processes the format that this software will recognize for
     * user input of an integral expression.
     *
     * The general format is:
     *
     * expression,lowerLimit,upperLimit,iterations(optional) e.g...
     * sin(3x-5),2,5.//assuming default number of iterations which will be
     * computed automatically sin(3x-5),2,5,50.//specifies 50 iterations. Please
     * ensure that the function is continuous in the specified range.
     *
     * @param expression The expression containing the function to integrate and
     * the lower and upper boundaries of integration.
     *
     * Produces an array which has:
     * At index 0.....the expression to integrate
     * At index 1.....the lower limit of integration At index 2.....the upper
     * limit of integration. At index 3(optional)...the number of iterations to
     * employ in evaluating this expression.
     *
     * F(x)=3x+1; poly( F,0,2,3 ) poly(F(x)=3x+1,0,2,5) OR poly(F(x),0,2,5) OR
     * poly(F,0,2,5)
     *
     */
    public void parsePolynomialCommand(String expression) {

        expression = expression.trim();

        if (expression.startsWith("poly(") && expression.endsWith(")")) {
            expression = expression.substring(expression.indexOf("(") + 1);
            expression = expression.substring(0, expression.length() - 1);//remove the last bracket
            double args[] = new double[3];
            args[0] = Double.NaN;
//The expression should look like...function,x1,x2,iterations(optional)
            int lastCommaIndex = expression.lastIndexOf(",");
            try {
                args[2] = Double.parseDouble(expression.substring(lastCommaIndex + 1).trim());
                expression = expression.substring(0, lastCommaIndex).trim();
            }//end try
            catch (NumberFormatException numErr) {
                throw new InputMismatchException("SYNTAX ERROR!");
            }//end catch

            lastCommaIndex = expression.lastIndexOf(",");
            try {
                args[1] = Double.parseDouble(expression.substring(lastCommaIndex + 1).trim());
                expression = expression.substring(0, lastCommaIndex).trim();
            }//end try
            catch (NumberFormatException numErr) {
                throw new InputMismatchException("SYNTAX ERROR!");
            }//end catch
            lastCommaIndex = expression.lastIndexOf(",");
            try {
                args[0] = Double.parseDouble(expression.substring(lastCommaIndex + 1).trim());
                expression = expression.substring(0, lastCommaIndex).trim();
            }//end try
            catch (NumberFormatException numErr) {
            }//end catch
            catch (IndexOutOfBoundsException indErr) {
                throw new InputMismatchException("SYNTAX ERROR!");
            }//end catch

            /**
             * test for a fourth argument and report an error if one is found,
             * else exit test quietly.
             */
            lastCommaIndex = expression.lastIndexOf(",");
            try {
                args[0] = Double.parseDouble(expression.substring(lastCommaIndex + 1).trim());
                expression = expression.substring(0, lastCommaIndex).trim();
                throw new InputMismatchException(" Max of 3 args allowed! ");
            }//end try
            catch (NumberFormatException numErr) {

            }//end catch
            catch (IndexOutOfBoundsException indErr) {

            }//end catch

            if (new Double(args[0]).isNaN()) {
                setxLower(args[1]);
                setxUpper(args[2]);
//setIterations(5);
                setFunction(new Function(expression));
            }//end if
            else if (!new Double(args[0]).isNaN()) {

                setxLower(args[0]);
                setxUpper(args[1]);
                setDegree((int) args[2]);
                setFunction(new Function(expression));
            }//end else if
            else {
                throw new InputMismatchException("Invalid Integral Expression!");
            }//end else

        }//end if
        else if (!expression.startsWith("poly(")) {
            throw new InputMismatchException("Invalid Integral Expression!");
        } else if (!expression.endsWith(")")) {
            throw new InputMismatchException("Missing Closing Parenthesis");
        }

    }//end method

    /**
     * Builds the polynomial expansion of the function.
     *
     * @param precisionMode The precision mode to employ in expanding the
     * Function.
     */
    public void buildPolynomial(int precisionMode) {
        if (precisionMode == DOUBLE_PRECISION) {

            Matrix mat = getMatrix();
            mat = mat.solveEquation();
            String var = function.getIndependentVariables().get(0).getName();
            String poly = "";

            int power = 0;
            double arr[][] = mat.getArray();
            for (int rows = 0; rows < mat.getRows(); rows++, power++) {
                for (int cols = 0; cols < mat.getCols(); cols++) {

                    poly = poly.concat(arr[rows][cols] + "*" + var + "^" + power + "+");

                }//end cols
            }//end rows

            poly = poly.replace("+-", "-");
            poly = poly.replace("-+", "-");
            poly = poly.replace("--", "+");
            poly = poly.replace("++", "+");

            setPolynomial(poly.substring(0, poly.length() - 1));//remove the ending "+".

        }//end if
        else if (precisionMode == BIGDECIMAL_PRECISION) {

            PrecisionMatrix mat = getPrecisionMatrix();
            mat = mat.solveEquation();
            String var = function.getIndependentVariables().get(0).getName();
            String poly = "";

            int power = 0;
            BigDecimal arr[][] = mat.getArray();
            for (int rows = 0; rows < mat.getRows(); rows++, power++) {
                for (int cols = 0; cols < mat.getCols(); cols++) {

                    poly = poly.concat(arr[rows][cols] + var + "^" + power + "+");

                }//end cols
            }//end rows

            poly = poly.replace("+-", "-");
            poly = poly.replace("-+", "-");
            poly = poly.replace("--", "+");
            poly = poly.replace("++", "+");

            setPolynomial(poly.substring(0, poly.length() - 1));//remove the ending "+".

        }//end else if
        else {
            throw new InputMismatchException("Choose A Relevant Precision Mode.");
        }
    }//end method

    /**
     * @return the derivative of the polynomial.
     */
    public String getPolynomialDerivative() {
        return new PolynomialCalculus().differentiate();
    }

    /**
     * @return the integral of the polynomial.
     */
    public String getPolynomialIntegral() {
        return new PolynomialCalculus().integrate();
    }

    /**
     * Finds the derivative of polynomial functions generated from above. Its
     * operations are trustworthy if the powers of the polynomial variable never
     * fall below zero. i.e it is valid for n>=0
     */
    private class PolynomialCalculus {

        private ArrayList<String> scanner = new ArrayList<String>();

        public PolynomialCalculus() {
            scan();
        }

        public String getPolynomial() {
            return polynomial;
        }

        /**
         *
         * @return a scanned version of the polynomial.
         */
        public void scan() {
            scanner = new MathScanner(polynomial).scanner();
        }

        /**
         * Differentiates polynomials.
         */
        public String differentiate() {
            ArrayList<String> myScan = new ArrayList<String>(scanner);
            for (int i = 0; i < myScan.size(); i++) {
                if (isPower(myScan.get(i))) {
                    try {
                        myScan.set(i - 3, String.valueOf(Double.valueOf(myScan.get(i + 1)) * Double.valueOf(myScan.get(i - 3))));
                        myScan.set(i + 1, String.valueOf(Double.valueOf(myScan.get(i + 1)) - 1));
                    }//end try
                    catch (IndexOutOfBoundsException indexExcep) {

                    }
                }//end if

            }//end for
            String derivative = myScan.toString().replaceAll("[,| ]", "");
            derivative = derivative.substring(1);
            derivative = derivative.substring(0, derivative.length() - 1);
            derivative = derivative.replace("--", "+");
            derivative = derivative.replace("-+", "-");
            derivative = derivative.replace("+-", "-");
            derivative = derivative.replace("++", "+");

            return derivative;
        }//end method

        /**
         * Integrates polynomials.
         */
        public String integrate() {

            ArrayList<String> myScan = new ArrayList<String>(scanner);

            for (int i = 0; i < myScan.size(); i++) {
                if (isPower(myScan.get(i))) {
                    try {
                        myScan.set(i - 3, String.valueOf(Double.valueOf(myScan.get(i - 3)) / (Double.valueOf(myScan.get(i + 1)) + 1.0)));
                        myScan.set(i + 1, String.valueOf(Double.valueOf(myScan.get(i + 1)) + 1));

                    }//end try
                    catch (IndexOutOfBoundsException indexExcep) {
                    }
                }//end if

            }//end for
//remove all commas and whitespaces.
            String integral = myScan.toString().replaceAll("[,| ]", "");
            integral = integral.substring(1);//remove the starting [
            integral = integral.substring(0, integral.length() - 1);//remove the starting ]
            return integral;
        }//end method

    }//end class PolynomialCalculus

    public static void main(String args[]) {
       
        FunctionExpander polynomial = new FunctionExpander("poly(@(x)(x-1)(x+2)(3+x),1,20,4)", DOUBLE_PRECISION);//var x=1;..is to initialize the variable x.
        System.out.println(polynomial.getPolynomial());
        
        FunctionExpander expand = new FunctionExpander("poly(@(x)asin(x),0.8,1.0,25)", DOUBLE_PRECISION);//var x=1;..is to initialize the variable x.
        String poly = expand.getPolynomial();
        System.out.println("polynomial function = " + poly + "\n\n\n");
        MathExpression me = new MathExpression(poly);
        me.setDRG(1);
        me.setValue("x", "0.9999");
        System.out.println("evaluating polynomial function with normal parser = " + me.solve());
        expand.getFunction().getIndependentVariable("x").setValue("0.9999");
        System.out.println("evaluating function = " + expand.getFunction().eval());

    }//end main

}//end class
/**
 * 
 * 
 * (x-1)(x+2)(x+3) = x^3+4x^2+x-6
 *  
 */