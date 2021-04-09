/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import math.numericalmethods.RootFinder;
import java.util.List;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.NoSuchElementException;
import java.util.Random;
import math.Maths;
import math.numericalmethods.NumericalIntegral;
import util.FunctionManager;

import java.util.Arrays;
import math.differentialcalculus.Derivative;
import math.matrix.expressParser.Matrix;
import math.quadratic.Quadratic_Equation;
import math.tartaglia.Tartaglia_Equation;
import util.VariableManager;

/**
 * Class that provides utility methods for carrying out statistical analysis on
 * a data set consisting of real numbers.
 *
 * @author GBENRO
 */
public class Set {

    /**
     * The data attribute of objects if this class
     */
    private List<String> data = new ArrayList<>();

    /**
     *
     * @param data
     */
    public Set(double... data) {

        for (int i = 0; i < data.length; i++) {
            this.data.add(data[i] + "");
        }
    }//end constructor

    /**
     * Creates a new Set object initialized with the specified data set.
     *
     * @param data the data set used to initialize the data attribute of this
     * class
     */
    public Set(List<String> data) {
        this.data = data;
    }

    /**
     * Creates a new Set object initialized with a set of data coming from a
     * mathematical MathExpression.
     *
     * @param function the Math MathExpression from which the set of data is
     * coming.
     * @param data the incoming data set
     */
    public Set(MathExpression function, List<String> data) {
        this.data = function.solveSubPortions(data);
    }//end constructor

    /**
     *
     * @param data sets the data to be operated on
     */
    public void setData(ArrayList<String> data) {
        this.data = data;
    }

    /**
     *
     * @return the data set
     */
    public List<String> getData() {
        return data;
    }

    /**
     *
     * @return the number of elements in the data set
     */
    public int size() {
        return data.size();
    }

    /**
     *
     * @return the sum of all elements in the data set
     */
    public double sum() {
        double u = 0;
        for (int i = 0; i < data.size(); i++) {
            u += Double.valueOf(data.get(i));
        }

        return u;
    }//end sum

    /**
     *
     * @return the sum of squares of values in the data set.
     */
    public double sumOfSquares() {
        double u = 0;
        for (int i = 0; i < data.size(); i++) {
            u += Math.pow(Double.valueOf(data.get(i)), 2);
        }
        return u;
    }

    /**
     *
     * @return the product of all elements in the data set.
     */

    public double prod() {
        double u = 1;
        for (int i = 0; i < data.size(); i++) {
            u *= Double.valueOf(data.get(i));
        }
        return u;
    }

    /**
     * this method determines the least value in a set of numbers
     *
     * @return the least value in a set of numbers
     * @throws NumberFormatException
     */
    public double min() {
//the maximum finder algorithm begins
        double minval = Double.valueOf(data.get(0));
        for (int i = 0; i < data.size(); i++) {
            double y = Double.valueOf(data.get(i));
            if (y < minval) {
                minval = y;
            }
        }

        return minval;
    }

    /**
     *
     * @return the maximum value in the data set.
     */
    public double max() {
//the maximum finder algorithm begins
        double maxval = Double.valueOf(data.get(0));
        for (int i = 0; i < data.size(); i++) {
            double y = Double.valueOf(data.get(i));
            if (y > maxval) {
                maxval = y;
            }
        }

        return maxval;
    }

    /**
     *
     * @return the meanor average value of a data set
     */
    public double avg() {
        return sum() / size();
    }//end avg

    /**
     *
     * @return the root mean squared value of the data set
     */
    public double rms() {
        return Math.sqrt(sumOfSquares()) / size();
    }//end rms

    /**
     *
     * @return the range of the data set
     */
    public double rng() {
        return max() - min();
    }//end range

    /**
     *
     * @return the midrange of the data set
     */
    public double mrng() {
        return 0.5 * (max() + min());
    }//end midrange

    /**
     *
     * @return the variance
     */
    public double var() {
        double u = 0;
        double u1 = 0;
        double avrg = avg();
        double N = size();
        for (int i = 0; i < N; i++) {
            u = (Double.valueOf(data.get(i)) - avrg);
            u1 += Math.pow(u, 2);
        }
        return u1 / N;
    }//end variance

    /**
     *
     * @return the standard deviation
     */
    public double std_dev() {//Finds the variance
        return Math.sqrt(var());
    }

    /**
     *
     * @return the standard error
     */
    public double std_err() {//Finds the variance
        return (std_dev() / Math.sqrt(size()));
    }

    /**
     *
     * @return the coefficient of variation
     */
    public String cov() {//Finds the coefficient of variastion
        return (100 * std_dev() / avg()) + "%";
    }

    /**
     *
     * @return displays the output of the result method as the sorting process
     * proceeds.
     */
    private List<String> displayOuputLineByLine() {
        List<String> sort = new ArrayList<String>();
        for (int i = 0; i < size(); i++) {
            sort.add(data.get(i));
        }
        return sort;
    }
    
    /**
     *
     * @return a number list sorted in ascending order
     */
    public List<String> sort() {
        List<String> sort = new ArrayList<String>();
        double[] array = new double[data.size()];
        int count = 0;
        for (String a : data) {
            array[count++] = Double.parseDouble(a);
        }
        Arrays.sort(array);
        count = 0;
        for (double d : array) {
            String c = String.valueOf(d);
            sort.add(c.endsWith(".0") ? c.substring(0 , c.length()-2) : c);
        }

        return sort;
    }//end result
    /**
     *
     * @return the inverse of the Matrix as a number list
     */
    public Matrix invert() {

        if (data.size() == 1) {
            Function f = FunctionManager.lookUp(data.get(0));
            return f.calcInverse();
        }

        int size = data.size();

        double dimen = Math.sqrt(size);

        if (Math.floor(dimen) == dimen) {
            double arr[][] = new double[(int) dimen][(int) dimen];
            int row = 0;
            int col = 0;
            int dim = (int) dimen;
            for (int i = 0; i < data.size(); i++) {
                arr[row][col] = Double.parseDouble(data.get(i));
                if (col == dim - 1) {
                    col = 0;
                    row++;
                } else {
                    col++;
                }
            }
            return new Matrix(arr).inverse();

        }
        return new Matrix("");
    }

    /**
     * The input list must be an n rows by n+1 columns matrix
     *
     * @return a the solution to the system of equations represented by the
     * number list
     */
    public Matrix solveSystem() {

        if (data.size() == 1) {
            Function f = FunctionManager.lookUp(data.get(0));
            return f.getMatrix().solveEquation();
        }

        List<String> result = new ArrayList<String>();
        int size = data.size();

        double dimen = Math.sqrt(size);

        int numRows = (int) dimen;
        int numCols = numRows + 1;

        if (numRows * numCols == size) {

            double arr[][] = new double[numRows][numCols];
            int row = 0;
            int col = 0;
            for (int i = 0; i < size; i++) {
                arr[row][col] = Double.parseDouble(data.get(i));
                if (col == numCols - 1) {
                    col = 0;
                    row++;
                } else {
                    col++;
                }
            }
            return new Matrix(arr).solveEquation();

        }
        return null;
    }

    /**
     * The input list is such that: The first 2 entries specify the number of
     * rows and columns. The remaining entries are the Matrix's entries.
     *
     * @return a number list which represents the row-reduced echelon Matrix
     * that the entries reduce to.
     */
    public Matrix listToMatrix() {

        int numRows = Integer.parseInt(data.get(0));
        int numCols = Integer.parseInt(data.get(1));

        data.subList(0, 2).clear();
        int size = data.size();

        if (numRows * numCols == size) {

            double arr[][] = new double[numRows][numCols];
            int row = 0;
            int col = 0;
            for (int i = 0; i < size; i++) {
                arr[row][col] = Double.parseDouble(data.get(i));
                if (col == numCols - 1) {
                    col = 0;
                    row++;
                } else {
                    col++;
                }
            }
            return new Matrix(arr);

        }
        return null;
    }

    /**
     * The input list is such that: The first 2 entries specify the number of
     * rows and columns. The remaining entries are the Matrix's entries.
     *
     * @return a number list which represents the row-reduced echelon Matrix
     * that the entries reduce to.
     */
    public Matrix echelon() {

        if (data.size() == 1) {
            Function f = FunctionManager.lookUp(data.get(0));
            return f.reduceToEchelon();
        }

        List<String> result = new ArrayList<String>();

        int numRows = Integer.parseInt(data.get(0));
        int numCols = Integer.parseInt(data.get(1));

        data.subList(0, 2).clear();
        int size = data.size();

        if (numRows * numCols == size) {

            double arr[][] = new double[numRows][numCols];
            int row = 0;
            int col = 0;
            for (int i = 0; i < size; i++) {
                arr[row][col] = Double.parseDouble(data.get(i));
                if (col == numCols - 1) {
                    col = 0;
                    row++;
                } else {
                    col++;
                }
            }
            return new Matrix(arr).reduceToRowEchelonMatrix();

        }
        return null;
    }

    /**
     * The input list is such that: The first 2 entries specify the number of
     * rows and columns. The remaining entries are the Matrix's entries.
     *
     * @return a number list which represents the triangular Matrix that the
     * entries reduce to.
     */
    public Matrix triMatrix() {

        if (data.size() == 1) {
            Function f = FunctionManager.lookUp(data.get(0));
            return f.triangularMatrix();
        }

        int numRows = Integer.parseInt(data.get(0));
        int numCols = Integer.parseInt(data.get(1));
        data.subList(0, 2).clear();
        int size = data.size();

        if (numRows * numCols == size) {

            double arr[][] = new double[numRows][numCols];
            int row = 0;
            int col = 0;
            for (int i = 0; i < size; i++) {
                arr[row][col] = Double.parseDouble(data.get(i));
                if (col == numCols - 1) {
                    col = 0;
                    row++;
                } else {
                    col++;
                }
            }
            return new Matrix(arr).reduceToTriangularMatrix();

        }
        return null;
    }

    /**
     *
     * @return the median of the data set
     */
    public double median() {
        double median = 0;
 
        List<String> scan = new ArrayList<String>();
        scan = sort();
        int g = size();
        if (STRING.isEven(g)) {
            double med = (Double.valueOf(scan.get(g / 2)) + Double.valueOf(scan.get((g / 2) - 1))) / 2;
            median = med;
        } else if (!STRING.isEven(g)) {
            median = Double.valueOf(scan.get((g - 1) / 2));
        }

        return median;
    }

    /**
     *
     * @return the mode of a number set as a list
     */
    public String mode() {
        int[] freq = new int[data.size()];
        int g = freq.length;
        int count = 0;
        //[3,5,1,2,5,1,2,5,3,2]
        for (int j = 0; j < g; j++) {
            for (int i = 0; i < g; i++) {
                if (Double.parseDouble(data.get(i)) == Double.parseDouble(data.get(j))) {
                    count++;
                }//end if
            }//end for
            freq[j] = count;
            count = 0;
        }//end for
//Get the highest frequency
        double big = 0;
        for (int i = 0; i < freq.length; i++) {
            if (big < freq[i]) {
                big = freq[i];
            }//end if
        }//end for

//Now get the values with the highest frequency calculated above.
        List<String> mode = new ArrayList<String>();
        for (int i = 0; i < freq.length; i++) {
            if (freq[i] == big && !mode.contains(data.get(i))) {
                mode.add(data.get(i));
            }//end if

        }// end for
        String dMode = String.valueOf(mode);
        dMode = STRING.delete(STRING.delete(dMode, "["), "]");
        return dMode;
        //return "The number(s) "+mode+" with "+big+" occurences has/have the highest frequency ";
    }

    /**
     *
     * if no value is found in the data set, the software will generate floating
     * point values randomly between 0.0 and 1.0 ( 0.0 inclusive and 1.0
     * exclusive). Else: If the data set has only one number, e.g [m] this
     * method will randomly generate a number between 0 and m-1 If the list has
     * 2 numbers, say m and n, e.g [m,n] The method will generate n numbers
     * between 0 and m-1
     *
     *
     *
     * @return a list of values generated randomly according to the format of
     * the random command.
     */
    public List<String> random() {
        List<String> vals = new ArrayList<String>();
        Random rnd = new Random();
//check if the random command came with only one argument
        if (size() == 1) {
            int val = Integer.valueOf(data.get(0));
            data.set(0, String.valueOf((int) val));
        } //check if it came with 2 arguments
        else if (size() == 2) {
            double val = Double.valueOf(data.get(0));
            double val1 = Double.valueOf(data.get(1));
            data.set(0, String.valueOf((int) val));
            data.set(1, String.valueOf((int) val1));
        }

        if (size() == 0) {
            vals.add(String.valueOf(rnd.nextFloat()));
        } else if (size() == 1) {
            String num = String.valueOf(rnd.nextInt(Integer.valueOf(data.get(0))));
            vals.add(num);
        } else if (size() == 2) {

            for (int i = 0; i < Integer.valueOf(data.get(1)); i++) {
                vals.add(String.valueOf(rnd.nextInt(Integer.valueOf((data.get(0))))));
            }
        } else {
            throw new NoSuchElementException("Allowed Formats For rnd Are rnd(),rnd1(number),"
                    + "rnd1( number1,number2)");
        }
        data.clear();
        data.addAll(vals);
        return vals;
    }

    /**
     *
     * @return the permutation of 2 values.
     */
    public String permutation() {
        String n = data.get(0);
        String r = data.get(1);

        double n_Factorial = Double.parseDouble(Maths.fact(n));
        double n_Minus_r_Factorial = Double.parseDouble(Maths.fact(String.valueOf(Double.valueOf(n) - Double.valueOf(r))));

        return String.valueOf(n_Factorial / n_Minus_r_Factorial);
    }

    /**
     *
     * @return the combination of 2 values.
     */
    public String combination() {
        String n = data.get(0);
        String r = data.get(1);

        double n_Factorial = Double.parseDouble(Maths.fact(n));
        double n_Minus_r_Factorial = Double.parseDouble(Maths.fact(String.valueOf(Double.valueOf(n) - Double.valueOf(r))));
        double r_Factorial = Double.parseDouble(Maths.fact(r));

        return String.valueOf(n_Factorial / (n_Minus_r_Factorial * r_Factorial));
    }

    /**
     *
     * @return Raises the number in index 0 to a power equal to the number in
     * index 1.
     */
    public String power() {
        double n = Double.parseDouble(data.get(0));
        double pow = Double.parseDouble(data.get(1));
        return String.valueOf(Math.pow(n, pow));
    }

    /**
     * Finds the numerical derivative of a Function which has been pre-defined
     * in the Workspace.
     *
     * @return the derivative at the specified value of the horizontal
     * coordinate.
     */
    public String differentiate() {

        int sz = data.size();
        switch (sz) {
            case 1:
            {
                String anonFunc = data.get(0);
                String solution = Derivative.eval("diff(" + anonFunc + ",1)");
                
                return solution;
            }
            case 2:
            {
                String anonFunc = data.get(0);
                double value = Double.valueOf(data.get(1));
                
                String solution = Derivative.eval("diff(" + anonFunc + "," + value + ")");
                /*  NumericalDerivative der = new NumericalDerivative(new Function(anonFunc), value );
                return der.findDerivativeByPolynomialExpander();*/
                return solution;
            }//end if
            case 3:
            {
                String anonFunc = data.get(0);
                double value = Double.valueOf(data.get(1));
                int order = Integer.parseInt(data.get(2));
                /*  NumericalDerivative der = new NumericalDerivative(FunctionManager.lookUp(data.get(0)),Double.valueOf(data.get(1)));
                return der.findDerivativeByPolynomialExpander();
                */
                String solution = Derivative.eval("diff(" + anonFunc + "," + value + "," + order + ")");
                
                return solution;
            }//end else if
            default:
                throw new InputMismatchException(" Parameter List " + data + " Is Invalid!");
        }
    }

    public String quadraticRoots() {
        int sz = data.size();
        //The function may come as a Function name to be looked up or as an anonymous Function.
        //If it comes as an anonymous Function,then the scanning process of the parser may
        //have broken it down into unmeaningful parts,whatever the situation, piece them together again!
        String function = "";
        for (int i = 0; i < sz; i++) {
            function = function.concat((String) data.get(i));
        }//end for

        Function f = FunctionManager.lookUp(function);

        String input = f.expressionForm();
        input = input.substring(1);//remove the @
        int closeBracOfAt = Bracket.getComplementIndex(true, 0, input);
        input = input.substring(closeBracOfAt + 1);
        if (input.startsWith("(")) {
            input = input.substring(1, input.length() - 1);
            input = input.concat("=0");
        }

        Quadratic_Equation solver = new Quadratic_Equation(input);

        //String reducedForm=solver.interpretedSystem();
        String soln = solver.solutions();

        return soln;
    }

    public String tartaglianRoots() {
        int sz = data.size();
        //The function may come as a Function name to be looked up or as an anonymous Function.
        //If it comes as an anonymous Function,then the scanning process of the parser may
        //have broken it down into unmeaningful parts,whatever the situation, piece them together again!
        String function = "";
        for (int i = 0; i < sz; i++) {
            function = function.concat((String) data.get(i));
        }//end for

        Function f = FunctionManager.lookUp(function);

        String input = f.expressionForm();
        input = input.substring(1);//remove the @
        int closeBracOfAt = Bracket.getComplementIndex(true, 0, input);
        input = input.substring(closeBracOfAt + 1);
        if (input.startsWith("(")) {
            input = input.substring(1, input.length() - 1);
            input = input.concat("=0");
        }
 
        Tartaglia_Equation solver = new Tartaglia_Equation(input);

        String reducedForm = solver.interpretedSystem();
        String soln = solver.solutions();

        return soln;
    }

    /**
     * Finds the zero of a Function which has been pre-defined in the Workspace.
     *
     * @return one of the roots of the Function object.
     */
    public String rootOfEquation() {
        int sz = data.size();
        boolean has2NumberArguments = false;
 System.out.println("LIST: "+data);
        if (Number.validNumber((String) data.get(sz - 1)) && !Number.validNumber((String) data.get(sz - 2))) {
            has2NumberArguments = true;
        }//end if

        //The function may come as a Function name to be looked up or as an anonymous Function.
        //If it comes as an anonymous Function,then the scanning process of the parser may
        //have broken it down into unmeaningful parts,whatever the situation, piece them together again!
        String function = "";
        if (has2NumberArguments) {
            for (int i = 0; i < sz - 1; i++) {
                function = function.concat((String) data.get(i));
            }//end for

            RootFinder rf = new RootFinder(FunctionManager.lookUp(function), Double.parseDouble((String) data.get(sz - 1)));
            return rf.findRoots();

        }//end if
        return "Function Syntax Error!";
    }

    /**
     * Finds the numerical integral of a Function which has been pre-defined in
     * the Workspace.
     *
     * @return the derivative at the specified value of the horizontal
     * coordinate.
     */
    public String integrate() {
        int sz = data.size();

        boolean has3NumberArguments = false;
        boolean has2NumberArguments = false;
        if (Number.validNumber(data.get(sz - 1)) && Number.validNumber(data.get(sz - 2)) && Number.validNumber(data.get(sz - 3))) {
            has3NumberArguments = true;
       
        }//end if
        if (Number.validNumber(data.get(sz - 1)) && Number.validNumber(data.get(sz - 2)) && !Number.validNumber(data.get(sz - 3))) {
            has2NumberArguments = true;
        }//end if

        //The function may come asa Function name to be looked up or as an anonmous Function.
        //If it comes as an anonymous Function,then the scanning process of the parser may
        //have broken it down into unmeaningful parts,whatever the situation, piece them together again!
        String function = "";
        if (has2NumberArguments) {
            for (int i = 0; i < sz - 2; i++) {
                function = function.concat(data.get(i));
            }//end for

            NumericalIntegral intg = new NumericalIntegral(Double.valueOf(data.get(sz - 2)), Double.valueOf(data.get(sz - 1)), 0, function);
            return String.valueOf(intg.findHighRangeIntegral()
            );

        }//end if
        else if (has3NumberArguments) {
            for (int i = 0; i < sz - 3; i++) {
                function = function.concat(data.get(i));
            }//end for

            NumericalIntegral intg = new NumericalIntegral(Double.valueOf(data.get(sz - 3)), Double.valueOf(data.get(sz - 2)),
                    (int) Math.round(Double.parseDouble(data.get(sz - 1))), function);
            double val = intg.findHighRangeIntegral();
            return String.valueOf(val);

        }//end else if
        return "Function Syntax Error!";
    }

    /**
     *
     * @return the determinant or {@link Double#NaN} if the determinant does not
     * exist.
     */
    public double determinant() {

        if (data.size() == 1) {
            Function f = FunctionManager.lookUp(data.get(0));
            return f.calcDet();
        }

        int size = data.size();

        double dimen = Math.sqrt(size);

        if (Math.floor(dimen) == dimen) {
            double arr[][] = new double[(int) dimen][(int) dimen];
            int row = 0;
            int col = 0;
            int dim = (int) dimen;
            for (int i = 0; i < data.size(); i++) {
                arr[row][col] = Double.parseDouble(data.get(i));
                if (col == dim - 1) {
                    col = 0;
                    row++;
                } else {
                    col++;
                }
            }
            Matrix mat = new Matrix(arr);
            return mat.determinant();
        }

        return Double.NaN;
    }//end result

    /**
     * The list must have been originally supplied: matrix_mul(A,B) {where A and
     * B are matrices} or matrix_mul(A,k) {where A is a matrix and k is a
     * scalar} It multiplies them out and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix product (scalar or vector)
     * depending on the second argument.
     */
    public Matrix multiplyMatrix() {

        if (data.size() == 2) {

            String token = data.get(0);
            String nextToken = data.get(1);

            if (Variable.isVariableString(token)) {
                if (Variable.isVariableString(nextToken)) {
                    Function f = FunctionManager.lookUp(token);
                    if (f == null) {
                        return new Matrix("");
                    }
                    Function f1 = FunctionManager.lookUp(nextToken);
                    if (f1 != null) {
                        return Matrix.multiply(f.getMatrix(), f1.getMatrix());
                    } else {
                        Variable v = VariableManager.lookUp(nextToken);
                        if (v != null) {
                            double val = Double.parseDouble(v.getValue());
                            return f.getMatrix().scalarMultiply(val);
                        }

                    }

                } else if (Number.validNumber(nextToken)) {
                    Function f = FunctionManager.lookUp(token);

                    return f.getMatrix().scalarMultiply(Double.parseDouble(nextToken));
                }

            }

        }

        return new Matrix("");

    }

    /**
     * The list must have been originally supplied: matrix_div(A,B) {where A and
     * B are matrices} or matrix_div(A,k) {where A is a matrix and k is a
     * scalar} It divides them out and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix division (scalar or
     * vector) depending on the second argument.
     */
    public Matrix divideMatrix() {

        if (data.size() == 2) {

            String token = data.get(0);
            String nextToken = data.get(1);

            if (Variable.isVariableString(token)) {
                if (Variable.isVariableString(nextToken)) {
                    Function f = FunctionManager.lookUp(token);
                    if (f == null) {
                        return new Matrix("");
                    }

                    Function f1 = FunctionManager.lookUp(nextToken);
                    if (f1 != null) {
                        return Matrix.multiply(f.getMatrix(), f1.getMatrix().inverse());
                    } else {
                        Variable v = VariableManager.lookUp(nextToken);
                        if (v != null) {
                            double val = Double.parseDouble(v.getValue());
                            return f.getMatrix().scalarDivide(val);
                        }

                    }

                } else if (Number.validNumber(nextToken)) {
                    Function f = FunctionManager.lookUp(token);

                    return f.getMatrix().scalarDivide(Double.parseDouble(nextToken));
                }

            }

        }
        return new Matrix("");

    }

    /**
     * The list must have been originally supplied: matrix_add(A,B) {where A and
     * B are matrices} It adds them out and returns the result as a list.
     *
     * @return a {@link Matrix}containing the matrix addition (scalar or
     * vector).
     */
    public Matrix addMatrix() {

        if (data.size() == 2) {

            String token = data.get(0);
            String nextToken = data.get(1);

            if (Variable.isVariableString(token)) {
                if (Variable.isVariableString(nextToken)) {
                    Function f = FunctionManager.lookUp(token);

                    Function f1 = FunctionManager.lookUp(nextToken);
                    if (f != null && f1 != null) {
                        return f.getMatrix().add(f1.getMatrix());
                    }

                }

            }

        }

        return new Matrix("");

    }

    /**
     * The list must have been originally supplied: matrix_sub(A,B) {where A and
     * B are matrices} It subtracts them out and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix difference.
     */
    public Matrix subtractMatrix() {

        if (data.size() == 2) {

            String token = data.get(0);
            String nextToken = data.get(1);

            if (Variable.isVariableString(token)) {
                if (Variable.isVariableString(nextToken)) {
                    Function f = FunctionManager.lookUp(token);

                    Function f1 = FunctionManager.lookUp(nextToken);
                    if (f != null && f1 != null) {
                        return f.getMatrix().subtract(f1.getMatrix());
                    }

                }

            }

        }

        return new Matrix("");

    }

    /**
     * The list must have been originally supplied: matrix_pow(A,n) {where A is
     * a Matrix and n is an integer. If n is a double, it is rounded down to the
     * nearest int.} It raises the Matrix to the power of the second argument
     * and returns the result as a list.
     *
     * @return a Matrix containing the matrix's power.
     */
    public Matrix powerMatrix() {

        if (data.size() == 2) {

            String token = data.get(0);
            String nextToken = data.get(1);

            if (Variable.isVariableString(token)) {
                if (Variable.isVariableString(nextToken) || Number.validNumber(nextToken)) {
                    Function f = FunctionManager.lookUp(token);
                    double pow = Variable.isVariableString(nextToken) ? Double.parseDouble(VariableManager.lookUp(nextToken).getValue()) : Double.parseDouble(nextToken);

                    if (f != null) {
                        return Matrix.pow(f.getMatrix(), (int) pow);
                    }

                }

            }

        }

              throw new InputMismatchException("Bad args for matrix powers");

    }

    /**
     * The list must have been originally supplied: transpose(A) {where A is a
     * Matrix} It transposes A and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix transpose.
     */
    public Matrix transpose() {

        if (data.size() == 1) {

            String token = data.get(0);

            if (Variable.isVariableString(token)) {
                Function f = FunctionManager.lookUp(token);

                if (f != null) {
                    return f.getMatrix().transpose();
                }

            }

        }
        
        throw new InputMismatchException("Bad args for matrix transpose");
 

    }
   /**
     * The list must have been originally supplied: adjoint(A) {where A is a
     * Matrix} It finds the adjoint of A and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix transpose.
     */
    public Matrix adjoint() {

        if (data.size() == 1) {

            String token = data.get(0);

            if (Variable.isVariableString(token)) {
                Function f = FunctionManager.lookUp(token);

                if (f != null) {
                    return f.getMatrix().adjoint();
                }

            }

        }
        
        throw new InputMismatchException("Bad args for matrix adjoint");
 

    }
      /**
     * The list must have been originally supplied: adjoint(A) {where A is a
     * Matrix} It finds the adjoint of A and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix transpose.
     */
    public Matrix cofactorMatrix() {

        if (data.size() == 1) {

            String token = data.get(0);

            if (Variable.isVariableString(token)) {
                Function f = FunctionManager.lookUp(token);

                if (f != null) {
                    return f.getMatrix().getCofactorMatrix();
                }

            }

        }
        
        throw new InputMismatchException("Bad args for matrix cofactors");
 

    }
      /**
     * The list must have been originally supplied: eigvec(A) {where A is a
     * Matrix} It finds the eigenvalues of A and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix transpose.
     */
    public Matrix eigenVectors() {

        if (data.size() == 1) {

            String token = data.get(0);

            if (Variable.isVariableString(token)) {
                Function f = FunctionManager.lookUp(token);

                if (f != null) {
                    return f.getMatrix().transpose();
                }

            }

        }
        
        throw new InputMismatchException("Bad args for matrix eigenValues");
 

    }
    
    private static final void printImpl(String data){
        System.out.println("ParserNG> "+data);
    }
       /**
     * The list must have been originally supplied: eigvec(A) {where A is a
     * Matrix} It finds the eigenvalues of A and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix transpose.
     */
    public void print() {

        if (data.size() == 1) {

            String token = data.get(0);

            if (Variable.isVariableString(token)) {
                Function f = FunctionManager.lookUp(token);
                
                if (f != null) {
                    
                    switch(f.getType()){
                        case Function.ALGEBRAIC:
                             printImpl(f.toString());
                              break;
                        case Function.MATRIX:
                            printImpl(f.getMatrix().toString());
                              break;
                        case Function.LIST:
                            printImpl(f.getMatrix().toString());
                              break;
                        default:
                            printImpl(f.toString());
                              break;
                    }
          
                }else{
                    Variable v = VariableManager.getVariable(token);
                    if(v != null){
                      printImpl(v.toString());  
                    }
                }

            }else if(parser.Number.isNumber(token)){
                printImpl(token);
            }
       

            return;
        }
        
        throw new InputMismatchException("Bad args for printing");
 

    }
      /**
     * The list must have been originally supplied: eigPoly(A) {where A is a
     * Matrix} It finds the characterisic polynomial whose solution yields the eigenvalues of A and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix transpose.
     */
    public String eigenPoly() {

        if (data.size() == 1) {

            String token = data.get(0);
 

            if (Variable.isVariableString(token)) {
                Function f = FunctionManager.lookUp(token);

                if (f != null) {
                    return f.getMatrix().getCharacteristicPolynomialForEigenVector();
                }

            }

        }
        
        throw new InputMismatchException("Bad args for matrix eigenVectors");
 

    }
    /**
     * The list must have been originally supplied: transpose(A) {where A is a
     * Matrix} It transposes A and returns the result as a list.
     *
     * @return a {@link Matrix} containing the matrix transpose.
     */
    public Matrix editMatrix() {

 
        if (data.size() == 4) {

            String token = data.get(0);

            if (Variable.isVariableString(token)) {
                Function f = FunctionManager.lookUp(token);
                if (f != null) {
                    Matrix m = f.getMatrix();
                    int row = Integer.parseInt(data.get(1));
                    int col = Integer.parseInt(data.get(2));
                    
                    
                    double val = Double.parseDouble(data.get(3));
                    boolean updated = m.update(val, row, col);
                    if(!updated){
                        throw new InputMismatchException("Trying to update outside matrix bounds. Matrix has: "+m.getRows()+" rows and "+m.getCols()+" columns\n But input was: row = "+row+" rows and columns = "+col);
                    }
                    return m;
                }
            }
        }
         throw new InputMismatchException("Bad args for matrix edit");
 

    }

    /**
     *
     * @param operator The operator.
     * @return the value of the user defined function.
     * @throws ClassNotFoundException if the function was never defined by the
     * user.
     */
    public String evaluateUserDefinedFunction(String operator) throws ClassNotFoundException {
        int sz = data.size();
        String fullname = operator.concat("(");
        for (int i = 0; i < sz; i++) {
            fullname = fullname.concat(data.get(i).concat(","));
        }//end for loop

        fullname = fullname.substring(0, fullname.length() - 1);
        fullname = fullname.concat(")");
        try {
            return FunctionManager.getFunction(operator).evalArgs(fullname);
        }//end try
        catch (Exception cnfe) {
            throw new ClassNotFoundException("Could Not Find Function! " + fullname);
        }

    }

}//end class
