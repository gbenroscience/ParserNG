/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.matrix.expressParser;

import java.util.ArrayList;
import java.util.HashMap;
import static parser.STRING.*;
import static parser.Operator.*;
import static parser.Variable.*;

import static parser.methods.Method.*;

/**
 *
 * @author GBENRO
 */
public class MNumber {

    /**
     * The string of digits that represent this MNumber object.
     */
    private String num;
    /**
     * The location of the MNumber in the scanner output of the parent Function
     * object that contains this MNumber object.
     */
    private int index;

    /**
     * @param num The string of digits that represent this MNumber object.
     *
     */
    public MNumber(String num) {
        this.num = num;
    }

    /**
     *
     * @param num The string of digits that represent this MNumber object.
     * @param index the location of this MOperator object in its parent Function
     * object's scanned ArrayList object.
     * @param scan The scanner output
     */
    public MNumber(String num, int index, ArrayList<String> scan) {
        this.num = isNumber(num) ? num : "";
        this.index = index;
        if (this.num.equals("")) {
            throw new IndexOutOfBoundsException("Invalid Name For Log or antilog to any base type MOperator.");
        }//end if
        else {
            this.index = (index >= 0 && scan.get(index).equals(num)) ? index : -1;
        }//end else
        if (this.index == -1) {
            throw new IndexOutOfBoundsException("Invalid Index");
        }//end if
    }//end constructor

    /**
     *
     * @param index sets the location of this MOperator object in its parent
     * Function
     */
    public void setIndex(int index) {
        this.index = index;
    }

    /**
     *
     * @return the location of this MOperator object in its parent Function
     */
    public int getIndex() {
        return index;
    }

    /**
     *
     * @param num sets the string of digits that represent this MNumber object.
     */
    public void setNum(String num) {
        this.num = num;
    }

    /**
     *
     * @return the string of digits that represent this MNumber object.
     */
    public String getNum() {
        return num;
    }

    /**
     * Built for use in my parser only.Serves to detect already and properly
     * scanned numbers in the list where string models of other objects reside,
     * e.g Variable objects,MOperator objects e.t.c It is a short cut method
     * that allows for lightning number detection in an environment that
     * contains numbers and other objects like variables, operators e.t.c as it
     * does away with matching patterns and tests just a maximum of 2 numbers to
     * know if the object is a number or not.
     *
     * @return true if the item is a number
     */
    public boolean isNumber() {
        //sign_num_._num_E_sign_num
        boolean verily = false;
        if (isDigit(firstElement(num))) {
            verily = true;
        } else if (num.length() > 1 && (firstElement(num).equals("+") || firstElement(num).equals("-"))) {
            if (isDigit(num.substring(1, 2)) || num.substring(1, 2).equals(".")) {
                verily = true;
            }
        } else if (firstElement(num).equals(".")) {
            verily = true;
        }

        return verily;
    }

    /**
     * This method may be used to test strings to see if or not they represent
     * valid numbers.
     *
     * @param num The string to test.
     * @return true if the string is a valid number
     */
    public static boolean validNumber(String num) {
        try {
            double number = Double.valueOf(num);
            return true;
        } catch (NumberFormatException numErr) {
            return false;
        }
    }

    /**
     * Built for use in my parser only.Serves to detect already and properly
     * scanned numbers in the list where string models of other objects reside,
     * e.g Variable objects,MOperator objects e.t.c
     *
     * @param num the string to be checked if it is a number or not
     * @return true if the item is a number
     */
    public static boolean isNumber(String num) {
        //sign_num_._num_E_sign_num
        boolean verily = false;
        if (isDigit(firstElement(num))) {
            verily = true;
        } else if (firstElement(num).equals("+") || firstElement(num).equals("-")) {
            if (isDigit(num.substring(1, 2)) || num.substring(1, 2).equals(".")) {
                verily = true;
            }
        } else if (firstElement(num).equals(".")) {
            verily = true;
        }

        return verily;
    }

    /**
     *
     * @return true if this MNumber object is a negative one.
     */
    public boolean isNegative() {
        return isNumber() && num.substring(0, 1).equals("-");
    }

    public MNumber getNumber() {
        return new MNumber(num);
    }

    public void validateNumber(MatrixFunction function) {
        ArrayList<String> scan = function.getScanner();
        //Numbers

        try {
            //specify valid tokens that can come before a number
            if (!isBracket(scan.get(index - 1))
                    && !isLogicOperator(scan.get(index - 1)) && !isUnaryPreOperator(scan.get(index - 1))
                    && !isBinaryOperator(scan.get(index - 1)) && !isAssignmentOperator(scan.get(index - 1))
                    && !isStatsMethod(scan.get(index - 1)) && !isNumber(scan.get(index - 1)) && !isVariableString(scan.get(index - 1))) {
                util.Utils.logError(
                        "ParserNG Does Not Allow " + num + " To Combine The Function Members \"" + scan.get(index - 1) + "\" And \"" + scan.get(index) + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Numbers!");
                function.setCorrectFunction(false);
                scan.clear();
            }//end if
            //specify valid tokens that can come after a number
            if (!isBracket(scan.get(index + 1)) && !isBinaryOperator(scan.get(index + 1))
                    && !isUnaryPostOperator(scan.get(index + 1))
                    && !isLogicOperator(scan.get(index + 1))
                    && !isUnaryPreOperator(scan.get(index + 1)) && !isNumberReturningStatsMethod(scan.get(index + 1))
                    && !isLogToAnyBase(scan.get(index + 1)) && !isAntiLogToAnyBase(scan.get(index + 1)) && !isNumber(scan.get(index + 1)) && !isVariableString(scan.get(index + 1))) {
                util.Utils.logError(
                        "ParserNG Does Not Allow " + num + " To Combine The Function Members \"" + scan.get(index) + "\" And \"" + scan.get(index + 1) + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Numbers!");
                function.setCorrectFunction(false);
                scan.clear();
            }//end if
        }//end try
        catch (IndexOutOfBoundsException ind) {

        }//end catch

    }

    public static void main(String args[]) {
        HashMap<Double,Double> map = new HashMap<>();
        
        double c = map.get(0);
        
        double v = c;
        
        System.out.println("c: "+c);
        System.out.println("v: "+v);
        
    }
}
