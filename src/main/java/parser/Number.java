/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import parser.methods.Method;

import java.util.ArrayList;

/**
 *
 * @author GBENRO
 */
public class Number {

    /**
     * The string of digits that represent this Number object.
     */
    private String num;
    /**
     * The location of the Number in the scanner output of the parent
     * MathExpression object that contains this Number object.
     */
    private int index;

    /**
     * @param num The string of digits that represent this Number object.
     *
     */
    public Number(String num) {
        this.num = num;
    }

    /**
     *
     * @param num The string of digits that represent this Number object.
     * @param index the location of this Operator object in its parent
     * MathExpression object's scanned ArrayList object.
     */
    public Number(String num, int index, ArrayList<String> scan) {
        this.num = isNumber(num) ? num : "";
        this.index = index;
        if (this.num.equals("")) {
            throw new IndexOutOfBoundsException("Invalid Name For Log or antilog to any base type Operator.");
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
     * @param index sets the location of this Operator object in its parent
     * MathExpression
     */
    public void setIndex(int index) {
        this.index = index;
    }

    /**
     *
     * @return the location of this Operator object in its parent MathExpression
     */
    public int getIndex() {
        return index;
    }

    /**
     *
     * @param num sets the string of digits that represent this Number object.
     */
    public void setNum(String num) {
        this.num = num;
    }

    /**
     *
     * @return the string of digits that represent this Number object.
     */
    public String getNum() {
        return num;
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
     * e.g Variable objects,Operator objects e.t.c
     *
     * @param num the string to be checked if it is a number or not
     * @return true if the item is a number
     */
    public static boolean isNumber(String num) {
        //sign_num_._num_E_sign_num
        boolean verily = false;
        if (STRING.isDigit(STRING.firstElement(num))) {
            verily = true;
        } else if (STRING.firstElement(num).equals(Operator.PLUS) || STRING.firstElement(num).equals(Operator.MINUS)) {
            if (num.length() > 1) {
                if (STRING.isDigit(num.substring(1, 2)) || num.substring(1, 2).equals(".")) {
                    verily = true;
                }
            }
        } else if (STRING.firstElement(num).equals(".")) {
            verily = true;
        }

        return verily;
    }

    public static boolean isNegative(String num) {
        return isNumber(num) && num.substring(0, 1).equals("-");
    }

    public Number getNumber() {
        return new Number(num);
    }

    public void validateNumber(MathExpression function) {
        ArrayList<String> scan = function.getScanner();
        //Numbers

        try {
            //specify valid tokens that can come before a number
            if (!Operator.isBracket(scan.get(index - 1))
                    && !Operator.isLogicOperator(scan.get(index - 1)) && !Operator.isUnaryPreOperator(scan.get(index - 1))
                    && !Operator.isBinaryOperator(scan.get(index - 1)) && !Operator.isAssignmentOperator(scan.get(index - 1))
                    && !Method.isStatsMethod(scan.get(index - 1)) && !isNumber(scan.get(index - 1)) && !Variable.isVariableString(scan.get(index - 1))) {
                util.Utils.logError(
                        "ParserNG Does Not Allow " + num + " To Combine The Function Members \"" + scan.get(index - 1) + "\" And \"" + scan.get(index) + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Numbers!");
                function.setCorrectFunction(false);
                scan.clear();
            }//end if
            //specify valid tokens that can come after a number
            if (!Operator.isBracket(scan.get(index + 1)) && !Operator.isBinaryOperator(scan.get(index + 1))
                    && !Operator.isUnaryPostOperator(scan.get(index + 1))
                    && !Operator.isLogicOperator(scan.get(index + 1))
                    && !Operator.isUnaryPreOperator(scan.get(index + 1)) && !Method.isNumberReturningStatsMethod(scan.get(index + 1))
                    && !Method.isLogToAnyBase(scan.get(index + 1)) && !Method.isAntiLogToAnyBase(scan.get(index + 1)) && !isNumber(scan.get(index + 1)) && !Variable.isVariableString(scan.get(index + 1))) {
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
        System.out.println(isNumber("-90g"));
        System.out.println(validNumber("-90g"));
    }
}
