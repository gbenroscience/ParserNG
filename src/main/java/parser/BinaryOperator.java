/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import parser.methods.Method;

import java.util.ArrayList;

/**
 *
 * @author GBEMIRO
 */
public class BinaryOperator extends Operator implements Validatable {

    /**
     * The precedence of this BinaryOperator object.
     */
    private final Precedence precedence;
    /**
     * The index of this operator in the scanned Function that it belongs to.
     */
    private int index;

    /**
     *
     * @param name The bame of the operator
     * @param index The index of the operator
     * @param scan The scanner output
     */
    public BinaryOperator(String name, int index, ArrayList<String> scan) {
        super(isBinaryOperator(name) ? name : "");

        if (this.getName().equals("")) {
            throw new IndexOutOfBoundsException("Invalid Name For Binary Operator.");
        }//end if
        else {
            this.index = (index >= 0 && scan.get(index).equals(name)) ? index : -1;
            this.precedence = Operator.getPrecedence(name);
        }//end else

        if (this.index == -1) {
            throw new IndexOutOfBoundsException("Invalid Index");
        }
    }//end constructor

    /**
     *
     * @return the precedence of this operator
     */
    public Precedence getPrecedence() {
        return precedence;
    }


    /**
     *
     * @param index sets the index of this operator in the scanned Function that
     * it belongs to.
     * @param scan The Function object that this object exists in.
     */
    public void setIndex(int index, ArrayList<String> scan) {

        this.index = (index >= 0 && scan.get(index).equals(this.getName())) ? index : -1;

        if (this.index == -1) {
            throw new IndexOutOfBoundsException("Invalid Index");
        }
    }

    /**
     *
     * @return the index of this operator in the scanned Function that it
     * belongs to.
     */
    public int getIndex() {
        return index;
    }

  
    /**
     * 
     * @param scan the Function object that this BinaryOperator object
     * exists in. validates the grammatical usage of this operator (by leaving
     * the correctFunction attribute of the function object un-modified) if the
     * usage of this operator in its immediate environment i.e to its left and
     * right is correct.
     * @return true if valid
     */
    @Override
    public boolean validate(ArrayList<String> scan) {

        boolean correct = true;
        try {
            //specify valid tokens that can come before a binary operator
            if (isPlusOrMinus(scan.get(index))) {

                if (!Number.isNumber(scan.get(index - 1)) && !Variable.isVariableString(scan.get(index - 1))
                        && !isUnaryPostOperator(scan.get(index - 1)) && !isClosingBracket(scan.get(index - 1)) && !isOpeningBracket(scan.get(index - 1))) {
                    util.Utils.logError(
                            "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + scan.get(index - 1) + "\" And \"" + scan.get(index) + "\""
                            + " As You Have Done."
                            + "ParserNG Error Detector For Binary Operators!");
                    System.out.println(scan);
                    correct = false;
                    scan.clear();

                }
                //specify valid tokens that can come after a binary operator
                if (!Number.isNumber(scan.get(index + 1)) && !Variable.isVariableString(scan.get(index + 1))
                        && !isOpeningBracket(scan.get(index + 1))
                        && !Method.isUnaryPreOperatorORDefinedMethod(scan.get(index + 1)) && !Method.isNumberReturningStatsMethod(scan.get(index + 1))
                        && !Method.isLogToAnyBase(scan.get(index + 1)) && !Method.isAntiLogToAnyBase(scan.get(index + 1))) {
                    util.Utils.logError(
                            "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + scan.get(index) + "\" And \"" + scan.get(index + 1) + "\""
                            + " As You Have Done."
                            + "ParserNG Error Detector For Binary Operators!");
                    System.out.println(scan);
                    correct = false;
                    scan.clear();
                }//end if

            }//end if
            else if (!isPlusOrMinus(scan.get(index))) {
                if (!Number.isNumber(scan.get(index - 1)) && !Variable.isVariableString(scan.get(index - 1))
                        && !isUnaryPostOperator(scan.get(index - 1)) && !isClosingBracket(scan.get(index - 1))) {
                    util.Utils.logError(
                            "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + scan.get(index - 1) + "\" And \"" + scan.get(index) + "\""
                            + " As You Have Done."
                            + "ParserNG Error Detector For Binary Operators!");
                    System.out.println(scan);
                    correct = false;
                    scan.clear();
                }//end if
                //specify valid tokens that can come after a binary operator
                if (!Number.isNumber(scan.get(index + 1)) && !Variable.isVariableString(scan.get(index + 1))
                        && !isOpeningBracket(scan.get(index + 1))
                        && !Method.isUnaryPreOperatorORDefinedMethod(scan.get(index + 1)) && !Method.isNumberReturningStatsMethod(scan.get(index + 1))
                        && !Method.isLogToAnyBase(scan.get(index + 1)) && !Method.isAntiLogToAnyBase(scan.get(index + 1))) {
                    util.Utils.logError(
                            "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + scan.get(index) + "\" And \"" + scan.get(index + 1) + "\""
                            + " As You Have Done."
                            + "ParserNG Error Detector For Binary Operators!");
                    System.out.println(scan);
                    correct = false;
                    scan.clear();
                }//end if

            }//end else if

        }//end try
        catch (IndexOutOfBoundsException ind) {

        }//end catch

        return correct;
    }

}//end class
