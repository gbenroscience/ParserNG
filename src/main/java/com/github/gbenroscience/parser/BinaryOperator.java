/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.parser;

import com.github.gbenroscience.parser.methods.Method;

import java.util.ArrayList;
import java.util.List;

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
     * @param name The name of the operator
     * @param index The index of the operator
     * @param scan The scanner output
     */
    public BinaryOperator(String name, int index, List<String> scan) {
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
    public boolean validate(List<String> scan) {


        int leftInd = index-1;
        int rightInd = index+1;
        int sz = scan.size();
        boolean correct = true;
        String prev = leftInd >= 0 ? scan.get(leftInd) : null;
        String curr = scan.get(index);
        String next = rightInd < sz ? scan.get(rightInd) : null;
  
        try {
            //specify valid tokens that can come before a binary operator
            if (isPlusOrMinus(curr)) {

                if (leftInd>=0 && !Number.isNumber(prev) && !Variable.isVariableString(prev)
                        && !isUnaryPostOperator(prev) && !isClosingBracket(prev) && !isOpeningBracket(prev)) {
                    com.github.gbenroscience.util.Utils.logError(
                            "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + prev + "\" And \"" + curr + "\""
                            + " As You Have Done."
                            + "ParserNG Error Detector For Binary Operators!");
                    correct = false;
                    scan.clear();

                }
                //specify valid tokens that can come after a binary operator
                if (rightInd<sz && !Number.isNumber(next) && !Variable.isVariableString(next)
                        && !isOpeningBracket(next)
                        && !Method.isUnaryPreOperatorORDefinedMethod(next) && !Method.isNumberReturningStatsMethod(next)
                        && !Method.isLogToAnyBase(next) && !Method.isAntiLogToAnyBase(next)) {
                    com.github.gbenroscience.util.Utils.logError(
                            "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + curr + "\" And \"" + next + "\""
                            + " As You Have Done."
                            + "ParserNG Error Detector For Binary Operators!");
                    correct = false;
                    scan.clear();
                }//end if

            }//end if
            else if (!isPlusOrMinus(curr)) {
                if (leftInd>=0 && !Number.isNumber(prev) && !Variable.isVariableString(prev)
                        && !isUnaryPostOperator(prev) && !isClosingBracket(prev)) {
                    com.github.gbenroscience.util.Utils.logError(
                            "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + prev + "\" And \"" + curr + "\""
                            + " As You Have Done."
                            + "ParserNG Error Detector For Binary Operators!");
                    correct = false;
                    scan.clear();
                }//end if
                //specify valid tokens that can come after a binary operator
                if (rightInd<sz && !Number.isNumber(next) && !Variable.isVariableString(next)
                        && !isOpeningBracket(next)
                        && !Method.isUnaryPreOperatorORDefinedMethod(next) && !Method.isNumberReturningStatsMethod(next)
                        && !Method.isLogToAnyBase(next) && !Method.isAntiLogToAnyBase(next)) {
                    com.github.gbenroscience.util.Utils.logError(
                            "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + curr + "\" And \"" + next + "\""
                            + " As You Have Done."
                            + "ParserNG Error Detector For Binary Operators!");
                    correct = false;
                    scan.clear();
                }//end if

            }//end else if

        }//end try
        catch (IndexOutOfBoundsException ind) {
ind.printStackTrace();
        }//end catch

        return correct;
    }

}//end class
