/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.parser;

import com.github.gbenroscience.parser.methods.Method;

import java.util.ArrayList;
import static com.github.gbenroscience.parser.Number.*;
import static com.github.gbenroscience.parser.Variable.*;
import java.util.List;

/**
 *
 * Models a post-operand Operator object e.g the
 * trigonometric,logarithmic,exponential e.t.c. operators.
 *
 * @author GBEMIRO
 */
public final class UnaryPreOperator extends Operator implements Validatable {

    /**
     * The precedence of this BinaryOperator object.
     */
    private final Precedence precedence;

    /**
     * The index of this operator in the scanned MathExpression that it belongs
     * to.
     */
    private int index;

    /**
     * Creates a new UnaryPreOperator object
     *
     * @param name The name of the operator
     * @param index The index of the operator in the list
     * @param scan The scanner output
     */
    public UnaryPreOperator(String name, int index, List<String> scan) {
        super(isUnaryPreOperator(name) ? name : "");
        if (this.getName().equals("")) {
            throw new IndexOutOfBoundsException("Invalid Name For Unary Pre-Number Operator.");
        }//end if
        else {

            this.index = (index >= 0 && scan.get(index).equals(name)) ? index : -1;
            this.precedence = getPrecedence(name);
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
     * @param scan the ArrayList object that this UnaryPreOperator object exists
     * in. validates the grammatical usage of this operator (by leaving the
     * correctFunction attribute of the function object un-modified) if the
     * usage of this operator in its immediate environment i.e to its left and
     * right is correct.
     * @return true if the grammatical usage of this token with repect to its 2
     * immediate neighbouring tokens to the left and to the right is correct.
     */
    @Override
    public boolean validate(List<String> scan) {

        int leftInd = index - 1;
        int rightInd = index + 1;
        int sz = scan.size();
        boolean correct = true;
        String prev = leftInd >= 0 ? scan.get(leftInd) : null;
        String curr = scan.get(index);
        String next = rightInd < sz ? scan.get(rightInd) : null;

        try {
            //specify valid tokens that can come before a pre-number operator
            if (leftInd >= 0 && !isBinaryOperator(prev) && !isNumber(prev)
                    && !isUnaryPreOperator(prev) && !isBracket(prev)
                    && !isVariableString(prev) && !isUnaryPostOperator(prev)
                    && !isLogicOperator(prev)
                    && !isAssignmentOperator(prev)) {
                com.github.gbenroscience.util.Utils.logError(
                        "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + prev + "\" And \"" + curr + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Pre-number operators!");
                correct = false;
                scan.clear();
            }//end if
            //specify valid tokens that can come after a pre-number operator
            if (rightInd < sz && !isOpeningBracket(next) && !isNumber(next)
                    && !Method.isUnaryPreOperatorORDefinedMethod(next)
                    && !isVariableString(next) && !Method.isNumberReturningStatsMethod(next)
                    && !Method.isLogToAnyBase(next) && !Method.isAntiLogToAnyBase(next)) {
                com.github.gbenroscience.util.Utils.logError(
                        "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + curr + "\" And \"" + next + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Pre-number operators!");
                correct = false;
                scan.clear();
            }//end if

        }//end try
        catch (IndexOutOfBoundsException ind) {
            ind.printStackTrace();
        }//end catch

        return correct;
    }

    /**
     *
     * @param function the MathExpression object that contains the power
     * operator
     */
    public static void assignCompoundTokens(MathExpression function) {
        List<String> scan = function.getScanner();

    }

}//end class UnaryPreOperator
