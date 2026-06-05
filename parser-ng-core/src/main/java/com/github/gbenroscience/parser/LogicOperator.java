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
 * @author GBEMIRO
 */
public class LogicOperator extends Operator implements Validatable {

    /**
     * The precedence of this LogicOperator object.
     */
    private final Precedence precedence;
    /**
     * The index of this operator in the scanned Function that it belongs to.
     */
    private int index;

    /**
     *
     * @param name The name that identifies the LogicOperator object
     * @param index The index of this operator in the scanned Function that it
     * belongs to.
     * @param scan the Function object that this LogicOperator object belongs
     * to.
     */
    public LogicOperator(String name, int index,  List<String> scan) {
        super(isLogicOperator(name) ? name : "");

        if (this.getName().equals("")) {
            throw new IndexOutOfBoundsException("Invalid Name For Binary Operator.");
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
     *
     * @param index sets the index of this operator in the scanned Function that
     * it belongs to.
     * @param scan The scanner output
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
     * @param scan the ArrayList object that this LogicOperator object exists
     * in. validates the grammatical usage of this operator (by leaving the
     * correctFunction attribute of the function object un-modified) if the
     * usage of this operator in its immediate environment i.e to its left and
     * right is correct.
     * 
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
  
            //specify valid tokens that can come before a logic operator
            if (leftInd>=0 && !isNumber(prev)
                    && !isVariableString(prev) && !isUnaryPostOperator(prev)
                    && !isClosingBracket(prev)) {
                com.github.gbenroscience.util.Utils.logError(
                        "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + prev + "\" And \"" + curr + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Logic operators!");
                correct = false;
                scan.clear();

            }//end if
            //specify valid tokens that can come after a logic operator
            if (rightInd<sz && isNumber(next) && !isVariableString(next)
                    && !isOpeningBracket(next)
                    && !Method.isUnaryPreOperatorORDefinedMethod(next) && !Method.isNumberReturningStatsMethod(next)
                    && !Method.isLogToAnyBase(next) && !Method.isAntiLogToAnyBase(next)) {
                com.github.gbenroscience.util.Utils.logError(
                        "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + curr + "\" And \"" + next + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Logic operators!");
                correct = false;
                scan.clear();
            }//end if
 
        return correct;
    }//end method

}//end class

class Greater extends LogicOperator {

    public Greater(int index, ArrayList<String> scan) {
        super(">", index, scan);
    }//end constructor

}//end class

//|&
final class Lesser extends LogicOperator {

    public Lesser(int index, ArrayList<String> scan) {
        super("<", index, scan);
    }//end constructor

}//end class

final class GreaterOrEquals extends LogicOperator {

    public GreaterOrEquals(int index, ArrayList<String> scan) {
        super("≥", index, scan);
    }//end constructor

}//end class

final class LesserOrEquals extends LogicOperator {

    public LesserOrEquals(int index, ArrayList<String> scan) {
        super("≤", index, scan);
    }//end constructor

}//end class

final class Equals extends LogicOperator {

    public Equals(int index, ArrayList<String> scan) {
        super("==", index, scan);
    }//end constructor
}//end class

final class AND extends LogicOperator {

    public AND(int index, ArrayList<String> scan) {
        super("&", index, scan);
    }//end constructor
}//end class

final class OR extends LogicOperator {

    public OR(int index, ArrayList<String> scan) {
        super("|", index, scan);
    }//end constructor

}//end class

