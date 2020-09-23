/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import parser.methods.Method;

import java.util.ArrayList;

import static parser.Number.*;
import static parser.Variable.*;

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
    public LogicOperator(String name, int index, ArrayList<String> scan) {
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
    public boolean validate(ArrayList<String> scan) {

        boolean correct = true;

        try {
            //specify valid tokens that can come before a logic operator
            if (!isNumber(scan.get(index - 1))
                    && !isVariableString(scan.get(index - 1)) && !isUnaryPostOperator(scan.get(index - 1))
                    && !isClosingBracket(scan.get(index - 1))) {
                util.Utils.logError(
                        "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + scan.get(index - 1) + "\" And \"" + scan.get(index) + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Logic operators!");
                correct = false;
                scan.clear();

            }//end if
            //specify valid tokens that can come after a logic operator
            if (!isNumber(scan.get(index + 1)) && !isVariableString(scan.get(index + 1))
                    && !isOpeningBracket(scan.get(index + 1))
                    && !Method.isUnaryPreOperatorORDefinedMethod(scan.get(index + 1)) && !Method.isNumberReturningStatsMethod(scan.get(index + 1))
                    && !Method.isLogToAnyBase(scan.get(index + 1)) && !Method.isAntiLogToAnyBase(scan.get(index + 1))) {
                util.Utils.logError(
                        "ParserNG Does Not Allow " + getName() + " To Combine The Function Members \"" + scan.get(index) + "\" And \"" + scan.get(index + 1) + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Logic operators!");
                correct = false;
                scan.clear();
            }//end if
        }//end try
        catch (IndexOutOfBoundsException ind) {

        }//end catch
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

