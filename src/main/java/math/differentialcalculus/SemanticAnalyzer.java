/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import parser.Bracket;
import parser.Operator;
import java.util.ArrayList;
import static parser.Number.*;
import static parser.Variable.*;
import static parser.Operator.*;
import static parser.Bracket.*;
import static math.differentialcalculus.Utilities.*;
import parser.PowerOperator;
import parser.UnaryPostOperator;

import static parser.methods.Method.*;

import java.util.List;

/**
 *
 * @author GBEMIRO
 */
public class SemanticAnalyzer {

    private Bracket[] brackets;
    private ArrayList<String> scanner;
    /**
     * Utility attribute used throughout the class for string appending
     * operations.
     */
    private StringBuilder utility = new StringBuilder("");

    public SemanticAnalyzer(String expression) throws Exception {
        DerivativeScanner ds = new DerivativeScanner(expression);

        if (!ds.getScanner().isEmpty()) {
            scanner = ds.getScanner();
            //Perform a back scan to appropriately wrap all square root and cube root operators up as appropriate

            for (int i = scanner.size() - 1; i >= 0; i--) {
                String operator = scanner.get(i);
                if (i + 1 >= scanner.size()) {
                    continue;
                }
                String token = scanner.get(i + 1);
                if (isUnaryPreOperator(operator)) {
                    scanner.set(i, isSquareRoot(operator) ? "sqrt" : "cbrt");

                    if (isVariableString(token) || isNumber(token)) {
                        if (i + 2 < scanner.size()) {
                            if (!isOpenBracket(scanner.get(i + 2))) {//but watch out for √,sin,(

                                scanner.add(i + 2, ")");
                                scanner.add(i + 1, "(");
                            } else {
                                int close = getComplementIndex(true, i + 2, scanner);
                                scanner.add(close + 1, ")");
                                scanner.add(i + 1, "(");
                            }
                        }

                    }
                }

            }//end for loop

            for (int i = 0; i < scanner.size(); i++) {
                try {
                    if (scanner.get(i).equals("-")) {
                        //the pattern (,-,var|num.. convert to (,-1,*,var|num
                        if ((isOpeningBracket(scanner.get(i - 1)) || isPower(scanner.get(i - 1)) || isMulOrDiv(scanner.get(i - 1)))
                                && (isVariableString(scanner.get(i + 1)) || isNumber(scanner.get(i + 1)))) {

                            if (isNumber(scanner.get(i + 1))) {
                                scanner.set(i + 1, "" + (Double.parseDouble(scanner.get(i + 1)) * -1));
                                scanner.remove(i);
                            } else {
                                scanner.set(i, "-1");
                                scanner.add(i + 1, "*");
                            }
                        }//end  if.
                    }//end if

                }//end try
                catch (IndexOutOfBoundsException boundsException) {

                }
            }//end for

            checkOperatorSemantics();

        } else {
            throw new Exception("Bad Math Expression");
        }
    }

    public Bracket[] getBrackets() {
        return brackets;
    }

    /**
     * Scans the bracket structure of the input expression and determines if it
     * is semantically correct
     *
     * @return
     */
    boolean checkBracketStructure() {
        int close = 0;
        ArrayList<String> scan = new ArrayList<>();
        ArrayList<String> scanner = getScanner();
        scan.addAll(scanner);
        ArrayList<Bracket> brackets = new ArrayList<>();
        int open = 0;
        int i = 0;
        while ((close = scan.indexOf(")")) != -1) {
            try {
                open = scan.subList(0, close + 1).lastIndexOf("(");
                Bracket opening = new Bracket("(");
                Bracket closing = new Bracket(")");
                opening.setIndex(open);
                closing.setIndex(close);
                opening.setComplement(closing);
                closing.setComplement(opening);
                brackets.add(opening);
                brackets.add(closing);
                scan.set(open, "");
                scan.set(close, "");

                ++i;
            }//end try
            catch (IndexOutOfBoundsException ind) {
                scanner.clear();
                break;
            }
        }//end while loop

//after the mapping, the algorithm demands that all ( and ) should have been used up in the function
        if (scan.indexOf("(") == -1 && scan.indexOf(")") == -1) {
            int size = brackets.size();
            this.brackets = new Bracket[size];
            this.brackets = brackets.toArray(this.brackets);
            return true;
        } else {
            scanner.clear();
            return false;
        }

    }//end method.

    private boolean checkOperatorSemantics() {
        //codeModifier();
        checkBracketStructure();
        functionComponentsAssociation();

        return !getScanner().isEmpty();
    }//end method

    /**
     * This method does final adjustments to the scanner function e.g it will
     * check for errors in operator combination in the scanner function and so
     * on.
     */
    private void functionComponentsAssociation() {
        ArrayList<String> scanner = getScanner();
        for (int i = 0; i < scanner.size(); i++) {
//check for the various valid arrangements for all members of the function.

            //Variables
            if (isVariableString(scanner.get(i))) {
                try {
                    //specify valid tokens that can come before a variable
                    if (!isOpeningBracket(scanner.get(i - 1))
                            && !isLogicOperator(scanner.get(i - 1)) && !isUnaryPreOperatorORDefinedMethod(scanner.get(i - 1))
                            && !isBinaryOperator(scanner.get(i - 1)) && !isAssignmentOperator(scanner.get(i - 1)) && !isNumber(scanner.get(i - 1)) && !isVariableString(scanner.get(i - 1))) {
                        //processLogger.writeLog("ParserNG Does Not Allow "+expression+" To Combine The MathExpression Members \""+scanner.get(i-1)+"\" And \""+scanner.get(i)+"\"\n");

                        scanner.clear();
                        break;
                    }//end if
                    //specify valid tokens that can come after a variable
                    if (!isBracket(scanner.get(i + 1)) && !isBinaryOperator(scanner.get(i + 1))
                            && !isUnaryPostOperator(scanner.get(i + 1)) && !isNumberReturningStatsMethod(scanner.get(i + 1))
                            && !isLogicOperator(scanner.get(i + 1)) && !isAssignmentOperator(scanner.get(i + 1))
                            && !isUnaryPreOperatorORDefinedMethod(scanner.get(i + 1))
                            && !isLogToAnyBase(scanner.get(i + 1)) && !isAntiLogToAnyBase(scanner.get(i + 1))
                            && !isNumber(scanner.get(i + 1)) && !isVariableString(scanner.get(i + 1))) {
                        //processLogger.writeLog("ParserNG Does Not Allow "+expression+" To Combine The MathExpression Members \""+scanner.get(i)+"\" And \""+scanner.get(i+1)+"\""+" As You Have Done.\n");

                        scanner.clear();
                        break;
                    }//end if
                }//end try
                catch (IndexOutOfBoundsException ind) {
                }//end catch
            }//end else if

        }//end for
        boolean valid = validateAll(scanner);

        if (valid) {
            // scanner.removeAll(whitespaceremover);
        }//end if
        else {
            scanner.clear();
            brackets = null;
        }//end else

    }//end method functionComponentAssociation

    /**
     * The method establishes meaning to some shorthand techniques in math that
     * the average mathematician might expect to see in a math device.
     * :e.g(3+4)(1+2) will become (3+4)*(1+2).It is essentially a stage that
     * generates code.
     */
    private void codeModifier() {
        ArrayList<String> scanner = getScanner();
        UnaryPostOperator.assignCompoundTokens(scanner);
        PowerOperator.assignCompoundTokens(scanner);

        /**
         * This stage serves for negative number detection. Prior to this stage,
         * all numbers are seen as positive ones. For example: turns [12,/,-,5]
         * to [12,/,-5].
         *
         * It also counts the number of list-returning operators in the system.
         */
        for (int i = 0; i < scanner.size(); i++) {
            try {
                if ((isBinaryOperator(scanner.get(i)) || isUnaryPreOperatorORDefinedMethod(scanner.get(i))
                        || isOpeningBracket(scanner.get(i))
                        || isLogicOperator(scanner.get(i)) || isAssignmentOperator(scanner.get(i))
                        || isComma(scanner.get(i)) || isStatsMethod(scanner.get(i)))
                        && Operator.isPlusOrMinus(scanner.get(i + 1)) && isNumber(scanner.get(i + 2))) {
                    utility.append(scanner.get(i + 1));
                    utility.append(scanner.get(i + 2));
                    scanner.set(i + 1, utility.toString());
                    scanner.set(i + 2, "");
                    utility.delete(0, utility.length());//clear the builder.
                }//end if

            }//end try
            catch (IndexOutOfBoundsException ind) {
            }//end catch

        }//end for

        /**
         * The theory behind this is that any sub-expression that lies between
         * any 2 inBetweenOperators or between an in between operator and the
         * enclosing bracket: is a compound token and as such must be evaluated
         * first before execution of other parts of the expression can proceed.
         * For example: 3/2sin4+5 is scanned as: [3,/,(,2,sin,4,),+,5] COME
         * HEREEEEEEEEEEEEEE!!!!FOR COMPOUND TOKENS IF NEEDED!!!!!!!!!!!!!!!!
         */
        freeSpaces(scanner);
        for (int i = 0; i < scanner.size(); i++) {

            /**
             * Remember to do for the format:
             * 3^sin2^sin3===3^sin(2^sin3)...correct not..(3^sin2)^sin3 change
             * things like 2sin3 to ( 2 * sin 3 ) Prevent the logic error:
             * 3/4sin2==3/4*sin2 The LHS == 3/(4*sin2) THE RHS == (3/4)*sin2
             */
            try {
                if ((isNumber(scanner.get(i)) || isVariableString(scanner.get(i))) && (isUnaryPreOperatorORDefinedMethod(scanner.get(i + 1))
                        || isNumberReturningStatsMethod(scanner.get(i + 1)) || isLogOrAntiLogToAnyBase(scanner.get(i + 1)))) {

                    //Determine the placement of the close bracket
                    int j = i + 1;
                    while (isUnaryPreOperatorORDefinedMethod(scanner.get(j)) || isNumberReturningStatsMethod(scanner.get(j)) || isLogOrAntiLogToAnyBase(scanner.get(j))) {
                        ++j;
                    }
                    if (isNumber(scanner.get(j)) || isVariableString(scanner.get(j))) {
                        scanner.add(j + 1, ")");
                        scanner.add(i, "(");
                        scanner.add(i + 2, "*");
                    } else if (isOpeningBracket(scanner.get(j))) {
                        int ind = Bracket.getComplementIndex(true, j, scanner);
                        scanner.add(ind + 1, ")");
                        scanner.add(i, "(");
                        scanner.add(i + 2, "*");

                    }

                }//end if
            }//end try.
            catch (IndexOutOfBoundsException indErr) {
            }//end catch

            try {
                /**
                 * tackles situations like sin-(2+3) by converting them into
                 * sin(-1*(2+3)). The generic situation is
                 * "preNumberOperator-(expr)"
                 */
                if (isUnaryPreOperatorORDefinedMethod(scanner.get(i - 1)) && Operator.isPlusOrMinus(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                    if (scanner.get(i).equals("-")) {
                        List<String> subList = scanner.subList(i - 1, Bracket.getComplementIndex(true, i + 1, scanner) + 1);

                        subList.set(1, "(");
                        subList.add(2, "-1");
                        subList.add(3, "*");
                        subList.add(")");
                    }//end if
                    else if (scanner.get(i).equals("+")) {
                        scanner.set(i, "");
                    }

                }//end if
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }//end catch

            try {
                /**
                 * tackles situations like (-sin2+3) which it converts into
                 * -1*sin2+3 if this is not done, it would be evaluated as
                 * -(sin2+3)
                 */
                if (isOpeningBracket(scanner.get(i - 1)) && Operator.isPlusOrMinus(scanner.get(i)) && isUnaryPreOperatorORDefinedMethod(scanner.get(i + 1))) {
                    if (scanner.get(i).equals("-")) {
                        scanner.set(i, "-1");
                        scanner.add(i + 1, "*");
                    } else if (scanner.get(i).equals("+")) {
                        scanner.set(i, "");
                    }
                }
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }//end catch

        }//end for

        freeSpaces(getScanner());

    }//end method codeModifier

    public ArrayList<String> getScanner() {
        return scanner;
    }

    public static void main(String[] args) {
        try {
            String expression = "2+3*(2*x*x*2*x*x*8*x*7*x*4*y*(3*x)-¹*3*x*x^2*2*3*4^x-5*1/x)";

            SemanticAnalyzer analyzer = new SemanticAnalyzer(expression);
            System.out.println(analyzer.getScanner());
        } catch (Exception e) {

        }

    }//end main method

}//end class
