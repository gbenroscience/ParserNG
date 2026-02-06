/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.matrix.expressParser;


import parser.Precedence;
import parser.UnaryPreOperator;
import static parser.Operator.*;

import java.util.ArrayList;
import parser.Operator;

/**
 *
 * @author GBENRO
 */
public class MOperator extends Operator {

    public static final String ROW_JOIN = "rowJoin";
    public static final String COL_JOIN = "colJoin";
    public static final String TRI_MATRIX = "tri";
    public static final String DET_BRACE = "|";
    public static final String UNIT = "unit";
    public static final String DET = "det";
    public static final String INV = "inv";
    
    
    /**
 * The instruction set of the
 * parser of this software.
 */
public static final String[] operators =
new String[]{PLUS,MINUS,MULTIPLY,POWER,
        ASSIGN,OPEN_CIRC_BRAC,CLOSE_CIRC_BRAC,COMMA,
        INVERSE,SQUARE,CUBE,OPEN_SQUARE_BRAC,CLOSE_SQUARE_BRAC,COLON,SPACE,
        SEMI_COLON,AT
};


    /**
     *
     * @param name creates a new form of a valid MOperator
     */
    public MOperator(String name) {
        super(name);
    }

    /**
     *
     * @param name set the name of the MOperator object
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     *
     * @return the name of the MOperator object
     */
    public String getName() {
        return name;
    }

    /**
     *
     * @param op The string to check.
     * @return true if the operator is a valid mathronian operator
     */
    public static boolean isOperatorString(String op) {

        return (op.equals(PLUS) || op.equals(MINUS) || op.equals(MULTIPLY) || op.equals(AT)
                || op.equals(POWER) || op.equals(ASSIGN) || op.equals(OPEN_CIRC_BRAC) || op.equals(CLOSE_CIRC_BRAC)
                || op.equals(ROW_JOIN) || op.equals(COL_JOIN) || op.equals(TRI_MATRIX)
                || op.equals(DET_BRACE) || op.equals(UNIT) || op.equals(DET) || op.equals(COLON)
                || op.equals(INVERSE) || op.equals(INV) || op.equals(SQUARE) || op.equals(CUBE) || op.equals(COMMA)
                || op.equals(OPEN_SQUARE_BRAC) || op.equals(CLOSE_SQUARE_BRAC));

    }//end method isOperatorString

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object contains a exit command
     */
    public static boolean isRowJoin(String op) {
        return op.equals(ROW_JOIN);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object contains a exit command
     */
    public static boolean isColJoin(String op) {
        return op.equals(COL_JOIN);
    }

    public static boolean isColon(String op) {
        return op.equals(COLON);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object contains a Variable storage command
     */
    public static boolean isTri(String op) {
        return op.equals(TRI_MATRIX);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object contains a constant storage command
     */
    public static boolean isUnit(String op) {
        return op.equals(UNIT);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object represents a [ character
     */
    public static boolean isOpeningBrace(String op) {
        return op.equals(OPEN_SQUARE_BRAC);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object represents a [ character
     */
    public static boolean isClosingBrace(String op) {
        return op.equals(CLOSE_SQUARE_BRAC);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "=" operator it means that we assign
     * or store the value of the RHS in the LHS and so the LHS must represent a
     * valid variable
     */
    public static boolean isAssignmentOperator(String op) {
        return (op.equals(ASSIGN));
    }

    /**
     * @param op the String to check
     * @return true if the operator is an operator that functions in between 2
     * numbers or variables i.e,+,-,*,/,^,%,Č,Р
     */
    public static boolean isBinaryOperator(String op) {

        return (op.equals(PLUS) || op.equals(MINUS) || op.equals(MULTIPLY) || op.equals(POWER)
                || op.equals(ROW_JOIN) || op.equals(COL_JOIN));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the + operator or is the - operator the
     * form is log-¹(num,base)
     */
    public static boolean isPlusOrMinus(String op) {
        return (op.equals(PLUS) || op.equals(MINUS));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the * operator the form is
     * log-¹(num,base)
     */
    public static boolean isMul(String op) {
        return (op.equals(MULTIPLY));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "%" operator
     */
    public static boolean isPower(String op) {
        return (op.equals(POWER));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "(" or the ")" operator
     */
    public static boolean isBracket(String op) {
        return (op.equals(OPEN_CIRC_BRAC) || op.equals(CLOSE_CIRC_BRAC));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "(" operator
     */
    public static boolean isOpeningBracket(String op) {
        return op.equals(OPEN_CIRC_BRAC);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the ")" operator
     */
    public static boolean isClosingBracket(String op) {
        return op.equals(CLOSE_CIRC_BRAC);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "-¹" operator
     */
    public static boolean isInverse(String op) {
        return op.equals(INVERSE);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "|" operator
     */
    public static boolean isDetHalfSymbol(String op) {
        return op.equals(DET_BRACE);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "|" operator
     */
    public static boolean isDet(String op) {
        return op.equals(DET);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "²" operator
     */
    public static boolean isSquare(String op) {
        return op.equals(SQUARE);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "³" operator
     */
    public static boolean isCube(String op) {
        return op.equals(CUBE);
    }

    /**
     * @param op the String to check
     * @return true if the operator is a pre-number operator e.g the trig
     * operators,exponential operators,logarithmic operators(not to any base)
     */
    public static boolean isUnaryPreOperator(String op) {
        return (op.equals(DET) || op.equals(TRI_MATRIX) || op.equals(UNIT));
    }

    /**
     * @param op the String to check
     * @return true if the operator is a post number operator e.g the inverse
     * operator,the factorial,the square and the cube
     */
    public static boolean isUnaryPostOperator(String op) {
        return (op.equals(INVERSE) || op.equals(SQUARE) || op.equals(CUBE));
    }

    /**
     * The precedence of the operators
     *
     * @param name the name of the MOperator object
     * @return the MOperator's Precedence attribute
     */
    public static Precedence getPrecedence(String name) {

        if (isUnaryPostOperator(name)) {
            return new Precedence(10000);
        } else if (isPower(name)) {
            return new Precedence(9999);
        } else if (isUnaryPreOperator(name)) {
            return new Precedence(9998);
        } else if (isMul(name)) {
            return new Precedence(9997);
        } else if (isPlusOrMinus(name)) {
            return new Precedence(9995);
        }

        return null;
    }


 

    /**
     *
     * @param scan An ArrayList object containing a scanned function.
     * @return true if validated successfully
     */
    public static boolean validateAll(ArrayList<String> scan) {
        boolean correct = true;
        for (int i = 0; i < scan.size(); i++) {

            if (isUnaryPostOperator(scan.get(i))) {
                correct = new MUnaryPostOperator(scan.get(i), i, scan).validate(scan);
            } else if (isUnaryPreOperator(scan.get(i))) {
                correct = new UnaryPreOperator(scan.get(i), i, scan).validate(scan);
            }
        }//end for

        return correct;
    }//end method

}
