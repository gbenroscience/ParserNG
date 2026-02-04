/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import java.util.*;

/**
 *
 * @author GBENRO
 */
public class Operator {

    /**
     * The name of the operator object
     */
    protected String name;

    public static final String PLUS = OperatorConstant.PLUS.getSymbol();
    public static final String MINUS = OperatorConstant.MINUS.getSymbol();
    public static final String DIVIDE = OperatorConstant.DIVIDE.getSymbol();
    public static final String MULTIPLY = OperatorConstant.MULTIPLY.getSymbol();
    public static final String EN_DASH = OperatorConstant.EN_DASH.getSymbol();
    public static final String FACTORIAL = OperatorConstant.FACTORIAL.getSymbol();
    public static final String POWER = OperatorConstant.POWER.getSymbol();
    public static final String LESS_THAN = OperatorConstant.LESS_THAN.getSymbol();
    public static final String GREATER_THAN = OperatorConstant.GREATER_THAN.getSymbol();
    public static final String ASSIGN = OperatorConstant.ASSIGN.getSymbol();
    public static final String EQUALS = OperatorConstant.EQUALS.getSymbol();
    public static final String LESS_OR_EQUALS = OperatorConstant.LESS_OR_EQUALS.getSymbol();
    public static final String GREATER_OR_EQUALS = OperatorConstant.GREATER_OR_EQUALS.getSymbol();
    public static final String AND = OperatorConstant.AND.getSymbol();
    public static final String OR = OperatorConstant.OR.getSymbol();
    public static final String REMAINDER = OperatorConstant.REMAINDER.getSymbol();
    public static final String OPEN_CIRC_BRAC = OperatorConstant.OPEN_CIRC_BRAC.getSymbol();
    public static final String CLOSE_CIRC_BRAC = OperatorConstant.CLOSE_CIRC_BRAC.getSymbol();
    public static final String COMMA = OperatorConstant.COMMA.getSymbol();
    public static final String ROOT = OperatorConstant.ROOT.getSymbol();
    public static final String CUBE_ROOT = OperatorConstant.CUBE_ROOT.getSymbol();
    public static final String PERMUTATION = OperatorConstant.PERMUTATION.getSymbol();
    public static final String COMBINATION = OperatorConstant.COMBINATION.getSymbol();
    public static final String INVERSE = OperatorConstant.INVERSE.getSymbol();
    public static final String SQUARE = OperatorConstant.SQUARE.getSymbol();
    public static final String CUBE = OperatorConstant.CUBE.getSymbol();
    public static final String OPEN_SQUARE_BRAC = OperatorConstant.OPEN_SQUARE_BRAC.getSymbol();
    public static final String CLOSE_SQUARE_BRAC = OperatorConstant.CLOSE_SQUARE_BRAC.getSymbol();
    public static final String COLON = OperatorConstant.COLON.getSymbol();
    public static final String CONST = OperatorConstant.CONST.getSymbol();
    public static final String STORE = OperatorConstant.STORE.getSymbol();
    public static final String EXIT = OperatorConstant.EXIT.getSymbol();
    public static final String SPACE = OperatorConstant.SPACE.getSymbol();
    public static final String SEMI_COLON = OperatorConstant.SEMI_COLON.getSymbol();
    public static final String AT = OperatorConstant.AT.getSymbol();

    /**
     * The instruction set of the parser of this software.
     */
    public static final String[] operators
            = new String[]{PLUS, MINUS, DIVIDE, MULTIPLY, FACTORIAL, POWER, LESS_THAN, GREATER_THAN,
                ASSIGN, EQUALS, LESS_OR_EQUALS, GREATER_OR_EQUALS, AND, OR, REMAINDER, OPEN_CIRC_BRAC, CLOSE_CIRC_BRAC, COMMA, ROOT, CUBE_ROOT,
                PERMUTATION, COMBINATION, INVERSE, SQUARE, CUBE, OPEN_SQUARE_BRAC, CLOSE_SQUARE_BRAC, COLON, CONST, STORE, EXIT, SPACE,
                SEMI_COLON, AT
            };

    /**
     *
     * @param name creates a new form of a valid Operator
     */
    public Operator(String name) {
        if (OperatorConstant.isOperatorString(name)) {
            this.name = name;
        } else {
            throw new InputMismatchException(name + " is not a valid operator!");
        }
    }

    /**
     *
     * @param name set the name of the Operator object
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     *
     * @return the name of the Operator object
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
        return OperatorConstant.isOperatorString(op);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object contains a exit command
     */
    public static boolean isExitCommand(String op) {
        return op.equals(EXIT);
    }

    public boolean isSemiColon(String op) {
        if (op.length() != 1) {
            return false;
        }
        // 2. Direct character comparison avoids the overhead of the .equals() loop
        return op.charAt(0) == SEMI_COLON.charAt(0);
    }

    public static boolean isAtOperator(String op) {
        if (op.length() != 1) {
            return false;
        }
        // 2. Direct character comparison avoids the overhead of the .equals() loop
        return op.charAt(0) == AT.charAt(0);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object contains a STORE command
     */
    public static boolean isStoreCommand(String op) {
        return op.equals(STORE);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object contains a constant storage command
     */
    public static boolean isConstantStoreCommand(String op) {
        return op.equals(CONST);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object represents a [ character
     */
    public static boolean isOpeningBrace(String op) {
        if (op.length() != 1) {
            return false;
        }
        // 2. Direct character comparison avoids the overhead of the .equals() loop
        return op.charAt(0) == OPEN_SQUARE_BRAC.charAt(0);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object represents a [ character
     */
    public static boolean isClosingBrace(String op) {
        if (op.length() != 1) {
            return false;
        }
        // 2. Direct character comparison avoids the overhead of the .equals() loop
        return op.charAt(0) == CLOSE_SQUARE_BRAC.charAt(0);
    }

    /**
     *
     * @param op the String object in consideration
     * @return true if the String object represents a : character This operator
     * is used to mark the end-point of Variable initialization e.g
     * [A=2,AA=3,acy=8.838383]:3A+6AA-8acy This instructs the parser to
     * initialize the variables A, AA, acy with the values given and then
     * evaluate the expression
     *
     */
    public static boolean isColon(String op) {
        if (op.length() != 1) {
            return false;
        }
        return op.charAt(0) == COLON.charAt(0);
    }

    /**
     * @param op the String to check
     * @return true if the operator is a logic operator the logic operators
     * defined here are ==,&lte;,&gte;,&lt;,&gt;,|,&amp;
     */
    public static boolean isLogicOperator(String op) {
         int len = op.length();
 
        // Case 1: "< > ≤ ≥ & |" (Length 1)
        if (len == 1) {
            return op.charAt(0) == LESS_OR_EQUALS.charAt(0) || op.charAt(0) == GREATER_OR_EQUALS.charAt(0) || op.charAt(0) == LESS_THAN.charAt(0) || op.charAt(0) == GREATER_THAN.charAt(0) || 
                    op.charAt(0) == OR.charAt(0) || op.charAt(0) == AND.charAt(0);
        }

        // Case 2: "==" (Length 2)
        if (len == 2) {
            return op.charAt(0) == EQUALS.charAt(0) && op.charAt(1) == EQUALS.charAt(1);
        }

        return false;
    }

    /**
     * @param op the String to check
     * @return true if the operator is the EQUALS operator
     */
    public static boolean isEqualsOperator(String op) {
        // 1. Length check is the fastest way to reject 90% of tokens
        if (op.length() != 2) {
            return false;
        }

        return op.charAt(0) == EQUALS.charAt(0) && op.charAt(1) == EQUALS.charAt(1);
    }
    
    public static boolean isGreaterThanOperator(String op) {
         if (op.length() != 1) {
            return false;
        }

        return op.charAt(0) == GREATER_THAN.charAt(0);
    }
        
    public static boolean isGreaterOrEqualsOperator(String op) {
        int len = op.length();
        if(len == 1){
            return op.charAt(0) == GREATER_OR_EQUALS.charAt(0);
        }
        if (len == 2) {
            return op.charAt(0) == GREATER_THAN.charAt(0) && op.charAt(1) == ASSIGN.charAt(1);
        }

        return false;
    }
        
    public static boolean isLessThanOperator(String op) {
      if (op.length() != 1) {
            return false;
        }

        return op.charAt(0) == LESS_THAN.charAt(0);
    }
        
    public static boolean isLessThanOrEqualsOperator(String op) {
      int len = op.length();
        if(len == 1){
            return op.charAt(0) == LESS_OR_EQUALS.charAt(0);
        }
        if (len == 2) {
            return op.charAt(0) == LESS_THAN.charAt(0) && op.charAt(1) == ASSIGN.charAt(1);
        }
        return false;
    }
    
    
    

    /**
     * @param op the String to check
     * @return true if the operator is the ASSIGN operator it means that we
     * assign or store the value of the RHS in the LHS and so the LHS must
     * represent a valid variable
     */
    public static boolean isAssignmentOperator(String op) {
        if (op.length() != 1) {
            return false;
        }
        return op.charAt(0) == ASSIGN.charAt(0);
    }

    /**
     * @param op the String to check
     * @return true if the operator is an operator that functions in between 2
     * numbers or variables i.e,+,-,*,/,^,%,Č,Р
     */
    public static boolean isBinaryOperator(String op) {
         if (op.length() != 1) {
            return false;
        }
        return (op.charAt(0) == PLUS.charAt(0) || op.charAt(0) == MINUS.charAt(0) || op.charAt(0) == DIVIDE.charAt(0) || op.charAt(0) == MULTIPLY.charAt(0)
                || op.charAt(0) == POWER.charAt(0) || op.charAt(0) == REMAINDER.charAt(0) || op.charAt(0) == COMBINATION.charAt(0) || op.charAt(0) == PERMUTATION.charAt(0));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the + operator or is the - operator the
     * form is log-¹(num,base)
     */
    public static boolean isPlusOrMinus(String op) {
        if (op.length() != 1) {
            return false;
        }
        return (op.charAt(0) == PLUS.charAt(0) || op.charAt(0) == MINUS.charAt(0));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the / operator or is the * operator the
     * form is log-¹(num,base)
     */
    public static boolean isMulOrDiv(String op) {
        if (op.length() != 1) {
            return false;
        }
        return (op.charAt(0) == DIVIDE.charAt(0) || op.charAt(0) == MULTIPLY.charAt(0));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the permutation operator or is the
     * combination operator the form is log-¹(num,base)
     */
    public static boolean isPermOrComb(String op) {
        if (op.length() != 1) {
            return false;
        }
        return (op.charAt(0) == COMBINATION.charAt(0) || op.charAt(0) == PERMUTATION.charAt(0));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the + operator or is the - operator the
     * form is log-¹(num,base)
     */
    public static boolean isMulOrDivOrRemOrPermOrCombOrPow(String op) {
        return (isBinaryOperator(op) && !isPlusOrMinus(op));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the REMAINDER operator
     */
    public static boolean isRemainder(String op) {
        if (op.length() != 1) {
            return false;
        }
        return (op.charAt(0) == REMAINDER.charAt(0));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the POWER operator
     */
    public static boolean isPower(String op) {
        if (op.length() != 1) {
            return false;
        }
        return (op.charAt(0) == POWER.charAt(0));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the OPEN_CIRC_BRAC or the CLOSE_CIRC_BRAC
     * operator
     */
    public static boolean isBracket(String op) {
        if (op.length() != 1) {
            return false;
        }
        return (op.charAt(0) == OPEN_CIRC_BRAC.charAt(0) || op.charAt(0) == CLOSE_CIRC_BRAC.charAt(0));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the OPEN_CIRC_BRAC operator
     */
    public static boolean isOpeningBracket(String op) {
        if (op.length() != 1) {
            return false;
        }
        return op.charAt(0) == OPEN_CIRC_BRAC.charAt(0);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the CLOSE_CIRC_BRAC operator
     */
    public static boolean isClosingBracket(String op) {
        if (op.length() != 1) {
            return false;
        }
        return op.charAt(0) == CLOSE_CIRC_BRAC.charAt(0);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the FACTORIAL operator
     */
    public static boolean isFactorial(String op) {
        if (op.length() != 1) {
            return false;
        }
        return op.charAt(0) == FACTORIAL.charAt(0);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the INVERSE operator
     */
    public static boolean isInverse(String op) {
        if (op.length() != 2) {
            return false;
        }
        return op.charAt(0) == INVERSE.charAt(0) && op.charAt(1) == INVERSE.charAt(1);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the ROOT operator
     */
    public static boolean isSquareRoot(String op) {
        if (op.length() != 1) {
            return false;
        }
        return op.charAt(0) == ROOT.charAt(0);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the "³√" operator
     */
    public static boolean isCubeRoot(String op) {
        if (op.length() != 2) {
            return false;
        }
        return op.charAt(0) == CUBE_ROOT.charAt(0) && op.charAt(1) == CUBE_ROOT.charAt(1);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the SQUARE operator
     */
    public static boolean isSquare(String op) {
        if (op.length() != 1) {
            return false;
        }
        return op.charAt(0) == SQUARE.charAt(0);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the CUBE operator
     */
    public static boolean isCube(String op) {
        if (op.length() != 1) {
            return false;
        }
        return op.charAt(0) == CUBE.charAt(0);
    }

    /**
     * @param op the String to check
     * @return true if the operator is a pre-number operator e.g the trig
     * operators,exponential operators,logarithmic operators(not to any base)
     */
    public static boolean isUnaryPreOperator(String op) {
       
        int len = op.length();

        // Case 1: "√" (Length 1)
        if (len == 1) {
            return op.charAt(0) == ROOT.charAt(0);
        }

        // Case 2: "³√" (Length 2)
        if (len == 2) {
            return op.charAt(0) == CUBE_ROOT.charAt(0) && op.charAt(1) == CUBE_ROOT.charAt(1);
        }

        return false;
    }

    /**
     * @param op the String to check
     * @return true if the operator is a post number operator e.g the inverse
     * operator,the factorial,the square and the cube
     */
    public static boolean isUnaryPostOperator(String op) {
  
        int len = op.length();

        // Case 1: "!", "²", "³" (Length 1)
        if (len == 1) {
            char c = op.charAt(0);
            return c == FACTORIAL.charAt(0) || c == SQUARE.charAt(0) || c == CUBE.charAt(0);
        }

        // Case 2: "-¹" (Length 2)
        if (len == 2) {
            // Direct char comparison is faster than .equals(INVERSE)
            return op.charAt(0) == INVERSE.charAt(0) && op.charAt(1) == INVERSE.charAt(1);
        }

        return false;
    }

    /**
     * @param op the String to check
     * @return true if the operator is the comma(,) operator
     */
    public static boolean isComma(String op) {
        if (op.length() != 1) {
            return false;
        }
        return op.charAt(0) == COMMA.charAt(0);

    }

    /**
     * The precedence of the operators
     *
     * @param name the name of the Operator object
     * @return the Operator's Precedence attribute
     */
    public static Precedence getPrecedence(String name) {

        if (isUnaryPostOperator(name)) {
            return new Precedence(10000);
        } else if (isPower(name)) {
            return new Precedence(9999);
        } else if (isUnaryPreOperator(name)) {
            return new Precedence(9998);
        } else if (isMulOrDiv(name)) {
            return new Precedence(9997);
        } else if (isRemainder(name)) {
            return new Precedence(9996);
        } else if (isPlusOrMinus(name)) {
            return new Precedence(9995);
        }

        return null;
    }

    /**
     *
     * @param scan An ArrayList object containing a scanned function.
     * @return true if validated
     */
    public static boolean validateAll(ArrayList<String> scan) {
        boolean correct = true;
        for (int i = 0; i < scan.size(); i++) {
            if (isBinaryOperator(scan.get(i))) {
                correct = new BinaryOperator(scan.get(i), i, scan).validate(scan);
            } else if (isLogicOperator(scan.get(i))) {
                correct = new LogicOperator(scan.get(i), i, scan).validate(scan);
            } else if (isUnaryPostOperator(scan.get(i))) {
                correct = new UnaryPostOperator(scan.get(i), i, scan).validate(scan);
            } else if (isUnaryPreOperator(scan.get(i))) {
                correct = new UnaryPreOperator(scan.get(i), i, scan).validate(scan);
            }
        }//end for

        return correct;
    }//end method

}//end method
